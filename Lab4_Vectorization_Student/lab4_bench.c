#define _GNU_SOURCE
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__AVX2__) || defined(__SSE2__)
#include <immintrin.h>
#endif

#define ALIGNMENT 64
#define NOINLINE __attribute__((noinline))
#if defined(__GNUC__) && !defined(__clang__)
#define NO_AUTOVECTORIZE \
    __attribute__((optimize("no-tree-loop-vectorize", "no-tree-slp-vectorize")))
#else
#define NO_AUTOVECTORIZE
#endif

static volatile double observable_sink;

static double now_seconds(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static uint64_t parse_u64(const char *s, const char *name) {
    if (!s || *s == '\0' || *s == '-') {
        fprintf(stderr, "invalid %s: %s\n", name, s ? s : "(null)");
        exit(EXIT_FAILURE);
    }
    errno = 0;
    char *end = NULL;
    unsigned long long value = strtoull(s, &end, 10);
    if (errno || end == s || *end != '\0') {
        fprintf(stderr, "invalid %s: %s\n", name, s);
        exit(EXIT_FAILURE);
    }
    return (uint64_t)value;
}

static int parse_threads(const char *s) {
    uint64_t value = parse_u64(s, "THREADS");
    if (value == 0 || value > 256) {
        fprintf(stderr, "THREADS must be in 1..256\n");
        exit(EXIT_FAILURE);
    }
    return (int)value;
}

static size_t checked_count(uint64_t count, const char *name) {
    if (count == 0 || count > SIZE_MAX / sizeof(double)) {
        fprintf(stderr, "%s is zero or too large\n", name);
        exit(EXIT_FAILURE);
    }
    return (size_t)count;
}

static double *alloc_doubles(size_t count) {
    double *p = NULL;
    if (posix_memalign((void **)&p, ALIGNMENT, count * sizeof(*p)) != 0) {
        fprintf(stderr, "allocation failed for %zu doubles\n", count);
        exit(EXIT_FAILURE);
    }
    return p;
}

static void init_vector(double *a, size_t n, unsigned seed) {
    for (size_t i = 0; i < n; ++i) {
        uint64_t x = (uint64_t)i * 6364136223846793005ULL + seed;
        a[i] = (double)(x & 1023U) / 1024.0 + 0.125;
    }
}

static double checksum(const double *a, size_t n) {
    double sum = 0.0;
    size_t step = n > 4096 ? n / 4096 : 1;
    for (size_t i = 0; i < n; i += step) sum += a[i];
    observable_sink = sum;
    return sum;
}

/* Stage 1: a regular data-parallel loop. */
static NOINLINE void kernel_axpy(double *restrict a,
                                 const double *restrict b,
                                 size_t begin, size_t end,
                                 uint64_t repeats) {
    const double alpha = 1.00000011920928955078125;
    for (uint64_t r = 0; r < repeats; ++r) {
        for (size_t i = begin; i < end; ++i) {
            a[i] += alpha * b[i];
        }
    }
}

/* Stage 2: the loop-carried dependence prevents ordinary SIMD execution. */
static NOINLINE void kernel_dependency(double *a, const double *b,
                                       size_t n, uint64_t repeats) {
    for (uint64_t r = 0; r < repeats; ++r) {
        for (size_t i = 1; i < n; ++i) {
            a[i] = 0.5 * a[i - 1] + b[i];
        }
    }
}

/* Stage 3: enough arithmetic per element to expose SIMD execution resources. */
static NOINLINE void kernel_poly(double *restrict a,
                                 const double *restrict b,
                                 size_t begin, size_t end,
                                 uint64_t repeats) {
    for (uint64_t r = 0; r < repeats; ++r) {
        for (size_t i = begin; i < end; ++i) {
            double x = a[i];
            const double y = b[i];
            x = x * 1.0000001 + y;
            x = x * 0.9999999 + y;
            x = x * 1.0000003 - y * 0.25;
            x = x * 0.9999997 + y * 0.125;
            x = x * 1.0000002 + 0.000001;
            x = x * 0.9999998 + y * 0.0625;
            x = x * 1.0000004 - 0.0000005;
            x = x * 0.9999996 + y * 0.03125;
            a[i] = x;
        }
    }
}

typedef struct {
    int tid;
    int threads;
    size_t n;
    uint64_t repeats;
    double *a;
    const double *b;
    pthread_barrier_t *barrier;
} poly_arg_t;

static void range_for_thread(size_t n, int tid, int threads,
                             size_t *begin, size_t *end) {
    size_t base = n / (size_t)threads;
    size_t rem = n % (size_t)threads;
    *begin = (size_t)tid * base + ((size_t)tid < rem ? (size_t)tid : rem);
    *end = *begin + base + ((size_t)tid < rem ? 1U : 0U);
}

static void *poly_worker(void *opaque) {
    poly_arg_t *arg = opaque;
    size_t begin, end;
    range_for_thread(arg->n, arg->tid, arg->threads, &begin, &end);
    int rc = pthread_barrier_wait(arg->barrier);
    if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) return NULL;
    rc = pthread_barrier_wait(arg->barrier);
    if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) return NULL;
    kernel_poly(arg->a, arg->b, begin, end, arg->repeats);
    rc = pthread_barrier_wait(arg->barrier);
    if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) return NULL;
    return NULL;
}

static void run_axpy_or_dependency(int dependency, size_t n,
                                   uint64_t repeats) {
    double *a = alloc_doubles(n);
    double *b = alloc_doubles(n);
    init_vector(a, n, 17);
    init_vector(b, n, 29);

    double t0 = now_seconds();
    if (dependency) kernel_dependency(a, b, n, repeats);
    else kernel_axpy(a, b, 0, n, repeats);
    double elapsed = now_seconds() - t0;
    double work = (double)n * (double)repeats;
    printf("mode=%s elements=%zu repeats=%" PRIu64
           " elapsed_s=%.6f ns_per_element=%.3f checksum=%.9g\n",
           dependency ? "dependency" : "axpy", n, repeats, elapsed,
           elapsed * 1e9 / work, checksum(a, n));
    free(a);
    free(b);
}

static void run_poly(int threads, size_t n, uint64_t repeats) {
    if (n < (size_t)threads) {
        fprintf(stderr, "ELEMENTS must be at least THREADS\n");
        exit(EXIT_FAILURE);
    }
    double *a = alloc_doubles(n);
    double *b = alloc_doubles(n);
    init_vector(a, n, 41);
    init_vector(b, n, 73);

    pthread_t *workers = calloc((size_t)threads, sizeof(*workers));
    poly_arg_t *args = calloc((size_t)threads, sizeof(*args));
    if (!workers || !args) {
        fprintf(stderr, "thread allocation failed\n");
        exit(EXIT_FAILURE);
    }
    pthread_barrier_t barrier;
    if (pthread_barrier_init(&barrier, NULL, (unsigned)threads + 1U) != 0) {
        fprintf(stderr, "barrier initialization failed\n");
        exit(EXIT_FAILURE);
    }
    for (int t = 0; t < threads; ++t) {
        args[t] = (poly_arg_t){t, threads, n, repeats, a, b, &barrier};
        if (pthread_create(&workers[t], NULL, poly_worker, &args[t]) != 0) {
            fprintf(stderr, "pthread_create failed at thread %d\n", t);
            exit(EXIT_FAILURE);
        }
    }
    int rc = pthread_barrier_wait(&barrier);
    if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
        fprintf(stderr, "ready barrier failed\n");
        exit(EXIT_FAILURE);
    }
    double t0 = now_seconds();
    rc = pthread_barrier_wait(&barrier);
    if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
        fprintf(stderr, "start barrier failed\n");
        exit(EXIT_FAILURE);
    }
    rc = pthread_barrier_wait(&barrier);
    if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
        fprintf(stderr, "completion barrier failed\n");
        exit(EXIT_FAILURE);
    }
    double elapsed = now_seconds() - t0;
    for (int t = 0; t < threads; ++t) {
        if (pthread_join(workers[t], NULL) != 0) {
            fprintf(stderr, "pthread_join failed at thread %d\n", t);
            exit(EXIT_FAILURE);
        }
    }
    double elements = (double)n * (double)repeats;
    printf("mode=poly threads=%d elements=%zu repeats=%" PRIu64
           " elapsed_s=%.6f ns_per_element=%.3f checksum=%.9g\n",
           threads, n, repeats, elapsed, elapsed * 1e9 / elements,
           checksum(a, n));
    pthread_barrier_destroy(&barrier);
    free(workers);
    free(args);
    free(a);
    free(b);
}

/* Paper-inspired locality-friendly loop: B[i] is reused immediately. */
static NOINLINE void tradeoff_fused(double *restrict a,
                                    double *restrict b,
                                    const double *restrict c,
                                    size_t n, uint64_t repeats) {
    for (uint64_t r = 0; r < repeats; ++r) {
        for (size_t i = 1; i < n; ++i) {
            a[i] = 3.0 * b[i - 1];
            b[i] = 0.5 * b[i] + c[i];
        }
    }
}

/* Paper-inspired SIMD-friendly loop distribution: easier to vectorize, but B
   is revisited only after a whole-array pass. */
static NOINLINE void tradeoff_split(double *restrict a,
                                    double *restrict b,
                                    const double *restrict c,
                                    size_t n, uint64_t repeats) {
    for (uint64_t r = 0; r < repeats; ++r) {
        for (size_t i = 1; i < n; ++i) b[i] = 0.5 * b[i] + c[i];
        for (size_t i = 1; i < n; ++i) a[i] = 3.0 * b[i - 1];
    }
}

static void run_tradeoff(int split, size_t n, uint64_t repeats) {
    double *a = alloc_doubles(n);
    double *b = alloc_doubles(n);
    double *c = alloc_doubles(n);
    init_vector(a, n, 11);
    init_vector(b, n, 23);
    init_vector(c, n, 47);
    double t0 = now_seconds();
    if (split) tradeoff_split(a, b, c, n, repeats);
    else tradeoff_fused(a, b, c, n, repeats);
    double elapsed = now_seconds() - t0;
    double result_checksum = checksum(a, n);
    result_checksum += checksum(b, n);
    printf("mode=tradeoff variant=%s elements=%zu working_set_MiB=%.2f"
           " repeats=%" PRIu64 " elapsed_s=%.6f ns_per_element=%.3f"
           " checksum=%.9g\n",
           split ? "split" : "fused", n,
           (double)n * 3.0 * sizeof(double) / 1048576.0, repeats, elapsed,
           elapsed * 1e9 / ((double)n * (double)repeats),
           result_checksum);
    free(a);
    free(b);
    free(c);
}

static NOINLINE void matrix_row(double *m, size_t n, uint64_t repeats) {
    const double alpha = 1.0000001, beta = 0.0000001;
    for (uint64_t r = 0; r < repeats; ++r)
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j) {
                size_t p = i * n + j;
                m[p] = alpha * m[p] + beta;
            }
}

static NOINLINE void matrix_col(double *m, size_t n, uint64_t repeats) {
    const double alpha = 1.0000001, beta = 0.0000001;
    for (uint64_t r = 0; r < repeats; ++r)
        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < n; ++i) {
                size_t p = i * n + j;
                m[p] = alpha * m[p] + beta;
            }
}

static void run_matrix(int column, size_t n, uint64_t repeats) {
    if (n > 20000 || n > SIZE_MAX / n) {
        fprintf(stderr, "matrix dimension is too large\n");
        exit(EXIT_FAILURE);
    }
    size_t count = n * n;
    if (count > SIZE_MAX / sizeof(double)) {
        fprintf(stderr, "matrix is too large\n");
        exit(EXIT_FAILURE);
    }
    double *m = alloc_doubles(count);
    init_vector(m, count, 97);
    double t0 = now_seconds();
    if (column) matrix_col(m, n, repeats);
    else matrix_row(m, n, repeats);
    double elapsed = now_seconds() - t0;
    printf("mode=matrix order=%s N=%zu matrix_MiB=%.2f repeats=%" PRIu64
           " elapsed_s=%.6f ns_per_element=%.3f checksum=%.9g\n",
           column ? "col" : "row", n,
           (double)count * sizeof(double) / 1048576.0, repeats, elapsed,
           elapsed * 1e9 / ((double)count * (double)repeats),
           checksum(m, count));
    free(m);
}

static void zero_matrix(double *c, size_t count) {
    memset(c, 0, count * sizeof(*c));
}

static NOINLINE void mm_ijk(const double *restrict a,
                            const double *restrict b,
                            double *restrict c, size_t n) {
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k)
                sum += a[i * n + k] * b[k * n + j];
            c[i * n + j] = sum;
        }
}

static NOINLINE void mm_ikj(const double *restrict a,
                            const double *restrict b,
                            double *restrict c, size_t n) {
    for (size_t i = 0; i < n; ++i)
        for (size_t k = 0; k < n; ++k) {
            double aik = a[i * n + k];
            for (size_t j = 0; j < n; ++j)
                c[i * n + j] += aik * b[k * n + j];
        }
}

/* SUPPLIED REFERENCE: cache tiling/blocking with a SIMD-friendly inner loop.
   Keep this function unchanged so it remains a comparison point. */
static NOINLINE void mm_blocked(const double *restrict a,
                                const double *restrict b,
                                double *restrict c, size_t n,
                                size_t block) {
    for (size_t ii = 0; ii < n; ii += block)
        for (size_t kk = 0; kk < n; kk += block)
            for (size_t jj = 0; jj < n; jj += block) {
                size_t i_end = ii + block < n ? ii + block : n;
                size_t k_end = kk + block < n ? kk + block : n;
                size_t j_end = jj + block < n ? jj + block : n;
                for (size_t i = ii; i < i_end; ++i)
                    for (size_t k = kk; k < k_end; ++k) {
                        double aik = a[i * n + k];
                        for (size_t j = jj; j < j_end; ++j)
                            c[i * n + j] += aik * b[k * n + j];
                    }
            }
}

/* STUDENT EDIT TARGET: manually vectorize only the inner j loop with guarded
   AVX2, SSE2, and AArch64 NEON intrinsics. The scalar loop is the correct
   starter and becomes the remainder/fallback after the SIMD loops are added.
   Auto-vectorization is disabled for this function so any packed instructions
   in its assembly must come from the intrinsics written here. */
static NOINLINE NO_AUTOVECTORIZE void
mm_blocked_intrinsics(const double *restrict a,
                      const double *restrict b,
                      double *restrict c, size_t n,
                      size_t block) {
    for (size_t ii = 0; ii < n; ii += block)
        for (size_t kk = 0; kk < n; kk += block)
            for (size_t jj = 0; jj < n; jj += block) {
                size_t i_end = ii + block < n ? ii + block : n;
                size_t k_end = kk + block < n ? kk + block : n;
                size_t j_end = jj + block < n ? jj + block : n;
                for (size_t i = ii; i < i_end; ++i)
                    for (size_t k = kk; k < k_end; ++k) {
                        double aik = a[i * n + k];
                        size_t j = jj;

#if defined(__AVX2__)
                        /* TODO: process four doubles per iteration with
                           _mm256_set1_pd, loadu, mul, add, and storeu. */
#elif defined(__aarch64__)
                        /* TODO: process two doubles per iteration with
                           vdupq_n_f64, vld1q_f64, vmulq_f64, vaddq_f64,
                           and vst1q_f64. */
#elif defined(__SSE2__)
                        /* TODO: process two doubles per iteration with
                           _mm_set1_pd, loadu, mul, add, and storeu. */
#endif
                        /* Keep this loop as the scalar tail and fallback. */
                        for (; j < j_end; ++j)
                            c[i * n + j] += aik * b[k * n + j];
                    }
            }
}

typedef enum { MM_IJK, MM_IKJ, MM_BLOCKED, MM_INTRINSICS } mm_variant_t;

static void execute_mm(mm_variant_t variant, const double *a, const double *b,
                       double *c, size_t n, size_t block) {
    zero_matrix(c, n * n);
    if (variant == MM_IJK) mm_ijk(a, b, c, n);
    else if (variant == MM_IKJ) mm_ikj(a, b, c, n);
    else if (variant == MM_BLOCKED) mm_blocked(a, b, c, n, block);
    else mm_blocked_intrinsics(a, b, c, n, block);
}

static const char *mm_name(mm_variant_t variant) {
    if (variant == MM_IJK) return "ijk";
    if (variant == MM_IKJ) return "ikj";
    if (variant == MM_BLOCKED) return "blocked";
    return "intrinsics";
}

static void init_mm(double *a, double *b, size_t n) {
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) {
            a[i * n + j] = (double)((i + 3 * j) % 17) / 17.0;
            b[i * n + j] = (double)((2 * i + j) % 19) / 19.0;
        }
}

static void run_mm(mm_variant_t variant, size_t n, size_t block) {
    if (n == 0 || n > 4096 || n > SIZE_MAX / n) {
        fprintf(stderr, "N must be in 1..4096\n");
        exit(EXIT_FAILURE);
    }
    if (block == 0 || block > n) {
        fprintf(stderr, "BLOCK must be in 1..N\n");
        exit(EXIT_FAILURE);
    }
    size_t count = n * n;
    double *a = alloc_doubles(count);
    double *b = alloc_doubles(count);
    double *c = alloc_doubles(count);
    init_mm(a, b, n);
    double t0 = now_seconds();
    execute_mm(variant, a, b, c, n, block);
    double elapsed = now_seconds() - t0;
    double flops = 2.0 * (double)n * (double)n * (double)n;
    printf("mode=mm variant=%s N=%zu block=%zu elapsed_s=%.6f GFLOP_s=%.3f"
           " checksum=%.9g\n", mm_name(variant), n, block, elapsed,
           flops / elapsed / 1e9, checksum(c, count));
    free(a);
    free(b);
    free(c);
}

static void run_verify(void) {
    /* Odd N and a non-dividing block exercise edge tiles and SIMD tails. */
    const size_t n = 35, block = 8, count = n * n;
    double *a = alloc_doubles(count);
    double *b = alloc_doubles(count);
    double *reference = alloc_doubles(count);
    double *candidate = alloc_doubles(count);
    init_mm(a, b, n);
    execute_mm(MM_IJK, a, b, reference, n, block);
    mm_variant_t variants[] = {MM_IKJ, MM_BLOCKED, MM_INTRINSICS};
    for (size_t v = 0; v < sizeof(variants) / sizeof(variants[0]); ++v) {
        execute_mm(variants[v], a, b, candidate, n, block);
        double max_error = 0.0;
        for (size_t i = 0; i < count; ++i) {
            double error = fabs(reference[i] - candidate[i]);
            if (error > max_error) max_error = error;
        }
        printf("verify variant=%s max_abs_error=%.3e %s\n",
               mm_name(variants[v]), max_error,
               max_error <= 1e-10 ? "PASS" : "FAIL");
        if (max_error > 1e-10) exit(EXIT_FAILURE);
    }
    free(a);
    free(b);
    free(reference);
    free(candidate);
}

static void usage(const char *program) {
    fprintf(stderr,
        "Usage:\n"
        "  %s axpy ELEMENTS REPEATS\n"
        "  %s dependency ELEMENTS REPEATS\n"
        "  %s poly THREADS ELEMENTS REPEATS\n"
        "  %s tradeoff fused|split ELEMENTS REPEATS\n"
        "  %s matrix row|col N REPEATS\n"
        "  %s mm ijk|ikj|blocked|intrinsics N BLOCK\n"
        "  %s verify\n", program, program, program, program,
        program, program, program);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }
    if (strcmp(argv[1], "verify") == 0 && argc == 2) {
        run_verify();
        return EXIT_SUCCESS;
    }
    if ((strcmp(argv[1], "axpy") == 0 ||
         strcmp(argv[1], "dependency") == 0) && argc == 4) {
        size_t n = checked_count(parse_u64(argv[2], "ELEMENTS"), "ELEMENTS");
        uint64_t repeats = parse_u64(argv[3], "REPEATS");
        if (repeats == 0) { fprintf(stderr, "REPEATS must be positive\n"); return EXIT_FAILURE; }
        run_axpy_or_dependency(strcmp(argv[1], "dependency") == 0, n, repeats);
        return EXIT_SUCCESS;
    }
    if (strcmp(argv[1], "poly") == 0 && argc == 5) {
        int threads = parse_threads(argv[2]);
        size_t n = checked_count(parse_u64(argv[3], "ELEMENTS"), "ELEMENTS");
        uint64_t repeats = parse_u64(argv[4], "REPEATS");
        if (repeats == 0) { fprintf(stderr, "REPEATS must be positive\n"); return EXIT_FAILURE; }
        run_poly(threads, n, repeats);
        return EXIT_SUCCESS;
    }
    if (strcmp(argv[1], "tradeoff") == 0 && argc == 5) {
        int split;
        if (strcmp(argv[2], "fused") == 0) split = 0;
        else if (strcmp(argv[2], "split") == 0) split = 1;
        else { usage(argv[0]); return EXIT_FAILURE; }
        size_t n = checked_count(parse_u64(argv[3], "ELEMENTS"), "ELEMENTS");
        uint64_t repeats = parse_u64(argv[4], "REPEATS");
        if (n < 2 || repeats == 0) {
            fprintf(stderr, "ELEMENTS must be >= 2 and REPEATS positive\n");
            return EXIT_FAILURE;
        }
        run_tradeoff(split, n, repeats);
        return EXIT_SUCCESS;
    }
    if (strcmp(argv[1], "matrix") == 0 && argc == 5) {
        int column;
        if (strcmp(argv[2], "row") == 0) column = 0;
        else if (strcmp(argv[2], "col") == 0) column = 1;
        else { usage(argv[0]); return EXIT_FAILURE; }
        size_t n = checked_count(parse_u64(argv[3], "N"), "N");
        uint64_t repeats = parse_u64(argv[4], "REPEATS");
        if (repeats == 0) { fprintf(stderr, "REPEATS must be positive\n"); return EXIT_FAILURE; }
        run_matrix(column, n, repeats);
        return EXIT_SUCCESS;
    }
    if (strcmp(argv[1], "mm") == 0 && argc == 5) {
        mm_variant_t variant;
        if (strcmp(argv[2], "ijk") == 0) variant = MM_IJK;
        else if (strcmp(argv[2], "ikj") == 0) variant = MM_IKJ;
        else if (strcmp(argv[2], "blocked") == 0) variant = MM_BLOCKED;
        else if (strcmp(argv[2], "intrinsics") == 0) variant = MM_INTRINSICS;
        else { usage(argv[0]); return EXIT_FAILURE; }
        size_t n = checked_count(parse_u64(argv[3], "N"), "N");
        size_t block = (size_t)parse_u64(argv[4], "BLOCK");
        run_mm(variant, n, block);
        return EXIT_SUCCESS;
    }
    usage(argv[0]);
    return EXIT_FAILURE;
}
