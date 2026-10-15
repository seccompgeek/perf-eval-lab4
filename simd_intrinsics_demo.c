#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if !defined(LAB4_DISABLE_INTRINSICS) && defined(__aarch64__)
#include <arm_neon.h>
#elif !defined(LAB4_DISABLE_INTRINSICS) && (defined(__AVX2__) || defined(__SSE2__))
#include <immintrin.h>
#endif

#define ALIGNMENT 64
#define NOINLINE __attribute__((noinline))

static volatile double observable_sink;

static double now_seconds(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static size_t parse_size(const char *s, const char *name) {
    errno = 0;
    char *end = NULL;
    unsigned long long value = strtoull(s, &end, 10);
    if (errno || end == s || *end != '\0' || value == 0 ||
        value > SIZE_MAX / sizeof(double)) {
        fprintf(stderr, "invalid %s: %s\n", name, s);
        exit(EXIT_FAILURE);
    }
    return (size_t)value;
}

static uint64_t parse_repeats(const char *s) {
    errno = 0;
    char *end = NULL;
    unsigned long long value = strtoull(s, &end, 10);
    if (errno || end == s || *end != '\0' || value == 0) {
        fprintf(stderr, "invalid REPEATS: %s\n", s);
        exit(EXIT_FAILURE);
    }
    return (uint64_t)value;
}

static double *alloc_vector(size_t n) {
    double *p = NULL;
    if (posix_memalign((void **)&p, ALIGNMENT, n * sizeof(*p)) != 0) {
        fprintf(stderr, "allocation failed for %zu doubles\n", n);
        exit(EXIT_FAILURE);
    }
    return p;
}

static void initialize(double *a, double *b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        a[i] = (double)((i * 17U + 3U) & 1023U) / 1024.0;
        b[i] = (double)((i * 29U + 7U) & 1023U) / 4096.0;
    }
}

static double checksum(const double *a, size_t n) {
    double sum = 0.0;
    size_t step = n > 4096 ? n / 4096 : 1;
    for (size_t i = 0; i < n; i += step) sum += a[i];
    observable_sink = sum;
    return sum;
}

/* STUDENT INTRINSICS EXERCISE
   The starter implements a[i] += b[i] three ways below. Extend every path to
   a[i] += 2.0 * b[i]: ordinary C in kernel_auto(), the hidden operation in
   opaque_add(), and all guarded ISA branches in kernel_intrinsics(). Confirm
   that the scalar tail calls the updated helper. Keep `./simd_demo verify`
   passing. */

/* Easy case: the loop exposes independent, contiguous additions. */
static NOINLINE void kernel_auto(double *restrict a,
                                 const double *restrict b,
                                 size_t n) {
    for (size_t i = 0; i < n; ++i) { /* VEC_CHECK_AUTO */
        a[i] += b[i];
    }
}

/* The operation is still data-parallel, but this non-inlined call hides the
   operation from the loop vectorizer. This is a compiler limitation, not a
   true data dependence. */
static NOINLINE double opaque_add(double x, double y) {
    return x + y;
}

static NOINLINE void kernel_opaque(double *restrict a,
                                   const double *restrict b,
                                   size_t n) {
    for (size_t i = 0; i < n; ++i) { /* VEC_CHECK_OPAQUE */
        a[i] = opaque_add(a[i], b[i]);
    }
}

/* WORKED INTRINSICS EXAMPLE + STUDENT EDIT TARGET
   Explicit SIMD restores vector execution even though the opaque scalar form
   above could not be auto-vectorized. The scalar tail handles every N. */
static NOINLINE void kernel_intrinsics(double *restrict a,
                                       const double *restrict b,
                                       size_t n) {
    size_t i = 0;
#if !defined(LAB4_DISABLE_INTRINSICS) && defined(__AVX2__)
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        _mm256_storeu_pd(a + i, _mm256_add_pd(va, vb));
    }
#elif !defined(LAB4_DISABLE_INTRINSICS) && defined(__aarch64__)
    for (; i + 2 <= n; i += 2) {
        float64x2_t va = vld1q_f64(a + i);
        float64x2_t vb = vld1q_f64(b + i);
        vst1q_f64(a + i, vaddq_f64(va, vb));
    }
#elif !defined(LAB4_DISABLE_INTRINSICS) && defined(__SSE2__)
    for (; i + 2 <= n; i += 2) {
        __m128d va = _mm_loadu_pd(a + i);
        __m128d vb = _mm_loadu_pd(b + i);
        _mm_storeu_pd(a + i, _mm_add_pd(va, vb));
    }
#endif
    for (; i < n; ++i) a[i] = opaque_add(a[i], b[i]);
}

static const char *intrinsics_isa(void) {
#if !defined(LAB4_DISABLE_INTRINSICS) && defined(__AVX2__)
    return "AVX2";
#elif !defined(LAB4_DISABLE_INTRINSICS) && defined(__aarch64__)
    return "AArch64_NEON";
#elif !defined(LAB4_DISABLE_INTRINSICS) && defined(__SSE2__)
    return "SSE2";
#else
    return "scalar_fallback";
#endif
}

typedef enum { DEMO_AUTO, DEMO_OPAQUE, DEMO_INTRINSICS } demo_variant_t;

static void execute(demo_variant_t variant, double *a, const double *b,
                    size_t n) {
    if (variant == DEMO_AUTO) kernel_auto(a, b, n);
    else if (variant == DEMO_OPAQUE) kernel_opaque(a, b, n);
    else kernel_intrinsics(a, b, n);
}

static const char *variant_name(demo_variant_t variant) {
    if (variant == DEMO_AUTO) return "auto";
    if (variant == DEMO_OPAQUE) return "opaque";
    return "intrinsics";
}

static void run_demo(demo_variant_t variant, size_t n, uint64_t repeats) {
    double *a = alloc_vector(n);
    double *b = alloc_vector(n);
    initialize(a, b, n);
    double start = now_seconds();
    for (uint64_t r = 0; r < repeats; ++r) execute(variant, a, b, n);
    double elapsed = now_seconds() - start;
    printf("mode=%s isa=%s elements=%zu repeats=%" PRIu64
           " elapsed_s=%.6f ns_per_element=%.3f checksum=%.12g\n",
           variant_name(variant), intrinsics_isa(), n, repeats, elapsed,
           elapsed * 1e9 / ((double)n * (double)repeats), checksum(a, n));
    free(a);
    free(b);
}

static void verify(void) {
    const size_t n = 1003;
    const uint64_t repeats = 3;
    double *b = alloc_vector(n);
    double *reference = alloc_vector(n);
    double *candidate = alloc_vector(n);
    initialize(reference, b, n);
    memcpy(candidate, reference, n * sizeof(*candidate));
    for (uint64_t r = 0; r < repeats; ++r) kernel_auto(reference, b, n);

    demo_variant_t variants[] = {DEMO_OPAQUE, DEMO_INTRINSICS};
    for (size_t v = 0; v < sizeof(variants) / sizeof(variants[0]); ++v) {
        initialize(candidate, b, n);
        for (uint64_t r = 0; r < repeats; ++r)
            execute(variants[v], candidate, b, n);
        double max_error = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double error = fabs(reference[i] - candidate[i]);
            if (error > max_error) max_error = error;
        }
        printf("verify variant=%s isa=%s max_abs_error=%.3e %s\n",
               variant_name(variants[v]), intrinsics_isa(), max_error,
               max_error == 0.0 ? "PASS" : "FAIL");
        if (max_error != 0.0) exit(EXIT_FAILURE);
    }
    free(b);
    free(reference);
    free(candidate);
}

static void usage(const char *program) {
    fprintf(stderr, "Usage:\n"
                    "  %s auto|opaque|intrinsics ELEMENTS REPEATS\n"
                    "  %s verify\n", program, program);
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "verify") == 0) {
        verify();
        return EXIT_SUCCESS;
    }
    if (argc != 4) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }
    demo_variant_t variant;
    if (strcmp(argv[1], "auto") == 0) variant = DEMO_AUTO;
    else if (strcmp(argv[1], "opaque") == 0) variant = DEMO_OPAQUE;
    else if (strcmp(argv[1], "intrinsics") == 0) variant = DEMO_INTRINSICS;
    else {
        usage(argv[0]);
        return EXIT_FAILURE;
    }
    run_demo(variant, parse_size(argv[2], "ELEMENTS"),
             parse_repeats(argv[3]));
    return EXIT_SUCCESS;
}
