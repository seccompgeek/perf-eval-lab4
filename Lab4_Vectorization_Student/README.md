# Lab 4 — Vectorization, SIMD, SMT, and Cache-Aware Matrices

## Central question

**When does SIMD make a loop faster, and when do dependencies, SMT resource
sharing, or the memory hierarchy erase the expected gain?**

We begin with one simple data-parallel loop, inspect what the compiler actually
generated, and then add one complication at a time. By the end, we use loop
interchange, vectorization, and cache blocking together to optimize matrix
multiplication.

## Learning objectives

After this lab, we should be able to:

1. distinguish scalar execution, SIMD vectorization, threading, and SMT;
2. verify vectorization with compiler diagnostics and assembly—not timing alone;
3. distinguish a truly non-vectorizable dependence from a loop the compiler
   merely cannot analyze, and use simple SIMD intrinsics when appropriate;
4. explain why vector width is an upper bound rather than a promised speedup;
5. recognize loop-carried dependencies and strided memory accesses;
6. measure the tradeoff between SIMD-friendly code and cache reuse;
7. transform matrix loops so the innermost loop is both contiguous and
   vectorizable;
8. write guarded AVX2/SSE2/NEON intrinsics for a blocked matrix kernel and
   compare them with the compiler-generated SIMD; and
9. use cycles, retired instructions, IPC, and cache events to explain—not merely
   report—a timing result.

## Requirements

- 64-bit Linux on either x86-64 (Intel or AMD) or AArch64
- GCC or a compatible C11 compiler
- `make`, pthreads, and `objdump`
- Python 3 (standard library only) for the guided assembly report
- Linux `perf` for the counter-analysis section; if hardware-counter access is
  restricted, use instructor-provided output and report that limitation
- A 64-bit Raspberry Pi OS is required on Raspberry Pi 3B+, 4, or 5; confirm
  that `uname -m` prints `aarch64`.

Most of the main benchmark uses portable C and compiler auto-vectorization. The
guarded AVX2/SSE2 and AArch64 NEON paths let the same source also demonstrate
explicit SIMD on every supported machine.

There are two separate coding assignments:

| Stage | File and edit target | What we write | SIMD mechanism |
|---|---|---|---|
| 1B | `simd_intrinsics_demo.c`: `kernel_auto()`, `opaque_add()`, and `kernel_intrinsics()` | extend `a[i] += b[i]` to `a[i] += 2.0*b[i]` | **we write AVX2/SSE2/NEON intrinsics** |
| 6 | `lab4_bench.c`: only `mm_blocked_intrinsics()` | manually vectorize the supplied blocked kernel's inner `j` loop | **we write AVX2/SSE2/NEON intrinsics and compare them with `mm_blocked()`** |

Stage 1B is the small rehearsal: we learn the vector types, loads, arithmetic,
stores, architecture guards, and scalar tail. Stage 6 applies exactly those
mechanics to the already-blocked matrix kernel, so the new variable is manual
SIMD—not a simultaneous redesign of the loop order and tiling.

Record the machine before measuring:

```bash
uname -m
lscpu -e=CPU,CORE,SOCKET,NODE
lscpu | egrep 'Model name|Socket|Core|Thread|NUMA|L1d|L2|L3'
grep -m1 -Ei '^(flags|features)' /proc/cpuinfo
```

Logical CPUs that share the same `CORE` value are SMT siblings. We determine
these IDs before using `taskset`; adjacent CPU numbers are not universally
siblings.

## Build three controlled variants

```bash
make
make check
```

The Makefile builds the main source three ways and also builds `simd_demo`:

| Binary | Relevant flags | Purpose |
|---|---|---|
| `lab4_scalar` | `-O3 -march=native -fno-tree-loop-vectorize -fno-tree-slp-vectorize` | optimized scalar baseline |
| `lab4_auto` | `-O3 -march=native -ftree-loop-vectorize -ftree-slp-vectorize` | normal compiler auto-vectorization |
| `lab4_fast` | above plus `-ffast-math` | optional relaxed floating-point semantics |

Keeping `-O3` in both the scalar and auto-vectorized builds is important. A
comparison between `-O0` and `-O3` would mix vectorization with many unrelated
compiler optimizations.

### Compiler options used in this lab

The flags answer different questions; none of them alone proves that a loop was
vectorized:

| GCC option | Role in the experiment |
|---|---|
| `-O3` | Runs the optimizing pipeline. GCC includes loop and SLP vectorization at this level, but we still spell out the vectorizer flags so the experimental knob is visible. |
| `-ftree-loop-vectorize` | Enables vectorization across independent loop iterations. This is the main mechanism for AXPY and the inner matrix loop. |
| `-ftree-slp-vectorize` | Enables basic-block or superword-level vectorization of similar adjacent operations. |
| `-fno-tree-loop-vectorize -fno-tree-slp-vectorize` | Disables both vectorizers while retaining `-O3`; this creates the optimized scalar control. |
| `-march=native` | Lets a native x86-64 or AArch64 build use the current CPU's ISA. It may expose AVX2 on one x86 machine but only SSE2 on another. The resulting binary is not intended for a different CPU. |
| `-march=armv8-a+simd` | Cross-compilation baseline that explicitly permits AArch64 Advanced SIMD/NEON. |
| `-fopt-info-vec-all=FILE` | Writes GCC's optimized and missed vectorization decisions to a report. It diagnoses code generation; it does not enable vectorization. |
| `-ffast-math` | Optionally relaxes floating-point semantics and can make reductions easier to vectorize. It is not required for the ordinary element-wise loops in this lab. |

The controlled commands represented by the Makefile are therefore equivalent
to:

```bash
# Optimized scalar control
cc -O3 -march=native \
  -fno-tree-loop-vectorize -fno-tree-slp-vectorize ...

# Auto-vectorized build
cc -O3 -march=native \
  -ftree-loop-vectorize -ftree-slp-vectorize ...
```

`-march` determines which instructions the compiler is allowed to choose;
`-ftree-*vectorize` asks it to vectorize profitable legal code. Neither means a
particular loop succeeded, which is why we still use `make report` and
`make assembly`.

The same native build commands work on x86-64 and a 64-bit Raspberry Pi:

```bash
make
make check
file lab4_auto
```

To check AArch64 compilation from an x86 development machine, install an
AArch64 Linux cross-compiler and run:

```bash
make cross-aarch64
file lab4_auto.aarch64
```

The cross target uses `-march=armv8-a+simd`, a common baseline for the Cortex-A53,
A72, and A76 cores in the Pi 3B+, 4, and 5. Cross-compilation confirms that the
source and flags are accepted; run `make check` on the actual board to test its
OS, libraries, compiler, and hardware.

Generate GCC's vectorization decisions and assembly:

```bash
make report
make assembly
less vectorization-report.txt
less assembly-guide.txt
```

`make assembly` still creates the raw compiler outputs—`lab4_scalar.s`,
`lab4_auto.s`, `lab4_fast.s`, and `simd_demo.s`—but we do not have to search
those large files unaided. It also creates `assembly-guide.txt`, organized into
seven checkpoints:

1. AXPY: scalar versus compiler-generated SIMD;
2. the loop-carried dependency;
3. the arithmetic-heavy polynomial;
4. automatic, opaque, and manual-intrinsics versions;
5. fused versus split loops;
6. row versus column matrix traversal; and
7. `ijk`, `ikj`, auto-vectorized blocked, and manually vectorized blocked matrix
   multiplication.

For every target function, the guide places the numbered C source immediately
above the emitted assembly. When both builds matter, the scalar and
auto-vectorized versions appear consecutively. Lines beginning with `>>>` are
likely packed-vector instructions for the current x86-64 or AArch64 compiler
target. The marker is a reading aid—not proof by itself—so we pair it with
`make report` or `make vec-check` and the measured time.

On an AVX2 x86 machine, a section looks like this:

```text
FUNCTION: kernel_axpy
SOURCE: lab4_bench.c
  89 | for (size_t i = begin; i < end; ++i) {
  90 |     a[i] += alpha * b[i];

SCALAR BUILD: lab4_scalar.s
    vmulsd ...                 # one double

AUTO-VECTORIZED BUILD: lab4_auto.s
>>> vmulpd ...                 # packed doubles
>>> vaddpd ...
```

For the focused automatic-versus-manual example, one command prints only the
important verdicts and then checks correctness:

```bash
make vec-check
```

It should report the `AUTO` loop as optimized, the `OPAQUE` loop as missed, and
the manual path as AVX2, SSE2, or AArch64 NEON. The complete focused diagnostic
is saved in `demo-vectorization-report.txt`.

If we want to inspect beyond the selected functions, the raw `.s` files remain
available. On x86, useful searches include `xmm`, `ymm`, `vadd`, `vmul`, and
`vfmadd`. On AArch64, look for vector registers such as `v0.2d` and instructions
such as `fadd`, `fmul`, `fmla`, `ld1`, and `st1`. Instruction names and vector
widths vary across architectures.

## Measurement protocol

For every A/B comparison we keep the input size, useful work, compiler baseline,
CPU placement, and checksum fixed. We change one mechanism at a time.

1. Calibrate so each measured configuration lasts at least about 0.1 seconds.
2. Run each configuration at least three times and use the median.
3. Use the program's `elapsed_s`; allocation and initialization are outside it.
4. Confirm that checksums match before comparing speed.
5. Pair timing with compiler evidence: was the intended loop actually vectorized?
6. Report speedup as `T_baseline / T_optimized`, not as a percentage.

`perf stat` measures the entire process, so its counters include allocation and
initialization even though `elapsed_s` does not. Use counter differences as
supporting evidence rather than pretending they describe only the timed region.

## Lab map (about 105–115 minutes)

| Stage | Topic | Suggested time |
|---|---|---:|
| 0 | topology, ISA, build, and calibration | 8 min |
| 1 | scalar loop → compiler-generated SIMD | 12 min |
| 1B | compiler limitation → explicit SIMD intrinsics | 10 min |
| 2 | dependency: why a loop does not vectorize | 8 min |
| 3 | SIMD and threading: SMT or multicore | 12 min |
| 4 | vectorization versus cache reuse | 15 min |
| 5 | vectorization and matrix locality | 10 min |
| 6 | blocked matrix multiplication: auto SIMD versus intrinsics | 25 min |
| 7 | `perf stat`: cycles, instructions, IPC, and cache evidence | 10 min |
| 8 | causal explanation and submission | 5 min |

## Stage 1 — Start with a data-parallel loop

The `axpy` kernel performs this update repeatedly:

```c
for (size_t i = 0; i < n; ++i)
    a[i] += alpha * b[i];
```

Each iteration is independent, so the compiler can operate on several `double`
values with one SIMD instruction.

```bash
./lab4_scalar axpy 1000000 200
./lab4_auto   axpy 1000000 200
```

Record the median time, calculate SIMD gain, and then confirm the compiler report.
Do not expect speedup to equal lane count. AVX2 commonly holds four doubles,
whereas a 128-bit AArch64 NEON vector holds two; this kernel does little
arithmetic per byte and can become limited by loads and stores on either ISA.

Questions:

1. Which source line was vectorized, and which vector width did GCC report?
2. In checkpoint 1 of `assembly-guide.txt`, which packed instructions appear in
   the auto build but not the scalar build?
3. Why is measured speedup smaller than the number of lanes?

## Stage 1B — When auto-vectorization needs help

A missed vectorization report does not always mean that SIMD is illegal. Compare
these two loops in `simd_intrinsics_demo.c`; both perform the same independent
element-wise addition:

```c
/* Easy for the compiler to see. */
for (size_t i = 0; i < n; ++i)
    a[i] += b[i];

/* Still data-parallel, but the non-inlined call hides the operation. */
for (size_t i = 0; i < n; ++i)
    a[i] = opaque_add(a[i], b[i]);
```

Run the focused check, then time all three implementations with identical work:

```bash
make vec-check
./simd_demo auto       1000000 200
./simd_demo opaque     1000000 200
./simd_demo intrinsics 1000000 200
```

Open `simd_intrinsics_demo.c` and search for `STUDENT INTRINSICS EXERCISE` or
`kernel_intrinsics`. The starter already contains this small worked intrinsic
example—we are not expected to invent the first vector loop from a blank file.
On AVX2 it contains:

```c
for (; i + 4 <= n; i += 4) {
    __m256d va = _mm256_loadu_pd(a + i);
    __m256d vb = _mm256_loadu_pd(b + i);
    _mm256_storeu_pd(a + i, _mm256_add_pd(va, vb));
}
```

The corresponding AArch64 NEON branch is:

```c
for (; i + 2 <= n; i += 2) {
    float64x2_t va = vld1q_f64(a + i);
    float64x2_t vb = vld1q_f64(b + i);
    vst1q_f64(a + i, vaddq_f64(va, vb));
}
```

`kernel_intrinsics()` therefore expresses the addition directly with the active
machine's intrinsics. The operation is the same on each path:

| Machine | Explicit vector type | Load / add / store |
|---|---|---|
| AVX2 x86-64 | `__m256d` | `_mm256_loadu_pd` / `_mm256_add_pd` / `_mm256_storeu_pd` |
| older x86-64 | `__m128d` | `_mm_loadu_pd` / `_mm_add_pd` / `_mm_storeu_pd` |
| AArch64 | `float64x2_t` | `vld1q_f64` / `vaddq_f64` / `vst1q_f64` |

Exercise:

1. Predict which of `auto` and `opaque` GCC will auto-vectorize.
2. Use `make vec-check` to test the prediction; timing alone is not proof.
3. In checkpoint 4 of `assembly-guide.txt`, find the load, add, store, loop
   increment, and scalar tail for `kernel_intrinsics()`.
4. Record three median times and confirm all checksums match.
5. Explain why intrinsics can help `opaque`, but cannot simply make the true
   loop-carried dependency in Stage 2 run concurrently.

### Student intrinsics coding task

Change the operation from `a[i] += b[i]` to `a[i] += 2.0 * b[i]` in all of these
places in `simd_intrinsics_demo.c`:

1. `kernel_auto()`—write ordinary scalar C and let the compiler vectorize it;
2. `opaque_add()`—preserve the non-inlined function-call case;
3. the AVX2 branch of `kernel_intrinsics()`—use `_mm256_set1_pd`,
   `_mm256_mul_pd`, and `_mm256_add_pd`;
4. the SSE2 branch—use the corresponding `_mm_set1_pd`, `_mm_mul_pd`, and
   `_mm_add_pd` operations;
5. the NEON branch—use `vmulq_n_f64` and `vaddq_f64`; and
6. the scalar tail at the bottom of `kernel_intrinsics()`—confirm that its call
   to the updated `opaque_add()` performs the scaled operation too.

Then rebuild and verify before timing:

```bash
make
./simd_demo verify
make vec-check
make assembly                 # inspect checkpoint 4
./simd_demo auto       1000000 200
./simd_demo opaque     1000000 200
./simd_demo intrinsics 1000000 200
```

This is the required handwritten-intrinsics exercise. On a single machine we
can execute only its native branch, but the submitted source should retain all
three guarded implementations so the instructor can cross-compile it for the
other architecture.

Intrinsics are not automatically better than auto-vectorization. Prefer clear
portable code when the compiler already emits the desired SIMD. Use intrinsics
only after the report and assembly identify a real compiler limitation and the
extra ISA-specific maintenance is justified.

## Stage 2 — Add a loop-carried dependency

```c
for (size_t i = 1; i < n; ++i)
    a[i] = 0.5 * a[i - 1] + b[i];
```

Iteration `i` needs the value produced by iteration `i-1`. Running ordinary SIMD
lanes simultaneously would violate that dependency.

```bash
./lab4_scalar dependency 1000000 20
./lab4_auto   dependency 1000000 20
```

Use `vectorization-report.txt` to quote the compiler's reason, then compare the
two versions in checkpoint 2 of `assembly-guide.txt`. Similar times and similar
loop bodies are expected; a small difference is measurement noise or another
compiler effect, not evidence that the dependency became SIMD.

## Stage 3 — SIMD and threading: SMT where available

SIMD and SMT expose different kinds of parallelism:

- **SIMD:** one instruction performs the same operation on multiple data lanes
  inside one hardware thread.
- **SMT:** two software threads share one physical core's front end, execution
  units, caches, and load/store resources.

The `poly` kernel performs a longer independent arithmetic chain per element, so
SIMD has useful work to pack into lanes.

First compare scalar and SIMD on one logical CPU:

```bash
taskset -c CPU_A ./lab4_scalar poly 1 2000000 100
taskset -c CPU_A ./lab4_auto   poly 1 2000000 100
```

On an x86 machine with SMT enabled, keep total work fixed and use the two SMT
siblings of one physical core:

```bash
taskset -c CPU_A       ./lab4_auto poly 1 2000000 100
taskset -c CPU_A,CPU_B ./lab4_auto poly 2 2000000 100
```

Repeat with `lab4_scalar`. On many machines, SIMD gives the larger gain because
SMT shares the same execution units; SMT may still hide stalls and improve
throughput. Some virtualized systems do not expose reliable SMT placement, so we
report the topology and observed result rather than forcing the expected story.

Raspberry Pi 3B+, 4, and 5 cores expose one hardware thread per core, so there
are no SMT siblings. Record the SMT experiment as **not applicable**, then retain
the fixed-work comparison using a second physical core:

```bash
taskset -c CPU_A       ./lab4_auto poly 1 2000000 100
taskset -c CPU_A,CPU_C ./lab4_auto poly 2 2000000 100
```

Call this a multicore result, not an SMT result. Raspberry Pi runs are also more
sensitive to thermal throttling, so record temperature and frequency state when
results drift across repetitions.

## Stage 4 — Vectorization versus cache reuse

This experiment adapts the motivating example from Shen, Chabbi, and Liu's
paper, *An Evaluation of Vectorization and Cache Reuse Tradeoffs on Modern CPUs*.

The fused form reuses `B[i]` immediately on the next iteration, but the resulting
loop-carried memory dependence blocks ordinary auto-vectorization:

```c
for (size_t i = 1; i < n; ++i) {
    A[i] = 3.0 * B[i - 1];
    B[i] = 0.5 * B[i] + C[i];
}
```

The split form distributes the loop into two independent, vectorizable passes:

```c
for (size_t i = 1; i < n; ++i) B[i] = 0.5 * B[i] + C[i];
for (size_t i = 1; i < n; ++i) A[i] = 3.0 * B[i - 1];
```

Start with a working set that fits in cache:

```bash
./lab4_auto tradeoff fused 4096 20000
./lab4_auto tradeoff split 4096 20000
```

Then exceed the last-level cache:

```bash
./lab4_auto tradeoff fused 8000000 10
./lab4_auto tradeoff split 8000000 10
```

Before interpreting the timing, open checkpoint 5 of `assembly-guide.txt` and
confirm that the expected packed instructions appear in `tradeoff_split` but not
in `tradeoff_fused`.

Repeat with `lab4_scalar` to separate SIMD gain from the source-level locality
change. The central question is not “which version is always best?” It is “where
does the winner change, and what resource explains the crossover?” Try several
sizes between the two endpoints and compare the printed `working_set_MiB` with
the cache capacities reported by `lscpu`.

## Stage 5 — Return to matrix locality

Both matrix modes update every element with the same arithmetic. Only traversal
order changes:

```bash
./lab4_scalar matrix row 2048 4
./lab4_auto   matrix row 2048 4
./lab4_scalar matrix col 2048 4
./lab4_auto   matrix col 2048 4
```

The row inner loop has unit-stride accesses and vectorizes. The column inner loop
uses a stride of `N * sizeof(double)` and normally does not vectorize. More
importantly, it wastes cache lines and stresses address translation. Compare all
four runs so we can distinguish SIMD gain from locality gain. Checkpoint 6 of
`assembly-guide.txt` puts `matrix_row` and `matrix_col` source and assembly next
to each other so we can connect unit stride to the emitted loop body.

## Stage 6 — Matrix multiplication challenge

All variants compute the same `C = A × B` and report approximately `2N³` floating-
point operations.

### Which function do we modify?

Open `lab4_bench.c` and search for the banner `STUDENT EDIT TARGET`. For this
exercise, we modify **only the SIMD TODOs inside `mm_blocked_intrinsics()`**:

| Function | Role | Modify it? |
|---|---|---|
| `mm_ijk()` | slow `ijk` baseline | no |
| `mm_ikj()` | supplied loop-interchanged reference | no |
| `mm_blocked()` | supplied blocked `ikj`; GCC auto-vectorizes its inner `j` loop | no |
| `mm_blocked_intrinsics()` | same blocking and loop order; we manually vectorize its inner `j` loop | **yes** |

Keeping the three reference functions unchanged gives us stable baselines. The
command-line variant `blocked` runs the compiler-vectorized `mm_blocked()`, while
`intrinsics` runs the code we write in `mm_blocked_intrinsics()`.

The outer tile loops and scalar tail are already supplied. Auto-vectorization is
disabled only for `mm_blocked_intrinsics()`, so packed instructions in that
function must come from our explicit intrinsics. This makes the comparison
unambiguous: the algorithms, tile size, and useful work match; only the SIMD
mechanism differs.

```bash
./lab4_auto mm ijk     512 64
./lab4_auto mm ikj     512 64
./lab4_auto mm blocked 512 64
./lab4_auto mm intrinsics 512 64
```

### How `ijk`, `ikj`, and blocked differ

All three variants perform the same matrix multiplication:

```text
C[i][j] = sum over k of A[i][k] * B[k][j]
```

They differ in the order in which they visit the matrices. Because our matrices
use row-major storage, changing the innermost loop changes both cache behavior
and the loop that the compiler can try to vectorize.

`ikj` is itself a cache-locality optimization: it uses **loop interchange** to
improve spatial locality. It is not the same optimization as cache
blocking/tiling, which improves temporal and capacity locality. The blocked
version combines the two ideas.

#### `ijk`: contiguous `A`, strided `B`, scalar reduction

```c
for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < n; ++j) {
        double sum = 0.0;
        for (size_t k = 0; k < n; ++k)
            sum += A[i*n + k] * B[k*n + j];
        C[i*n + j] = sum;
    }
```

**Cache behavior:** `A[i*n+k]` moves contiguously through one row of `A`, but
`B[k*n+j]` moves down a column of row-major `B`. Consecutive `B` accesses are
`N*sizeof(double)` bytes apart. A cache line is fetched for one value and most
of its neighboring values are not used by the current inner loop. Large matrices
also stress the TLB and repeatedly bring `B` data back through the cache
hierarchy. `C[i][j]` is favorable—it stays in the scalar `sum` and is stored once.

**SIMD behavior:** every `k` iteration updates the same scalar `sum`, creating a
reduction dependency. The compiler would also need strided loads or gathers for
`B`, followed by a horizontal reduction of the vector lanes. Under our normal
floating-point rules, ordinary packed SIMD is therefore unlikely for this loop.

The inner `k` loop therefore has all four of these properties:

- contiguous access through a row of `A`;
- strided access down a column of `B`;
- a scalar reduction into `sum`;
- poor ordinary SIMD opportunities.

#### `ikj`: contiguous `B` and `C`, with SIMD across `j`

Interchanging the `j` and `k` loops gives:

```c
for (size_t i = 0; i < n; ++i)
    for (size_t k = 0; k < n; ++k) {
        double aik = A[i*n + k];
        for (size_t j = 0; j < n; ++j)
            C[i*n + j] += aik * B[k*n + j];
    }
```

`C` is initialized to zero before this function runs.

**Cache behavior:** for fixed `i` and `k`, both `B[k*n+j]` and `C[i*n+j]` move
contiguously through rows. We use every value brought in by each cache line.
`A[i*n+k]` is loaded once into `aik` and reused for the whole inner loop. The
same `C` row is updated repeatedly as `k` changes and may remain in a nearby
cache. This loop interchange is therefore itself a cache-locality optimization,
primarily improving spatial locality and cache-line utilization. It does not,
however, bound the working set when the matrices are much larger than cache.

For fixed `i` and `k`, the inner `j` loop:

- reads `B[k][j]` contiguously;
- reads and writes `C[i][j]` contiguously;
- reuses `A[i][k]` from the scalar `aik`, which can remain in a register;
- has independent `j` iterations; and
- can therefore be auto-vectorized.

**SIMD behavior:** adjacent `j` iterations update different elements of `C`, so
they are independent:

```text
C[i][j+0] += aik * B[k][j+0]
C[i][j+1] += aik * B[k][j+1]
C[i][j+2] += aik * B[k][j+2]
C[i][j+3] += aik * B[k][j+3]
```

Conceptually, the compiler transforms those iterations into:

```text
broadcast A[i][k]
load multiple adjacent B values
load multiple adjacent C values
multiply and add corresponding lanes
store multiple adjacent C values
```

For example, an AVX2 vector holds four doubles:

```text
C vector = [C[i][j+0], C[i][j+1], C[i][j+2], C[i][j+3]]
B vector = [B[k][j+0], B[k][j+1], B[k][j+2], B[k][j+3]]
A vector = [aik,         aik,         aik,         aik        ]

C vector = C vector + A vector * B vector
```

Representative AVX2 assembly for that operation resembles:

```asm
vbroadcastsd  aik,    %ymm0
vmovupd       B[j],   %ymm1
vmulpd        %ymm0,  %ymm1, %ymm1
vaddpd        C[j],   %ymm1, %ymm1
vmovupd       %ymm1,  C[j]
```

The compiler may replace the packed multiply and add with a packed FMA when the
target and compiler options allow it. AArch64 NEON performs the same pattern
with two double-precision lanes rather than four.

The important dependency is **across successive `k` iterations for each one
`C[i][j]` value**. It is not between adjacent `j` iterations:

```text
C[i][j] is independent of C[i][j+1]
```

The compiler can therefore vectorize across `j` while leaving `k` sequential.
This is the useful SIMD pattern: one scalar broadcast, two contiguous vectors,
and independent lanes. The `restrict` qualifiers also tell the compiler that
`A`, `B`, and `C` do not overlap, removing another possible barrier to
vectorization.

#### Blocked: keep the `ikj` access pattern inside cache-sized tiles

Plain `ikj` still traverses large portions of `B` repeatedly. When the matrices
exceed cache capacity, data can be evicted before it is reused. The supplied
blocked implementation restricts work to submatrices while retaining the
SIMD-friendly `ikj` order:

```c
for (size_t ii = 0; ii < n; ii += block)
    for (size_t kk = 0; kk < n; kk += block)
        for (size_t jj = 0; jj < n; jj += block)
            for (size_t i = ii; i < min(ii + block, n); ++i)
                for (size_t k = kk; k < min(kk + block, n); ++k) {
                    double aik = A[i*n + k];

                    for (size_t j = jj;
                         j < min(jj + block, n); ++j)
                        C[i*n + j] += aik * B[k*n + j];
                }
```

Here `min(...)` represents the end-of-matrix checks used for partial edge tiles.

**Cache behavior:** rather than operating over full matrix dimensions at once,
we repeatedly work on smaller regions of `A`, `B`, and `C`. Ideally, the active
tiles remain in cache long enough to be reused before eviction. A rough
three-tile footprint for square tiles is `3 * block² * sizeof(double)`, although
associativity, other data, and the actual loop order also matter. A block that is
too large loses the cache benefit; a very small block adds loop overhead and
provides little work per tile.

**SIMD behavior:** blocking alone does not guarantee vectorization. This blocked
implementation deliberately preserves `j` as the innermost loop, so `B` and `C`
remain contiguous and adjacent `j` iterations remain independent. The compiler
can use the same packed SIMD operation as in plain `ikj`, now within each tile.
Partial tiles are handled by the bounds above. When a tile width is not a
multiple of the hardware vector width, compiler-generated cleanup code handles
the remaining elements in `mm_blocked()`. We will write the corresponding
scalar-tail logic explicitly in `mm_blocked_intrinsics()`.

### Interchange versus tiling

| Technique | Main effect |
|---|---|
| `ijk` to `ikj` loop interchange | Improves spatial locality and exposes SIMD |
| Tiling/blocking | Keeps reusable submatrices within cache |
| Tiled `ikj` | Combines cache reuse, contiguous access, and SIMD |

The accurate progression is therefore:

```text
ijk
  |
  | loop interchange
  v
ikj: spatial locality + vectorization
  |
  | cache blocking
  v
tiled ikj: spatial locality + temporal/capacity locality + vectorization
```

Loop interchange and blocking are both cache optimizations, but they address
different problems. `ikj` repairs the access direction and cache-line usage;
blocking restricts the active working set so reused data has a better chance of
remaining in cache.

Block size is machine-dependent:

```bash
for b in 16 32 48 64 96 128; do
  ./lab4_auto mm blocked 512 "$b"
done
```

### Our task

`mm_blocked_intrinsics()` already has the same `ii`/`kk`/`jj` tiling and
`i`/`k`/`j` loop order as `mm_blocked()`. Its initial inner loop is a correct
scalar fallback. We replace the architecture-specific TODOs immediately above
that loop; the existing scalar loop must remain to handle leftover elements and
machines without one of the selected SIMD ISAs.

For each fixed `i` and `k`, start with `j = jj` and implement the active branch:

| Target | Lanes | Operations required inside the `j` loop |
|---|---:|---|
| AVX2 x86-64 | 4 doubles | `_mm256_set1_pd`, `_mm256_loadu_pd`, `_mm256_mul_pd`, `_mm256_add_pd`, `_mm256_storeu_pd` |
| SSE2 x86-64 | 2 doubles | `_mm_set1_pd`, `_mm_loadu_pd`, `_mm_mul_pd`, `_mm_add_pd`, `_mm_storeu_pd` |
| AArch64 NEON | 2 doubles | `vdupq_n_f64`, `vld1q_f64`, `vmulq_f64`, `vaddq_f64`, `vst1q_f64` |

The vector loop must stop before crossing `j_end`. The supplied scalar loop then
continues from the current `j` value to `j_end`, covering both a short final
vector and a partial edge tile.

1. Run the unmodified starter first:

   ```bash
   make
   ./lab4_auto verify
   ./lab4_auto mm blocked    512 64
   ./lab4_auto mm intrinsics 512 64
   ```

   Initially, `intrinsics` uses only the deliberately non-auto-vectorized scalar
   fallback and should normally lose to `blocked`.
2. In the AVX2 branch, broadcast `aik`, load four adjacent `B` values and four
   adjacent `C` values, calculate `C + aik*B`, store the result, and advance `j`
   by four.
3. Implement the equivalent two-double operation in the SSE2 and NEON branches.
   The `#if` guards mean only the current machine's branch is compiled natively;
   `make cross-aarch64` checks the NEON branch from an x86 host when that compiler
   is installed.
4. Rebuild and run `./lab4_auto verify`. The verification size deliberately
   exercises partial tiles and scalar tails. Do not time incorrect code.
5. Run `make report && make assembly`. In checkpoint 7:

   - use the compiler report to confirm that `mm_blocked()` auto-vectorized;
   - find packed instructions in `mm_blocked()` generated from ordinary C; and
   - find packed instructions in `mm_blocked_intrinsics()` that correspond to
     the intrinsics we wrote.

6. Compare automatic and manual SIMD with identical `N` and `BLOCK` values:

   ```bash
   ./lab4_auto mm blocked    512 64
   ./lab4_auto mm intrinsics 512 64
   ```

   Run each at least three times and compare medians. Handwritten intrinsics are
   not guaranteed to beat the compiler; equal performance is an informative
   result when both versions generate essentially the same vector loop.
7. Sweep the same block sizes for both variants. Do not compare different block
   sizes and attribute the difference only to automatic versus manual SIMD:

   ```bash
   for b in 16 32 48 64 96 128; do
     ./lab4_auto mm blocked    512 "$b"
     ./lab4_auto mm intrinsics 512 "$b"
   done
   ```

## Stage 7 — Explain the timing with `perf stat`

Elapsed time tells us which version won. Hardware counters help explain why. The
Makefile supplies a portable starting point:

```bash
make perf-stat PERF_CPU=CPU_A \
  PERF_COMMAND='./lab4_scalar mm blocked 512 64'

make perf-stat PERF_CPU=CPU_A \
  PERF_COMMAND='./lab4_auto mm blocked 512 64'
```

These two commands execute the same blocked source algorithm, matrix size, tile
size, and useful arithmetic. The intended difference is compiler-generated SIMD.
The default event set records `task-clock`, cycles, retired instructions,
branches, branch misses, and the generic cache events. `perf` normally prints IPC
beside `instructions` when both instructions and cycles are counted.

Read the counters as a group:

| Counter | Question it helps answer | Important caution |
|---|---|---|
| elapsed time / cycles | Did execution actually become faster? | Frequency and migration affect cycles; pin the same CPU. |
| instructions | Did SIMD or a loop transformation reduce retired instructions? | One SIMD instruction can perform several arithmetic operations. Equal useful work does **not** imply equal instruction count. |
| IPC | How many instructions retired per cycle? | Higher IPC is not automatically faster; SIMD can finish sooner with fewer instructions and a lower IPC. |
| cache misses / references | Did data movement change? | Generic events are architecture-dependent and do not mean “all L1/L2 accesses.” |

After implementing `mm_blocked_intrinsics()`, compare automatic and manual SIMD
inside the same `lab4_auto` binary:

```bash
make perf-stat PERF_CPU=CPU_A \
  PERF_COMMAND='./lab4_auto mm blocked 512 64'

make perf-stat PERF_CPU=CPU_A \
  PERF_COMMAND='./lab4_auto mm intrinsics 512 64'
```

These two runs use the same blocked algorithm. Differences in retired
instructions and cycles now reflect compiler-generated versus explicitly
written SIMD, including any differences in unrolling and tail handling.

Next hold the auto-vectorized binary fixed and compare loop organization:

```bash
make perf-stat PERF_CPU=CPU_A PERF_COMMAND='./lab4_auto mm ijk 512 64'
make perf-stat PERF_CPU=CPU_A PERF_COMMAND='./lab4_auto mm ikj 512 64'
make perf-stat PERF_CPU=CPU_A PERF_COMMAND='./lab4_auto mm blocked 512 64'
```

Here the useful matrix multiplication is the same, but both the instruction
stream and memory traffic may change. We expect `ikj` and `blocked` to spend
fewer cycles than `ijk`; blocking should reduce long-latency traffic when its
working tiles fit the cache. We test those hypotheses rather than assuming every
counter moves in one predetermined direction.

### Ask the PMU for cache-level detail

The most portable way to request the cache events supported by the current
machine is:

```bash
make perf-stat PERF_CPU=CPU_A PERF_STAT_FLAGS=-d \
  PERF_COMMAND='./lab4_auto mm blocked 512 64'
```

`perf stat -d` commonly adds L1-data-cache and last-level-cache statistics, but
the available events depend on the CPU and Linux kernel. It does not guarantee an
L2 result. Discover the native names before requesting explicit L1/L2/LLC events:

```bash
perf list cache
perf list | grep -Ei 'L1|L2|LLC|cache.*(miss|refill)'
```

For example, a machine may accept `L1-dcache-loads` and
`L1-dcache-load-misses`; some Intel machines expose `l2_rqsts.*`, while AArch64
PMUs often use names containing `l2d_cache` and `l2d_cache_refill`. We use only
events shown by `perf list` on the machine being measured and record any
`<not supported>` result. Do not copy raw event numbers between Intel, AMD, and
Raspberry Pi systems.

After confirming that an event pair exists, we can request it explicitly:

```bash
make perf-stat PERF_CPU=CPU_A \
  PERF_STAT_FLAGS='-e L1-dcache-loads,L1-dcache-load-misses' \
  PERF_COMMAND='./lab4_auto mm blocked 512 64'
```

If normal execution is denied, repeat the same command with permission according
to the lab machine's policy, for example:

```bash
make perf-stat PERF='sudo perf' PERF_CPU=CPU_A \
  PERF_COMMAND='./lab4_auto mm blocked 512 64'
```

Use a small event set—or separate core and cache runs—to avoid heavy counter
multiplexing. `perf stat` measures the entire process, including allocation and
initialization, whereas the benchmark's `elapsed_s` covers only the timed kernel.
Choose enough work that the kernel dominates, and use counters as supporting
causal evidence rather than as a replacement for elapsed time.

## Floating-point semantics

`lab4_fast` allows reassociation and contraction that strict IEEE-style compilation
may reject. It can expose additional vectorization, especially for reductions, but
it may change rounding, NaN, infinity, and signed-zero behavior. We use it only as
an explicit experiment—not as a default “make it faster” switch.

## Deliverable

Submit:

1. a table of scalar, auto-vectorized, and where relevant SMT results;
2. the `auto` / `opaque` / `intrinsics` table with the focused compiler verdict;
3. one excerpt from `assembly-guide.txt` or a compiler report proving successful
   vectorization;
4. a working-set-size plot for fused versus split code;
5. a four-way matrix table: scalar/auto × row/column;
6. a completed `mm_blocked_intrinsics()` with guarded AVX2, SSE2, NEON, and
   scalar-tail paths that passes verification;
7. a block-size versus GFLOP/s plot containing both `blocked` and `intrinsics`;
8. one `perf stat` table comparing blocked scalar versus blocked auto,
   auto-vectorized blocked versus manual intrinsics, and `ijk` versus `ikj`
   versus blocked; and
9. three explanations in the form **claim → evidence → mechanism → next test**.

## References and design provenance

- Du Shen, Milind Chabbi, and Xu Liu, “An Evaluation of Vectorization and Cache
  Reuse Tradeoffs on Modern CPUs,” PMAM 2018,
  <https://doi.org/10.1145/3178442.3178445>.
- MIT OpenCourseWare 6.172, Homework 3: Vectorization,
  <https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/072651a8229a63376d5720c9a500ae45_MIT6_172F18hw3.pdf>.
- GCC vectorization options and diagnostics,
  <https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html>.
- LLVM Loop Vectorizer documentation,
  <https://llvm.org/docs/Vectorizers.html>.

The benchmark code and lab sequence here are original course materials. The MIT
assignment inspired the evidence workflow—controlled compiler flags, diagnostics,
assembly inspection, runtime bounds, strides, and reductions—while the PMAM paper
motivated the working-set crossover experiment.
