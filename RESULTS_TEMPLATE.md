# Lab 4 results worksheet

## Machine and build

- CPU model:
- physical cores / logical CPUs:
- SMT sibling pair used, or N/A on Raspberry Pi:
- L1d / L2 / L3:
- compiler and version:
- vector ISA observed in assembly:

## Stage 1 — scalar versus auto-vectorized

| Kernel | Binary | Median time | Speedup | Vectorized? | Evidence |
|---|---|---:|---:|---|---|
| axpy | scalar | | 1.00 | no | |
| axpy | auto | | | | |
| dependency | scalar | | 1.00 | no | |
| dependency | auto | | | | |
| poly | scalar | | 1.00 | no | |
| poly | auto | | | | |

## Stage 1B — automatic versus explicit SIMD

Coding checkpoint: change all three paths to `a[i] += 2.0*b[i]`, including the
AVX2, SSE2, NEON, and scalar-tail code guarded in `kernel_intrinsics()`.

| Variant | Median time | Compiler verdict | ISA/instructions | Checksum |
|---|---:|---|---|---:|
| `auto` | | auto-vectorized / missed | | |
| `opaque` | | auto-vectorized / missed | scalar call | |
| `intrinsics` | | explicit—not auto-vectorized | | |

Why can intrinsics recover SIMD for `opaque`, while they cannot simply remove
the real dependency in the `dependency` kernel?

## Stage 3 — SIMD versus SMT or multicore

| Binary | Threads | Placement | Median time | Speedup |
|---|---:|---|---:|---:|
| scalar | 1 | one logical CPU | | 1.00 |
| scalar | 2 | SMT siblings or two Pi cores | | |
| auto | 1 | one logical CPU | | 1.00 |
| auto | 2 | SMT siblings or two Pi cores | | |

## Stage 4 — vectorization/locality crossover

| Elements | Working set | Binary | Fused time | Split time | Winner |
|---:|---:|---|---:|---:|---|
| | | scalar | | | |
| | | auto | | | |
| | | scalar | | | |
| | | auto | | | |

## Stage 5 — matrix traversal

| Binary | Row time | Column time | Column/row | Explanation |
|---|---:|---:|---:|---|
| scalar | | | | |
| auto | | | | |

## Stage 6 — matrix multiplication

`blocked` is the supplied compiler-vectorized implementation. We modify only
the guarded SIMD TODOs in `mm_blocked_intrinsics()` and retain its scalar tail.
Use the same matrix and block size when comparing automatic and manual SIMD.

| Variant | Block | SIMD mechanism / ISA | Median time | GFLOP/s | Speedup vs. `ijk` | Verified? |
|---|---:|---|---:|---:|---:|---|
| `ijk` | — | compiler verdict: | | | 1.00 | |
| `ikj` | — | compiler verdict: | | | | |
| `blocked` | | compiler-generated: | | | | |
| `intrinsics` | | manually written: | | | | |

For `mm_blocked()` and `mm_blocked_intrinsics()`, identify the packed load,
broadcast, multiply/add, store, loop increment, and scalar cleanup in checkpoint
7 of `assembly-guide.txt`. Did handwritten intrinsics beat the compiler? Explain
the result from the generated loops rather than assuming manual code must win.

## Stage 7 — hardware-counter evidence

Use the same CPU, input size, block size, repetitions, and event set for every
row. Write `N/A` rather than inventing a value when the PMU does not expose an
event on our machine.

| Binary / variant | Time | Cycles | Instructions | IPC | Generic cache misses | L1D miss rate | L2 or LLC miss rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| scalar / blocked | | | | | | | |
| auto / blocked | | | | | | | |
| auto / intrinsics | | | | | | | |
| auto / `ijk` | | | | | | | |
| auto / `ikj` | | | | | | | |

Counter interpretation:

1. Did compiler-generated SIMD reduce instructions, cycles, or both for the
   identical blocked source code?
2. How do the auto-vectorized and manual-intrinsics blocked versions differ?
3. Did IPC move in the same direction as elapsed time? Why is that not required?
4. Which supported cache event best distinguishes `ijk`, `ikj`, and blocked?
5. Were any counters unsupported or multiplexed, and how does that limit the
   claim?

## Causal explanations

For each result, write:

**Claim → evidence → mechanism → next test**

1. SIMD gain or limit:
2. SIMD versus SMT:
3. Vectorization/locality crossover:
4. Matrix-multiplication optimization:
5. Counter evidence and its limitation:
