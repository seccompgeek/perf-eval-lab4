CC ?= gcc
AARCH64_CC ?= aarch64-linux-gnu-gcc
PYTHON ?= python3
PERF ?= perf

BASE_FLAGS = -std=c11 -Wall -Wextra -Wpedantic -pthread
ARCH_FLAGS ?= -march=native
OPT_FLAGS = -O3 $(ARCH_FLAGS)
VECTOR_FLAGS = -ftree-loop-vectorize -ftree-slp-vectorize
SCALAR_FLAGS = -fno-tree-loop-vectorize -fno-tree-slp-vectorize
AARCH64_ARCH_FLAGS ?= -march=armv8-a+simd
AARCH64_OPT_FLAGS = -O3 $(AARCH64_ARCH_FLAGS)

# `make perf-stat` has a useful default, but every part is overridable. Examples:
#   make perf-stat PERF_COMMAND='./lab4_scalar mm blocked 512 64'
#   make perf-stat PERF_CPU=3 PERF_RUNS=5 PERF_STAT_FLAGS=-d
PERF_RUNS ?= 3
PERF_CPU ?= 0
PERF_EVENTS ?= task-clock,cycles,instructions,branches,branch-misses,cache-references,cache-misses
PERF_STAT_FLAGS ?= -e $(PERF_EVENTS)
PERF_COMMAND ?= ./lab4_auto mm blocked 512 64

.PHONY: all check report vec-check assembly perf-stat cross-aarch64 clean

all: lab4_scalar lab4_auto lab4_fast simd_demo

lab4_scalar: lab4_bench.c
	$(CC) $(BASE_FLAGS) $(OPT_FLAGS) $(SCALAR_FLAGS) $< -o $@ -lm

lab4_auto: lab4_bench.c
	$(CC) $(BASE_FLAGS) $(OPT_FLAGS) $(VECTOR_FLAGS) $< -o $@ -lm

lab4_fast: lab4_bench.c
	$(CC) $(BASE_FLAGS) $(OPT_FLAGS) $(VECTOR_FLAGS) -ffast-math $< -o $@ -lm

simd_demo: simd_intrinsics_demo.c
	$(CC) $(BASE_FLAGS) $(OPT_FLAGS) $(VECTOR_FLAGS) $< -o $@ -lm

check: all
	./lab4_scalar verify
	./lab4_auto verify
	./lab4_fast verify
	./simd_demo verify
	./lab4_auto axpy 100003 2
	./lab4_auto dependency 100003 2
	./lab4_auto poly 2 100003 2
	./lab4_auto tradeoff fused 100003 2
	./lab4_auto tradeoff split 100003 2
	./lab4_auto matrix row 128 2
	./lab4_auto matrix col 128 2
	./simd_demo auto 100003 2
	./simd_demo opaque 100003 2
	./simd_demo intrinsics 100003 2

report:
	$(RM) vectorization-report.txt
	$(CC) $(BASE_FLAGS) $(OPT_FLAGS) $(VECTOR_FLAGS) \
	  -fopt-info-vec-all=vectorization-report.txt \
	  -c lab4_bench.c -o report-only.o
	@echo "Wrote vectorization-report.txt"

# A focused, readable auto-vectorization check for the three-way SIMD demo.
vec-check: simd_demo
	$(RM) demo-vectorization-report.txt
	$(CC) $(BASE_FLAGS) $(OPT_FLAGS) $(VECTOR_FLAGS) \
	  -fopt-info-vec-all=demo-vectorization-report.txt \
	  -c simd_intrinsics_demo.c -o demo-report-only.o
	$(CC) $(BASE_FLAGS) $(OPT_FLAGS) $(VECTOR_FLAGS) -S simd_intrinsics_demo.c -o simd_demo.s
	@auto_line=$$(awk '/VEC_CHECK_AUTO/{print NR}' simd_intrinsics_demo.c); \
	 opaque_line=$$(awk '/VEC_CHECK_OPAQUE/{print NR}' simd_intrinsics_demo.c); \
	 echo "AUTO loop (expected: optimized/vectorized):"; \
	 grep "simd_intrinsics_demo.c:$${auto_line}:.*optimized" demo-vectorization-report.txt | head -n 2 || true; \
	 echo "OPAQUE loop (expected: missed because of the function call):"; \
	 grep "simd_intrinsics_demo.c:$${opaque_line}:.*missed" demo-vectorization-report.txt | head -n 2 || true; \
	 echo "MANUAL path: explicit $$(./simd_demo intrinsics 32 1 | sed -n 's/.*isa=\([^ ]*\).*/\1/p') intrinsics; run 'make assembly' and open checkpoint 4."
	./simd_demo verify

assembly:
	$(CC) $(BASE_FLAGS) $(OPT_FLAGS) $(SCALAR_FLAGS) -S lab4_bench.c -o lab4_scalar.s
	$(CC) $(BASE_FLAGS) $(OPT_FLAGS) $(VECTOR_FLAGS) -S lab4_bench.c -o lab4_auto.s
	$(CC) $(BASE_FLAGS) $(OPT_FLAGS) $(VECTOR_FLAGS) -ffast-math -S lab4_bench.c -o lab4_fast.s
	$(CC) $(BASE_FLAGS) $(OPT_FLAGS) $(VECTOR_FLAGS) -S simd_intrinsics_demo.c -o simd_demo.s
	LAB4_CC="$(CC)" $(PYTHON) make_assembly_guide.py
	@echo "Open assembly-guide.txt; it places each target's C source directly above its assembly."

# Count the whole process. Choose a long-running kernel so initialization and
# process-start overhead are small compared with the measured work.
perf-stat: all
	@echo "Command: $(PERF_COMMAND)"
	@echo "CPU: $(PERF_CPU); repetitions: $(PERF_RUNS)"
	$(PERF) stat -r $(PERF_RUNS) $(PERF_STAT_FLAGS) -- \
	  taskset -c $(PERF_CPU) $(PERF_COMMAND)

# Build portable ARMv8-A/Advanced-SIMD binaries on an x86 host. These targets
# verify compilation only; run `make check` natively on the Raspberry Pi.
cross-aarch64: lab4_scalar.aarch64 lab4_auto.aarch64 lab4_fast.aarch64 simd_demo.aarch64

lab4_scalar.aarch64: lab4_bench.c
	$(AARCH64_CC) $(BASE_FLAGS) $(AARCH64_OPT_FLAGS) $(SCALAR_FLAGS) $< -o $@ -lm

lab4_auto.aarch64: lab4_bench.c
	$(AARCH64_CC) $(BASE_FLAGS) $(AARCH64_OPT_FLAGS) $(VECTOR_FLAGS) $< -o $@ -lm

lab4_fast.aarch64: lab4_bench.c
	$(AARCH64_CC) $(BASE_FLAGS) $(AARCH64_OPT_FLAGS) $(VECTOR_FLAGS) -ffast-math $< -o $@ -lm

simd_demo.aarch64: simd_intrinsics_demo.c
	$(AARCH64_CC) $(BASE_FLAGS) $(AARCH64_OPT_FLAGS) $(VECTOR_FLAGS) $< -o $@ -lm

clean:
	rm -f lab4_scalar lab4_auto lab4_fast simd_demo *.aarch64 *.s \
	  report-only.o demo-report-only.o vectorization-report.txt \
	  demo-vectorization-report.txt vectorization-optimized.txt vectorization-missed.txt \
	  assembly-guide.txt
