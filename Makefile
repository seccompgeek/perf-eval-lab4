CC ?= gcc
AARCH64_CC ?= aarch64-linux-gnu-gcc
PYTHON ?= python3
PERF ?= perf

BASE_FLAGS ?= -std=c11 -Wall -Wextra -Wpedantic -pthread
HOST_OS := $(shell uname -s)
HOST_ARCH := $(shell uname -m)
IS_CLANG := $(shell $(CC) --version 2>/dev/null | head -n 1 | grep -i clang)
TARGET_ARCH := $(shell $(CC) -dumpmachine 2>/dev/null)
ifneq ($(filter aarch64% arm64%,$(TARGET_ARCH)),)
ARCH_FLAGS ?= -mcpu=native
else
ARCH_FLAGS ?= -march=native
endif
OPT_FLAGS ?= -O3 $(ARCH_FLAGS)
ifneq ($(IS_CLANG),)
VECTOR_FLAGS ?= -fvectorize -fslp-vectorize
SCALAR_FLAGS ?= -fno-vectorize -fno-slp-vectorize
REPORT_FLAGS = -Rpass=\(loop\|slp\)-vectorize -Rpass-missed=\(loop\|slp\)-vectorize -Rpass-analysis=\(loop\|slp\)-vectorize
AUTO_DIAGNOSTIC = remark:.*vectorized
MISSED_DIAGNOSTIC = remark:.*not vectorized
else
VECTOR_FLAGS ?= -ftree-loop-vectorize -ftree-slp-vectorize
SCALAR_FLAGS ?= -fno-tree-loop-vectorize -fno-tree-slp-vectorize
REPORT_FLAGS = -fopt-info-vec-all
AUTO_DIAGNOSTIC = optimized
MISSED_DIAGNOSTIC = missed
endif

# Detect the cross-compiler independently from CC (which may be Apple Clang).
AARCH64_IS_CLANG := $(shell $(AARCH64_CC) --version 2>/dev/null | head -n 1 | grep -i clang)
ifneq ($(AARCH64_IS_CLANG),)
AARCH64_VECTOR_FLAGS ?= -fvectorize -fslp-vectorize
AARCH64_SCALAR_FLAGS ?= -fno-vectorize -fno-slp-vectorize
else
AARCH64_VECTOR_FLAGS ?= -ftree-loop-vectorize -ftree-slp-vectorize
AARCH64_SCALAR_FLAGS ?= -fno-tree-loop-vectorize -fno-tree-slp-vectorize
endif
AARCH64_ARCH_FLAGS ?= -march=armv8-a+simd
AARCH64_OPT_FLAGS ?= -O3 $(AARCH64_ARCH_FLAGS)

# Explicit SIMD and auto-vectorization are independent controls. Scalar targets
# always disable both; INTRINSICS=0 also disables explicit SIMD in other targets.
INTRINSICS ?= 1
ifeq ($(INTRINSICS),0)
INTRINSICS_FLAGS = -DLAB4_DISABLE_INTRINSICS
else ifneq ($(INTRINSICS),1)
$(error INTRINSICS must be 0 or 1)
endif
LDLIBS ?= -lm
AARCH64_CPPFLAGS ?= $(CPPFLAGS)
AARCH64_CFLAGS ?= $(CFLAGS)
AARCH64_LDFLAGS ?= $(LDFLAGS)
AARCH64_LDLIBS ?= $(LDLIBS)
NATIVE_FLAGS = $(CPPFLAGS) $(BASE_FLAGS) $(OPT_FLAGS) $(CFLAGS)
CROSS_FLAGS = $(AARCH64_CPPFLAGS) $(BASE_FLAGS) $(AARCH64_OPT_FLAGS) $(AARCH64_CFLAGS)

# `make perf-stat` has a useful default, but every part is overridable. Examples:
#   make perf-stat PERF_COMMAND='./lab4_scalar mm blocked 512 64'
#   make perf-stat PERF_CPU=3 PERF_RUNS=5 PERF_STAT_FLAGS=-d
PERF_RUNS ?= 3
PERF_CPU ?= 0
PERF_EVENTS ?= task-clock,cycles,instructions,branches,branch-misses,cache-references,cache-misses
PERF_STAT_FLAGS ?= -e $(PERF_EVENTS)
PERF_COMMAND ?= ./lab4_auto mm blocked 512 64

.PHONY: all check report vec-check assembly codegen-check perf-stat cross-aarch64 clean FORCE

all: lab4_scalar lab4_auto lab4_fast simd_demo simd_demo_scalar

# Make normally ignores changes to command-line variables. Compare the complete
# configuration so a compiler/flag change rebuilds existing binaries, without
# forcing recompilation when the configuration is unchanged.
shell_quote = '$(subst ','"'"',$(1))'
FORCE:

.build-native: FORCE
	@printf '%s\n' $(foreach var,CC NATIVE_FLAGS SCALAR_FLAGS VECTOR_FLAGS INTRINSICS_FLAGS LDFLAGS LDLIBS HOST_OS HOST_ARCH,$(call shell_quote,$($(var)))) > $@.tmp
	@cmp -s $@.tmp $@ && rm $@.tmp || mv $@.tmp $@

.build-aarch64: FORCE
	@printf '%s\n' $(foreach var,AARCH64_CC CROSS_FLAGS AARCH64_SCALAR_FLAGS AARCH64_VECTOR_FLAGS INTRINSICS_FLAGS AARCH64_LDFLAGS AARCH64_LDLIBS,$(call shell_quote,$($(var)))) > $@.tmp
	@cmp -s $@.tmp $@ && rm $@.tmp || mv $@.tmp $@

lab4_scalar: lab4_bench.c macos_barrier.h Makefile .build-native
	$(CC) $(NATIVE_FLAGS) $(SCALAR_FLAGS) -DLAB4_DISABLE_INTRINSICS $< $(LDFLAGS) -o $@ $(LDLIBS)

lab4_auto: lab4_bench.c macos_barrier.h Makefile .build-native
	$(CC) $(NATIVE_FLAGS) $(VECTOR_FLAGS) $(INTRINSICS_FLAGS) $< $(LDFLAGS) -o $@ $(LDLIBS)

lab4_fast: lab4_bench.c macos_barrier.h Makefile .build-native
	$(CC) $(NATIVE_FLAGS) $(VECTOR_FLAGS) $(INTRINSICS_FLAGS) -ffast-math $< $(LDFLAGS) -o $@ $(LDLIBS)

simd_demo: simd_intrinsics_demo.c Makefile .build-native
	$(CC) $(NATIVE_FLAGS) $(VECTOR_FLAGS) $(INTRINSICS_FLAGS) $< $(LDFLAGS) -o $@ $(LDLIBS)

simd_demo_scalar: simd_intrinsics_demo.c Makefile .build-native
	$(CC) $(NATIVE_FLAGS) $(SCALAR_FLAGS) -DLAB4_DISABLE_INTRINSICS $< $(LDFLAGS) -o $@ $(LDLIBS)

check: all
	./lab4_scalar verify
	./lab4_auto verify
	./lab4_fast verify
	./simd_demo verify
	./simd_demo_scalar verify
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
	$(CC) $(NATIVE_FLAGS) $(VECTOR_FLAGS) $(INTRINSICS_FLAGS) \
	  $(REPORT_FLAGS) \
	  -c lab4_bench.c -o report-only.o 2>vectorization-report.txt
	@echo "Wrote vectorization-report.txt"

# A focused, readable auto-vectorization check for the three-way SIMD demo.
vec-check: simd_demo simd_demo.s
	$(RM) demo-vectorization-report.txt
	$(CC) $(NATIVE_FLAGS) $(VECTOR_FLAGS) $(INTRINSICS_FLAGS) \
	  $(REPORT_FLAGS) \
	  -c simd_intrinsics_demo.c -o demo-report-only.o 2>demo-vectorization-report.txt
	@auto_line=$$(awk '/VEC_CHECK_AUTO/{print NR}' simd_intrinsics_demo.c); \
	 opaque_line=$$(awk '/VEC_CHECK_OPAQUE/{print NR}' simd_intrinsics_demo.c); \
	 echo "AUTO loop (expected: optimized/vectorized):"; \
	 grep "simd_intrinsics_demo.c:$${auto_line}:.*$(AUTO_DIAGNOSTIC)" demo-vectorization-report.txt | head -n 2 || true; \
	 echo "OPAQUE loop (expected: missed because of the function call):"; \
	 grep "simd_intrinsics_demo.c:$${opaque_line}:.*$(MISSED_DIAGNOSTIC)" demo-vectorization-report.txt | head -n 2 || true; \
	 echo "MANUAL path: explicit $$(./simd_demo intrinsics 32 1 | sed -n 's/.*isa=\([^ ]*\).*/\1/p') intrinsics; run 'make assembly' and open checkpoint 4."
	./simd_demo verify

lab4_scalar.s: lab4_bench.c macos_barrier.h Makefile .build-native
	$(CC) $(NATIVE_FLAGS) $(SCALAR_FLAGS) -DLAB4_DISABLE_INTRINSICS -S lab4_bench.c -o lab4_scalar.s

lab4_auto.s: lab4_bench.c macos_barrier.h Makefile .build-native
	$(CC) $(NATIVE_FLAGS) $(VECTOR_FLAGS) $(INTRINSICS_FLAGS) -S lab4_bench.c -o lab4_auto.s

lab4_fast.s: lab4_bench.c macos_barrier.h Makefile .build-native
	$(CC) $(NATIVE_FLAGS) $(VECTOR_FLAGS) $(INTRINSICS_FLAGS) -ffast-math -S lab4_bench.c -o lab4_fast.s

simd_demo.s: simd_intrinsics_demo.c Makefile .build-native
	$(CC) $(NATIVE_FLAGS) $(VECTOR_FLAGS) $(INTRINSICS_FLAGS) -S simd_intrinsics_demo.c -o simd_demo.s

simd_demo_scalar.s: simd_intrinsics_demo.c Makefile .build-native
	$(CC) $(NATIVE_FLAGS) $(SCALAR_FLAGS) -DLAB4_DISABLE_INTRINSICS -S simd_intrinsics_demo.c -o simd_demo_scalar.s

assembly: lab4_scalar.s lab4_auto.s lab4_fast.s simd_demo.s
	LAB4_CC="$(CC)" $(PYTHON) make_assembly_guide.py
	@echo "Open assembly-guide.txt; it places each target's C source directly above its assembly."

codegen-check: assembly simd_demo_scalar.s
	$(PYTHON) tests/check_codegen.py --intrinsics=$(INTRINSICS)

# Count the whole process. Choose a long-running kernel so initialization and
# process-start overhead are small compared with the measured work.
perf-stat: all
ifeq ($(HOST_OS),Darwin)
	@echo "perf-stat requires Linux perf and taskset. On macOS use benchmark elapsed_s or Instruments."
	@exit 1
else
	@echo "Command: $(PERF_COMMAND)"
	@echo "CPU: $(PERF_CPU); repetitions: $(PERF_RUNS)"
	$(PERF) stat -r $(PERF_RUNS) $(PERF_STAT_FLAGS) -- \
	  taskset -c $(PERF_CPU) $(PERF_COMMAND)

endif

# Build portable ARMv8-A/Advanced-SIMD binaries on an x86 host. These targets
# verify compilation only; run `make check` natively on the Raspberry Pi.
cross-aarch64: lab4_scalar.aarch64 lab4_auto.aarch64 lab4_fast.aarch64 simd_demo.aarch64 simd_demo_scalar.aarch64

lab4_scalar.aarch64: lab4_bench.c macos_barrier.h Makefile .build-aarch64
	$(AARCH64_CC) $(CROSS_FLAGS) $(AARCH64_SCALAR_FLAGS) -DLAB4_DISABLE_INTRINSICS $< $(AARCH64_LDFLAGS) -o $@ $(AARCH64_LDLIBS)

lab4_auto.aarch64: lab4_bench.c macos_barrier.h Makefile .build-aarch64
	$(AARCH64_CC) $(CROSS_FLAGS) $(AARCH64_VECTOR_FLAGS) $(INTRINSICS_FLAGS) $< $(AARCH64_LDFLAGS) -o $@ $(AARCH64_LDLIBS)

lab4_fast.aarch64: lab4_bench.c macos_barrier.h Makefile .build-aarch64
	$(AARCH64_CC) $(CROSS_FLAGS) $(AARCH64_VECTOR_FLAGS) $(INTRINSICS_FLAGS) -ffast-math $< $(AARCH64_LDFLAGS) -o $@ $(AARCH64_LDLIBS)

simd_demo.aarch64: simd_intrinsics_demo.c Makefile .build-aarch64
	$(AARCH64_CC) $(CROSS_FLAGS) $(AARCH64_VECTOR_FLAGS) $(INTRINSICS_FLAGS) $< $(AARCH64_LDFLAGS) -o $@ $(AARCH64_LDLIBS)

simd_demo_scalar.aarch64: simd_intrinsics_demo.c Makefile .build-aarch64
	$(AARCH64_CC) $(CROSS_FLAGS) $(AARCH64_SCALAR_FLAGS) -DLAB4_DISABLE_INTRINSICS $< $(AARCH64_LDFLAGS) -o $@ $(AARCH64_LDLIBS)

clean:
	rm -f lab4_scalar lab4_auto lab4_fast simd_demo simd_demo_scalar *.aarch64 *.s \
	  .build-native .build-native.tmp .build-aarch64 .build-aarch64.tmp \
	  report-only.o demo-report-only.o vectorization-report.txt \
	  demo-vectorization-report.txt vectorization-optimized.txt vectorization-missed.txt \
	  assembly-guide.txt
