#!/usr/bin/env python3
"""Build a source-to-assembly reading guide for the Lab 4 target kernels."""

from pathlib import Path
import os
import platform
import re
import shlex
import subprocess


OUTPUT = Path("assembly-guide.txt")


# title, source file, functions, assembly variants, reading cue
CHECKPOINTS = [
    (
        "1. AXPY — the compiler's easy auto-vectorization case",
        Path("lab4_bench.c"),
        ["kernel_axpy"],
        [("SCALAR BUILD", Path("lab4_scalar.s")),
         ("AUTO-VECTORIZED BUILD", Path("lab4_auto.s"))],
        "Compare the loop bodies. In the auto build, find packed loads/adds/stores "
        "and the scalar cleanup for leftover elements.",
    ),
    (
        "2. DEPENDENCY — a true recurrence",
        Path("lab4_bench.c"),
        ["kernel_dependency"],
        [("SCALAR BUILD", Path("lab4_scalar.s")),
         ("AUTO BUILD", Path("lab4_auto.s"))],
        "The auto build should remain scalar because a[i] consumes a[i-1]. "
        "Similar assembly is expected here.",
    ),
    (
        "3. POLYNOMIAL — enough arithmetic to expose SIMD",
        Path("lab4_bench.c"),
        ["kernel_poly"],
        [("SCALAR BUILD", Path("lab4_scalar.s")),
         ("AUTO-VECTORIZED BUILD", Path("lab4_auto.s"))],
        "Look for repeated packed multiply/add operations in the auto build. "
        "This function is longer, so search for the >>> markers first.",
    ),
    (
        "4. AUTO, OPAQUE, AND MANUAL SIMD",
        Path("simd_intrinsics_demo.c"),
        ["kernel_auto", "kernel_opaque", "kernel_intrinsics"],
        [("EMITTED ASSEMBLY", Path("simd_demo.s"))],
        "kernel_auto should contain compiler-generated SIMD; kernel_opaque should "
        "retain calls to opaque_add; kernel_intrinsics should contain the explicit "
        "AVX2, SSE2, or NEON operations from the source.",
    ),
    (
        "5. FUSED REUSE VERSUS SPLIT VECTORIZATION",
        Path("lab4_bench.c"),
        ["tradeoff_fused", "tradeoff_split"],
        [("AUTO BUILD", Path("lab4_auto.s"))],
        "The fused dependence normally prevents packed operations. The two split "
        "loops are independent and should show packed SIMD instructions.",
    ),
    (
        "6. MATRIX ROW VERSUS COLUMN TRAVERSAL",
        Path("lab4_bench.c"),
        ["matrix_row", "matrix_col"],
        [("AUTO BUILD", Path("lab4_auto.s"))],
        "The row loop has unit-stride packed accesses. The column loop advances by "
        "N*sizeof(double), so ordinary packed SIMD is unlikely.",
    ),
    (
        "7. MATRIX MULTIPLICATION — AUTO SIMD VERSUS INTRINSICS",
        Path("lab4_bench.c"),
        ["mm_ijk", "mm_ikj", "mm_blocked", "mm_blocked_intrinsics"],
        [("AUTO BUILD", Path("lab4_auto.s"))],
        "Start with mm_ikj: its inner j loop should contain packed operations. Then "
        "find the compiler-generated packed loop inside mm_blocked; mm_ijk has a "
        "strided B access and scalar accumulation. Finally inspect "
        "mm_blocked_intrinsics: before the exercise it has no packed inner loop; "
        "afterward its packed instructions must come from our guarded intrinsics.",
    ),
]


PACKED_X86 = re.compile(
    r"(?:%[yz]mm\d+|\b(?:v?addpd|v?subpd|v?mulpd|v?divpd|v?movupd|"
    r"v?movapd|vfmadd\w*pd|vfmsub\w*pd|vbroadcastsd)\b)",
    re.IGNORECASE,
)
PACKED_ARM = re.compile(
    r"(?:\bv\d+\.(?:2d|4s|4h|8h|16b)\b|\b(?:ld1|st1)\b|"
    r"\b(?:ldr|str|ldp|stp)\s+q\d+\b|"
    r"\bf(?:add|sub|mul|mla|mls)\.(?:2d|4s|4h|8h)\b|"
    r"\bf(?:add|sub|mul|mla|mls)\s+v\d+\.)",
    re.IGNORECASE,
)


def compiler_command():
    return shlex.split(os.environ.get("LAB4_CC", "cc"))


def compiler_version():
    try:
        result = subprocess.run(
            compiler_command() + ["--version"], check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        return result.stdout.splitlines()[0]
    except OSError:
        return "compiler version unavailable"


def compiler_target():
    try:
        result = subprocess.run(
            compiler_command() + ["-dumpmachine"], check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        return result.stdout.strip() or "target unavailable"
    except OSError:
        return "target unavailable"


def extract_source(path, function):
    lines = path.read_text(encoding="utf-8").splitlines()
    match_line = None
    pattern = re.compile(rf"\b{re.escape(function)}\s*\(")
    for index, value in enumerate(lines):
        if pattern.search(value):
            match_line = index
            break
    if match_line is None:
        raise RuntimeError(f"cannot find source function {function} in {path}")

    start = match_line
    while start > 0 and not re.match(r"\s*static\b", lines[start]):
        start -= 1

    depth = 0
    opened = False
    end = match_line
    for index in range(start, len(lines)):
        # The selected kernels contain no braces inside strings or comments.
        depth += lines[index].count("{")
        if "{" in lines[index]:
            opened = True
        depth -= lines[index].count("}")
        end = index
        if opened and depth == 0:
            break

    return [f"{index + 1:4d} | {lines[index]}" for index in range(start, end + 1)]


def assembly_symbols(lines, function):
    symbols = []
    pattern = re.compile(rf"^(_?{re.escape(function)}(?:\.[A-Za-z0-9_.]+)?):(?:\s*[#;].*)?$")
    for value in lines:
        found = pattern.match(value.strip())
        if found and found.group(1) not in symbols:
            symbols.append(found.group(1))
    symbols.sort(key=lambda value: (value != function, "constprop" not in value,
                                    value))
    return symbols


def clean_assembly(raw_lines):
    cleaned = []
    for value in raw_lines:
        stripped = value.strip()
        if not stripped:
            continue
        if stripped.startswith((".cfi", ".loc", ".file", ".p2align",
                                ".type", ".size", ".section", ".ident",
                                ".globl", ".hidden", ".previous",
                                ".note.GNU-stack")):
            continue
        if stripped.startswith(".") and not stripped.endswith(":"):
            continue
        marker = ">>>" if (PACKED_X86.search(stripped) or
                           PACKED_ARM.search(stripped)) else "   "
        cleaned.append(f"{marker} {stripped}")
    return cleaned


def extract_assembly(path, function):
    lines = path.read_text(encoding="utf-8").splitlines()
    symbols = assembly_symbols(lines, function)
    if not symbols:
        return "(function symbol not emitted; the compiler may have inlined it)\n"

    # Use one canonical emitted body. Static functions sometimes exist only as a
    # .constprop clone when all call-site arguments are compile-time constants.
    symbol = symbols[0]
    start = next(i for i, value in enumerate(lines)
                 if value.strip().split(":", 1)[0] == symbol)
    end_pattern = re.compile(rf"^\s*\.size\s+{re.escape(symbol)}\s*,")
    end = len(lines) - 1
    for index in range(start + 1, len(lines)):
        if (end_pattern.search(lines[index]) or
                lines[index].strip().startswith(".cfi_endproc")):
            end = index
            break
    cleaned = clean_assembly(lines[start:end + 1])
    vector_lines = sum(1 for value in cleaned if value.startswith(">>>"))
    heading = (f"emitted symbol: {symbol}\n"
               f"packed-vector-looking lines marked >>>: {vector_lines}\n")
    return heading + "\n".join(cleaned) + "\n"


def divider(character="=", width=88):
    return character * width


def build_guide():
    required = {source for _, source, _, variants, _ in CHECKPOINTS}
    required.update(path for _, _, _, variants, _ in CHECKPOINTS
                    for _, path in variants)
    missing = [str(path) for path in sorted(required) if not path.exists()]
    if missing:
        raise RuntimeError("missing inputs; run make assembly first: " +
                           ", ".join(missing))

    out = []
    out.extend([
        divider(),
        "LAB 4 — SOURCE-TO-ASSEMBLY READING GUIDE",
        divider(),
        f"Host architecture: {platform.machine()}",
        f"Compiler: {compiler_version()}",
        f"Compiler target: {compiler_target()}",
        "",
        "HOW TO USE THIS FILE",
        "  1. Read the C source immediately above each assembly block.",
        "  2. Compare SCALAR BUILD with AUTO-VECTORIZED BUILD where both appear.",
        "  3. Lines prefixed >>> look like packed SIMD instructions for this target.",
        "  4. Labels, branches, bounds checks, and scalar cleanup are left visible.",
        "  5. Instruction names differ between x86-64 and AArch64; use the cue",
        "     printed at the start of each checkpoint.",
        "",
        "IMPORTANT",
        "  A >>> marker is a reading aid, not proof by itself. Pair this guide with",
        "  make vec-check or make report and with the benchmark timing.",
        "",
        "CONTENTS",
    ])
    for title, _, _, _, _ in CHECKPOINTS:
        out.append(f"  {title}")
    out.append("")

    for title, source, functions, variants, cue in CHECKPOINTS:
        out.extend([divider(), title, divider(), "READING CUE", f"  {cue}", ""])
        for function in functions:
            out.extend([divider("-"), f"FUNCTION: {function}", divider("-"),
                        f"SOURCE: {source}", ""])
            out.extend(extract_source(source, function))
            out.append("")
            for label, assembly in variants:
                out.extend([f"{label}: {assembly}", divider(".")])
                out.extend(extract_assembly(assembly, function).rstrip().splitlines())
                out.append("")

    out.extend([
        divider(),
        "END OF GUIDE",
        divider(),
        "Suggested evidence sentence:",
        "  Claim → source property → compiler/assembly evidence → timing → mechanism",
        "",
    ])
    OUTPUT.write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    build_guide()
    print(f"Wrote {OUTPUT}")
