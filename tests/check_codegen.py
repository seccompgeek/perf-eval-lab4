#!/usr/bin/env python3
"""Check the lab's scalar/SIMD distinction in x86-64 and AArch64 assembly.

Only packed floating-point arithmetic counts: scalar FP uses SIMD registers
too, and vectorized copies do not prove that the arithmetic was vectorized.
Run through `make codegen-check` with the default vectorizer flags.
"""

import argparse
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from make_assembly_guide import extract_assembly


PACKED_ARITHMETIC = re.compile(
    r"\b(?:v?(?:add|sub|mul|div)pd|vfmadd\w*pd|vfmsub\w*pd)\b"
    r"|\bf(?:add|sub|mul|div|mla|mls)\s+v\d+\.2d\b"
    r"|\bf(?:add|sub|mul|div|mla|mls)\s+z\d+\.d\b"
    r"|\bf(?:add|sub|mul|div|mla|mls)\.2d\b",
    re.IGNORECASE,
)


def check(path, function, expected):
    body = extract_assembly(Path(path), function)
    if not body.startswith("emitted symbol:"):
        raise AssertionError(f"{path}: missing {function}")
    packed = bool(PACKED_ARITHMETIC.search(body))
    if packed != expected:
        raise AssertionError(
            f"{path}: {function}: packed arithmetic={packed}, expected {expected}"
        )
    print(f"PASS {path}: {function}: packed arithmetic={packed}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intrinsics", choices=("0", "1"), default="1")
    args = parser.parse_args()
    check("lab4_scalar.s", "kernel_axpy", False)
    check("lab4_auto.s", "kernel_axpy", True)
    check("lab4_auto.s", "kernel_dependency", False)
    check("simd_demo.s", "kernel_auto", True)
    check("simd_demo.s", "kernel_opaque", False)
    check("simd_demo.s", "kernel_intrinsics", args.intrinsics == "1")
    check("simd_demo_scalar.s", "kernel_auto", False)
    check("simd_demo_scalar.s", "kernel_intrinsics", False)


if __name__ == "__main__":
    main()
