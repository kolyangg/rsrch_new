#!/usr/bin/env python3
"""Run the small architectural checks in the selected backend environment."""

import argparse
import runpy

from ba_dit.config import ROOT
from ba_dit.runtime import prepare_imports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("flux", "qwen"))
    args = parser.parse_args()
    prepare_imports(args.backend)
    files = ["test_reference_read_delta", "test_branch_checkpoint", "test_training_state", f"test_{args.backend}_branch_seam", f"test_{args.backend}_layout"]
    passed = 0
    for name in files:
        for function, value in runpy.run_path(str(ROOT / "tests" / f"{name}.py")).items():
            if function.startswith("test_"):
                value()
                print(f"PASS {name}.{function}", flush=True)
                passed += 1
    print(f"{passed} critical checks passed ({args.backend})")


if __name__ == "__main__":
    main()
