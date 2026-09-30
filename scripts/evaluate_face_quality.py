#!/usr/bin/env python3
"""Score the original four face-quality models in their isolated CPU environment."""

import argparse

from ba_dit.face_quality import DEFAULT_THREADS, evaluate

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--validation", required=True)
parser.add_argument("--no-comet", action="store_true")
parser.add_argument("--log-dir")
parser.add_argument("--global-step", type=int, default=0)
parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
args = parser.parse_args()
if args.threads < 1:
    parser.error("--threads must be positive")
evaluate(args.validation, args.no_comet, args.log_dir, args.global_step, args.threads)
