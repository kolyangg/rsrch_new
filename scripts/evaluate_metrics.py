#!/usr/bin/env python3
"""Score generated images in the separate metrics environment."""

import argparse

from ba_dit.metrics import evaluate

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--validation", required=True)
parser.add_argument("--ownership-boxes", help="JSON sample_id -> xyxy box on these generated images")
parser.add_argument("--no-clip", action="store_true")
parser.add_argument("--no-comet", action="store_true")
parser.add_argument("--log-dir", help="Training run directory containing the immutable Comet key")
parser.add_argument("--global-step", type=int, default=0)
parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
args = parser.parse_args()
evaluate(args.validation, args.ownership_boxes, not args.no_clip, args.no_comet, args.log_dir, args.global_step, args.device)
