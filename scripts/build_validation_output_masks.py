#!/usr/bin/env python3
"""Generate and freeze output face masks from this backbone's native validation."""

import argparse

from ba_dit.validation_masks import prepare

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--validation", required=True)
parser.add_argument("--overrides", help="Optional JSON sample_id -> xyxy box for reviewed detections")
args = parser.parse_args()
prepare(args.validation, args.overrides)
