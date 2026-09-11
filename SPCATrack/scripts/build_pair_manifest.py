#!/usr/bin/env python3
"""Build the adjacent-frame identity manifest used by full ISPS-SOD training."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spcatrack.training.data import build_pair_manifest

p = argparse.ArgumentParser()
p.add_argument("--frames-root", required=True)
p.add_argument("--annotations-root", required=True)
p.add_argument("--out", required=True)
p.add_argument("--allow-no-shared-id", action="store_true")
a = p.parse_args()
n = build_pair_manifest(a.frames_root, a.annotations_root, a.out, require_shared_identity=not a.allow_no_shared_id)
print(f"Wrote {n} adjacent-frame pairs to {a.out}")
