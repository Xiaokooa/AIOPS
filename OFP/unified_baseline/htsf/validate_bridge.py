"""Validate that formal strict B2 and sampled B2 form a controlled bridge."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
UNIFIED_DIR = BASE_DIR.parent
sys.path[:0] = [str(BASE_DIR), str(UNIFIED_DIR)]

from ofp_htsf.bridge import validate_strict_b2_bridge


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-b2-manifest", type=Path, required=True)
    parser.add_argument("--sampled-b2-manifest", type=Path, required=True)
    parser.add_argument("--bridge-suite-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = validate_strict_b2_bridge(
        args.strict_b2_manifest,
        args.sampled_b2_manifest,
        args.bridge_suite_manifest,
    )
    encoded = json.dumps(result, indent=2)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
