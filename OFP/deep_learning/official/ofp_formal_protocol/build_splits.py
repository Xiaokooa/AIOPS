from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from OFP.deep_learning.official.ofp_formal_protocol.protocol import build_formal_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the frozen formal OFP split manifest.")
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/training"))
    parser.add_argument("--index_path", type=Path, default=Path("dataset/train_test_set_index(in).csv"))
    parser.add_argument("--out_dir", type=Path, default=Path("output/ofp_formal_protocol/splits"))
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_df, manifest_df = build_formal_manifest(
        data_dir=args.data_dir,
        index_path=args.index_path,
        out_dir=args.out_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print(f"[splits] modules={len(metadata_df)} manifest_rows={len(manifest_df)}")
    print(f"[splits] metadata={args.out_dir / 'module_metadata.csv'}")
    print(f"[splits] manifest={args.out_dir / 'formal_manifest.csv'}")
    print("[splits] role counts:")
    print(manifest_df.groupby(["fold", "role"])["file_name"].count().unstack(fill_value=0))
    mismatches = int(metadata_df["label_mismatch"].sum())
    print(f"[splits] label_mismatches={mismatches}")


if __name__ == "__main__":
    main()
