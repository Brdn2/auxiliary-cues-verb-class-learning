from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare independently randomized AUX annotation packets.")
    parser.add_argument(
        "--input",
        default=str(ROOT / "audits/ACL_static_analysis_v1/aux_context_sample_for_blind_review.csv"),
    )
    parser.add_argument(
        "--output-dir", default=str(ROOT / "audits/ACL_static_analysis_v1/human_annotation")
    )
    parser.add_argument("--seed-a", type=int, default=61001)
    parser.add_argument("--seed-b", type=int, default=61002)
    args = parser.parse_args()

    source = read_csv(Path(args.input))
    if len(source) != 500 or len({row["sample_id"] for row in source}) != 500:
        raise RuntimeError("Expected exactly 500 unique blinded AUX contexts")
    fields = list(source[0])
    output_dir = Path(args.output_dir)
    for annotator, seed in (("A", args.seed_a), ("B", args.seed_b)):
        rows = [dict(row) for row in source]
        random.Random(seed).shuffle(rows)
        write_csv(output_dir / f"annotator_{annotator}.csv", rows, fields)
    print(f"Prepared two independent 500-item packets in {output_dir}")


if __name__ == "__main__":
    main()
