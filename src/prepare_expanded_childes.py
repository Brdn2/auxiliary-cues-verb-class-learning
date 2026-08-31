from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


REJECT_MARKERS = ("[", "xxx", "yyy", "www")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile_plaintext(path: Path) -> dict[str, Any]:
    lines = 0
    tokens = 0
    unique: set[str] = set()
    lengths: list[int] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            text = raw.strip()
            if not text:
                continue
            count = len(text.split())
            lines += 1
            tokens += count
            lengths.append(count)
            unique.add(text)
    ordered = sorted(lengths)
    quantile = lambda q: ordered[min(int((len(ordered) - 1) * q), len(ordered) - 1)] if ordered else 0
    return {
        "utterances": lines,
        "whitespace_tokens": tokens,
        "unique_utterances": len(unique),
        "duplicate_rate": round(1 - len(unique) / max(lines, 1), 6),
        "length_min": min(lengths, default=0),
        "length_p50": quantile(0.50),
        "length_p90": quantile(0.90),
        "length_p99": quantile(0.99),
        "length_max": max(lengths, default=0),
        "sha256": sha256(path),
    }


def extract_chat(
    input_path: Path,
    roles: set[str],
    min_words: int,
    max_words: int,
) -> tuple[list[str], dict[str, Any]]:
    accepted: list[str] = []
    speaker_counts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    total_lines = 0
    speaker_lines = 0
    with input_path.open(encoding="utf-8") as handle:
        for raw in handle:
            total_lines += 1
            line = raw.rstrip("\n")
            if not line.startswith("*") or ":\t" not in line:
                rejection_counts["non_speaker_or_metadata"] += 1
                continue
            speaker_lines += 1
            speaker, text = line[1:].split(":\t", 1)
            speaker_counts[speaker] += 1
            if speaker not in roles:
                rejection_counts["speaker_not_selected"] += 1
                continue
            text = " ".join(text.strip().split())
            if not text:
                rejection_counts["empty"] += 1
                continue
            if text.startswith('"'):
                rejection_counts["quoted_or_imitation"] += 1
                continue
            marker = next((marker for marker in REJECT_MARKERS if marker in text), None)
            if marker:
                rejection_counts[f"chat_marker_{marker}"] += 1
                continue
            words = len(text.split())
            if words < min_words:
                rejection_counts["too_short"] += 1
                continue
            if words > max_words:
                rejection_counts["too_long"] += 1
                continue
            accepted.append(text)
    return accepted, {
        "input_lines": total_lines,
        "speaker_lines": speaker_lines,
        "selected_roles": sorted(roles),
        "speaker_counts": dict(speaker_counts),
        "rejection_counts": dict(rejection_counts),
    }


def nested_sample_indices(lines: list[str], budgets: list[int], seed: int) -> dict[int, set[int]]:
    indices = list(range(len(lines)))
    random.Random(seed).shuffle(indices)
    selected: dict[int, set[int]] = {budget: set() for budget in budgets}
    running_tokens = 0
    for index in indices:
        running_tokens += len(lines[index].split())
        for budget in budgets:
            if running_tokens <= budget or not selected[budget]:
                selected[budget].add(index)
        if running_tokens >= max(budgets):
            break
    return selected


def write_lines(path: Path, lines: list[str], indices: set[int] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, text in enumerate(lines):
            if indices is None or index in indices:
                handle.write(text + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract comparable MOT/FAT CDS and build nested corpus scales.")
    parser.add_argument("input_path")
    parser.add_argument("output_dir")
    parser.add_argument(
        "--old-cleaned",
        help="Optional earlier cleaned corpus used only for the scale-comparison profile.",
    )
    parser.add_argument("--roles", default="MOT,FAT")
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument("--max-words", type=int, default=80)
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--budgets", default="720000,2000000")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    old_path = Path(args.old_cleaned) if args.old_cleaned else None
    roles = {item.strip() for item in args.roles.split(",") if item.strip()}
    budgets = sorted(int(item) for item in args.budgets.split(",") if item.strip())
    lines, extraction = extract_chat(input_path, roles, args.min_words, args.max_words)
    full_path = output_dir / "childes_train2_cds_full_cleaned.txt"
    write_lines(full_path, lines)
    samples = nested_sample_indices(lines, budgets, args.sample_seed)
    sample_paths: dict[str, str] = {}
    for budget, indices in samples.items():
        label = f"{budget // 1000:04d}k"
        path = output_dir / f"childes_train2_cds_{label}_cleaned.txt"
        write_lines(path, lines, indices)
        sample_paths[label] = str(path)

    old_profile = profile_plaintext(old_path) if old_path else None
    full_profile = profile_plaintext(full_path)
    sample_profiles = {label: profile_plaintext(Path(path)) for label, path in sample_paths.items()}
    report = {
        "input": str(input_path),
        "old_cleaned": str(old_path) if old_path else None,
        "cleaning_rule": {"roles": sorted(roles), "min_words": args.min_words, "max_words": args.max_words, "rejected_markers": list(REJECT_MARKERS), "reject_initial_quote": True, "preserve_duplicates": True},
        "extraction": extraction,
        "old_profile": old_profile,
        "expanded_full_profile": full_profile,
        "nested_sample_profiles": sample_profiles,
        "scale_ratio": (
            {
                "utterances": round(full_profile["utterances"] / old_profile["utterances"], 4),
                "whitespace_tokens": round(
                    full_profile["whitespace_tokens"] / old_profile["whitespace_tokens"], 4
                ),
            }
            if old_profile
            else None
        ),
        "sample_seed": args.sample_seed,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "corpus_profile.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = ["# Expanded CHILDES corpus profile", ""]
    if old_profile:
        markdown.extend(
            [
                f"- Earlier cleaned corpus: {old_profile['utterances']:,} utterances; "
                f"{old_profile['whitespace_tokens']:,} whitespace tokens.",
                f"- Scale ratio: {report['scale_ratio']['utterances']:.2f}× utterances; "
                f"{report['scale_ratio']['whitespace_tokens']:.2f}× tokens.",
            ]
        )
    markdown.extend(
        [
            f"- Expanded cleaned corpus: {full_profile['utterances']:,} utterances; "
            f"{full_profile['whitespace_tokens']:,} whitespace tokens.",
            f"- Expanded duplicate rate: {full_profile['duplicate_rate']:.2%}.",
            f"- Length p50/p90/p99: {full_profile['length_p50']}/"
            f"{full_profile['length_p90']}/{full_profile['length_p99']} whitespace tokens.",
            "",
            "## Nested samples",
            "",
            "| Sample | Utterances | Tokens | SHA-256 |",
            "|---|---:|---:|---|",
        ]
    )
    for label, profile in sample_profiles.items():
        markdown.append(f"| {label} | {profile['utterances']:,} | {profile['whitespace_tokens']:,} | `{profile['sha256']}` |")
    (output_dir / "corpus_profile.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
