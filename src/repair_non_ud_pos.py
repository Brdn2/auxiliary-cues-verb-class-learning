from __future__ import annotations

import argparse
from pathlib import Path

UD_UPOS = {
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
}


def repair(path: Path) -> tuple[int, int]:
    temporary = path.with_suffix(path.suffix + ".tmp")
    changed_lines = 0
    changed_tokens = 0
    with path.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as target:
        for line in source:
            output = []
            line_changed = False
            for item in line.rstrip("\n").split():
                if "/" not in item:
                    output.append(item)
                    continue
                token, upos = item.rsplit("/", 1)
                if upos not in UD_UPOS:
                    upos = "X"
                    changed_tokens += 1
                    line_changed = True
                output.append(f"{token}/{upos}")
            if line_changed:
                changed_lines += 1
            target.write(" ".join(output) + "\n")
    temporary.replace(path)
    return changed_lines, changed_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Map non-UD POS labels to X in-place.")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        changed_lines, changed_tokens = repair(path)
        print(f"{path}: repaired {changed_tokens} tokens across {changed_lines} lines")


if __name__ == "__main__":
    main()
