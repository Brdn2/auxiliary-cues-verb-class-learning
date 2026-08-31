from __future__ import annotations

import argparse
from pathlib import Path

import spacy


UD_UPOS = {
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
}


def token_for_slash_format(text: str) -> str:
    return text.replace(" ", "_")


def annotate(input_path: Path, output_path: Path, model: str, batch_size: int, limit: int | None) -> None:
    nlp = spacy.load(model, disable=["parser", "ner", "textcat"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if limit is not None:
        lines = lines[:limit]
    with output_path.open("w", encoding="utf-8") as out:
        for doc in nlp.pipe(lines, batch_size=batch_size):
            items: list[str] = []
            for token in doc:
                if token.is_space:
                    continue
                # spaCy may emit its internal SPACE label for malformed CHAT
                # overlap markers even when token.is_space is false. The corpus
                # parser accepts UD UPOS only, so preserve the token and degrade
                # unsupported labels to X rather than dropping the utterance.
                upos = token.pos_ if token.pos_ in UD_UPOS else "X"
                items.append(f"{token_for_slash_format(token.text)}/{upos}")
            if items:
                out.write(" ".join(items) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("--model", default="en_core_web_sm")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    annotate(Path(args.input_path), Path(args.output_path), args.model, args.batch_size, args.limit)
    print(f"Annotated {args.input_path} -> {args.output_path} with {args.model}")


if __name__ == "__main__":
    main()
