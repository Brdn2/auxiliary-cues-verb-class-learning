from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pilot_utils import (
    load_json,
    normalize_token,
    project_path,
    read_jsonl,
    save_json,
    write_jsonl,
)


UD_UPOS = {
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
}


@dataclass(frozen=True)
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    pad_id: int
    unk_id: int
    bos_id: int
    eos_id: int
    mask_id: int
    mask_verb_id: int
    fw_mask_id: int
    content_mask_id: int

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.token_to_id.get(token, self.unk_id) for token in tokens]


def parse_utterance(line: str, line_id: int, lowercase: bool) -> dict[str, Any] | None:
    raw_items = line.strip().split()
    if not raw_items:
        return None
    tokens: list[str] = []
    pos: list[str] = []
    for item in raw_items:
        if "/" not in item:
            raise ValueError(f"Line {line_id}: token lacks UPOS separator: {item}")
        word, upos = item.rsplit("/", 1)
        if upos not in UD_UPOS:
            raise ValueError(f"Line {line_id}: invalid UPOS tag {upos!r} in {item!r}")
        tokens.append(normalize_token(word, lowercase))
        pos.append(upos)
    return {
        "line_id": line_id,
        "tokens": tokens,
        "pos": pos,
        "raw": line.rstrip("\n"),
    }


def parse_corpus(config: dict[str, Any]) -> list[dict[str, Any]]:
    raw_path = project_path(config["paths"]["raw_corpus"])
    min_len = int(config["data"]["min_utterance_tokens"])
    max_len = int(config["data"]["max_utterance_tokens"])
    records: list[dict[str, Any]] = []
    with raw_path.open("r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            parsed = parse_utterance(line, line_id, bool(config["lowercase"]))
            if parsed is None:
                continue
            length = len(parsed["tokens"])
            if min_len <= length <= max_len:
                records.append(parsed)
    return records


def split_records(
    records: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, list[int]]:
    seed = int(config["seed"])
    ratios = config["split"]
    indices = list(range(len(records)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    train_n = int(len(indices) * float(ratios["train"]))
    dev_n = int(len(indices) * float(ratios["dev"]))
    return {
        "train": indices[:train_n],
        "dev": indices[train_n : train_n + dev_n],
        "test": indices[train_n + dev_n :],
    }


def build_vocab(
    train_records: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    special = config["special_tokens"]
    ordered_specials = [
        special["pad"],
        special["unk"],
        special["bos"],
        special["eos"],
        special["mask"],
        special["mask_verb"],
        special["fw_mask"],
        special["content_mask"],
    ]
    counts: Counter[str] = Counter()
    for record in train_records:
        counts.update(record["tokens"])
    min_freq = int(config["data"]["vocab_min_freq"])
    tokens = sorted(token for token, count in counts.items() if count >= min_freq)
    id_to_token = ordered_specials + [t for t in tokens if t not in set(ordered_specials)]
    token_to_id = {token: idx for idx, token in enumerate(id_to_token)}
    return {
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
        "counts": dict(counts),
    }


def vocab_from_data(vocab_data: dict[str, Any], config: dict[str, Any]) -> Vocab:
    special = config["special_tokens"]
    token_to_id = {str(k): int(v) for k, v in vocab_data["token_to_id"].items()}
    id_to_token = [str(t) for t in vocab_data["id_to_token"]]
    return Vocab(
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        pad_id=token_to_id[special["pad"]],
        unk_id=token_to_id[special["unk"]],
        bos_id=token_to_id[special["bos"]],
        eos_id=token_to_id[special["eos"]],
        mask_id=token_to_id[special["mask"]],
        mask_verb_id=token_to_id[special["mask_verb"]],
        fw_mask_id=token_to_id[special["fw_mask"]],
        content_mask_id=token_to_id[special["content_mask"]],
    )


def load_prepared(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[int]], Vocab]:
    processed_dir = project_path(config["paths"]["processed_dir"])
    splits_dir = project_path(config["paths"]["splits_dir"])
    records = read_jsonl(processed_dir / "utterances.jsonl")
    splits = load_json(splits_dir / "splits.json")
    vocab_data = load_json(processed_dir / "vocab.json")
    return records, splits, vocab_from_data(vocab_data, config)


def save_prepared(
    records: list[dict[str, Any]],
    splits: dict[str, list[int]],
    vocab_data: dict[str, Any],
    config: dict[str, Any],
) -> None:
    processed_dir = project_path(config["paths"]["processed_dir"])
    splits_dir = project_path(config["paths"]["splits_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(processed_dir / "utterances.jsonl", records)
    save_json(splits_dir / "splits.json", splits)
    save_json(processed_dir / "vocab.json", vocab_data)


def build_verb_maps(
    config: dict[str, Any], vocab: Vocab | None = None
) -> tuple[dict[str, dict[str, str]], dict[str, list[str]]]:
    surface_map: dict[str, dict[str, str]] = {}
    class_to_surfaces: dict[str, list[str]] = {}
    allowed_vocab = set(vocab.token_to_id) if vocab is not None else None
    for verb_class, lemma_map in config["verb_lexicon"].items():
        class_to_surfaces.setdefault(verb_class, [])
        for lemma, forms in lemma_map.items():
            norm_lemma = normalize_token(lemma, bool(config["lowercase"]))
            for form in forms:
                surface = normalize_token(form, bool(config["lowercase"]))
                if allowed_vocab is not None and surface not in allowed_vocab:
                    continue
                surface_map[surface] = {"lemma": norm_lemma, "class": verb_class}
                class_to_surfaces[verb_class].append(surface)
    class_to_surfaces = {
        cls: sorted(set(surfaces)) for cls, surfaces in class_to_surfaces.items()
    }
    return surface_map, class_to_surfaces


def is_comp_token(token: str, upos: str, config: dict[str, Any]) -> bool:
    return upos == "SCONJ" and token in set(config["comp_allowlist"])


def load_split_records(
    records: list[dict[str, Any]], splits: dict[str, list[int]], split: str
) -> list[dict[str, Any]]:
    return [records[i] for i in splits[split]]


def path_exists_for_prepared(config: dict[str, Any]) -> bool:
    return (
        project_path(config["paths"]["processed_dir"]) / "utterances.jsonl"
    ).exists() and (project_path(config["paths"]["splits_dir"]) / "splits.json").exists()
