# Data statement: expanded child-directed speech corpus

## Provenance and scope

The local source file is named `childes.train-2.txt`. Its macOS download metadata identifies the source as `BabyLM-community/BabyLM-2026-Strict/childes.train.txt`; its size (152,297,628 bytes) and SHA-256 (`d6c0e39e2c283bfa5489cde80031a4fb9237ca8065a8301580df10e3b6843a98`) exactly match the Git-LFS record added in BabyLM repository commit `c10defe96630b187709593c25fd60c58a3648efa`. The BabyLM card describes this component as 28,410,878 tokens in the 2026 Strict training set and labels the repository MIT.

The repository-level tag does not by itself establish that all underlying transcript rights were relicensed. TalkBank states that, except where otherwise indicated, its data use is governed by CC BY-NC-SA 3.0 and remains subject to corpus-specific citation, access, ethics, and confidentiality rules. This project therefore applies the more restrictive boundary: non-commercial research use with CHILDES/BabyLM attribution and no public redistribution of raw or derived utterance text. The present repository contains no transcript text.

## Speaker and utterance selection

`src/preprocess_childes.py` reads CHAT speaker lines and retains only `MOT` and `FAT` roles. The frozen cleaning rule in `data/expanded_childes/corpus_profile.json`:

- retains utterances of 5--80 whitespace-delimited words;
- rejects lines containing `[`, `xxx`, `yyy`, or `www`;
- rejects initial quoted/imitation lines;
- preserves duplicate utterances.

The source contains 5,638,783 total lines and 5,199,468 speaker lines. The filtering pipeline produces 904,228 utterances and 6,952,986 whitespace-delimited tokens. There are 819,576 unique utterance strings, for a duplicate rate of 9.3618%. Length percentiles are 7, 12, and 21 words at p50, p90, and p99.

These role codes are transcript metadata, not verified demographic categories. The project does not infer caregiver gender, family structure, race, socioeconomic status, geography, or child characteristics from them.

## Annotation and model records

The cleaned text is lowercased and automatically POS-tagged with spaCy 3.4.4 using `en_core_web_sm` 3.4.1; parser, NER, and text categorization components are disabled. Punctuation segmentation yields 904,222 model records and 8,449,144 POS-token units. Automatic POS errors are therefore a known source of measurement error. AUX tags receive an exhaustive rule-based lexical audit; the reported audit is not a human validation study.

## Splits

A deterministic 80/10/10 split contains:

| Split | Model records |
|---|---:|
| Train | 723,377 |
| Development | 90,422 |
| Test | 90,423 |

The split manifest SHA-256 is `05909372bd0245c90ff07bfe88f11a1cc5120c9b4fbe07e8a52d61026fd223be`. All experimental conditions use identical development and test indices; interventions alter training input only.

## Integrity identifiers

- Cleaned utterance file: SHA-256 `1360cb99960028a3d9013a8a82ed028722611fbd94d6e1a6f58e4e81def7f749`.
- Upstream BabyLM CHILDES component: SHA-256 `d6c0e39e2c283bfa5489cde80031a4fb9237ca8065a8301580df10e3b6843a98`.
- POS-tagged file: SHA-256 `c9abf03b5252565c2fa18213a8a2e09c0ef9b909d5fdd14cdfb533ec03bd5eaf`.
- Nested 720k-token sample: 93,606 utterances / 719,992 tokens, SHA-256 `5c60bac0a70141373a87cfcc1885343dc0c92d732ce88bb0c105e7f8e69079d3`.
- Nested 2M-token sample: 259,743 utterances / 1,999,991 tokens, SHA-256 `f88ac23293b7e8f34036023fd1f196e9c7d8008265d9b85b463e0efb6b47276f`.

## Intended use and exclusions

The derived data support research on distributional learnability and controlled language-model interventions. They are not suitable for profiling children or caregivers, estimating individual language ability, clinical decisions, educational assessment, or deployment. Results from English parental speech should not be generalized to all children, languages, families, or interaction settings.

## Privacy and release boundary

The cleaning procedure removes several CHAT omission/markup patterns but is not a comprehensive personally identifying information audit. Transcripts can still contain names and sensitive conversational content. No claim of de-identification is made beyond protections in the upstream distribution. This release therefore contains only aggregate statistics, hashes, code, and non-text configurations—not source or derived utterance strings. Per-example outputs and item identifiers linked to source contexts are also excluded.

## Reproduction files

The source and derived files identified above are local and excluded from Git.
The public processing entry points are `src/preprocess_childes.py` and
`src/annotate_pos.py`; the frozen paths and experimental settings are in
`configs/`.

## External provenance evidence

- BabyLM 2026 Strict dataset card: <https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict>
- Commit containing the Git-LFS identifier: <https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict/commit/c10defe96630b187709593c25fd60c58a3648efa>
- TalkBank ground rules: <https://talkbank.org/0share/rules.html>
- TalkBank access levels: <https://talkbank.org/0share/access.html>
