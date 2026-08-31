# Auxiliary cues and novel verb-class learning

Public research materials for **“Are Auxiliary Cues Uniquely Informative for
Coarse Verb-Class Learning in Masked Language Models Trained on CHILDES?”**

This repository accompanies an ACL-format manuscript on whether auxiliary
identity contributes to coarse verb-class expectations in small masked
language models trained from scratch on child-directed English. It contains the
analysis and training code, frozen configurations, aggregate results, tests,
paper source, figures, and release documentation.

> **Release boundary.** CHILDES/BabyLM utterances, derived transcript text,
> per-example predictions, evaluation prompts, tokenizers, and model
> checkpoints are not redistributed. See [Data access and provenance](#data-access-and-provenance).

## Main findings

- Mean five-way verb-class accuracy rises from **53.7%** at 720k training
  tokens to **72.2%** at the full training scale.
- Leakage-controlled accuracies are **30.5%** for AUX-only Naive Bayes,
  **44.5%** for local POS frames, and **52.0%** for lexical context.
- In the five-seed confirmatory experiment, AUX-identity ablation reduces class
  preference score (CPS) by **0.144**, 95% CI **[0.085, 0.207]**, relative to
  unmodified input. The loss is not reliably larger than both matched-damage
  controls, so the evidence does **not** establish an AUX-specific effect.
- In the nonce-form experiment, shuffling AUX identities reduces
  cross-template accuracy by **3.27 percentage points** and CPS by **0.096**.
  Matched function-word ablation also reduces CPS.

The supported interpretation is deliberately narrow: auxiliary information is
one useful cue in a redundant distributional system, not a uniquely necessary
source of verb semantics and not evidence for a child-level causal mechanism.

![Training-scale and baseline results](paper/figures/baseline_gradient.png)

## Repository map

| Path | Contents |
|---|---|
| `config/` | Frozen data, model, intervention, and evaluation settings |
| `src/` | Corpus preparation, training, evaluation, statistics, and audits |
| `scripts/` | Portable experiment and verification entry points |
| `results/` | Aggregate tables, confidence intervals, and decision records |
| `audits/` | Non-text release-safe protocol and matching summaries |
| `paper/` | ACL LaTeX source, references, figures, and generated tables |
| `output/pdf/` | Current compiled manuscript |
| `docs/` | Data statement, model card, compute report, and release policy |

## Verify the public artifact

The lightweight verifier uses only the Python standard library. It checks the
released file set and recomputes the headline values from aggregate tables:

```bash
python3 scripts/verify_public_artifact.py
```

To run the source-level tests:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
PYTHONPATH=src pytest -q
```

The reference environment is Python 3.11. Dependency versions are pinned in
`requirements.txt`; model and training settings are frozen in `config/`.

## Reproduce the experiments

Full retraining requires lawful local access to the BabyLM 2026 Strict CHILDES
component. After placing the source outside version control, prepare and tag it:

```bash
python src/prepare_expanded_childes.py \
  /path/to/childes.train.txt data/expanded_childes
python src/annotate_plaintext_spacy.py \
  data/expanded_childes/childes_train2_cds_full_cleaned.txt \
  data/expanded_childes/pos/childes_train2_cds_full_spacy_pos.txt
```

Then build the five intervention corpora and run/resume the 25-model
confirmatory matrix:

```bash
export PYTHONPATH=src
python src/acl_conditions.py --force
python src/run_confirmatory_ablation.py \
  --config config/ACL_confirmatory_aux.yaml --resume
```

`scripts/finalize_acl_paper.sh` recomputes statistics and paper artifacts after
the completion gate passes. It refuses to update the manuscript unless all five
conditions contain the same five seeds and the same 600 evaluation items per
seed. The complete experiment registry and evidence mapping are in
[`docs/EXPERIMENT_REGISTRY.md`](docs/EXPERIMENT_REGISTRY.md).

## Data access and provenance

The source is the CHILDES component of
[BabyLM 2026 Strict](https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict),
ultimately governed by the applicable
[TalkBank access and ground rules](https://talkbank.org/0share/rules.html) and
any corpus-specific terms. The exact source SHA-256, selection rules, split
sizes, privacy boundary, and intended use are documented in
[`docs/DATA_STATEMENT.md`](docs/DATA_STATEMENT.md).

The repository-level BabyLM license label is not treated as proof that every
underlying transcript was relicensed. Users must obtain the source themselves,
follow the applicable terms, and keep transcript material out of forks and pull
requests.

## Citation

Please use [`CITATION.cff`](CITATION.cff). Until a proceedings record is
available, cite this repository and the manuscript title shown above.

## License and responsible use

Original code is released under the MIT License. Original documentation,
aggregate results, and figures are released under CC BY 4.0. Third-party ACL
style files and all upstream datasets retain their own terms. See
[`LICENSE`](LICENSE) for the exact scope.

The models and metrics are research probes. They must not be used to assess
children, caregivers, language development, clinical status, educational
ability, or demographic traits. See [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md)
and [`CONTRIBUTING.md`](CONTRIBUTING.md).
