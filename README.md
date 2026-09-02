# Auxiliary cues and novel verb-class learning

Research materials for the ACL-format course paper **“Are Auxiliary Cues
Uniquely Informative for Coarse Verb-Class Learning in Masked Language Models
Trained on CHILDES?”**

The paper studies whether English auxiliary identity contributes uniquely to
coarse verb-class expectations in small masked language models trained from
scratch on child-directed speech. The supported conclusion is deliberately
narrow: auxiliary information is useful within a redundant distributional
system, but the matched-control experiments do not establish an AUX-specific
effect.

[Read the paper (PDF)](output/pdf/paper.pdf)

![Training-scale and baseline results](paper/figures/baseline_gradient.png)

## Main results

- Five-way verb-class accuracy increases from **53.7%** at 720k training
  tokens to **72.2%** at the full training scale.
- Leakage-controlled accuracies are **30.5%** for AUX-only Naive Bayes,
  **44.5%** for local POS frames, and **52.0%** for lexical context.
- AUX-identity ablation lowers the five-seed class preference score by
  **0.144** relative to unmodified input, but the loss is not reliably larger
  than both matched-damage controls.
- In the nonce-form experiment, shuffling AUX identities lowers cross-template
  accuracy by **3.27 percentage points** and class preference by **0.096**.

## Repository contents

| Path | Contents |
|---|---|
| `paper/` | ACL LaTeX source, references, figures, and tables used by the paper |
| `output/pdf/` | Compiled course-paper manuscript |
| `src/` | Data preparation, training, evaluation, and statistical analysis code |
| `config/` | Frozen configurations for the reported experiments |
| `results/` | Release-safe aggregate results supporting the reported values |
| `docs/DATA_STATEMENT.md` | Data provenance, processing, privacy, and access limits |

## Verify the released results

The lightweight verifier uses only the Python standard library and recomputes
the headline values from the released aggregate tables:

```bash
python3 scripts/verify_public_artifact.py
```

## Reproduction scope

The repository contains the principal code and frozen configurations, but not
the licensed source corpus, derived utterance text, per-example predictions,
tokenizers, or model checkpoints. Consequently, the released aggregate values
can be checked directly, while confidence intervals and trained models can be
recomputed only after independently obtaining the source data and regenerating
the local intermediate files.

The reference environment is Python 3.11 with versions pinned in
`requirements.txt`. After obtaining the CHILDES component of BabyLM 2026
Strict, keep it outside version control and prepare it locally:

```bash
python src/prepare_expanded_childes.py \
  /path/to/childes.train.txt data/expanded_childes
python src/annotate_plaintext_spacy.py \
  data/expanded_childes/childes_train2_cds_full_cleaned.txt \
  data/expanded_childes/pos/childes_train2_cds_full_spacy_pos.txt
```

Run the five-condition confirmatory experiment with:

```bash
bash scripts/run_full_confirmatory.sh
```

After the Original checkpoints for seeds 2026--2028 are available, run the
nonce experiment and its summary:

```bash
PYTHONPATH=src python src/run_nonce_cross_template.py
PYTHONPATH=src python src/summarize_nonce.py
```

The static baseline and cue-behavior analyses additionally require local
per-example predictions and the locally generated evaluation inventory. These
are intentionally excluded because they retain links to source contexts.

## Data access and release boundary

The source is the CHILDES component of
[BabyLM 2026 Strict](https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict),
subject to the applicable [TalkBank access and ground
rules](https://talkbank.org/0share/rules.html) and any corpus-specific terms.
Users must obtain the source independently.

No raw or derived transcript text, per-example predictions, evaluation
contexts, tokenizers, checkpoints, or model logs are distributed here. See
[`docs/DATA_STATEMENT.md`](docs/DATA_STATEMENT.md) for hashes, selection rules,
split sizes, and privacy limitations.

## Citation and license

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Until a
proceedings record exists, cite the manuscript title shown above.

Original code is licensed under the MIT License. Original documentation,
aggregate results, and figures are licensed under CC BY 4.0. The manuscript,
ACL style files, and upstream datasets retain the separate terms described in
[`LICENSE`](LICENSE).
