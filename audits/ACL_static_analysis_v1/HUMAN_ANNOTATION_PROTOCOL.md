# Blinded AUX context audit protocol

## Purpose

This audit checks whether tokens tagged `AUX` are contextually valid auxiliaries. It is independent of the paper's verb-class outcomes. Annotators must not inspect `aux_context_sample_automatic_key.csv`, model predictions, or one another's labels before both packets are frozen.

## Labels

- `valid_aux`: the highlighted token functions as an auxiliary in context, including accepted contractions and child-directed variants.
- `mistag`: the token is not an auxiliary in context and the AUX POS tag is linguistically wrong.
- `transcription_or_tokenization_artifact`: the apparent AUX tag is caused by spelling-out, transcript markup, token-boundary corruption, or a comparable transcription artifact.
- `uncertain`: the available context is insufficient. Explain the uncertainty in `review_notes`.

Enter exactly one label in `review_label` for every row. Do not alter `sample_id` or the context fields. Annotators A and B work independently and return only their own CSV.

## Execution

The two packets in `human_annotation/` use different deterministic row orders. After both are complete, run:

```bash
PYTHONPATH=src python3 src/score_aux_human_audit.py
```

The scorer validates all 500 IDs and labels, reports raw agreement and Cohen's kappa, and creates `results/disagreements_for_adjudication.csv`. Adjudication must remain blind to the automatic key. Construct a complete 500-row `human_annotation/adjudicated.csv`, then rerun the scorer to report adjudicated agreement with the automatic audit and accuracy by token-frequency stratum.

Because all automatically suspicious items are intentionally oversampled, the unweighted 500-item accuracy is not a corpus-prevalence estimate. Report the exhaustive lexical event counts separately and describe the contextual sample as a targeted validation audit.
