# Released results

This directory contains aggregate results only. Files with transcript text,
evaluation contexts, item identifiers tied to source records, or per-example
predictions are intentionally excluded.

- `natural_corpus/`: five-seed condition summaries, paired hierarchical
  bootstrap intervals, and the prespecified AUX-specificity decision.
- `nonce/`: three-seed nonce-transfer summaries and paired hierarchical
  bootstrap intervals.
- `statistics/`: learning-curve values, leakage-controlled baselines, random
  initialization results, and cue--behavior correlations.

Run `python3 scripts/verify_results.py` from the repository root to recompute
the headline values reported in `README.md` from these files.
