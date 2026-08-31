# Contributing

Bug reports and reproducibility improvements are welcome. Please keep issues
and pull requests focused on the released code, configurations, aggregate
results, or documentation.

Before opening a pull request:

1. Run `python3 scripts/verify_public_artifact.py`.
2. Run `PYTHONPATH=src pytest -q` in the pinned Python 3.11 environment.
3. Do not attach or commit CHILDES/BabyLM utterances, derived transcript text,
   per-example predictions, evaluation prompts, checkpoints, tokenizers,
   blinded annotation contexts, personal data, or machine-specific paths.
4. Explain any result-affecting change and identify the experiment IDs from
   `docs/EXPERIMENT_REGISTRY.md`.

Security or privacy concerns involving research data should be reported
privately to the contact address in the manuscript rather than posted with
examples in a public issue.
