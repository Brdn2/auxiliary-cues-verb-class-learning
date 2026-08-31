# Public artifact policy

This project keeps a complete local research workspace and a smaller public
artifact as separate objects. The local workspace contains restricted CHILDES
derivatives, trained checkpoints, logs, exploratory outputs, and annotation
materials. Their presence in the workspace does not authorize redistribution.

## Public boundary

The public artifact is generated only from the explicit allowlist in
`config/public_artifact_manifest.json`. It contains:

- source code, tests, pinned configurations, and portable runner scripts;
- manuscript sources, generated tables and figures, and the compiled paper;
- aggregate metrics, confidence intervals, decision records, and non-text audit
  summaries;
- provenance, model-card, data-statement, responsible-research, and release
  documentation.

It deliberately excludes:

- `data/`, including raw and derived utterance text and train/dev/test splits;
- `models/`, including checkpoints and tokenizers;
- `archive/`, `logs/`, `literature/`, temporary/build directories, and local
  notes or word-processing documents;
- per-example predictions, exposure/evaluation prompts, matched corpus rows,
  blinded contexts, automatic annotation keys, and lexical inventories;
- every file marked debug or smoke.

The model-card and source-license record explain why checkpoints and corpus
text require a separate author/institutional decision. This conservative
package is code-and-aggregate-results only; it is not a declaration that the
underlying transcript material has been relicensed.

## Fail-closed build

Before copying any file, `src/build_public_artifact.py` resolves the allowlist
and rejects symlinks, forbidden roots or filenames, oversized files, and text
files containing machine-specific home-directory paths. An actual archive
cannot be built without `--require-e7`; check-only mode is available while E7
is running. The builder emits both a deterministic ZIP archive and
`MANIFEST.sha256`, so every packaged byte can be audited.

The allowlist is intentionally explicit. Adding a new analysis output to the
working directory does not add it to the public artifact. A maintainer must
review its privacy and provenance implications and then update the manifest.

## Pre-release human checks

The package builder prevents known unsafe file classes from entering the
archive, but it does not replace institutional or legal review. Before public
release, authors must also:

1. complete and record the source-license verification;
2. inspect the generated file list and checksum manifest;
3. confirm that no newly allowlisted result contains recoverable transcript
   content or personal data;
4. complete the responsible-NLP checklist and human AUX annotation gate.
