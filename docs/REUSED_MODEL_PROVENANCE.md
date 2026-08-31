# Reused model provenance

Nine full-corpus checkpoints are reused from the earlier expanded experiment rather than retrained:

- conditions: `original`, `AUX_target_ablation`, `AUX_identity_shuffle`;
- model seeds: 2026, 2027, 2028;
- source: `../MVP_Pilot实验/models/Expanded_learning_curve_full`;
- destination: `models/ACL_confirmatory_aux_v2`.

Reuse is valid because the condition files are byte-identical. SHA-256 values from both manifests are:

| Condition | SHA-256 |
|---|---|
| Original | `c177bc6d5f2ca514686e920425ffee981df68e349647bebc7801d1d831315111` |
| AUX target ablation | `6a8a05c1338af64770a8871a1b04b49c1c8a83c5182d329150c85c6cf6429bc2` |
| AUX identity shuffle | `ace4f5fd9c29d5fb6c2e57d5cb455bb2ccb9a024b0b0eb072950fd0d772f68e6` |

The identity generator uses `identity_shuffle_salt: Expanded_learning_curve_full` specifically to reproduce the earlier deterministic derangement. The new matched-control corpora do not have reusable checkpoints and must be trained from scratch.
