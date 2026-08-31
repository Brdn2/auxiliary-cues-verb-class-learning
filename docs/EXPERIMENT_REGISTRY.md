# Experiment registry

Status labels: **complete** means full machine-checkable outputs exist; **automatic phase complete** means a human validation remains pending; **runtime-qualified** means an accepted device/version mixture limits a causal comparison.

| ID | Paper role | Design | Primary metric | Status | Output |
|---|---|---|---|---|---|
| E0 | Scale validation | 720k, 2M, full CHILDES; 3 seeds | verb-class accuracy | complete | `results/Expanded_learning_curve_summary` |
| E1 | Legacy ablation | Original, AUX mask, 2 unmatched random masks, AUX identity shuffle; 3 seeds | CPS, class accuracy | complete/exploratory | `results/Expanded_learning_curve_full` |
| E2 | Annotation validity | exhaustive lexical AUX audit + 500-context blinded review sheet | suspicious-event rate | automatic phase complete; human review pending | `audits/ACL_static_analysis_v1` |
| E3 | Simple baselines | majority, POS frame, lexical NB, AUX-only NB, lexical ceiling | class accuracy | complete | `results/ACL_static_analysis_v1` |
| E4 | Architecture bias | 5 random initializations | class accuracy, CPS | complete | `results/ACL_random_init` |
| E5 | Mechanism | AUX reliability, entropy, MI and item-level cue diagnosticity | item CPS/accuracy association | complete | `results/ACL_static_analysis_v1` |
| E6 | Novel-verb generalization | diagnostic nonce exposure, exact POS-frame-disjoint evaluation; intact/AUX-delete/AUX-shuffle/matched-function; 3 base seeds | CPS primary, accuracy secondary; Holm correction over six intervention tests | complete | `results/ACL_nonce_cross_template_v2` |
| E7 | Confirmatory natural-corpus ablation | Original/AUX mask/content damage/function-word matched/AUX identity; 5 seeds | CPS primary, accuracy secondary | complete; total AUX loss, preregistered specificity criterion not met; matched/identity contrasts runtime-qualified | `results/ACL_confirmatory_aux_v2` |

The anonymous manuscript source is `paper/main.tex`. Its E7 table and prose are generated only after the matrix-completion gate; the current decision record states that the total AUX effect is supported but the preregistered specificity criterion is not.

The public artifact preflight is implemented with an explicit code-and-aggregate-results allowlist (`config/public_artifact_manifest.json`). Corpus text, models, per-example outputs, annotation contexts, debug/smoke files, and local paths are excluded. Archive generation is gated on completed E7 evidence.

## Preregistered E7 decision rule

The paper may claim an AUX-specific contribution only if AUX target ablation is worse than both matched controls on the primary CPS metric, with hierarchical paired-bootstrap confidence intervals excluding zero, and if this direction holds in at least four of five model seeds. Original-versus-target alone supports usefulness, not specificity.

This rule is executable rather than editorial: `src/update_paper_from_confirmatory.py` validates the complete evaluation matrix, applies the rule, records it in `CONFIRMATORY_DECISION.json`, and only then replaces the gated table and prose fragments used by `paper/main.tex`.

The same completion gate reads all 25 `training_summary.json` files and creates `training_diagnostics.csv` plus an appendix table of final train loss, unchanged-development loss, and wall-clock time. This makes generic intervention-induced language-modeling degradation visible alongside the primary CPS contrasts.

## Human annotation gate

Before submission, two annotators who do not see the automatic label should independently complete `aux_context_sample_for_blind_review.csv`. Report raw agreement, Cohen's kappa, adjudicated accuracy, and accuracy by frequency stratum. The current automatic audit must not be called a human annotation study.
