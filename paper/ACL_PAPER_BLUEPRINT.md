# ACL paper blueprint

## Working title

**Do Auxiliary Cues Support Novel Verb Meaning Induction from Child-Directed Speech?**

## One-sentence answer

Small masked language models trained on naturalistic child-directed speech use auxiliary–context regularities when learning verb semantic classes, but the effect is graded and shared with broader function-word structure rather than uniquely attributable to auxiliaries.

## Abstract skeleton

Children can infer aspects of verb meaning from the linguistic environments in which verbs occur. We test whether English auxiliaries contribute to this process in masked language models trained from scratch on child-directed speech. Across increasing data scales, verb-class prediction rises from 53.7% to 72.2%. Removing auxiliaries from training reduces performance, but unmatched random ablations leave damage-based explanations unresolved. We therefore introduce complementary sentence-damage and token-covariate controls, an identity-shuffle intervention, leakage-controlled statistical baselines, and a held-out-template nonce-verb task. Simple AUX-only statistics reach 30.5%, compared with 44.5% for POS frames, 52.0% for lexical co-occurrence, and 72.2% for the trained model. In diagnostic nonce exposures, shuffling AUX forms lowers cross-template accuracy by 3.27 points and class preference by 0.096 relative to intact exposures, while deleting matched non-AUX function words also affects the continuous preference measure. These results support auxiliaries as one component of a redundant cue system, not as a unique source of verb semantics.

E7 is complete. Its total Original-to-target CPS loss is $-0.144$ (95\% CI [$-0.207$, $-0.085$]), but neither preregistered specificity contrast excludes zero. The final abstract must retain the conclusion ``overall contribution, not AUX-specificity''.

## Research questions and claim strength

1. **RQ1: Can models learn verb semantic classes from child-directed speech at realistic data scales?**
   - Supported: accuracy is 53.7% at 720k tokens, 61.4% at 2M, and 72.2% at the full training scale.
2. **RQ2: Are AUX statistics useful beyond generic corpus damage?**
   - Overall target removal is associated with a CPS loss, but the completed E7 primary comparisons do not establish an AUX-specific loss beyond both matched damages.
3. **RQ3: Do AUX form–context mappings help induce meanings for previously unseen verb forms across unseen templates?**
   - Supported narrowly in diagnostic exposures: AUX shuffle causes a significant 3.27-point accuracy drop; AUX deletion significantly lowers CPS but not discrete accuracy.
4. **RQ4: Are auxiliaries unique?**
   - Not supported. Matched non-AUX function-word removal also lowers CPS. The mature claim is a redundant multi-cue account.

## Contributions

1. A scale-controlled study of verb semantic class learning from 904,228 cleaned child-directed utterances (904,222 post-annotation model records) using models trained from scratch.
2. A causal intervention suite that distinguishes total AUX usefulness, form–context identity, sentence-level damage, and token-level closed-class damage.
3. A nonce-verb benchmark with equal tokenizer piece lengths and exact local POS-frame disjointness between exposure and evaluation.
4. A mechanism analysis connecting corpus-level AUX diagnosticity to item-level model behavior.
5. Reproducible event-level audits, five-seed random-init controls, and hierarchical seed-by-item uncertainty estimates.

## Recommended section structure

### 1. Introduction

Frame the work as computational evidence about cue use, not a cognitive equivalence claim. End with the four research questions and the redundant-cue hypothesis.

### 2. Background

Cover syntactic bootstrapping, function-word/prosodic cues, computational acquisition from child-directed speech, and masked-LM diagnostics. Explicitly distinguish this paper from broad function-word cue studies: the contribution is causal verb-semantic evaluation plus novel-form transfer.

### 3. Data and task

Report speaker filtering, 904,222 utterances, tokenization, train/dev/test split, five verb classes, candidate balancing, and target-only masking at evaluation. Include the AUX automatic audit and the pending human-review protocol.

### 4. Models and interventions

Describe the 2-layer RoBERTa model, training budget, seeds, and five E7 conditions. State what each control identifies. Put full covariate-balance tables in the appendix.

### 5. Experiments

1. Scale curve.
2. Simple and random-init baselines.
3. Confirmatory five-condition natural-corpus ablation.
4. Cue-statistic mechanism analysis.
5. Nonce-verb cross-template adaptation.

### 6. Results

Lead with answer-first statements. Report seed-level points and hierarchical confidence intervals. Do not use `p=0`; report `p<1/B` for B bootstrap draws.

### 7. Discussion

Argue for redundant cue integration. Separate evidence about model behavior from claims about children. Discuss why identity shuffle is more diagnostic than deletion and why content controls cannot simultaneously match AUX position and frequency.

### 8. Limitations and ethics

Include English-only data, automatic POS tags, selected verb ontology, small architecture, no direct child behavioral fit, diagnostic nonce subset selection, three base seeds for nonce adaptation, and substantial compute cost. Describe CHILDES licensing and privacy handling.

## Main tables

### Table 1: Data and task

Corpus sizes, split sizes, AUX event counts, verb-class examples, tokenizer vocabulary, model parameters.

### Table 2: Baselines and scale

| System | Accuracy |
|---|---:|
| Uniform random, 5-seed random init | 17.8–20.0% |
| Majority class | 20.0% |
| AUX-only Naive Bayes | 30.5% |
| POS frame count | 44.5% |
| Lexical co-occurrence NB | 52.0% |
| MLM, 720k | 53.7% |
| MLM, 2M | 61.4% |
| MLM, full | 72.2% |

### Table 3: Confirmatory E7

Report Original, AUX target, content damage, matched function word, and identity shuffle with mean and seed SD. Report the primary CPS contrasts, their hierarchical 95% CIs, and seed-direction counts in text; report intact-dev-loss diagnostics in the appendix.

### Table 4: Nonce cross-template transfer

Report zero-shot and three exposure interventions. Primary contrast: intact minus AUX shuffle. Secondary contrasts: intact minus AUX deletion and intact minus matched-function deletion.

## Main figures

1. Learning curve with seed points.
2. E7 paired seed-and-item effects, not bar charts alone.
3. Nonce transfer deltas with hierarchical confidence intervals.
4. AUX cue reliability/entropy versus model item preference.

## Results currently safe to report

- Automatic AUX lexical plausibility: 99.9925%; 70/935,364 events flagged for contextual review.
- Baselines: AUX-only 30.5%, POS-frame 44.5%, lexical co-occurrence 52.0%, lexical identity ceiling 100% (explicitly leakage-prone and not a task baseline).
- Original model cue association: AUX log-odds correlates with accuracy at 0.20–0.24 and with CPS at Spearman 0.15–0.19 across seeds.
- Diagnostic nonce exposures: intact versus AUX shuffle accuracy delta +3.27 points, 95% CI [1.07, 5.47], Holm-adjusted bootstrap p=.0136; CPS delta +0.096, CI [0.048, 0.138], Holm-adjusted p<.0006. The multiplicity family contains the three intervention contrasts across both metrics (six tests).
- Intact versus AUX deletion: accuracy delta +2.07 points, CI [-0.87, 5.27]; CPS delta +0.095, CI [0.029, 0.154].
- Intact versus matched-function deletion: accuracy delta +1.20 points, CI [-1.33, 3.80]; CPS delta +0.083, CI [0.048, 0.116].

## Claims that are not safe

- “Auxiliaries uniquely cause verb semantic learning.”
- “The matched content control is statistically equivalent to AUX positions.”
- “The completed E7 experiment establishes an AUX-specific effect beyond both matched controls.”
- “The AUX audit is human-validated.”
- “The model reproduces child behavior.”

## Submission gate

E7 is complete: 25/25 models, aligned evaluation, hierarchical contrasts, and dev-loss diagnostics are generated. Submission still requires two-person AUX annotation, source-term confirmation, completion of the ACL responsible-NLP checklist, and at least one clean-room rerun from the frozen environment.
