# Model card: small RoBERTa MLMs for auxiliary-cue interventions

## Model family

The project trains RoBERTa-style masked language models from scratch. Each main model has 946,208 trainable parameters: two Transformer layers, hidden size 128, four attention heads, intermediate size 512, maximum position embeddings 128, and a 4,000-type byte-level BPE vocabulary. Parameters use float32 checkpoints.

## Training

Main models train for five epochs with batch size 16, learning rate `5e-4`, weight decay `0.01`, gradient clipping at `1.0`, MLM probability `0.15`, and dropout `0.1`. E7 uses seeds 2026--2030 with no E7 hyperparameter search. The five training conditions are Original, AUX target ablation, matched content damage, matched non-AUX function-word damage, and AUX identity shuffle. Development and test text remain unchanged.

Nine E7 checkpoints are reused from byte-identical earlier conditions. Six
additional MPS checkpoints are imported after hash verification; the local
overlap for `original/2029` is preserved rather than overwritten. The import
record, including device/version heterogeneity, is
`audits/ACL_confirmatory_aux_v2/MBP_E7_SIX_IMPORT_PROVENANCE.json`. The
remaining checkpoints are trained within this project. E7's Original and AUX
target cells are MPS, matched controls are CPU, and identity shuffle is mixed;
cross-runtime contrasts are therefore qualified sensitivity evidence.

## Inputs and outputs

The models consume lowercased English child-directed utterances tokenized with the shared BPE tokenizer. At evaluation, the target verb is replaced by one or more mask tokens. Outputs are token log probabilities, from which the project computes target-surface probabilities, five-way verb-class accuracy, and class preference score (CPS).

## Intended use

These models are research probes for whether distributional cues in child-directed text can support coarse verb-class learning and nonce-form transfer. They are suitable for controlled ablation, cue-correlation, and reproducibility analyses within the documented corpus and task.

## Out-of-scope use

The models are not cognitive models of individual children and must not be used to assess children, caregivers, families, language development, clinical status, education, or demographic traits. They are not general-purpose language models and are not intended for deployment or text generation.

## Evaluation evidence

- Scale experiment: five-way class accuracy increases from 53.7% at 720k tokens to 61.4% at 2M and 72.2% on the full split, averaged over three seeds.
- Leakage-controlled baselines: AUX-only NB 30.5%, POS-frame count 44.5%, lexical-context NB 52.0%.
- Random initialization: 17.8--20.0% across five seeds.
- Nonce transfer: intact exposure exceeds AUX identity shuffle by 3.27 accuracy points and 0.096 CPS under hierarchical paired bootstrap; matched-function deletion also affects CPS.
- E7: the complete 5-condition $\times$ 5-seed $\times$ 600-item matrix finds a total Original-to-AUX-target CPS loss of -0.144 (95% CI [-0.207, -0.085]), but neither preregistered CPS specificity contrast excludes zero. The matched-control and identity contrasts retain the documented runtime qualification.

## Limitations

The corpus is English-only and text-only; POS tags are automatic; the semantic ontology has five researcher-defined classes; the model is deliberately small; and nonce exposure contexts are selected to contain diagnostically informative AUX cues. Accuracy and CPS do not establish childlike representations or behavior. Ablations alter the training distribution, and unchanged-development loss must be considered alongside class metrics.

## Compute and environment

Experiments run on a single Apple M3 machine with eight CPU cores and 16GB memory, using CPU or MPS as recorded in each training summary. `results/ACL_compute_budget/compute_budget_summary.json` reports a deduplicated lower bound over logged runs and explicitly lists untimed work.

## Data and release

See `docs/DATA_STATEMENT.md`. Source and derived utterance text should not be publicly redistributed until the exact source terms and a release-specific privacy review are complete. Code, aggregate results, hashes, manifests, and model checkpoints require a project license decision before public release.
