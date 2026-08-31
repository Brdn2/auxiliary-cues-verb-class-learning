# ARR Responsible NLP Research checklist evidence map

This is a working evidence map for the current ARR submission form, not a substitute for entering the checklist in the official form. Status is deliberately fail-closed.

| Item | Status | Evidence or required action |
|---|---|---|
| A1 Limitations | ready | Unnumbered `Limitations` section scopes language, ontology, architecture, nonce selection, seeds, and behavioral interpretation. |
| A2 Risks | ready | `Ethical Considerations` limits use to research and discusses privacy, representational, and environmental risks. |
| A3 Claims summarized | ready | Abstract, Introduction, Results, Discussion, and Conclusion are updated from the completed E7 matrix; the text states total AUX contribution without unsupported specificity. |
| B1 Artifact citation | partial | CHILDES and the exact BabyLM 2026 source distribution are cited; the merged file does not retain corpus-level identifiers needed to guarantee every contributor-specific citation. **[TODO—authors]** Confirm the required citations for every contributing source corpus before submission. |
| B2 Licenses/terms | awaiting authors | File origin and hash are verified. The BabyLM repository is tagged MIT, while TalkBank's default data terms are CC BY-NC-SA 3.0 plus corpus-specific obligations. Apply the stricter non-commercial/no-text-redistribution boundary and obtain author/institutional confirmation before release. **[TODO—authors]** Record the approval and its date. |
| B3 Intended use | ready | Research-only interpretation and no deployment claim are stated. |
| B4 Identifiers/offensive content | partial | Only parental speech is selected and marked transcript omissions are removed, but a release-specific privacy audit is still needed if derived text is distributed. **[TODO—authors]** Document the audit decision before any text-containing release. |
| B5 Artifact documentation | partial | English, child-directed domain, speaker-role filter, preprocessing and scope are documented; source-corpus demographic coverage remains to be added if available. **[TODO—authors]** Add source-corpus demographic coverage if it can be documented without inference. |
| B6 Data statistics | ready | Utterance/token counts and fixed 80/10/10 split are reported with manifests and hashes. |
| C1 Parameters/compute/infrastructure | ready | 946,208 trainable parameters and Apple M3/8-core/16GB infrastructure are recorded. `results/ACL_compute_budget` gives an 83.0 device-hour lower bound across 243 deduplicated logged runs and lists untimed work. |
| C2 Experimental setup | ready | Architecture, fixed hyperparameters, seeds, no E7 hyperparameter search, preprocessing, conditions and primary metric are specified. |
| C3 Descriptive statistics | ready | The completed 5-condition $\times$ 5-seed $\times$ 600-item E7 matrix reports seed means, 10,000-resample hierarchical intervals, seed deltas, and intact-development-loss diagnostics. |
| C4 Packages | ready | `requirements.txt`, spaCy model version, configs and model manifests record implementations and versions. |
| D1 Annotation instructions | ready | Full blinded AUX protocol and label definitions are in `audits/ACL_static_analysis_v1/HUMAN_ANNOTATION_PROTOCOL.md`. |
| D2 Recruitment/payment | awaiting authors | **[TODO—authors]** Record how the two annotators are recruited and compensated. |
| D3 Consent | awaiting authors | **[TODO—authors]** Record informed consent for the annotation task if required by the institutional context. |
| D4 Ethics review | awaiting authors | **[TODO—authors]** Determine and report approval/exemption requirements without breaking anonymity. |
| D5 Annotator characteristics | awaiting authors | **[TODO—authors]** Report only relevant, voluntarily provided characteristics; do not infer them. |
| E1 AI assistance disclosure | awaiting authors | **[TODO—authors]** Disclose coding/writing assistance in the checklist or acknowledgments in accordance with ACL policy; AI tools are not authors. |

Official guidance consulted on 2026-08-26: <https://aclrollingreview.org/responsibleNLPresearch/> and <https://acl-org.github.io/ACLPUB/formatting.html>.
