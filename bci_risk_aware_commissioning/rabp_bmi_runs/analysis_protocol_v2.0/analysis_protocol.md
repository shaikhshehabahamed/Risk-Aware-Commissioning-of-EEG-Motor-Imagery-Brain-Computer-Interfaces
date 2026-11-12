# Frozen Analysis Protocol v2.0

Date: 2026-06-09

## Status

`frozen_v2_for_reproducible_analysis_not_manuscript_submission`

This is not a manuscript draft. It is the locked analysis protocol for the next reproducibility and paper-preparation phase.

This v2.0 protocol supersedes v1.0 after an explicitly recorded exploratory audit of target coverage, analysis window, and classifier alternatives.

## Primary Research Claim

A risk-aware, budgeted, calibration-dependent operations-research framework can support cue-paced EEG command gating by explicitly trading channel burden, calibration burden, deployment, command coverage, accepted-command reliability, and false-command exposure.

## Primary Policy

```text
topk21_m15_rule_t03_ba065_fcr010
0.5-3.5 s post-cue window
top-k 21 channels, 60 calibration trials
target coverage 0.3
deploy if calibration accepted BA >= 0.65 and calibration FCR <= 0.10
```
Primary validated operating point:
```text
deployment rate: 0.571 [0.505, 0.641]
accepted BA: 0.858 [0.825, 0.886]
unconditional FCR: 0.016 [0.010, 0.023]
unconditional coverage: 0.146 [0.125, 0.167]
```
## Operating-Point Hierarchy

| Rank | Mode | Role | Channels | Trials | Deploy | BA | Uncond. FCR | Uncond. Cov. |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | topk21_m15_rule_t03_ba065_fcr010 | primary_policy | 21.0 | 60.0 | 0.571 [0.505, 0.641] | 0.858 [0.825, 0.886] | 0.016 [0.010, 0.023] | 0.146 [0.125, 0.167] |
| 2 | topk16_m15_rule_t03_ba070_fcr010 | minimal_channel_secondary | 16.0 | 60.0 | 0.549 [0.479, 0.619] | 0.852 [0.818, 0.882] | 0.016 [0.011, 0.022] | 0.140 [0.119, 0.162] |
| 3 | all64_m15_no_gate | primary_comparator | 64.0 | 60.0 | 1.000 [1.000, 1.000] | 0.852 [0.827, 0.875] | 0.046 [0.034, 0.060] | 0.326 [0.305, 0.347] |
| 4 | topk16_m15_no_gate | always_deploy_budgeted_baseline | 16.0 | 60.0 | 1.000 [1.000, 1.000] | 0.818 [0.790, 0.842] | 0.051 [0.039, 0.064] | 0.279 [0.256, 0.301] |
| 5 | topk21_m15_no_gate | secondary_budgeted_baseline | 21.0 | 60.0 | 1.000 [1.000, 1.000] | 0.826 [0.800, 0.852] | 0.048 [0.037, 0.061] | 0.285 [0.262, 0.307] |
| 6 | topk16_m15_rule_ba075_fcr010 | safety_secondary | 16.0 | 60.0 | 0.413 [0.340, 0.486] | 0.855 [0.820, 0.884] | 0.017 [0.011, 0.024] | 0.150 [0.122, 0.178] |
| 7 | topk21_m15_t03_meta_p060 | high_confidence_secondary | 21.0 | 60.0 | 0.130 [0.083, 0.181] | 0.927 [0.870, 0.969] | 0.003 [0.000, 0.008] | 0.042 [0.025, 0.060] |
| 8 | central21_m15_no_gate | biological_sensitivity | 21.0 | 60.0 | 1.000 [1.000, 1.000] | 0.792 [0.764, 0.818] | 0.061 [0.049, 0.074] | 0.311 [0.290, 0.330] |
| 9 | fixed_motor21_m15_no_gate | motor_only_negative_sensitivity | 21.0 | 60.0 | 1.000 [1.000, 1.000] | 0.785 [0.756, 0.811] | 0.062 [0.051, 0.075] | 0.305 [0.285, 0.324] |

## Locked Decisions

| ID | Decision | Locked Choice | Rationale |
|---|---|---|---|
| D01 | Primary scientific framing | Risk-aware operations-research framework for cue-paced EEG command gating. | This matches the validated evidence and avoids overclaiming pure motor decoding. |
| D02 | Primary operating policy | Primary21: top-k 21 channels, 60 calibration trials, target coverage 0.3, deploy if calibration BA >= 0.65 and FCR <= 0.10. | Strongest validated budgeted reliability-safety-access tradeoff after the v2 window and target sweep. |
| D03 | Always-deploy budgeted baseline | TopK16 no-gate. | Shows resource reduction relative to all channels without readiness gating. |
| D04 | Primary comparator | All64 no-gate. | Conventional full-channel benchmark. |
| D05 | High-confidence secondary | TopK16 target0.4 Rule75 and TopK21 target0.3 Meta21. | Useful sensitivity modes for high-confidence operation, but not the best primary deployment/coverage tradeoff. |
| D06 | Biological sensitivity | Central21 and Motor21. | Quantifies what remains when posterior/frontal or non-motor channels are constrained. |
| D07 | Statistical validation | Subject-level bootstrap confidence intervals and paired subject-level comparisons. | Preserves subject heterogeneity and avoids trial-level pseudo-replication. |
| D08 | Non-inferiority interpretation | Use 0.05 BA margin only as a screening/engineering margin. | No clinical margin has been externally established for this dataset and task. |
| D09 | Protocol amendment | v2.0 supersedes v1.0 after exploratory operating-point and window audits. | The 0.5-3.5 s window improved the validated budgeted policy frontier; this amendment records the data-driven pivot explicitly. |

## Out Of Scope Claims

| Item | Status | Reason |
|---|---|---|
| new_classifier_claim | out_of_scope | Novelty is in decision framework, budget formulation, evaluation design, and deployment policy. |
| pure_motor_cortex_decoding_claim | out_of_scope | Top-k policies select meaningful frontal/posterior fractions; cue-paced command-gating framing is more honest. |
| self_paced_asynchronous_bmi_claim | out_of_scope | Dataset and labels support cue-paced analysis, not self-paced asynchronous command detection. |
| universal_deployment_claim | out_of_scope | Readiness gates deliberately reject many sessions; lower-tail subjects remain a limitation. |
| clinical_effectiveness_claim | out_of_scope | The public dataset supports methodological validation, not clinical efficacy. |

## Change Control

Any change to the primary cohort, task definition, primary time window, primary operating policy, endpoint hierarchy, or statistical validation plan must create a new protocol version with a rationale.
