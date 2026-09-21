# B01 — S1 representative model selection

Date: 2026-09-21  
Base inspected: `dev@be21c1dd3f8140e89b5706d4f58178d7af98f43d`

This is a candidate-audit result, not a model implementation PR.

## 1. Why S1 first

The current Model Factory already contains linear/statistical control, ResNet/TCN-style CNNs, recurrent models and conventional Transformer-family implementations. S1 supervised diagnosis therefore has an existing Task and broad baseline families; starting here does not require a new Task contract.

S2 forecasting remains gated by the separate point-forecasting Task decision recorded in B00.

## 2. Scientific role test

The first new S1 model should add a clearly different inductive bias rather than another generic CNN/Transformer label.

| Candidate | Distinct role | Source status | First-pilot judgment |
| --- | --- | --- | --- |
| ModernTCN | modern large-kernel/depthwise convolution | official classification repository, MIT | defer: overlaps existing TCN/ResNet until G2/E2 justify incremental value |
| TSLANet | adaptive spectral filtering + interactive convolution | official classification source, MIT | defer: distinct but heavier fidelity/dependency boundary |
| iTransformer | variate/channel-token attention | official main repository is forecasting-only | defer: classification variant/source identity unresolved |
| TimesNet | frequency-selected periodic 2D variation | THUML Time-Series-Library classification branch, MIT | **preferred first S1 candidate**, but not yet QUEUED |

## 3. TimesNet evidence actually established

Exact source selected for the candidate audit:

```text
repository: thuml/Time-Series-Library
revision: 4e938a1767106324dd753b2a44832bf870a0252e
file: models/TimesNet.py
license: MIT
variant: classification
```

The upstream file explicitly implements `task_name == 'classification'`; its core block:
1. computes an RFFT over time;
2. excludes the zero-frequency bin from selection;
3. chooses top periodic frequencies;
4. reshapes 1-D variation into period-aligned 2-D grids;
5. applies inception-style 2-D convolution;
6. adaptively aggregates period-specific representations.

That role is materially different from the current TCN/ResNet baselines and is the reason for preference; paper popularity is not the inclusion reason.

Historical source material in closed #266 is **not current-dev verification**, but its JUnit artifact provides useful feasibility evidence:
- 19 TimesNet component/contract cases total about 0.175 s;
- one real public Dummy CLI case took 7.696 s;
- the historical implementation exercised forward/backward/update, state restoration, negative contracts and a period-grid reference test.

These measurements stay `historical_E1_reference_only`; they do not establish current E1/E2/E3 or final G7.

## 4. Gates

TimesNet:

```text
G1 scientific role          PASS
G2 non-duplicate information PASS
G3 auditable source         PASS
G4 code license             PASS
G5 current S1 Task fit      PASS
G6 fidelity verifiability   PARTIAL
G7 maintenance/resource     PARTIAL
```

G6 still needs a bounded current-dev implementation with an oracle against the fixed upstream revision.

G7 still needs candidate-specific memory/import evidence in the actual implementation environment. Historical timing is below the B−1 time warning thresholds but does not close unmeasured memory/install dimensions.

Therefore:

```text
lifecycle = AUDITED
disposition = <unset>
queue state = NOT QUEUED
selection = PREFERRED FIRST S1 CANDIDATE
```

## 5. Implementation gate

Do not open the TimesNet implementation PR while another critical implementation PR occupies the repository's WIP slot.

When the slot is free, the next model PR must be exactly one model and must reconstruct the implementation from current dev plus the fixed upstream source, not cherry-pick the old #266 port wholesale.

Required current-dev evidence before merge:

```text
E0 source/license
E1 PHMFactory constructor/input/output/config/public CLI
E2 fixed-source fidelity oracle for FFT-period selection and 2-D periodic convolution path
candidate-specific G7 measurement
current-head CI
fresh-context review
```

No S2 Task work, no second model, no aggregate native-model workflow and no broad registry redesign belong in that PR.

## 6. Non-selection is not rejection

ModernTCN, TSLANet and iTransformer remain AUDITED candidates. Their deferral only means they are not the lowest-risk, highest-information first S1 pilot. Future coverage gaps or paper requirements may reopen them through G1–G7.
