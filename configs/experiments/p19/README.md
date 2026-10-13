# P19 native experiment configurations

The sole local execution guide is [P19 Local Execution SOP](../../../docs/P19_TSPN_SIGNAL_STUDY.md). These configurations use `phmfactory research` and the maintained native data/model/task/trainer factories; they do not launch the historical Agent Benchmark runtime.

| Configuration | Scope | Training budget |
|---|---|---|
| [op1.yaml](op1.yaml) | Original 32-bearing OP1 with TSPN, unchanged 0.80 tune-Accuracy stop gate | Fixed optimizer, 100 epochs, seeds 11/23/37 |
| [paderborn_dg_0.yaml](paderborn_dg_0.yaml) | Source conditions 1/2/3; target 0 | Four source-only trials at seed 11, selected repeats 23/37, per model |
| [paderborn_dg_1.yaml](paderborn_dg_1.yaml) | Source conditions 0/2/3; target 1 | Same |
| [paderborn_dg_2.yaml](paderborn_dg_2.yaml) | Source conditions 0/1/3; target 2 | Same |
| [paderborn_dg_3.yaml](paderborn_dg_3.yaml) | Source conditions 0/1/2; target 3 | Same |

The DG configs inherit OP1's fixed signal/model/training settings and add ResNet1D and TCN to TSPN. Source and target physical bearings are disjoint as well as conditions. The minimum-val-loss checkpoint and source Macro-F1/Accuracy/Brier trial rule are common; no target score selects a configuration. The 0.80 source diagnostic does not exclude DG folds or seeds. Validation/calibration/evaluation batch size is 1.

Baseline architectures are fixed before the shared optimizer search. ResNet1D uses basic blocks `[2,2,2,2]` and `initial_channels: 64`, matching the [native model defaults](../../../src/model_factory/CNN/ResNet1D.py); the source's width-32 example is explicitly for testing. TCN uses channels `[32,64,64,64]` and kernel size 7 from the existing [P04 comparator configuration](../p04/baseline_tcn_indomain.yaml), with the [native TCN](../../../src/model_factory/CNN/TCN.py) default dropout 0.2 made explicit. Only these architecture settings are reused from P04; its data, labels, split and training protocol are not adopted. This provenance does not qualify either baseline under P19 DG or establish competitive performance. OP1, the proposed method and the shared DG search/epoch/seed budget remain unchanged.

`study.role_map: REQUIRED_PHYSICAL_SPLIT_JSON` is intentionally unresolved. Supply the real approved physical assignment through the same explicit local-config/override arguments in every stage. Historical `assignments`, `grow_units` and `select_units` identify the fixed five roles. `sanity` validates real metadata and creates separate source/evaluation populations without changing raw metadata. Do not replace physical identities with record IDs or create example IDs to make a config pass.

The supplied Paderborn profile is 64 kHz, one selected vibration channel, 4,096 samples and 42 signal coordinates. A lower-rate or multichannel dataset needs its own source-only, scientifically qualified profile. The second PHM dataset and HAR are not represented by placeholder runnable configs.

`cuda:0` is a logical device: the SOP first queries physical GPU 0's UUID and sets `CUDA_VISIBLE_DEVICES` to that exact UUID before Python imports. Numeric visibility `0`, GPU 2 and multi-GPU execution are prohibited. CPU diagnostics use an empty visibility value.

Implementation and synthetic CPU checks are software evidence. Real-data/GPU execution, source baseline qualification and scientific gains remain unmeasured for this implementation. Four Paderborn folds are still one PHM dataset; this directory does not establish the required multi-PHM result or competitive/SOTA superiority.
