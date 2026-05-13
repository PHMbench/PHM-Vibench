#!/usr/bin/env bash
set -euo pipefail

# Launch shard for CUDA device 1
# Run only on the local 2x4090 machine after this preflight passes.
nvidia-smi -L
python -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 2; names=[torch.cuda.get_device_name(i) for i in range(2)]; assert all('4090' in name for name in names), names; print(names[0]); print(names[1])"

# Queue validation can_execute at generation time: False
# Queue validation resource reason: blocked; no accepted GPU evidence can be generated in this session
# Launchable commands: 48

# Q1 TII_operator_attention baselines B01 device=1 workdir=.: NSN/TSPN_UXFD without operator attention
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/TII_operator_attention/configs/vibench/min.yaml --override model.name=NSN --override model.uxfd.operator_attention.enable=false --override trainer.num_epochs=1 --override data.num_workers=0

# Q1 TII_operator_attention baselines B03 device=1 workdir=.: SincNet baseline
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/TII_operator_attention/configs/vibench/min.yaml --override model.name=Sincnet --override trainer.num_epochs=1 --override data.num_workers=0

# Q1 TII_operator_attention baselines B05 device=1 workdir=.: WKN baseline
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/TII_operator_attention/configs/vibench/min.yaml --override model.name=WKN --override trainer.num_epochs=1 --override data.num_workers=0

# Q1 TII_operator_attention baselines B07 device=1 workdir=.: Feature/self-attention CNN baseline
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/TII_operator_attention/configs/vibench/min.yaml --override model.type=CNN --override model.name=AttentionCNN --override model.input_dim=2 --override trainer.num_epochs=1 --override data.num_workers=0

# Q1 TII_operator_attention ablations A02 device=1 workdir=.: identity-only operator subset
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/TII_operator_attention/configs/vibench/min.yaml --override model.name=NSN --override model.uxfd.operator_attention.operators='["I"]' --override trainer.num_epochs=1 --override data.num_workers=0

# Q1 TII_operator_attention ablations A04 device=1 workdir=.: FFT-only operator subset
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/TII_operator_attention/configs/vibench/min.yaml --override model.name=NSN --override model.uxfd.operator_attention.operators='["FFT"]' --override trainer.num_epochs=1 --override data.num_workers=0

# Q1 TII_operator_attention ablations A06 device=1 workdir=.: high temperature sensitivity
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/TII_operator_attention/configs/vibench/min.yaml --override model.name=NSN --override model.uxfd.operator_attention.temperature=2.0 --override trainer.num_epochs=1 --override data.num_workers=0

# Q2 1D-2D_fusion_explainable baselines B01 device=1 workdir=.: NSN/TSPN_UXFD with 2D signal path disabled
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/1D-2D_fusion_explainable/configs/vibench/min.yaml --override model.signal_processing_2d.enable=false --override trainer.num_epochs=1 --override data.num_workers=0

# Q2 1D-2D_fusion_explainable baselines B03 device=1 workdir=.: SincNet baseline
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/1D-2D_fusion_explainable/configs/vibench/min.yaml --override model.name=Sincnet --override trainer.num_epochs=1 --override data.num_workers=0

# Q2 1D-2D_fusion_explainable baselines B05 device=1 workdir=.: WKN baseline
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/1D-2D_fusion_explainable/configs/vibench/min.yaml --override model.name=WKN --override trainer.num_epochs=1 --override data.num_workers=0

# Q2 1D-2D_fusion_explainable ablations A01 device=1 workdir=.: disable 2D signal-processing path
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/1D-2D_fusion_explainable/configs/vibench/min.yaml --override model.signal_processing_2d.enable=false --override trainer.num_epochs=1 --override data.num_workers=0

# Q2 1D-2D_fusion_explainable ablations A03 device=1 workdir=.: smaller STFT hop length
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/1D-2D_fusion_explainable/configs/vibench/min.yaml --override model.signal_processing_2d.stft.hop_length=32 --override trainer.num_epochs=1 --override data.num_workers=0

# Q2 1D-2D_fusion_explainable ablations A05 device=1 workdir=paper/UXFD_paper/1D-2D_fusion_explainable: paper-local demo class-count sanity
(cd paper/UXFD_paper/1D-2D_fusion_explainable && CUDA_VISIBLE_DEVICES=1 python scripts/run_minimal_demo.py --use_dummy --num_epochs=1 --batch_size=8 --input_dim=128 --num_classes=10 --output_root /tmp/uxfd_paper02_minimal_demo)

# Q2 1D-2D_fusion_explainable ablations A07 device=1 workdir=.: legacy 1D-only / 2D-only / no-statistical configs
CUDA_VISIBLE_DEVICES=1 python paper/UXFD_paper/1D-2D_fusion_explainable/scripts/run_fusion_ablation_smoke.py --condition legacy_ablation_surface --output /tmp/uxfd_paper02_fusion_ablation_smoke --seed 0

# Q3 Explainable_FD_Toolkit baselines B01 device=1 workdir=.: NSN/TSPN_UXFD baseline
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/Explainable_FD_Toolkit/configs/vibench/min.yaml --override model.name=NSN --override trainer.num_epochs=1 --override data.num_workers=0

# Q3 Explainable_FD_Toolkit baselines B03 device=1 workdir=.: SincNet baseline
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/Explainable_FD_Toolkit/configs/vibench/min.yaml --override model.name=Sincnet --override trainer.num_epochs=1 --override data.num_workers=0

# Q3 Explainable_FD_Toolkit baselines B05 device=1 workdir=.: WKN baseline
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/Explainable_FD_Toolkit/configs/vibench/min.yaml --override model.name=WKN --override trainer.num_epochs=1 --override data.num_workers=0

# Q3 Explainable_FD_Toolkit ablations A01 device=1 workdir=.: disable PHM-Vibench explain extension
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/Explainable_FD_Toolkit/configs/vibench/min.yaml --override trainer.extensions.explain.enable=false --override trainer.num_epochs=1 --override data.num_workers=0

# Q3 Explainable_FD_Toolkit ablations A03 device=1 workdir=paper/UXFD_paper/Explainable_FD_Toolkit: faithfulness/stability metric-family removal
(cd paper/UXFD_paper/Explainable_FD_Toolkit && CUDA_VISIBLE_DEVICES=1 python scripts/run_toolkit_ablations.py --condition metrics_subset_off --output /tmp/uxfd_paper01_toolkit_ablation_smoke --seed 0)

# Q3 Explainable_FD_Toolkit ablations A05 device=1 workdir=paper/UXFD_paper/Explainable_FD_Toolkit: fixed seed/config snapshot off
(cd paper/UXFD_paper/Explainable_FD_Toolkit && CUDA_VISIBLE_DEVICES=1 python scripts/run_toolkit_ablations.py --condition snapshot_off --output /tmp/uxfd_paper01_toolkit_ablation_smoke --seed 0)

# Q4 MOE_explainable proposed P00 device=1 workdir=.: PHM-Vibench UXFD route/MoE proxy smoke
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/MOE_explainable/configs/vibench/min.yaml --override trainer.num_epochs=1 --override data.num_workers=0

# Q4 MOE_explainable baselines B02 device=1 workdir=.: ResNet baseline
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/MOE_explainable/configs/vibench/min.yaml --override model.name=Resnet --override trainer.num_epochs=1 --override data.num_workers=0

# Q4 MOE_explainable baselines B04 device=1 workdir=.: TFN baseline
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/MOE_explainable/configs/vibench/min.yaml --override model.name=TFN --override trainer.num_epochs=1 --override data.num_workers=0

# Q4 MOE_explainable baselines B06 device=1 workdir=.: ConvTransformer baseline
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/MOE_explainable/configs/vibench/min.yaml --override model.type=Transformer --override model.name=ConvTransformer --override model.input_dim=2 --override trainer.num_epochs=1 --override data.num_workers=0

# Q4 MOE_explainable ablations A02 device=1 workdir=.: remove load-balance regularization
CUDA_VISIBLE_DEVICES=1 python paper/UXFD_paper/MOE_explainable/scripts/run_moe_ablation_smoke.py --condition no_load_balance --output /tmp/uxfd_paper04_moe_ablation_smoke --seed 0

# Q4 MOE_explainable ablations A04 device=1 workdir=.: router temperature sweep
CUDA_VISIBLE_DEVICES=1 python paper/UXFD_paper/MOE_explainable/scripts/run_moe_ablation_smoke.py --condition temperature_sweep --output /tmp/uxfd_paper04_moe_ablation_smoke --seed 0

# Q4 MOE_explainable ablations A06 device=1 workdir=.: uniform/equal-weight router
CUDA_VISIBLE_DEVICES=1 python paper/UXFD_paper/MOE_explainable/scripts/run_moe_ablation_smoke.py --condition uniform_router --output /tmp/uxfd_paper04_moe_ablation_smoke --seed 0

# Q5 Paper_fuzzy_XFD baselines B01 device=1 workdir=.: NSN/TSPN_UXFD without fuzzy rules
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/Paper_fuzzy_XFD/configs/vibench/min.yaml --override model.decision_configs.type=linear --override trainer.num_epochs=1 --override data.num_workers=0

# Q5 Paper_fuzzy_XFD baselines B03 device=1 workdir=.: SincNet baseline
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/Paper_fuzzy_XFD/configs/vibench/min.yaml --override model.name=Sincnet --override trainer.num_epochs=1 --override data.num_workers=0

# Q5 Paper_fuzzy_XFD baselines B05 device=1 workdir=.: WKN baseline
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/Paper_fuzzy_XFD/configs/vibench/min.yaml --override model.name=WKN --override trainer.num_epochs=1 --override data.num_workers=0

# Q5 Paper_fuzzy_XFD baselines B07 device=1 workdir=.: Classical fuzzy/rule baseline
CUDA_VISIBLE_DEVICES=1 python paper/UXFD_paper/Paper_fuzzy_XFD/scripts/run_fuzzy_baseline.py --features paper/UXFD_paper/Paper_fuzzy_XFD/results/evidence/t044/fuzzy_features_demo_missing.npz --output paper/UXFD_paper/Paper_fuzzy_XFD/results/evidence/t044/classical_fuzzy_dummy.json --max_samples 20

# Q5 Paper_fuzzy_XFD ablations A02 device=1 workdir=.: uncalibrated fuzzy residual scale
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/Paper_fuzzy_XFD/configs/vibench/min.yaml --override model.decision_configs.fuzzy.logit_scale=1.0 --override trainer.num_epochs=1 --override data.num_workers=0

# Q5 Paper_fuzzy_XFD ablations A04 device=1 workdir=.: low rule-count fuzzy head
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/Paper_fuzzy_XFD/configs/vibench/min.yaml --override model.decision_configs.fuzzy.num_rules=2 --override trainer.num_epochs=1 --override data.num_workers=0

# Q5 Paper_fuzzy_XFD ablations A06 device=1 workdir=.: narrow fuzzy feature bottleneck
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/Paper_fuzzy_XFD/configs/vibench/min.yaml --override model.decision_configs.fuzzy.num_fuzzy_features=8 --override trainer.num_epochs=1 --override data.num_workers=0

# Q6 Neuralsymbolic_theory baselines B01 device=1 workdir=.: NSN/TSPN_UXFD without symbolic decision constraints
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/Neuralsymbolic_theory/configs/vibench/min.yaml --override model.decision_configs.type=linear --override trainer.num_epochs=1 --override data.num_workers=0

# Q6 Neuralsymbolic_theory baselines B03 device=1 workdir=.: SincNet baseline
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/Neuralsymbolic_theory/configs/vibench/min.yaml --override model.name=Sincnet --override trainer.num_epochs=1 --override data.num_workers=0

# Q6 Neuralsymbolic_theory baselines B05 device=1 workdir=.: WKN baseline
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/Neuralsymbolic_theory/configs/vibench/min.yaml --override model.name=WKN --override trainer.num_epochs=1 --override data.num_workers=0

# Q6 Neuralsymbolic_theory ablations A01 device=1 workdir=.: remove symbolic constraints
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/Neuralsymbolic_theory/configs/vibench/min.yaml --override model.decision_configs.type=linear --override trainer.num_epochs=1 --override data.num_workers=0

# Q6 Neuralsymbolic_theory ablations A03 device=1 workdir=.: low symbolic residual strength
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/Neuralsymbolic_theory/configs/vibench/min.yaml --override model.decision_configs.logic.logit_scale=0.1 --override trainer.num_epochs=1 --override data.num_workers=0

# Q6 Neuralsymbolic_theory ablations A05 device=1 workdir=paper/UXFD_paper/Neuralsymbolic_theory: independent proposition validation
(cd paper/UXFD_paper/Neuralsymbolic_theory && CUDA_VISIBLE_DEVICES=1 python simple_validation_demo.py)

# Q6 Neuralsymbolic_theory ablations A07 device=1 workdir=.: remove cross-method mapping module from training/evaluation
CUDA_VISIBLE_DEVICES=1 python paper/UXFD_paper/Neuralsymbolic_theory/scripts/run_mapping_ablation_smoke.py --condition no_mapping --output /tmp/uxfd_paper06_mapping_ablation_smoke --seed 0

# Q7 LLM_Explainable_FD_Toolkit baselines B01 device=1 workdir=.: PHM-Vibench structured output without agent extension
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/LLM_Explainable_FD_Toolkit/configs/vibench/min.yaml --override trainer.extensions.agent.enable=false --override trainer.num_epochs=1 --override data.num_workers=0

# Q7 LLM_Explainable_FD_Toolkit baselines B03 device=1 workdir=.: ResNet diagnostic baseline
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/LLM_Explainable_FD_Toolkit/configs/vibench/min.yaml --override model.name=Resnet --override trainer.num_epochs=1 --override data.num_workers=0

# Q7 LLM_Explainable_FD_Toolkit baselines B05 device=1 workdir=.: TFN diagnostic baseline
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/LLM_Explainable_FD_Toolkit/configs/vibench/min.yaml --override model.name=TFN --override trainer.num_epochs=1 --override data.num_workers=0

# Q7 LLM_Explainable_FD_Toolkit baselines B07 device=1 workdir=.: ConvTransformer diagnostic baseline
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/LLM_Explainable_FD_Toolkit/configs/vibench/min.yaml --override model.type=Transformer --override model.name=ConvTransformer --override model.input_dim=2 --override trainer.num_epochs=1 --override data.num_workers=0

# Q7 LLM_Explainable_FD_Toolkit ablations A02 device=1 workdir=paper/UXFD_paper/LLM_Explainable_FD_Toolkit: single-case dialogue instead of pipeline demo
(cd paper/UXFD_paper/LLM_Explainable_FD_Toolkit && CUDA_VISIBLE_DEVICES=1 python experiments/scripts/run_minimal_llm_demo_standalone.py --mode single --case 0)

# Q7 LLM_Explainable_FD_Toolkit ablations A04 device=1 workdir=paper/UXFD_paper/LLM_Explainable_FD_Toolkit: core toolkit unit-test gate
(cd paper/UXFD_paper/LLM_Explainable_FD_Toolkit && CUDA_VISIBLE_DEVICES=1 python -m pytest -q code/tests/test_basic_functionality.py)

# Q7 LLM_Explainable_FD_Toolkit ablations A06 device=1 workdir=paper/UXFD_paper/LLM_Explainable_FD_Toolkit: remove retrieval/domain knowledge context
(cd paper/UXFD_paper/LLM_Explainable_FD_Toolkit && CUDA_VISIBLE_DEVICES=1 python experiments/scripts/run_llm_evidence_smoke.py --condition no_domain_context --output /tmp/uxfd_paper03_llm_evidence_smoke --seed 0)
