#!/usr/bin/env bash
set -euo pipefail

# Launch shard for CUDA device 1
# Run only after the experiment launch gate passes without --allow-not-ready.
# Queue validation can_execute at generation time: False
# Queue validation resource reason: blocked; no accepted GPU evidence can be generated in this session
# Launchable commands: 48

# Static queue validation failed at generation time.
# Regenerate this launch plan only after queue, owner-review, and resource gates pass.
printf '%s\n' 'Blocked: static queue validation can_execute=False'
printf '%s\n' 'Resource reason: blocked; no accepted GPU evidence can be generated in this session'
printf '%s\n' 'Structural issues: 0'
printf '%s\n' 'Experiment launch gate: python -m scripts.uxfd_experiment_launch_gate --format markdown'
printf '%s\n' 'Do not launch queue scripts until the experiment launch gate passes without --allow-not-ready.'
exit 2

python -m scripts.uxfd_experiment_launch_gate --format markdown

nvidia-smi -L
python -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 2; names=[torch.cuda.get_device_name(i) for i in range(2)]; assert all('RTX 4090' in name for name in names), names; print(names[0]); print(names[1])"

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

# Q2 1D-2D_fusion_explainable ablations A05 device=1 workdir=.: paper-local demo class-count sanity
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/1D-2D_fusion_explainable/configs/vibench/min.yaml --override model.out_channels=10 --override trainer.num_epochs=1 --override data.num_workers=0

# Q2 1D-2D_fusion_explainable ablations A07 device=1 workdir=.: legacy 1D-only / 2D-only / no-statistical configs
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/1D-2D_fusion_explainable/configs/vibench/min.yaml --override model.feature_extractor_configs='["Mean"]' --override trainer.num_epochs=1 --override data.num_workers=0

# Q3 Explainable_FD_Toolkit baselines B01 device=1 workdir=.: NSN/TSPN_UXFD baseline
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/Explainable_FD_Toolkit/configs/vibench/min.yaml --override model.name=NSN --override trainer.num_epochs=1 --override data.num_workers=0

# Q3 Explainable_FD_Toolkit baselines B03 device=1 workdir=.: SincNet baseline
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/Explainable_FD_Toolkit/configs/vibench/min.yaml --override model.name=Sincnet --override trainer.num_epochs=1 --override data.num_workers=0

# Q3 Explainable_FD_Toolkit baselines B05 device=1 workdir=.: WKN baseline
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/Explainable_FD_Toolkit/configs/vibench/min.yaml --override model.name=WKN --override trainer.num_epochs=1 --override data.num_workers=0

# Q3 Explainable_FD_Toolkit ablations A01 device=1 workdir=.: disable PHM-Vibench explain extension
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/Explainable_FD_Toolkit/configs/vibench/min.yaml --override trainer.extensions.explain.enable=false --override trainer.num_epochs=1 --override data.num_workers=0

# Q3 Explainable_FD_Toolkit ablations A03 device=1 workdir=.: faithfulness/stability metric-family removal
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/Explainable_FD_Toolkit/configs/vibench/min.yaml --override trainer.extensions.explain.metric_families='["latency"]' --override trainer.num_epochs=1 --override data.num_workers=0

# Q3 Explainable_FD_Toolkit ablations A05 device=1 workdir=.: fixed seed/config snapshot off
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/Explainable_FD_Toolkit/configs/vibench/min.yaml --override trainer.extensions.predictions.save_config_snapshot=false --override trainer.num_epochs=1 --override data.num_workers=0

# Q4 MOE_explainable proposed P00 device=1 workdir=.: PHM-Vibench UXFD route/MoE proxy smoke
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/MOE_explainable/configs/vibench/min.yaml --override trainer.num_epochs=1 --override data.num_workers=0

# Q4 MOE_explainable baselines B02 device=1 workdir=.: ResNet baseline
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/MOE_explainable/configs/vibench/min.yaml --override model.name=Resnet --override trainer.num_epochs=1 --override data.num_workers=0

# Q4 MOE_explainable baselines B04 device=1 workdir=.: TFN baseline
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/MOE_explainable/configs/vibench/min.yaml --override model.name=TFN --override trainer.num_epochs=1 --override data.num_workers=0

# Q4 MOE_explainable baselines B06 device=1 workdir=.: ConvTransformer baseline
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/MOE_explainable/configs/vibench/min.yaml --override model.type=Transformer --override model.name=ConvTransformer --override model.input_dim=2 --override trainer.num_epochs=1 --override data.num_workers=0

# Q4 MOE_explainable ablations A02 device=1 workdir=.: remove load-balance regularization
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/MOE_explainable/configs/vibench/min.yaml --override model.uxfd.operator_attention.load_balance_weight=0.0 --override trainer.num_epochs=1 --override data.num_workers=0

# Q4 MOE_explainable ablations A04 device=1 workdir=.: router temperature sweep
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/MOE_explainable/configs/vibench/min.yaml --override model.uxfd.operator_attention.temperature=0.5 --override trainer.num_epochs=1 --override data.num_workers=0

# Q4 MOE_explainable ablations A06 device=1 workdir=.: uniform/equal-weight router
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/MOE_explainable/configs/vibench/min.yaml --override model.uxfd.operator_attention.enable=false --override trainer.num_epochs=1 --override data.num_workers=0

# Q5 Paper_fuzzy_XFD baselines B01 device=1 workdir=.: NSN/TSPN_UXFD without fuzzy rules
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/Paper_fuzzy_XFD/configs/vibench/min.yaml --override model.decision_configs.type=linear --override trainer.num_epochs=1 --override data.num_workers=0

# Q5 Paper_fuzzy_XFD baselines B03 device=1 workdir=.: SincNet baseline
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/Paper_fuzzy_XFD/configs/vibench/min.yaml --override model.name=Sincnet --override trainer.num_epochs=1 --override data.num_workers=0

# Q5 Paper_fuzzy_XFD baselines B05 device=1 workdir=.: WKN baseline
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/Paper_fuzzy_XFD/configs/vibench/min.yaml --override model.name=WKN --override trainer.num_epochs=1 --override data.num_workers=0

# Q5 Paper_fuzzy_XFD baselines B07 device=1 workdir=.: Classical fuzzy/rule baseline
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/Paper_fuzzy_XFD/configs/vibench/min.yaml --override model.decision_configs.fuzzy.logit_scale=1.0 --override trainer.num_epochs=1 --override data.num_workers=0

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

# Q6 Neuralsymbolic_theory ablations A05 device=1 workdir=.: independent proposition validation
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/Neuralsymbolic_theory/configs/vibench/min.yaml --override model.decision_configs.logic.logit_scale=0.5 --override trainer.num_epochs=1 --override data.num_workers=0

# Q6 Neuralsymbolic_theory ablations A07 device=1 workdir=.: remove cross-method mapping module from training/evaluation
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/Neuralsymbolic_theory/configs/vibench/min.yaml --override model.decision_configs.type=linear --override trainer.num_epochs=1 --override data.num_workers=0

# Q7 LLM_Explainable_FD_Toolkit baselines B01 device=1 workdir=.: PHM-Vibench structured output without agent extension
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/LLM_Explainable_FD_Toolkit/configs/vibench/min.yaml --override trainer.extensions.agent.enable=false --override trainer.num_epochs=1 --override data.num_workers=0

# Q7 LLM_Explainable_FD_Toolkit baselines B03 device=1 workdir=.: ResNet diagnostic baseline
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/LLM_Explainable_FD_Toolkit/configs/vibench/min.yaml --override model.name=Resnet --override trainer.num_epochs=1 --override data.num_workers=0

# Q7 LLM_Explainable_FD_Toolkit baselines B05 device=1 workdir=.: TFN diagnostic baseline
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/LLM_Explainable_FD_Toolkit/configs/vibench/min.yaml --override model.name=TFN --override trainer.num_epochs=1 --override data.num_workers=0

# Q7 LLM_Explainable_FD_Toolkit baselines B07 device=1 workdir=.: ConvTransformer diagnostic baseline
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/LLM_Explainable_FD_Toolkit/configs/vibench/min.yaml --override model.type=Transformer --override model.name=ConvTransformer --override model.input_dim=2 --override trainer.num_epochs=1 --override data.num_workers=0

# Q7 LLM_Explainable_FD_Toolkit ablations A02 device=1 workdir=.: single-case dialogue instead of pipeline demo
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/LLM_Explainable_FD_Toolkit/configs/vibench/min.yaml --override trainer.extensions.agent.mode=single_case --override trainer.num_epochs=1 --override data.num_workers=0

# Q7 LLM_Explainable_FD_Toolkit ablations A04 device=1 workdir=paper/UXFD_paper/LLM_Explainable_FD_Toolkit: core toolkit unit-test gate
(cd paper/UXFD_paper/LLM_Explainable_FD_Toolkit && CUDA_VISIBLE_DEVICES=1 python -m pytest -q code/tests/test_basic_functionality.py)

# Q7 LLM_Explainable_FD_Toolkit ablations A06 device=1 workdir=.: remove retrieval/domain knowledge context
CUDA_VISIBLE_DEVICES=1 python main.py --config paper/UXFD_paper/LLM_Explainable_FD_Toolkit/configs/vibench/min.yaml --override trainer.extensions.agent.domain_context.enable=false --override trainer.num_epochs=1 --override data.num_workers=0
