#!/usr/bin/env bash
set -euo pipefail

# Launch shard for CUDA device 0
# Run only on the local 2x4090 machine after this preflight passes.
# Queue validation can_execute at generation time: False
# Queue validation resource reason: blocked; no accepted GPU evidence can be generated in this session
# Launchable commands: 49

# Static queue validation failed at generation time.
# Regenerate this launch plan only after the queue and resource gates pass.
printf '%s\n' 'Blocked: static queue validation can_execute=False'
printf '%s\n' 'Resource reason: blocked; no accepted GPU evidence can be generated in this session'
printf '%s\n' 'Structural issues: 0'
exit 2

python -m scripts.uxfd_experiment_launch_gate --format markdown

nvidia-smi -L
python -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 2; names=[torch.cuda.get_device_name(i) for i in range(2)]; assert all('RTX 4090' in name for name in names), names; print(names[0]); print(names[1])"

# Q1 TII_operator_attention proposed P00 device=0 workdir=.: XOAN/DSOA operator attention
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/TII_operator_attention/configs/vibench/min.yaml --override trainer.num_epochs=1 --override data.num_workers=0

# Q1 TII_operator_attention baselines B02 device=0 workdir=.: ResNet baseline
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/TII_operator_attention/configs/vibench/min.yaml --override model.name=Resnet --override trainer.num_epochs=1 --override data.num_workers=0

# Q1 TII_operator_attention baselines B04 device=0 workdir=.: TFN baseline
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/TII_operator_attention/configs/vibench/min.yaml --override model.name=TFN --override trainer.num_epochs=1 --override data.num_workers=0

# Q1 TII_operator_attention baselines B06 device=0 workdir=.: ConvTransformer baseline
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/TII_operator_attention/configs/vibench/min.yaml --override model.type=Transformer --override model.name=ConvTransformer --override model.input_dim=2 --override trainer.num_epochs=1 --override data.num_workers=0

# Q1 TII_operator_attention ablations A01 device=0 workdir=.: remove operator attention
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/TII_operator_attention/configs/vibench/min.yaml --override model.name=NSN --override model.uxfd.operator_attention.enable=false --override trainer.num_epochs=1 --override data.num_workers=0

# Q1 TII_operator_attention ablations A03 device=0 workdir=.: Hilbert-only operator subset
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/TII_operator_attention/configs/vibench/min.yaml --override model.name=NSN --override model.uxfd.operator_attention.operators='["HT"]' --override trainer.num_epochs=1 --override data.num_workers=0

# Q1 TII_operator_attention ablations A05 device=0 workdir=.: low temperature sensitivity
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/TII_operator_attention/configs/vibench/min.yaml --override model.name=NSN --override model.uxfd.operator_attention.temperature=0.5 --override trainer.num_epochs=1 --override data.num_workers=0

# Q2 1D-2D_fusion_explainable proposed P00 device=0 workdir=.: PHM-Vibench NSN proxy with 1D-2D signal_processing_2d config
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/1D-2D_fusion_explainable/configs/vibench/min.yaml --override trainer.num_epochs=1 --override data.num_workers=0

# Q2 1D-2D_fusion_explainable baselines B02 device=0 workdir=.: ResNet baseline
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/1D-2D_fusion_explainable/configs/vibench/min.yaml --override model.name=Resnet --override trainer.num_epochs=1 --override data.num_workers=0

# Q2 1D-2D_fusion_explainable baselines B04 device=0 workdir=.: TFN baseline
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/1D-2D_fusion_explainable/configs/vibench/min.yaml --override model.name=TFN --override trainer.num_epochs=1 --override data.num_workers=0

# Q2 1D-2D_fusion_explainable baselines B06 device=0 workdir=.: ConvTransformer baseline
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/1D-2D_fusion_explainable/configs/vibench/min.yaml --override model.type=Transformer --override model.name=ConvTransformer --override model.input_dim=2 --override trainer.num_epochs=1 --override data.num_workers=0

# Q2 1D-2D_fusion_explainable ablations A02 device=0 workdir=.: smaller STFT window
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/1D-2D_fusion_explainable/configs/vibench/min.yaml --override model.signal_processing_2d.stft.n_fft=64 --override trainer.num_epochs=1 --override data.num_workers=0

# Q2 1D-2D_fusion_explainable ablations A04 device=0 workdir=.: concat fusion switch
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/1D-2D_fusion_explainable/configs/vibench/min.yaml --override model.signal_processing_2d.fusion.type=concat --override trainer.num_epochs=1 --override data.num_workers=0

# Q2 1D-2D_fusion_explainable ablations A06 device=0 workdir=.: FFT-only signal layer
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/1D-2D_fusion_explainable/configs/vibench/min.yaml --override model.signal_processing_configs.layer1='["FFT"]' --override trainer.num_epochs=1 --override data.num_workers=0

# Q3 Explainable_FD_Toolkit proposed P00 device=0 workdir=.: Toolkit NSN smoke with explain extension
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/Explainable_FD_Toolkit/configs/vibench/min.yaml --override trainer.num_epochs=1 --override data.num_workers=0

# Q3 Explainable_FD_Toolkit baselines B02 device=0 workdir=.: ResNet baseline
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/Explainable_FD_Toolkit/configs/vibench/min.yaml --override model.name=Resnet --override trainer.num_epochs=1 --override data.num_workers=0

# Q3 Explainable_FD_Toolkit baselines B04 device=0 workdir=.: TFN baseline
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/Explainable_FD_Toolkit/configs/vibench/min.yaml --override model.name=TFN --override trainer.num_epochs=1 --override data.num_workers=0

# Q3 Explainable_FD_Toolkit baselines B06 device=0 workdir=.: ConvTransformer baseline
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/Explainable_FD_Toolkit/configs/vibench/min.yaml --override model.type=Transformer --override model.name=ConvTransformer --override model.input_dim=2 --override trainer.num_epochs=1 --override data.num_workers=0

# Q3 Explainable_FD_Toolkit ablations A02 device=0 workdir=.: schema removal
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/Explainable_FD_Toolkit/configs/vibench/min.yaml --override trainer.extensions.explain.schema_validation=false --override trainer.num_epochs=1 --override data.num_workers=0

# Q3 Explainable_FD_Toolkit ablations A04 device=0 workdir=.: standardized manifest off
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/Explainable_FD_Toolkit/configs/vibench/min.yaml --override trainer.extensions.report.enable=false --override trainer.num_epochs=1 --override data.num_workers=0

# Q3 Explainable_FD_Toolkit ablations A06 device=0 workdir=.: post-hoc comparator only
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/Explainable_FD_Toolkit/configs/vibench/min.yaml --override trainer.extensions.explain.explainer=gradcam_xfd --override trainer.num_epochs=1 --override data.num_workers=0

# Q4 MOE_explainable baselines B01 device=0 workdir=.: NSN/TSPN_UXFD without operator-routing proxy
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/MOE_explainable/configs/vibench/min.yaml --override model.name=NSN --override model.uxfd.operator_attention.enable=false --override trainer.num_epochs=1 --override data.num_workers=0

# Q4 MOE_explainable baselines B03 device=0 workdir=.: SincNet baseline
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/MOE_explainable/configs/vibench/min.yaml --override model.name=Sincnet --override trainer.num_epochs=1 --override data.num_workers=0

# Q4 MOE_explainable baselines B05 device=0 workdir=.: WKN baseline
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/MOE_explainable/configs/vibench/min.yaml --override model.name=WKN --override trainer.num_epochs=1 --override data.num_workers=0

# Q4 MOE_explainable ablations A01 device=0 workdir=.: expert-count sweep 3/5/8
CUDA_VISIBLE_DEVICES=0 python paper/UXFD_paper/MOE_explainable/scripts/run_expert_ablation_probe.py --output-dir paper/UXFD_paper/MOE_explainable/results/t043/expert_ablation_probe --datasets CWRU --expert-counts 3 5 8 --epochs 1 --batch-size 16 --max-train-batches 4 --max-test-batches 4

# Q4 MOE_explainable ablations A03 device=0 workdir=.: remove sparsity regularization
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/MOE_explainable/configs/vibench/min.yaml --override model.uxfd.operator_attention.sparsity_weight=0.0 --override trainer.num_epochs=1 --override data.num_workers=0

# Q4 MOE_explainable ablations A05 device=0 workdir=.: expert family removal
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/MOE_explainable/configs/vibench/min.yaml --override model.uxfd.operator_attention.operators='["I","HT"]' --override trainer.num_epochs=1 --override data.num_workers=0

# Q5 Paper_fuzzy_XFD proposed P00 device=0 workdir=.: Fuzzy-XFD / NSN with fuzzy residual head
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/Paper_fuzzy_XFD/configs/vibench/min.yaml --override trainer.num_epochs=1 --override data.num_workers=0

# Q5 Paper_fuzzy_XFD baselines B02 device=0 workdir=.: ResNet baseline
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/Paper_fuzzy_XFD/configs/vibench/min.yaml --override model.name=Resnet --override trainer.num_epochs=1 --override data.num_workers=0

# Q5 Paper_fuzzy_XFD baselines B04 device=0 workdir=.: TFN baseline
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/Paper_fuzzy_XFD/configs/vibench/min.yaml --override model.name=TFN --override trainer.num_epochs=1 --override data.num_workers=0

# Q5 Paper_fuzzy_XFD baselines B06 device=0 workdir=.: ConvTransformer baseline
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/Paper_fuzzy_XFD/configs/vibench/min.yaml --override model.type=Transformer --override model.name=ConvTransformer --override model.input_dim=2 --override trainer.num_epochs=1 --override data.num_workers=0

# Q5 Paper_fuzzy_XFD ablations A01 device=0 workdir=.: remove fuzzy decision head
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/Paper_fuzzy_XFD/configs/vibench/min.yaml --override model.decision_configs.type=linear --override trainer.num_epochs=1 --override data.num_workers=0

# Q5 Paper_fuzzy_XFD ablations A03 device=0 workdir=.: weak fuzzy residual scale
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/Paper_fuzzy_XFD/configs/vibench/min.yaml --override model.decision_configs.fuzzy.logit_scale=0.1 --override trainer.num_epochs=1 --override data.num_workers=0

# Q5 Paper_fuzzy_XFD ablations A05 device=0 workdir=.: single membership function
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/Paper_fuzzy_XFD/configs/vibench/min.yaml --override model.decision_configs.fuzzy.num_membership_functions=1 --override trainer.num_epochs=1 --override data.num_workers=0

# Q6 Neuralsymbolic_theory proposed P00 device=0 workdir=.: Constrained NSN/TSPN_UXFD logic-slot neural-symbolic model
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/Neuralsymbolic_theory/configs/vibench/min.yaml --override trainer.num_epochs=1 --override data.num_workers=0

# Q6 Neuralsymbolic_theory baselines B02 device=0 workdir=.: ResNet baseline
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/Neuralsymbolic_theory/configs/vibench/min.yaml --override model.name=Resnet --override trainer.num_epochs=1 --override data.num_workers=0

# Q6 Neuralsymbolic_theory baselines B04 device=0 workdir=.: TFN baseline
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/Neuralsymbolic_theory/configs/vibench/min.yaml --override model.name=TFN --override trainer.num_epochs=1 --override data.num_workers=0

# Q6 Neuralsymbolic_theory baselines B06 device=0 workdir=.: ConvTransformer baseline
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/Neuralsymbolic_theory/configs/vibench/min.yaml --override model.type=Transformer --override model.name=ConvTransformer --override model.input_dim=2 --override trainer.num_epochs=1 --override data.num_workers=0

# Q6 Neuralsymbolic_theory ablations A02 device=0 workdir=paper/UXFD_paper/Neuralsymbolic_theory: physical-informed robustness hook
(cd paper/UXFD_paper/Neuralsymbolic_theory && CUDA_VISIBLE_DEVICES=0 python experiments/proposition2_simple.py)

# Q6 Neuralsymbolic_theory ablations A04 device=0 workdir=.: high symbolic residual strength
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/Neuralsymbolic_theory/configs/vibench/min.yaml --override model.decision_configs.logic.logit_scale=1.0 --override trainer.num_epochs=1 --override data.num_workers=0

# Q6 Neuralsymbolic_theory ablations A06 device=0 workdir=paper/UXFD_paper/Neuralsymbolic_theory: cross-method mapping validation
(cd paper/UXFD_paper/Neuralsymbolic_theory && CUDA_VISIBLE_DEVICES=0 python code/validate_mapping.py && python scripts/build_source_backed_mapping.py)

# Q7 LLM_Explainable_FD_Toolkit proposed P00 device=0 workdir=.: PHM-Vibench NSN smoke with agent/distillation extension enabled
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/LLM_Explainable_FD_Toolkit/configs/vibench/min.yaml --override trainer.num_epochs=1 --override data.num_workers=0

# Q7 LLM_Explainable_FD_Toolkit baselines B02 device=0 workdir=.: standalone template LLM baseline
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/LLM_Explainable_FD_Toolkit/configs/vibench/min.yaml --override trainer.extensions.agent.enable=false --override trainer.extensions.agent.mode=standalone --override trainer.num_epochs=1 --override data.num_workers=0

# Q7 LLM_Explainable_FD_Toolkit baselines B04 device=0 workdir=.: SincNet diagnostic baseline
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/LLM_Explainable_FD_Toolkit/configs/vibench/min.yaml --override model.name=Sincnet --override trainer.num_epochs=1 --override data.num_workers=0

# Q7 LLM_Explainable_FD_Toolkit baselines B06 device=0 workdir=.: WKN diagnostic baseline
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/LLM_Explainable_FD_Toolkit/configs/vibench/min.yaml --override model.name=WKN --override trainer.num_epochs=1 --override data.num_workers=0

# Q7 LLM_Explainable_FD_Toolkit ablations A01 device=0 workdir=.: disable PHM-Vibench agent/distillation extension
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/LLM_Explainable_FD_Toolkit/configs/vibench/min.yaml --override trainer.extensions.agent.enable=false --override trainer.num_epochs=1 --override data.num_workers=0

# Q7 LLM_Explainable_FD_Toolkit ablations A03 device=0 workdir=.: package-based template pipeline
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/LLM_Explainable_FD_Toolkit/configs/vibench/min.yaml --override trainer.extensions.agent.mode=pipeline --override trainer.extensions.agent.save_outputs=true --override trainer.num_epochs=1 --override data.num_workers=0

# Q7 LLM_Explainable_FD_Toolkit ablations A05 device=0 workdir=.: remove hallucination checker
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/LLM_Explainable_FD_Toolkit/configs/vibench/min.yaml --override trainer.extensions.agent.hallucination_checker.enable=false --override trainer.num_epochs=1 --override data.num_workers=0

# Q7 LLM_Explainable_FD_Toolkit ablations A07 device=0 workdir=.: short/medium/long template latency sweep
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/LLM_Explainable_FD_Toolkit/configs/vibench/min.yaml --override trainer.extensions.agent.length_sweep='["short","medium","long"]' --override trainer.num_epochs=1 --override data.num_workers=0
