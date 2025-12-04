# 替换 data_dir

# CWRU 分类任务
python main.py --config configs/demo/Single_DG/CWRU.yaml

# Cross-dataset genealization
python main.py --config configs/demo/Multiple_DG/CWRU_THU_using_ISFM.yaml

# pretrain
python main.py --config configs/demo/Pretraining/Pretraining_demo.yaml

# general few-shot
python main.py --config configs/demo/Few_Shot/CWRU.yaml


# pretrain + few-shot
python main.py \
--pipeline Pipeline_02_pretrain_fewshot \
--fs_config_path configs/demo/GFS/GFS_demo.yaml \
--config_path configs/demo/Pretraining/Pretraining_demo.yaml

# === Unified Metric Learning Pipeline ===

# Quick 1-epoch validation test
python script/unified_metric/test_1epoch.py

# Full unified metric learning pipeline
python script/unified_metric/run_unified_experiments.py \
    --config script/unified_metric/configs/unified_experiments.yaml \
    --mode complete

# Individual stages
python script/unified_metric/run_unified_experiments.py --mode pretraining
python script/unified_metric/run_unified_experiments.py --mode zero_shot_eval
python script/unified_metric/run_unified_experiments.py --mode finetuning

# Results analysis and visualization
python script/unified_metric/collect_results.py --mode analyze
python script/unified_metric/paper_visualization.py --demo

# SOTA comparison
python script/unified_metric/sota_comparison.py --methods all