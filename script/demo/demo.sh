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
python main.py --pipeline Pipeline_02_pretrain_fewshot --config_path [configs/demo/Pretraining/Pretraining_demo.yaml,configs/demo/Few_Shot/pretrain_few_shot.yaml]