CUDA_VISIBLE_DEVICES=5 python main_LQ.py \
    --config_path script/LQ1/Pretraining/Pretraining_C+P_patchtst.yaml \
    --fs_config_path script/LQ1/GFS/GFS_C+M.yaml \
    --pipeline Pipeline_01_default \
    --notes "EXP1_mask08" \
    # --pipeline 

CUDA_VISIBLE_DEVICES=5 python main_LQ.py \
    --config_path script/LQ1/Pretraining/Pretraining_C+P_patchtst.yaml \
    --fs_config_path script/LQ1/GFS/GFS_C+M.yaml \
    --pipeline Pipeline_01_default \
    --notes "EXP1_newtask" \