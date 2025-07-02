CUDA_VISIBLE_DEVICES=5 python main_LQ.py \
    --config_path script/LQ1/Pretraining/Pretraining_C+P_patchtst.yaml \
    --fs_config_path script/LQ1/GFS/GFS_C+M.yaml \
    --pipeline Pipeline_01_default \
    --notes "EXP1_mask08" \
    # --pipeline 

CUDA_VISIBLE_DEVICES=0 python main_LQ.py \
    --config_path script/LQ1/Pretraining/Pretraining_C+P_patchtst.yaml \
    --fs_config_path script/LQ1/GFS/GFS_C+M.yaml \
    --pipeline Pipeline_01_default \
    --notes "EXP1_mask0.1+predict+reshape_norm" \
export WANDB_BASE_URL=HTTP://api.bandw.top
CUDA_VISIBLE_DEVICES=4,5,6,7 python main_LQ.py \
--config_path script/LQ1/Pretraining/Pretrain_MS.yaml \
--fs_config_path script/LQ1/GFS/GFS_C+M.yaml \
--pipeline Pipeline_01_default \
--notes "EXP1_mask0.1+predict+reshape_norm" \