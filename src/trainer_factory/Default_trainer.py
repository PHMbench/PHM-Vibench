import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, ModelPruning, EarlyStopping
from torch.utils.tensorboard.writer import SummaryWriter
import os
import swanlab
from swanlab.integration.pytorch_lightning import SwanLabLogger

# 获取当前进程的排名
is_main_process = True  # 默认为主进程
if 'LOCAL_RANK' in os.environ:
    local_rank = int(os.environ['LOCAL_RANK'])
    is_main_process = local_rank == 0
    

def trainer(args_e,args_t, args_d, path):
    """
    设置训练器的配置，包括日志记录、回调函数和数据加载器等。
    
    参数:
    - args_t: 包含训练配置的对象
    - args_d: 包含数据配置的对象
    - path: 存储日志、检查点的路径
    
    返回:
    - trainer: 训练器对象
    """
    # 获取回调列表
    callback_list = call_backs(args_t, path)
    log_list = [CSVLogger(path, name="logs")]
    use_wandb = getattr(args_e, 'wandb', False)
    use_swanlab = getattr(args_e, 'swanlab', False)    
    # 根据 wandb_flag 确定日志记录器列表
    if use_wandb:
        # 配置 WandB 日志记录
        wandb_logger = WandbLogger(project=args_e.project,
                                   offline= not is_main_process,
                                  )
        log_list.append(wandb_logger)

    if use_swanlab:
        # swanlab
        swanlab_logger = SwanLabLogger(
                project = args_e.project,
                experiment_name=args_t.logger_name
            )
        log_list.append(swanlab_logger)
            
    # 设置设备类型：CPU 或自动选择
    accelerate_type = 'cpu' if args_t.device == 'cpu' else 'auto'

    # 如果不存在log_every_n_steps，使用默认值50 # TODO @liq22
    if not getattr(args_t, 'log_every_n_steps', None):
        args_t.log_every_n_steps = 50

    # 初始化训练器
    trainer = pl.Trainer(
        callbacks=callback_list,
        accelerator=accelerate_type,
        max_epochs=args_t.num_epochs,
        devices=args_t.gpus,
        logger=log_list,
        log_every_n_steps=args_t.log_every_n_steps,
        strategy= "ddp_find_unused_parameters_true" if args_t.gpus > 1 else 'auto',
    )
    return trainer

def call_backs(args, path):
    """
    配置训练时所需的回调函数，包括检查点保存、模型修剪、早期停止等。
    
    参数:
    - args: 包含训练配置的对象
    - path: 存储检查点的路径
    
    返回:
    - callback_list: 配置好的回调函数列表
    """
    # 检查点回调（保存最好的模型）
    checkpoint_callback = ModelCheckpoint(
        monitor=args.monitor,
        filename='model-{epoch:02d}-{val_loss:.4f}',
        save_top_k=8,
        mode='min',
        dirpath=path
    )
    
    callback_list = [checkpoint_callback]

    # 模型修剪回调（根据需求添加）
    if args.pruning:
        prune_callback = Prune_callback(args)
        callback_list.append(prune_callback)
    
    # 早期停止回调
    if args.early_stopping:
        early_stopping = create_early_stopping_callback(args)
        callback_list.append(early_stopping)
    
    return callback_list

def Prune_callback(args):
    """
    根据训练配置，返回模型修剪回调函数。
    
    参数:
    - args: 包含训练配置的对象
    
    返回:
    - prune_callback: 配置好的修剪回调（如果有）
    """
    def compute_amount(epoch):
        # 根据训练进度动态调整修剪比例
        if epoch == args.num_epochs // 4:
            return args.pruning[0]
        elif epoch == args.num_epochs // 2:
            return args.pruning[1]
        elif 3 * args.num_epochs // 4 < epoch:
            return args.pruning[2]
            
    if isinstance(args.pruning, (int, float)):
        prune_callback = ModelPruning("l1_unstructured", parameter_names=['weight'], amount=args.pruning)
    elif isinstance(args.pruning, list):
        prune_callback = ModelPruning("l1_unstructured", parameter_names=['weight'], amount=compute_amount)
    else:
        prune_callback = None
    return prune_callback

def create_early_stopping_callback(args):
    """
    创建并返回早期停止回调函数。
    
    参数:
    - args: 包含训练配置的对象
    
    返回:
    - early_stopping: 配置好的早期停止回调
    """
    # 配置早期停止回调，监控验证集的损失
    early_stopping = EarlyStopping(
        monitor=args.monitor, # TODO @liq22
        min_delta=0.00,
        patience=args.patience,  # 从args中读取patience值
        verbose=True,
        mode='min',
        check_finite=True,  # 防止无穷大或NaN值时停止训练
    )
    return early_stopping