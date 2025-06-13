import argparse
import importlib
# 调用默认 pipeline
# from src.Pipeline_01_default import pipeline

def main():
    """
    Vbench 主入口，配置环境变量并调用实验流水线

    """
    # 加载环境变量配置
    parser = argparse.ArgumentParser(description="任务流水线")
    
    parser.add_argument('--config_path', 
                        type=str, 
                        # default= '/home/user/LQ/B_Signal/Signal_foundation_model/Vbench/configs/demo/Single_DG/CWRU.yaml', # CWRU.yaml
                        # default='/home/user/LQ/B_Signal/Signal_foundation_model/Vbench/configs/demo/Multiple_DG/CWRU_THU_using_ISFM.yaml',
                        # default='/home/user/LQ/B_Signal/Signal_foundation_model/Vbench/configs/demo/dummy_test.yaml',
                        # default='/home/user/LQ/B_Signal/Signal_foundation_model/Vbench/configs/demo/Multiple_DG/all.yaml',
                        # default='/home/lq/LQcode/2_project/PHMBench/PHM-Vibench/configs/demo/Multiple_DG/test.yaml',
                        # default='/home/lq/LQcode/2_project/PHMBench/PHM-Vibench/configs/demo/Pretraining/test.yaml',
                        # default='/home/lq/LQcode/2_project/PHMBench/PHM-Vibench/configs/demo/GFS/test.yaml',
                         default=[
                                 '/home/lq/LQcode/2_project/PHMBench/PHM-Vibench/configs/demo/Pretraining/test.yaml',
                                 '/home/lq/LQcode/2_project/PHMBench/PHM-Vibench/configs/demo/GFS/test.yaml'],
                        help='配置文件路径')
    parser.add_argument('--notes', 
                        type=str, 
                        default='',
                        help='实验备注')

    parser.add_argument('--pipeline', 
                        type=str, 
                        # default='test.Pipeline_gfs_sampler_test', # 修改：指向新的测试流水线
                        # default='src.Pipeline_01_default', # 修改：指向默认流水线
                        default='src.Pipeline_02_pretrain_fewshot', # Pretraining-fewshot pipeline
                        help='实验流水线模块路径 (例如 src.Pipeline_01_default 或 test.Pipeline_gfs_sampler_test)')
    
    args = parser.parse_args()
    # 修改：直接使用 args.pipeline 作为模块路径
    pipeline_module = importlib.import_module(args.pipeline) 
    # 执行DG流水线
    results = pipeline_module.pipeline(args) # 修改：调用导入模块的 pipeline 函数
    print(f"完成所有实验！")
    
    return results

if __name__ == "__main__":
    main()


