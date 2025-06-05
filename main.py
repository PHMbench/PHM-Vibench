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
                        default= '/home/user/LQ/B_Signal/Signal_foundation_model/Vbench/configs/demo/Single_DG/CWRU.yaml', # CWRU.yaml
                        # default='/home/user/LQ/B_Signal/Signal_foundation_model/Vbench/configs/demo/Multiple_DG/CWRU_THU_using_ISFM.yaml',
                        # default='/home/user/LQ/B_Signal/Signal_foundation_model/Vbench/configs/demo/dummy_test.yaml',
                        help='配置文件路径')
    parser.add_argument('--notes',
                        type=str,
                        default='',
                        help='实验备注')

    parser.add_argument('--fs_config_path',
                        type=str,
                        default=None,
                        help='few-shot config for pretrain pipeline')

    parser.add_argument('--pipeline', 
                        type=str, 
                        default='Pipeline_01_default',
                        help='实验流水线模块路径')
    
    args = parser.parse_args()
    pipeline = importlib.import_module(f'src.{args.pipeline}')
    # 执行DG流水线
    results = pipeline.pipeline(args)
    print(f"完成所有实验！")
    
    return results

if __name__ == "__main__":
    main()


