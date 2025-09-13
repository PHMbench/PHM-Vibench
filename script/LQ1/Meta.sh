#!/bin/bash

# 设置为 "fail on error"，这样任何命令失败都会使脚本停止
set -e

# --- 1. 定义您要遍历的实验参数 ---
TARGET_SYSTEM_IDS=(1 2 6 14)  # 您想要遍历的目标数据集ID
SHOTS=(1 5)                     # 您想要测试的 k-shot 值

# --- 2. 定义输入和输出路径 ---
BASE_CONFIG_PATH="configs/demo/template.yaml" # 您的模板配置文件路径
OUTPUT_CONFIG_DIR="configs/generated/"        # 存放新生成的配置文件的目录

# --- 3. 配置文件生成 ---
echo "=================================================="
echo "STEP 1: Generating YAML configuration files using sed..."
echo "=================================================="

# 确保输出目录存在
mkdir -p "$OUTPUT_CONFIG_DIR"

# 循环遍历所有实验组合
for target_id in "${TARGET_SYSTEM_IDS[@]}"; do
  for shot in "${SHOTS[@]}"; do
    
    # 定义本次实验的唯一名称和新配置文件的路径
    experiment_name="target_${target_id}_shot_${shot}"
    new_config_path="${OUTPUT_CONFIG_DIR}/config_${experiment_name}.yaml"
    
    echo "Generating config for: ${experiment_name}"

    # --- 使用sed命令进行查找和替换 ---
    # sed -e "s/旧内容/新内容/" 模板文件 > 新文件
    # -e 选项允许我们执行多个替换表达式
    # 我们用双引号 "..." 包围表达式，以便shell可以解析变量如 $target_id
    sed \
      -e "s/project: .*/project: \"${experiment_name}\"/" \
      -e "s|output_dir: .*|output_dir: \"results/${experiment_name}\"|" \
      -e "s/target_system_id: .*/target_system_id: [${target_id}]/" \
      -e "s/num_support: .*/num_support: ${shot}/" \
      -e "s/num_query: .*/num_query: ${shot}/" \
      "$BASE_CONFIG_PATH" > "$new_config_path"
      
    echo "  -> Saved new config to: ${new_config_path}"
  done
done

echo "YAML configuration files generated successfully."
echo ""


# --- 4. 实验循环执行 ---
echo "=================================================="
echo "STEP 2: Starting experiments from directory: $OUTPUT_CONFIG_DIR"
echo "=================================================="

# 遍历目录中的每一个.yaml文件
for config_file in "$OUTPUT_CONFIG_DIR"/*.yaml; do
  echo "--------------------------------------------------"
  echo "Executing experiment with config: $config_file"
  echo "--------------------------------------------------"
  
  # 调用你的主训练程序
  python main.py --config_file "$config_file"
  
  echo "Finished experiment with config: $config_file"
  echo ""
done


echo "=================================================="
echo "All experiments have been completed."
echo "=================================================="