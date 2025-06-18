

TODO 
# 1领域泛化(DG)任务
task:
  name: classification
  type: DG
  selector: default  # 使用默认的DG选择器
  target_system_id: [RM_001_CWRU]
  source_domain_id: [0, 1, 2]
  target_domain_id: [3, 4]

# 2小样本学习(Few-Shot)任务
task:
  name: classification
  type: few_shot
  selector: few_shot
  target_system_id: [RM_001_CWRU]
  n_way: 5         # 5类分类
  k_shot: 1        # 每类1个样本用于训练
  n_query: 15      # 每类最多15个样本用于测试
  label_column: Label

# 3. 不平衡数据任务
task:
  name: classification
  type: imbalanced
  selector: imbalanced
  target_system_id: [RM_001_CWRU]
  imbalance_ratio: 0.1  # 少数类与多数类的比例
  minority_labels: [2, 4]  # 指定少数类标签
  stratify: true  # 使用分层抽样