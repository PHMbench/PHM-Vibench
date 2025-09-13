# Implementation Plan

## Task Overview
PHM-Vibench代码重构采用严格的原子化任务分解，每个任务专注于1-3个相关文件的具体修改，确保15-30分钟内完成。所有任务保持向后兼容性，优先使用现有代码组件。

## Steering Document Compliance
严格遵循项目结构约定：
- 保持`src/{factory_type}/`目录结构
- 新增`src/services/`服务层
- 测试文件对应`test/`结构
- 配置文件遵循现有YAML格式

## Atomic Task Requirements
**每个任务必须满足：**
- **文件范围**: 涉及1-3个相关文件
- **时间限制**: 15-30分钟内完成
- **单一目标**: 一个可测试的具体成果
- **明确文件**: 指定确切的文件路径
- **Agent友好**: 清晰的输入输出，最小上下文切换

## Good vs Bad Task Examples
❌ **Bad Examples (Too Broad)**:
- "实现ExperimentCoordinator实验协调器" (affects many files, multiple purposes)
- "建立代码质量监控仪表板" (vague scope, no file specification)
- "重构BaseTask任务基类" (affects all task implementations)

✅ **Good Examples (Atomic)**:
- "创建TypedFactory基类在src/infrastructure/typed_factory.py中定义泛型接口"
- "添加类型检查方法到Registry类在src/utils/registry.py第45行"
- "修改BaseReader添加validate_data方法在src/data_factory/reader/BaseReader.py"

## Tasks

### Phase 1: 基础设施层增强

- [ ] 1. 创建基础设施层包初始化
  - 文件: src/infrastructure/__init__.py
  - 创建空的包初始化文件
  - 添加基本的导入语句占位符
  - 目的: 建立基础设施层目录结构
  - _Requirements: 1.1_

- [ ] 2. 增强Registry类添加类型检查
  - 文件: src/utils/registry.py (修改现有文件)
  - 在Registry类中添加validate_component方法
  - 添加类型检查和依赖验证逻辑
  - 保持现有register装饰器兼容性
  - _Leverage: src/utils/registry.py:Registry类_
  - _Requirements: 4.1_

- [ ] 3. 创建组件验证器基础类
  - 文件: src/infrastructure/component_validator.py
  - 定义ComponentValidator类和基本接口
  - 实现validate_interface方法框架
  - 添加基本的类型检查逻辑
  - _Requirements: 4.1_

- [ ] 4. 实现TypedFactory泛型基类接口
  - 文件: src/infrastructure/typed_factory.py
  - 创建Generic[T]基类定义
  - 实现register方法签名和基础逻辑
  - 集成ComponentValidator依赖
  - _Leverage: src/utils/registry.py_
  - _Requirements: 4.1, 4.4_

- [ ] 5. 添加TypedFactory的create方法实现
  - 文件: src/infrastructure/typed_factory.py (继续上个任务)
  - 实现create方法的组件实例化逻辑
  - 添加错误处理和类型转换
  - 集成Registry.get方法调用
  - _Requirements: 4.1_

- [ ] 6. 创建基础配置模型结构
  - 文件: src/configs/models.py
  - 定义BaseConfig基类使用Pydantic
  - 创建ExperimentConfig类框架
  - 添加基本字段和验证规则
  - _Requirements: 4.1, 4.4_

- [ ] 7. 添加DataConfig配置模型
  - 文件: src/configs/models.py (扩展上个任务)
  - 定义DataConfig类继承BaseConfig
  - 添加data_dir, metadata_file, batch_size字段
  - 实现字段验证和默认值设置
  - _Leverage: src/configs/config_utils.py中的配置项_
  - _Requirements: 4.1_

- [ ] 8. 添加ModelConfig和TaskConfig模型
  - 文件: src/configs/models.py (继续扩展)
  - 定义ModelConfig和TaskConfig类
  - 添加模型和任务相关的配置字段
  - 实现跨配置的验证逻辑
  - _Requirements: 4.1_

- [ ] 9. 创建配置服务基础框架
  - 文件: src/services/__init__.py
  - 创建服务层包初始化
  - 添加基本导入和模块结构
  - 定义服务层的命名约定
  - _Requirements: 3.2_

- [ ] 10. 实现ConfigService类基础结构
  - 文件: src/services/config_service.py
  - 创建ConfigService类定义
  - 添加__init__方法和基本属性
  - 集成load_config函数调用
  - _Leverage: src/configs/__init__.py:load_config_
  - _Requirements: 4.1_

- [ ] 11. 添加ConfigService的验证方法
  - 文件: src/services/config_service.py (扩展上个任务)
  - 实现validate_config方法
  - 集成Pydantic配置模型验证
  - 添加详细的错误报告逻辑
  - _Requirements: 4.4_

- [ ] 12. 实现配置转换和兼容性方法
  - 文件: src/services/config_service.py (继续扩展)
  - 添加convert_legacy_config方法
  - 实现ConfigWrapper到新模型的转换
  - 保持向后兼容的配置访问
  - _Leverage: src/configs/config_utils.py:ConfigWrapper_
  - _Requirements: 5.3_

### Phase 1 Testing

- [ ] 13. 创建TypedFactory单元测试基础
  - 文件: test/infrastructure/test_typed_factory.py
  - 创建测试类和基础测试方法
  - 添加注册功能的单元测试
  - 验证类型检查和错误处理
  - _Requirements: 5.2_

- [ ] 14. 添加ConfigService单元测试
  - 文件: test/services/test_config_service.py
  - 测试配置加载和验证功能
  - 验证Pydantic验证错误处理
  - 测试向后兼容性转换
  - _Requirements: 5.2_

### Phase 2: 数据服务层重构

- [ ] 15. 创建BaseService抽象基类
  - 文件: src/services/base_service.py
  - 定义BaseService抽象基类
  - 添加基本的initialize和cleanup方法
  - 实现服务生命周期管理框架
  - _Requirements: 3.3_

- [ ] 16. 从data_factory提取MetadataManager
  - 文件: src/services/metadata_manager.py
  - 复制data_factory中的元数据访问逻辑
  - 创建MetadataManager类包装现有功能
  - 保持现有元数据文件格式兼容
  - _Leverage: src/data_factory/__init__.py中元数据处理_
  - _Requirements: 2.1, 3.1_

- [ ] 17. 重构H5DataDict为CacheManager服务
  - 文件: src/services/cache_manager.py
  - 将H5DataDict逻辑转换为服务类
  - 添加缓存策略和生命周期管理
  - 保持现有缓存接口兼容性
  - _Leverage: src/data_factory/H5DataDict.py_
  - _Requirements: 2.1_

- [ ] 18. 创建DatasetBuilder服务类
  - 文件: src/services/dataset_builder.py
  - 提取数据集构建的公共逻辑
  - 创建统一的Reader调用接口
  - 集成MetadataManager依赖
  - _Leverage: src/data_factory/reader/BaseReader.py_
  - _Requirements: 2.1_

- [ ] 19. 实现DataService协调类
  - 文件: src/services/data_service.py
  - 创建DataService类整合三个子服务
  - 实现prepare方法协调数据准备
  - 添加get_datasets和get_loaders方法
  - _Requirements: 2.1, 3.1_

- [ ] 20. 增强BaseReader添加标准接口
  - 文件: src/data_factory/reader/BaseReader.py (修改现有)
  - 添加validate_data抽象方法
  - 强化load_data方法的类型提示
  - 添加统一的错误处理机制
  - _Leverage: 现有BaseReader实现_
  - _Requirements: 2.1, 4.1_

- [ ] 21. 创建ModelService基础结构
  - 文件: src/services/model_service.py
  - 定义ModelService类和基本接口
  - 集成TypedFactory[BaseModel]依赖
  - 实现create_model基础方法
  - _Leverage: src/model_factory/build_model.py_
  - _Requirements: 3.1_

- [ ] 22. 添加ModelService的加载和验证方法
  - 文件: src/services/model_service.py (扩展上个任务)
  - 实现load_pretrained方法
  - 添加validate_model_config方法
  - 集成模型元数据收集逻辑
  - _Requirements: 3.1, 4.1_

- [ ] 23. 创建TaskService基础框架
  - 文件: src/services/task_service.py
  - 定义TaskService类和核心接口
  - 集成TypedFactory[BaseTask]
  - 实现create_task基础方法
  - _Leverage: src/task_factory/T_*.py公共模式_
  - _Requirements: 2.3, 3.1_

- [ ] 24. 提取BaseTask公共训练逻辑
  - 文件: src/task_factory/base_task.py (修改现有)
  - 添加setup_common_metrics方法
  - 提取公共的优化器配置逻辑
  - 标准化训练步骤的接口
  - _Leverage: src/task_factory/T_*.py重复代码_
  - _Requirements: 2.3_

- [ ] 25. 创建TrainerService封装类
  - 文件: src/services/trainer_service.py
  - 封装PyTorch Lightning Trainer创建
  - 添加统一的训练配置管理
  - 实现检查点和回调管理
  - _Leverage: src/trainer_factory/构建逻辑_
  - _Requirements: 3.1_

### Phase 2 Testing

- [ ] 26. 测试DataService集成功能
  - 文件: test/services/test_data_service.py
  - 测试数据服务的端到端流程
  - 验证与现有data_factory的兼容性
  - 测试错误处理和恢复机制
  - _Requirements: 5.2_

- [ ] 27. 测试ModelService和TaskService
  - 文件: test/services/test_model_task_services.py
  - 测试模型和任务服务的创建功能
  - 验证类型安全和配置验证
  - 测试服务间的协调工作
  - _Requirements: 5.2_

### Phase 3: 实验协调层

- [ ] 28. 创建ExperimentCoordinator接口定义
  - 文件: src/services/experiment_coordinator.py
  - 定义ExperimentCoordinator类接口
  - 添加基本的run, prepare, finalize方法签名
  - 集成所有服务层依赖声明
  - _Requirements: 3.2_

- [ ] 29. 实现ExperimentCoordinator初始化逻辑
  - 文件: src/services/experiment_coordinator.py (扩展)
  - 实现__init__方法和服务依赖注入
  - 添加配置验证和服务初始化
  - 集成路径管理和日志设置
  - _Leverage: src/utils/path_name, makedir_
  - _Requirements: 3.2_

- [ ] 30. 添加实验生命周期管理方法
  - 文件: src/services/experiment_coordinator.py (继续)
  - 实现prepare方法的环境准备逻辑
  - 添加run_iteration模板方法
  - 实现finalize方法的结果收集
  - _Requirements: 3.2, 5.1_

- [ ] 31. 实现检查点和恢复机制
  - 文件: src/services/experiment_coordinator.py (继续)
  - 添加save_checkpoint方法
  - 实现resume_from_checkpoint逻辑
  - 集成PyTorch Lightning检查点系统
  - _Leverage: 现有检查点管理逻辑_
  - _Requirements: 5.2_

- [ ] 32. 创建具体Pipeline实现类
  - 文件: src/services/pipeline_implementations.py
  - 实现DefaultPipeline继承ExperimentCoordinator
  - 添加PretrainFinetunePipeline具体实现
  - 保持现有Pipeline接口兼容
  - _Leverage: src/Pipeline_*.py实验流程_
  - _Requirements: 5.3_

### Phase 3 Testing

- [ ] 33. 测试ExperimentCoordinator核心功能
  - 文件: test/services/test_experiment_coordinator.py
  - 测试实验协调的完整流程
  - 验证检查点和恢复机制
  - 测试错误处理和优雅降级
  - _Requirements: 5.2_

- [ ] 34. 验证Pipeline兼容性
  - 文件: test/compatibility/test_pipeline_compatibility.py
  - 测试新Pipeline与现有接口的兼容性
  - 验证所有现有Pipeline脚本正常工作
  - 性能基准对比测试
  - _Leverage: 现有Pipeline测试用例_
  - _Requirements: 5.3_

### Phase 4: 兼容性和质量改进

- [ ] 35. 创建兼容性适配器框架
  - 文件: src/compatibility/__init__.py
  - 建立兼容性层包结构
  - 定义适配器基础接口
  - 添加版本检测机制
  - _Requirements: 5.3_

- [ ] 36. 实现build_*函数适配器
  - 文件: src/compatibility/legacy_adapter.py
  - 包装build_data函数调用新DataService
  - 保持完全相同的API签名和行为
  - 添加deprecation警告和迁移提示
  - _Leverage: 现有build_data函数_
  - _Requirements: 5.3_

- [ ] 37. 添加build_model和build_task适配器
  - 文件: src/compatibility/legacy_adapter.py (扩展)
  - 适配build_model到ModelService
  - 适配build_task到TaskService
  - 保持错误处理行为一致性
  - _Leverage: build_model, build_task函数_
  - _Requirements: 5.3_

- [ ] 38. 实现代码复杂度检查器
  - 文件: src/infrastructure/complexity_checker.py
  - 使用AST分析文件行数和函数复杂度
  - 检查嵌套层级和圈复杂度
  - 生成违规报告和修复建议
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [ ] 39. 创建import语句分析工具
  - 文件: src/infrastructure/import_analyzer.py
  - 扫描和识别通配符导入语句
  - 生成具体导入的替换建议
  - 验证导入路径的正确性
  - _Requirements: 4.2_

- [ ] 40. 实现print语句检测和替换
  - 文件: src/infrastructure/print_migrator.py
  - 使用AST检测所有print语句
  - 分类print语句的用途(调试、输出、错误)
  - 生成结构化日志的替换代码
  - _Requirements: 4.3_

- [ ] 41. 批量清理deprecated目录
  - 文件: scripts/cleanup_deprecated.py
  - 扫描src/configs/deprecated/目录内容
  - 识别重复和无用的配置文件
  - 生成安全删除脚本和备份
  - _Requirements: 2.2_

- [ ] 42. 执行import语句自动修复
  - 文件: scripts/fix_imports_batch.py
  - 应用import_analyzer的分析结果
  - 批量替换通配符导入为具体导入
  - 验证修复后代码的正确性
  - _Leverage: src/infrastructure/import_analyzer.py_
  - _Requirements: 4.2_

- [ ] 43. 执行print语句批量替换
  - 文件: scripts/migrate_prints_batch.py
  - 应用print_migrator的替换建议
  - 建立统一的日志配置
  - 验证日志输出的功能等价性
  - _Leverage: src/infrastructure/print_migrator.py_
  - _Requirements: 4.3_

- [ ] 44. 扫描和处理TODO/FIXME标记
  - 文件: scripts/process_todos.py
  - 使用grep扫描所有TODO和FIXME
  - 分类标记并生成处理建议
  - 创建GitHub issues或实现简单修复
  - _Requirements: 6.2_

- [ ] 45. 添加类型提示到公共API
  - 文件: scripts/add_type_hints.py
  - 使用mypy分析缺失的类型提示
  - 为工厂函数添加完整类型注解
  - 验证类型提示的正确性和完整性
  - _Requirements: 4.1, 4.4_

- [ ] 46. 创建性能基准测试套件
  - 文件: benchmarks/refactoring_benchmarks.py
  - 测量配置加载、模型创建时间
  - 对比重构前后的性能数据
  - 生成性能回退检测报告
  - _Requirements: Performance NFRs_

- [ ] 47. 生成API文档配置
  - 文件: docs/source/conf.py
  - 配置Sphinx自动文档生成
  - 设置API文档覆盖率检查
  - 添加代码示例和使用指南
  - _Requirements: 6.4_

- [ ] 48. 创建重构完成验证脚本
  - 文件: scripts/verify_refactoring.py
  - 运行所有质量检查工具
  - 验证所有需求的满足情况
  - 生成重构完成报告
  - _Requirements: 所有Requirements_

## Implementation Notes

### 执行顺序要求
- Phase 1 (任务1-14): 必须按序执行，建立基础设施
- Phase 2 (任务15-27): 可部分并行，遵循服务依赖
- Phase 3 (任务28-34): 需要Phase 2完成后执行
- Phase 4 (任务35-48): 多数可并行执行

### 成功标准
- 每个任务完成后运行相关测试
- 修改现有文件时保持向后兼容
- 新增功能必须有对应的单元测试
- 关键功能需要集成测试验证

### Agent执行要求
- 每个任务指定明确的文件路径
- 利用现有代码避免从零开始
- 保持API兼容性和行为一致
- 添加适当的错误处理和日志