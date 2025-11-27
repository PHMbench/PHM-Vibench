• 是的，run_pretrain 里拿到的 stage_cfg 已经是「应用完所有 override 之后的阶段完整配置」。

  更具体一点：

  - 对于 单 YAML + stages.overrides 的统一格式：
      - _load_unified_yaml_config() 会做三层合并：
          1. 全局块：data/model/task/trainer/environment（YAML 顶层）
          2. 该 stage 的 overrides（YAML 里 stages[i].overrides）
          3. CLI 参数：parse_stage_overrides(self.cli_overrides) 拆分出的
              - 全局 overrides（同时作用于所有 stage）
              - 按索引的 stage‑specific overrides（例如 stage_1.* / stages[0].*）
      - 合并结果变成一个 stage_ns，放进 self.cfg.stages[i]，同时 self.cfg.stage_1 =
        self.cfg.stages[0] 等。
  - run_pretrain(self, stage_cfg, ...) 调用时传入的就是这个 stage_ns（或等价的 dict/
    ConfigWrapper）：
      - _stage_to_namespaces(stage_cfg) 只是从 stage_ns.__dict__ 中拆出 environment/data/model/
        task/trainer，不再做任何 merge。
      - 所以 env/data/model/task/trainer 都已经包含了：
          - YAML 顶层配置
          - 对应 stage 的 overrides
          - CLI 的 global + stage overrides

  因此：

  - 你在命令行里用的 --override stages[0].trainer.max_epochs=1、--override
    stage_1.trainer.save_dir=... 等，都是在 orchestrator 初始化阶段就已经合并进 stage_cfg；
  - run_pretrain / run_adapt 阶段不会再重新应用 override，只是按这个最终的阶段配置去构建 data/
    model/task/trainer 并跑训练。