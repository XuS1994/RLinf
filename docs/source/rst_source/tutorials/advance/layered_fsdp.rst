分层FSDP (Layered FSDP)
========================

概述
----

分层FSDP是RLinf框架中的一个高级功能，允许对不同模块使用不同的精度策略。这对于需要保持某些关键模块高精度（如fp32）同时其他模块使用低精度（如bf16）以提高训练效率的场景非常有用。

工作原理
--------

分层FSDP采用"嵌套包裹"的策略：

1. **内层FSDP**: 对指定的子模块使用不同精度进行FSDP包裹
2. **外层FSDP**: 对整个模型使用模型配置的精度进行FSDP包裹
3. **自动识别**: FSDP会自动识别已包裹的模块，避免重复包裹

这种设计确保了：
- 关键模块（如归一化层、输出层）保持高精度
- 大部分模块使用低精度以提高训练效率
- 内存使用得到优化

配置方法
--------

在配置文件中启用分层FSDP：

.. code-block:: yaml

  actor:
    model:
      # 启用分层FSDP功能
      enable_layered_fsdp: true
      
      # 指定模块精度配置，格式为 {module_path: precision_type}
      modules_precision:
        # fp32精度的模块（关键模块，需要高精度）
        'normalize_inputs.buffer_observation_state': 'fp32'
        'normalize_targets.buffer_action': 'fp32'
        'model.state_proj': 'fp32'
        'model.value_proj': 'fp32'
        
        # 使用模型默认精度的模块
        'model.paligemma_with_expert.paligemma': 'bf16'
      
      # 外层FSDP精度设置
      precision: "bf16"

**配置格式说明：**

- **module_path**: 模块的完整路径，如 'model.state_proj'
- **precision_type**: 精度类型，支持以下值：
  - 'fp32': 单精度浮点数
  - 'bf16': 半精度浮点数（bfloat16）
  - 'fp16': 半精度浮点数（float16）
  - 具体的torch.dtype对象

**注意事项：**
- 模块路径应该是模块名，而不是具体的参数名（如 'model.state_proj' 而不是 'model.state_proj.weight'）
- 这样可以避免重复包裹，提高效率
- 系统会自动处理该模块下的所有参数和缓冲区

使用场景
--------

分层FSDP特别适用于以下场景：

1. **数值稳定性要求高的模块**
   - 归一化层的统计信息
   - 输出投影层
   - 价值函数头

2. **混合精度训练**
   - 大部分模块使用bf16/fp16
   - 关键模块保持fp32

3. **内存优化**
   - 减少整体内存占用
   - 保持关键模块精度

示例配置
--------

完整的分层FSDP配置示例：

.. code-block:: yaml

  actor:
    model:
      enable_layered_fsdp: true
      modules_precision:
        # 归一化缓冲区
        'normalize_inputs.buffer_observation_state': 'fp32'
        'normalize_targets.buffer_action': 'fp32'
        
        # 关键投影层
        'model.state_proj': 'fp32'
        'model.action_proj': 'fp32'
        'model.value_proj': 'fp32'
        
        # 输出层
        'model.lm_head': 'fp32'
        
        # 使用模型默认精度的模块
        'model.paligemma_with_expert.paligemma': 'bf16'
      
      precision: "bf16"
      sharding_strategy: "no_shard"
    
    training_backend: "fsdp"
    micro_batch_size: 8
    global_batch_size: 64

故障排除
--------

如果遇到问题，请检查：

1. 模块路径是否正确
2. 是否启用了enable_layered_fsdp
3. modules_precision配置是否正确
4. 模型结构是否与配置匹配
