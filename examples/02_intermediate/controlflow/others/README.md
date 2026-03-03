# 其他运行时特性样例 (Other Runtime Features)

本目录包含 PyPTO 运行时的其他高级特性示例。

## 总览介绍

除循环和条件分支外，PyPTO 还提供了以下运行时特性：
- **动态形状 (Dynamic Shapes)**: 处理推理过程中波动的 Batch Size 或序列长度。
- **输入顺序灵活性 (Kernel Input Order)**: JIT 内核对输入参数顺序的容错能力。

## 代码结构

- **`dynamic.py`**: 动态形状处理示例，展示 `dynamic_axis` 的标记方法及动态维度下的 View/Assemble 分块策略。
  - `test_dynamic_mul`: 基础动态 Batch 维度。
  - `test_dynamic_partial`: 部分动态维度。
  - `test_dynamic_attention`: 动态 Batch 的多头注意力。
  - `test_dynamic_multi_dim`: 多动态维度。
- **`kernel_input.py`**: 输入顺序灵活性示例，展示 JIT 内核对参数顺序的容错能力。

## 运行方法

### 环境准备

```bash
# 配置 CANN 环境变量
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash

# 设置设备 ID
export TILE_FWK_DEVICE_ID=0
```

### 执行脚本

```bash
# 运行动态形状示例
python3 dynamic.py

# 运行输入顺序示例
python3 kernel_input.py
```

## 注意事项

- 动态维度仅标记必要的轴，保持其他维度为具体值以获得更好的编译优化。
- 使用动态形状时需配合 `pypto.view` / `pypto.assemble` 进行显式分块管理。
