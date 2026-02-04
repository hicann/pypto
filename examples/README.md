# PyPTO 样例代码 (Examples)

本目录包含了一系列 PyPTO 的开发样例代码，旨在指导开发者如何使用该 AI 编程框架。样例代码根据开发者的学习路径，由浅入深地展示了框架的各项特性。

## 目录结构

样例代码分为以下几个等级：

- **00_hello_world (入门)**： 简单的张量加法，适合入门者了解框架初始化的 Hello World 示例。
- **01_beginner (初级)**: 基础操作与核心概念，适合刚接触 PyPTO 的开发者。
- **02_intermediate (中级)**: 神经网络组件、算子组合以及运行时（Runtime）特性。
- **03_advanced (高级)**: 复杂架构（如 Attention）、高级模式和系统级优化。
- **models (模型)**: 真实世界的大模型（LLM）算子实现样例。

## 快速开始

1. **初次使用？** 请从 [初级样例 (01_beginner)](01_beginner/README.md) 开始。
2. **构建神经网络？** 参考 [中级样例 (02_intermediate)](02_intermediate/README.md)。
3. **探索高级模式？** 查阅 [高级样例 (03_advanced)](03_advanced/README.md)。
4. **大模型算子实现？** 探索 [模型样例 (models)](../models/)。

### 环境准备
请参考[环境准备](../docs/install/prepare_environment.md)，完成基础环境搭建

### 软件安装
请参考[软件安装](../docs/install/build_and_install.md)，完成PyPTO软件安装

### 运行前配置（可选）
如需运行在真实NPU环境中，请参考如下配置

```bash
# 配置 CANN 环境变量
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash

# 设置 NPU 设备 ID（运行 NPU 样例时必需）
export TILE_FWK_DEVICE_ID=0
```
补充说明：如需运行models相关样例，请在真实设备运行

## 如何运行样例

大多数样例脚本支持运行全部测试或指定特定测试：

```bash

# 运行所有初级基础操作样例（默认为NPU模式运行）
python3 examples/01_beginner/basic/basic_ops.py

# 运行特定的样例
python3 examples/01_beginner/basic/basic_ops.py view_operations::test_view_operations

# 列出脚本中所有可用的样例
python3 examples/01_beginner/basic/basic_ops.py --list

# 指定以仿真（CPU）模式运行
python3 examples/01_beginner/basic/basic_ops.py --run_mode sim

```

## 学习路径建议

1. **第一阶段：夯实基础**
   - [Hello World](00_hello_world/hello_world.py)
   - [01_beginner/basic](01_beginner/basic/README.md)
   - [01_beginner/compute](01_beginner/compute/README.md)

2. **第二阶段：进阶组件**
   - [02_intermediate/nn/layer_normalization](02_intermediate/basic_nn/layer_normalization/README.md)
   - [02_intermediate/operators/softmax](02_intermediate/operators/softmax/README.md)
   - [02_intermediate/nn/ffn](02_intermediate/basic_nn/ffn/README.md)

3. **第三阶段：深度实践**
   - [03_advanced/nn/attention](03_advanced/advanced_nn/attention/README.md)
   - [models/deepseek_v32_exp](../models/deepseek_v32_exp/README.md)
   - [models/glm_v4_5](../models/glm_v4_5/README.md)

---

**祝您在 PyPTO 的编程之旅中收获满满！ 🚀**
