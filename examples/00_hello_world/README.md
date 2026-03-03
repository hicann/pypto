# Hello World 样例

本样例展示了 PyPTO 最简单的张量加法操作，适合初次接触框架的开发者快速了解基本工作流。

## 总览介绍

本样例通过一个简单的张量加法演示了 PyPTO 的完整开发流程：
- 使用 `@pypto.frontend.jit` 定义内核函数。
- 通过 PyTorch 张量创建输入数据。
- 调用 JIT 内核执行计算并验证结果。

## 代码结构

- **`hello_world.py`**: Hello World 示例脚本，包含最基础的张量加法操作。

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
# 运行 Hello World 示例
python3 hello_world.py

# 以仿真模式运行
python3 hello_world.py --run_mode sim
```

## 注意事项

- 本样例是入门的第一步，完成后建议继续学习 [初级样例 (01_beginner)](../01_beginner/README.md)。
