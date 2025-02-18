# PyPTO

## 🔥Latest News

- [2025/12] PyPTO 项目首次上线.

## 🚀概述

PyPTO（发音: pai p-t-o）是一款面向 AI 加速器的高性能编程框架, 旨在简化复杂融合算子乃至整个模型网络的开发流程, 同时保持高性能计算能力. 该框架采用创新的 **PTO（Parallel Tensor/Tile Operation）编程范式**, 以 **基于 Tile 的编程模型** 为核心设计理念, 通过多层次的中间表示（IR）系统, 将用户通过 API 构建的 AI 模型应用从高层次的 Tensor 图逐步编译成硬件指令, 最终生成可在目标平台上高效执行的可执行代码.

### 核心特性

- **基于 Tile 的编程模型**: 所有计算都基于 Tile（硬件感知的数据块）进行, 充分利用硬件并行计算能力和内存层次结构
- **多层级计算图转换**: 通过编译 Pass 将 Tensor Graph 转换为 Tile Graph、Block Graph 和 Execution Graph, 每一步包括一系列 Pass 优化流程
- **自动化代码生成**: 编译结果通过 CodeGen 生成底层 PTO 虚拟指令代码, 再通过编译器将虚拟指令代码编译成目标平台的可执行代码
- **MPMD 执行调度**: 可执行代码被加载到设备侧, 通过 MPMD（Multiple Program Multiple Data）的方式调度到设备上的处理器核
- **完整的工具链支持**: 全流程的编译中间产物和运行时性能数据可通过 IDE 集成的工具链可视化识别性能瓶颈, 开发者也可以通过工具链控制编译和调度行为
- **Python 友好 API**: 提供直观的 Tensor 级别抽象, 贴近算法开发者的思维模式, 支持动态 Shape 和符号化编程
- **分层抽象设计**: 对不同开发者暴露不同抽象层次, 算法开发者使用 Tensor 层次, 性能专家使用 Tile 层次, 系统开发者使用 Block 层次

### 目标用户

- **算法开发者**: 主要使用 Tensor 层次编程, 快速实现和验证算法, 专注于算法逻辑
- **性能优化专家**: 可使用 Tile 或 Block 层次, 进行深度性能调优, 以实现极致性能
- **系统开发者**: 可在 Tensor/Tile/Block 和 PTO 虚拟指令集层次上进行三方框架对接或集成, 以及工具链开发

## ⚡️最佳实践样例

PyPTO 提供了丰富的示例代码, 涵盖从基础操作到复杂模型实现的多个层级. 一些最佳实践样例参考:

### 大模型实现样例

- [DeepSeekV3.2 SFA](examples/models/deepseek_v32_exp/deepseekv32_sparse_flash_attention_quant.py) - 稀疏 Flash Attention 量化实现
- [DeepSeekV3.2 MLA-PROLOG](examples/models/deepseek_v32_exp/deepseekv32_mla_indexer_prolog_quant.py) - MLA Indexer Prolog 量化实现
- [GLM V4.5 Attention](examples/models/glm_v4_5/glm_attention.py) - GLM 注意力机制实现
- [GLM V4.5 ExpertsSelector](examples/models/glm_v4_5/glm_select_experts.py) - GLM 专家选择器实现

### 学习路径

在 [examples](examples) 目录下, 我们规划了多个层级的样例:

- [beginner/](examples/01_beginner): 基础操作示例, 帮助初学者快速上手 PyPTO 编程
- [intermediate/](examples/02_intermediate): 中级示例, 包括自定义操作、神经网络模块等
- [advanced/](examples/03_advanced): 高级示例, 包括复杂模式和多函数组合
- [models/](examples/models): 完整的大模型实现样例, 供快速移植和部署

这些示例可以帮助开发者学习如何编写 PyPTO 算子, 从简单的 Tensor 操作到复杂的模型网络实现.

## 🔥性能表现

PyPTO 在多个 AI 工作负载上展现出优异的性能表现. 框架通过多层级优化和硬件感知的代码生成, 能够充分利用 AI 加速器的计算资源.

（性能对比数据待补充）

## ⚡️快速入门

若您希望快速体验 PyPTO 的使用和开发过程, 请访问如下文档获取简易教程.

- [环境部署](docs/context/prepare_environment.md): 介绍项目基础环境的搭建, 包括软件包和第三方依赖的获取和安装.
- [编译安装](docs/context/build_and_install.md): 环境部署后, 介绍如何快速获取或编译 PyPTO 软件包并安装.
- [样例运行](docs/invocation/examples_invocation.md): 安装 PyPTO 软件包后, 介绍如何快速实现样例运行.

## 📖文档资源

若您希望深入体验项目功能并修改源码, 请访问如下文档获取详细教程.
- **[docs](./docs/)**: 当前发布版本的详细文档, 包括编程指南、API 参考等
- **[示例代码](./examples/)**: 丰富的示例代码, 从基础到高级应用

## 🔍目录结构

关键目录如下:
```
├── examples/                   # 示例代码
│   ├── 01_beginner/            # 初级示例
│   ├── 02_intermediate/        # 中级示例
│   ├── 03_advanced/            # 高级示例
│   └── models/                 # 模型实现示例
│
├── python/                     # Python 源码
│   ├── pypto/                  # Python 包源码根目录
│   ├── src/                    # pybind11 源码根目录
│   └── tests/                  # Python 测试用例源码（UTest, STest）
│
├── framework/                  # C++ 源码根目录
│   ├── include/                # C++ 对外头文件
│   ├── src/                    # C++ 源码
│   │   ├── codegen/            # 代码生成模块
│   │   ├── passes/             # 编译 Pass 模块
│   │   └── ...
│   └── tests/                  # C++ 测试用例源码
│
├── tools/                      # 工具脚本
│
├── cmake/                      # 构建所需的 CMake 公共配置及脚本
├── build_ci.py                 # CI执行构建、执行 UTest、执行 STest 辅助脚本
├── CMakeLists.txt              # 顶层 CMakeLists.txt, 定义所有对外公开编译开关
├── pyproject.toml              # Python 编译工具配置文件
├── LICENSE                     # 许可证文件
└── setup.py                    # Python 编译工具脚本文件(setuptools)
```

## 📝相关信息

- [贡献指南](CONTRIBUTION.md)
- 安全声明

    PyPTO 项目致力于保障用户数据和应用的安全. 在使用本框架时, 请注意:
    - 请从官方渠道获取 PyPTO 软件包
    - 定期更新到最新版本以获取安全补丁
    - 在生产环境中使用前, 请进行充分的安全测试
    - 如发现安全问题, 请通过安全渠道报告

- [许可证](LICENSE)

## 联系我们

- **问题反馈**: 通过 GitCode【Issues】提交问题
- **功能建议**: 通过 GitCode【讨论】参与交流
- **技术支持**: 参考文档或提交 Issue

---

**注意**: 本文档会持续更新, 请关注最新版本.
