# PyPTO 通信算子超时定位指南

## 概述

PyPTO 通信算子用于实现多卡间的数据传输与同步，是分布式计算场景的核心组件。通信算子超时问题是分布式场景中常见且复杂的调试场景。在实际使用中，排除驱动包、依赖组件相关问题后，绝大多数超时问题的根本原因是 shmem_wait_until 等不到信号值。因此本文主要侧重于快速定位 shmem_wait_until 导致超时的根本原因。

根据超时发生的时机，可将问题分为两类：

- HOST 侧超时：kernel 未下发，需检查 host 侧问题
- DEVICE 侧超时：kernel 已下发但执行超时，本文重点阐述此类问题定位

## HOST 超时

操作步骤：

1. 设置全局日志级别，可参考 [README.md](../../trouble_shooting/README.md) 日志环境变量，重点关注 ASCEND_MODULE_LOG_LEVEL，开启所需模块日志。

说明：通过开启环境变量查看 HOST 日志是否有打印 Kernel Launch，有打印则 DEVICE 超时，否则 HOST 超时。在定位前需先基础预检，来排除环境或常见问题，可参考 [精度调试指南](../debug/precision.md) 中基础预检，确保问题确实是通信超时而非环境等问题。本文暂无 HOST 超时指导，必要时，可提供日志提 ISSUE 解决。

## DEVICE 超时

PyPTO 提供多个 DFX 调试开关，同时开启 DEVICE 日志，检查详细日志帮助定位超时问题。

操作步骤：

1. 开启详细日志开关，可参考 [machine.md](../../trouble_shooting/machine.md) 启用追踪日志章节。

2. 设置全局日志级别，可参考 [README.md](../../trouble_shooting/README.md) 日志环境变量，控制 CANN 日志输出行为。

3. 检查落盘日志。

```bash
# 检查 Device 日志（运行阶段错误）
grep -i "error" $ASCEND_PROCESS_LOG_PATH/debug/device*/device*.log

# 检查 plog 日志（编译阶段错误）
grep -i "ERROR" $ASCEND_PROCESS_LOG_PATH/debug/plog/pypto-*.log
```

### 环境、CANN包等检查

检查目的：排除 CANN 包、驱动包等问题。

检查要点：

- $ASCEND_PROCESS_LOG_PATH/debug/device* 目录无 pypto 启动 device 日志（`Initialize "Device trace already...`）
- $ASCEND_PROCESS_LOG_PATH/debug/plog 目录中有 ERROR 日志
- 查找首次出现的 ERROR 日志，根据报错模块定界（如 HCCL、驱动等）

日志检查示例：

```
# ASCEND_PROCESS_LOG_PATH/debug/plog 日志报错
[ERROR] HCCL(<pid>,python3):YYYY-MM-DD HH:MM:SS.mmm [json_parser.cc:100][<pid>]JSON parse error: [json.exception.parse_error.101] parse error at line 2, column 23: syntax error while parsing object key - invalid literal; last read: '"2.0",   /'; expected string literal at byte 25
```

当前首报错 ERROR 日志为 HCCL 相关，需先解决相关依赖模块问题。

### Task 调度检查

检查目的：根据首 ERROR 日志定界错误问题类型。

- 首次 ERROR 日志是 `#sche.dtask.leave: Aicpu[%d] proc finish: finishedFunctionCnt=%x, coreFunctionCnt=%x, ..., but timeout !` 则大部分为 Task 调度异常或 shmem_wait_until Task 超时导致，其中：`finishedFunctionCnt` 表示已完成 Task 数量，`coreFunctionCnt` 表示总 Task 数量。
- 否则需先定位造成此 ERROR 的原因，需关注环境、前端代码、计算图是否符合预期等。

根据当前 DFX 日志，可重点关注以下关键日志（日志关键字可能随代码变化有所出入）：

| 日志关键字(函数名称[行号]: 部分日志) | 含义 |
|------------|------|
| `Initialize "Device trace already...` | PyPTO 启动 Device 成功 |
| `DumpTaskDetail "aiv taskId...` | Task Ready 状态 |
| `RunManager "Schedule run init...` | PyPTO init 初始化成功，准备 Task Dispatch |
| `Init "Init aicpu...` | AICPU 控制器初始化（shmem_wait_until Task 由 AICPU 控制） |
| `SendTaskToAicore "Send task...` | Task Dispatch（非 AICPU Task） |
| `ReleaseCoreByRegVal "resolve task core...` | Task Complete |
| `PushAicpuTaskQueue "PushAicpuTaskQueue...` | Task 依赖解析完成 |
| `TaskDispatch "Dispatch...` | Task Dispatch（AICPU Task） |
| `PrepareTask "PrepareTask...` | Task 信息（地址、shape 等） |
| `PollCompleted "expectedSum_=...` | Task Complete（shmem_wait_until 等到信号） |
| `RunTask "#sche.dtask.leave: Aicpu[X] proc finish: finishedFunctionCnt=%X, coreFunctionCnt=%X, ..., but timeout` | Task 超时 |

说明：建议同时开启泳道图，方便检查任务间的执行顺序，参考 [快速入门](../../tools/introduction/快速入门.md) 中查看泳道图，OUTPUT 目录下 dyn_topo.txt 在定位中可能会多次用到。

#### Device 初始化检查

检查目的：DeviceCtrl、DeviceArgs 等初始化成功。

检查要点：

- 日志 `RunManager "Schedule run init...` 有打印，AICore 控制器初始化成功。
- 日志 `Init "Init aicpu...` 有打印，AICPU 控制器初始化成功。

若无以上日志打印，会导致超时等问题，需先重点定位此原因。

#### Task Dispatch 检查

根据 dyn_topo.txt 和 CodeGen 生成的 CCE 文件，关联 shmem_op CCE 信息和 device 侧 taskId，检查所有任务的 Dispatch 情况。其中 dyn_topo.txt 的 taskId 和 device 日志对应，根据 leafHash 和 cce 文件中的 funcHash 对应，找到具体的 op，coreType 表示 task 类型，successors 表示后继 task。
dyn_topo.txt 示例：

```txt
seqNo,taskId,rootIndex,rootHash,opmagic,leafIndex,leafHash,coreType,psgId,wrapId,staticSuccCount,successors
0,0,0,3599269173890440401,10001,17,6153214905358987192,4,0,-1,0,196618,196628,196630,196632,196634,196636,196638,196640
```

CCE 文件示例：

```txt
// funcHash 6153214905358987192

extern "C" [aicore] void TENSOR_LOOP_MM_ALLREDUCE_ADD_RMSNORM_Unroll1_PATH0_hiddenfunc2_16_0_9007199254740992(CoreFuncParam* param, int64_t GMStackBase, __gm__ int64_t *hcclContext, __gm__ TaskStat* taskStat)
...
TileOp::Distributed::ShmemPut<...>...
...
```

1. shmem_put+shmem_signal、shmem_get 未全部 Dispatch

根据日志 `SendTaskToAicore "Send task...` 以及 dyn_topo.txt 的 taskId 列，检查 task 是否全部 Dispatch 结束，需保证所有 task Dispatch 完成。

2. shmem_wait_until task 未全部 Dispatch

根据日志 `TaskDispatch "Dispatch...` 检查 shmem_wait_until task 是否 Dispatch 成功，未 Dispatch 成功需要关注以下：

- 根据日志 `PrepareTask "PrepareTask...` 检查 shmem_wait_until task 是否准备完成。
- 根据日志 `PushAicpuTaskQueue "PushAicpuTaskQueue...` 检查 shmem_wait_until task 依赖解析是否成功。

#### Task Completion 检查

检查目的：已 Dispatch 的 Task 是否正常 Complete。

1. shmem_put+shmem_signal、shmem_get 未全部 Complete

某一 op task 是否 Complete 可根据日志 `ReleaseCoreByRegVal "resolve task core...` taskId 和前文 Dispatch 日志的 taskId 进行对比做差，判定是否有未 Complete。

2. shmem_wait_until task 未全部 Complete

根据日志 `PollCompleted "expectedSum_=...` 检查 shmem_wait_until task 是否 Complete，未 Complete 的 task 可以根据日志 `PrepareTask "PrepareTask...` 打印的 taskId 对比做差。

说明：shmem_put+shmem_signal 全部 Dispatch 并 Complete，且 shmem_wait_until task 全部 Dispatch，但 shmem_wait_until 未全部 Complete 则是 shmem_wait_until task 超时。

### shmem_wait_until task 超时

判定哪些 shmem_wait_until task 未等到信号：

- 根据泳道图点击所有的 AICPU 任务，没有后序箭头的 task 表示该 task 超时。
- 根据 PrepareTask 日志中所有的 AICPU taskId 和已经等到信号的 taskId 对比做差，找到未执行的 shmem_wait_until task。

#### 检查 shmem_signal 写入信号地址和 shmem_wait_until 等待信号地址是否一致

- 检查两个 op 的切块是否一致。
- 检查是否多个信号写入同一地址。
- 必要时增加 kernel 侧日志，打印详细信息进行对比，在目标 TileOp 执行后添加打印，打印需要验证的信息，例如打印 addr、shape 等信息。示例代码如下：

```cpp
GMTileTensorFP32DIM2_1 gmTensor_1((__gm__float*)(RUNTIME_COA_GET_PARAM_ADDR_MAYBE_CONST(2, 0, 0, 1), DynLayout2Dim(Shape2Dim(GET_PARAM_RAWSHAPE_2(param, 0, 1)), Stride2Dim(GET_PARAM_STRIDE_2(param, 0, 1)))))
// 打印gmTensor_1的 addr
AicoreLogF(param->ctx, "gmTensor_1 addr: %lu", RUNTIME_COA_GET_PARAM_ADDR_MAYBE_CONST(2, 0, 0, 1));
```

#### 检查信号值是否符合预期

- 检查 `shmem_signal` 的 `signal` 参数是否正确传递。
- 检查 CCE 文件 `output/output_<id>/kernel_aicore/` 中 signal 值是否符合预期，atomicType 是否符合预期。
- atomicType 是否保证原子操作。
- 必要时增加 kernel 侧日志。

#### 判定信号写入共享信号区域

目的：通过 shmem_get 读取共享信号数据

操作步骤：将 shmem_get 的输出加入算子输出列表：

```python
# 正常的通信流程完成后，读取 Win 区数据
win_data = pypto.distributed.shmem_get(
    shmem_tensor, my_pe, shmem_shape, [0, 0],
    pred=[wait_until_out], valid_shape=shmem_shape)

# 将 win_data 加入算子输出列表，用于后续精度对比
return [output, win_data]
```

## 定位技巧

### 缩小问题规模

缩小问题规模旨在简化场景，提高定位效率。缩小后可查看 Pass 图是否符合预期，分析各算子的连接关系、数据流、切分方式。需确保缩小后仍能复现原始问题。

1. 减小 world_size

将 world_size 从大值减小到最小可复现值（如 world_size = 2）。

2. 减少切块数量或不切块

增大 TileShape 减少切块数量，或直接使用不切分的配置。不切分时问题消失，说明切分逻辑有问题；不切分时问题仍存在，说明问题与切分无关。

3. 减小 Shape 规格

调小模型的 Shape 规格（如 batch_size、hidden_size），降低计算和通信规模，便于直接对计算结果进行分析。

4. 减小维度

减少 tensor 的维度数量（如从 3D 降到 2D），简化数据结构和计算逻辑。维度减少后问题消失，说明问题可能与特定维度的处理相关；维度减少后问题仍存在，说明问题与维度数量无关。

5. 单一算子

如果是融合算子或在整网中出现超时现象，先根据章节"Task 调度状态检查"定位到具体的通信算子，构造单一算子再进行定位。

### 可视化查看 task 依赖情况

1. leafHash → 前端代码映射

如果问题规模较大且无法缩小问题规模，可以通过项目组提供 SKILL，可视化检查 task 间的依赖关系，映射方法参考 [AGENTS.md](../../../../AGENTS.md) pypto-op-perf-tune 目录下 leafhash-to-code-mapping.md，使用 agent 可视化检查 task 依赖关系和执行情况。

### 偶现超时

1. 插入同步点

偶现性问题定位比较困难，当前遇到的问题大部分和同步相关，可以在 [tileop_shmem.h](../../../../framework/src/interface/tileop/distributed/tileop_shmem.h) 涉及写信号的代码插入 PIPE_ALL，根据插入的 PIPE_ALL 二分定位。

## 典型错误场景

### 场景一：HCCL 配置错误

场景说明：未真正开始执行通信任务，首报错为 HCCL 相关，属于非 PyPTO 问题。

错误日志特征：

```txt
# ASCEND_PROCESS_LOG_PATH/debug/plog 日志报错
[ERROR] HCCL(<pid>,python3):YYYY-MM-DD HH:MM:SS.mmm [json_parser.cc:100][<pid>]JSON parse error: ...
[ERROR] HCCL(<pid>,python3):YYYY-MM-DD HH:MM:SS.mmm [hcclCommOp.cc:124] errNo[0x0000000005010001] load allocated resource to json fail.
```

处理方式：首报错为 HCCL 相关，需先解决 HCCL 配置或依赖问题，提 ISSUE 到 HCCL 仓解决。

### 场景二：AICPU 未拉起导致超时

场景说明：未真正开始执行通信任务，AICPU 线程未拉起，属于非 PyPTO 问题。

日志特征：

```txt
# ASCEND_PROCESS_LOG_PATH/debug/device_* 日志未搜索到 Init aicpu task manager
# shmem_wait_until 任务无法正常拉起
```

处理方式：检查 CANN/驱动版本，确保 AICPU 功能正常。

### 场景三：信号值写入同一地址

场景说明：信号内存地址计算错误，信号写入同一块地址。

错误日志特征：

```txt
[DEBUG] PrepareTask "PrepareTask taskId=65558, actualAddr=0x10004ca001a0, actual offset=[1, 4]"
[DEBUG] PrepareTask "PrepareTask taskId=65564, actualAddr=0x10004ca001a0, actual offset=[1, 4]"  # 映射相同地址
[ERROR] Aicpu[1] proc finish: finishedFunctionCnt=29, coreFunctionCnt=30, but timeout !.
```

关键日志：

- `PrepareTask` 中存在相同的 `actualAddr`，表示信号写入同一块地址
- `finishedFunctionCnt < coreFunctionCnt` 表示部分 task 未 Complete

问题定位：

- 对应前端代码检查中的输入一致性检查
- 修复：尾块计算有误导致映射相关地址，修改计算方式

## 常见问题 FAQ

### Q1：多轮通信场景下，为什么首轮正常但后续轮次超时？

常见原因：多轮通信下，pypto.AtomicType.ADD 场景下，前后轮没有插入同步时，快卡可能会比慢卡快一轮，导致信号区域不符合预期。

| 原因 | 检查方法 |
|------|----------|
| 未执行 barrier | 检查每轮通信前是否有 `shmem_barrier_all` |

---

## 相关文档

- [分布式 API 文档](../../api/distributed/index.md)
- [shmem_wait_until API](../../api/distributed/pypto-distributed-shmem_wait_until.md)
- [shmem_signal API](../../api/distributed/pypto-distributed-shmem_signal.md)
- [分布式故障排查](../../trouble_shooting/distributed.md)
- [通信算子切块设置指南](distributed_operation_tiling_guide.md)
- [通信算子精度定位指南](distributed_precision_guide.md)
