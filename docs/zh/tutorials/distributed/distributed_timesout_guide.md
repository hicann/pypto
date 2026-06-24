# PyPTO通信算子超时定位指南

## 概述

PyPTO通信算子用于实现多卡间的数据传输与同步，是分布式计算场景的核心组件。通信算子超时问题是分布式场景中常见且复杂的调试场景。在实际使用中，排除驱动包、依赖组件相关问题后，绝大多数超时问题的根本原因是shmem_wait_until等不到信号值。因此本文主要侧重于快速定位shmem_wait_until导致超时的根本原因。

根据超时发生的时机，可将问题分为两类：

- HOST侧超时：kernel未下发，需检查host侧问题
- DEVICE侧超时：kernel已下发但执行超时，本文重点阐述此类问题定位

## HOST超时

操作步骤：

1. 设置全局日志级别，可参考 [README.md](../../trouble_shooting/README.md)日志环境变量，重点关注ASCEND_MODULE_LOG_LEVEL，开启所需模块日志。

> [!NOTE]说明
> 通过开启环境变量查看HOST日志是否有打印Kernel Launch，有打印则DEVICE超时，否则HOST超时。在定位前需先基础预检，来排除环境或常见问题，可参考[精度调试](../debug/precision.md)中基础预检，确保问题确实是通信超时而非环境等问题。本文暂无HOST超时指导，必要时，可提供日志提ISSUE解决。

## DEVICE超时

PyPTO提供多个DFX调试开关，同时开启DEVICE日志，检查详细日志帮助定位超时问题。

操作步骤：

1. 开启详细日志开关，可参考 [machine.md](../../trouble_shooting/machine.md)启用追踪日志章节。

2. 设置全局日志级别，可参考 [README.md](../../trouble_shooting/README.md)日志环境变量，控制CANN日志输出行为。

3. 检查落盘日志。

```bash
# 检查Device日志（运行阶段错误）
grep -i "error" $ASCEND_PROCESS_LOG_PATH/debug/device*/device*.log

# 检查plog日志（编译阶段错误）
grep -i "ERROR" $ASCEND_PROCESS_LOG_PATH/debug/plog/pypto-*.log
```

### 环境、CANN包等检查

检查目的：排除CANN包、驱动包等问题。

检查要点：

- $ASCEND_PROCESS_LOG_PATH/debug/device*目录无pypto启动device日志（`Initialize "Device trace already...`）
- $ASCEND_PROCESS_LOG_PATH/debug/plog目录中有ERROR日志
- 查找首次出现的ERROR日志，根据报错模块定界（如HCCL、驱动等）

日志检查示例：

```.txt
# ASCEND_PROCESS_LOG_PATH/debug/plog日志报错
[ERROR] HCCL(<pid>,python3):YYYY-MM-DD HH:MM:SS.mmm [json_parser.cc:100][<pid>]JSON parse error: [json.exception.parse_error.101] parse error at line 2, column 23: syntax error while parsing object key - invalid literal; last read: '"2.0",   /'; expected string literal at byte 25
```

当前首报错ERROR日志为HCCL相关，需先解决相关依赖模块问题。

### Task调度检查

检查目的：根据首ERROR日志定界错误问题类型。

- 首次ERROR日志是`#sche.dtask.leave: Aicpu[%d] proc finish: finishedFunctionCnt=%x, coreFunctionCnt=%x, ..., but timeout !`则大部分为Task调度异常或shmem_wait_until Task超时导致，其中：`finishedFunctionCnt`表示已完成Task数量，`coreFunctionCnt`表示总Task数量。
- 否则需先定位造成此ERROR的原因，需关注环境、前端代码、计算图是否符合预期等。

根据当前DFX日志，可重点关注以下关键日志（日志关键字可能随代码变化有所出入）：

| 日志关键字(函数名称[行号]:部分日志) | 含义 |
|------------|------|
| `Initialize "Device trace already...` | PyPTO启动Device成功 |
| `DumpTaskDetail "aiv taskId...` | Task Ready状态 |
| `RunManager "Schedule run init...` | PyPTO init初始化成功，准备Task Dispatch |
| `Init "Init aicpu...` | AICPU控制器初始化（shmem_wait_until Task由AICPU控制） |
| `SendTaskToAicore "Send task...` | Task Dispatch（非AICPU Task） |
| `ReleaseCoreByRegVal "resolve task core...` | Task Complete |
| `PushAicpuTaskQueue "PushAicpuTaskQueue...` | Task依赖解析完成 |
| `TaskDispatch "Dispatch...` | Task Dispatch（AICPU Task） |
| `PrepareTask "PrepareTask...` | Task信息（地址、shape等） |
| `PollCompleted "expectedSum_=...` | Task Complete（shmem_wait_until等到信号） |
| `RunTask "#sche.dtask.leave: Aicpu[X] proc finish: finishedFunctionCnt=%X, coreFunctionCnt=%X, ..., but timeout` | Task超时 |

> [!NOTE]说明
> 建议同时开启泳道图，方便检查任务间的执行顺序，参考[查看泳道图](../introduction/quick_start.md#查看泳道图)，OUTPUT目录下dyn_topo.txt在定位中可能会多次用到。

#### Device初始化检查

检查目的：DeviceCtrl、DeviceArgs等初始化成功。

检查要点：

- 日志`RunManager "Schedule run init...`有打印，AICore控制器初始化成功。
- 日志`Init "Init aicpu...`有打印，AICPU控制器初始化成功。

若无以上日志打印，会导致超时等问题，需先重点定位此原因。

#### Task Dispatch检查

根据dyn_topo.txt和CodeGen生成的CCE文件，关联shmem_op CCE信息和device侧taskId，检查所有任务的Dispatch情况。其中dyn_topo.txt的taskId和device日志对应，根据leafHash和cce文件中的funcHash对应，找到具体的op，coreType表示task类型，successors表示后继task。
dyn_topo.txt示例：

```txt
seqNo,taskId,rootIndex,rootHash,opmagic,leafIndex,leafHash,coreType,psgId,wrapId,staticSuccCount,successors
0,0,0,3599269173890440401,10001,17,6153214905358987192,4,0,-1,0,196618,196628,196630,196632,196634,196636,196638,196640
```

CCE文件示例：

```txt
// funcHash 6153214905358987192

extern "C" [aicore] void TENSOR_LOOP_MM_ALLREDUCE_ADD_RMSNORM_Unroll1_PATH0_hiddenfunc2_16_0_9007199254740992(CoreFuncParam* param, int64_t GMStackBase, __gm__ int64_t *hcclContext, __gm__ TaskStat* taskStat)
...
TileOp::Distributed::ShmemPut<...>...
...
```

1. shmem_put+shmem_signal、shmem_get未全部Dispatch

    根据日志`SendTaskToAicore "Send task...`以及dyn_topo.txt的taskId列，检查task是否全部Dispatch结束，需保证所有task Dispatch完成。

2. shmem_wait_until task未全部Dispatch

    根据日志`TaskDispatch "Dispatch...`检查shmem_wait_until task是否Dispatch成功，未Dispatch成功需要关注以下：

    - 根据日志`PrepareTask "PrepareTask...`检查shmem_wait_until task是否准备完成。
    - 根据日志`PushAicpuTaskQueue "PushAicpuTaskQueue...`检查shmem_wait_until task依赖解析是否成功。

#### Task Completion检查

检查目的：已Dispatch的Task是否正常Complete。

1. shmem_put+shmem_signal、shmem_get未全部Complete

    某一op task是否Complete可根据日志`ReleaseCoreByRegVal "resolve task core...` taskId和前文Dispatch日志的taskId进行对比做差，判定是否有未Complete。

2. shmem_wait_until task未全部Complete

    根据日志`PollCompleted "expectedSum_=...`检查shmem_wait_until task是否Complete，未Complete的task可以根据日志`PrepareTask "PrepareTask...`打印的taskId对比做差。

    > [!NOTE]说明
    > shmem_put+shmem_signal全部Dispatch并Complete，且shmem_wait_until task全部Dispatch，但shmem_wait_until未全部Complete则是shmem_wait_until task超时。

### shmem_wait_until task超时

判定哪些shmem_wait_until task未等到信号：

- 根据泳道图点击所有的AICPU任务，没有后序箭头的task表示该task超时。
- 根据PrepareTask日志中所有的AICPU taskId和已经等到信号的taskId对比做差，找到未执行的shmem_wait_until task。

#### 检查shmem_signal写入信号地址和shmem_wait_until等待信号地址是否一致

- 检查两个op的切块是否一致。
- 检查是否多个信号写入同一地址。
- 必要时增加kernel侧日志，打印详细信息进行对比，在目标TileOp执行后添加打印，打印需要验证的信息，例如打印addr、shape等信息。示例代码如下：

```cpp
GMTileTensorFP32DIM2_1 gmTensor_1((__gm__float*)(RUNTIME_COA_GET_PARAM_ADDR_MAYBE_CONST(2, 0, 0, 1), DynLayout2Dim(Shape2Dim(GET_PARAM_RAWSHAPE_2(param, 0, 1)), Stride2Dim(GET_PARAM_STRIDE_2(param, 0, 1)))))
// 打印gmTensor_1的addr
AicoreLogF(param->ctx, "gmTensor_1 addr: %lu", RUNTIME_COA_GET_PARAM_ADDR_MAYBE_CONST(2, 0, 0, 1));
```

#### 检查信号值是否符合预期

- 检查`shmem_signal`的`signal`参数是否正确传递。
- 检查CCE文件`output/output_<id>/kernel_aicore/`中signal值是否符合预期，atomicType是否符合预期。
- atomicType是否保证原子操作。
- 必要时增加kernel侧日志。

#### 判定信号写入共享信号区域

目的：通过shmem_get读取共享信号数据

操作步骤：将shmem_get的输出加入算子输出列表：

```python
# 正常的通信流程完成后，读取Win区数据
win_data = pypto.distributed.shmem_get(
    shmem_tensor, my_pe, shmem_shape, [0, 0],
    pred=[wait_until_out], valid_shape=shmem_shape)

# 将win_data加入算子输出列表，用于后续精度对比
return [output, win_data]
```

## 定位技巧

### 缩小问题规模

缩小问题规模旨在简化场景，提高定位效率。缩小后可查看Pass图是否符合预期，分析各算子的连接关系、数据流、切分方式。需确保缩小后仍能复现原始问题。

1. 减小world_size

    将world_size从大值减小到最小可复现值（如world_size = 2）。

2. 减少切块数量或不切块

    增大TileShape减少切块数量，或直接使用不切分的配置。不切分时问题消失，说明切分逻辑有问题；不切分时问题仍存在，说明问题与切分无关。

3. 减小Shape规格

    调小模型的Shape规格（如batch_size、hidden_size），降低计算和通信规模，便于直接对计算结果进行分析。

4. 减小维度

    减少tensor的维度数量（如从3D降到2D），简化数据结构和计算逻辑。维度减少后问题消失，说明问题可能与特定维度的处理相关；维度减少后问题仍存在，说明问题与维度数量无关。

5. 单一算子

    如果是融合算子或在整网中出现超时现象，先根据章节"Task调度状态检查"定位到具体的通信算子，构造单一算子再进行定位。

### 可视化查看task依赖情况

1. leafHash → 前端代码映射

如果问题规模较大且无法缩小问题规模，可以通过项目组提供SKILL，可视化检查task间的依赖关系，映射方法参考 [AGENTS.md](../../../../AGENTS.md) pypto-op-perf-tune目录下leafhash-to-code-mapping.md，使用agent可视化检查task依赖关系和执行情况。

### 偶现超时

1. 插入同步点

偶现性问题定位比较困难，当前遇到的问题大部分和同步相关，可以在 [tileop_shmem.h](../../../../framework/src/interface/tileop/distributed/tileop_shmem.h)涉及写信号的代码插入PIPE_ALL，根据插入的PIPE_ALL二分定位。

## 典型错误场景

### 场景一：HCCL配置错误

场景说明：未真正开始执行通信任务，首报错为HCCL相关，属于非PyPTO问题。

错误日志特征：

```txt
# ASCEND_PROCESS_LOG_PATH/debug/plog日志报错
[ERROR] HCCL(<pid>,python3):YYYY-MM-DD HH:MM:SS.mmm [json_parser.cc:100][<pid>]JSON parse error: ...
[ERROR] HCCL(<pid>,python3):YYYY-MM-DD HH:MM:SS.mmm [hcclCommOp.cc:124] errNo[0x0000000005010001] load allocated resource to json fail.
```

处理方式：首报错为HCCL相关，需先解决HCCL配置或依赖问题，提ISSUE到HCCL仓解决。

### 场景二：AICPU未拉起导致超时

场景说明：未真正开始执行通信任务，AICPU线程未拉起，属于非PyPTO问题。

日志特征：

```txt
# ASCEND_PROCESS_LOG_PATH/debug/device_*日志未搜索到Init aicpu task manager
# shmem_wait_until任务无法正常拉起
```

处理方式：检查CANN/驱动版本，确保AICPU功能正常。

### 场景三：信号值写入同一地址

场景说明：信号内存地址计算错误，信号写入同一块地址。

错误日志特征：

```txt
[DEBUG] PrepareTask "PrepareTask taskId=65558, actualAddr=0x10004ca001a0, actual offset=[1, 4]"
[DEBUG] PrepareTask "PrepareTask taskId=65564, actualAddr=0x10004ca001a0, actual offset=[1, 4]"  # 映射相同地址
[ERROR] Aicpu[1] proc finish: finishedFunctionCnt=29, coreFunctionCnt=30, but timeout !.
```

关键日志：

- `PrepareTask`中存在相同的`actualAddr`，表示信号写入同一块地址
- `finishedFunctionCnt < coreFunctionCnt`表示部分task未Complete

问题定位：

- 对应前端代码检查中的输入一致性检查
- 修复：尾块计算有误导致映射相关地址，修改计算方式

## 常见问题FAQ

### Q1：多轮通信场景下，为什么首轮正常但后续轮次超时？

常见原因：多轮通信下，pypto.AtomicType.ADD场景下，前后轮没有插入同步时，快卡可能会比慢卡快一轮，导致信号区域不符合预期。

| 原因 | 检查方法 |
|------|----------|
| 未执行barrier | 检查每轮通信前是否有`shmem_barrier_all` |

---

## 相关文档

- [分布式API文档](../../api/distributed/index.md)
- [shmem_wait_until API](../../api/distributed/pypto-distributed-shmem_wait_until.md)
- [shmem_signal API](../../api/distributed/pypto-distributed-shmem_signal.md)
- [分布式故障排查](../../trouble_shooting/distributed.md)
- [通信算子切块设置指南](distributed_operation_tiling_guide.md)
- [通信算子精度定位指南](distributed_precision_guide.md)
