# pypto.distributed 模块介绍

## 概述

`pypto.distributed` 模块提供了分布式场景下的共享内存通信能力，支持多个 PE（Processing Element）之间的数据传输、同步和协同计算。该模块基于对称内存概念设计，实现了高效的跨卡数据交换机制。

## 核心概念：ShmemTensor

### ShmemTensor 的设计理念

ShmemTensor（Shared Memory Tensor）是分布式通信的核心数据结构，与普通 Tensor 有以下关键区别：

- **对称内存访问**：通过指定访问的 PE，可以访问其他卡的 ShmemTensor，实现了跨 PE 的数据共享
- **视图操作支持**：与普通 Tensor 一样，ShmemTensor 支持 View 操作，允许对共享内存张量的部分视图进行操作
- **通信组隔离**：通过 `group_name` 参数实现不同通信域的隔离，支持多个独立的分布式任务

### 与普通 Tensor 的对比

| 特性 | 普通 Tensor | ShmemTensor |
|------|------------|-------------|
| 作用域 | 单 PE 内部 | 跨 PE 共享 |
| 访问方式 | 直接访问 | 通过 PE 编号访问 |
| 视图操作 | 支持 | 支持 |
| 同步机制 | 不需要 | 需要信号同步 |

## 分布式通信设计模式

### 通信模型

`pypto.distributed` 采用基于共享内存的通信模型，主要包含以下操作类型：

1. **数据传输**：通过 `shmem_put` 和 `shmem_get` 实现 PE 间的数据读写
2. **信号同步**：通过 `shmem_signal` 和 `shmem_wait_until` 实现 PE 间的同步通知
3. **视图操作**：通过 `shmem_view` 创建共享内存的部分视图
4. **集合通信**：通过 `shmem_barrier_all` 实现全局同步

### 同步机制

信号同步是分布式通信的关键机制，确保数据的一致性和正确性：

- **信号发送**：使用 `shmem_signal` 向目标 PE 发送信号通知
- **信号等待**：使用 `shmem_wait_until` 等待信号满足指定条件
- **原子操作**：支持 SET（覆盖）和 ADD（累加）两种原子操作类型
- **广播支持**：支持向所有 PE 广播信号

### 典型通信流程

一个典型的分布式通信流程包含以下步骤：

1. 创建 ShmemTensor（数据张量和信号张量）
2. 设置 TileShape（必须步骤）
3. 数据写入（shmem_put）
4. 信号发送（shmem_signal）
5. 等待信号（shmem_wait_until）
6. 数据读取（shmem_get）
7. 可选：全局同步（shmem_barrier_all）

## 应用场景

### AllReduce/AllGather 等集合通信

在分布式训练和推理中，AllReduce 和 AllGather 是常见的集合通信模式：

- **AllReduce**：所有 PE 的数据聚合后分发到所有 PE
- **AllGather**：收集所有 PE 的数据并分发到所有 PE
- **ReduceScatter**：聚合数据后分片分发到不同 PE

### MoE（Mixture of Experts）分布式推理

MoE 模型需要动态路由到不同的专家，分布式通信模块支持：

- 专家路由和分发
- 专家计算结果的聚合
- 多专家协同计算

### 自定义分布式算法

用户可以基于 `pypto.distributed` 实现自定义的分布式算法：

- 自定义通信模式
- 特定应用的数据交换
- 多阶段协同计算

## 使用最佳实践

### TileShape 设置

**重要性**：TileShape 的正确设置是分布式通信正常工作的前提

- **必须设置**：在调用任何 ShmemTensor 相关函数前，必须通过 `set_vec_tile_shapes` 设置 TileShape
- **维度匹配**：TileShape 的维度应与数据张量的维度一致
- **一致性要求**：`shmem_signal` 和 `shmem_wait_until` 的 TileShape 设置必须保持一致

```python
# 正确示例
pypto.set_vec_tile_shapes(16, 64)  # 对应 [m, n] 形状的数据
```

### 依赖关系管理

**重要性**：正确管理操作依赖关系确保数据一致性和执行顺序

- **依赖传递**：通过 `pred` 参数正确传递操作依赖关系
- **操作顺序**：`shmem_get` 通常在 `shmem_wait_until` 之后执行，确保数据已写入
- **流水优化**：在切块数据大于 1 的场景下，保持相同的切块配置以优化流水排布

### 性能优化建议

- **批量传输**：尽量使用较大的数据块进行传输，减少通信次数
- **视图复用**：合理使用 `shmem_view` 避免重复创建共享内存张量
- **信号合并**：在可能的情况下，合并多个信号操作
- **流水并行**：合理设置 TileShape 以优化流水排布

## 使用示例

本节提供的快速入门示例仅用于演示 `pypto.distributed` 模块的基本用法。更详细的使用示例和完整的实现案例，请参考以下文档：

  - [combine_shmem_implementation.md](../../tutorials/distributed/combine_shmem_implementation.md) - 使用通信shmem API实现combine算子

### 基础数据传输

```python
import pypto

# 设置 TileShape（必须步骤）
pypto.set_vec_tile_shapes(16, 64)

# 创建本地数据
local_data = pypto.tensor([16, 64], pypto.DT_FP32, "local_data")

# 创建共享内存张量，形状与 local_data 一致
shmem_tensor = pypto.distributed.create_shmem_tensor(
    group_name="example",
    n_pes=4,
    dtype=pypto.DT_FP32,
    shape=[16, 64]
)

# 将数据写入目标 PE 的共享内存
put_out = pypto.distributed.shmem_put(
    src=local_data,
    offsets=[0, 0],
    dst=shmem_tensor,
    dst_pe=1,
    put_op=pypto.AtomicType.SET,
)

# 从目标 PE 的共享内存读取数据
get_out = pypto.distributed.shmem_get(
    src=shmem_tensor,
    src_pe=1,
    pred=[put_out],
)
```

### 信号同步

```python
import pypto

# 创建信号张量
signal_tensor = pypto.distributed.create_shmem_signal(
    group_name="sync",
    n_pes=4
)

# 设置 TileShape
pypto.set_vec_tile_shapes(32, 64)

# PE 0 发送信号通知 PE 1
signal_out = pypto.distributed.shmem_signal(
    src=signal_tensor,
    src_pe=0,
    signal=1,
    target_pe=1,
    sig_op=pypto.AtomicType.SET,
)

# PE 1 等待信号
wait_out = pypto.distributed.shmem_wait_until(
    src=signal_tensor,
    src_pe=0,
    cmp=pypto.OpType.EQ,
    cmp_value=1,
    clear_signal=True,
    pred=[signal_out],
)
```

## 相关文档

- [《分布式 API 详细文档》](index.md) - 查看各个函数的详细说明和约束条件
- [分布式故障排查](../../trouble_shooting/distributed.md) - 常见问题和解决方案
