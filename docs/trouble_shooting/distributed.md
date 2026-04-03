# DISTRIBUTED 组件错误码

- **范围**：0xAXXXX
- 本文档说明 DISTRIBUTED 组件的错误码定义、场景说明与排查建议。

## 错误码定义

相关错误码的统一定义，参见 `framework/src/interface/utils/distributed_error.h` 文件。

---

## 排查建议

根据日志中不同ErrorCode关联到下述排查建议：

### 参数错误（0xA0000 - 0xA0009）

#### 0xA0000 INVALID_GROUP_NAME

1. **检查通信域名是否为空**：确认传入的 group_name 不为 nullptr，且不为空字符串。
2. **检查通信域名长度范围**：确认传入的 group_name 长度在 [1, 128) 范围内，避免过长或非法长度。

#### 0xA0001 INVALID_WORLD_SIZE

1. **检查创建通信域**：确保在调用 create_shmem_tensor 时，输入的总进程数不为 0。重复调用 create_shmem_tensor 时，确保传入的 group_name 一致。

#### 0xA0002 INVALID_TENSOR_DIM

1. **检查张量维度**：确认张量的维度为 2 维，符合要求。

#### 0xA0003 INVALID_TENSOR_SHAPE

1. **检查张量形状**：确保张量形状为 2 维，且每个维度的大小为正整数，避免出现零维、负维等非法情况。
2. **检查张量形状有效性**：确认每个维度的形状符合预期，满足后续运算需求。

#### 0xA0004 INVALID_TENSOR_DTYPE

1. **检查张量类型**：确认张量不包含当前硬件不支持的低精度或高精度类型，且数据类型符合预期。

#### 0xA0005 INVALID_TENSOR_FORMAT

1. **检查张量格式**：确认张量格式为 ND 格式，确保数据格式符合规范。

### INVALID_SHMEM_TENSOR

1. **检查输入Shmem Tensor**：请根据报错信息确定原因，可能原因是ShmemTensor中没有合法的data或者signal Tensor。

### INVALID_SHMEM_VIEW_PARAM

1. **检查ShmemView接口参数**：请根据报错信息确定原因，可能原因是ShmemView传入shape或者offset信息不合法。

### INVALID_OP_TYPE

1. **检查ShmemWaitUntil**：请直接根据报错信息确定原因，可能原因是ShmemWaitUntil接口传入了不支持的比较类型。

#### 0xA0006 INVALID_SHMEM_TENSOR

1. **检查输入Shmem Tensor**：请根据报错信息确定原因，可能原因是ShmemTensor中没有合法的data或者signal Tensor。

#### 0xA0007 INVALID_SHMEM_VIEW_PARAM

1. **检查ShmemView接口参数**：请根据报错信息确定原因，可能原因是ShmemView传入shape或者offset信息不合法。

#### 0xA0008 INVALID_OP_TYPE

1. **检查ShmemWaitUntil**：请直接根据报错信息确定原因，可能原因是ShmemWaitUntil接口传入了不支持的比较类型。

#### 0xA0009 INVALID_OPERAND_NUM

1. **检查输入输出参数个数**：确保传入的输入和输出参数数量与 API 定义一致。

### 配置错误（0xA1000-0xA1002）

#### 0xA1000 INVALID_TILE_DIM

1. **检查tile的维度**：确定tile的维度为2维。

#### 0xA1001 INVALID_TILE_SHAPE

1. **检查tile的维度**：确定tile每个维度的值必须大于0，均为合法有效值。

#### 0xA1002 INVALID_ALIGNMENT

1. **检查UB buffer的总字节大小**：确保UB buffer的总字节大小是 256 字节的整数倍。

### 运行时错误（0xA2000 - 0xA2002）

#### 0xA2000 WIN_SIZE_EXCEED_LIMIT

1. **检查创建的shmem_tensor**： 确认创建的所有 shmem_tensor 的总元素大小小于 1024 * 1024 * 200字节。

#### 0xA2001 TILE_NUM_EXCEED_LIMIT

1. **检查分块总个数**： 确认在调用 shmem_wait_until 之前，设置的分块总数不超过 1024。

#### 0xA2002 DIVISION_BY_ZERO

1. **检查分块合理性**：确认各维度分块大小无非法零值。

### machine 相关的错误（0xA3000 - 0xA3005）

#### 0xA3000 AICPU_TASK_TIMEOUT

1. **aicpu等待超时**：确认 shmem_signal 发送的信号能够被 shmem_wait_until 正常接收并等待完成。
2. **查日志上下文**：参考 `docs\trouble_shooting\machine.md` 文件，打开DEBUG日志。

#### 0xA3001 AICPU_TASK_NUM_EXCEED_LIMIT

1. **检查任务队列大小**：确认 SignalTileOp 队列的任务数未超出最大容量限制。
2. **检查任务数量限制**：确认当前任务数量 (taskCount) 不超过 1024。
3. **查日志上下文**：参考 `docs\trouble_shooting\machine.md` 文件，打开DEBUG日志。

#### 0xA3002 AICPU_TASK_QUEUE_EMPTY

1. **检查 AICPU 任务队列**：确认任务队列在执行任务前不为空，避免在空队列上执行任务操作。
2. **查日志上下文**：参考 `docs\trouble_shooting\machine.md` 文件，打开DEBUG日志。

#### 0xA3003 AICPU_TASKID_NOT_IN_MAP

1. **检查任务 ID**：确认给定的 taskId 存在于任务 ID 映射表中，避免任务 ID 查找失败。
2. **查日志上下文**：参考 `docs\trouble_shooting\machine.md` 文件，打开DEBUG日志。

#### 0xA3004 INVALID_GROUP_INDEX

1. **检查通信域索引**：确认给定的 groupIndex 小于通信域的总数 commGroupNum_，确保索引在有效范围内。
2. **查日志上下文**：参考 `docs\trouble_shooting\machine.md` 文件，打开DEBUG日志。

#### 0xA3005 NULLPTR

1. **检查运行时管理对象**：确保 AicoreManager等运行时依赖对象已正确初始化并传入。
2. **查日志上下文**：参考 `docs\trouble_shooting\machine.md` 文件，打开DEBUG日志。
