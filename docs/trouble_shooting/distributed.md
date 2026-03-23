# DISTRIBUTED 组件错误码

（待补充）

- **范围**：FAXXXX
- 本文档说明 DISTRIBUTED 组件的错误码定义、场景说明与排查建议。

## 错误码定义

相关错误码的统一定义，参见 `framework/src/codegen/utils/distributed_error.h` 文件。

---

## 排查建议

根据日志中不同ErrorCode关联到下述排查建议：

### INVALID_GROUP_NAME
1. **检查通信域名是否为空**：确认传入的 group_name 不为 nullptr，且不为空字符串。
2. **检查通信域名长度范围**：确认传入的 group_name 长度在 [1, 128) 范围内，避免过长或非法长度。

### INVALID_WORLD_SIZE
1. **检查创建通信域**：确保在调用 create_shmem_tensor 时，输入的总进程数不为 0。重复调用 create_shmem_tensor 时，确保传入的 group_name 一致。

### INVALID_TENSOR_DIM
1. **检查张量维度**：确认张量的维度为 2 维，符合要求。

### INVALID_TENSOR_SHAPE
1. **检查张量形状**：确保张量形状为 2 维，且每个维度的大小为正整数，避免出现零维、负维等非法情况。
2. **检查张量形状有效性**：确认每个维度的形状符合预期，满足后续运算需求。

### INVALID_TENSOR_DTYPE
1. **检查张量类型**：确认张量不包含当前硬件不支持的低精度或高精度类型，且数据类型符合预期。

### INVALID_TENSOR_FORMAT
1. **检查张量格式**：确认张量格式为 ND 格式，确保数据格式符合规范。

### INVALID_OPERAND_NUM
1. ***检查输入输出参数个数**：确保传入的输入和输出参数数量与 API 定义一致。

### INVALID_TILE_DIM 
1. ***检查tile的维度**：确定tile的维度为2维。

### INVALID_TILE_SHAPE
1. ***检查tile的维度**：确定tile每个维度的值必须大于0，均为合法有效值。

### INVALID_ALIGNMENT
1. **检查UB buffer的总字节大小**：确保UB buffer的总字节大小是 256 字节的整数倍。

### WIN_SIZE_EXCEED_LIMIT
1. **检查创建的shmem_tensor**： 确认创建的所有 shmem_tensor 的总元素大小小于 1024 * 1024 * 200字节。

### TILE_NUM_EXCEED_LIMIT
1. **检查分块总个数**： 确认在调用 shmem_wait_until 之前，设置的分块总数不超过 1024。

### DIVISION_BY_ZERO
1. **检查分块合理性**：确认各维度分块大小无非法零值。

### AICPU_TASK_TIMEOUT
1. **aicpu等待超时**：确认 shmem_signal 发送的信号能够被 shmem_wait_until 正常接收并等待完成。
2. **查日志上下文**：参考 `docs\trouble_shooting\machine.md` 文件，打开DEBUG日志。

### AICPU_TASK_NUM_EXCEED_LIMIT
1. **检查任务队列大小**：确认 SignalTileOp 队列的任务数未超出最大容量限制。
2. **检查任务数量限制**：确认当前任务数量 (taskCount) 不超过 1024。
3. **查日志上下文**：参考 `docs\trouble_shooting\machine.md` 文件，打开DEBUG日志。

### AICPU_TASK_QUEUE_EMPTY
1. **检查 AICPU 任务队列**：确认任务队列在执行任务前不为空，避免在空队列上执行任务操作。
2. **查日志上下文**：参考 `docs\trouble_shooting\machine.md` 文件，打开DEBUG日志。

### AICPU_TASKID_NOT_IN_MAP
1. **检查任务 ID**：确认给定的 taskId 存在于任务 ID 映射表中，避免任务 ID 查找失败。
2. **查日志上下文**：参考 `docs\trouble_shooting\machine.md` 文件，打开DEBUG日志。

### INVALID_GROUP_INDEX
1. **检查通信域索引**：确认给定的 groupIndex 小于通信域的总数 commGroupNum_，确保索引在有效范围内。
2. **查日志上下文**：参考 `docs\trouble_shooting\machine.md` 文件，打开DEBUG日志。

### NULLPTR
1. **检查运行时管理对象**：确保 AicoreManager等运行时依赖对象已正确初始化并传入。
2. **查日志上下文**：参考 `docs\trouble_shooting\machine.md` 文件，打开DEBUG日志。

