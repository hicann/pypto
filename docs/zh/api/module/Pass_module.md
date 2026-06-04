# PyPTO Pass 模块功能总结

## 1. Pass 总体介绍

PyPTO的 Pass 是通过多层中间表示的渐降式编译转化，并允许开发者显式注入优化策略，从而实现人机协同以生成极致性能AI加速器代码的编译工序。这些 Pass 模块按照执行顺序分为三个主要阶段。

### 1.1 Pass 阶段说明

| 阶段 | 主要目标 |
|------|----------|
| Tensor Graph Pass | 对前端构建的张量计算图施以常量折叠、算子融合等硬件解耦优化，剔除冗余计算逻辑、规整图结构，简化高层计算语义，同时完成图结构的语法校验与规范统一，确保化简后的图结构符合后续降级要求，平稳下译为 Tile 图，为后续分层优化奠定基础 |
| Tile Graph Pass | 将张量操作精准分片为严丝合缝匹配本地缓冲区容量与对齐约束的 Tile，规避内存访问冲突；借助双缓冲技术、流水线编排策略及局部内存生命周期精细化管理，充分释放硬件并行计算潜力，平衡计算与内存访问效率，同时完成 Tile 级别的合法性校验，最终将优化后的图结构转换为 Block 图，实现计算粒度向硬件适配的进一步下沉 |
| Block Graph Pass | 将 Tile 计算拆解为贴合多核执行模型的基本块，执行寄存器分配、指令调度与显式数据预取等近硬件优化，优化指令执行顺序、提升资源利用率，减少数据访问延迟；同时完成硬件资源约束校验，确保优化后的基本块适配 AI 加速器多核架构，最终由 CodeGen 模块遣译为可执行指令序列，实现高层计算语义向底层硬件执行指令的精准落地，保障代码执行的极致性能与稳定性 |

---

## 2. Pass 三个阶段介绍

### 2.1 Tensor Graph Pass 阶段

Tensor Graph Pass 阶段处理 Tensor Graph 层面的优化，主要关注：

- **冗余操作消除**: 删除不必要的 Reshape、View 等操作
- **类型转换优化**: 自动插入和优化 Cast 操作
- **内存冲突处理**: 推断和解决内存访问冲突
- **图结构转换**: 将 Tensor Graph 展开为 Tile Graph

### 2.2 Tile Graph Pass 阶段

Tile Graph Pass 阶段处理 Tile Graph 层面的优化，主要关注：

- **操作合并**: 合并 View、Assemble、Reduce Copy 等操作
- **内存优化**: 分配内存类型、复用内存、拆分 tensor
- **图划分**: 将计算图划分为多个子图
- **子图优化**: 子图合并、子图复用、子图转换
- **数据路径优化**: 生成搬运操作、处理边界 tensor
- **形状优化**: 推断动态 shape、对齐 tensor shape

### 2.3 Block Graph Pass 阶段

Block Graph Pass 阶段处理 Block Graph 层面的优化，主要关注：

- **参数管理**: 推断参数索引、处理动态形状
- **内存复用**: 合并源目标 buffer、复用内存
- **调度优化**: 乱序调度、优化操作执行顺序
- **同步管理**: 插入同步操作、优化同步开销
- **代码生成准备**: 内存分配、copy out resolve、代码生成预处理

---

## 3. Pass 详细介绍

### 3.1 Tensor Graph Pass 阶段

| Pass 名称 | 简要描述 | 主要功能 |
|-----------|----------|----------|
| RemoveRedundantReshape | 删除冗余 Reshape 操作 | 识别并删除输入输出形状相同或连续 Reshape 的冗余操作 |
| AutoCast | 自动插入类型转换操作 | 根据 FP16/BF16/INT32 等类型支持，插入必要 Cast 操作并缩短冗余 Cast 链 |
| InferMemoryConflict | 推断内存冲突 | 通过前向和后向传播分析 tensor 内存使用，检测冲突并插入 Copy 操作 |
| RemoveUndrivenView | 删除未被驱动的 View | 为 AssembleSSA 删除未驱动的 View 并降级为 Assemble |
| ExpandFunction | 展开 Tensor Graph | 将高层操作展开为 Tile 操作，是 Tensor Graph 到 Tile Graph 的关键转换 |

### 3.2 Tile Graph Pass 阶段

| Pass 名称 | 简要描述 | 主要功能 |
|-----------|----------|----------|
| MergeViewAssemble | 合并 View 和 Assemble | 将连续的 View/Assemble 合并为一个，减少操作数量 |
| SplitReshape | 拆分 Reshape | 当输入输出存在重叠时，拆分为多个 View 和 Assemble |
| SplitRawTensor | 拆分 RawTensor | 当 LogicalTensor shape 小于 RawTensor shape 时创建新 RawTensor |
| SplitLargeFanoutTensor | 拆分大扇出 tensor | 将多消费者消费的大 tensor 拆分为小 tensor，提高并行度 |
| DuplicateOp | 复制 View 和 GatherIn | 为多消费者创建新操作，避免操作共享 |
| AssignMemoryType | 分配内存类型 | 根据操作需求和硬件限制分配合适内存类型（UB/L1/L0 等） |
| InferDiscontinuousInput | 推断非连续输入 | 从 InCast 前向传播，检测冲突并插入 Copy |
| InsertOpForViewAssemble | 插入 Copy 操作 | 在 View 和 Assemble 间处理内存类型差异 |
| RemoveRedundantOp | 消除冗余操作 | 识别并删除冗余的 View/Assemble/Register Copy |
| ProcessAtomic | 消除 ReduceAcc及OP_ATOMIC_RMW | 优化 K 轴归约，将多个 A_MUL_B 的 CopyOut 直连到 GM，并消除OP_ATOMIC_RMW，把相关模式属性刷新到Copyout上 |
| GraphPartition | 图划分 | 通过同构子图分组和合并算法进行图划分 |
| ReduceCopyMerge | 合并 Reduce Copy | 将连续的 Reduce Acc 合并为一个 |
| NBufferMerge | NBuffer 合并 | 通过着色算法将同构子图分组合并，减少切换开销 |
| L1CopyInReuseMerge | L1 Copy In 复用 | 合并重复的 L1 Copy In 操作 |
| IntraSubgraphAdapter | 适配边界 tensor | 处理跨子图 tensor 传递，插入 ASSEMBLE 和 VIEW |
| GenerateMoveOp | 生成搬运操作 | 将 VIEW/ASSEMBLE/CONVERT 转换为 CopyIn/CopyOut 等搬运操作 |
| CommonOperationEliminate | 消除重复计算 | 通过哈希特征识别相同计算，用一操作替换多个 |
| AxisCombine | 对齐广播输入 | 插入 BRCB 或 EXPAND 确保输入最后一维对齐 |
| PadLocalBuffer | 对齐 tensor shape | 通过 padding 确保 tensor 满足 32B 等硬件对齐要求 |
| RemoveUnalignedReshape | 删除未对齐 Reshape | 移除尾轴非对齐的 reshape，插入 CopyOut/CopyIn |
| ReplaceTensor | 内存复用 | 针对 inplace 操作等进行 tensor 内存复用 |
| PreGraphProcess | 预处理图结构 | 设置子图颜色、边界、Cube 操作属性 |
| InferDynShape | 推断动态 shape | 通过拓扑排序遍历操作并调用 infer shape |
| SubgraphToFunction | 转换子图为函数 | 构建子图调用关系，处理 Incast/Outcast 和符号化 |

### 3.3 Block Graph Pass 阶段

| Pass 名称 | 简要描述 | 主要功能 |
|-----------|----------|----------|
| InferParamIndex | 推断参数索引 | 为子函数推断参数索引，处理动态形状 |
| SrcDstBufferMerge | 合并源目标 buffer | 通过 inplace 语义和 L0 内存复用减少分配 |
| AddAlloc | 添加 Alloc 操作 | 为需要分配内存的 tensor 插入 Alloc |
| OoOSchedule | 乱序调度 | 分析依赖关系，优化执行顺序，提升并行度 |
| TuneTileOpSeqForVF | 优化 TileOp 序列 | 调整 Pipe V 操作执行顺序，优化同步开销 |
| RemoveAlloc | 移除 Alloc | 清理不需要的内存分配操作 |
| CopyOutResolve | 解析 CopyOut | 为 Outcast 插入操作触发 copy out resolve |
| InsertSync | 插入同步操作 | 插入 SetFlag 和 WaitFlag 确保数据依赖正确 |
| TuneSyncForVF | 优化同步操作 | 调整 SetFlag 和 WaitFlag 位置，优化同步开销 |
| MixSubgraphSplit | 拆分 Mix 子图 | 将 Mix 子图拆分为独立的 Cube 和 Vector 子图 |
| GlobalMemoryReuse | 全局内存复用 | 通过 TensorBucket 实现跨操作内存复用 |
| LoopAxesProc | 处理循环轴 | 为 Vector Fusion 操作标记 loopGroup 和 loopAxes |
| CodegenPreproc | 代码生成预处理 | 保存 GM tensor 参数索引、强制 axis 合并 |

---
