# PyPTO Pass模块功能总结

## 1. Pass总体介绍

PyPTO的Pass是通过多层中间表示的渐降式编译转化，并允许开发者显式注入优化策略，从而实现人机协同以生成极致性能AI加速器代码的编译工序。这些Pass模块按照执行顺序分为三个主要阶段。

### 1.1 Pass阶段说明

| 阶段 | 主要目标 |
|------|----------|
| Tensor Graph Pass | 对前端构建的张量计算图施以常量折叠、算子融合等硬件解耦优化，剔除冗余计算逻辑、规整图结构，简化高层计算语义，同时完成图结构的语法校验与规范统一，确保化简后的图结构符合后续降级要求，平稳下译为Tile图，为后续分层优化奠定基础 |
| Tile Graph Pass | 将张量操作精准分片为严丝合缝匹配本地缓冲区容量与对齐约束的Tile，规避内存访问冲突；借助双缓冲技术、流水线编排策略及局部内存生命周期精细化管理，充分释放硬件并行计算潜力，平衡计算与内存访问效率，同时完成Tile级别的合法性校验，最终将优化后的图结构转换为Block图，实现计算粒度向硬件适配的进一步下沉 |
| Block Graph Pass | 将Tile计算拆解为贴合多核执行模型的基本块，执行寄存器分配、指令调度与显式数据预取等近硬件优化，优化指令执行顺序、提升资源利用率，减少数据访问延迟；同时完成硬件资源约束校验，确保优化后的基本块适配AI加速器多核架构，最终由CodeGen模块遣译为可执行指令序列，实现高层计算语义向底层硬件执行指令的精准落地，保障代码执行的极致性能与稳定性 |

---

## 2. Pass三个阶段介绍

### 2.1 Tensor Graph Pass阶段

Tensor Graph Pass阶段处理Tensor Graph层面的优化，主要关注：

- **冗余操作消除**: 删除不必要的Reshape、View等操作
- **类型转换优化**: 自动插入和优化Cast操作
- **内存冲突处理**: 推断和解决内存访问冲突
- **图结构转换**: 将Tensor Graph展开为Tile Graph

### 2.2 Tile Graph Pass阶段

Tile Graph Pass阶段处理Tile Graph层面的优化，主要关注：

- **操作合并**: 合并View、Assemble、Reduce Copy等操作
- **内存优化**: 分配内存类型、复用内存、拆分tensor
- **图划分**: 将计算图划分为多个子图
- **子图优化**: 子图合并、子图复用、子图转换
- **数据路径优化**: 生成搬运操作、处理边界tensor
- **形状优化**: 推断动态shape、对齐tensor shape

### 2.3 Block Graph Pass阶段

Block Graph Pass阶段处理Block Graph层面的优化，主要关注：

- **参数管理**: 推断参数索引、处理动态形状
- **内存复用**: 合并源目标buffer、复用内存
- **调度优化**: 乱序调度、优化操作执行顺序
- **同步管理**: 插入同步操作、优化同步开销
- **代码生成准备**: 内存分配、copy out resolve、代码生成预处理

---

## 3. Pass详细介绍

### 3.1 Tensor Graph Pass阶段

| Pass名称 | 简要描述 | 主要功能 |
|-----------|----------|----------|
| RemoveRedundantReshape | 删除冗余Reshape操作 | 识别并删除输入输出形状相同或连续Reshape的冗余操作 |
| AutoCast | 自动插入类型转换操作 | 根据FP16/BF16/INT32等类型支持，插入必要Cast操作并缩短冗余Cast链 |
| InferMemoryConflict | 推断内存冲突 | 通过前向和后向传播分析tensor内存使用，检测冲突并插入Copy操作 |
| RemoveUndrivenView | 删除未被驱动的View | 为AssembleSSA删除未驱动的View并降级为Assemble |
| ExpandFunction | 展开Tensor Graph | 将高层操作展开为Tile操作，是Tensor Graph到Tile Graph的关键转换 |

### 3.2 Tile Graph Pass阶段

| Pass名称 | 简要描述 | 主要功能 |
|-----------|----------|----------|
| MergeViewAssemble | 合并View和Assemble | 将连续的View/Assemble合并为一个，减少操作数量 |
| SplitReshape | 拆分Reshape | 当输入输出存在重叠时，拆分为多个View和Assemble |
| SplitRawTensor | 拆分RawTensor | 当LogicalTensor shape小于RawTensor shape时创建新RawTensor |
| SplitLargeFanoutTensor | 拆分大扇出tensor | 将多消费者消费的大tensor拆分为小tensor，提高并行度 |
| DuplicateOp | 复制View和GatherIn | 为多消费者创建新操作，避免操作共享 |
| AssignMemoryType | 分配内存类型 | 根据操作需求和硬件限制分配合适内存类型（UB/L1/L0等） |
| InferDiscontinuousInput | 推断非连续输入 | 从InCast前向传播，检测冲突并插入Copy |
| InsertOpForViewAssemble | 插入Copy操作 | 在View和Assemble间处理内存类型差异 |
| RemoveRedundantOp | 消除冗余操作 | 识别并删除冗余的View/Assemble/Register Copy |
| ProcessAtomic | 消除ReduceAcc及OP_ATOMIC_RMW | 优化K轴归约，将多个A_MUL_B的CopyOut直连到GM，并消除OP_ATOMIC_RMW，把相关模式属性刷新到Copyout上 |
| GraphPartition | 图划分 | 通过同构子图分组和合并算法进行图划分 |
| ReduceCopyMerge | 合并Reduce Copy | 将连续的Reduce Acc合并为一个 |
| NBufferMerge | NBuffer合并 | 通过着色算法将同构子图分组合并，减少切换开销 |
| L1CopyInReuseMerge | L1 Copy In复用 | 合并重复的L1 Copy In操作 |
| IntraSubgraphAdapter | 适配边界tensor | 处理跨子图tensor传递，插入ASSEMBLE和VIEW |
| GenerateMoveOp | 生成搬运操作 | 将VIEW/ASSEMBLE/CONVERT转换为CopyIn/CopyOut等搬运操作 |
| CommonOperationEliminate | 消除重复计算 | 通过哈希特征识别相同计算，用一操作替换多个 |
| AxisCombine | 对齐广播输入 | 插入BRCB或EXPAND确保输入最后一维对齐 |
| PadLocalBuffer | 对齐tensor shape | 通过padding确保tensor满足32B等硬件对齐要求 |
| RemoveUnalignedReshape | 删除未对齐Reshape | 移除尾轴非对齐的reshape，插入CopyOut/CopyIn |
| ReplaceTensor | 内存复用 | 针对inplace操作等进行tensor内存复用 |
| PreGraphProcess | 预处理图结构 | 设置子图颜色、边界、Cube操作属性 |
| InferDynShape | 推断动态shape | 通过拓扑排序遍历操作并调用infer shape |
| SubgraphToFunction | 转换子图为函数 | 构建子图调用关系，处理Incast/Outcast和符号化 |

### 3.3 Block Graph Pass阶段

| Pass名称 | 简要描述 | 主要功能 |
|-----------|----------|----------|
| InferParamIndex | 推断参数索引 | 为子函数推断参数索引，处理动态形状 |
| SrcDstBufferMerge | 合并源目标buffer | 通过inplace语义和L0内存复用减少分配 |
| AddAlloc | 添加Alloc操作 | 为需要分配内存的tensor插入Alloc |
| OoOSchedule | 乱序调度 | 分析依赖关系，优化执行顺序，提升并行度 |
| TuneTileOpSeqForVF | 优化TileOp序列 | 调整Pipe V操作执行顺序，优化同步开销 |
| RemoveAlloc | 移除Alloc | 清理不需要的内存分配操作 |
| CopyOutResolve | 解析CopyOut | 为Outcast插入操作触发copy out resolve |
| InsertSync | 插入同步操作 | 插入SetFlag和WaitFlag确保数据依赖正确 |
| TuneSyncForVF | 优化同步操作 | 调整SetFlag和WaitFlag位置，优化同步开销 |
| MixSubgraphSplit | 拆分Mix子图 | 将Mix子图拆分为独立的Cube和Vector子图 |
| GlobalMemoryReuse | 全局内存复用 | 通过TensorBucket实现跨操作内存复用 |
| LoopAxesProc | 处理循环轴 | 为Vector Fusion操作标记loopGroup和loopAxes |
| CodegenPreproc | 代码生成预处理 | 保存GM tensor参数索引、强制axis合并 |

---
