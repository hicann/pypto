# PASS 组件错误码

（待补充）

- **范围**：F4-F5XXXX
- 本文档说明 PASS 组件的错误码定义、场景说明与排查建议。
- 补充错误码时，可注明 **关联 Skill**（链接至 [.opencode/skills](../../.opencode/skills) 下对应技能，便于排查时加载）。


 ## 错误码定义与使用说明
 	 
相关错误码的统一定义，参见 `framework\src\passes\pass_utils\pass_error.h` 文件。
---

### 前端传入的错误内容
前端用户自检
 	 
**Tensor相关错误**：
1. TENSOR_NULL_POINTER
## 描述：Tensor或其关联的操作存在空指针引用
行为：
- Tensor的producer为null
- Tensor的consumer为null
- Operation的input tensor为null
- Operation的output tensor为null
- Tensor的消费者中存在null consumer
- Tensor的生产者中存在null producer

2. TENSOR_INVALID_MEMORY_TYPE
## 描述：Tensor的内存类型配置不合法或不匹配

3. TENSOR_SUBGRAPH_BOUNDARY
## 描述：跨子图使用的Tensor未正确标记边界
行为：
- DDR tensor未标记为subgraph boundary
- 跨子图的tensor未标记为subgraph boundary
- Tensor的subgraph id为NOT_IN_SUBGRAPH

4. TENSOR_SHAPE_MISMATCH
## 描述：Tensor的shape配置与操作语义不匹配
行为：
- 特定OP的输入输出tensor shape或者memType不合规

5. TENSOR_UNSUPPORTED_DATATYPE
## 描述：Tensor的数据类型不被操作支持
行为：
- OP与输入输出tensor支持的数据类型不符

6. TENSOR_MEMORY_ALLOCATION
## 描述：Tensor的内存分配配置不合法


7. TENSOR_DYNAMIC_ATTR
## 描述：动态形状相关属性缺失或配置错误
行为：
- OP的动态相关属性缺失
- Tensor的dynValidShape为空

8. TENSOR_MEMORY_CORRUPTION
## 描述：Tensor内存已损坏、内存状态非法或内存数据完整性异常，无法正常参与计算
行为：
- 检测到Tensor内存指针非法、内存越界、内存已释放或数据被篡改

**Operation相关错误**：
1. OP_INVALID_OPERAND_COUNT
## 描述：操作的操作数数量不符合预期

2. OP_NULL_POINTER
## 描述：操作或其属性存在空指针引用
行为：
- Operation为null
- Operation的op attribute为null
- Operation的IOperands或OOperands为null

3. OP_INVALID_OPCODE
## 描述：操作的opcode在当前上下文中不合法
行为：
- OP不合规


4. OP_PRODUCER_CONSUMER
## 描述：操作的输入输出依赖关系不完整
行为：
- OP没有生产者或者消费者

5. OP_SPECIAL_CONSTRAINT
## 描述：特殊操作违反了特定的约束条件
行为：
- 特定OP的生产者消费者OP类型不合规
- 特定OP的to memType类型不合规

6. OP_NESTING_DEPTH
## 描述：特定操作的嵌套深度超过限制
行为：
- 特定OP嵌套深度超过限制

7. OP_SEQUENCE_ERROR
## 描述：操作序列中存在不允许的操作组合
行为：
- 存在不允许的OP或OP组合

**Function相关错误**：
1. FUNCTION_GRAPH_STRUCTURE
## 描述：Function的图结构不完整或不合法
行为：
- Function中存在null operation
- Function的incast为空
- Function的outcast为空
- Function中存在循环依赖
- 子图拓扑结构不正确
- 子图ID超出范围
- 空子图存在

2. FUNCTION_BOUNDARY_COMPLETENESS
## 描述：Function的输入输出边界不完整
行为：
- Incast没有consumer
- Outcast没有producer
- Operation的subgraphID为负数且不是NOP操作

3. FUNCTION_GRAPH_CONNECTION
## 描述：Function的图连接关系不正确
行为：
- 输入输出图不匹配
- 子图边界tensor未正确标记
- 边索引超出operations_ size
- 操作的magic number找不到

4. FUNCTION_EXPAND_FEATURE
## 描述：Function展开功能的状态不正确
行为：
- ExpandFunctionAccelerate标志未重置为false
- 局部定义的临时tensor用作操作输入（没有producer）

5. FUNCTION_MEMORY_REACHABILITY
## 描述：Function中的内存类型转换不可达
行为：
- 特定OP的输入输出memory type不可达
- 输入输出memory type转换路径不存在

6. FUNCTION_UNIQUENESS
描述：Function中存在重复的标识符
行为：
- Operation的magic number重复
- Tensor的magic number重复

7. FUNCTION_SPECIAL_STRUCTURE
## 描述：Function中存在特殊的结构性问题

**Graph相关错误**：
1. GRAPH_LOOP_DETECTION
##描述：图中存在循环依赖
行为：
- OperationLoopCheck失败，存在循环依赖
- LoopCheck失败，存在循环

2. GRAPH_TOPOLOGY_STRUCTURE
## 描述：图的拓扑结构不正确
行为：
- 子图拓扑结构不正确
- 父子图ID关系不正确（parent subGraphId应小于等于subGraphId）
- 边索引超出operations_ size

3. GRAPH_SUBGRAPH_EMPTY
## 描述：存在空的子图
行为：
- 子图为空
- 空子图存在

4. GRAPH_SUBGRAPH_ID_INVALID
## 描述：子图ID配置不合法
行为：
- 子图ID为负数且不是NOP操作
- 子图ID超出totalSubGraphNum范围

5. GRAPH_EDGE_CONSISTENCY
## 描述：图的边连接关系不一致
行为：
- inEdgeGraph和outEdgeGraph大小不匹配
- 节点在inGraph_中的位置超出outGraph_范围
- 节点在inGraph_中但在outGraph_中找不到
- outEdgeGraph中有未被遍历的边

6. GRAPH_COLOR_CONSISTENCY
## 描述：图的着色信息不一致
行为：
- colorInGraph_和colorOutGraph_一致性检查失败
- colorOutGraph_和输入匹配失败
- 原始操作和子图操作之间的边在colorOutGraph_中缺失
- colorOutGraph_中的边在outGraph_中没有对应边

7. GRAPH_READY_STATE
## 描述：图的就绪状态不一致
行为：
- 拓扑结构中就绪状态不一致
- readyState与负的前驱计数不匹配

8. GRAPH_AIV_AIC_MIX
## 描述：子图中混合了不兼容的计算单元
行为：
- 子图中同时存在AIV和AIC操作
- 子图中同时存在UB和L0/L1内存类型tensor

**Config相关错误**：
1. CONFIG_MEMORY_TYPE_REACHABLE
## 描述：内存类型之间不存在可达的转换路径
行为：
- 输入输出内存类型不可达
- 内存类型转换路径不存在

2. CONFIG_SUBGRAPH_BOUNDARY
## 描述：跨子图的Tensor边界标记缺失
行为：
- DDR tensor未标记为子图边界
- 跨子图的tensor未标记为子图边界

3. CONFIG_TENSOR_MEMORY_TYPE
## 描述：Tensor的内存类型配置不合法
行为：
- 内存类型不匹配

**Manager相关错误**：
