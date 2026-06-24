# PASS组件错误码说明文档

- 范围：F40000-F44002
- 本文档说明PASS组件的错误码定义、场景说明与排查建议。
- 补充错误码时，可注明关联Skill（链接至.agents/skills下对应技能）。

## 错误码定义与使用说明

相关错误码的枚举与码值统一定义在`framework/include/tilefwk/error_code.h`（PASS侧见TensorErr、OperationErr、FunctionErr、GraphErr、ConfigErr等）。

---

## 前端传入的错误内容

### 前端用户通用排查方法

#### 排查步骤

**步骤1：日志落盘**

export ASCEND_PROCESS_LOG_PATH=$(pwd)/logs/$(date +%Y%m%d%H%M%S)\
export ASCEND_GLOBAL_LOG_LEVEL=0

**步骤2：识别错误码**

根据报错信息中的错误码识别问题类别（如`F40000`、`F41000`）。

**步骤3：对照错误码定义**

根据错误码前缀定位问题归属：

- `F40***` → Tensor定义或属性问题
- `F41***` → Operation定义或连接问题
- `F42***` → Function图结构问题
- `F43***` → Graph拓扑问题
- `F44***` → 配置问题

**步骤4：定位用户配置**

根据错误码定义中的"行为"描述，定位对应的前端配置项。

#### 常见前端问题修复建议

| 问题类型 | 修复方法 |
|---------|---------|
| Tile Shape超限 | 减小tile shape参数值 |
| Shape不匹配 | 调整Tensor shape或更换OP |
| dtype不支持 | 更换为OP支持的dtype |
| 空指针 | 确保Tensor/OP正确创建和连接 |
| 边界标记缺失 | 对跨子图Tensor设置boundary属性 |
| 循环依赖 | 修改计算逻辑，消除数据循环 |
| memType不合法 | 使用框架支持的内存类型路径 |

---

### Tensor相关错误

#### F40000 TENSOR_NULL_POINTER

描述：Tensor或其关联的操作存在空指针引用\
行为：

- Tensor的producer为null
- Tensor的consumer为null
- Operation的input tensor为null
- Operation的output tensor为null
- Tensor的消费者中存在null consumer
- Tensor的生产者中存在null producer

#### F40001 TENSOR_INVALID_MEMORY_TYPE

描述：Tensor的内存类型配置不合法或不匹配\
行为：

- Tensor的内存类型为无效 / 未定义值
- Tensor的内存类型与所在子图 / 计算单元要求不兼容
- 动态形状Tensor使用了不合法的内存类型配置
- 边界Tensor未使用规定的内存类型

#### F40002 TENSOR_SUBGRAPH_BOUNDARY

描述：跨子图使用的Tensor未正确标记边界\
行为：

- DDR tensor未标记为subgraph boundary
- 跨子图的tensor未标记为subgraph boundary
- Tensor的subgraph id为NOT_IN_SUBGRAPH

#### F40003 TENSOR_SHAPE_MISMATCH

描述：Tensor的shape配置与操作语义不匹配\
行为：

- 特定OP的输入输出tensor shape或者memType不合规

#### F40004 TENSOR_UNSUPPORTED_DATATYPE

描述：Tensor的数据类型不被操作支持\
行为：

- OP与输入输出tensor支持的数据类型不符

#### F40005 TENSOR_MEMORY_ALLOCATION

描述：Tensor的内存分配配置不合法\
行为：

- 同一内存区域被多个Tensor非法重叠占用
- 内存段划分不合理导致地址越界
- Tensor内存大小为0或超出合法分配范围
- 动态内存分配属性缺失或配置非法
- Tensor内存对齐方式不符合硬件约束

#### F40006 TENSOR_DYNAMIC_ATTR

描述：动态形状相关属性缺失或配置错误\
行为：

- OP的动态相关属性缺失
- Tensor的dynValidShape为空

---

### Operation相关错误

#### F41000 OP_INVALID_OPERAND_COUNT

描述：OP的输入输出数量不符合预期\
行为：

- OP的实际输入Tensor数不合规
- OP的实际输出Tensor数不合规
- 控制依赖 / 边带输入数量不符合约束

#### F41001 OP_NULL_POINTER

描述：操作或其属性存在空指针引用\
行为：

- Operation为null
- Operation的op attribute为null
- Operation的IOperands或OOperands为null

#### F41002 OP_INVALID_OPCODE

描述：操作的opcode在当前上下文中不合法\
行为：

- OP不合规

#### F41003 OP_PRODUCER_CONSUMER

描述：操作的输入输出依赖关系不完整\
行为：

- OP没有生产者或者消费者

#### F41004 OP_SPECIAL_CONSTRAINT

描述：特殊操作违反了特定的约束条件\
行为：

- 特定OP的生产者消费者OP类型不合规
- 特定OP的to memType类型不合规

#### F41005 OP_NESTING_DEPTH

描述：特定操作的嵌套深度超过限制\
行为：

- 特定OP嵌套深度超过限制

#### F41006 OP_SEQUENCE_ERROR

描述：操作序列中存在不允许的操作组合\
行为：

- 存在不允许的OP或OP组合

---

### Function相关错误

#### F42000 FUNCTION_GRAPH_STRUCTURE

描述：Function的图结构不完整或不合法\
行为：

- Function中存在null operation
- Function的incast为空
- Function的outcast为空
- Function中存在循环依赖
- 子图拓扑结构不正确
- 子图ID超出范围
- 空子图存在

#### F42001 FUNCTION_BOUNDARY_COMPLETENESS

描述：Function的输入输出边界不完整\
行为：

- Incast没有consumer
- Outcast没有producer
- Operation的subgraphID为负数且不是NOP操作

#### F42002 FUNCTION_GRAPH_CONNECTION

描述：Function的图连接关系不正确\
行为：

- 输入输出图不匹配
- 子图边界tensor未正确标记
- 边索引超出operations_ size
- 操作的magic number找不到

#### F42003 FUNCTION_EXPAND_FEATURE

描述：Function展开功能的状态不正确\
行为：

- ExpandFunctionAccelerate标志未重置为false
- 局部定义的临时tensor用作操作输入（没有producer）

#### F42004 FUNCTION_MEMORY_REACHABILITY

描述：Function中的内存类型转换不可达\
行为：

- 特定OP的输入输出memory type不可达
- 输入输出memory type转换路径不存在

#### F42005 FUNCTION_UNIQUENESS

描述：Function中存在重复的标识符\
行为：

- Operation的magic number重复
- Tensor的magic number重复

#### F42006 FUNCTION_SPECIAL_STRUCTURE

描述：Function中存在特殊的结构性问题\
行为：

- 存在不符合拓扑规范的特殊节点连接方式
- 子图嵌套结构不符合框架约束
- Function内存在不允许的特殊算子组合结构

---

### Graph相关错误

#### F43000 GRAPH_LOOP_DETECTION

描述：图中存在循环依赖\
行为：

- OperationLoopCheck失败，存在循环依赖
- LoopCheck失败，存在循环

#### F43001 GRAPH_TOPOLOGY_STRUCTURE

描述：图的拓扑结构不正确\
行为：

- 子图拓扑结构不正确
- 父子图ID关系不正确（parent subGraphId应小于等于subGraphId）
- 边索引超出operations_ size

#### F43002 GRAPH_SUBGRAPH_EMPTY

描述：存在空的子图\
行为：

- 子图为空
- 空子图存在

#### F43003 GRAPH_SUBGRAPH_ID_INVALID

描述：子图ID配置不合法\
行为：

- 子图ID为负数且不是NOP操作
- 子图ID超出totalSubGraphNum范围

#### F43004 GRAPH_EDGE_CONSISTENCY

描述：图的边连接关系不一致\
行为：

- inEdgeGraph和outEdgeGraph大小不匹配
- 节点在inGraph_中的位置超出outGraph_范围
- 节点在inGraph_中但在outGraph_中找不到
- outEdgeGraph中有未被遍历的边

#### F43005 GRAPH_COLOR_CONSISTENCY

描述：图的着色信息不一致\
行为：

- colorInGraph_和colorOutGraph_一致性检查失败
- colorOutGraph_和输入匹配失败
- 原始操作和子图操作之间的边在colorOutGraph_中缺失
- colorOutGraph_中的边在outGraph_中没有对应边

#### F43006 GRAPH_READY_STATE

描述：图的就绪状态不一致\
行为：

- 拓扑结构中就绪状态不一致
- readyState与负的前驱计数不匹配

#### F43007 GRAPH_AIV_AIC_MIX

描述：子图中混合了不兼容的计算单元\
行为：

- 子图中同时存在AIV和AIC操作
- 子图中同时存在UB和L0/L1内存类型tensor

---

### Config相关错误

#### F44000 CONFIG_MEMORY_TYPE_REACHABLE

描述：内存类型之间不存在可达的转换路径\
行为：

- 输入输出内存类型不可达
- 内存类型转换路径不存在

#### F44001 CONFIG_SUBGRAPH_BOUNDARY

描述：跨子图的Tensor边界标记缺失\
行为：

- DDR tensor未标记为子图边界
- 跨子图的tensor未标记为子图边界

#### F44002 CONFIG_TENSOR_MEMORY_TYPE

描述：Tensor的内存类型配置不合法\
行为：

- 内存类型不匹配

#### F44003 CONFIG_TENSOR_MEMORY_TYPE

描述：配置文件读取、解析或加载失败 
 行为： 

 - 配置文件打开失败 
 - 配置文件读取失败 
 - 配置文件中不存在对应的配置项 
 - 配置文件对应tab下不存在指定的配置key 
 - 配置项读取失败，不存在指定的配置信息
