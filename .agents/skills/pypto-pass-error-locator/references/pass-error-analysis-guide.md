# PyPTO Pass 异常分析流程指导

本文档针对 PyPTO Pass 模块常见异常类型，提供不同异常类型的分析流程指导。

---

## 一、参数配置异常

### 日志特征
- `vec_nbuffer_setting`
- `cube_l1_reuse_setting`
- `cube_nbuffer_setting`
- `sg_set_scope`
- 包含pass配置参数名关键字

### 分析流程

#### 步骤 1：获取参数约束
1. 从 `docs/api/config/pypto-set_pass_options.md` 获取指定配置参数的使用说明

#### 步骤 2：检查用户参数配置
1. 检查用户提供的代码，定位到参数配置代码行，获取用户参数配置
2. 检查用户参数配置，对比参数使用说明，检查是否满足参数约束

#### 步骤 3：检查pass业务逻辑
1. 分析pass异常位置代码逻辑，检查约束实现是否正确，约束是否有例外场景

#### 步骤 4：确定根因
1. 如果用户配置参数不满足约束：给出异常的参数配置信息，代码位置，不满足的约束点
2. 如果是pass逻辑问题：给出pass逻辑异常代码位置，不满足的约束点

---

## 二、算子约束违反

### 分析流程

#### 步骤 1：提取关键信息
1. 从日志中提取 opmagic 或 tensor magic
2. 识别 magic 类型（op_magic 或 tensor_magic）

#### 步骤 2：加载计算图
1. 从 `output` 目录下，按时间戳找到最新的计算图输出目录，例如：`output/output_20260324_162232_912961_1605290_C0A8451A`
2. 在该目录下查找异常pass模块的 `.json` 计算图文件，获取文件路径，例如：`output/output_20260402_211910_059229_787054_C0A8451A/Pass_01_AutoCast/After_001_AutoCast_TENSOR_default_loop_1_Unroll1_PATH0_hiddenfunc0_5.json`
3. 使用 `ComputationGraphAnalyzer.load_graph(json_path)` 加载计算图
4. 获取分析器实例

#### 步骤 3：获取算子信息
1. **如果是 op_magic**：
   - 调用 `find_operation_by_magic(opmagic)` 获取 OperationInfo
2. **如果是 tensor_magic**：
   - 调用 `find_producer_of_tensor(tensor_magic)` 获取生产者 op
   - 调用 `find_consumers_of_tensor(tensor_magic)` 获取消费者 op 列表
   - 对生产者和每个消费者分别执行后续分析步骤

#### 步骤 4：定位用户代码
1. 从 OperationInfo 获取 `file` 和 `line` 属性
2. 定位到用户代码的具体位置，确定该位置的算子类型

#### 步骤 5：检查算子约束
1. 从 `docs/api/operation/` 目录下获取对应算子的使用说明文档
2. 检查用户代码中该算子的使用是否满足文档约束
3. 检查pass异常位置对该算子的处理逻辑，是否可能导致pass处理后不满足算子约束

#### 步骤 6：确定根因
1. 如果用户代码中算子使用不满足约束：给出异常的参数配置信息，代码位置，不满足的约束点
2. 如果是pass逻辑问题：给出pass逻辑异常代码位置，代码逻辑的解释说明，经过pass处理后不满足约束的原因

---

## 通用问题分析流程

在没有对应的异常类型指导时，使用该分析流程进行分析。本流程通过日志、计算图与源码的交叉验证，快速定位问题根因。

### 步骤 1：提取日志关键信息
1. **定位日志文件**：从 `$ASCEND_PROCESS_LOG_PATH/debug/plog` 目录下获取最新的 `pypto-*.log` 文件。
2. **解析错误行**：搜索 `[ERROR]` 或 `[WARN]` 级别日志，提取以下关键信息：
   - **Magic ID**：`op_magic`（算子唯一标识）或 `tensor_magic`（张量唯一标识）
   - **Pass 上下文**：触发异常的 Pass 名称、执行阶段（Before/After）、子图 ID（subgraph_id）
   - **错误描述**：具体的错误提示文本（如约束违反、索引越界、内存冲突等）
3. **验证检查点**：
   - [ ] Magic ID 准确提取
   - [ ] Pass 名称与阶段明确
   - [ ] 错误描述完整记录

### 步骤 2：根据关键信息定位计算图异常节点
1. **选择计算图文件**：在 `output/` 目录下按时间戳找到最新输出，定位到异常 Pass 对应的 `Before_*.json` 和 `After_*.json` 文件。
2. **加载计算图分析器**：使用 `ComputationGraphAnalyzer` 工具加载 JSON 文件。
   ```python
   analyzer = ComputationGraphAnalyzer()
   graph = analyzer.load_graph(json_path)
   ```
3. **查询异常节点**：
   - 若日志提供 `op_magic`：调用 `analyzer.find_operation_by_magic(op_magic)` 获取 OperationInfo
   - 若日志提供 `tensor_magic`：调用 `analyzer.find_tensor_by_magic(tensor_magic)` 获取 TensorInfo，并进一步查询其生产者/消费者
4. **验证检查点**：
   - [ ] 节点成功定位
   - [ ] 节点属性（opcode, shape, mem_type, offset 等）准确获取
   - [ ] 节点所属 Function/Subgraph 确认

### 步骤 2.1：定位异常算子对应代码行（仅当异常信息包含 op 信息时执行）
1. **提取代码位置**：从 OperationInfo 中获取 `file` 和 `line` 属性，定位到用户代码的具体位置。
2. **识别算子类型**：根据 OperationInfo 的 `opcode` 字段确定算子类型（如 `ADD`, `MUL`, `VIEW`, `INDEX_OUTCAST` 等）。
3. **查找算子使用文档**：
   - 从 `docs/api/operation/` 目录下查找对应算子的使用说明文档（如 `pypto-add.md`、`pypto-view.md`）
   - 若文档不存在，可搜索 `python/pypto/op/` 目录下的算子实现源码
4. **理解算子用法**：
   - 阅读文档中的**参数说明**：输入输出类型、支持的 dtype、shape 约束
   - 阅读文档中的**约束说明**：broadcast 规则、tile shape 要求、内存层级限制
   - 阅读文档中的**调用示例**：正确使用方式与常见错误写法
5. **对比用户代码**：将用户代码中的算子调用与文档约束逐项对比，识别可能的违规使用。
6. **验证检查点**：
   - [ ] 代码行准确定位（文件路径 + 行号）
   - [ ] 算子类型明确识别
   - [ ] 算子文档获取并阅读完成
   - [ ] 用户代码与文档约束对比完成

### 步骤 3：对比 Pass 前后计算图节点变化
1. **提取节点属性**：分别从 `Before` 和 `After` JSON 中提取同一节点的关键属性。
2. **对比关键字段**：
   - **结构属性**：shape, dtype, format, validshape
   - **内存属性**：mem_type (asis/tobe), mem_id, offset, life_range
   - **连接关系**：ioperands（输入操作数）, ooperands（输出操作数）
3. **结合 Pass 源码分析**：
   - 阅读对应 Pass 的 C++ 实现代码，理解该 Pass 的预期行为（如内存提升、算子融合、张量替换等）
   - 对比预期行为与实际变化，识别**非预期变更**（如 shape 错误修改、mem_type 未提升、连接断裂等）
4. **验证检查点**：
   - [ ] 前后差异点明确列出
   - [ ] Pass 预期行为与实际变更对比完成
   - [ ] 异常变更点精准定位

### 步骤 4：分析异常节点生产者/消费者链路
1. **向上追溯生产者链路（Producer Chain）**：
   - 从异常节点出发，沿 `ioperands` 递归查找数据生产者
   - 检查数据生成逻辑是否符合预期，是否存在非法输入传播
2. **向下追踪消费者链路（Consumer Chain）**：
   - 沿 `ooperands` 查找数据消费者
   - 检查下游算子是否满足输入约束（如 dtype 匹配、shape 兼容、内存层级要求）
3. **识别特殊 Op 链路**：重点关注链路中的特殊操作，如 `VIEW`, `RESHAPE`, `INDEX_OUTCAST`, `ASSEMBLE`, `TRANSPOSE` 等，这些节点常是数据流断裂或约束违反的高发区。
4. **验证检查点**：
   - [ ] 完整数据流链路绘制
   - [ ] 异常传播路径识别
   - [ ] 特殊 Op 约束检查完成

### 步骤 5：综合推断错误原因
1. **构建根因链**：按照以下格式串联证据：
   `现象(日志报错) -> 触发位置(Pass源码行) -> 状态异常(计算图Before/After差异) -> 违反规则(算子约束/内存策略)`
2. **交叉验证**：
   - 将日志错误信息与代码逻辑分支进行匹配
   - 验证计算图实际状态是否满足算子/Pass 的官方约束文档
   - 排除环境干扰、配置错误等外部因素
3. **输出结论与修复建议**：
   - **错误类型**：明确分类（配置错误/代码逻辑缺陷/约束违反/环境问题）
   - **根因定位**：精确到文件、行号、具体逻辑分支
   - **修复方案**：提供可执行的修改建议（如调整参数、修复 Pass 逻辑、修改用户代码写法）
4. **验证检查点**：
   - [ ] 根因链完整闭环
   - [ ] 证据链（日志+源码+计算图）相互支撑
   - [ ] 修复建议具备可操作性

