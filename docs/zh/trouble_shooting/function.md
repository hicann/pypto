# FUNCTION组件错误码

- **范围**：F2-F3XXXX
- 本文档说明FUNCTION组件的错误码定义、场景说明与排查建议。

---

## 错误码定义与使用说明

相关错误码的枚举与码值统一定义在`framework/include/tilefwk/error_code.h`。

其中定义了以下错误码（`FeError`）：

### 通用错误码（0x21001U - 0x21009U）

- **INVALID_OPERATION (0x21001U)**：不允许的操作

- **INVALID_TYPE (0x21002U)**：错误的类型

- **INVALID_VAL (0x21003U)**：无效的值

- **INVALID_PTR (0x21004U)**：无效的指针

- **OUT_OF_RANGE (0x21005U)**：参数超出范围

- **IS_EXIST (0x21006U)**：参数/操作已存在

- **NOT_EXIST (0x21007U)**：参数/操作不存在

- **DYNAMIC_SHAPE_COMPUTE_UNSUPPORTED (0x21008U)**：不支持Shape为动态的Tensor直接参与计算

- **OP_DEPENDENCY_CYCLE (0x21009U)**：算子依赖图存在环，拓扑排序失败

### 文件错误码（0x29001U - 0x29002U）

- **BAD_FD (0x29001U)**：错误的文件描述符状态

- **INVALID_FILE (0x29002U)**：无效的文件内容

### 未知错误码

- **UNKNOWN (0x3FFFFU)**：未知错误

---

## 排查建议

### 通用排查建议

#### 1. 启用详细日志

在遇到FUNCTION组件错误时，可以启用详细日志获取更多信息：

```bash
export ASCEND_GLOBAL_LOG_LEVEL=0 # Debug级别日志
export ASCEND_PROCESS_LOG_PATH=./debug_logs # 指定日志落盘路径
```

#### 2. 开启图编译阶段调试模式开关

Function作为前端，需要根据开发者用法/语法总结出上下文，提供给后续组件使用，比如计算图，当开发者的计算图出问题时，使用该调试开关，可查看Function Dump出来的program.json是否符合预期。开启方法请参考[查看计算图](../tutorials/introduction/quick_start.md#查看计算图)。

---

## 错误码相关示例

### INVALID_OPERATION (0x21001U)

**错误描述：**不允许的操作

**可能原因：**

- 尝试执行不被允许的操作
- 如Tensor二次写入不同数据
- 操作上下文不正确

**解决办法：**

- 检查操作是否在正确的上下文中执行
- 确保操作符合系统约束

---

### INVALID_TYPE (0x21002U)

**错误描述：**错误的类型

**可能原因：**

- 入参类型不匹配
- 使用了接口不支持的数据类型

**解决办法：**

- 检查传入参数的类型是否匹配
- 使用接口支持的类型

---

### INVALID_VAL (0x21003U)

**错误描述：**无效的值

**可能原因：**

- 参数值(shape, offset等维数)不匹配，无法计算
- 参数值不合法

**解决办法：**

- 检查参数格式
- 使用有效的参数值

---

### INVALID_PTR (0x21004U)

**错误描述：**无效的指针

**可能原因：**

- 指针为空
- 指针未正确初始化

**解决办法：**

- 确保指针已正确初始化
- 使用前，检查指针有效性

---

### OUT_OF_RANGE (0x21005U)

**错误描述：**参数超出范围

**可能原因：**

- 索引超出范围
- 参数值超出有效范围

**解决办法：**

- 检查索引范围,使用有效的索引值
- 使用有效范围内的参数值

---

### IS_EXIST (0x21006U)

**错误描述：**参数/操作已存在

**可能原因：**

- 尝试创建已存在的对象
- 对象名称重复

**解决办法：**

- 检查对象是否已存在
- 使用唯一的对象名称

---

### NOT_EXIST (0x21007U)

**错误描述：**参数/操作不存在

**可能原因：**

- 访问不存在的对象
- 对象未正确注册

**解决办法：**

- 检查对象是否存在
- 确保对象已正确注册

---

### DYNAMIC_SHAPE_COMPUTE_UNSUPPORTED (0x21008U)

**错误描述：**不支持Shape为动态的Tensor直接参与计算

**可能原因：**

- 直接使用包含`pypto.DYNAMIC`维度的Tensor作为operation操作数
- 未在`pypto.loop`中通过view切分出静态shape后再进行计算

**解决办法：**

- 对动态维度按固定view大小切分
- 在`pypto.loop`中对切分后的静态shape Tensor进行计算表达

---

### OP_DEPENDENCY_CYCLE (0x21009U)

**错误描述：**算子依赖图存在环，`GetSortedOperations`拓扑排序失败

**可能原因：**

- 同一JIT Function内，对**同一Tensor槽位**先**读**后**写回**（如先`view`/`where`等读取，再`assemble`把读取结果写回同一Tensor），导致依赖边首尾相接，图不再是DAG
```python
# 错误示例 - 先读后写回同一槽位
tile = pypto.view(buf, ...)           # 读buf
pypto.assemble(tile, [i, 0], buf)     # 把tile写回buf → 成环
```
- Python变量改名不改变槽位，框架将Tensor视为整体节点，不区分局部切片

**解决办法：**

- **读写分槽**：读`buf_rd`，写`buf_wr`，结果落到第三个变量
- **变换前置 / 不再写回**：在`assemble`之前完成`where`等变换；或先写后读，写缓冲只assemble、读后不再写回同一缓冲
```python
# 正确示例 - 先写后读，且不再写回
for i in pypto.loop(M, name="i_loop", idx_name="i"):
    pypto.assemble(tile, [i, 0], buf) # 只写buf
out = pypto.mul(buf, scale)           # 只读buf，写入新张量
```
- **拆图**：读写无法分离时，拆成两个`@jit`函数

---

### BAD_FD (0x29001U)

**错误描述：**错误的文件描述符状态

**可能原因：**

- 文件描述符状态错误
- 文件未正确打开或关闭

**解决办法：**

- 检查文件描述符状态
- 确保文件正确打开和关闭

---

### INVALID_FILE (0x29002U)

**错误描述：**无效的文件内容

**可能原因：**

- 文件内容格式错误
- 文件内容不符合预期

**解决办法：**

- 检查文件内容格式
- 使用正确的文件内容
