# FUNCTION 组件错误码

- **范围**：F2-F3XXXX
- 本文档说明 FUNCTION 组件的错误码定义、场景说明与排查建议。
---

## 错误码定义与使用说明

相关错误码的统一定义，参见 `framework/src/interface/utils/function_error.h` 文件。

该文件中定义了以下错误码（FError）：

### 通用错误码（0x21001U - 0x21008U）

- **EINTERNAL (0x21001U)**：内部错误

- **INVALID_OPERATION (0x21002U)**：不允许的操作

- **INVALID_TYPE (0x21003U)**：错误的类型

- **INVALID_VAL (0x21004U)**：无效的值

- **INVALID_PTR (0x21005U)**：无效的指针

- **OUT_OF_RANGE (0x21006U)**：参数超出范围

- **IS_EXIST (0x21007U)**：参数/操作已存在

- **NOT_EXIST (0x21008U)**：参数/操作不存在

### 文件错误码（0x29001U - 0x29002U）

- **BAD_FD (0x29001U)**：错误的文件描述符状态

- **INVALID_FILE (0x29002U)**：无效的文件内容

### 未知错误码

- **UNKNOWN (0x3FFFFU)**：未知错误

---

## 排查建议

### 通用排查建议

#### 1. 启用详细日志

在遇到 FUNCTION 组件错误时，可以启用详细日志获取更多信息：

```bash
export ASCEND_GLOBAL_LOG_LEVEL=0 # Debug级别日志
export ASCEND_PROCESS_LOG_PATH=./debug_logs # 指定日志落盘路径
```

#### 2. 开启图编译阶段调试模式开关

Function作为前端，需要根据开发者用法/语法总结出上下文，提供给后续组件使用，比如计算图，当开发者的计算图出问题时，使用该调试开关，可查看Function Dump出来的program.json是否符合预期。

开启方法: [查看计算图.md](../../docs/tools/computation_graph/查看计算图.md)

---

## 错误码相关示例

### EINTERNAL (0x21001U)

**错误描述：** 内部错误

**出现原因：**
- 系统内部发生无法预期的错误
- 内部状态不一致

**解决办法：**
- 检查系统状态
- 联系技术支持

---

### INVALID_OPERATION (0x21002U)

**错误描述：** 不允许的操作

**出现原因：**
- 尝试执行不被允许的操作
- 如Tensor二次写入不同数据
- 操作上下文不正确

**解决办法：**
- 检查操作是否在正确的上下文中执行
- 确保操作符合系统约束

**错误用例：**

```cpp
// 错误示例 - 禁止向有实际数据的Tensor进行不同Tensor的赋值
Tensor lhs(DT_FP32, tshape, "lhs");
Tensor rhs(DT_FP32, tshape, "rhs");

auto ptr1 = std::make_unique<uint8_t>(0);
auto ptr2 = std::make_unique<uint8_t>(0);

lhs.SetData(ptr1.get());
rhs.SetData(ptr2.get());

lhs = rhs;  // 错误使用，lhs里已存在实际Data

// 正确示例
Tensor lhs(DT_FP32, tshape, "lhs");
Tensor rhs(DT_FP32, tshape, "rhs");

auto ptr1 = std::make_unique<uint8_t>(0);
auto ptr2 = std::make_unique<uint8_t>(0);

rhs.SetData(ptr2.get());
lhs = rhs;

```

```python
# 错误示例 - 不在动态函数中
def not_under_dynamic_function_example():
    x = pypto.tensor([4, 4], pypto.DT_INT32)
    y = x[0, 0]  # GetTensorData 需要在动态函数中执行
    return y

# 正确示例
@pypto.frontend.jit
def correct_dynamic_function_example(x):
    y = x[0, 0]  # GetTensorData 在动态函数中执行
    return y
```

---

### INVALID_TYPE (0x21003U)

**错误描述：** 错误的类型

**出现原因：**
- 类型不匹配
- 使用了不支持的数据类型

**解决办法：**
- 检查数据类型
- 使用正确的类型

**示例：**

```python
# 错误示例 - 数据类型不匹配
a = pypto.tensor((4, 4), pypto.DT_INT32)
b = pypto.tensor((4, 4), pypto.DT_FP32)
a[0, 0] = 1.3 # SetTensorData, supports only DT_INT32 tensors
data = b[0, 0] # GetTensorData, supports only DT_INT32 tensors

# 正确示例
a = pypto.tensor((4, 4), pypto.DT_INT32)
b = pypto.tensor((4, 4), pypto.DT_INT32)
a[0, 0] = 1
data = b[0, 0]
```

---

### INVALID_VAL (0x21004U)

**错误描述：** 无效的值

**出现原因：**
- 参数值(shape, offset等)不匹配
- 参数值格式不正确

**解决办法：**
- 检查参数格式
- 使用有效的参数值

**示例：**

```python
# 错误示例 - 无效形状
x = pypto.tensor([-2, 4], pypto.DT_FP32)

# 正确示例
x = pypto.tensor([-1, 4], pypto.DT_FP32)
y = pypto.tensor([4, 4], pypto.DT_FP32)
```

```python
# 错误示例 - shape, offset维度不一致
x = pypto.tensor([8, 8, 16, 16], pypto.DT_FP32)
# y = pypto.view(x, shape, offsets, valid_shape) offset和shape的维数不一致
y = pypto.view(x, [16, 16, 8, 8], [0, 0, 0], [])

# 正确示例
y = pypto.view(x, [16, 16, 8, 8], [0, 0, 0, 0], [])
```

```python
# 错误示例 - shape, offset维度不一致
x = pypto.tensor([8, 8, 16, 16], pypto.DT_FP32)
# y = pypto.view(x, shape, offsets, valid_shape) offset和shape的维数不一致
y = pypto.view(x, [16, 16, 8, 8], [0, 0, 0], [])
# 且 view目标shape必须与tensor(x) 保持维数一致
z = pypto.view(x, [16, 16, 64], [0, 0, 0, 0], [])

# 正确示例
y = pypto.view(x, [16, 16, 8, 8], [0, 0, 0, 0], [])
```

```python
# 错误示例 - offset
x = pypto.tensor([8, 8, 16, 16], pypto.DT_FP32)
# y = pypto.view(x, shape, offsets, valid_shape) offset和shape的维数不一致
y = pypto.view(x, [16, 16, 8, 8], [0, 0, 0], [])

# 正确示例
y = pypto.view(x, [16, 16, 8, 8], [0, 0, 0, 0], [])
```

---

### INVALID_PTR (0x21005U)

**错误描述：** 无效的指针

**出现原因：**
- 指针为空
- 指针未正确初始化

**解决办法：**
- 确保指针已正确初始化
- 检查指针有效性

**示例：**

```cpp
// 错误示例 - Tensor 构造函数中 storage_ 为 nullptr
Tensor tensor(nullptr);  // 触发 FError::INVALID_PTR 错误
```

---

### OUT_OF_RANGE (0x21006U)

**错误描述：** 参数超出范围

**出现原因：**
- 索引超出范围
- 参数值超出有效范围

**解决办法：**
- 检查索引范围, 使用有效的索引值
- 使用有效的参数值

**示例：**

```python
# 错误示例 - 轴索引超出范围
@pypto.frontend.jit
def axis_out_of_range_example(x):
    # x 形状为 [4, 4]，但尝试访问轴 2
    return pypto.sum(x, axis=2)  # 轴 2 超出范围（有效轴为 0, 1）

# 正确示例
@pypto.frontend.jit
def correct_axis_example(x):
    # 访问有效的轴 0 或 1
    return pypto.sum(x, axis=0)  # 轴 0 在范围内
```

```python
# 错误示例 - 视图偏移不匹配
@pypto.frontend.jit
def view_offset_mismatch_example(x):
    # x 形状为 [8, 8]，但偏移 [10, 10] 超出范围
    return pypto.view(x, [4, 4], offset=[10, 10])  # 偏移超出范围

# 正确示例
@pypto.frontend.jit
def correct_view_offset_example(x):
    # 偏移在有效范围内
    return pypto.view(x, [4, 4], offset=[2, 2])  # 正确的偏移
```

```json
// 错误 - 配置值溢出
// C++ 代码调用 SetOptionsNg
config::SetOptionsNg("runtime.device_sched_mode", 4);  // 超出范围 [0, 3]，会报错
config::SetOptionsNg("runtime.stitch_function_num_initial", 129);  // 超出范围 [1, 128]，会报错

// tile_fwk_config_schema.json 中定义了范围
{
    "properties": {
        "runtime": {
            "properties": {
                "device_sched_mode": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 3
                },
                "stitch_function_num_initial": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 128
                }
            }
        }
    }
}
```

---

### IS_EXIST (0x21007U)

**错误描述：** 参数/操作已存在

**出现原因：**
- 尝试创建已存在的对象
- 对象名称重复

**解决办法：**
- 检查对象是否已存在
- 使用唯一的对象名称

**示例：**

```python
# 错误示例 - loop的idx_name重复
for b_idx in pypto.loop(b_scalar, name="LOOP_b", idx_name="b_idx"):
    for s1_idx in pypto.loop(s1_scalar, name="LOOP_s1", idx_name="b_idx"):
       ...

# 正确示例
for b_idx in pypto.loop(b_scalar, name="LOOP_b", idx_name="b_idx"):
    for s1_idx in pypto.loop(s1_scalar, name="LOOP_s1", idx_name="s1_idx"):
       ...
```

```cpp
// 错误示例 - 函数名称重复
// 在静态函数的子函数中使用与当前函数相同的名称
auto &program = npu::tile_fwk::Program::GetInstance();
// 创建一个静态函数
std::string funcName = "test_function";
program.BeginFunction(funcName, npu::tile_fwk::FunctionType::STATIC, 
                        npu::tile_fwk::GraphGraphType::TENSOR_GRAPH, {}, false);
// 再次执行会触发CHECK断言，由于funcName重复
// program.BeginFunction(funcName, npu::tile_fwk::FunctionType::STATIC, 
//                      npu::tile_fwk::GraphType::TENSOR_GRAPH, {}, false);
```

---

### NOT_EXIST (0x21008U)

**错误描述：** 参数/操作不存在

**出现原因：**
- 访问不存在的对象
- 对象未正确注册

**解决办法：**
- 检查对象是否存在
- 确保对象已正确注册

**示例：**

```cpp
// C++ 代码调用 GetAnyConfig
// 错误示例 - 会导致 "key[xx.no_exist] has been not loaded form tile_fwk_config_schema.json." 错误
auto &cm = ConfigManagerNg::GetInstance();
auto scope = cm.CurrentScope();
auto value = AnyCast<int64_t>(scope->GetAnyConfig("xx.no_exist"));

// 正确示例
auto &cm = ConfigManagerNg::GetInstance();
auto scope = cm.CurrentScope();
// 获取的键值必须在tile_fwk_config.json文件中存在
auto value = AnyCast<int64_t>(scope->GetAnyConfig("pass.pg_parallel_lower_bound"));
```

```json
// 错误示例 - 配置字段缺失
// tile_fwk_config_schema.json 中缺少 'type' 或 'properties' 字段
{
    "properties": {
        "pg_parallel_lower_bound": {
            // "type": "integer",
            "label": "...",
        }
    }
}

// 正确示例
{
    "properties": {
        "pg_parallel_lower_bound": {
            "type": "integer",
            "label": "...",
        }
    }
}
```

---

### BAD_FD (0x29001U)

**错误描述：** 错误的文件描述符状态

**出现原因：**
- 文件描述符状态错误
- 文件未正确打开或关闭
- 文件不存在
- 文件正在被使用

**解决办法：**
- 检查文件描述符状态
- 确保文件正确打开和关闭

---

### INVALID_FILE (0x29002U)

**错误描述：** 无效的文件内容

**出现原因：**
- 文件内容格式错误
- 文件内容不符合预期

**解决办法：**
- 检查文件内容格式
- 使用正确的文件内容