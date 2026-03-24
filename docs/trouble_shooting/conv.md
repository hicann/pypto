# CONV 组件错误码

（待补充）

- **范围**：FC6XXX - FC8XXX
- 本文档说明 CONV 子类 OP 的错误码定义、场景说明与排查建议。

## 错误码定义与使用说明

相关错误码的统一定义，参见 `framework/src/interface/utils/conv_error.h` 文件。
---

## 错误码定义和场景说明

### 1. Operation（Operation非法拦截类报错，`ConvError::Operation`，FC61xxx）

| 场景枚举 | 错误码 | 报错阶段 | 场景说明 |
|---------|------|----------------------------------|----------|
| `INPUT_INVALID` | **FC6101** | `conv.operation.checkinput` | Operation校验输入参数不合法（维度，shape，数据类型等）。 |
| `OVER_BUFFER_LIMIT` | **FC6102** | `conv.operation.checkweight` | Operation校验超出空间限制不合法。 |
| `UNKNOWN` | **FC6199** | `conv.operation.reserved` | Operation阶段未知报错预留错误码。 |

### 2. Tile切分（Tile图切分，`ConvError::ExpandFunction`，FC62xxx）

| 场景枚举 | 错误码 | 报错阶段 | 场景说明 |
|---------|------|----------|------|
| `EXPANDFUNC_TENSOR_OP_NULLPTR` | **FC6201** | `conv.expandfunc.tensor_nullptr` | Tile图切分，tensor图处理节点空指针报错。 |
| `EXPANDFUNC_TENSOR_ATTR_GET_FAILED` | **FC6202** | `conv.expandfunc.get_attr` | Tile图切分，tensor图节点属性获取失败。 |
| `EXPANDFUNC_TILE_OP_NULLPTR` | **FC6203** | `conv.expandfunc.tile_nullptr` | Tile图切分，tile图新生成节点空指针报错。 |
| `EXPANDFUNC_PARAMS_INVALID` | **FC6204** | `conv.expandfunc.params_check` | Tile图切分，参数不匹配错误（维度，类型，Tile块配置）。 |
| `EXPANDFUNC_INNER_STATUS_FAILED` | **FC6205** | `conv.expandfunc.check_status` | Tile图切分，内部功能函数返回值异常错误。 |
| `UNKNOWN` | **FC6299** | `conv.operation.reserved` | ExpandFunc Tile图切分阶段未知报错预留错误码。 |

### 3. CodenGen

| 场景枚举 | 错误码 | 报错阶段 | 场景说明 |
|---------|------|----------|------|
| `CODEGEN_GET_ATTR_FAILED` | **FC6301** | `conv.codegen.get_attr` | Codegen代码生成，tensor图节点属性获取失败。 |
| `CODEGEN_CHECK_ATTR_INVALID` | **FC6302** | `conv.codegen.check_attr` | Codegen代码生成，tensor图节点属性校验非法。 |
| `CODEGEN_CHECK_DIM_INVALID` | **FC6303** | `conv.codegen.check_dim` | TCodegen代码生成，shape/offset校验dim非法。 |
| `UNKNOWN` | **FC6399** | `conv.operation.reserved` | Codegen代码生成阶段未知报错预留错误码。 |

### 4. TileOp

| 场景枚举 | 错误码 | 报错阶段 | 场景说明 |
|---------|------|----------|------|
| `TILEOP_TENSOR_FORMAT_FAILED` | **FC6401** | `conv.tileop.check_tensor_format` | TileOp，tensor硬件FORMAT校验失败。 |
| `TILEOP_SHAPE_SIZE_FAILED` | **FC6402** | `conv.tileop.check_shape_size` | TileOp，shape size校验失败。 |
| `TILEOP_STC_SHAPE_INVALID` | **FC6403** | `conv.tileop.check_stc_shape` | TileOp，static shape非法。 |
| `TILEOP_INDEX_INVALID` | **FC6404** | `conv.tileop.check_index` | TileOp，获取shape/stride的index校验非法。 |
| `UNKNOWN` | **FC6499** | `conv.operation.reserved` | TileOp未知报错预留错误码。 |

---

## 排查建议

### Operation shape/TileShape 拦截编译报错
可根据报错参考约束说明：`docs/api/config/pypto-set_conv_tile_shapes.md`


### Pass 图阶段 拦截编译报错
1. 打开编译debug模式，dump pass阶段图，配置 `debug_options={"compile_debug_mode": 1}`
```python
@pypto.frontend.jit(debug_options={"compile_debug_mode": 1})
def conv_kernel()
```

2. 复跑问题用例，在output下生成对应时间戳的dump结果，根据报错日志所示图阶段，使用pto-toolkit打开，查看执行图阶段之前的dump图，对conv operation 切成的 Tile子图进行排查；

