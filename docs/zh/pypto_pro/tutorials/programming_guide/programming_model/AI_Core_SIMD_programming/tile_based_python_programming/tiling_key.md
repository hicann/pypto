# TilingKey

本文档介绍 **TilingKey**：用有限个编译期配置为同一份PyPTO Pro kernel源码生成多个专用
实例，并在launch时选择其中一个实例。它用于会改变代码路径、tile模板或数据布局的少数
模式。

本文以
`python/tests/ut/block/frontend/a5/fa/test_fa_perf_tkv_preload_dn_vf_bufid_dynrank.py`
中的FlashAttention为主例，使用`FaTilingKey`为causal attention和full attention
分别生成专用kernel。

源码参考：

- `python/pypto_pro/runtime/tilingkey.py`：schema校验、64-bit打包和合法组合枚举。
- `python/tests/ut/block/frontend/system/tiling_key/`：字段、launch参数及`is_valid`的
  校验用例。
- `python/tests/ut/block/frontend/a5/fa/test_fa_perf_tkv_preload_dn_vf_bufid_dynrank.py`：
  多字段schema、`is_valid`、TilingKey launch和二进制头文件生成的完整示例。

所有示例使用以下导入：

```python
from pypto_pro.runtime.tilingkey import TilingKeyField
import pypto_pro.language as pl
```

---

## . 何时使用TilingKey

TilingKey字段在parser阶段会被折叠为常量。因此每个具体key都编译为独立kernel，而不是
在一个kernel内保留运行时分支。适合以下情况：

- 可选功能会改变较大代码路径，例如是否应用attention mask；
- tile尺寸、layout、dtype模板只有有限候选值；
- 希望在launch前拒绝不支持的字段组合；
- 二进制交付时，需要枚举可编译的tiling key。

不要把任意运行时shape放入TilingKey。候选值的笛卡尔积会决定可枚举组合数量；TilingKey
应只描述有限的、会影响代码生成的模式。

---

## . 声明schema

TilingKey是一个普通Python类。类属性使用`TilingKeyField(bits=..., values=...)`声明：

```python
class AttentionKey:
    # 二进制开关，取值只能为 0 或 1。
    HasAtten = TilingKeyField(bits=1, values=[0, 1])

    # 两种固定 tile 模板。
    BlockM = TilingKeyField(bits=8, values=[64, 128])

    def is_valid(self, key):
        has_atten, block_m = key
        # 仅举例：mask 模式只支持 128 行 tile。
        return has_atten == 0 or block_m == 128
```

字段按**类定义顺序**收集。该顺序同时决定：

1. `is_valid()`中`key` tuple的元素顺序；
2. 64-bit packed key的bit offset；
3. 二进制交付头文件中的字段和selector顺序。

`is_valid`是可选谓词，参数`key`是定义顺序的tuple，而不是dict。它既会在JIT launch时
校验具体key，也会在二进制交付时过滤枚举出的组合。

### 字段约束

框架在`@pl.jit`装饰时立即检查schema：

| 约束 | 说明 |
|---|---|
| `tiling_key` | 必须是class，且至少包含一个`TilingKeyField`。 |
| `bits` | 必须大于0。 |
| `values` | 必须非空、元素必须是互不重复的`int`，不能是`bool`。 |
| 编码容量 | 候选数量不得超过`2**bits`。字段bit保存候选在`values`中的下标，而不是候选值本身。 |
| 总位宽 | 所有字段位宽之和不得超过64。 |
| `is_valid` | 若定义，必须可调用。 |

字段位宽用于为候选**下标**分配编码空间，不限制候选值本身的数值范围。实际值可以是稀疏的
模板编号，例如`bits=3`的候选集合可以是`[16, 64, 128]`；它们分别编码为0、1、2。

### 编码与AscendC对齐

`values`的顺序具有语义：packed key的字段bit是该值在`values`中的下标，解码后才得到
kernel内使用的实际值。这与AscendC模板tiling key的selector一致。

```python
class MaskKey:
    NeedAttnMask = TilingKeyField(bits=1, values=[1, 0])
```

此字段的映射为：

| 实际值 | `values`下标 | 字段bit / tiling key |
|---:|---:|---:|
| `1` | `0` | `0` |
| `0` | `1` | `1` |

因此tiling key为`1`时，`NeedAttnMask`的实际值为`0`。launch时仍传入实际值，例如
`{"NeedAttnMask": 0}`；无需也不应手工传入下标。

---

## . 将schema绑定到kernel

通过`@pl.jit(tiling_key=...)`关联schema。kernel体内直接以字段名引用它们；这些名称不是
kernel参数：

```python
@pl.jit(auto_mutex=True, tiling_key=AttentionKey)
def attention_kernel(
    q: pl.Ptr[pl.DT_FP16],
    k: pl.Ptr[pl.DT_FP16],
    out: pl.Ptr[pl.DT_FP16],
):
    for qi in pl.range(0, 32):
        kv_end = 32
        # HasAtten 在此处是当前 TilingKey 实例对应的编译期常量。
        if HasAtten == 1:
            kv_end = qi + 1
        for ki in pl.range(0, kv_end):
            # ...
            pass
```

例如`HasAtten=1`时，parser保留`kv_end = qi + 1`的分支；`HasAtten=0`时，parser
移除该分支，保留初始值`kv_end = 32`。二者共享源码，却不会在最终kernel中留下对
`HasAtten`的运行时判断。

TilingKey字段名不得与kernel形参或模块级普通变量冲突。字段名应采用描述编译期语义的名称，
例如`HasAtten`、`S1TemplateType`，不要使用`n`、`shape`之类可能与运行时变量冲突的名称。

---

## . 选择具体实例并launch

带TilingKey的kernel必须在方括号launch参数中提供完整的key dict，位置在`stream`和
`block_dim`之后：

```python
key = {"HasAtten": 1, "BlockM": 128}
attention_kernel[None, num_cores, key](q, k, out)
```

dict的字段必须与schema **完全一致**：

- 每个字段都必须出现，且不能有额外字段；
- 值必须属于该字段的`values`；
- 整个组合必须通过`is_valid()`；
- 不能直接调用`attention_kernel(...)`，也不能用list或tuple代替key dict。

若同时使用`datatype`特化，TilingKey仍是第三个参数，datatype dict紧随其后：

```python
attention_kernel[None, num_cores, key, datatype](q, k, out)
```

框架按字段定义顺序把具体dict中的实际值转换为各自的`values`下标，再打包为唯一的
64-bit key，并按这个key缓存相应的专用编译结果。

---

## . FlashAttention用例

指定用例以`FaTilingKey`为`flash_attention_score`生成causal attention和full attention
两种专用实例。其余kernel实参不参与TilingKey schema，也不影响具体key的选择。

### `FaTilingKey`的字段

同一用例的`FaTilingKey`声明14个编译期字段：

| 字段 | bits | 候选值 | 此用例的固定值 |
|---|---:|---|---:|
| `KernelTypeKey` | 2 | 0, 1 | 0 |
| `ImplMode` | 2 | 0, 1, 2 | 0 |
| `Layout` | 4 | 0, 1, 2, 3, 4 | 1 |
| `S1TemplateType` | 10 | 0, 16, 64, 128, 256 | 128 |
| `S2TemplateType` | 10 | 0, 16, 32, 64, 128, 256, 512 | 128 |
| `DTemplateType` | 12 | 0, 16, 32, 48, 64, 80, 96, 128, 160, 192, 256, 768 | 128 |
| `DvTemplateType` | 12 | 同`DTemplateType` | 128 |
| `PseMode` | 4 | 0, 1, 2, 3, 4, 9 | 9 |
| `HasAtten` | 1 | 0, 1 | 0或1 |
| `HasDrop` | 1 | 0, 1 | 0 |
| `HasRope` | 1 | 0, 1 | 0 |
| `OutDtype` | 2 | 0, 1, 2 | 0 |
| `Regbase` | 1 | 0, 1 | 1 |
| `OptionalDn` | 1 | 0, 1 | 0 |

这些字段总计63 bits。该用例的`is_valid()`将除`HasAtten`之外的字段限制为表中的固定值，
所以大量候选值的笛卡尔积最终只保留两个合法key：`HasAtten=0`和`HasAtten=1`。

kernel在Cube和Vector两个循环中都使用`HasAtten`选择causal和full两条专用路径：

```python
causal_skv = skv_tiles
if HasAtten == 1:
    causal_skv = qi + 1
```

`HasAtten=0`的专用kernel只保留full attention的`skv_tiles`路径；`HasAtten=1`的
专用kernel只保留causal attention的`qi + 1`路径。这正是TilingKey的典型用途：源码中
可以写清晰的`if/else`，但每个具体key的最终代码只保留可达分支。

### 两次launch

用例的基础key如下：

```python
base_key = {
    "KernelTypeKey": 0, "ImplMode": 0, "Layout": 1,
    "S1TemplateType": 128, "S2TemplateType": 128,
    "DTemplateType": 128, "DvTemplateType": 128,
    "PseMode": 9, "HasAtten": 0, "HasDrop": 0, "HasRope": 0,
    "OutDtype": 0, "Regbase": 1, "OptionalDn": 0,
}

causal_key = {**base_key, "HasAtten": 1}
flash_attention_score[None, actual_num_cores, causal_key, datatype](
    query, key, value, ...
)

full_key = {**base_key, "HasAtten": 0}
flash_attention_score[None, actual_num_cores, full_key, datatype](
    query, key, value, ...
)
```

两次launch使用相同的大部分key字段，只改变`HasAtten`。测试分别将输出与PyTorch
causal attention和full attention参考实现比较，并覆盖FP16与BF16。

---

## . 二进制交付

对带TilingKey的kernel调用`generate_binary_headers()`，会输出TilingKey头文件：

```python
from pypto_pro.runtime.opc.pypto_compile import generate_binary_headers

binary_dir = generate_binary_headers(flash_attention_score)
```

在FlashAttention用例中，生成的`FaTilingKey_tilingkey.h`包含字段声明及通过`is_valid()`
的key selector。它用`ASCENDC_TPL_ARGS_DECL`描述各字段和允许值，并以
`ASCENDC_TPL_SEL`仅列出合法组合。字段bit选择`values`中对应下标的实际值；因此应尽量
收紧`values`，并在存在字段关联约束时实现`is_valid()`，避免生成无用的二进制实例。

---

## . 常见错误

| 现象 | 原因与处理 |
|---|---|
| 装饰kernel时失败 | 检查`bits > 0`、候选值为互不重复的整数、候选数量不超过`2**bits`，且总位宽不超过64。 |
| launch报字段不匹配 | key dict必须包含所有字段且没有未知字段；字段名大小写必须与class属性一致。 |
| launch报值不在候选集中 | 将实际值加入声明的`values`，或使用已有候选值；不要传入候选下标。 |
| launch被`is_valid`拒绝 | 按字段定义顺序检查传给谓词的组合约束。 |
| kernel内找不到字段名 | 在`@pl.jit(tiling_key=...)`中绑定schema，并避免字段名与kernel参数或模块变量冲突。 |
| 为每个shape都新增key | 仅将真正影响代码生成的有限模式放入TilingKey。 |

---

## . 速查

```python
import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField


class MyKey:
    UseFastPath = TilingKeyField(bits=1, values=[0, 1])

    def is_valid(self, key):
        (use_fast_path,) = key
        return use_fast_path in (0, 1)


@pl.jit(auto_mutex=True, tiling_key=MyKey)
def kernel(x: pl.Ptr[pl.DT_FP16]):
    if UseFastPath == 1:       # 编译期常量
        pass
    for i in pl.range(0, 8):
        pass


kernel[None, 1, {"UseFastPath": 1}](x)
```

用`TilingKey`选择有限的专用实现，从而消除关键模式分支的运行时开销。
