# Tiling结果传输

## TilingData

本节介绍TilingData的声明、传入和字段访问方法。TilingData用于向已编译Kernel传递
shape、stride、循环边界、算子选择器和缩放系数等运行时参数，无需将具体取值固化在
Kernel签名中。TilingData既支持标量字段，也支持定长数组字段。

示例代码使用以下导入：

```python
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch_npu
import pypto_pro.language as pl
```

---

### TilingData类

TilingData类是一个普通的Python `@dataclass`，其字段**全部**为以下之一：

- `int`  —— 标量整数  （降低为`INDEX` / `int64_t`）
- `float`—— 标量浮点  （降低为`FP32`）
- `bool` —— 标量布尔  （降低为`BOOL`）
- `T[N]` —— 定长数组，含`N`个`T`类型元素（`T` ∈ `{int, float, bool}`）

TilingData类至少包含一个字段，并且每个字段均使用上述类型之一进行标注。

字段支持Python `dataclass`默认值。框架序列化实例中的全部字段值，未显式传入的字段
使用`dataclass`默认值。设备侧结构体包含全部字段。数组字段需要使用
`dataclasses.field(default_factory=...)`提供默认列表，
且带默认值的字段必须位于无默认值字段之后。

将TilingData实例传给JIT Kernel时，框架会：

1. 根据字段标注推导C `struct`布局；
2. 按照该布局序列化实例，并传递给设备侧代码。

字段类型与C struct成员一一对应（采用原生ctypes对齐），因此Python侧与生成的CCE
struct在`sizeof`与字段偏移上保持一致。

```python
TS = 128       # Q方向的Tile大小，编译期常量
TKV = 128      # K/V方向的Tile大小，编译期常量


@dataclass
class OpTiling:
    sq: int       # 标量
    skv: int      # 标量
    d: int        # 标量
```

上述代码定义了一个包含三个整型标量字段的TilingData类。

---

### 在Kernel签名中声明TilingData

在Kernel函数形参末尾声明TilingData类型参数。该参数按运行时结构体传递，可与裸指针
输入（[`pypto_pro.language.Ptr[dtype]`](../../../../../api/pro_api/SIMD-API/basic_data_structures/Ptr.md)）配合，使用TilingData中的shape重建固定rank的Tensor视图：

```python
@dataclass
class OpTiling:
    sq: int
    skv: int
    d: int

@pl.jit(auto_mutex=True)
def fa_kernel(
    q: pl.Ptr[pl.DT_FP16],
    k: pl.Ptr[pl.DT_FP16],
    v: pl.Ptr[pl.DT_FP16],
    o: pl.Ptr[pl.DT_FP16],
    tiling: OpTiling,          # <-- TilingData 参数
):
    # 重建二维带类型视图；两个维度的运行时取值来自 tiling，而非函数签名。
    tensor_q = pl.make_tensor(q, [tiling.sq,  tiling.d])
    tensor_k = pl.make_tensor(k, [tiling.skv, tiling.d])
    tensor_v = pl.make_tensor(v, [tiling.skv, tiling.d])
    tensor_o = pl.make_tensor(o, [tiling.sq,  tiling.d])

    sq_dim    = tiling.sq
    skv_dim   = tiling.skv
    sq_tiles  = (sq_dim  + (TS  - 1)) // TS
    skv_tiles = (skv_dim + (TKV - 1)) // TKV
    # Kernel中将tiling字段作为普通运行时标量使用
```

上述示例使用[`pypto_pro.language.make_tensor`](../../../../../api/pro_api/SIMD-API/resource_management/make_tensor.md)从裸指针构造固定rank的Tensor视图。也可以保留带类型的[`pypto_pro.language.Tensor`](../../../../../api/pro_api/SIMD-API/basic_data_structures/Tensor.md)输入，并仅使用TilingData传递循环边界、标志和缩放系数。
两种用法相互独立。

> [!NOTE]说明
> 当前每次Kernel调用仅支持一个TilingData实例，且该实例必须位于Kernel形参列表和
> 启动实参列表的末尾。

---

### 在Kernel中读取TilingData字段

在Kernel中，TilingData字段是普通的运行时值：

| 访问形式             | 含义                                                |
|----------------------|-----------------------------------------------------|
| `tiling.field`       | 读取标量字段（int / float / bool）。               |
| `tiling.arr[k]`      | 读取`T[N]`字段的第`k`个元素。           |

它们可用于：

- `pypto_pro.language.make_tensor`的shape和stride实参，
- `pypto_pro.language.range(...)`的循环边界，
- 算术运算（`(tiling.sq + TS - 1) // TS`），
- 运行时条件分支（`if tiling.opkind[4] == 0:`）。

```python
# 标量字段用作循环边界
for kv in pl.range(0, skv_tiles, 1):
    ...

# 数组元素用于在运行时选择算子
if tiling.opkind[4] == 0:
    pl.add(tile_c, tile_a, tile_b)
elif tiling.opkind[4] == 1:
    pl.sub(tile_c, tile_a, tile_b)
else:
    pl.mul(tile_c, tile_a, tile_b)
```

---

### 在启动时构造并传入TilingData实例

在Host侧构造dataclass，并通过方括号启动语法作为对应参数传入。标量字段可直接传值：

```python
tiling = OpTiling(sq=8192, skv=8192, d=128)
fa_kernel[None, num_cores](q_t, k_t, v_t, o_t, tiling)
torch.npu.synchronize()
```

框架自动将`tiling`序列化为C struct字节并管理相应字节缓冲。

![PyPTO Pro TilingData从Host实例到Kernel字段访问的传递流程](../../../../figures/pro/pro_tilingdata_host_kernel_flow.png)

运行时按照字段定义顺序和原生ctypes对齐规则生成C struct字节，再把它包装为设备侧
`uint8` 缓冲区传给Kernel。生成的CCE结构体与Host侧在`sizeof`和字段偏移上保持一致。

---

### 数组字段

数组字段持有固定数量的同质元素。用`T[N]`声明字段，用Python原生`list`
保存运行时值。包含`T[N]`的模块必须启用`from __future__ import annotations`，
避免Python在定义class时求值`int[N]`。

#### 声明数组字段

```python
@dataclass
class OpTiling:
    offsets: int[4]     # 4 个 int
    scales:  float[2]   # 2 个 float
    opkind:  int[8]     # 8 个 int
```

数组字段声明需满足以下规则：

- 元素类型必须为`int`、`float`或`bool`；
- 数组长度`N`必须直接写成1～2048的整数值（例如`int[4]`），不能使用变量、
  算术表达式或布尔值。

`int[N]`是PyPTO Pro DSL注解，不是标准Python类型；部分静态类型检查器可能提示
`int`不可下标，但PyPTO Pro会从延迟注解字符串中安全解析该字段。

#### 构造数组值

`T[N]`只描述字段，不负责构造值。使用原生列表即可：

```python
[0] * 4                         # -> [0, 0, 0, 0]
[0, 1, 2]                       # 直接给出元素
[i for i in range(60)]          # 由迭代生成

arr = [0] * 8                  # 可变：支持下标赋值
arr[4] = 1                     # 设置第 4 个元素
```

`list`支持`arr[i]`读/写、`len(arr)`与迭代。长度必须与声明的大小一致，
否则序列化会抛出`ValueError`。

---

### 完整、可执行示例 —— 运行时shape与尾块处理

以下示例使用两个定长数组：`shape`用于将Host侧2～4维输入的shape折叠为Kernel中的二维Tensor视图，
`opkind[4]`用于在运行时选择加、减或乘。示例同时使用`valid_shape`处理不能被
`128 × 128`整除的尾块。

以`shape=[1, 1, 513, 511]`、`opkind[4]=1`为例：同一份TilingData同时决定Tensor
视图、Tile数量、尾块有效形状和算子分支，这些值均在运行时生效。

```python
from __future__ import annotations

import os

from dataclasses import dataclass

import logging
import torch
import torch_npu
import pypto_pro.language as pl
from pypto_pro.runtime.platform import get_platform_info

MAX_RANK = 4
TILE_M = 128
TILE_N = 128

@dataclass
class AddTiling:
    shape: int[4]       # 未使用的前导维度填1
    opkind: int[8]      # opkind[4]保存算子选择值


@pl.jit(auto_mutex=True)
def add_dynrank_kernel(
    x: pl.Ptr[pl.DT_FP16],
    y: pl.Ptr[pl.DT_FP16],
    z: pl.Ptr[pl.DT_FP16],
    tiling: AddTiling,
):
    # 把Host侧2～4维输入的shape折叠成Kernel内固定的二维[M, N] Tensor视图。
    n = tiling.shape[3]
    m = tiling.shape[0] * tiling.shape[1] * tiling.shape[2]
    tensor_x = pl.make_tensor(x, [m, n])
    tensor_y = pl.make_tensor(y, [m, n])
    tensor_z = pl.make_tensor(z, [m, n])

    tile_type = pl.TileType(
        shape=[TILE_M, TILE_N],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Vec,
        valid_shape=[-1, -1],
    )
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x10000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x20000, mutex_ids=[30, 31])
    with pl.section_vector():
        num_cores = pl.get_block_num()
        core_id = pl.get_block_idx()
        m_tiles = (m + TILE_M - 1) // TILE_M
        n_tiles = (n + TILE_N - 1) // TILE_N
        total_tiles = m_tiles * n_tiles

        for idx in pl.range(core_id, total_tiles, num_cores):
            i = idx // n_tiles
            j = idx % n_tiles
            tile_a = a_db.next()
            tile_b = b_db.next()
            tile_c = c_db.next()

            # 分别覆盖满块、尾列、尾行和尾角四种情况。
            rem_r = m - i * TILE_M
            rem_c = n - j * TILE_N
            if rem_r >= TILE_M:
                if rem_c >= TILE_N:
                    pl.set_validshape(tile_a, [TILE_M, TILE_N])
                    pl.set_validshape(tile_b, [TILE_M, TILE_N])
                    pl.set_validshape(tile_c, [TILE_M, TILE_N])
                else:
                    pl.set_validshape(tile_a, [TILE_M, rem_c])
                    pl.set_validshape(tile_b, [TILE_M, rem_c])
                    pl.set_validshape(tile_c, [TILE_M, rem_c])
            else:
                if rem_c >= TILE_N:
                    pl.set_validshape(tile_a, [rem_r, TILE_N])
                    pl.set_validshape(tile_b, [rem_r, TILE_N])
                    pl.set_validshape(tile_c, [rem_r, TILE_N])
                else:
                    pl.set_validshape(tile_a, [rem_r, rem_c])
                    pl.set_validshape(tile_b, [rem_r, rem_c])
                    pl.set_validshape(tile_c, [rem_r, rem_c])

            pl.load_tile(tile_a, tensor_x, [i, j])
            pl.load_tile(tile_b, tensor_y, [i, j])

            if tiling.opkind[4] == 0:
                pl.add(tile_c, tile_a, tile_b)
            elif tiling.opkind[4] == 1:
                pl.sub(tile_c, tile_a, tile_b)
            else:
                pl.mul(tile_c, tile_a, tile_b)

            pl.store_tile(tensor_z, tile_c, [i, j])

OP_CASES = [
    (0, lambda a, b: a + b, "add"),
    (1, lambda a, b: a - b, "sub"),
    (2, lambda a, b: a * b, "mul"),
]


def ceildiv(a, b):
    return (a + b - 1) // b


def _run_case(shape, opkind, ref_fn, op_name):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    torch.npu.set_device(device)
    torch.manual_seed(0)
    dtype = torch.float16

    rank = len(shape)
    assert 2 <= rank <= MAX_RANK, f"rank must be in [2, {MAX_RANK}], got {rank}"

    numel = 1
    for s in shape:
        numel *= s

    x = torch.rand(shape, device=device, dtype=dtype)
    y = torch.rand(shape, device=device, dtype=dtype)
    z = torch.empty(shape, device=device, dtype=dtype)

    # 使用前导1补齐到长度4，使最内层维度始终位于shape[3]。
    dims = [1] * MAX_RANK
    for i in range(rank):
        dims[MAX_RANK - rank + i] = shape[i]

    opkind_arr = [0] * 8
    opkind_arr[4] = opkind
    tiling = AddTiling(shape=dims, opkind=opkind_arr)

    n = shape[-1]
    m = numel // n
    total_tiles = ceildiv(m, TILE_M) * ceildiv(n, TILE_N)
    # block_dim取平台可用AIV数量和任务Tile数量中的较小值。
    num_cores = min(get_platform_info().vector_core_num, total_tiles)

    add_dynrank_kernel[None, num_cores](x, y, z, tiling)
    torch.npu.synchronize()

    z_ref = ref_fn(x.float(), y.float()).half()
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
    logging.info("dynamic-rank %s %s (rank=%d, numel=%d) passed!", op_name, list(shape), rank, numel)


def test_add_dynamic_rank():
    shapes = [
        [512, 512],
        [8, 256, 256],
        [2, 4, 256, 256],
        [513, 513],
        [513, 511],
        [200, 300],
        [2, 3, 513],
        [2, 2, 3, 200],
    ]
    for shape in shapes:
        for opkind, ref_fn, op_name in OP_CASES:
            _run_case(shape, opkind, ref_fn, op_name)


if __name__ == "__main__":
    test_add_dynamic_rank()
```

本例演示了：

- 使用多个`int[N]`数组字段的TilingData类；
- Kernel中的**数组元素访问**（`tiling.opkind[4]`）驱动不同的计算分支；
- `tiling.shape[0..3]`（`int[4]`数组）保存Host输入的运行时shape；Kernel构造的`[M, N]` Tensor视图的rank固定为2；
- `valid_shape`和`pypto_pro.language.set_validshape`用于安全处理任意二维尾块；
- 同一个已编译Kernel可根据启动时传入的TilingData值运行三种不同算子，无需重新编译。

---

### 纯标量TilingData示例

仅需传递少量运行时标量时，可以使用只包含标量字段的TilingData类：

```python
@dataclass
class LoopTiling:
    n_iters: int          # 运行时循环边界

@pl.jit(auto_mutex=True)
def copy_kernel(
    x: pl.Tensor[[1024, 256], pl.DT_FP16],
    z: pl.Tensor[[1024, 256], pl.DT_FP16],
    tiling: LoopTiling,
):
    tt = pl.TileType(shape=[1, 256], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    g  = pl.make_tile_group(type=tt, addrs=0x0, mutex_ids=[0, 1])
    with pl.section_vector():
        for i in pl.range(0, tiling.n_iters, 1):     # 使用TilingData中的运行时循环边界
            buf = g.next()
            pl.load_tile(buf, x, [i, 0])
            pl.store_tile(z, buf, [i, 0])

# Host侧构造并启动
tiling = LoopTiling(n_iters=4)
copy_kernel[None, 1](x, z, tiling)
```

---

### 字段类型与dtype对应关系

TilingData字段标注与IR dtype、C struct成员的对应关系如下：

| 标注           | IR dtype | C struct成员（CCE）             |
|----------------|----------|----------------------------------|
| `int`          | `INDEX`  | `int64_t`                        |
| `float`        | `FP32`   | `float`                          |
| `bool`         | `BOOL`   | 布尔大小的整数                   |
| `int[N]`       | `INDEX`  | `int64_t[N]`                     |
| `float[N]`     | `FP32`   | `float[N]`                       |
| `bool[N]`      | `BOOL`   | 布尔大小的整数`[N]`             |

布局采用原生ctypes对齐计算，并与代码生成的C struct保持一致。字段顺序决定结构体布局，
因此Python dataclass的字段顺序必须与设备侧预期一致。

---

### 使用限制

- **不符合规范的字段类型。** 每个字段必须是`int`/`float`/`bool`或`T[N]`。
  其他类型标注不会被识别为有效的TilingData字段。
- **数组长度错误。** 赋给`int[8]`字段的序列必须恰好包含8个元素；长度不符会在
  JIT启动序列化时抛出`ValueError`。
- **数组声明过大。** `T[N]`中的`N`必须是正整数，且不能超过2048。
- **参数位置错误。** TilingData必须位于Kernel形参和启动实参的末尾。
- **字段顺序与布局不匹配。** dataclass的字段顺序决定struct布局，应与设备侧预期的
  字段顺序和填充方式保持一致。
- **运行时数据与编译期常量混用。** TilingData字段是运行时值，可用于`pypto_pro.language.range`、
  `pypto_pro.language.make_tensor`、算术和`if`条件，但不能用于需要编译期Python `int`的参数，
  例如`TileType`的静态`shape`。

---

### 使用摘要

```python
from __future__ import annotations

# 声明TilingData
@dataclass
class MyTiling:
    n: int
    scale: float
    flags: int[8]

# 在Kernel签名中声明TilingData参数
@pl.jit(auto_mutex=True)
def k(x: pl.Ptr[pl.DT_FP16], tiling: MyTiling):
    t = pl.make_tensor(x, [tiling.n, 128])             # shape 中的标量字段
    for i in pl.range(0, tiling.n, 1):                 # 作循环边界的标量字段
        if tiling.flags[4] == 1:                       # 分支中的数组元素
            ...
    ...

# 在Host侧构造并启动
flags = [0] * 8
flags[4] = 1
tiling = MyTiling(n=256, scale=2.0, flags=flags)
k[None, num_cores](x, tiling)
```

## TilingKey

本节介绍TilingKey的声明和使用方法。TilingKey使用有限的编译期配置，为同一份
PyPTO Pro Kernel生成多个专用实例，并在启动时选择目标实例。TilingKey适用于会改变
代码路径、Tile模板或数据布局的离散模式。

所有示例使用以下导入：

```python
from pypto_pro.runtime.tilingkey import TilingKeyField
import pypto_pro.language as pl
```

---

### 何时使用TilingKey

TilingKey字段在编译阶段会被折叠为常量。因此，每个具体Key都会生成独立Kernel，
不会在单个Kernel中保留对应的运行时分支。TilingKey适合以下场景：

- 可选功能会改变较大代码路径，例如是否应用attention mask；
- Tile尺寸、Layout或Dtype模板只有有限候选值；
- 需要在启动前拒绝不支持的字段组合；
- 二进制交付时需要枚举可编译的TilingKey。

TilingKey不适用于任意运行时shape。候选值的笛卡尔积决定可枚举的组合数量，因此
TilingKey应仅描述有限且会影响代码生成的模式。

---

### 声明Schema

TilingKey是一个普通Python类。类属性使用`TilingKeyField(bits=..., values=...)`声明：

```python
class AttentionKey:
    # 二进制开关，取值只能为 0 或 1。
    HasAtten = TilingKeyField(bits=1, values=[0, 1])

    # 两种固定 Tile 模板。
    BlockM = TilingKeyField(bits=8, values=[64, 128])

    def is_valid(self, key):
        has_atten, block_m = key
        # 仅举例：mask 模式只支持 128 行 Tile。
        return has_atten == 0 or block_m == 128
```

字段按**类定义顺序**收集。该顺序同时决定：

1. `is_valid()`中`key`元组的元素顺序；
2. 64-bit编码中各字段的bit offset；
3. 二进制交付头文件中的字段和selector顺序。

`is_valid()`是可选校验函数，参数`key`使用按照字段定义顺序排列的元组。
该函数既用于在JIT启动时校验具体Key，也用于在二进制交付时过滤枚举组合。

#### 字段约束

框架在应用`@pl.jit`装饰器时检查schema：

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

#### 编码与AscendC对齐

`values`的顺序具有语义：编码后的TilingKey字段保存该值在`values`中的下标，解码后
得到Kernel中使用的实际值。该行为与Ascend C模板TilingKey的Selector一致。

```python
class MaskKey:
    NeedAttnMask = TilingKeyField(bits=1, values=[1, 0])
```

此字段的映射为：

| 实际值 | `values`下标 | 字段Bit / TilingKey |
|---:|---:|---:|
| `1` | `0` | `0` |
| `0` | `1` | `1` |

因此，TilingKey为`1`时，`NeedAttnMask`的实际值为`0`。启动时仍传入实际值，
例如`{"NeedAttnMask": 0}`，不需要手工传入下标。

---

### 将Schema绑定到Kernel

通过`@pl.jit(tiling_key=...)`关联schema。TilingKey字段在Kernel中作为编译期变量直接引用，
无需在Kernel形参列表中声明：

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

例如，`HasAtten=1`时，parser保留`kv_end = qi + 1`分支；`HasAtten=0`时，
编译阶段移除该分支并保留初始值`kv_end = 32`。二者共享源码，但最终Kernel中不会保留对
`HasAtten`的运行时判断。

TilingKey字段名不得与Kernel形参或模块级普通变量冲突。字段名应描述其编译期语义，
例如`HasAtten`、`S1TemplateType`，并避免使用`n`、`shape`等可能与运行时变量冲突的名称。

---

### 选择实例并启动

带TilingKey的Kernel必须在方括号启动参数中提供完整的Key字典，位置在`stream`和
`block_dim`之后：

```python
key = {"HasAtten": 1, "BlockM": 128}
attention_kernel[None, num_cores, key](q, k, out)
```

Key字典的字段必须与Schema **完全一致**：

- 每个字段都必须出现，且不能有额外字段；
- 值必须属于该字段的`values`；
- 整个组合必须通过`is_valid()`；
- 不能直接调用`attention_kernel(...)`，也不能使用list或tuple代替Key字典。

若同时使用`datatype`特化，TilingKey仍是第三个参数，datatype dict紧随其后：

```python
attention_kernel[None, num_cores, key, datatype](q, k, out)
```

框架按照字段定义顺序，将Key字典中的实际值转换为对应的`values`下标，再打包为唯一的
64-bit Key，并缓存对应的专用编译结果。

---

### FlashAttention特化示例

以下示例使用`FaTilingKey`为`flash_attention_score`生成causal attention和
full attention两种专用实例。其他Kernel实参不参与TilingKey Schema，也不影响具体Key的选择。

#### `FaTilingKey`的字段

`FaTilingKey`声明14个编译期字段：

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

这些字段总计63 bits。`is_valid()`将除`HasAtten`之外的字段限制为表中的固定值，
因此候选值的笛卡尔积最终只保留两个合法Key：`HasAtten=0`和`HasAtten=1`。

Kernel在Cube和Vector两个循环中均使用`HasAtten`选择causal和full专用路径：

```python
causal_skv = skv_tiles
if HasAtten == 1:
    causal_skv = qi + 1
```

`HasAtten=0`的专用Kernel仅保留full attention的`skv_tiles`路径；`HasAtten=1`的
专用Kernel仅保留causal attention的`qi + 1`路径。源码中可以保留清晰的`if/else`结构，
而每个具体Key的最终代码只包含可达分支。

#### 启动两个专用实例

基础Key如下：

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

两次启动使用相同的大部分Key字段，仅改变`HasAtten`，分别选择causal attention和
full attention专用实例。该示例支持FP16和BF16。

---

### 二进制交付

对带TilingKey的Kernel调用`generate_binary_headers()`可生成TilingKey头文件：

```python
from pypto_pro.runtime.opc.pypto_compile import generate_binary_headers

binary_dir = generate_binary_headers(flash_attention_score)
```

生成的`FaTilingKey_tilingkey.h`包含字段声明及通过`is_valid()`
的Key Selector。该文件使用`ASCENDC_TPL_ARGS_DECL`描述各字段和允许值，并以
`ASCENDC_TPL_SEL`仅列出合法组合。字段bit选择`values`中对应下标的实际值；因此应尽量
收紧`values`，并在存在字段关联约束时实现`is_valid()`，避免生成无用的二进制实例。

---

### 常见错误

| 现象 | 原因与处理 |
|---|---|
| 应用Kernel装饰器时失败 | 检查`bits > 0`、候选值为互不重复的整数、候选数量不超过`2**bits`，且总位宽不超过64。 |
| 启动时字段不匹配 | Key字典必须包含所有字段且不能包含未知字段；字段名大小写必须与类属性一致。 |
| 启动值不在候选集中 | 将实际值加入声明的`values`，或使用已有候选值；不能传入候选下标。 |
| 启动被`is_valid()`拒绝 | 按字段定义顺序检查组合约束。 |
| Kernel中找不到字段名 | 在`@pl.jit(tiling_key=...)`中绑定Schema，并避免字段名与Kernel参数或模块变量冲突。 |
| 为每个shape新增Key | 仅将真正影响代码生成的有限模式放入TilingKey。 |

---

### 最小示例

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
