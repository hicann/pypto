# TilingData

本文档介绍 **TilingData**（亦称 *tiling类* / *tiling结构体*）：如何声明、如何传入
kernel、如何在kernel体内读取其字段，以及如何使用数组字段。TilingData是把
**运行时参数**——shape、stride、循环边界、算子选择器、缩放系数等——喂给已编译kernel
的方式，而无需把它们固化进kernel签名。

源码参考：

- `python/pypto_pro/language/typing/_tiling.py` —— `T[N]`注解解析、
  `is_tiling_class`、`get_tiling_fields`、ctypes序列化。
- `python/tests/st/pypto_pro/frontend/element_wise/test_tiling_op.py` —— 一个简单、
  可执行的端到端示例（本文档贯穿使用）。
- `python/tests/st/pypto_pro/frontend/fa/test_fa_perf_tkv_preload_dn_vf_bufid_dynrank.py`
  —— 复杂的`OpTiling`示例（FlashAttention，动态rank）。请先阅读本文这里的较简单
  示例。

所有示例使用的前置导入：

```python
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch_npu
import pypto_pro.language as pl
```

执行脚本：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh; python3 test_case.py
```

---

## 什么是TilingData类？

TilingData类是一个普通的Python `@dataclass`，其字段**全部**为以下之一：

- `int`  —— 标量整数  （降低为`INDEX` / `int64_t`）
- `float`—— 标量浮点  （降低为`FP32`）
- `bool` —— 标量布尔  （降低为`BOOL`）
- `T[N]` —— 定长数组，含`N`个`T`类型元素（`T` ∈ `{int, float, bool}`）

它由`is_tiling_class()`（`_tiling.py`）识别：一个至少有一个字段、且每个字段都用上述
类型之一标注的dataclass。

当你把一个tiling实例传给JIT kernel时，框架会：

1. 由字段标注推导出一个C `struct`布局（`get_tiling_ctype_struct`），并且
2. 把实例序列化为该精确的字节布局（`tiling_instance_to_bytes`），以便交给设备侧代码。

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

这就是一个完整、合法的tiling类。

---

## 声明接收TilingData的kernel

加入一个用你的tiling类标注的参数。它**不是** tensor —— 而是运行时参数结构体。一种
常见而强大的模式是把它与裸指针输入（`pl.Ptr[dtype]`）结合，从tiling的shape重建
tensor视图（即 "动态rank" kernel）：

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
    # 重建带类型视图；rank/shape 来自 tiling，而非签名。
    tensor_q = pl.make_tensor(q, [tiling.sq,  tiling.d], [tiling.d, 1])
    tensor_k = pl.make_tensor(k, [tiling.skv, tiling.d], [tiling.d, 1])
    tensor_v = pl.make_tensor(v, [tiling.skv, tiling.d], [tiling.d, 1])
    tensor_o = pl.make_tensor(o, [tiling.sq,  tiling.d], [tiling.d, 1])

    sq_dim    = tiling.sq
    skv_dim   = tiling.skv
    sq_tiles  = (sq_dim  + (TS  - 1)) // TS
    skv_tiles = (skv_dim + (TKV - 1)) // TKV
    # ... kernel 体把 tiling.* 当作普通运行时标量使用 ...
```

你也完全可以保留带类型的`pl.Tensor`输入，仅用tiling来传循环边界、flag、缩放系数
等 —— 两种用法相互独立。

> [!NOTE]说明
> 当前JIT只识别调用实参中的最后一个TilingData实例。因此，TilingData参数必须放在
> Kernel形参列表末尾，launch时也必须作为最后一个实参传入。

---

## 在kernel体内读取TilingData字段

在kernel体内，tiling字段就是普通的运行时值：

| 访问形式             | 含义                                                |
|----------------------|-----------------------------------------------------|
| `tiling.field`       | 读取标量字段（int / float / bool）。               |
| `tiling.arr[k]`      | 读取`T[N]`字段的第`k`个元素。           |

它们可用于：

- `pl.make_tensor`的shape / stride实参，
- `pl.range(...)`的循环边界，
- 算术运算（`(tiling.sq + TS - 1) // TS`），
- **类似编译期风格的分支**（`if tiling.opkind[4] == 0:`）。

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

## 在launch时构造并传入TilingData实例

在host侧，照常构造dataclass，并在方括号launch中作为对应参数传入。对标量字段，直接
传值即可：

```python
tiling = OpTiling(sq=8192, skv=8192, d=128)
fa_kernel[None, num_cores](q_t, k_t, v_t, o_t, tiling)
torch.npu.synchronize()
```

框架会自动把`tiling`序列化为其C struct字节 —— 你无需自行管理字节缓冲。

---

## 数组字段

数组字段持有固定数量的同质元素。用`T[N]`声明字段，用Python原生`list`
保存运行时值。包含`T[N]`的模块必须启用`from __future__ import annotations`，
避免Python在定义class时求值`int[N]`。

### 声明数组字段

```python
@dataclass
class OpTiling:
    offsets: int[4]     # 4 个 int
    scales:  float[2]   # 2 个 float
    opkind:  int[8]     # 8 个 int
```

规则（由`_tiling.py`的注解解析器强制）：

- 元素类型必须为`int`、`float`或`bool`；
- 大小必须为正整数字面量，且不能超过2048。

`int[N]`是PyPTO Pro DSL注解，不是标准Python类型；部分静态类型检查器可能提示
`int`不可下标，但PyPTO Pro会从延迟注解字符串中安全解析该字段。

### 构造数组值

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

## 完整、可执行示例 —— 动态Rank与尾块处理

下面的示例来自
`python/tests/st/pypto_pro/frontend/element_wise/test_eltwise_dynamic_rank.py`。
TilingData包含两个定长数组：`shape`用于把Rank为2～4的输入重建为二维Tensor视图，
`opkind[4]`用于在运行时选择加、减或乘。示例同时使用`valid_shape`处理不能被
`128 × 128`整除的尾块。

```python
from __future__ import annotations

from dataclasses import dataclass

import logging
import torch
import torch_npu
import pypto_pro.language as pl

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
    # 把Rank为2～4的逻辑shape折叠成二维[M, N]。
    n = tiling.shape[3]
    m = tiling.shape[0] * tiling.shape[1] * tiling.shape[2]
    tensor_x = pl.make_tensor(x, [m, n], [n, 1])
    tensor_y = pl.make_tensor(y, [m, n], [n, 1])
    tensor_z = pl.make_tensor(z, [m, n], [n, 1])

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
    device = "npu:0"
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
    num_cores = min(32, total_tiles)

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

- 使用多个`int[N]`数组字段的tiling类；
- kernel体内的**数组元素访问**（`tiling.opkind[4]`）驱动不同的计算分支；
- `tiling.shape[0..3]`（`int[4]`数组）用于支持动态Rank；
- `valid_shape`和`pl.set_validshape`用于安全处理任意二维尾块；
- 同一个已编译kernel仅凭launch时传入的tiling值即可运行三种不同算子 —— 无需重新编译。

---

## 一个最小的纯标量示例

如果你只需要少量运行时标量（例如动态循环边界），一个很小的tiling类就够了：

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
        for i in pl.range(0, tiling.n_iters, 1):     # <-- 来自 tiling 的运行时循环边界
            buf = g.next()
            pl.load_tile(buf, x, [i, 0])
            pl.store_tile(z, buf, [i, 0])

# Host：
tiling = LoopTiling(n_iters=4)
copy_kernel[None, 1](x, z, tiling)
```

---

## 字段类型 → dtype参考

出自`tiling.py`（`_PYTHON_TYPE_TO_DTYPE` / `get_tiling_fields`）：

| 标注           | IR dtype | C struct成员（CCE）             |
|----------------|----------|----------------------------------|
| `int`          | `INDEX`  | `int64_t`                        |
| `float`        | `FP32`   | `float`                          |
| `bool`         | `BOOL`   | 布尔大小的整数                   |
| `int[N]`       | `INDEX`  | `int64_t[N]`                     |
| `float[N]`     | `FP32`   | `float[N]`                       |
| `bool[N]`      | `BOOL`   | 布尔大小的整数`[N]`             |

布局采用原生ctypes对齐计算，因此与codegen发出的C struct一致。由于布局是按位置
决定的，**字段顺序很重要**：保持Python dataclass的字段顺序与设备侧预期完全一致。

---

## 常见坑

- **不符合规范的字段类型。** 每个字段必须是`int`/`float`/`bool`或`T[N]`。
  任何其他标注都会使该类无法通过`is_tiling_class()`，从而不会被当作TilingData。
- **数组长度错误。** 赋给`int[8]`字段的序列必须恰好包含8个元素；长度不符会在
  JIT启动序列化时抛出`ValueError`。
- **数组声明过大。** `T[N]`中的`N`必须是正整数字面量，且不能超过2048。
- **TilingData不是最后一个参数。** 当前JIT只检查最后一个调用实参；应将TilingData
  放在Kernel形参和launch实参的末尾。
- **字段顺序 / 布局不匹配。** dataclass的字段顺序*就是* struct布局。如果你的设备侧
  struct期望某种填充或顺序，请精确镜像它。
- **忘了它是运行时数据。** tiling字段是运行时值，而非parser能看作字面量的Python
  常量；请在`pl.range`、`pl.make_tensor`、算术与`if`条件中使用它们，而不要用在需要
  编译期Python `int`的地方（例如`TileType`的静态`shape`）。

---

## 速查

```python
from __future__ import annotations

# --- 声明 ---
@dataclass
class MyTiling:
    n: int
    scale: float
    flags: int[8]

# --- kernel 参数 ---
@pl.jit(auto_mutex=True)
def k(x: pl.Ptr[pl.DT_FP16], tiling: MyTiling):
    t = pl.make_tensor(x, [tiling.n, 128], [128, 1])   # shape 中的标量字段
    for i in pl.range(0, tiling.n, 1):                 # 作循环边界的标量字段
        if tiling.flags[4] == 1:                       # 分支中的数组元素
            ...
    ...

# --- 构造并 launch ---
flags = [0] * 8
flags[4] = 1
tiling = MyTiling(n=256, scale=2.0, flags=flags)
k[None, num_cores](x, tiling)
```
