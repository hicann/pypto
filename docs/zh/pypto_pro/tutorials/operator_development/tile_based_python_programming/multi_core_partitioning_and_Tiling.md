# 多核切分与Tiling：把数据分到多个AI Core

本文档介绍PyPTO Pro编程里的**多核切分（tiling）**：如何把一份数据切开、分配到多个
AI Core上并行处理。内容包括：

- 在kernel里用`pl.get_block_idx()` / `pl.get_block_num()`拿到“我是第几个核 / 一共几个核”；
- 在launch时用方括号语法设置核数（`block_dim`）；
- host与device之间怎么传tiling数据（标量参数 / TilingData结构体 / 动态shape）；
- 参考Ascend C tiling设计方法论，用PyPTO Pro写出**负载均衡**的切分。

源码参考：

- `python/pypto_pro/ir/op/system_ops.py` — `get_block_idx`、`get_subblock_idx`、`get_block_num`。
- `python/pypto_pro/runtime/jit.py` — `_TileJitKernel.__getitem__`（方括号launch）、
  `_clamp_block_dim`、`_pack_tiling_arg`（tiling结构体序列化）、`_args_to_ctypes`。
- `python/pypto_pro/runtime/platform.py` — `get_platform_info().core_num`（查询可用核数）。
- `python/pypto_pro/language/typing/_tiling.py` — TilingData类的字段解析与序列化。

可执行范例：

- `python/tests/st/pypto_pro/frontend/element_wise/test_add.py`（最简单的动态shape多核加法）。
- `python/tests/st/pypto_pro/frontend/element_wise/test_eltwise_dynamic_rank.py`
  （TilingData + 多核 + 尾块，端到端）。

均衡tiling方法论参考：参见仓内examples/ 目录。

配套阅读：[TilingData](TilingData.md)、[尾块Tile](tail_block_handling.md)、
[Tensor、Tile与TileGroup](Python_programming_overview.md)。

统一前置导入：

```python
import torch
import pypto_pro.language as pl
```

---

## 心智模型：block == AI Core

在PyPTO Pro / Ascend术语里，一个**block**就是一个**AI Core**。启动一个kernel时，
**同一份kernel代码**会在`block_dim`个核上各跑一遍（SPMD模型）。每个核靠自己的
**核编号**去认领整份工作里的一小块：

- `pl.get_block_num()` → 本次一共有几个核（即launch时传入的`block_dim`）。
- `pl.get_block_idx()` → 当前核的编号，取值`0 .. block_num - 1`。
- `pl.get_subblock_idx()` → 核内子块编号（`0`或`1`，用于Vector双子核；一般
  只在需要区分子核时才用）。

它们没有参数，直接在kernel体里调用，返回运行时整数：

```python
num_cores = pl.get_block_num()     # 例如 32
core_id   = pl.get_block_idx()     # 0..31
```

### `get_subblock_idx` —— Vector子核

Ascend 950PR/Ascend 950DT的Vector单元每个block内有**两个子核**。`pl.get_subblock_idx()`返回
`0`或`1`，让两个子核在同一份数据里再各分一半。典型用法是让子核沿某一轴各处理半块
（`fa/test_fa_serial_dn_auto.py`）：

```python
with pl.section_vector():
    sub_id  = pl.get_subblock_idx()      # 0 或 1
    row_off = sub_id * TS_HALF           # 两个子核各处理 [0:TS_HALF] 和 [TS_HALF:TS]
```

一般kernel不需要它 —— 只有当你要在block内进一步切分、榨干Vector双子核的并行度时才
用。多核切分的主线仍是`get_block_idx` / `get_block_num`。

三个API的层级关系：

```text
block_dim 个 block（AI Core）        <- launch 时决定，get_block_num() 读到
   └── 每个 block 内 2 个 subblock   <- Vector 双子核，get_subblock_idx() 读到
```

---

## 跨步循环：标准的多核切分惯用法

把总任务切成`total_tiles`个tile后，最常见、最均衡的分法是**跨步（strided）**分配：
第`core_id`个核处理第`core_id, core_id+num_cores, core_id+2*num_cores, ...`个tile。
这正是`pl.range(start, stop, step)`的用途（语义同Python `range`）：

```python
with pl.section_vector():
    num_cores = pl.get_block_num()
    core_id   = pl.get_block_idx()

    total_tiles = m_tiles * n_tiles          # 整份工作被切成这么多 tile

    # 每个核跨步遍历扁平化的 tile 网格。
    for idx in pl.range(core_id, total_tiles, num_cores):
        i = idx // n_tiles                   # 行 tile 索引
        j = idx % n_tiles                    # 列 tile 索引
        ...
```

为什么用跨步而不是“连续分段”（第k个核拿`[k*chunk : (k+1)*chunk]`）？

- **天然均衡**：若`total_tiles`不能被`num_cores`整除，跨步写法让前
  `total_tiles % num_cores`个核各多做1个tile，最多相差1 —— 无需手写首/尾核逻辑。
- **循环边界统一**：所有核共用同一个`pl.range(core_id, total_tiles, num_cores)`，代码
  只有一份。

> 也可以按二维直接切（外层`range(core_id, m_tiles, num_cores)`切行、内层
> `range(0, n_tiles, 1)`遍历列），见`test_add.py`。选哪种取决于哪种切分让每个核的数据
> 更连续、更均衡。

### 跨步为什么均衡 —— 一个具体的数

设`total_tiles = 100`、`num_cores = 32`。`100 = 3 * 32 + 4`，跨步分配的结果是：

- 核`0..3`（共4个）各处理**4**个tile：如核0拿`{0, 32, 64, 96}`；
- 核`4..31`（共28个）各处理**3**个tile：如核4拿`{4, 36, 68}`。

最忙的核和最闲的核只差**1**个tile —— 这就是“负载均衡”。而且你**没写任何**首/尾核分支
代码，`pl.range(core_id, 100, 32)`一行就做到了。对比“连续分段”，如果写成核k拿
`[k*4 : k*4+4]`，前25个核各4个、后7个核0个 —— 严重不均。

### 二维切vs扁平切

| 方式        | 循环写法                                                   | 适用                         |
|-------------|------------------------------------------------------------|------------------------------|
| **扁平切** | `for idx in pl.range(core_id, m_tiles*n_tiles, num_cores)` | tile总数多、想要最细粒度均衡 |
| **二维切** | 外`range(core_id, m_tiles, num_cores)` + 内`range(0, n_tiles, 1)` | 每核处理整行、数据更连续 |

扁平切把`idx`还原成`(i, j)`：`i = idx // n_tiles`、`j = idx % n_tiles`。二维切让同一
核处理连续的若干整行，GM访问更连续，但当`m_tiles < num_cores`时会有核空转（此时应改
扁平切）。

---

## 在launch时设置核数（block_dim）

编译后的kernel用方括号语法启动，语法是`kernel[stream, block_dim](*args)`：

```python
kernel[stream, block_dim](x, y, z)   # 显式 stream + 核数
kernel[block_dim](x, y, z)           # 默认 stream，只给核数
kernel(x, y, z)                       # 默认 stream，block_dim=1
```

- `stream` —— NPU stream；传`None`用默认stream。
- `block_dim` ——**核数**，也就是kernel里`get_block_num()`会返回的值。

```python
# 例：用 32 个核跑
kernel[None, 32](x, y, z)
torch.npu.synchronize()
```

### 让核数适配硬件

不要写死一个可能超过硬件核数的值。两种做法：

**(a)查询平台核数**（`platform.py`）：

```python
from pypto_pro.runtime.platform import get_platform_info

info = get_platform_info()
print(info.soc_version)   # 例如 "DAV_3510"
print(info.core_num)      # 例如 20 / 32 ...
kernel[None, info.core_num](q, k, v, o)
```

**(b)按任务量取下界**，避免核多于tile（多出来的核会空转）：

```python
num_cores = min(info.core_num, total_tiles)
kernel[None, num_cores](x, y, z)
```

此外，运行时还有一层保护：`_clamp_block_dim`（`jit.py`）会把超过平台核数的
`block_dim`**截断**到硬件上限并打印告警。但**依赖它兜底不是好习惯**——请在host侧就
按`min(core_num, 任务tile数)`算好。

---

## host ↔ device传tiling数据

kernel需要知道“数据多大、切多少块、循环几次、用哪个算子”等运行时信息。PyPTO Pro有三条
互补的通道把这些从host传到device：

### 动态shape（`pl.DYNAMIC`）—— 从tensor自动解析

在签名里用`pl.DYNAMIC`声明动态维度；kernel体内通过`tensor.shape[i]`取运行时值。

```python
@pl.jit(auto_mutex=True)
def add_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    with pl.section_vector():
        num_cores = pl.get_block_num()
        core_id   = pl.get_block_idx()
        m_tile_num = x.shape[0] // TILE_M
        n_tile_num = x.shape[1] // TILE_N
        for i in pl.range(core_id, m_tile_num, num_cores):
            for j in pl.range(0, n_tile_num, 1):
                ...

# host：shape 自动从传入 tensor 推断
add_kernel[None, num_cores](x, y, z)
```

### 标量参数 —— 传一两个运行时值

需要传单个运行时标量（缩放系数、循环边界、flag）时，直接在签名里加一个`pl.DT_*`参数，
launch时传值即可（`_append_scalar_ctype_arg`按dtype映射到ctypes）：

```python
@pl.jit()
def scaled_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_INT8],
    scale_bits: pl.DT_INT32,          # <-- 运行时标量
):
    ...
    pl.store(out, acc, [0, 0], pre_quant_scalar=scale_bits)

# host
scaled_kernel(a, out, scale_bits)      # 直接传 int
```

### TilingData结构体 —— 传一整包tiling参数

参数一多，就把它们打包进一个`@dataclass`的**TilingData**类（字段为
`int`/`float`/`bool`或定长数组`T[N]`）。框架会把它序列化成一段C struct字节、放进一个
device上的`uint8` buffer，kernel入口再拷进来（`jit.py:_pack_tiling_arg`、
`_tiling.py:tiling_instance_to_bytes`）。

```python
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class AddTiling:
    shape: int[4]       # 每维大小（动态 rank）
    opkind: int[8]      # 算子选择器，值在 opkind[4]

@pl.jit(auto_mutex=True)
def add_dynrank_kernel(
    x: pl.Ptr[pl.DT_FP16],
    y: pl.Ptr[pl.DT_FP16],
    z: pl.Ptr[pl.DT_FP16],
    tiling: AddTiling,             # <-- TilingData 参数
):
    N = tiling.shape[3]                          # 字段当普通运行时值用
    M = tiling.shape[0] * tiling.shape[1] * tiling.shape[2]
    tensor_x = pl.make_tensor(x, [M, N], [N, 1])
    ...
    if tiling.opkind[4] == 0:                    # 数组元素驱动运行时分支
        pl.add(tile_c, tile_a, tile_b)

# host：普通构造 dataclass，作为最后一个参数传入
tiling = AddTiling(shape=[1, 1, 513, 513], opkind=[0,0,0,0,1,0,0,0])
add_dynrank_kernel[None, num_cores](x, y, z, tiling)
```

TilingData的完整规则（字段类型、数组、序列化、字段顺序即布局）见
[TilingData文档](TilingData.md)。三条通道可以自由组合：常见做法是`pl.Ptr`输入 +
一个TilingData携带shape/循环边界/算子选择。

<a id="tiling-key--编译期特化而非运行时传值"></a>

### tiling_key —— 编译期特化（而非运行时传值）

前三条都是**运行时**通道（一份kernel处理所有取值）。若某个“模式开关”会让整段代码走
不同分支，且你希望**每种模式各编一份专用kernel**（消除死分支、拿到最优指令），用
`tiling_key`：声明一个字段类，launch时用方括号第三位传具体key，parser把字段折叠成
编译期常量（`python/pypto_pro/runtime/tilingkey.py`）。

```python
from pypto_pro.runtime.tilingkey import TilingKeyField

class FaTilingKey:
    # 每个字段：占几 bit + 允许的取值
    NeedAttnMask = TilingKeyField(bits=1, values=[0, 1])

    # 可选：拒绝非法组合（key 是按定义顺序的取值元组）
    def is_valid(self, key):
        (need_attn_mask,) = key
        return True

@pl.jit(auto_mutex=True, tiling_key=FaTilingKey)
def fa_kernel(q: pl.Ptr[pl.DT_FP16], k: pl.Ptr[pl.DT_FP16],
              v: pl.Ptr[pl.DT_FP16], o: pl.Ptr[pl.DT_FP16], tiling: OpTiling):
    ...
    if NeedAttnMask == 1:            # 被折叠成常量：另一分支在该 key 下是死代码，直接消除
        ...                          # 载入并应用 mask
    ...

# launch：方括号第三位是具体 key dict —— 每个 key 编成独立 kernel
fa_kernel[None, num_cores, {"NeedAttnMask": 1}](q, k, v, o, tiling)   # causal + mask
fa_kernel[None, num_cores, {"NeedAttnMask": 0}](q, k, v, o, tiling)   # full，同一份源码
```

**运行时flag（§4.2/§4.3）还是`tiling_key`？**

| 需求                                        | 用哪个           |
|---------------------------------------------|------------------|
| 值经常变、想避免重复编译                    | 运行时标量 / TilingData |
| 值只有少数几种、分支很重、想要每种最优代码  | `tiling_key`     |
| 想让非法组合在launch时就报错              | `tiling_key` + `is_valid` |

`tiling_key`与其它三条通道正交，可同时用（`tiling_key`选分支，TilingData传shape）。
完整示例见`python/tests/st/pypto_pro/frontend/fa/test_fa_tilingkey_attn_mask.py`。

---

## 用PyPTO Pro写出均衡的tiling

上面讲了“怎么把tile分给核”的机制。这一节讲“**每个核该分多少、tile该多大**”的
方法论。设计Tiling时，需要综合考虑多核负载、片上存储容量、数据对齐和尾块处理，下面
介绍这些因素在PyPTO Pro里的落地方式。

### 多核切分：负载均衡

多核切分的目标是：**每个核的任务量尽量相等；相邻数据尽量同核；tile粒度适中
（太小调度开销大，太大并行度低）。**

以逐元素算子为例，Host侧可以根据每核最小处理量（如4KB）估算实际使用的核数，避免
小数据量启用过多核：

```python
# host 侧：算实际要用几个核，避免小数据开太多核
MIN_ELEMS_PER_CORE = 4 * 1024 // elem_bytes         # 每核至少 4KB
core_by_load = (numel + MIN_ELEMS_PER_CORE - 1) // MIN_ELEMS_PER_CORE
num_cores = min(core_by_load, info.core_num)
```

在PyPTO Pro里，“每核任务量相等”由第2节的**跨步循环**自动保证 —— 不必像Ascend C
那样手写`blockFormer` / `blockTail` / 首尾核分支。你只需：

1. host侧算出总tile数`total_tiles`和要用的`num_cores`；
2. kernel里`for idx in pl.range(core_id, total_tiles, num_cores)`。

跨步分配天然让各核tile数最多相差1，即负载均衡。

### UB（片上buffer）切分：单次处理多少

UB切分的要点是：**受UB容量限制，单次处理量要对齐到Vector指令友好的粒度
（如256B）。**

在PyPTO Pro里，“UB切分”就是**选tile的shape**：一个`[TILE_M, TILE_N]`的tile就是
一次处理的数据块。选tile尺寸时：

- **不超UB**：所有并存tile的字节和（含双缓冲的多份）不能超过UB。每份
  `prod(tile.shape) * dtype_bytes`，`N`缓冲就乘`N`。
- **对齐友好**：`TILE_N`（内层连续维）选成Vector对齐粒度的整数倍（如FP16取128、
  FP32取64等），提升指令效率。
- **能整除更好，不能整除就靠尾块**：tile不必整除tensor —— 用向上取整的tile数 +
  `set_validshape`处理尾块（见[尾块Tile文档](tail_block_handling.md)）。

```python
TILE_M, TILE_N = 128, 128                      # 一次处理 128x128 = 16384 元素
tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16,
                        target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
m_tiles = (M + TILE_M - 1) // TILE_M           # 向上取整覆盖尾块
n_tiles = (N + TILE_N - 1) // TILE_N
```

**UB预算的一个具体算例。**逐元素加法`c = a + b`，FP16，三个操作数各双缓冲：

```text
单份 tile 字节 = TILE_M * TILE_N * sizeof(FP16) = 128 * 128 * 2 = 32 KB
buffer 份数    = 3 个操作数(a,b,c) x 2(双缓冲) = 6 份
总占用         = 6 * 32 KB = 192 KB
```

当前支持平台的UB容量为248KB，因此192KB可以放下。若换成FP32（单份64KB），
`6 * 64 = 384KB`，会**超出UB容量**——这时要么把tile缩小（如`128 x 64`），要么减少缓冲份数。选tile
尺寸本质上就是在解这道“所有并存buffer字节和 ≤ UB容量”的不等式。

### Buffer规划：双缓冲重叠

指南（§3）要点：**规划输入/输出/中间buffer，并用Double Buffer让搬运与计算重叠。**

在PyPTO Pro里用`pl.make_tile_group` + `auto_mutex=True`实现双 / N缓冲，框架自动插入
同步（细节见[Tensor、Tile与TileGroup](Python_programming_overview.md)）：

```python
a_db = pl.make_tile_group(type=tile_type, addrs=0x0000,  mutex_ids=[0, 1])   # 双缓冲
b_db = pl.make_tile_group(type=tile_type, addrs=0x10000, mutex_ids=[2, 3])
c_db = pl.make_tile_group(type=tile_type, addrs=0x20000, mutex_ids=[30, 31])
```

### 分支场景覆盖

指南（§4）要点：**覆盖dtype / shape大小 / 对齐 / 边界等不同场景。**在PyPTO Pro里：

- **dtype / 模式分支**：用TilingData的字段在kernel里`if`分支（如`opkind[4]`），或
  用`tiling_key`做**编译期特化**（每个key编成一份专用kernel，见
  `python/pypto_pro/runtime/tilingkey.py`）。
- **对齐 / 边界**：用尾块机制（`valid_shape=[-1,-1]` + `set_validshape`）统一覆盖非对齐
  shape，无需为对齐/非对齐各写一份kernel。

---

## 完整可执行示例

下面是`test_add.py`：动态shape、多核、双缓冲，`auto_mutex`免手写同步。它把
`[M, N]`按`128 × 128`切块，外层按行跨步分核。

```python
import os
import torch
import pypto_pro.language as pl

TILE_M = 128
TILE_N = 128


@pl.jit(auto_mutex=True)
def add_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16,
                            target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000,  mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x10000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x20000, mutex_ids=[30, 31])

    with pl.section_vector():
        num_cores = pl.get_block_num()
        core_id   = pl.get_block_idx()
        m_tile_num = x.shape[0] // TILE_M
        n_tile_num = x.shape[1] // TILE_N

        # 每个核跨步遍历 M 方向的 tile；内层遍历 N 方向。
        for i in pl.range(core_id, m_tile_num, num_cores):
            for j in pl.range(0, n_tile_num, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


def test_add():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    torch.npu.set_device(device)
    M_SIZE, N_SIZE = 8192, 4096
    num_cores = M_SIZE // TILE_M           # 每个核一行 tile；此处能整除
    torch.manual_seed(0)
    x = torch.rand([M_SIZE, N_SIZE], device=device, dtype=torch.float16)
    y = torch.rand([M_SIZE, N_SIZE], device=device, dtype=torch.float16)
    z = torch.empty([M_SIZE, N_SIZE], device=device, dtype=torch.float16)

    add_kernel[None, num_cores](x, y, z)   # launch：[stream, block_dim]
    torch.npu.synchronize()
    torch.testing.assert_close(z, x + y)


if __name__ == "__main__":
    test_add()
```

要点：

- `kernel[None, num_cores](...)` —— `None`是默认stream，`num_cores`是核数。
- `pl.get_block_num()` / `pl.get_block_idx()`给出核总数 / 当前核。
- `for i in pl.range(core_id, m_tile_num, num_cores)` —— 标准跨步多核切分。
- 这里`M_SIZE`恰好被`TILE_M`整除；不整除时把`num_cores`改成
  `min(info.core_num, ceildiv(M, TILE_M) * ceildiv(N, TILE_N))`并按[尾块文档](tail_block_handling.md)
  处理边界。

### matmul的多核切分

matmul的切分思路一样：把输出`[M, N]`按`128 × 128`分块，外层按输出行块跨步分核，
内层遍历输出列块。区别在于每块tile走的是Cube数据流
`GM →(MTE2)→ Mat →(MTE1)→ Left/Right →(M)→ Acc →(FIX)→ GM`，用`pl.section_cube()` +
TileGroup + `auto_mutex`免手写同步（`matmul/test_matmul_8K_example.py`）：

```python
@pl.jit(auto_mutex=True)
def matmul_example(
    a:   pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b:   pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    num_cores = pl.get_block_num()
    core_id   = pl.get_block_idx()
    M = a.shape[0]
    N = b.shape[1]

    with pl.section_cube():
        a_mat = pl.make_tile_group(
            type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0,       mutex_ids=[0, 1, 10, 11])   # 4 缓冲：更多在途的 GM->L1 加载
        b_mat = pl.make_tile_group(
            type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0x20000, mutex_ids=[2, 3, 12, 13])
        a_left  = pl.make_tile_group(
            type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),  addrs=0, mutex_ids=[4, 5])
        b_right = pl.make_tile_group(
            type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right), addrs=0, mutex_ids=[6, 7])
        acc     = pl.make_tile_group(
            type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),   addrs=0, mutex_ids=[8, 9])

        # 每个核跨步认领输出行块；内层遍历输出列块。
        for i in pl.range(core_id, M // 128, num_cores):
            for j in pl.range(0, N // 128, 1):
                a_l1 = a_mat.next(); pl.load_tile(a_l1, a, [i, 0])
                b_l1 = b_mat.next(); pl.load_tile(b_l1, b, [0, j])
                cur_a = a_left.next();  pl.move(cur_a, a_l1)      # L1 -> L0A
                cur_b = b_right.next(); pl.move(cur_b, b_l1)      # L1 -> L0B
                acc_t = acc.next()
                pl.matmul(acc_t, cur_a, cur_b)                    # L0C = A @ B
                pl.store_tile(out, acc_t, [i, j])                 # L0C -> GM
```

> 上面为简洁只切了K = 128的单步；真实matmul还要在K方向累加（`pl.matmul_acc`）。这里的
> 重点是**多核切分与Vec完全一致**：`for i in pl.range(core_id, M//128, num_cores)`。矩阵
> 乘、L1/L0摆放的细节见[Tensor、Tile与TileGroup](Python_programming_overview.md)。

想看**多核 + TilingData + 尾块**三者结合的完整例子，见
`test_eltwise_dynamic_rank.py`（也在[尾块Tile文档](tail_block_handling.md) §6详解）。

---

<a id="常见坑"></a>

## 常见坑

- **`block_dim`超过硬件核数。**虽然`_clamp_block_dim`会截断并告警，但请在host侧就用
  `min(get_platform_info().core_num, total_tiles)`算好，别依赖兜底。
- **核数多于tile数。**多出来的核在`pl.range(core_id, total_tiles, num_cores)`里一次都
  不进循环，纯空转。用`num_cores = min(core_num, total_tiles)`。
- **连续分段导致不均衡。**手写`[k*chunk:(k+1)*chunk]`且不处理余数时，最后一个核可能显
  著多做或少做。优先用跨步`pl.range(core_id, total, num_cores)`。
- **把TilingData字段当编译期常量用。**tiling字段是运行时值，不能用在需要编译期
  `int`的地方（例如`TileType`的静态`shape`）；用在`pl.range`、`make_tensor`的
  shape/stride、算术、`if`条件里。
- **TilingData必须是最后一个参数。**`_pack_tiling_arg`只识别args的最后一个元素是否为
  tiling实例。

---

## 速查

```python
# --- kernel 里：认领本核的工作 ---
num_cores = pl.get_block_num()      # == launch 的 block_dim
core_id   = pl.get_block_idx()      # 0 .. num_cores-1
sub_id    = pl.get_subblock_idx()   # 0/1，Vector子核

total_tiles = m_tiles * n_tiles
for idx in pl.range(core_id, total_tiles, num_cores):   # 跨步均衡切分
    ...

# --- launch：设核数 ---
from pypto_pro.runtime.platform import get_platform_info
num_cores = min(get_platform_info().core_num, total_tiles)
kernel[None, num_cores](x, y, z)                  # [stream, block_dim]

# --- 传 tiling 数据 ---
# (a) 动态 shape：签名用 pl.DYNAMIC，kernel 内用 x.shape[0] 取运行时维度
# pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16]
# (b) 标量：签名加 pl.DT_*，launch 传值
def k(a, out, scale: pl.DT_INT32): ...
# (c) TilingData：@dataclass，作为最后一个参数
@dataclass
class T:
    shape: int[4]
    flags: int[8]
tiling = T(shape=[1,1,513,513], flags=[0]*8)
kernel[None, num_cores](x, y, z, tiling)
```
