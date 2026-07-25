# Tile矢量计算编程

本章介绍基于Tile的Vector侧编程方法，包括Tile分配、数据搬运、双缓冲流水、尾块处理和多核切分。

## Tile分配

### make_tile —— 分配单个tile

`pl.make_tile`分配一块固定的片上buffer。指定`addr`时**必须**同时指定`size`（buffer的字节大小）：

```python
tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
tile_a = pl.make_tile(tt, addr=0x0000, size=8192)
tile_b = pl.make_tile(tt, addr=0x2000, size=8192)
tile_out = pl.make_tile(tt, addr=0x4000, size=8192)
```

`size`是buffer的字节大小：`prod(shape) * dtype_bytes`。例如`[64, 64]`的FP16 tile = `64*64*2 = 8192`字节。

### make_tile的手动同步

由`make_tile`创建的tile只是一块裸buffer，框架**不会**为它插入任何跨pipe的同步。当一个tile在某条硬件pipe上被生产（例如MTE2加载），又在另一条pipe上被消费（例如V计算）时，**必须自己**用`pl.system.sync_src`/`pl.system.sync_dst`插入同步：

```python
with pl.section_vector():
    pl.load(tile_a, a, [0, 0])
    pl.load(tile_b, b, [0, 0])
    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
    pl.add(tile_out, tile_a, tile_b)
    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
    pl.store(out, tile_out, [0, 0])
```

- `sync_src(set_pipe, wait_pipe, event_id)` —— SET flag（生产方）
- `sync_dst(set_pipe, wait_pipe, event_id)` —— WAIT flag（消费方）
- `event_id` —— flag id；仅在上一次使用已被消费后才可复用同一id

Pipe类型（`pl.PipeType`）：`MTE2`（GM→L1/UB加载）、`MTE1`（L1→L0搬运）、`M`（矩阵计算）、`FIX`（L0C结果搬运）、`V`（向量计算）、`MTE3`（UB→GM存储）等。

## TileGroup —— 自动同步的双缓冲/N缓冲

`pl.make_tile_group`声明一组轮转的tile，是实现双缓冲乃至N缓冲的惯用方式。相对`make_tile`的决定性优势：**无需手动`sync_src`/`sync_dst`**。配合`auto_mutex=True`，框架会在每次使用轮转tile的前后自动发出`mutex_lock`/`mutex_unlock`。

```python
tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)

# 双缓冲：mutex_ids长度为2
a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
b_db = pl.make_tile_group(type=tile_type, addrs=0x10000, mutex_ids=[2, 3])
c_db = pl.make_tile_group(type=tile_type, addrs=0x20000, mutex_ids=[30, 31])
```

### 游标接口

| 方法 | 游标效果 | 返回值 |
|:---|:---|:---|
| `g.next()` | 前进+1 | 新索引处的tile（`(cur+1) % N`） |
| `g.current()` | 不变 | 当前索引处的tile |
| `g.previous()` | 不变 | 前一个tile（`(cur-1) % N`） |

`next()`是主力：每次循环迭代调用一次以取"下一块buffer"，游标按N取模回绕。`current()`在同一迭代中多个算子共享同一块buffer时使用。`previous()`在不扰动游标的情况下窥视前一块buffer。

### addrs的两种写法

- **单个基地址** → tile连续排布：`base + i * slot_size`
- **地址列表**（长度 == `len(mutex_ids)`）→ 每个tile一个显式的、可不连续的地址

```python
# 连续基地址
a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])

# 离散地址
db = pl.make_tile_group(type=tile_type, addrs=[0x0, 0x10000], mutex_ids=[0, 1])
```

## 数据搬运

GM tensor与片上tile之间的数据搬运通过以下接口完成：

| 算子 | 方向 | 偏移语义 |
|:---|:---|:---|
| `pl.load` | GM → tile | 绝对元素偏移 |
| `pl.load_tile` | GM → tile | 按tile索引：offset = `index * tile_shape` |
| `pl.store` | tile → GM | 绝对元素偏移 |
| `pl.store_tile` | tile → GM | 按tile索引 |

```python
# 按tile索引：[i, j]选取第(i,j)个TILE_M x TILE_N块
pl.load_tile(tile_a, x, [i, j])      # 先目标tile，再源tensor，最后坐标
pl.store_tile(z, tile_c, [i, j])     # 先目标tensor，再源tile，最后坐标

# 绝对偏移：[row, col]是进入tensor的元素偏移
pl.load(tile_a, x, [i, j])
pl.store(z, tile_c, [i, j])
```

`load_tile`/`store_tile`会自动把每个tile坐标乘以tile的shape，因此按"tile"来索引，无需手动计算字节/元素偏移。

## 完整示例 —— 逐元素加法（双缓冲）

```python
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
    device = "npu:0"
    torch.npu.set_device(device)
    M_SIZE, N_SIZE = 8192, 4096
    num_cores = M_SIZE // TILE_M
    torch.manual_seed(0)
    x = torch.rand([M_SIZE, N_SIZE], device=device, dtype=torch.float16)
    y = torch.rand([M_SIZE, N_SIZE], device=device, dtype=torch.float16)
    z = torch.empty([M_SIZE, N_SIZE], device=device, dtype=torch.float16)

    add_kernel[None, num_cores](x, y, z)
    torch.npu.synchronize()
    torch.testing.assert_close(z, x + y)
```

要点：

- `@pl.jit(auto_mutex=True)` —— `auto_mutex`是省去`sync_src`/`sync_dst`的关键
- `kernel[None, num_cores](...)` —— 方括号launch为`[stream, block_dim]`；`None`表示默认stream

## 多核切分与Tiling

多核切分（跨步循环、launch核数、UB容量预算、host↔device传tiling数据、均衡tiling方法论）的详细说明请参考[多核切分与Tiling](multi_core_partitioning_and_Tiling.md)。

## 尾块处理

当GM上的`pl.Tensor`的shape不能被tile shape整除时，边界上会出现比tile小的"不完整块"。尾块处理涉及`valid_shape`、`set_validshape`、`pad`、`fillpad`和`compact`等参数的协同，详细说明请参考[尾块处理](tail_block_handling.md)。

## N缓冲（循环）用法

N缓冲就是一个带N个mutex id的TileGroup，在循环里用`next()`驱动。游标按N取模前进，buffer像一个环一样被复用：

```python
# 三缓冲（N=3）：游标走 0,1,2,0,1,2,...
ring = pl.make_tile_group(
    type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
    addrs=0x0000, mutex_ids=[0, 1, 2])

for t in pl.range(0, num_tiles, 1):
    buf = ring.next()            # 环中的下一个槽，并自动插入同步
    pl.load_tile(buf, src, [t, 0])
    pl.matmul(acc, buf, weights)
```

经验法则：**环深度 ≥ 预取深度 + 1**。深度为N的环最多可让`N-1`个tile同时在途，之后生产者才会覆盖消费者仍需要的buffer。

## make_tile vs make_tile_group小结

| 方面 | `make_tile` | `make_tile_group` |
|:---|:---|:---|
| 分配 | 一块固定buffer（`addr`+`size`） | N块轮转buffer（`addrs`+`mutex_ids`） |
| buffer选择 | 手动（你持有变量） | `next()/current()/previous()`游标 |
| 跨pipe同步 | **手动**`sync_src`/`sync_dst`对 | 配合`auto_mutex=True`**自动** |
| 双/N缓冲 | 手工（多个tile + 乒乓同步） | 内建（`mutex_ids`的长度） |
| 适用场景 | 紧凑、手工调优的单趟流水线 | 大多数kernel；流水化/重叠的循环 |

> [!NOTE]说明
> 新kernel优先使用`make_tile_group` + `auto_mutex=True`，它在构造上即正确，远不易出错。仅在你需要精确、手工放置flag时才使用`make_tile` + 显式同步。两者可在同一kernel中混用。

## 常见坑

> 尾块相关的常见坑请参考[尾块处理](tail_block_handling.md#常见坑)。
> 多核切分相关的常见坑请参考[多核切分与Tiling](multi_core_partitioning_and_Tiling.md#常见坑)。

## 速查

```python
# --- Tile（片上）---
tt   = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
tile = pl.make_tile(tt, addr=0x0, size=128*128*2)  # 给addr必须给size
# pipe间手动同步：
pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)

# --- TileGroup（轮转、自动同步）---
g = pl.make_tile_group(type=tt, addrs=0x0, mutex_ids=[0, 1])  # 双缓冲
a = g.next()       # 游标前进，返回下一个槽（+ 自动mutex）
c = g.current()    # 同一个槽，不前进
p = g.previous()   # 前一个槽，不前进

# --- 数据搬运 ---
pl.load(dst_tile, src_tensor, [r, c])         # 绝对偏移
pl.load_tile(dst_tile, src_tensor, [i, j])    # 按tile索引
pl.store(dst_tensor, src_tile, [r, c])
pl.store_tile(dst_tensor, src_tile, [i, j])
```
