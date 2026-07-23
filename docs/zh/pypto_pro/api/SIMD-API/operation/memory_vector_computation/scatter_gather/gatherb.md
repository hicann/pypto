# pypto_pro.language.gatherb

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
<!-- end id3 -->

## 功能说明

按 32 字节块的字节偏移聚合：从源 tile 中按 offsets 指定的字节偏移，每次取 32 字节（如 16 个 FP16 元素）块，拼入目标 tile。

## 函数原型

```python
pypto_pro.language.gatherb(out, src, offsets)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 目标 tile，按字节偏移聚合结果 |
| `src` | 输入 | 源 tile |
| `offsets` | 输入 | 偏移 tile（字节偏移），指定每个 32 字节块从源中读取的位置 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 数据类型：b8、b16、b32、b64<br>shape 须与 `src` 一致 |
| `src` | 输入 | 数据类型：与 `out` 一致<br>shape：与 `out` 一致 |
| `offsets` | 输入 | 数据类型：`pypto_pro.language.DT_UINT32`<br>shape：`[行数, 列数 / BLOCK_ELEMS]`，其中 `BLOCK_ELEMS = 32 / dtype_size`<br>值须为合法的字节偏移（0 ≤ offset < 源 tile 总字节数），越界行为不确定<br>每次取 32 字节（如 FP16 时为 16 个元素） |

## 流水类型

V（向量计算流水）。

## 调用示例

下面是一个完整 kernel：用 identity offset（行顺序字节偏移）从 64×128 FP16 源 tile 中 gatherb，结果应复现源数据。纯 vector kernel 使用 `make_tile_group` 管理 Tile 资源，并通过 `auto_mutex` 完成流水同步。

```python
import pypto_pro.language as pl

DTYPE_SIZE = 2          # FP16
BLOCK_BYTES = 32
BLOCK_ELEMS = BLOCK_BYTES // DTYPE_SIZE   # 16
ROWS, COLS = 64, 128
OFFSETS_PER_ROW = COLS // BLOCK_ELEMS     # 8


@pl.jit(auto_mutex=True)
def gatherb_kernel(
    src: pl.Tensor[[64, 128], pl.DT_FP16],
    offsets: pl.Tensor[[64, 8], pl.DT_UINT32],
    dst: pl.Tensor[[64, 128], pl.DT_FP16],
):
    tile_src_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000, mutex_ids=[0])
    tile_offsets_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 8], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x4000, mutex_ids=[1])
    tile_dst_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x4800, mutex_ids=[2])
    with pl.section_vector():
        tile_src = tile_src_group.current()
        tile_offsets = tile_offsets_group.current()
        tile_dst = tile_dst_group.current()
        pl.load(tile_src, src, [0, 0])
        pl.load(tile_offsets, offsets, [0, 0])
        pl.gatherb(tile_dst, tile_src, tile_offsets)
        pl.store(dst, tile_dst, [0, 0])
```

实测结果示例如下：

<!-- pypto-doc-output:gatherb:start -->
```bash
输入数据src：[[1 1.25 1.5 1.75 2 2.25 2.5 2.75 ...], [33 33.25 33.5 33.75 34 34.25 34.5 34.75 ...], [65 65.25 65.5 65.75 66 66.25 66.5 66.75 ...], [97 97.25 97.5 97.75 98 98.25 98.5 98.75 ...], ...]
输入数据offsets：[[0 32 64 96 128 160 192 224], [256 288 320 352 384 416 448 480], [512 544 576 608 640 672 704 736], [768 800 832 864 896 928 960 992], ...]
输出数据dst：[[1 1.25 1.5 1.75 2 2.25 2.5 2.75 ...], [33 33.25 33.5 33.75 34 34.25 34.5 34.75 ...], [65 65.25 65.5 65.75 66 66.25 66.5 66.75 ...], [97 97.25 97.5 97.75 98 98.25 98.5 98.75 ...], ...]
```
<!-- pypto-doc-output:gatherb:end -->
