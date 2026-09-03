# pypto_pro.language.fill_index

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

向目标tile填充从start开始的连续整数序列：out[j] = start + j。

典型场景：生成位置编码、初始化索引tile用于排序或gather操作。

## 函数原型

```python
pypto_pro.language.fill_index(
    out: Tile,
    start: Union[int, Scalar],
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| out | 输出 | 目标tile，存放生成的索引序列。shape为[1, N]（行数为1，列数为索引个数），须位于UB。 |
| start | 输入 | 起始值，支持整数或运行时整型标量表达式，生成代码时转换为out的元素类型。 |

## 返回值说明

无。索引序列写入out。

## 约束说明

- out支持DT_INT16、DT_UINT16、DT_INT32和DT_UINT32。
- out必须为单行Tile（shape第0维为1）。
- out须位于UB。
- 生成的索引序列为[start, start+1, ..., start+N-1]，其中N为out的有效列数（valid_shape[1]）。
- 必须在section_vector内调用。

## 调用示例

```python
import pypto_pro.language as pl

START = 0


@pl.jit(auto_mutex=True)
def fill_index_kernel(
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
):
    tt = pl.TileType(shape=[1, 64], valid_shape=[-1, -1],
                     dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    tile_out = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    with pl.section_vector():
        m_dim = out.shape[0]
        n_dim = out.shape[1]
        for i in pl.range(0, m_dim, 1):
            for j in pl.range(0, n_dim, 64):
                cur_out = tile_out.current()
                valid_n = pl.min(64, n_dim - j)
                pl.set_validshape(cur_out, [1, valid_n])
                pl.fill_index(cur_out, START + j)
                pl.store(out, cur_out, [i, j])
```
