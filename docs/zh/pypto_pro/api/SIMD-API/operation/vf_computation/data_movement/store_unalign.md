# vf.store_unalign

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

非对齐存储，将变长向量数据或掩码寄存器（MaskReg）数据写入 UB。配合 `vf.unalign_reg_for_store()` 和 `vf.store_unalign_post()` 使用。当 `src` 为 MaskReg 时，后端自动分派 MaskReg 非对齐存储路径。

### 非对齐搬出原理

将源寄存器 srcReg 中的非对齐数据写入 UB 地址 dstAddr，根据 ureg 当前状态，分为两种场景：

**场景一：ureg 为空**（第一次迭代）

如下图所示，从源寄存器 srcReg（256B）读取数据，并将其搬运至目标 UB 地址（dstAddr ~ 304）。处理流程如下：

① 调用 **store_unalign**，此时 ureg 内无有效数据，表示连续非对齐搬出的起始状态，将 srcReg 中对应 UB 地址 48 ~ 288 的数据写入 dstAddr。此外，srcReg 中对应 UB 地址 288 ~ 304 的数据会被写入 ureg。

② 调用 **store_unalign_post** 进行非对齐搬出后处理。将 ureg 中缓存的数据写入 UB 地址 288 ~ 304。

**图 1** 非对齐数据搬出（ureg 为空）

![](<../../../../../figures/非对齐数据搬出(ureg为空).jpg>)

**场景二：ureg 不为空**（除第一次迭代）

如下图所示，从源寄存器 srcReg（256B）读取数据，并将其搬运至目标 UB 地址（dstAddr ~ 304）。处理流程如下：

① 调用 **store_unalign**，此时 ureg 内有有效数据，系统将 ureg 中 UB 地址 32 ~ dstAddr 对应的数据与 srcReg 中 UB 地址 dstAddr ~ 288 对应的数据进行拼接，结果写入 UB 地址 dstAddr。此外，srcReg 中对应 UB 地址 288 ~ 304 的数据会被写入 ureg。

② 调用 **store_unalign_post** 进行非对齐搬出后处理。将 ureg 中缓存的数据写入 UB 地址 288 ~ 304。

**图 2** 非对齐数据搬出（ureg 不为空）

![](<../../../../../figures/非对齐数据搬出(ureg不为空).jpg>)

### 连续非对齐搬入搬出示例

**图 3** 连续非对齐搬入搬出示例（数据类型 uint32_t）

![](<../../../../../figures/连续非对齐搬入搬出示例(storeunalign).jpg>)

连续非对齐搬入时，`vf.load_unalign` 会将后续未对齐的数据缓存至 ureg，所以下一次搬入不需要再次调用 `vf.load_unalign_pre`，只需在迭代开始前调用一次 `vf.load_unalign_pre`，从而实现非对齐搬入的性能优化。

连续非对齐搬出时，下次迭代的 `vf.store_unalign` 会将本次迭代 `vf.store_unalign` 缓存至 ureg 中的数据写入 UB，所以本次迭代不需要调用 `vf.store_unalign_post` 将 ureg 数据写入 UB，只需在迭代结束后调用一次 `vf.store_unalign_post`，从而实现非对齐搬出的性能优化。

## 函数原型

```python
# 3 参数形式（无步长，legacy）
vf.store_unalign(tile, src, align_reg, *, post_update=False)

# 4 参数形式（带步长）
vf.store_unalign(tile, src, align_reg, stride, *, post_update=False)

# MaskReg 非对齐存储（src 为 MaskReg 时自动分派）
vf.store_unalign(tile, mask_reg, align_reg)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `tile` | 输出 | 目标 UB tile |
| `src` | 输入 | 源向量寄存器或 MaskReg |
| `align_reg` | 输入 | alignment tracker 寄存器（由 `vf.unalign_reg_for_store()` 创建） |
| `stride` | 输入 | 可选，存储元素个数，post_update=True 时同时作为地址更新步长（整型标量） |
| `post_update` | 输入 | 可选，`True` 时 tracker 自动累进到下一段，默认 `False` |

## 数据类型

| src | tile |
|---|---|
| INT8 | INT8 |
| UINT8 | UINT8 |
| INT16 | INT16 |
| UINT16 | UINT16 |
| FP16 | FP16 |
| BF16 | BF16 |
| INT32 | INT32 |
| UINT32 | UINT32 |
| FP32 | FP32 |
| INT64 | INT64 |
| UINT64 | UINT64 |

## 返回值说明

无

## 约束说明

- 目标地址不需要 32 字节对齐。
- 3 参数形式（legacy）在当前硬件上可能导致挂死，推荐使用 4 参数形式（带步长）。
- 必须与 `vf.store_unalign_post()` 配对使用，在 `vf.store_unalign` 之后调用以 flush 剩余未对齐字节。

## 调用示例

```python
import pypto_pro.language as pl
import torch
import torch_npu


@pl.vector_function
def example_vf(src_tile, dst_tile):
    # vf 是 @pl.vector_function 函数内的保留命名空间，无需 import
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    align_reg = vf.unalign_reg_for_store()
    reg = vf.load_align(src_tile, 0)
    # 4 参数形式：stride=64 存储 64 个 FP32 元素，post_update=True 让 tracker 累进
    vf.store_unalign(dst_tile, reg, align_reg, 64, post_update=True)
    vf.store_unalign_post(dst_tile, align_reg, 0, post_update=True)


@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a = pl.make_tile(tf, addr=0, size=256)
    t_out = pl.make_tile(tf, addr=256, size=256)
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        example_vf(in_a, t_out)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, t_out, [0, 0])


def test_example():
    device = "npu:0"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 64], device=device, dtype=torch.float32)
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_example()
    print("PASSED")
```

## MaskReg 非对齐存储示例

当 `src` 为 MaskReg 时，`vf.store_unalign` 自动分派 MaskReg 非对齐存储路径。MaskReg 32 字节数据按 b16 打包为 16 字节或按 b32 打包为 8 字节写入 UB。硬件从每 2bit(b16)/4bit(b32) 中提取最低有效位(LSB)。

```python
import pypto_pro.language as pl
import torch
import torch_npu


@pl.vector_function
def example_vf(src_tile, mask_buf_tile):
    # vf 是 @pl.vector_function 函数内的保留命名空间，无需 import
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_a = vf.load_align(src_tile, 0)
    cmp_mask = vf.ge(reg_a, 0.0, preg)
    # MaskReg 非对齐存储（pstu 指令），b32 模式将 32B MaskReg 打包为 8B 写入 UB
    ureg = vf.unalign_reg_for_store()
    vf.store_unalign(mask_buf_tile, cmp_mask, ureg)
    # 必须 flush alignment tracker 中剩余的未对齐字节，否则数据滞留不到达 UB
    vf.store_unalign_post(mask_buf_tile, ureg)


@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tu = pl.TileType(shape=[1, 64], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    in_a = pl.make_tile(tf, addr=0, size=256)
    t_mask = pl.make_tile(tu, addr=256, size=256)
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        example_vf(in_a, t_mask)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, t_mask, [0, 0])


def test_example_2():
    device = "npu:0"
    core_nums = 1
    torch.npu.set_device(device)
    # pstu b32: 32B MaskReg → 8B (2 uint32)，bit i = (a[i] >= 0)
    # 全正输入 → 全部掩码位为 1 → 打包结果非零
    a = torch.ones([1, 64], device=device, dtype=torch.float32)
    out = torch.zeros([1, 64], device=device, dtype=torch.int32)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    assert out[0, 0] != 0 or out[0, 1] != 0, "全正输入应产生非零打包掩码"
    # 全负输入 → 全部掩码位为 0 → 打包结果为零
    a = -torch.ones([1, 64], device=device, dtype=torch.float32)
    out = torch.zeros([1, 64], device=device, dtype=torch.int32)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    assert out[0, 0] == 0 and out[0, 1] == 0, "全负输入应产生零打包掩码"


if __name__ == "__main__":
    test_example_2()
    print("PASSED")
```
