# vf.load_unalign_pre

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

非对齐搬入初始化接口。在读非对齐地址前，应该先通过 `vf.load_unalign_pre` 进行初始化，保存非 32 字节对齐的数据，然后再调用 `vf.load_unalign` 进行数据搬入。

连续非对齐搬入时，`vf.load_unalign` 会将后续未对齐的数据缓存至 ureg，所以下一次搬入不需要再次调用 `vf.load_unalign_pre`，只需在迭代开始前调用一次 `vf.load_unalign_pre`，从而实现非对齐搬入的性能优化。

## 函数原型

```python
vf.load_unalign_pre(align_reg, tile)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `align_reg` | 输入/输出 | 非对齐寄存器，UnalignRegForLoad 类型，用于缓存非 32 字节对齐的数据（由 `vf.load_unalign_init()` 创建） |
| `tile` | 输入 | 源 UB tile，起始地址不需要 32 字节对齐 |

## 数据类型

目的操作数与源操作数的数据类型需要保持一致。支持的数据类型为：INT8、UINT8、INT16、UINT16、FP16、BF16、INT32、UINT32、FP32、INT64、UINT64。

## 返回值说明

无

## 约束说明

- `vf.load_unalign_pre` 与 `vf.load_unalign` 接口需要组合使用。
- 连续非对齐搬入时，只需在迭代开始前调用一次 `vf.load_unalign_pre`。

## 调用示例

```python
import pypto_pro.language as pl
import torch
import torch_npu


@pl.vector_function
def example_vf(src_tile, dst_tile):
    # vf 是 @pl.vector_function 函数内的保留命名空间，无需 import
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    # 分配非对齐搬入缓存寄存器
    ureg = vf.load_unalign_init()
    # 非对齐搬入初始化，只需在迭代开始前调用一次
    vf.load_unalign_pre(ureg, src_tile)
    # 非对齐搬入
    src_reg = vf.load_unalign(ureg, src_tile, post_update=True)
    vf.store_align(dst_tile, src_reg, preg)


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
