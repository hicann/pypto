# vf.store_unalign_post

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

非对齐存储后处理，flush alignment tracker 中剩余的未对齐字节。须在 `vf.store_unalign` 之后调用。

## 函数原型

```python
# 2 参数形式（无步长，legacy）
vf.store_unalign_post(tile, align_reg)

# 3 参数形式（带步长）
vf.store_unalign_post(tile, align_reg, stride, *, post_update=False)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `tile` | 输出 | 目标 UB tile |
| `align_reg` | 输入 | alignment tracker 寄存器（由 `vf.unalign_reg_for_store()` 创建） |
| `stride` | 输入 | 可选，存储元素个数，post_update=True 时同时作为地址更新步长（整型标量） |
| `post_update` | 输入 | 可选，`True` 时 tracker 自动累进，默认 `False` |

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

- 本接口操作数为寄存器，不涉及地址对齐。
- 必须在 `vf.store_unalign` 之后调用。
- 2 参数形式（legacy）在当前硬件上可能导致挂死，推荐使用 3 参数形式（带步长）。

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
    vf.store_unalign(dst_tile, reg, align_reg, 64, post_update=True)
    # 3 参数形式：stride=0 仅 flush 剩余字节，不写入新数据，post_update=True 完成 tracker 收尾
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
