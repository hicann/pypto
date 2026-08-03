# vf.get_mask_spr

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

从SetVectorMask设置的掩码寄存器 {MASK1, MASK0} 中读取Mask值，并按数据类型对应的格式转换后写入返回值MaskReg。

本接口对应AscendC `MoveMask<T>`接口。具体转换方式：

- **b32类型**：读取64bit的MASK0数据，将每个bit复制为4bit，写入MaskReg。
- **b16类型**：读取完整128bit的 {MASK1, MASK0} 数据，将每个bit复制为2bit，写入MaskReg。

## 函数原型

```python
# 默认 b32 宽度
mask_reg = vf.get_mask_spr()

# 指定 b16 宽度
mask_reg = vf.get_mask_spr(width=pl.MaskWidth.B16)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `mask_reg` | 输出 | 返回的MaskReg，从SPR {MASK1, MASK0} 读取并转换 |
| `width` | 输入 | 掩码宽度，决定读取的SPR位宽及扩展方式。`pl.MaskWidth.B32`（默认）：读取64bit MASK0，每bit扩展为4bit，对应`movp_b32()`指令；`pl.MaskWidth.B16`：读取128bit {MASK1, MASK0}，每bit扩展为2bit，对应`movp_b16()`指令 |

## 数据类型

| width | 返回值 |
|---|---|
| b32 | MaskReg（b32） |
| b16 | MaskReg（b16） |

## 返回值说明

返回MaskReg类型

## 约束说明

- 本接口为兼容性接口，建议优先采用`vf.create_mask`和`vf.update_mask`进行MaskReg计算。
- 使用前需选择与掩码含义一致的模式，并通过`pl.set_vec_mask`或产生掩码的VF指令设置SPR {MASK1, MASK0}。按位掩码使用norm模式；元素计数使用count模式。

## 调用示例

先用`pl.set_vec_mask`设置SPR {MASK1, MASK0}，再用`vf.get_mask_spr`读取到MaskReg，最后用该MaskReg控制计算：

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu


@pl.vector_function
def example_vf(src_tile, dst_tile):
    # vf 是 @pl.vector_function 函数内的保留命名空间，无需 import
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg = vf.load_align(src_tile, 0)
    # 从 SPR 读取掩码到 MaskReg（movp_b32 指令）
    spr_mask = vf.get_mask_spr(width=pl.MaskWidth.B32)
    # 使用读取的掩码做 abs：前 32 个元素取 abs，其余置零
    reg_dst = vf.abs(reg, spr_mask)
    vf.store_align(dst_tile, reg_dst, preg)


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
        pl.set_mask_norm()
        pl.set_vec_mask(0, 0xFFFFFFFF)
        example_vf(in_a, t_out)
        pl.reset_mask()
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, t_out, [0, 0])


def test_example():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 64], device=device, dtype=torch.float32)
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    # 前 32 个元素取 abs，后 32 个置零
    expected = torch.zeros_like(a)
    expected[:, :32] = torch.abs(a[:, :32])
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_example()
    print("PASSED")
```
