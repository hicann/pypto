# vf.mask_reg（概念说明）

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

MaskReg（掩码寄存器）用于指示在计算过程中哪些元素参与计算，宽度为[RegTensor](../reg_tensor.md)的八分之一（VL/8）。

如下图所示，当操作数类型为b8时，每一个element对应1bit MaskReg；当操作数类型为b16时，每一个element对应2bit MaskReg，且仅2bit中的最低位是有效的；当操作数类型为b32时，每一个element对应4bit MaskReg，且仅4bit中的最低位是有效的。

**图1**MaskReg计算过程
![](../../../../figures/mask_reg_calculation.jpg "MaskReg计算过程")

mask_reg由`vf.create_mask`或`vf.update_mask`产生，作为`MaskReg`类型的参数直接传递给矢量计算API，控制哪些元素参与运算。

## 函数原型

- CreateMask接口

```python
preg = vf.create_mask(*, pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
```

- UpdateMask接口

```python
preg = vf.update_mask(scalar, *, dtype=pl.DT_FP32)
```

## 参数说明

**CreateMask参数说明**

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `pattern` | 输入 | 创建MaskReg的模式，字符串类型。取值见下表，默认``pl.MaskPattern.ALL`` |
| `dtype` | 输入 | 掩码对应的数据类型，决定掩码覆盖的元素个数 |

**MaskPattern取值说明**

| 取值 | 含义 |
|---|---|
| `ALL` | 所有元素设置为有效数据 |
| `VL1` | 最低1个元素设置为有效数据 |
| `VL2` | 最低2个元素设置为有效数据 |
| `VL4` | 最低4个元素设置为有效数据 |
| `VL8` | 最低8个元素设置为有效数据 |
| `VL16` | 最低16个元素设置为有效数据 |
| `VL32` | 最低32个元素设置为有效数据 |
| `VL64` | 最低64个元素设置为有效数据 |
| `VL128` | 最低128个元素设置为有效数据 |
| `M3` | 3的倍数设置为有效数据 |
| `M4` | 4的倍数设置为有效数据 |
| `H` | 最低一半元素设置为有效数据 |
| `Q` | 最低四分之一元素设置为有效数据 |
| `ALLF` | 所有元素设置为无效数据 |

**UpdateMask参数说明**

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `scalar` | 输入 | 标量值，其比特位定义新的掩码模式 |
| `dtype` | 输入 | 掩码对应的数据类型，决定掩码宽度（默认`pl.DT_FP32`） |

## 数据类型

MaskReg支持的数据类型为：b8、b16、b32、b64（即`pl.DT_UINT8`/`pl.DT_INT8`、`pl.DT_UINT16`/`pl.DT_FP16`/`pl.DT_BF16`、`pl.DT_UINT32`/`pl.DT_FP32`/`pl.DT_INT32`、`pl.DT_INT64`/`pl.DT_UINT64`）。

## 返回值说明

返回一个`MaskReg`实例，供后续VF算子使用。

## 约束说明

- MaskReg寄存器数量上限为8。超出限制上限的寄存器数据会写入预留的8K UB内存中，可能会引起性能劣化。编译器会自动复用生命周期结束的寄存器和预留内存，若寄存器与预留内存均存在可用空间，将优先复用寄存器。

## 关键特性

### astype精度转换中的MaskReg

不同数据类型下元素对应的mask位宽不一致，在astype进行类型转换时，MaskReg根据输入的源操作数进行有效元素筛选。

下图展示了MaskReg和RegLayout同时作用时b16和b32进行类型转换的过程：

**图2**b16到b32类型转换过程

![maskReg-b16到b32类型转换过程](../../../../figures/mask_reg_b16_to_b32_conversion.jpg)

**图3**b32到b16类型转换过程

![MaskReg-b32到b16类型转换过程](../../../../figures/mask_reg_b32_to_b16_conversion.jpg)

### UpdateMask掩码生成

UpdateMask根据当前scalarValue的值生成对应长度的有效位掩码，并自动将scalarValue减去当前向量长度以更新剩余待处理元素数量。以b16数据类型为例，掩码生成过程如下图所示：

**图4**b16数据类型下UpdateMask接口基于scalarValue的掩码生成

![maskreg-b16数据类型下UpdateMask接口基于scalerValue的掩码生成](../../../../figures/maskreg_b16_update_mask_gen.jpg)

## Mask设置方式

在Mask设置中，Reg矢量计算支持多种灵活配置方式，可根据实际计算场景选择：

| 编号 | 设置方式 | 涉及接口 | 说明 |
|---|---|---|---|
| 1 | 调用接口设置 | `vf.create_mask` | 以固定的pattern设置Mask，每次循环均使用此Mask |
| 2 | 调用接口设置 | `vf.update_mask` | 根据count生成对应长度的有效位掩码，并自动更新剩余待处理元素数量 |
| 3 | 从UB搬运 | `vf.load_align` | 从UB搬运Mask数据到MaskReg，通过`dist`关键字参数区分MaskReg目标 |
| 4 | 从RegTensor生成 | `vf.mask_gen_with_reg_tensor` | 从RegTensor的指定bit位生成MaskReg |
| 5 | 从SPR读取 | `vf.get_mask_spr` | 从SetVectorMask设置的掩码寄存器 {MASK1, MASK0} 中读取Mask值 |

## 调用示例

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu


@pl.vector_function
def example_vf(src_tile, dst_tile):
    # create_mask 以固定 pattern 设置 Mask
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg = vf.load_align(src_tile, 0)
    # update_mask 从标量值生成掩码
    preg_tail = vf.update_mask(0xFFFFFFFF, dtype=pl.DT_FP32)
    vf.store_align(dst_tile, reg, preg_tail)


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
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
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
