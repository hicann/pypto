# vf.create_mask

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

创建掩码寄存器（MaskReg），指定参与后续VF运算的元素范围。

### 掩码寄存器（MaskReg）工作原理

掩码寄存器是VF运算中控制元素级有效性的专用寄存器。VF算子（如`vf.add`、`vf.mul`等）在执行时，会根据掩码寄存器中每个元素的对应比特位决定该元素是否参与运算：

- **比特位为1（有效）**：该元素参与运算，结果写入目的寄存器对应位置。
- **比特位为0（无效）**：该元素不参与运算，目的寄存器对应位置的行为由算子的`mode`参数决定（`MODE_ZEROING`置零，`MODE_MERGING`保留原值）。

掩码寄存器的总位宽为256 bit，其粒度由`dtype`参数决定；每个数据元素对应的掩码位数随元素位宽变化。例如：

| dtype | 元素位宽 | 元素个数 | 每元素掩码位数 | 总掩码位数 |
|---|---|---|---|---|
| `DT_INT8` / `DT_UINT8` | 8 bit | 256 | 1 bit（b8粒度） | 256 bit |
| `DT_FP16` / `DT_UINT16` / `DT_BF16` | 16 bit | 128 | 2 bit（b16粒度） | 256 bit |
| `DT_FP32` / `DT_INT32` / `DT_UINT32` | 32 bit | 64 | 4 bit（b32粒度） | 256 bit |
| `DT_INT64` / `DT_UINT64` | 64 bit | 32 | 8 bit（b64粒度） | 256 bit |

> [!CAUTION]注意
> `dtype`参数决定的是掩码粒度（即掩码寄存器中每多少个bit对应一个数据元素），而非掩码寄存器本身的类型。掩码寄存器始终为`MaskReg`类型。

### MaskPattern模式说明

`pattern`参数决定掩码寄存器中哪些元素被设置为有效（1），哪些被设置为无效（0）：

| 取值 | 含义 | 示意（以DT_FP32 / 64元素为例） |
|---|---|---|
| `ALL` | 所有元素有效 | `1111111111111111...1111`（全1） |
| `ALLF` | 所有元素无效 | `0000000000000000...0000`（全0） |
| `VL1` | 最低1个元素有效 | `1000000000000000...0000` |
| `VL2` | 最低2个元素有效 | `1100000000000000...0000` |
| `VL4` | 最低4个元素有效 | `1111000000000000...0000` |
| `VL8` | 最低8个元素有效 | `1111111100000000...0000` |
| `VL16` | 最低16个元素有效 | 前16个1，其余0 |
| `VL32` | 最低32个元素有效 | 前32个1，其余0 |
| `VL64` | 最低64个元素有效 | 前64个1，其余0 |
| `VL128` | 最低128个元素有效 | 全部有效（仅b8/b16粒度下有意义） |
| `H` | 最低一半元素有效 | 前32个1，后32个0（64元素时） |
| `Q` | 最低四分之一元素有效 | 前16个1，后48个0（64元素时） |
| `M3` | 3的倍数位置有效 | 每第3个元素为1 |
| `M4` | 4的倍数位置有效 | 每第4个元素为1 |

### 掩码寄存器的典型使用场景

1. **全量运算**：`pattern=ALL`，所有元素参与运算（最常用）。
2. **尾块处理**：当数据长度不是寄存器宽度的整数倍时，用`VL1`~`VL128`限制最后一块的参与元素数。
3. **条件选择**：通过`vf.eq`、`vf.gt`等比较算子生成掩码，再用`vf.select`按掩码选择元素。
4. **交替处理**：用`H`、`Q`、`M3`、`M4`等模式对寄存器中的部分元素进行筛选运算。

## 函数原型

```python
preg = vf.create_mask(*, pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `pattern` | 输入 | 掩码模式，取值见上方「MaskPattern模式说明」表格。默认``pl.MaskPattern.ALL`` |
| `dtype` | 输入 | 掩码对应的数据类型，决定掩码粒度（即每多少bit对应一个数据元素）。如`pl.DT_FP32`对应b32粒度（64元素 × 4 bit），`pl.DT_FP16`对应b16粒度（128元素 × 2 bit），`pl.DT_UINT8`对应b8粒度（256元素 × 1 bit），默认`pl.DT_FP32` |

## 数据类型

| dst |
|---|---|
| MaskReg |

## 返回值说明

返回一个掩码寄存器，供后续VF算子使用。

## 约束说明

- 本接口操作数为寄存器，不涉及地址对齐。
- 本接口不修改全局寄存器的值。

## 调用示例

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu


@pl.vector_function
def example_vf(src_tile, dst_tile):
    # vf 是 @pl.vector_function 函数内的保留命名空间，无需 import
    # create_mask 创建掩码寄存器，供后续算子做掩码控制
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg = vf.load_align(src_tile, 0)
    vf.store_align(dst_tile, reg, preg)


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
