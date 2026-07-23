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

创建掩码寄存器（MaskReg），指定参与后续 VF 运算的元素范围。

### 掩码寄存器（MaskReg）工作原理

掩码寄存器是 VF 运算中控制元素级有效性的专用寄存器。VF 算子（如 `vf.add`、`vf.mul` 等）在执行时，会根据掩码寄存器中每个元素的对应比特位决定该元素是否参与运算：

- **比特位为 1（有效）**：该元素参与运算，结果写入目的寄存器对应位置。
- **比特位为 0（无效）**：该元素不参与运算，目的寄存器对应位置的行为由算子的 `mode` 参数决定（`MODE_ZEROING` 置零，`MODE_MERGING` 保留原值）。

掩码寄存器的总位宽为 256 bit，其粒度由 `dtype` 参数决定：每个数据元素对应 4 bit 的掩码信息。例如：

| dtype | 元素位宽 | 元素个数 | 每元素掩码位数 | 总掩码位数 |
|---|---|---|---|---|
| `DT_INT8` / `DT_UINT8` | 8 bit | 256 | 1 bit（b8 粒度） | 256 bit |
| `DT_FP16` / `DT_UINT16` / `DT_BF16` | 16 bit | 128 | 2 bit（b16 粒度） | 256 bit |
| `DT_FP32` / `DT_INT32` / `DT_UINT32` | 32 bit | 64 | 4 bit（b32 粒度） | 256 bit |
| `DT_INT64` / `DT_UINT64` | 64 bit | 32 | 8 bit（b64 粒度） | 256 bit |

> **注意**：`dtype` 参数决定的是掩码粒度（即掩码寄存器中每多少个 bit 对应一个数据元素），而非掩码寄存器本身的类型。掩码寄存器始终为 `MaskReg` 类型。

### MaskPattern 模式说明

`pattern` 参数决定掩码寄存器中哪些元素被设置为有效（1），哪些被设置为无效（0）：

| 取值 | 含义 | 示意（以 DT_FP32 / 64 元素为例） |
|---|---|---|
| `ALL` | 所有元素有效 | `1111111111111111...1111`（全 1） |
| `ALLF` | 所有元素无效 | `0000000000000000...0000`（全 0） |
| `VL1` | 最低 1 个元素有效 | `1000000000000000...0000` |
| `VL2` | 最低 2 个元素有效 | `1100000000000000...0000` |
| `VL4` | 最低 4 个元素有效 | `1111000000000000...0000` |
| `VL8` | 最低 8 个元素有效 | `1111111100000000...0000` |
| `VL16` | 最低 16 个元素有效 | 前 16 个 1，其余 0 |
| `VL32` | 最低 32 个元素有效 | 前 32 个 1，其余 0 |
| `VL64` | 最低 64 个元素有效 | 前 64 个 1，其余 0 |
| `VL128` | 最低 128 个元素有效 | 全部有效（仅 b8/b16 粒度下有意义） |
| `H` | 最低一半元素有效 | 前 32 个 1，后 32 个 0（64 元素时） |
| `Q` | 最低四分之一元素有效 | 前 16 个 1，后 48 个 0（64 元素时） |
| `M3` | 3 的倍数位置有效 | 每第 3 个元素为 1 |
| `M4` | 4 的倍数位置有效 | 每第 4 个元素为 1 |

### 掩码寄存器的典型使用场景

1. **全量运算**：`pattern=ALL`，所有元素参与运算（最常用）。
2. **尾块处理**：当数据长度不是寄存器宽度的整数倍时，用 `VL1`~`VL128` 限制最后一块的参与元素数。
3. **条件选择**：通过 `vf.eq`、`vf.gt` 等比较算子生成掩码，再用 `vf.select` 按掩码选择元素。
4. **交替处理**：用 `H`、`Q`、`M3`、`M4` 等模式对寄存器中的部分元素进行筛选运算。

## 函数原型

```python
preg = vf.create_mask(*, pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `pattern` | 输入 | 掩码模式，取值见上方「MaskPattern 模式说明」表格。默认 ``pl.MaskPattern.ALL`` |
| `dtype` | 输入 | 掩码对应的数据类型，决定掩码粒度（即每多少 bit 对应一个数据元素）。如 `pl.DT_FP32` 对应 b32 粒度（64 元素 × 4 bit），`pl.DT_FP16` 对应 b16 粒度（128 元素 × 2 bit），`pl.DT_UINT8` 对应 b8 粒度（256 元素 × 1 bit），默认 `pl.DT_FP32` |

## 数据类型

| dst |
|---|---|
| MaskReg |

## 返回值说明

返回一个掩码寄存器，供后续 VF 算子使用。

## 约束说明

- 本接口操作数为寄存器，不涉及地址对齐。
- 本接口不修改全局寄存器的值。

## 调用示例

```python
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
