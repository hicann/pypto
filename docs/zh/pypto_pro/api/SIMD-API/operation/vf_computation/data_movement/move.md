# vf.move

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

将源寄存器中的元素复制到目标寄存器的对应位置。支持 RegTensor 和 MaskReg 两种寄存器类型：

- **RegTensor**：对 src 中的有效元素逐个复制写入 dst 中对应位置，无效位置保留 dst 原值（MERGING 模式）。
- **MaskReg**：将 src 中的 bit 复制到 dst 中对应位置。如有输入 mask，则仅复制被 mask 选定的有效 bit，无效位置填 0。

## 函数原型

```python
# RegTensor — 有 mask（赋值形式，dst 隐式声明）
dst = vf.move(src, mask)

# RegTensor — 无 mask
dst = vf.move(src)

# MaskReg — 有 mask
dst = vf.move(src, mask)

# MaskReg — 无 mask
dst = vf.move(src)
```

> 本接口为统一接口，同时支持 RegTensor 和 MaskReg 输入。当源操作数为 MaskReg 时，目标寄存器自动推断为 MaskReg。

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `src` | 输入 | 源寄存器，RegTensor 或 MaskReg 类型 |
| `mask` | 输入 | 可选，掩码寄存器。RegTensor 时控制哪些元素参与复制；MaskReg 时控制哪些 bit 有效 |
| `mode` | 输入 | 可选，`pl.MergeMode.MERGING`（默认，仅支持的模式） |

## 数据类型

| src | dst |
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

返回目标寄存器（`RegTensor` 或 `MaskReg` 类型，与源操作数类型一致）。

## 约束说明

- RegTensor 的 move 仅支持 `MERGING` 模式（被 mask 筛选掉的元素保留 dst 原值），不支持 `ZEROING` 模式。
- 目标操作数与源操作数的数据类型需要保持一致。

## 调用示例

```python
import pypto_pro.language as pl
import torch
import torch_npu


@pl.vector_function
def example_vf(src_tile, dst_tile):
    # vf 是 @pl.vector_function 函数内的保留命名空间，无需 import
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    # RegTensor move — 将 reg_a 的内容复制到 reg_b
    reg_a = vf.load_align(src_tile, 0)
    reg_b = vf.move(reg_a, preg)
    src_mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    dst_mask = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_FP32)
    dst_mask = vf.move(src_mask)
    vf.store_align(dst_tile, reg_b, dst_mask)


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

## MaskReg 调用示例

当源操作数为 MaskReg 时，`vf.move` 将掩码的 bit 复制到目标 MaskReg，对应 `pmov` 指令。

MaskReg move（mask 模式）的机制如下图所示：b16 类型读取完整 128bit 的 {MASK1, MASK0}，将每个 bit 复制为 2bit；b32 类型读取 64bit 的 MASK0，并将每个 bit 复制为 4bit。

![](<../../../../../figures/move(mask模式)示意图.jpg>)

```python
import pypto_pro.language as pl
import torch
import torch_npu


@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg = vf.load_align(src_tile, 0)
    # 生成比较掩码：reg >= 0 的位置为 1
    mask_a = vf.ge(reg, 0.0, preg)
    # 复制掩码
    dst_mask = vf.move(mask_a)
    # 使用复制后的掩码做 abs：reg >= 0 处取 abs（即自身），否则置零
    reg_dst = vf.abs(reg, dst_mask)
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
    expected = torch.where(a >= 0, a, torch.zeros_like(a))
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_example()
    print("PASSED")
```
