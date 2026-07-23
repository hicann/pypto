# vf.ge

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

逐元素比较a是否大于等于b，将比较结果写入目的操作数MaskReg中对应比特位，如果比较后的结果为真，则输出结果的对应比特位为1，否则为0。

第二个参数可以是标量或RegTensor，接口自动识别并分发到对应的硬件指令（标量比较走vcmps_ge，向量比较走vcmp_ge）。

## 函数原型

```python
dst_mask = vf.ge(a, b, preg)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dst_mask` | 输出 | 目标掩码寄存器，存放比较结果 |
| `a` | 输入 | 源操作数 |
| `b` | 输入 | 比较操作数，可以是标量或RegTensor |
| `preg` | 输入 | 掩码寄存器，指定参与比较的元素范围 |
| `cmp_dtype` | 输入 | 向量比较时指定比较位宽的数据类型。若未传入，则根据 `a` 的 dtype 自动推断。例如将 UINT16 寄存器按 UINT8 宽度比较时，传入 `cmp_dtype=pl.DT_UINT8` |

## 数据类型

| a、b 数据类型 | dst_mask 数据类型 |
|---|---|
| INT8 | MaskReg |
| UINT8 | MaskReg |
| INT16 | MaskReg |
| UINT16 | MaskReg |
| FP16 | MaskReg |
| BF16 | MaskReg |
| INT32 | MaskReg |
| UINT32 | MaskReg |
| FP32 | MaskReg |
| INT64 | MaskReg |
| UINT64 | MaskReg |

## 返回值说明

返回 `MaskReg` 类型的掩码寄存器，存放比较结果。

## 约束说明

- 通过pred参数控制的未选中元素在目的操作数中被置零。
- 操作数重叠约束：a和b可以是同一个RegTensor。

## 调用示例

```python
import pypto_pro.language as pl
import torch
import torch_npu


@pl.vector_function
def example_vf(src_a, src_b, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_a = vf.load_align(src_a, 0)
    reg_b = vf.load_align(src_b, 0)
    # 向量比较
    dst_mask = vf.ge(reg_a, reg_b, preg)
    # 标量比较：dst_mask = vf.ge(reg_a, 0.0, preg)
    reg_out = vf.select(reg_a, reg_b, dst_mask)
    vf.store_align(dst_tile, reg_out, preg)


@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a = pl.make_tile(tf, addr=0, size=256)
    in_b = pl.make_tile(tf, addr=256, size=256)
    t_out = pl.make_tile(tf, addr=512, size=256)
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.load(in_b, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        example_vf(in_a, in_b, t_out)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, t_out, [0, 0])


def test_example():
    device = "npu:0"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 64], device=device, dtype=torch.float32)
    b = torch.randn([1, 64], device=device, dtype=torch.float32)
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.where(a >= b, a, b), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_example()
    print("PASSED")
```

## 另请参阅

- `vf.eq` / `vf.ne` / `vf.lt` / `vf.gt` / `vf.le`：其他比较模式
