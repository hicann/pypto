# pl.reset_ctrl_spr

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

将CTRL特殊寄存器中指定比特区间恢复为硬件默认值。CTRL寄存器的默认值为`0x1000000000000008`。

通常在通过`pl.set_ctrl_spr`或`pl.set_saturation_flag`修改CTRL寄存器后，用于恢复默认状态。

## 函数原型

```python
reset_ctrl_spr(start_bit: int, end_bit: int) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `start_bit` | 输入 | 起始比特位（0-63），编译期常量。 |
| `end_bit` | 输入 | 结束比特位（0-63），编译期常量。 |

## 约束说明

- 可重置的CTRL比特位与`set_ctrl_spr`一致：6-10、45、48、50、53、59、60。
- 恢复后CTRL寄存器中指定比特区间回到默认值，其他比特位不受影响。

## 返回值说明

无

## 调用示例

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg = vf.load_align(src_tile, 0)
    # FP32→INT16，缩窄转换，layout=ZERO放偶数半区
    # 不指定saturate参数，使用set_saturation_flag设置的全局饱和模式
    reg_i16 = vf.astype(reg, preg, dtype=pl.DT_INT16, layout=pl.CastLayout.ZERO)
    # INT16→FP32，扩展回FP32用于搬出
    reg_f32 = vf.astype(reg_i16, preg, dtype=pl.DT_FP32)
    vf.store_align(dst_tile, reg_f32, preg)

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    # 设置CAST饱和模式
    pl.set_saturation_flag(mode=pl.SaturationFlagMode.CAST, enable=True)
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
    # 恢复CTRL[59]到默认值
    pl.reset_ctrl_spr(59, 59)

def test_example():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    torch.npu.set_device(device)
    # 使用超出INT16范围的值测试饱和效果
    a = torch.randn([1, 64], device=device, dtype=torch.float32) * 50000
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, 1](a, out)
    torch.npu.synchronize()
    # 饱和模式下，超出[-32768,32767]的值被钳位，再转回FP32
    expected = a.clamp(-32768, 32767).to(torch.int16).to(torch.float32)
    torch.testing.assert_close(out, expected, rtol=0, atol=1.0)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```
