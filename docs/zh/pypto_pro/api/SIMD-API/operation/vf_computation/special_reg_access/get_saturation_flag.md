# pl.get_saturation_flag

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

读取CTRL特殊寄存器中的饱和模式标志位的当前状态。返回布尔值表示指定类别的饱和模式是否开启。

## 函数原型

```python
get_saturation_flag(mode: SaturationFlagMode) -> bool
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `mode` | 输入 | 饱和模式类别，对应[SaturationFlagMode](../types/SaturationFlagMode.md)枚举。取值与`pl.set_saturation_flag`的`mode`参数一致。 |

## 约束说明

无

## 返回值说明

返回`bool`类型。`True`表示指定类别的饱和模式当前处于开启状态，`False`表示关闭状态。

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
    # 设置CAST饱和模式为开启
    pl.set_saturation_flag(mode=pl.SaturationFlagMode.CAST, enable=True)
    # 读取当前饱和模式状态
    is_enabled = pl.get_saturation_flag(mode=pl.SaturationFlagMode.CAST)
    # is_enabled为True时执行饱和转换
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
    # 恢复不饱和模式
    pl.set_saturation_flag(mode=pl.SaturationFlagMode.CAST, enable=False)

def test_example():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    # 使用超出INT16范围的值测试饱和效果
    a = torch.randn([1, 64], device=device, dtype=torch.float32) * 50000
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    # 饱和模式下，超出[-32768,32767]的值被钳位，再转回FP32
    expected = a.clamp(-32768, 32767).to(torch.int16).to(torch.float32)
    torch.testing.assert_close(out, expected, rtol=0, atol=1.0)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```
