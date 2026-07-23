# vf.mem_bar

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

内存屏障，对 `src` 类内存操作与其后的 `dst` 类内存操作施加顺序保证，确保屏障前的操作对屏障后的操作可见。对应 AscendC 的 `LocalMemBar<src, dst>`。

通过 `mode` 选择 src→dst 的类型组合，共支持 12 种合法组合（V*=矢量，*_LD/*_ST/ST_*/LD_*=标量，*_ALL=该单元全量屏障）：

| mode | src → dst 含义 |
|---|---|
| `VST_VLD` | 矢量 store → 矢量 load（默认，RAW） |
| `VLD_VST` | 矢量 load → 矢量 store（WAR） |
| `VST_VST` | 矢量 store → 矢量 store（WAW） |
| `VST_LD` | 矢量 store → 标量 load |
| `VST_ST` | 矢量 store → 标量 store |
| `VLD_ST` | 矢量 load → 标量 store |
| `ST_VLD` | 标量 store → 矢量 load |
| `ST_VST` | 标量 store → 矢量 store |
| `LD_VST` | 标量 load → 矢量 store |
| `VV_ALL` | 全部矢量 ↔ 全部矢量 |
| `VS_ALL` | 全部矢量 ↔ 全部标量 |
| `SV_ALL` | 全部标量 ↔ 全部矢量 |

## 函数原型

```python
vf.mem_bar()                              # 默认 VST_VLD
vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `mode` | 输入 | 可选，屏障模式，`pl.MemBarMode` 枚举（见上表 12 种组合）。默认 `VST_VLD` |

## 数据类型

不涉及数据类型。

## 返回值说明

无

## 约束说明

- `mode` 只能取上表 12 种合法组合之一。

## 调用示例

```python
import pypto_pro.language as pl
import torch
import torch_npu


@pl.vector_function
def example_vf(in_a, t_f0):
    # vf 是 @pl.vector_function 函数内的保留命名空间，无需 import
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_a = vf.load_align(in_a, 0)
    reg_t = vf.add(reg_a, reg_a, preg)
    vf.store_align(t_f0, reg_t, preg)
    # 写-写屏障：保证对同一 tile 的两次 store 有序
    vf.mem_bar(mode=pl.MemBarMode.VST_VST)
    reg_r = vf.load_align(t_f0, 0)
    # 全量矢量屏障
    vf.mem_bar(mode=pl.MemBarMode.VV_ALL)
    vf.store_align(t_f0, reg_r, preg)


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
    torch.testing.assert_close(out, a + a, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_example()
    print("PASSED")
```
