# pypto_pro.language.make_ptr

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

为已有裸指针创建新的元素类型视图。返回的指针与源指针底层地址相同，只改变后续指针运算和 Tensor view 使用的元素类型。

常用于把一段以字节粒度（如 `pl.Ptr[pl.DT_UINT8]`）申请的 workspace 重新解释为更宽的数据类型（如 `pl.DT_FP16`），再配合 [`pypto_pro.language.make_tensor`](make_tensor.md) 包装成可 load/store 的 tensor view。与 [`pypto_pro.language.addptr`](addptr.md) 的区别：`addptr` 平移指针起点但不改变元素类型；`make_ptr` 不移动起点只改变元素类型。

## 函数原型

```python
pypto_pro.language.make_ptr(ptr, dtype=None) -> Ptr
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `ptr` | 输入 | 裸指针，类型为 `pypto_pro.language.Ptr[dtype]`（PtrType） |
| `dtype` | 输入 | 可选的目标元素类型 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `ptr` | 输入 | 须为 `pypto_pro.language.Ptr[dtype]` 标注的裸指针；不接受 Tensor 类型 |
| `dtype` | 输入 | [`pypto_pro.language.DataType`](../../basic_data_structures/DataType.md) 枚举值或 `None`<br>传入时返回的指针以该 dtype 解释后续指针运算（相当于 reinterpret cast）<br>不传时保留源指针的元素类型 |

## 返回值

返回与源指针地址相同、元素类型为 `dtype` 的 `Ptr`。

## 调用示例

下面是一个完整 kernel：kernel 接收一段 `pl.Ptr[pl.DT_UINT8]` 的 GM workspace，用 `pypto_pro.language.make_ptr` 将其重新解释为 `pl.DT_FP16` 指针，再通过 `make_tensor` 包装成 `[64, 128]` 的 tensor view，从中 load 数据并 store 到输出 Tensor。vector kernel 开 `auto_mutex`，同步由 `make_tile_group` 自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def make_ptr_kernel(
    workspace: pl.Ptr[pl.DT_UINT8],
    out: pl.Tensor[[64, 128], pl.DT_FP16],
):
    fp16_ptr = pl.make_ptr(workspace, dtype=pl.DT_FP16)
    ws_buf = pl.make_tensor(fp16_ptr, [64, 128], [128, 1])

    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])

    with pl.section_vector():
        t = tile.current()
        pl.load(t, ws_buf, [0, 0])
        pl.store(out, t, [0, 0])
```

`make_ptr` 不传 `dtype` 时等价于身份转换，返回与源指针同类型的指针：

```python
same = pl.make_ptr(ptr)
```
