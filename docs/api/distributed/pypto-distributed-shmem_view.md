# pypto.distributed.shmem_view

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 推理系列产品 |    √     |
| Atlas A2 推理系列产品 |    √     |

## 功能说明

从输入的 shared memory tensor 中提取一个部分视图，以供后续计算。

## 函数原型

```python
shmem_view(
    src: ShmemTensor,
    shape: list[int] = None,
    offsets: list[Union[int, SymbolicScalar]] = None,
    *,
    valid_shape: Optional[list[Union[int, SymbolicScalar]]] = None,
) -> ShmemTensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| src   | 输入      |  要提取局部视图的 shared memory tensor。 |
| shape   | 输入      | 需要获取的视图大小。 |
| offsets   | 输入      | 需要获取的视图偏移量。 <br> offsets 的维度应与 src 的维度一致，且每个维度的偏移量值应小于 src 对应维度的大小。 |
| valid_shape  | 输入      | 用于指定需要获取的有效数据大小。 <br> 需要保证 valid_shape 小于参数 shape。 |

## 返回值说明

返回一个 从 src 中提取的部分视图，形状为 shape 参数，如果指定了 valid_shape，则实际形状为 valid_shape。

## 约束说明

无

## 调用示例

- 示例 1：从  shared memory tensor 的提取一个部分视图，该部分视图的 shape 为 [1, 64, 64]，offset 为 [0, 0, 1]，实际获取的数据有效大小为 [1, 64, 32]。

    ```python
    shmem_tensor = pypto.distributed.create_shmem_tensor(group_name="tp", n_pes=8, dtype=pypto.DT_FP16, shape=[1, 64, 128])
    y = pypto.distributed.shmem_view(src=x, shape=[1, 64, 64], offsets=[0, 0, 1], valid_shape=[1, 64, 32])
    ```
