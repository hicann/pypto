# pypto.distributed.shmem_clear_data

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 推理系列产品 |    √     |
| Atlas A2 推理系列产品 |    √     |

## 功能说明

用于清除当前 pe 对应的 shared memory tensor 的部分视图

## 函数原型

```python
shmem_clear_data(
    src: ShmemTensor,
    shape: list[int] = None,
    offsets: list[Union[int, SymbolicScalar]] = None,
    *,
    pred: list[Tensor] = None,
) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| src   | 输入      |  要清除的 shared memory tensor。|
| shape   | 输入      | 需要清除的视图大小。 <br> 参数类型为 list[int] 类型。 |
| offsets   | 输入      | 需要清除的视图的偏移量。 <br> 支持 int 或 SymbolicScalar 类型的列表。 <br> offsets 的维度应与 src 的维度一致，且每个维度的偏移量值应小于 src 对应维度的大小。 |
| pred   | 输入      | 用于控制操作执行的依赖关系张量列表。 <br> 对数据类型无要求。 <br> 不支持空 Tensor；Shape 仅支持 2 维。 |

## 返回值说明

返回一个 Tensor，用于表示操作完成的依赖关系。

## 约束说明

无

## 调用示例

- 示例 1：创建了一个 shape = [1, 128, 256] 的 shared memory tensor，清除当前 pe 对应的 shared memory tensor 的部分视图的数据。该部分视图的 shape 为 [1, 128, 128], offsets 为 [0, 0, 0]。

    ```python
    shmem_tensor = pypto.distributed.create_shmem_tensor(group_name="tp", n_pes=8, dtype=pypto.DT_FP16, shape=[1, 128, 256])
    data_clear_dummy = pypto.distributed.shmem_clear_data(
        src=shmem_tensor,
        shape=[1, 128, 128],
        offsets=[0, 0, 0],
        pred=predToken,
    )
    ```

- 示例 2：创建了一个 shape = [1, 128, 256] 的 shared memory tensor，清除当前 pe 对应的 shared memory tensor 的全部视图的数据。

    ```python
    shmem_tensor = pypto.distributed.create_shmem_tensor(group_name="tp", n_pes=8, dtype=pypto.DT_FP16, shape=[1, 128, 256])
    data_clear_dummy = pypto.distributed.shmem_clear_data(
        src=shmem_tensor,
        pred=predToken,
    )
    ```
