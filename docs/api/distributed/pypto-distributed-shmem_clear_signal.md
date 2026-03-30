# pypto.distributed.shmem_clear_signal

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 推理系列产品 |    √     |
| Atlas A2 推理系列产品 |    √     |

## 功能说明

用于清除当前 pe 通过 create_shmem_tensor 或 create_shmem_signal 接口创建的 shared memory tensor 的信号值。在清除信号之前，如果 shared memory tensor 是通过 create_shmem_signal 创建的，则不得对其执行任何视图操作（如切片、偏移等）。

## 函数原型

```python
shmem_clear_signal(
    src: ShmemTensor,
    *,
    pred: list[Tensor] = None
) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| src   | 输入      |  要清除的 shared memory tensor。 |
| pred   | 输入      | 用于控制操作执行的依赖关系张量列表。 <br> 对数据类型无要求。 <br> 不支持空 Tensor；Shape 仅支持 2 维。 |

## 返回值说明

返回一个 Tensor，用于表示操作完成的依赖关系。

## 约束说明

1. 在执行 shmem_clear_signal 操作之前，作为 src 参数传入的 shared memory tensor 如果是通过 create_shmem_signal 接口创建，则不得执行任何视图操作，如切片、偏移等。

## 调用示例

- 示例1：创建了一个 shape = [1, 128, 256] 的 shared memory tensor，清除当前 pe 对应的 shared memory tensor 的信号值。

    ```python
    shmem_tensor = pypto.distributed.create_shmem_tensor(group_name="tp", n_pes=8, dtype=pypto.DT_FP16, shape=[1, 128, 256])
    data_clear_dummy = pypto.distributed.shmem_clear_signal(
        src=shmem_tensor,
        pred=predToken,
    )
    ```
