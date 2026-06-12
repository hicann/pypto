# pypto.distributed.shmem_clear_signal

## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持

## 功能说明

用于清除当前pe通过create_shmem_tensor或create_shmem_signal接口创建的shared memory tensor的信号值。在清除信号之前，如果shared memory tensor是通过create_shmem_signal创建的，则不得对其执行任何视图操作（如切片、偏移等）。

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
| src   | 输入      |  要清除的shared memory tensor。 |
| pred   | 输入      | 用于控制操作执行的依赖关系张量列表。 <br> 对数据类型无要求。 |

## 返回值说明

返回一个Tensor，用于表示操作完成的依赖关系。

## 约束说明

1. 在执行shmem_clear_signal操作之前，作为src参数传入的shared memory tensor如果是通过create_shmem_signal接口创建，则不得执行任何视图操作，如切片、偏移等。

## 调用示例

- 示例：创建了一个shape = [128, 256] 的shared memory tensor，清除当前pe对应的shared memory tensor的信号值。

    ```python
    shmem_tensor = pypto.distributed.create_shmem_tensor(group_name="tp", n_pes=8, dtype=pypto.DT_FP16, shape=[128, 256])
    data_clear_dummy = pypto.distributed.shmem_clear_signal(
        src=shmem_tensor,
    )
    ```
