# pypto.distributed.shmem_barrier_all

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 推理系列产品 |    √     |
| Atlas A2 推理系列产品 |    √     |

## 功能说明

用于在通信域内同步多个 pe，确保各 pe 在继续执行后续任务前都完成了当前的任务。

## 函数原型

```python
shmem_barrier_all(
    src: ShmemTensor,
    pred: list[Tensor] = None,
) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| src   | 输入      | 一个 shared memory tensor，用于在多个 pe 之间的传递的同步信号，确保在同步信号满足时，多个 pe 才能继续执行。 |
| pred   | 输入      | 用于控制操作执行的依赖关系张量列表。 <br> 对数据类型无要求。 <br> 不支持空 Tensor；Shape 仅支持 2 维。 |

## 返回值说明

返回一个 Tensor，用于表示操作完成的依赖关系。

## 约束说明

无

## 调用示例

```python
matmul_out = pypto.matmul(A_tile, B_tile, pypto.DT_FP16)
shmem_signal = pypto.distributed.create_shmem_signal(group_name="tp", n_pes=8)
barrier_out = pypto.distributed.shmem_barrier_all(src=shmem_signal, pred=[matmul_out])
```
