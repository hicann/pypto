# pypto.distributed.create_shmem_signal

## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持

## 功能说明

为指定通信域创建一个shared memory tensor，用于不同pe之间进行同步的信号。

## 函数原型

```python
create_shmem_signal(group_name: str, n_pes: int) -> ShmemTensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| group_name   | 输入      | 集合通信操作所在通信域的名字，字符串长度: 1~128。<br> 支持的类型为：str类型。 |
| n_pes   | 输入      | 通信域中的pe总数，n_pes > 0。 <br> 同一个group_name下的创建信号张量必须保证n_pes一致。 <br> 支持的类型为int类型。 |

## 返回值说明

生成用于pe间同步的信号。

## 约束说明

1. 调用create_shmem_signal创建的shared memory tensor不可切分。

## 调用示例

```python
signal = pypto.distributed.create_shmem_signal(group_name="tp", n_pes=8)
```
