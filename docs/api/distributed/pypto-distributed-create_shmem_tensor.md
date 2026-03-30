# pypto.distributed.create_shmem_tensor

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 推理系列产品 |    √     |
| Atlas A2 推理系列产品 |    √     |

## 功能说明

为指定通信域创建一个 shared memory tensor，用于不同 pe 之间进行数据访问。

## 函数原型

```python
create_shmem_tensor(group_name: str, n_pes: int, dtype: DataType, shape: list[int]) -> ShmemTensor
```

## 参数说明


| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| group_name   | 输入      | 指定需要创建 shared memory tensor 的通信域的名字，字符串长度: 1~128。<br> 支持的类型为：str 类型。 |
| n_pes   | 输入      | 通信域中的 pe 总数，n_pes > 0。 <br> 同一个 group_name 下的创建 shared memory tensor 必须保证 n_pes 一致。 <br> 支持的类型为 int 类型。 |
| dtype   | 输入      |创建的 shared memory tensor 的数据类型。 <br> 支持的类型为 pypto 的数据类型，可选值：DT_INT32、DT_FP16、DT_FP32、DT_BF16。 |
| shape   | 输入      |创建的 shared memory tensor 的形状。 <br> 参数类型为 list[int] 类型。 <br> 运行时判断当前创建的 shared memory tensor 是否超出共享区大小，进行报错提示。 |

## 返回值说明

生成一个 shared memory tensor 用于不同 pe 之间进行数据访问。

## 约束说明

无

## 调用示例

```python
data = pypto.distributed.create_shmem_tensor(group_name="tp", n_pes=8, dtype=pypto.DT_FP16, shape=[1, 64, 128])
```
