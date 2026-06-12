# pypto.distributed.create_shmem_tensor

## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持

## 功能说明

为指定通信域创建一个shared memory tensor，用于不同pe之间进行数据访问。

## 函数原型

```python
create_shmem_tensor(group_name: str, n_pes: int, dtype: DataType, shape: list[int]) -> ShmemTensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| group_name   | 输入      | 指定需要创建shared memory tensor的通信域的名字，字符串长度: 1~128。<br> 支持的类型为：str类型。 |
| n_pes   | 输入      | 通信域中的pe总数，n_pes > 0。 <br> 同一个group_name下的创建shared memory tensor必须保证n_pes一致。 <br> 支持的类型为int类型。 |
| dtype   | 输入      |创建的shared memory tensor的数据类型。 |
| shape   | 输入      |创建的shared memory tensor的形状。 <br> 参数类型为list[int] 类型。 <br> 运行时会检查shmem tensor的总字节大小（shape各维度乘积 × dtype字节大小）是否超过共享区限制（200MB），若超过则报错WIN_SIZE_EXCEED_LIMIT（0xA2000）。 <br> 支持的shmem Tensor维度：2 - 4维。 |

## 返回值说明

生成一个shared memory tensor用于不同pe之间进行数据访问。

## 约束说明

1. dtype支持的数据类型说明：
    - **Ascend 950PR**：DT_INT32、DT_FP16、DT_FP32、DT_BF16
    - **Atlas A3 训练系列产品/Atlas A3 推理系列产品**：DT_INT32、DT_FP16、DT_FP32、DT_BF16

## 调用示例

```python
data = pypto.distributed.create_shmem_tensor(group_name="tp", n_pes=8, dtype=pypto.DT_FP16, shape=[64, 128])
```
