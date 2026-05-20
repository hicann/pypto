# pypto.index\_add\__ub

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Ascend 950PR/Ascend 950DT |    √     |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

pypto.index_add_的ub版本，可参考[pypto.index_add_](pypto-index_add_.md)。

## 函数原型

```python
index_add__ub(input: Tensor, dim: int, index: Tensor, source: Tensor, *, alpha: Union[int, float] = 1) -> Tensor
```

## 约束说明

1. index必须是整数类型（DT\_INT32 或 DT\_INT64），值不超过input在dim维度上的Shape大小，维数为1，Shape大小与 source 所在dim轴的Shape大小相同；

2. dim为int类型，取值范围：-input.dim <= dim < input.dim；

3. input和source的数据类型和维数均相同；

4. input.shape和source.shape的dim轴viewshape不可切，要求viewshape\[dim\]\>=max\(input.shape\[dim\], source.shape\[dim\]\)，其余维度的Shape大小不做限制；

5. TileShape的维度与input相同，input, source 的 dim 轴以及 index 均不可切，所有输入和输出的TileShape大小总和不能超过UB内存的大小。

## 调用示例

### TileShape设置示例

调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。
如输入input为[m, n, p]，dim为1，输入source为[m, t, p]，输入index为[t]，输出为[m, n, p]，TileShape设置为[m1, t1, p1]，则m1, p1分别用于切分m, p轴。 n轴，t轴不可切，必须保证n轴t轴全载。

```python
pypto.set_vec_tile_shapes(4, 16, 32)
```