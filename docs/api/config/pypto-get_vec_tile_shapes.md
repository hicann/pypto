# pypto.get\_vec\_tile\_shapes

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

设置vector计算中的TileShape大小。

## 函数原型

```python
get_vec_tile_shapes() -> List[int]
```

## 参数说明

void

## 返回值说明

返回每个维度的TileShape大小。

## 约束说明

无。

## 调用示例

```python
pypto.get_vec_tile_shapes()
```

