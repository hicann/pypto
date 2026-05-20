# pypto.get\_conv\_tile\_shapes

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Ascend 950PR/Ascend 950DT |    √     |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    ×     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    ×     |

## 功能说明

获取卷积（conv）计算中设置的TileShape大小以及L0TileInfo的开关使能。

## 函数原型

```python
def get_conv_tile_shapes() -> Tuple[pypto_impl.TileL1Info, pypto_impl.TileL0Info, bool]
```

## 参数说明

无。

## 返回值说明

返回L0和L1上的TileShape大小、是否开启L0TileInfo的开关。

## 约束说明

无。

## 调用示例

```python
pypto.get_conv_tile_shapes()
```
