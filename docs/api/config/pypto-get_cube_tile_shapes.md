# pypto.get\_cube\_tile\_shapes

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

获取cube计算中设置的TileShape大小、L1大包搬运功能以及多核切K功能的开关使能。

## 函数原型

```python
get_cube_tile_shapes() -> Tuple[List[int], List[int], List[int], bool, bool]:
```

## 参数说明

void

## 返回值说明

返回包含m, k, n方向上的TileShape大小、是否开启L1的大包搬运功能以及是否开启多核切K功能。

## 约束说明

无。

## 调用示例

```python
pypto.get_cube_tile_shapes()
```

