# pypto_pro.language.Tile

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
<!-- end id3 -->

## 功能说明

Tile 的类型标注（下标语法）。`pypto_pro.language.Tile` **不能直接通过构造函数创建**，所有 Tile 都通过 [`pypto_pro.language.make_tile`](../operation/resource_management/make_tile.md) 或 [`pypto_pro.language.make_tile_group`](../operation/resource_management/make_tile_group.md) 创建。

`pypto_pro.language.Tile` 主要用于：

1. kernel 辅助函数参数和返回值的类型标注
2. kernel 内部局部变量的类型标注
3. 通过 `=` 赋值创建 Tile 别名

Tile 的完整规格（形状、数据类型、内存空间、排布方式等）由 [`pypto_pro.language.TileType`](TileType.md) 描述。

## 函数原型

```python
pypto_pro.language.Tile[[shape], dtype]
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `shape` | 输入 | 各维大小列表，如 `[64, 128]` |
| `dtype` | 输入 | 元素数据类型，如 `pypto_pro.language.DT_FP16` |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `shape` | 输入 | 整数列表，各维大小须为正整数；当前 codegen 最多支持二维 Tile<br>须与 `make_tile` 时使用的 `TileType.shape` 一致 |
| `dtype` | 输入 | [`pypto_pro.language.DataType`](DataType.md) 枚举值<br>须与 `make_tile` 时使用的 `TileType.dtype` 一致 |

## 调用示例

以下为 kernel 函数体内的局部变量标注片段：

```python
tile_a: pl.Tile[[64, 128], pl.DT_FP16] = pl.make_tile(
    pl.TileType(shape=[64, 128], dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Vec),
    addr=0x0000,
    size=16384,
)
```

### Tile 别名

Tile 别名与原变量指向同一个 Tile，可直接用于 `load`、`move` 等 Tile 操作。重新绑定原变量不会改变已有别名的指向。

以下代码为 kernel 函数体内的使用片段，其中 `input_tensor`、`replacement_tensor` 为 Tensor 参数，`input_tile_group`、`replacement_tile_group` 为已创建的 TileGroup，`left_tile` 为已创建的 L0A Tile：

```python
input_tile = input_tile_group.current()
saved_input_tile = input_tile

# 重新绑定 input_tile 后，saved_input_tile 仍指向原 Tile
input_tile = replacement_tile_group.current()

pl.load(saved_input_tile, input_tensor, [0, 0])
pl.load(input_tile, replacement_tensor, [0, 0])
pl.move(left_tile, saved_input_tile)
```

实际分配 tile 的完整示例见 [`pypto_pro.language.make_tile`](../operation/resource_management/make_tile.md) 和 [`pypto_pro.language.make_tile_group`](../operation/resource_management/make_tile_group.md)。
