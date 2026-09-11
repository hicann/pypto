# pypto.full

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-09-02T08:03:00.500Z pushedAt=2026-09-05T07:36:26.329Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Creates a tensor of the specified size, filled with the given **fill\_value**. The data type of the tensor is determined by **dtype**.

## Prototype

```python
full(size: List[int], fill_value: Union[int, float, Element], dtype: DataType, *, valid_shape: Optional[Union[List[int], List[SymbolicScalar]]] = None ) -> Tensor
```

## Parameters

| Parameter    | Input/Output | Description                                                                 |
|--------------|-----------|----------------------------------------------------------------------|
| size         | Input      | Source operand that defines the shape of the output tensor.<br>Supported data type: List[int].<br>Supported dimension range: 1 to 4 dimensions. |
| fill_value   | Input      | Source operand used to fill the values of the output tensor.<br>Supported data types: int, float, and Element.<br>When the value is of type int or float, it is automatically converted to the Element type, where int corresponds to DT_INT32 and float corresponds to DT_FP32. When other data types are required, they can be constructed through Element.<br>Data types supported by Element: DT_FP32, DT_FP16, DT_BF16, DT_INT8, DT_INT16, DT_INT32, DT_UINT8, DT_UINT16, DT_UINT32, and DT_BOOL.<br>The input must be of the same type as dtype; implicit conversion is not supported. |
| dtype        | Input      | Source operand that defines the type of the output tensor.<br>Supported data types: DT_FP32, DT_FP16, DT_BF16, DT_INT8, DT_INT16, DT_INT32, DT_UINT8, DT_UINT16, DT_UINT32, and DT_BOOL.<br>The input must be of the same type as **fill_value**; implicit conversion is not supported. |
| valid_shape  | Input      | Source operand that defines the dynamic shape of the output tensor. It is a keyword parameter used in dynamic graphs and can be omitted in static graphs.<br>Supported types: List[SymbolicScalar] and List[int]. |

## Return Value

Returns an output tensor. The data type of the tensor matches `dtype`, its shape is determined by `size`, and all elements are set to `fill_value`.

## Constraints

1. **valid\_shape** is used in dynamic graph scenarios.

    In a dynamic graph scenario, if a [5,5] tensor needs to be generated with the ViewShape set to [2,2], the framework generates [2,2] tiles through the pypto.loop loop and concatenates them by offset. In this case, if valid\_shape is not passed, the code generates a tensor filled entirely with \[2,2\] by default (for example, pypto.full\(\[2,2\], 1, pypto.DT\_INT32\)).

    However, when the total size \[5,5\] is not divisible by the tile size \[2,2\], the valid shape of the tail tile (for example, \[1,1\]) cannot be automatically inferred by the framework. For example, the last row/column may contain only one element instead of a complete \[2,2\] tile. In this case, the actual valid shape of the tail tile must be explicitly specified through valid\_shape, as follows:

    pypto.full\(\[2, 2\], 1, pypto.DT\_INT32, valid\_shape=\[pypto.min\(2, 5 - 2 \* b\_idx\), pypto.min\(2, 5 - 2 \* s\_idx\)\]\), where b\_idx and s\_idx represent the loop indices.

2. The dimension range of **size** is 1 to 4 dimensions, that is, the length range of **size** is \[1, 4\].

## Examples

### TileShape Setting Example

Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The dimensions of the TileShape must be the same as those of the output.

For example, if the input size is [m, n], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
# Valid shapes use keyword argument
x1 = 1.0 # must be 1.0; implicit conversion is not supported
y1 = pypto.full([2,2], x1, pypto.DT_FP32, valid_shape = [pypto.symbolic_scalar(2), pypto.symbolic_scalar(2)])

x2 = pypto.Element(pypto.DT_INT32,1)
y2 = pypto.full([2,2], x2, pypto.DT_INT32, valid_shape = [pypto.symbolic_scalar(2), pypto.symbolic_scalar(2)])

# In static graphs, validshape can be ignored
x3 = pypto.Element(pypto.DT_INT32,1)
y3 = pypto.full([2,2], x3, pypto.DT_INT32)
```

The results are as follows:

```python
y1 output data: [[1.0,1.0], [1.0,1.0]]
y2 output data: [[1,1], [1,1]]
y3 output data: [[1,1], [1,1]]
```
