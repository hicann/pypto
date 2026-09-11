# pypto.view

<!-- md-trans-meta sourceCommit=14b69d18385f30130c8cd22ddda0e75ec338bdd9 translatedAt=2026-09-02T09:08:00.002Z pushedAt=2026-09-05T07:36:26.385Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Extracts a partial view from the input tensor for subsequent computation.

## Precautions

- **`valid_shape` requires `pypto.view`**: When `valid_shape` (dynamic valid data size) needs to be specified, the `[]` slicing syntax cannot be used; the explicit `pypto.view` API must be used instead.
- The number of dimensions of the input tensor `input` and the input `shape` must be consistent.

## Prototype

```python
view(input: Tensor, shape: List[int] = None, offsets: List[Union[int, SymbolicScalar]] = None, *, valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None, dtype: DataType = None,
) -> Tensor:
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|------------------------------------------------------------------------------|
| input     | Input        | Source operand.<br>Supported data types: data types supported by PyPto.<br>Empty tensors are not supported. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| shape     | Input        | Size of the view to be extracted, which must be consistent with the number of dimensions of `input`.<br>The shape size must not exceed **2147483647** (that is, **INT32_MAX**). **`shape` supports only the List [int] type, not the SymbolicScalar type.** |
| offsets   | Input        | Offset of each dimension relative to `input` when extracting the view.<br>`offsets` must be smaller than the shape of `input`. |
| valid_shape | Input      | Valid data size of the extracted view block.<br>`valid_shape` must be smaller than the shape of `input`. In scenarios such as page_attention, when the input tensors such as kv_cache contain invalid data, the output valid shape cannot be correctly inferred and must be passed in manually. |
| dtype     | Input        | Data type of the return value, which allows the input data to be interpreted as a different data type. |

## Return Value

Returns the output tensor. The data type of the tensor is the same as that of `input`, and its shape is the size specified by the `shape` parameter. If valid\_shape is specified, the actual size is valid\_shape. If `dtype` is specified, the input is read according to `dtype`.

## Examples

- Basic usage

    ```python
    x = pypto.tensor([4, 8], pypto.DT_FP32)
    shape = [4, 4]
    offsets = [0, 4]
    y = pypto.view(x, shape, offsets)
    ```

    The results are as follows:

    ```python
    Input data x: [[1 1 2 2 3 3 4 4],
                [1 1 2 2 3 3 4 4],
                [1 1 2 2 3 3 4 4],
                [1 1 2 2 3 3 4 4]]
    Output data y: [[3 3 4 4],
                [3 3 4 4],
                [3 3 4 4],
                [3 3 4 4]]
    ```

- Adding **valid\_shape**

    ```python
    x = pypto.tensor([4, 8], pypto.DT_FP32)
    shape = [4, 4]
    offsets = [2, 4]
    valid_shape = [2, 4]
    y = pypto.view(x, shape, offsets, valid_shape)
    ```

    The results are as follows:

    ```python
    Input data x: [[1 1 2 2 3 3 4 4],
                [1 1 2 2 3 3 4 4],
                [1 1 2 2 5 5 6 6],
                [1 1 2 2 5 5 6 6]]
    Output data y: [[5 5 6 6],
                [5 5 6 6],
                [0 0 0 0],
                [0 0 0 0]]
    ```

- Specifying **dtype**

    ```python
    x = pypto.tensor([2, 2], pypto.DT_FP32)
    y = pypto.view(x, dtype=pypto.DT_INT8)
    ```

    The result is as follows:

    ```python
    Input data x:
    [[0.9405094  0.20237109],
     [0.99819463 0.13246714]]

    Output data y:
    [[  57  -59  112   63   94   58   79   62],
     [ -81 -119  127   63  119  -91    7   62]]

    ```
