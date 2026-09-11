# pypto.Tensor Constructor

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-26T02:39:55.077Z pushedAt=2026-08-28T11:36:17.365Z -->

## Applicable Products

- Ascend 950PR/Ascend 950DT: Supported
- Atlas A3 training products/Atlas A3 inference products: Supported
- Atlas A2 training products/Atlas A2 inference products: Supported

## Description

Creates a tensor object. The tensor contains uninitialized random values upon creation.

## Prototype

```python
__init__(self,
         shape=None,
         dtype: Union[DataType, None] = None,
         name: str = "",
         format: TileOpFormat = TileOpFormat.TILEOP_ND,
         data_ptr: Optional[int] = None,
         device=None,
         ori_shape=None
)
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|------------|-----------|----------------------------------------------------------------------|
| shape      | Input      | Shape of the tensor, which can be one of the following types:<br> - **None**: Creates an empty tensor.<br> - **List[int]**: A list of integers specifying the size of each dimension.<br> - **List[Union[int, SymbolicScalar]]**: A list containing integers or symbolic scalars, used for dynamic shapes. |
| dtype      | Input      | Data type of the tensor. |
| name       | Input      | Name of the tensor. |
| format     | Input      | Format of the tensor. Available values include:<br> - **TileOpFormat.TILEOP_ND** (default)<br> - **TileOpFormat.TILEOP_NZ** |
| data_ptr   | Input      | Data pointer, which defaults to **None**. Currently used only internally by the frontend framework and can be ignored by operator developers. |
| device     | Input      | Device information, which defaults to **None**. |
| ori_shape  | Input      | Original shape, used to store the original shape information of the tensor, which defaults to **None**. |

## Return Value

Returns a tensor object.

## Constraints

None

## Example

```python
# Create an empty tensor.
empty_tensor = pypto.Tensor()

# Create a tensor with the specified shape and data type.
tensor1 = pypto.Tensor(shape=(4, 4), dtype=pypto.DT_FP32)
tensor2 = pypto.Tensor(shape=[8, 16, 32], dtype=pypto.DT_INT32)

# Create a named tensor.
named_tensor = pypto.Tensor(shape=(4, 4),
                            dtype=pypto.DT_FP32,
                            name="input_tensor" )

# Create a tensor with the specified format.
sparse_tensor = pypto.Tensor(shape=(4, 32),
                             dtype=pypto.DT_FP32,
                             format=pypto.TileOpFormat.TILEOP_NZ )

# Create a tensor with a dynamic shape (using symbolic scalars).
dynamic_shape = [pypto.SymbolicScalar("N"), 4, 8]
dynamic_tensor = pypto.Tensor(shape=dynamic_shape,
                              dtype=pypto.DT_FP32 )

# Create a tensor using the pypto.tensor convenience function (recommended).
tensor3 = pypto.tensor((4, 4), pypto.DT_FP32)
tensor4 = pypto.tensor((4, 4), pypto.DT_FP32, name="my_tensor")
```
