# Tensor Creation

<!-- md-trans-meta sourceCommit=d665ab95092c497ed3fc231aebc833bc88e9cfe6 translatedAt=2026-08-11T09:45:13.090Z pushedAt=2026-09-04T04:13:41.653Z -->

Tensor is the fundamental data structure in PyPTO, representing a multi-dimensional array that will be used in the compute graph and executed on the NPU.

In PyPTO, a tensor represents the structure and attributes of its data, which enables PyPTO to build a compute graph and optimize it before execution. A tensor contains actual values only at execution time. Values in an uninitialized tensor are random and must be initialized as needed during execution.

## Creating a Tensor

- Creating a Basic Tensor

    ```python
    # Create a tensor with shape [2, 3] and data type FP16.
    tensor = pypto.tensor([2, 3], pypto.DT_FP16, "my_tensor")
    ```

    Parameters:

    - shape: Dimension, which supports an integer list.
    - dtype: Data type stored in the tensor. It supports the DataType type. For example, DT\_FP16 indicates a 16-bit half-precision floating-point number.
    - name: Name of the tensor. It supports the string type and is optional. However, it is recommended to provide a meaningful name for the tensor to facilitate debugging and understanding of the compute graph structure.
    - format: Data layout format. It supports the TileOpFormat type and is optional. The default value is TILEOP\_ND.
                When format is explicitly specified, better performance can be achieved. The incoming torch tensor must be consistent with the format declared by pypto.Tensor.

- Creating a formatted tensor

    ```python
    #Create a tensor in NZ format.
    tensor = pypto.tensor([-1, 32], pypto.DT_FP16, "nz_tensor", pypto.TileOpFormat.TILEOP_NZ)
    ```

    Supported formats:

    - TILEOP\_ND: ND format, an N-dimensional array using row-major mode in PyPTO.
    - TILEOP\_NZ: NZ format, a special format for matrix multiplication. A two-dimensional matrix is divided into multiple fractals (the fractal size is better suited for a single Cube computation). The fractals are arranged in column-major order, forming an N-shaped layout. Each fractal is arranged in row-major order, forming a Z-shaped layout. For details, see [Data Layout Formats](https://www.hiascend.com/document/detail/en/canncommercial/latest/opdevg/Ascendcopdevg/atlas_ascendc_10_0099.html).

- Creating a tensor in a sub-function and returning it to the main function

    ```python
    def sub_function():
        # Create a tensor with shape [2, 3] and data type FP16.
        tensor = pypto.tensor([2, 3], pypto.DT_FP16, "my_tensor")
        return tensor

    def main_function():
        sub_tensor = sub_function()
    ```

- Converting a PyTorch tensor to a PyPTO tensor

    ```python
    # prepare data
    input_data = torch.rand(shape, dtype=torch.float, device='npu')
    output_data = torch.zeros(shape, dtype=torch.float, device='npu')

    #convert from torch tensor to pypto tensor
    pto_input = pypto.from_torch(input_data, "in_0")
    pto_output = pypto.from_torch(output_data, "out_0")
    ```

## Viewing Tensor Attributes

A tensor has basic attributes such as shape, data type (dtype), data layout (format), number of dimensions (dim), and name. You can query these attributes through the pypto.tensor APIs.

```python
tensor = pypto.tensor([2,3, 4], pypto.DT_FP16, "example")

# Shape
print(tensor.shape)  #[2, 3, 4]

# Data type
print(tensor.dtype)  #DataType.DT_FP16

# Dimension
print(tensor.dim)    # 3

# Format
print(tensor.format)  #TILEOP_ND

# Name
print(tensor.name)    # "example"
tensor.name = "new_name"  #Can be changed
```

## Handling Dynamic Dimension Tensors

In actual app scenarios, a tensor is usually variable-length data. You can define a tensor with a dynamic shape using the following method and mark dynamic dimensions with -1:

```python
tensor = pypto.tensor([-1, 32], pypto.DT_FP16, "dynamic")

# Print the tensor dimensions. SymbolicScalar indicates that the current shape is a symbolic scalar.
print(tensor.shape)
>>> [SymbolicScalar(RUNTIME_GetInputShapeDim(ARG_dynamic,0)), 32]
```

You can obtain the symbolic scalar of dynamic dimensions using the following method and get the specific value at runtime:

```python
b = pypto.symbolic_scalar(tensor_shape[0])
```

If a tensor inherits from a PyTorch tensor, you can define a tensor with dynamic dimensions through the dynamic\_axis = \[int\] parameter of the pypto.from\_torch API.

```python
# prepare data
input_data = torch.rand(shape, dtype=torch.float, device='npu')
output_data = torch.zeros(shape, dtype=torch.float, device='npu')

#convert from torch tensor to pypto tensor with dynamic axis
pto_input = pypto.from_torch(input_data, "in_0", dynamic_axis=[0])
pto_output = pypto.from_torch(output_data, "out_0", dynamic_axis=[0])
```
