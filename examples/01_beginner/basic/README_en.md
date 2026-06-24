# Basic Operations

This sample demonstrates the basic operator operations and typical programming patterns of PyPTO. It is very suitable for beginners to quickly get started.

## Overview

This sample covers the core concepts in PyPTO programming, including:
- Tensor creation and attribute access.
- Basic element-wise arithmetic operations (addition, scalar multiplication, and so on).
- Matrix multiplication (Matmul) operations.
- Reduction operations (such as Sum).
- Tiling configuration (Vec and Cube Tile Shapes).
- Transform operations (View and Assemble).

## Sample Code Features

This sample highlights the following features of PyPTO:
- **JIT Compilation**: Use the `@pypto.frontend.jit` decorator to define kernel functions that execute on the NPU.
- **PyTorch Integration**: Directly accept PyTorch NPU tensors as input and output, seamlessly integrating with existing deep learning workflows.
- **Explicit Tiling Control**: Manually optimize hardware execution efficiency through `set_vec_tile_shapes` and `set_cube_tile_shapes`.
- **Symbolic Tensors**: Define tensor shapes and types even outside kernel functions.

## Code Structure

- **`basic_ops.py`**: Main script containing all samples, serving as the entry point for a quick overview.
  - `test_tensor_creation()`: Sample 1 - Tensor creation.
  - `test_elementwise_ops()`: Sample 2 - Element-wise operations (addition, scalar multiplication).
  - `test_matmul()`: Sample 3 - Matrix multiplication.
  - `test_reduce_ops()`: Sample 4 - Reduction operations (Sum).
  - `test_tiling_config()`: Sample 5 - Tiling configuration.
  - `test_transform_ops()`: Sample 6 - Transform operations (View and Assemble).
- **`tensor_creation.py`**: Tensor creation operation samples, including creation methods such as Arange, Full, and data types.
- **`symbolic_scalar.py`**: Sample demonstrating the use of Symbolic Scalar.

For more detailed usage of various operators, refer to the sibling directories:
- `../compute/`: Element-wise operators, matrix multiplication, reduction operators.
- `../tiling/`: Tiling configuration strategies.
- `../transform/`: Transform operators.

## How To Run

### Environment Preparation

```bash
# Configure CANN environment variables
# After installation, configure the environment variables. Execute the following command based on the actual path of set_env.sh.
# The above environment variable configuration only takes effect in the current window. You can write the above commands into the environment variable configuration file (such as .bashrc) as needed.

# Default path installation, using root user as a sample (for non-root users, replace /usr/local with ${HOME})
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Set device ID
export TILE_FWK_DEVICE_ID=0
```

### Execute the Script

```bash
# Run all samples
python3 basic_ops.py

# Run a specific sample (for example, Sample 2: Element-wise operations)
python3 basic_ops.py elementwise_ops::test_elementwise_ops

# View the list of all available samples
python3 basic_ops.py --list
```

## Key Code Analysis

### 1. JIT Function Definition and Tiling Configuration

```python
@pypto.frontend.jit()
def elementwise_kernel(
    a: pypto.Tensor(shape, pypto.DT_FP16),
    b: pypto.Tensor(shape, pypto.DT_FP16),
    out: pypto.Tensor(shape, pypto.DT_FP16)
):
    # Set the tile shape for vector computation
    pypto.set_vec_tile_shapes(8, 8)
    # Operator combination
    out[:] = pypto.mul(pypto.add(a, b), 2.0)

```

### 2. Execute the JIT Function

```python
# Execute the JIT function
result = elementwise_kernel(a, b)
```

## Precautions

- **Data Types**: The Ascend NPU provides native hardware acceleration for FP16 and BF16. Prioritize these types in operator development.
- **Tile Size**: The choice of Tiling shapes significantly affects operator performance. Set them based on the vector or matrix computation unit size of the NPU architecture.
- **Environment**: Ensure that `torch_npu` is correctly installed and can recognize the Ascend GPU.
