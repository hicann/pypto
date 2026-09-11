# FC6XXX-FC8XXX

<!-- md-trans-meta sourceCommit=bb16a7106cfcf454f45d57218422f73e61f61ec0 translatedAt=2026-08-11T09:08:35.630Z pushedAt=2026-09-04T08:49:53.671Z -->

## FC6101 OVER_BUFFER_LIMIT

**Error Description**

During the Operation stage, the TileShape configuration exceeds hardware cache space limits, including L0A, L0B, L0C, and L1 caches.

**Possible Causes**

- L0A space limit exceeded: `tileH * tileW * tileK * sizeof(dtype) > L0A_size`.

   ```python
   # Error example - L0A space limit exceeded
   # Take Ascend 950PR/Ascend 950DT as an example: L0A_size=65536 bytes
   # FP16(sizeof=2), tileH=16, tileW=16, tileK=256
   # 16 * 16 * 256 * 2 = 131072 > 65536(L0A_size)
   tile_l0_info = pypto_impl.TileL0Info(
       tileH=16, tileW=16, tileK=256, tileN=16
   )
   ```

- L0B space limit exceeded: `tileK * tileN * sizeof(dtype) > L0B_size`.
- L0C space limit exceeded: `tileH * tileW * tileN * sizeof(FP32) > L0C_size`.
- L1 space limit exceeded: The total occupancy of input, weight, and bias in L1 exceeds L1_size. Formula: `CeilAlign(hinL1 * winL1 * kAL1 * sizeof(dtype), 32) + CeilAlign(nL1 * kBL1 * sizeof(dtype), 32) + CeilAlign(tileN * sizeof(dtype), 32) > L1_size`.

**Solution**

1. Based on the cache name (L0A/L0B/L0C/L1) and actual tile values indicated in the error log, calculate whether the occupied space exceeds the limit by referring to the buffer size of the current chip model.
2. Reduce the tile size of the corresponding dimension so that constraints such as `tileH * tileW * tileK * sizeof(dtype) ≤ L0A_size` are satisfied.
3. Refer to the TileShape space constraint description: [pypto.set_conv_tile_shapes](../../../api/config/pypto-set_conv_tile_shapes.md).

   ```python
   # Correct example - based on test_conv.py: test_conv2d_fp16_basic_with_bias
   # For Ascend 950PR/Ascend 950DT, L0A_size=65536, L0B_size=65536, L0C_size=131072
   # FP16(sizeof=2), tileH=1, tileW=16, tileK=16
   # L0A: 1 * 16 * 16 * 2 = 512 <= 65536
   # L0B: 16 * 16 * 2 = 512 <= 65536
   # L0C: 1 * 16 * 16 * 4 = 1024 <= 131072
   tile_l0_info = pypto_impl.TileL0Info(
       tileH=1, tileW=16, tileK=16, tileN=16
   )
   ```

## FC6201 EXPANDFUNC_TENSOR_OP_NULLPTR

**Error Description**

During the Tile graph partitioning stage, when setting attributes for the MMAD (Matrix Multiply-Accumulate) Operation, the tensor graph node pointers of fmap/weight/res are null.

**Possible Causes**

- The input/output tensors of the Conv Operation were not correctly passed or were released prematurely, causing tensor nodes to be missing during Tile graph expansion.

   ```python
   # Error example - In dynamic scenarios, the offset or shape of the view operation is passed abnormally, causing the fmap sub-tensor node to be incorrectly constructed during Tile graph expansion.
   input_a_view = pypto.view(input_a, [tile_batch, 16, 64], [batch_offset, 0, 0])
   # If input_a itself is None or not correctly initialized, fmapTensorPtr is empty during Tile graph expansion.
   out = pypto.conv(input_a_view, input_b, dtype, [1], [1, 1], [1], extend_params={}, groups=1)
   ```

- In dynamic shape scenarios, the sub-tensor chain constructed by view/assemble operations is abnormal, causing the tensor pointer passed to the MMAD node to be lost.

**Solution**

1. Enable the compilation debug mode to dump the compute graph, and check whether the input/output tensor nodes of the Conv Operation are complete:

   ```python
   @pypto.frontend.jit(debug_options={"compile_debug_mode": 1})
   def conv_kernel():
   ```

2. In the output directory, use pto-toolkit to open the dumped compute graph, locate the Conv Operation node that reported the error, and check whether its input tensors (fmap, weight) and output tensor (res) all exist and are non-empty.
3. If it is a dynamic shape scenario, check whether the offset and shape parameters of `pypto.view` are valid, and confirm that the sub-tensor after view can be correctly passed to `pypto.conv`.
4. If the above troubleshooting finds no issues, submit an issue or ticket via the CONV component, and attach the complete error log, dumped compute graph, and reproduction case.

## FC6202 EXPANDFUNC_TENSOR_ATTR_GET_FAILED

**Error Description**

During the Tile graph partitioning stage, the original fmap/weight shape attributes (`CONV_ORI_FMAP_SHAPE_ATTR`/`CONV_ORI_WEIGHT_SHAPE_ATTR`) of the Conv Operation are not set. This verification is triggered in NZ2NZ mode.

**Possible Causes**

- Conv is used in NZ2NZ mode, but the original fmap/weight shape attributes are not correctly set during the Operation construction process.

   ```
   # Error example - In NZ2NZ mode, Conv Operation is missing the ori_fmap_shape attribute.
   # Error log example:
   # Conv ori fmapshape should be set when InOut Tensor NZ mode.
   ```

- The Conv Operation is constructed in a non-standard way (for example, a custom pass modified the Conv node attributes), causing necessary attributes to be lost.

**Solution**

1. Confirm the runtime chip model and check whether the NZ2NZ path (`ConstructTensorGraphNZ2NZ`) is used.
2. Enable the compilation debug mode to dump the compute graph, search for Conv Operation nodes in the dumped graph, and check whether their attribute lists contain `ori_fmap_shape` and `ori_weight_shape`.
3. Confirm whether the Conv Operation is constructed via the standard `pypto.conv` API. If a custom pass is used to modify the Conv node, check whether shape attributes were accidentally deleted.
4. If the above troubleshooting finds no issues, file an issue or ticket with the CONV component, and attach the complete error log, chip model information, and reproduction case.

## FC6203 EXPANDFUNC_TILE_OP_NULLPTR

**Error Description**

During the Tile graph partitioning stage, a null pointer occurs in a newly generated node of the Tile graph, including: the current Function pointer is null, or the tile tensor pointer of fmap/weight/bias/res is null.

**Possible Causes**

- The Conv Operation is not called within a dynamic function decorated by `@pypto.frontend.jit`, causing failure to obtain the current Function pointer (`functionPtr` is null).

   ```python
   # Error example: conv not called within a JIT dynamic function, functionPtr is null.
   def not_under_jit_example(fmap, weight):
       output = pypto.conv(fmap, weight, dtype, [1, 1], [0, 0, 0, 0], [1, 1])
       return output  # functionPtr is null, triggering FC6203.
   ```

- `hasBias=True` is configured but no bias tensor is passed, or the bias tensor is lost during Tile graph expansion, causing `biasTensorPtr` to be null.

   ```python
   # Error example: hasBias is True but bias is passed abnormally in extend_params.
   extend_params = {'bias_tensor': None}  # bias is None, hasBias is marked as True but biasTensorPtr is null.
   output = pypto.conv(fmap, weight, dtype, [1, 1], [0, 0, 0, 0], [1, 1], extend_params=extend_params)
   ```

- During Tile graph expansion, the construction of fmap/weight/res sub-tensor nodes at the L0 level fails.

**Solution**

1. Confirm that the `pypto.conv` call is inside a function decorated with `@pypto.frontend.jit`. It cannot be directly called in a regular Python function.
2. If using bias, confirm that `extend_params['bias_tensor']` is a valid tensor object, not None.
3. Enable the compilation debug mode to dump the compute graph, and check whether the tensor connection relationships of all nodes after Tile graph expansion are complete.
4. If the above troubleshooting finds no issues, submit an issue or ticket through the CONV component, and attach the complete error log and reproduction case.

## FC6204 EXPANDFUNC_PARAMS_INVALID

**Error Description**

During the Tile graph partitioning stage, the number of input operands of the Conv Operation does not match the expected value. Expected operand count = 2 (fmap + weight) + hasBias (0 or 1).

**Possible Causes**

- The operands of the Conv Operation were abnormally added or removed during graph transfer, for example, a custom pass incorrectly added or deleted input edges of the Conv Operation.

   ```
   # Error example - The operand count of the Conv Operation does not match the hasBias flag.
   # Error log example:
   # Operand vector size mismatch: Expected size: 3, actual size: 2, Conv Common Input: 2, hasBias: True
   ```

- Abnormal bias operand passing logic: The `hasBias` flag is **True** but the actual operands lack a bias, or `hasBias` is **False** but an extra bias exists in the operands.

**Solution**

1. Enable the compilation debug mode to dump the compute graph, locate the erroneous Conv Operation node in the dumped graph, and check the number and types of its input edges.
2. Confirm that the number of input operands is consistent with the `hasBias` flag: when there is no bias, the number of operands is 2 (fmap + weight); when there is a bias, the number of operands is 3 (fmap + weight + bias).
3. If a custom pass is used, check whether the input edges of the Conv Operation have been mistakenly modified.
4. If the above troubleshooting finds no issues, submit an issue or ticket through the CONV component, and attach the complete error log, dumped compute graph, and reproduction case.

## FC6205 EXPANDFUNC_INNER_STATUS_FAILED

**Error Description**

During the Tile graph partitioning stage, an internal function returns an abnormal value. This error code is a reserved error code.

**Possible Causes**

N/A

**Solution**

1. This error code is a reserved code and is not used in the current version. If you encounter this error, submit an issue or ticket through the CONV component, and attach the complete error log and reproduction case.

## FC6301 CODEGEN_GET_ATTR_FAILED

**Error Description**

During the Codegen code generation stage, obtaining the **CopyInMode** or **CopyOutMode** attribute of Conv TileOp fails. These two attributes are set during the Tile graph expansion stage and read during the Codegen stage.

**Possible Causes**

- The CopyInMode/CopyOutMode attributes were not correctly set during the Tile graph expansion stage, usually caused by an abnormal Tile graph expansion process.

   ```
   # Error example: Failed to read the CopyInMode attribute during the Codegen stage.
   # Error log example:
   # GenMemL1CopyInConv get CopyInMode failed.
   # GenMemL0CCopyOutConv get CopyOutMode failed.
   ```

- The Load/Store operation nodes of the Conv Operation are abnormally modified during the pass stage, causing attribute loss.

**Solution**

1. Enable the compilation debug mode to dump the compute graph, locate the Load (CopyIn)/Store (CopyOut) operation nodes of Conv in the dumped graph, and check whether their attributes contain `COPY_IN_MODE`/`COPY_OUT_MODE`.
2. Check whether a custom pass has modified the Load/Store nodes of Conv, and confirm that the attributes have not been accidentally deleted.
3. In the generated kernel code file (`TENSOR***.cpp` in the `kernel_aicore` directory), search for the corresponding TileOp call and confirm whether the parameters are complete.
4. If the above troubleshooting finds no issues, submit an issue or ticket through the Codegen component, and attach the complete error log, dumped compute graph, and reproduction case.

## FC6302 CODEGEN_CHECK_ATTR_INVALID

**Error Description**

During the Codegen code generation stage, the `CopyInMode`/`CopyOutMode` attribute value of the Conv TileOp is outside the valid range, or the `cutW` attribute is **0**.

**Possible Causes**

- The CopyInMode value is not within the range of `[ND2NZ, DN2NZ]`.

   ```
   # Error example - CopyInMode/CopyOutMode attribute value is invalid or cutW is 0.
   # Error log example:
   # GenMemL1CopyInConv CopyInMode is invalid: -1
   # GenMemL0CCopyOutConv CopyOutMode is invalid: 99
   # GenMemL0CCopyOutConv cutW should not be 0!
   ```

- The CopyOutMode value is not within the `{NZ2ND, NZ2NZ, NZ2DN}` range.
- The `cutW` attribute is **0**, causing the L0C-to-GM copy-out operation to fail to correctly split into blocks.

**Solution**

1. Enable the compilation debug mode to dump the compute graph, locate the Conv Load/Store operation nodes in the dumped graph, and check whether the `COPY_IN_MODE`/`COPY_OUT_MODE`/`CUT_W` attribute values are valid.
2. Valid values for `CopyInMode`: `ND2NZ` (1), `NZ2NZ` (2), `DN2NZ` (3); valid values for `CopyOutMode`: `NZ2ND` (0), `NZ2NZ` (1), `NZ2DN` (3).
3. Confirm that the Tile graph expansion process is not interfered with by custom passes. These attributes are automatically derived by the framework based on the input tensor format and chip model.
4. If the above troubleshooting finds no issues, submit an issue or ticket with the Codegen component, and attach the complete error log, dumped compute graph, and reproduction case.

## FC6303 CODEGEN_CHECK_DIM_INVALID

**Error Description**

During the Codegen code generation stage, the src shape or offset dimension of the Conv TileOp does not match the expected value. The shape/offset of a 2D conv should be 4-dimensional, that of a 3D conv should be 5-dimensional, and the valid shape of L0C should be 2-dimensional.

**Possible Causes**

- During the Tile graph expansion stage, the generated tensor shape dimensions are inconsistent with the convolution type. For example, the fmap shape of a 2D conv is incorrectly generated as 5-dimensional.

   ```
   # Error example: shape/offset dimensions do not match the convolution type.
   # Error log example:
   # GenMemL1CopyInConv shape should be 4-dim! (2D conv expects 4 dimensions, but received non-4-dim.)
   # GenMemL1CopyInConv offset should be 4-dim! (2D conv expects 4 dimensions, but received non-4-dim.)
   # GenMemL0CCopyOutConv valid shape should be 2-dim! (L0C expects 2 dimensions, but received non-2-dim.)
   ```

- In dynamic shape scenarios, the dimension derivation of the valid shape is abnormal.
- The shape of the L0C tensor is not 2D (M, N), which may be caused by an L0C node construction error during Tile graph expansion.

**Solution**

1. Enable the compilation debug mode to dump the compute graph, locate the erroneous Conv Load/Store node in the dumped graph, and check the shape dimensions of its src tensor.
2. Confirm that the shape dimensions match the convolution type: 1D conv corresponds to 3D (NCL), 2D conv corresponds to 4D (NCHW), and 3D conv corresponds to 5D (NCDHW).
3. If it is a dynamic shape scenario, check whether the sub-tensor shape dimensions constructed by `pypto.view` are consistent with the original input.
4. Check the generated kernel code file (in the `kernel_aicore` directory) and confirm that the `shape`/`offset` parameter dimensions of the TileOp call are correct.
5. If the above troubleshooting finds no issues, submit an issue or ticket through the Codegen component, and attach the complete error log, dumped compute graph, and reproduction case.

## FC6401 TILEOP_TENSOR_FORMAT_FAILED

**Error Description**

During the TileOp stage, the tensor hardware FORMAT verification fails. The Conv Load operation requires the src to be in GM format and the dst to be in L1 format. The Store operation requires the src to be in L0C format and the dst to be in GM format.

**Possible Causes**

- During the Tile graph expansion stage, an incorrect memory hierarchy is assigned to the tensor. For example, the dst of a Load operation is assigned to L0C instead of L1.

```
   # Error example: src/dst hardware format mismatch for Load/Store operations.
   # Error log example (compile-time static_assert):
   # [TLoadConv Error]: Src format shoulde be GM and Dst format shoulde be L1
   # [TStoreConv Error]: Src format shoulde be L0C and Dst format shoulde be GM
   ```

- A custom pass modified the memory type attribute of the tensor, causing a src/dst format mismatch for Load/Store operations.

**Solution**

1. The verification corresponding to this error code is a compile-time `static_assert`, which reports an error during the kernel code compilation stage. In the generated kernel code file (under the `kernel_aicore` directory), search for `TLoadConv`/`TStoreConv` calls and check the tensor type declarations passed in.
2. Verify that the dst tensor of the Load operation is declared as L1 type (for example, `ConvTile<TileType::Mat, ..., Hardware::L1>`), and the src tensor is of GM type.
3. Verify that the src tensor of the Store operation is declared as L0C type, and the dst tensor is of GM type.
4. If the Tile graph expansion logic has not been modified but this error still occurs, submit an issue or ticket through the CONV component, with the kernel code file and reproduction case attached.

## FC6402 TILEOP_SHAPE_SIZE_FAILED

**Error Description**

In the TileOp stage, the shape size validation of the L0C tensor fails. The Store operation requires the L0C shape to be 2-dimensional (M, N).

**Possible Causes**

- In the Tile graph expansion stage, a non-2D shape is constructed for the L0C tensor, which may be caused by an anomaly in the output shape derivation of the Conv MMAD node.

```
   # Error example - L0C tensor shape is not 2D.
   # Error log example (compile-time static_assert):
   # L0C shape size should be 2 Dim
   ```

- A custom pass modified the shape definition of the L0C tensor.

**Solution**

1. The verification corresponding to this error code is a compile-time `static_assert`. In the generated kernel code file, search for the `TStoreConv` call and check the shape declaration of the L0C tensor.
2. Confirm that the shape of the L0C tensor is 2-dimensional, that is, `(M, N)`, where `M = tileH * tileW` and `N = tileN`.
3. Enable the compilation debug mode to dump the compute graph, and check the shape of the L0C tensor node from the MMAD output to the Store operation in the dumped graph.
4. If the above troubleshooting finds no issues, submit an issue or ticket through the CONV component, and attach the kernel code file and reproduction case.

## FC6403 TILEOP_STC_SHAPE_INVALID

**Error Description**

During the TileOp stage, the static shape (compile-time constant shape) verification fails. Some dimensions of Conv TileOp must be compile-time constants and cannot be modified at runtime.

**Possible Causes**

- Some dimensions in the TileShape configuration must be compile-time constants but are set to dynamic values, causing `std::tuple_element` to fail during compilation.

```
   # Error example: The static dimension configuration of TileShape is abnormal, for example, tileCinFmap or tileN is configured as 0, causing bufferSize = 0.
   ```

- The buffer size calculation result of TileShape is **0** or negative (for example, a dimension is configured as **0**).

**Solution**

1. Check whether all dimension values of TileShape set by `pypto.set_conv_tile_shapes` are positive integers.
2. Confirm that the dimensions in TileShape that need to be compile-time constants (such as `tileN` and `tileWout`) are correctly declared using `pypto.symbolic_scalar` in dynamic shape scenarios.
3. Enable the compilation debug mode and check whether the TileShape expansion results in the compilation log are valid.
4. If the above troubleshooting finds no issues, submit an issue or ticket via the CONV component and attach the complete error log and reproduction case.

## FC6404 TILEOP_INDEX_INVALID

**Error Description**

During the TileOp stage, the index validation for shape/stride access failed. For Conv TileOp, the shape/stride access index must be less than 5.

**Possible Causes**

- During the Tile graph expansion stage, an attempt was made to access the fifth dimension or higher (index >= 5) of a tensor shape/stride. This may be caused by an anomaly in tensor shape dimension derivation.

```
   # Error example - shape/stride access with index >= 5
   # Error log example (compile-time static_assert):
   # Idx should be less than 5
   ```

- The tensor of a 2D conv is incorrectly processed as a 3D conv, or vice versa, causing an out-of-bounds dimension access.

**Solution**

1. The corresponding verification for this error code is a compile-time `static_assert`. Check the TileOp call corresponding to the error, and verify whether the passed index parameter is valid (must be < 5).
2. Enable the debug mode for compilation to dump the compute graph, and check whether the tensor shape dimensions of the Conv Load/Store node match the convolution type (4D + pad = 5D for 2D conv, 5D + pad = 6D internal processing for 3D conv).
3. Verify that the convolution type is correctly determined: 1D conv input is 3D, 2D conv input is 4D, and 3D conv input is 5D.
4. If the above troubleshooting finds no issues, submit an issue or ticket through the CONV component, and attach the kernel code file and reproduction case.
