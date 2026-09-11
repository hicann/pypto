# AI Core Exception

<!-- md-trans-meta sourceCommit=bb16a7106cfcf454f45d57218422f73e61f61ec0 translatedAt=2026-08-11T08:58:59.135Z pushedAt=2026-09-04T08:32:59.582Z -->

## Symptom

An exception occurs during AI Core kernel execution (hardware trap, execution timeout, or core hang).
```bash
[Error]: aicore exception, device_id: 6, stream_id: 47, task_id: 2, retcode: 507015, kernelName: PyPTO_matmul_add_0_mix_aic
        Rectify the fault based on the error information in the ascend log.
PyPTO error: PyPTO Inner Error. Please rectify the fault based on the error information in the ascend log. (function PyPTOExceptionInfoCallBack)
```

## Possible Causes

- The kernel code has out-of-bounds memory access.
- Data dependency edges are lost (the consumer reads before the producer finishes writing).
- Tiling/Shape parameters do not match the kernel.
- Issues in the MACHINE scheduling framework itself.

## Solution
1. Run **export ASCEND_WORK_PATH=./wk**. For details, see [Environment Variable Reference](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/maintenref/envvar/envref_07_0001.html).
2. Use a fixed CCE compilation mode.
    ```python
    # Call example
    @pypto.frontend.jit(debug_options={"compile_debug_mode": 2})
    def pypto_kernel():
    ```
3. Enable the single-commit mode to run each instruction step by step.
    ```bash
    /usr/local/Ascend/driver/tools/msnpureport config --set --singlecommit 1 -d device-id
    ```
4. Run the use case again.
5. Search for **kernel_symbol_locator.cpp** in the plog log.
    ```bash
        # Example
        grep -rn "kernel_symbol_locator.cpp" wk/log/debug/plog/plog-2341095_20260707170139095.log

        # Register information
        62:[ERROR] IDEDD(2341095,python):2026-07-07-17:01:41.620.723 [kernel_symbol_locator.cpp:583][tid:2341490] [Dump][Exception] Error register information. coreId=6, coreType=0, AIC_ERR_0=0x0 AIC_ERR_1=0x0 AIC_ERR_2=0x0 AIC_ERR_3=0x40000000 AIC_ERR_4=0x0 AIC_ERR_5=0x0 BIU_ERR_0=0x0 BIU_ERR_1=0x0 CCU_ERR_0=0x0 CCU_ERR_1=0x63851b81 CUBE_ERR_0=0x4000036 CUBE_ERR_1=0x0 IFU_ERR_0=0xde06800 IFU_ERR_1=0x212c3 MTE_ERR_0=0x3bcdf8f6 MTE_ERR_1=0x13 VEC_ERR_0=0x0 VEC_ERR_1=0x0 FIXP_ERR_0=0xbcdf8f6 FIXP_ERR_1=0x13 AIC_COND_0=0x0 AIC_COND_1=0x0
        # PC information
        64:[ERROR] IDEDD(2341095,python):2026-07-07-17:01:41.620.746 [kernel_symbol_locator.cpp:602][tid:2341490] [Dump][Exception] Error PC information. coreId=6, coreType=0, originalStartPC=0x124a00001130, fixedStartPC=0x124a00001000, originalCurrentPC=0x124a000010dc, fixedCurrentPC=0x124a000010d8, fixedPCOffset=0xd8.
        # Symbol information
        65:[ERROR] IDEDD(2341095,python):2026-07-07-17:01:41.620.750 [kernel_symbol_locator.cpp:608][tid:2341490] [Dump][Exception] Error symbol information. coreId=6, coreType=0, symbol=TENSOR_s0_Unroll1_PATH0_hiddenfunc0_8_0_4294967296+0xd8.

        # If symbol=TENSOR_s0_Unroll1_PATH0_hiddenfunc0_8_0_4294967296, it indicates that the hang occurs at the CCE file. If symbol=PyPTO_matmul_add_0_mix_aic, it indicates that the hang occurs at the framework AI Core processing source code.
    ```


7. llvm-symbolizer --obj=${aicore_kernel_bin_file_path} fixedPCOffset
    - Install llvm-symbolizer by running apt install llvm or yum install llvm.
    - After an AI Core exception is triggered, aicore_kernel_bin_file is automatically dumped to the ${ASCEND_WORK_PATH}/extra-info/data-dump/device-id/ directory.
    - fixedPCOffset, that is, 0xd8 in line 65, which is the offset of the core's CCE instruction relative to the base address of aicore_kernel_bin_file.
    ```bash
        # Example
        llvm-symbolizer --obj=wk/extra-info/data-dump/5/PyPTO_matmul_add_0_host.o 0xd8

        void pto::TMatmul<(pto::AccPhase)0, pto::Tile<(pto::TileType)4, int, 32, 32, (pto::BLayout)1, -1, -1, (pto::SLayout)1, 1024, (pto::PadValue)0, (pto::CompactMode)0>, pto::Tile<(pto::TileType)2, signed char, 32, 32, (pto::BLayout)0, -1, -1, (pto::SLayout)1, 512, (pto::PadValue)0, (pto::CompactMode)0>, pto::Tile<(pto::TileType)3, signed char, 32, 32, (pto::BLayout)0, -1, -1, (pto::SLayout)2, 512, (pto::PadValue)0, (pto::CompactMode)0>, false, true, false>(pto::Tile<(pto::TileType)4, int, 32, 32, (pto::BLayout)1, -1, -1, (pto::SLayout)1, 1024, (pto::PadValue)0, (pto::CompactMode)0>::TileDType, pto::Tile<(pto::TileType)2, signed char, 32, 32, (pto::BLayout)0, -1, -1, (pto::SLayout)1, 512, (pto::PadValue)0, (pto::CompactMode)0>::TileDType, pto::Tile<(pto::TileType)3, signed char, 32, 32, (pto::BLayout)0, -1, -1, (pto::SLayout)2, 512, (pto::PadValue)0, (pto::CompactMode)0>::TileDType, unsigned short, unsigned short, unsigned short, bool)
        /root/pto-isa/include/pto/npu/a2a3/TMatmul.hpp:51:5
        void pto::TMATMUL_IMPL<(pto::AccPhase)0, pto::Tile<(pto::TileType)4, int, 32, 32, (pto::BLayout)1, -1, -1, (pto::SLayout)1, 1024, (pto::PadValue)0, (pto::CompactMode)0>, pto::Tile<(pto::TileType)2, signed char, 32, 32, (pto::BLayout)0, -1, -1, (pto::SLayout)1, 512, (pto::PadValue)0, (pto::CompactMode)0>, pto::Tile<(pto::TileType)3, signed char, 32, 32, (pto::BLayout)0, -1, -1, (pto::SLayout)2, 512, (pto::PadValue)0, (pto::CompactMode)0>>(pto::Tile<(pto::TileType)4, int, 32, 32, (pto::BLayout)1, -1, -1, (pto::SLayout)1, 1024, (pto::PadValue)0, (pto::CompactMode)0>&, pto::Tile<(pto::TileType)2, signed char, 32, 32, (pto::BLayout)0, -1, -1, (pto::SLayout)1, 512, (pto::PadValue)0, (pto::CompactMode)0>&, pto::Tile<(pto::TileType)3, signed char, 32, 32, (pto::BLayout)0, -1, -1, (pto::SLayout)2, 512, (pto::PadValue)0, (pto::CompactMode)0>&)
        /root/pto-isa/include/pto/npu/a2a3/TMatmul.hpp:161:5
        pto::RecordEvent pto::TMATMUL<pto::Tile<(pto::TileType)4, int, 32, 32, (pto::BLayout)1, -1, -1, (pto::SLayout)1, 1024, (pto::PadValue)0, (pto::CompactMode)0>, pto::Tile<(pto::TileType)2, signed char, 32, 32, (pto::BLayout)0, -1, -1, (pto::SLayout)1, 512, (pto::PadValue)0, (pto::CompactMode)0>, pto::Tile<(pto::TileType)3, signed char, 32, 32, (pto::BLayout)0, -1, -1, (pto::SLayout)2, 512, (pto::PadValue)0, (pto::CompactMode)0>>(pto::Tile<(pto::TileType)4, int, 32, 32, (pto::BLayout)1, -1, -1, (pto::SLayout)1, 1024, (pto::PadValue)0, (pto::CompactMode)0>&, pto::Tile<(pto::TileType)2, signed char, 32, 32, (pto::BLayout)0, -1, -1, (pto::SLayout)1, 512, (pto::PadValue)0, (pto::CompactMode)0>&, pto::Tile<(pto::TileType)3, signed char, 32, 32, (pto::BLayout)0, -1, -1, (pto::SLayout)2, 512, (pto::PadValue)0, (pto::CompactMode)0>&)
        /root/pto-isa/include/pto/common/pto_instr.hpp:661:5
        void TMatmulImpl<true, (TransMode)0, true, TileTensor<int, TileOp::Layout<Std::tuple<unsigned long, unsigned long>, Std::tuple<Std::integral_constant<unsigned long, 32ul>, Std::integral_constant<unsigned long, 1ul>>, Std::tuple<Std::integral_constant<unsigned long, 32ul>, Std::integral_constant<unsigned long, 32ul>>>, (Hardware)5>, TileTensor<signed char, TileOp::Layout<Std::tuple<unsigned long, unsigned long>, Std::tuple<Std::integral_constant<unsigned long, 32ul>, Std::integral_constant<unsigned long, 1ul>>, Std::tuple<Std::integral_constant<unsigned long, 32ul>, Std::integral_constant<unsigned long, 32ul>>>, (Hardware)3>, TileTensor<signed char, TileOp::Layout<Std::tuple<unsigned long, unsigned long>, Std::tuple<Std::integral_constant<unsigned long, 32ul>, Std::integral_constant<unsigned long, 1ul>>, Std::tuple<Std::integral_constant<unsigned long, 32ul>, Std::integral_constant<unsigned long, 32ul>>>, (Hardware)4>>(TileTensor<int, TileOp::Layout<Std::tuple<unsigned long, unsigned long>, Std::tuple<Std::integral_constant<unsigned long, 32ul>, Std::integral_constant<unsigned long, 1ul>>, Std::tuple<Std::integral_constant<unsigned long, 32ul>, Std::integral_constant<unsigned long, 32ul>>>, (Hardware)5>&, TileTensor<signed char, TileOp::Layout<Std::tuple<unsigned long, unsigned long>, Std::tuple<Std::integral_constant<unsigned long, 32ul>, Std::integral_constant<unsigned long, 1ul>>, Std::tuple<Std::integral_constant<unsigned long, 32ul>, Std::integral_constant<unsigned long, 32ul>>>, (Hardware)3>&, TileTensor<signed char, TileOp::Layout<Std::tuple<unsigned long, unsigned long>, Std::tuple<Std::integral_constant<unsigned long, 32ul>, Std::integral_constant<unsigned long, 1ul>>, Std::tuple<Std::integral_constant<unsigned long, 32ul>, Std::integral_constant<unsigned long, 32ul>>>, (Hardware)4>&)
        /root/miniconda/envs/mq/lib/python3.10/site-packages/pypto/lib/include/tileop/cube/impl/mmad_impl.h:63:9
        void TMatmul<true, (TransMode)0, true, TileTensor<int, TileOp::Layout<Std::tuple<unsigned long, unsigned long>, Std::tuple<Std::integral_constant<unsigned long, 32ul>, Std::integral_constant<unsigned long, 1ul>>, Std::tuple<Std::integral_constant<unsigned long, 32ul>, Std::integral_constant<unsigned long, 32ul>>>, (Hardware)5>, TileTensor<signed char, TileOp::Layout<Std::tuple<unsigned long, unsigned long>, Std::tuple<Std::integral_constant<unsigned long, 32ul>, Std::integral_constant<unsigned long, 1ul>>, Std::tuple<Std::integral_constant<unsigned long, 32ul>, Std::integral_constant<unsigned long, 32ul>>>, (Hardware)3>, TileTensor<signed char, TileOp::Layout<Std::tuple<unsigned long, unsigned long>, Std::tuple<Std::integral_constant<unsigned long, 32ul>, Std::integral_constant<unsigned long, 1ul>>, Std::tuple<Std::integral_constant<unsigned long, 32ul>, Std::integral_constant<unsigned long, 32ul>>>, (Hardware)4>>(TileTensor<int, TileOp::Layout<Std::tuple<unsigned long, unsigned long>, Std::tuple<Std::integral_constant<unsigned long, 32ul>, Std::integral_constant<unsigned long, 1ul>>, Std::tuple<Std::integral_constant<unsigned long, 32ul>, Std::integral_constant<unsigned long, 32ul>>>, (Hardware)5>&, TileTensor<signed char, TileOp::Layout<Std::tuple<unsigned long, unsigned long>, Std::tuple<Std::integral_constant<unsigned long, 32ul>, Std::integral_constant<unsigned long, 1ul>>, Std::tuple<Std::integral_constant<unsigned long, 32ul>, Std::integral_constant<unsigned long, 32ul>>>, (Hardware)3>&, TileTensor<signed char, TileOp::Layout<Std::tuple<unsigned long, unsigned long>, Std::tuple<Std::integral_constant<unsigned long, 32ul>, Std::integral_constant<unsigned long, 1ul>>, Std::tuple<Std::integral_constant<unsigned long, 32ul>, Std::integral_constant<unsigned long, 32ul>>>, (Hardware)4>&)
        /root/miniconda/envs/mq/lib/python3.10/site-packages/pypto/lib/include/tileop/cube/cube_pto.h:307:5

        /data/m00794585/pypto/wk/pypto/kernel_aicore/TENSOR_s0_Unroll1_PATH0_hiddenfunc0_8_7936091181990093848_0_aic.cpp:45:1
    ```
8. The preceding information indicates that the core performs the TMatmul operation at TENSOR_s0_Unroll1_PATH0_hiddenfunc0_8_7936091181990093848_0_aic.cpp:45.
9. After locating the hang position, you can use `aicore print` to print the parameters involved in the problematic CCE instruction to locate the issue.

## AiCore Print

### Function Description

AiCore Print is used to print tensor data and debug information in AI Core kernels, supporting GM, UB, and L1 memory hierarchies and multiple data types.

### APIs

| API Name | Function | Applicable Scenario | Ascend 950PR/Ascend 950DT |
|---------|------|---------|:---:|
| **AiCoreLogF** | Formatted log printing | Print addresses, scalars, and prompt information | Supported |
| **AiCorePrintShape** | Print shape information | View tensor shape dimensions | Supported |
| **AiCorePrintGmTensor** | Print GM tensor | View global memory data | Supported |
| **AiCorePrintUbTensor** | Print UB tensor | View unified buffer data (AIV kernel only) | Supported |
| **AiCorePrintL1Tensor** | Print L1 tensor | View circular buffer data (AIC kernel only) | Not supported |
| **AiCorePrintL0CTensor** | Print L0C tensor | View accumulator buffer data (AIC kernel only) | Supported |

### Supported Data Types

AiCore Print supports the following data types:

**Floating-point types**:

- Ascend 950PR/Ascend 950DT: supported
- **fp32**: `float`
- **fp16**: `half`
- **bf16**: `bfloat16_t`

**Integer types**:

- Ascend 950PR/Ascend 950DT: supported
- **int8**: `int8_t`
- **uint8**: `uint8_t`
- **int16**: `int16_t`
- **uint16**: `uint16_t`
- **int32**: `int32_t`
- **uint32**: `uint32_t`
- **int64**: `int64_t`
- **uint64**: `uint64_t`

**FP8 types** (with platform restrictions):

- Ascend 950PR/Ascend 950DT: supported
- **fp8_e4m3**: `float8_e4m3_t`
- **fp8_e5m2**: `float8_e5m2_t`
- **fp8_e8m0**: `float8_e8m0_t`
- **hifloat8**: `hifloat8_t`

**Platform restriction**: FP8 and HiFloat8 types are supported only on Ascend 950PR/Ascend 950DT (`SUPPORT_FP8_HF8_PRINT=1`, corresponding to `__NPU_ARCH__ == 3510`).

Other platforms do not support FP8/HiFloat8 print functionality.

### Procedure

### 1. Enabling Trace Logging

Modify the configuration file:

`framework/src/interface/configs/tile_fwk_config.json`

```json
"fixed_output_path": true,
"force_overwrite": false,
```

Modify the header file:

`framework/src/interface/machine/device/tilefwk/aicore_print.h`

```cpp
#define ENABLE_AICORE_PRINT 1
```

### 2. Recompiling and Reinstalling

```bash
rm -rf build_out/ && python build_ci.py && pip install build_out/pypto*whl --force-reinstall --no-deps
```

### 3. Adding Print Code to the Kernel CCE File

**Important process notes:**

When to delete the kernel_aic* directory:

- For the first run or when switching cases, delete the **kernel_aic*** directory.
- For repeated runs of the same use case, keep the **kernel_aic*** directory (retain modifications).

**Step 3.1: Generate kernel CCE files on first run**.

For the first run or when switching cases:

```bash
rm -rf kernel_aic* output/ wk/
export ASCEND_PROCESS_LOG_PATH=./wk && export ASCEND_GLOBAL_LOG_LEVEL=1 && python xxx.py
```

For repeated runs of the same use case (print code already added):

```bash
rm -rf output/ wk/
export ASCEND_PROCESS_LOG_PATH=./wk && export ASCEND_GLOBAL_LOG_LEVEL=1 && python xxx.py
```

**Step 3.2: Add print code to the generated CCE files**.

View the generated kernel files:

```bash
ls kernel_aicore/*.cpp
```

Modification steps:
(1) Add `#include "tilefwk/aicore_print.h"` at the beginning of the file.
(2) Add a print call at an appropriate location (a synchronization point after data loading or computation).

Print API call format:

```cpp
AiCoreLogF(param->ctx, "format string", args...);
AiCorePrintShape(param->ctx, Shape2Dim(dim0, dim1), "name");
AiCorePrintGmTensor(param->ctx, (__gm__ T*)addr, end, begin, "name");
AiCorePrintUbTensor(param->ctx, (__ubuf__ T*)addr, end, begin, "name");
AiCorePrintL1Tensor(param->ctx, (__cbuf__ T*)addr, end, begin, l1_staging, "name");
AiCorePrintL0CTensor(param->ctx, (__cc__ T*)addr, end, begin, l0cShape0, l0cShape1, l0c_staging, "name");
```

**Step 3.3: Configure L1/L0C staging buffer (required only for AiCorePrintL1Tensor/AiCorePrintL0CTensor)**.

```cpp
// L1 staging buffer (allocated from workspace)
__gm__ T* l1_staging = (__gm__ T*)(param->funcData->workspaceAddr);

// L0C staging buffer (allocated from workspace, 32-byte alignment required)
__gm__ T* l0c_staging = (__gm__ T*)(param->funcData->workspaceAddr);
```

**Note**: For the first run or when switching cases, delete **kernel_aic***. For repeated runs of the same use case, keep the modifications.

### 4. Running the Test and Viewing the Print Results

**Important**: The following commands must be executed **completely in one go** (connected with `&&`). Do not split them into multiple commands:

```bash
export ASCEND_PROCESS_LOG_PATH=./wk && export ASCEND_GLOBAL_LOG_LEVEL=1 && rm -rf output/ wk/ && python xxx.py && grep -rn "DumpAicoreLog" ./wk
```

**Command description**:

1. `export ASCEND_PROCESS_LOG_PATH=./wk`: Sets the log output directory to `./wk`.
2. `export ASCEND_GLOBAL_LOG_LEVEL=1`: Sets the log level to INFO (level 1) and enables log output.
3. `rm -rf output/ wk/`: Clears old logs and build artifacts to avoid interference.
4. `python xxx.py`: Runs the test case to trigger kernel compilation and execution.
5. `grep -rn "DumpAicoreLog" ./wk`: Searches for and prints all AiCore Print outputs (including tensor data and debug information).

### Print Examples for Different Data Types

The following examples demonstrate the print usage for each data type. Insert the print code at an appropriate location (such as the synchronization point after TLoad/TAdd).

### Floating-Point Types

```cpp
AiCorePrintGmTensor(param->ctx, (__gm__ float*)gmTensor_fp32.GetAddr(), 8, 0, "fp32_gm");

AiCorePrintUbTensor(param->ctx, (__ubuf__ half*)ubTensor_fp16.GetAddr(), 16, 0, "fp16_ub");

__gm__ bfloat16_t* l1_staging_bf16 = (__gm__ bfloat16_t*)(param->funcData->workspaceAddr);
AiCorePrintL1Tensor(param->ctx, (__cbuf__ bfloat16_t*)l1Tensor_bf16.GetAddr(), 16, 0, l1_staging_bf16, "bf16_l1");
```

### Integer Types

```cpp
AiCorePrintGmTensor(param->ctx, (__gm__ int8_t*)gmTensor_int8.GetAddr(), 16, 0, "int8_gm");

AiCorePrintUbTensor(param->ctx, (__ubuf__ uint8_t*)ubTensor_uint8.GetAddr(), 16, 0, "uint8_ub");

AiCorePrintUbTensor(param->ctx, (__ubuf__ int16_t*)ubTensor_int16.GetAddr(), 8, 0, "int16_ub");

AiCorePrintGmTensor(param->ctx, (__gm__ uint16_t*)gmTensor_uint16.GetAddr(), 8, 0, "uint16_gm");

AiCorePrintUbTensor(param->ctx, (__ubuf__ int32_t*)ubTensor_int32.GetAddr(), 16, 0, "int32_ub");

AiCorePrintGmTensor(param->ctx, (__gm__ uint32_t*)gmTensor_uint32.GetAddr(), 8, 0, "uint32_gm");

AiCorePrintGmTensor(param->ctx, (__gm__ int64_t*)gmTensor_int64.GetAddr(), 8, 0, "int64_gm");

AiCorePrintUbTensor(param->ctx, (__ubuf__ uint64_t*)ubTensor_uint64.GetAddr(), 8, 0, "uint64_ub");
```

### FP8 Types (with Platform Restrictions)

```cpp
AiCorePrintGmTensor(param->ctx, (__gm__ float8_e4m3_t*)gmTensor_fp8e4m3.GetAddr(), 8, 0, "fp8e4m3_gm");

AiCorePrintGmTensor(param->ctx, (__gm__ float8_e5m2_t*)gmTensor_fp8e5m2.GetAddr(), 8, 0, "fp8e5m2_gm");

AiCorePrintGmTensor(param->ctx, (__gm__ float8_e8m0_t*)gmTensor_fp8e8m0.GetAddr(), 8, 0, "fp8e8m0_gm");

AiCorePrintGmTensor(param->ctx, (__gm__ hifloat8_t*)gmTensor_hf8.GetAddr(), 8, 0, "hifloat8_gm");
```

### Other APIs

AiCorePrintShape:

```cpp
AiCorePrintShape(param->ctx, Shape2Dim(sym_161_dim_0, sym_161_dim_1), "sym_161");
AiCorePrintShape(param->ctx, Shape3Dim(dim0, dim1, dim2));
AiCorePrintShape(param->ctx, Shape4Dim(dim0, dim1, dim2, dim3), "conv_out");
```

L1 Tensor print example:

```cpp
__gm__ half* l1_staging = (__gm__ half*)(param->funcData->workspaceAddr);
AiCorePrintL1Tensor(param->ctx, (__cbuf__ half*)l1Tensor.GetAddr(), 16, 0, l1_staging, "fp16_l1");
```

L0C Tensor print example (L0C data is printed after being moved to the GM staging buffer via DMA):

```cpp
__gm__ int32_t* l0c_staging = (__gm__ int32_t*)(param->funcData->workspaceAddr);
AiCorePrintL0CTensor(param->ctx, (__cc__ int32_t*)l0cTensor.GetAddr(), 1024, 0, 32, 32, l0c_staging, "int32_l0c");
```

AiCoreLogF:

```cpp
AiCoreLogF(param->ctx, "GM address=%p", ((__gm__ float*)gmTensor.GetAddr()));
AiCoreLogF(param->ctx, "Shape=[%ld,%ld]", dim0, dim1);
AiCoreLogF(param->ctx, "INT8 input loaded");
```

### Notes

1. **L1/L0C staging buffer alignment**: The l1_staging and l0c_staging addresses must be 32-byte aligned. The workspaceAddr meets this requirement by default.

2. **Print quantity control**: PRINT_BUFFER_SIZE is currently 128 KB (defined in `framework/src/interface/machine/device/tilefwk/aicpu_common.h`). If an overflow warning is triggered, increase this value and recompile.

3. **FP8/HiFloat8 supported platforms**: Only Ascend 950PR/Ascend 950DT (`__NPU_ARCH__ == 3510`) are supported (see the `SUPPORT_FP8_HF8_PRINT` macro definition).

4. **AiCorePrintL1Tensor supported platforms**: Ascend 950PR/Ascend 950DT are not supported. Atlas A2 training products/Atlas A2 inference products and Atlas A3 training products/Atlas A3 inference products are supported (see the `SUPPORT_L1_COPY` macro definition).

5. **AiCorePrintUbTensor cannot be used in the AIC (Cube core)**. The scalar processor (SP) of the AIC has no physical path to the UB address space, and thus cannot read scalar data from the UB. This restriction is enforced at compile time via `static_assert`. Calling `AiCorePrintUbTensor` in an AIC kernel will trigger a compilation error:

   ```text
   error: static assertion failed: [AIC UB Print Error] AiCorePrintUbTensor is not supported on AIC (Cube) kernel.
   ```

   To check UB data, perform the check in an AIV (Vector core) kernel, or use `AiCorePrintGmTensor` in the AIC to print data that has been moved to GM.

6. **AiCoreLogF triggers a runtime error when printing UB data values in the AIC**: When `AiCoreLogF` uses formats such as `%f` and `%d` to print UB data values in an AIC kernel (for example, `((__ubuf__ float*)addr)[521]`), the compiler generates a scalar load instruction from the UB address space. The AIC SP does not support this operation, triggering MPU error 271:

   ```text
   error from aicore error exception, core id is 0, error code = 271
   errorStr: The MPU address access is invalid
   ```

   Using `%p` to print address values (without reading UB data) is safe. **Correct approach**: Do not directly read UB data values in the AIC kernel. Move the UB print logic to the AIV kernel.

7. **Do not use DMA to move UB data to GM for printing**. The AIC (Cube core) does not have an MTE3 DMA engine (intrinsics such as `copy_ubuf_to_gm` and `copy_ubuf_to_gm_align_v2` do not support the cube target). The `OpCoreType` of `TStoreVec` (`OP_UB_COPY_OUT`) is `AIV`, which is exclusive to the Vector core. Calling these APIs in an AIC kernel causes a compilation error:

   ```text
   error: function type '...' of 'copy_ubuf_to_gm' does not support the given target feature
   ```

### FAQs

### 1. No Print Output Displayed

Ensure that ENABLE_AICORE_PRINT=1 is set, recompilation and installation are complete, the log dump path is specified, the log level is set to info (1), and the grep search results are as expected.

### 2. L1/L0C Print Alignment Warning

Ensure that l1_staging/l0c_staging addresses are 32-byte aligned. workspaceAddr itself is already aligned.

### 3. Overflow Warning

Reduce the print quantity or increase PRINT_BUFFER_SIZE and recompile.

### 4. FP8/HiFloat8 Printing Not Supported

This is not supported on the current platform (check the `SUPPORT_FP8_HF8_PRINT` macro; it is enabled (set to **1**) only on Ascend 950PR/Ascend 950DT, that is, when `__NPU_ARCH__ == 3510`).

### 5. AiCorePrintL1Tensor Cannot Find API Definition

The current platform does not support this (check the `SUPPORT_L1_COPY` macro).

### 6. ld.lld: error: undefined symbol

The `ld.lld: error: undefined symbol` link error occurs during compilation, causing the compilation to fail.

**Cause**: When the `parallel_compile` value is greater than 1, CodeGen compiles multiple subgraphs in parallel. In this mode, symbol dependencies between some compilation units are not correctly handled, resulting in a link failure.

**Solution**: Modify `framework/src/interface/configs/tile_fwk_config.json` by setting `parallel_compile` to `1` (one compilation thread, that is, serial compilation). Note: This configuration item indicates the **number of parallel compilation threads**, not a Boolean switch (`1` indicates a single thread, while `128` and other values indicate multi-threaded parallelism). After the modification, rerun to resolve the issue.

```json
"parallel_compile": 1
```

### Compilation Error Occurs When AiCorePrintUbTensor Is Called in an AIC Kernel

When `AiCorePrintUbTensor` is used in an AIC (Cube core) kernel, the compiler triggers `static_assert`:

```text
error: static assertion failed due to requirement '!std::is_same_v<float, float>':
  [AIC UB Print Error] AiCorePrintUbTensor is not supported on AIC (Cube) kernel.
  AIC Scalar Processor cannot scalar-load from UB address space.
  Please use AiCorePrintUbTensor in AIV (Vector) kernel instead,
  or use AiCorePrintGmTensor to print data that has been moved to GM.
```

**Cause**: The scalar processor (SP) of the AIC (Cube core) has no physical path to the UB address space and thus cannot read scalar data from UB.

**Solution**: Move the `AiCorePrintUbTensor` call to an AIV (Vector core) kernel, or use `AiCorePrintGmTensor` to print data that has been moved to GM.

### 8. Using AiCoreLogF to Print UB Data Values in AIC Kernel Triggers Error 271

The following code is used in the CCE file of the AIC kernel:

```cpp
AiCoreLogF(param->ctx, "ubTensor val=%f", ((__ubuf__ float*)ubTensor.GetAddr())[521]);
```

An aicore error is triggered at runtime:

```text
error from aicore error exception, core id is 0, error code = 271
errorStr: The MPU address access is invalid
```

**Cause**: `((__ubuf__ float*)addr)[521]` generates a scalar load instruction from the UB address space, which is not supported by the AIC SP. **Note**: Scalar reads from the UB address space in the AIC kernel cannot be trapped by `static_assert` at compile time (because the `__ubuf__` attribute is no longer retained after the variadic template of `AiCoreLogF` evaluates the parameter expression), nor can they be caught at runtime (such MPU errors are hardware traps with no software recovery mechanism).

**Solution**:

- Move the UB data printing logic to the AIV kernel.
- In the AIC kernel, use only `%p` to print the UB address value (without reading data). This is safe.
- Check the CCE code of the AIC kernel and delete all expressions that perform `[]` subscript access on the UB address space.

### 9. Compilation Error When TStoreVec/copy_ubuf_to_gm Is Used in AIC Kernel to Move Data

Compilation fails when APIs such as `TStoreVec`, `copy_ubuf_to_gm`, and `copy_ubuf_to_gm_align_v2` are called in an AIC kernel:

```text
error: function type 'void (__gm__ void *, __ubuf__ void *, ...)' of 'copy_ubuf_to_gm' does not support the given target feature
```

**Cause**: The Cube core does not have an MTE3 DMA output engine, and all intrinsics that transfer data from the UB source address do not support the cube target. The `OpCoreType` of `TStoreVec` (`OP_UB_COPY_OUT`) is `AIV`, which is a Vector core-specific operation.

**Solution**: This operation can only be used in AIV (Vector core) kernels. Do not perform this operation in AIC kernels. To print UB data, perform the operation within the AIV core.
