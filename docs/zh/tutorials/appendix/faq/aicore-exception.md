# aicore exception

## 问题现象描述

AICore kernel执行期间发生异常（硬件trap、执行超时、core挂死）。
```bash
[Error]: aicore exception, device_id: 6, stream_id: 47, task_id: 2, retcode: 507015, kernelName: PyPTO_matmul_add_0_mix_aic
        Rectify the fault based on the error information in the ascend log.
PyPTO error: PyPTO Inner Error. Please rectify the fault based on the error information in the ascend log. (function PyPTOExceptionInfoCallBack)
```

## 可能原因

- Kernel代码存在内存越界访问。
- 数据依赖边丢失（producer未写完consumer已读）。
- Tiling/Shape参数与kernel不匹配。
- MACHINE调度框架自身问题。

## 处理方式
1. export ASCEND_WORK_PATH=./wk，详细介绍请参考《[环境变量参考](https://www.hiascend.com/document/redirect/CannCommunityEnvRef)》。
2. 固定cce编译模式
    ```python
    #调用示例
    @pypto.frontend.jit（debug_options={"compile_debug_mode": 2}）
    def pypto_kernel（）:
    ```
3. 开启singlecommit，每条指令单步跑
    ```bash
    /usr/local/Ascend/driver/tools/msnpureport config --set --singlecommit 1 -d device-id
    ```
4. 重新执行用例
5. plog日志里搜kernel_symbol_locator.cpp
    ```bash
        #示例
        grep -rn "kernel_symbol_locator.cpp" wk/log/debug/plog/plog-2341095_20260707170139095.log

        #寄存器信息
        62:[ERROR] IDEDD(2341095,python):2026-07-07-17:01:41.620.723 [kernel_symbol_locator.cpp:583][tid:2341490] [Dump][Exception] Error register information. coreId=6, coreType=0, AIC_ERR_0=0x0 AIC_ERR_1=0x0 AIC_ERR_2=0x0 AIC_ERR_3=0x40000000 AIC_ERR_4=0x0 AIC_ERR_5=0x0 BIU_ERR_0=0x0 BIU_ERR_1=0x0 CCU_ERR_0=0x0 CCU_ERR_1=0x63851b81 CUBE_ERR_0=0x4000036 CUBE_ERR_1=0x0 IFU_ERR_0=0xde06800 IFU_ERR_1=0x212c3 MTE_ERR_0=0x3bcdf8f6 MTE_ERR_1=0x13 VEC_ERR_0=0x0 VEC_ERR_1=0x0 FIXP_ERR_0=0xbcdf8f6 FIXP_ERR_1=0x13 AIC_COND_0=0x0 AIC_COND_1=0x0
        #PC信息
        64:[ERROR] IDEDD(2341095,python):2026-07-07-17:01:41.620.746 [kernel_symbol_locator.cpp:602][tid:2341490] [Dump][Exception] Error PC information. coreId=6, coreType=0, originalStartPC=0x124a00001130, fixedStartPC=0x124a00001000, originalCurrentPC=0x124a000010dc, fixedCurrentPC=0x124a000010d8, fixedPCOffset=0xd8.
        #符号信息
        65:[ERROR] IDEDD(2341095,python):2026-07-07-17:01:41.620.750 [kernel_symbol_locator.cpp:608][tid:2341490] [Dump][Exception] Error symbol information. coreId=6, coreType=0, symbol=TENSOR_s0_Unroll1_PATH0_hiddenfunc0_8_0_4294967296+0xd8.

        #如果symbol=TENSOR_s0_Unroll1_PATH0_hiddenfunc0_8_0_4294967296，即表示挂在该cce文件，如果symbol=PyPTO_matmul_add_0_mix_aic，即表示挂在框架aicore处理源码处。
    ```


7. llvm-symbolizer --obj=${aicore_kernel_bin_file_path} fixedPCOffset
    - llvm-symbolizer通过apt install llvm或yum install llvm安装
    - 触发aicore exception后，aicore_kernel_bin_fil会在${ASCEND_WORK_PATH}/extra-info/data-dump/device-id/目录下自动落盘。
    - fixedPCOffset，即65行中的0xd8，即core的cce指令基于aicore_kernel_bin_file基地址的偏移量。
    ```bash
        #示例
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
8. 基于上述信息，即表示core在TENSOR_s0_Unroll1_PATH0_hiddenfunc0_8_7936091181990093848_0_aic.cpp:45处TMatmul操作。
9. 找到挂的位置后，可以通过`aicore print`打印问题cce指令涉及的参数，以便定位问题。

## aicore print

### 功能说明

AiCore Print用于在AICore kernel中打印tensor数据和调试信息，支持GM、UB、L1内存层次和多种数据类型。

### 对外接口

| 接口名称 | 功能 | 适用场景 | Ascend 950PR |
|---------|------|---------|:---:|
| **AiCoreLogF** | 格式化日志打印 | 打印地址、标量、提示信息 | 支持 |
| **AiCorePrintShape** | 打印Shape信息 | 查看tensor shape维度 | 支持 |
| **AiCorePrintGmTensor** | 打印GM Tensor | 查看Global Memory数据 | 支持 |
| **AiCorePrintUbTensor** | 打印UB Tensor | 查看Unified Buffer数据（仅AIV kernel） | 支持 |
| **AiCorePrintL1Tensor** | 打印L1 Tensor | 查看Circular Buffer数据（仅AIC kernel） | 不支持 |
| **AiCorePrintL0CTensor** | 打印L0C Tensor | 查看Accumulator Buffer数据（仅AIC kernel） | 支持 |

### 支持的数据类型

AiCore Print支持以下数据类型：

**浮点类型**：

- Ascend 950PR：支持
- **fp32**：`float`
- **fp16**：`half`
- **bf16**：`bfloat16_t`

**整数类型**：

- Ascend 950PR：支持
- **int8**：`int8_t`
- **uint8**：`uint8_t`
- **int16**：`int16_t`
- **uint16**：`uint16_t`
- **int32**：`int32_t`
- **uint32**：`uint32_t`
- **int64**：`int64_t`
- **uint64**：`uint64_t`

**FP8类型**（平台限制）：

- Ascend 950PR：支持
- **fp8_e4m3**：`float8_e4m3_t`
- **fp8_e5m2**：`float8_e5m2_t`
- **fp8_e8m0**：`float8_e8m0_t`
- **hifloat8**：`hifloat8_t`

**平台限制说明**：FP8和HiFloat8类型仅在Ascend 950PR上支持（`SUPPORT_FP8_HF8_PRINT=1`，对应`__NPU_ARCH__ == 3510`）。

其他平台不支持FP8/HiFloat8打印功能。

### 使用步骤

### 1. 启用追踪日志

修改配置文件：

`framework/src/interface/configs/tile_fwk_config.json`

```json
"fixed_output_path": true,
"force_overwrite": false,
```

修改头文件：

`framework/src/interface/machine/device/tilefwk/aicore_print.h`

```cpp
#define ENABLE_AICORE_PRINT 1
```

### 2. 重新编译安装

```bash
rm -rf build_out/ && python build_ci.py && pip install build_out/pypto*whl --force-reinstall --no-deps
```

### 3. 在kernel CCE文件中添加打印代码

**重要流程说明**：

**何时删除kernel_aic*目录**：

- 首次运行或切换用例：删除kernel_aic*目录
- 同一用例重复运行：保留kernel_aic*目录（保留修改）

**步骤3.1：首次运行生成kernel CCE文件**

首次运行或切换用例：

```bash
rm -rf kernel_aic* output/ wk/
export ASCEND_PROCESS_LOG_PATH=./wk && export ASCEND_GLOBAL_LOG_LEVEL=1 && python xxx.py
```

同一用例重复运行（已添加打印代码）：

```bash
rm -rf output/ wk/
export ASCEND_PROCESS_LOG_PATH=./wk && export ASCEND_GLOBAL_LOG_LEVEL=1 && python xxx.py
```

**步骤3.2：在生成的CCE文件中添加打印代码**

查看生成的kernel文件：

```bash
ls kernel_aicore/*.cpp
```

修改步骤：
（1）在文件开头添加：`#include "tilefwk/aicore_print.h"`
（2）在合适位置（数据加载或计算后的同步点）添加打印调用

打印接口调用格式：

```cpp
AiCoreLogF(param->ctx, "format string", args...);
AiCorePrintShape(param->ctx, Shape2Dim(dim0, dim1), "name");
AiCorePrintGmTensor(param->ctx, (__gm__ T*)addr, end, begin, "name");
AiCorePrintUbTensor(param->ctx, (__ubuf__ T*)addr, end, begin, "name");
AiCorePrintL1Tensor(param->ctx, (__cbuf__ T*)addr, end, begin, l1_staging, "name");
AiCorePrintL0CTensor(param->ctx, (__cc__ T*)addr, end, begin, l0cShape0, l0cShape1, l0c_staging, "name");
```

**步骤3.3：配置L1/L0C staging buffer（仅AiCorePrintL1Tensor / AiCorePrintL0CTensor需要）**

```cpp
// L1 staging buffer（从workspace分配）
__gm__ T* l1_staging = (__gm__ T*)(param->funcData->workspaceAddr);

// L0C staging buffer（从workspace分配，需32字节对齐）
__gm__ T* l0c_staging = (__gm__ T*)(param->funcData->workspaceAddr);
```

**关键注意事项**：首次运行或切换用例删除kernel_aic*，同一用例重复运行保留修改。

### 4. 运行测试并查看打印结果

**重要**：以下命令必须**一次性完整执行**（使用`&&`连接），不要拆分为多个命令：

```bash
export ASCEND_PROCESS_LOG_PATH=./wk && export ASCEND_GLOBAL_LOG_LEVEL=1 && rm -rf output/ wk/ && python xxx.py && grep -rn "DumpAicoreLog" ./wk
```

**命令说明**：

1. `export ASCEND_PROCESS_LOG_PATH=./wk`：设置日志输出目录为`./wk`
2. `export ASCEND_GLOBAL_LOG_LEVEL=1`：设置日志级别为INFO（级别1），开启日志输出
3. `rm -rf output/ wk/`：清理旧日志和编译产物，避免干扰
4. `python xxx.py`：运行测试用例，触发kernel编译和执行
5. `grep -rn "DumpAicoreLog" ./wk`：搜索并打印所有AiCore Print输出（包含tensor数据和调试信息）

### 不同数据类型打印示例

以下示例展示每种数据类型的打印用法。打印代码需在合适位置插入（如TLoad/TAdd后的同步点）。

### 浮点类型

```cpp
AiCorePrintGmTensor(param->ctx, (__gm__ float*)gmTensor_fp32.GetAddr(), 8, 0, "fp32_gm");

AiCorePrintUbTensor(param->ctx, (__ubuf__ half*)ubTensor_fp16.GetAddr(), 16, 0, "fp16_ub");

__gm__ bfloat16_t* l1_staging_bf16 = (__gm__ bfloat16_t*)(param->funcData->workspaceAddr);
AiCorePrintL1Tensor(param->ctx, (__cbuf__ bfloat16_t*)l1Tensor_bf16.GetAddr(), 16, 0, l1_staging_bf16, "bf16_l1");
```

### 整数类型

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

### FP8类型（平台限制）

```cpp
AiCorePrintGmTensor(param->ctx, (__gm__ float8_e4m3_t*)gmTensor_fp8e4m3.GetAddr(), 8, 0, "fp8e4m3_gm");

AiCorePrintGmTensor(param->ctx, (__gm__ float8_e5m2_t*)gmTensor_fp8e5m2.GetAddr(), 8, 0, "fp8e5m2_gm");

AiCorePrintGmTensor(param->ctx, (__gm__ float8_e8m0_t*)gmTensor_fp8e8m0.GetAddr(), 8, 0, "fp8e8m0_gm");

AiCorePrintGmTensor(param->ctx, (__gm__ hifloat8_t*)gmTensor_hf8.GetAddr(), 8, 0, "hifloat8_gm");
```

### 其他接口

AiCorePrintShape：

```cpp
AiCorePrintShape(param->ctx, Shape2Dim(sym_161_dim_0, sym_161_dim_1), "sym_161");
AiCorePrintShape(param->ctx, Shape3Dim(dim0, dim1, dim2));
AiCorePrintShape(param->ctx, Shape4Dim(dim0, dim1, dim2, dim3), "conv_out");
```

L1 Tensor打印示例：

```cpp
__gm__ half* l1_staging = (__gm__ half*)(param->funcData->workspaceAddr);
AiCorePrintL1Tensor(param->ctx, (__cbuf__ half*)l1Tensor.GetAddr(), 16, 0, l1_staging, "fp16_l1");
```

L0C Tensor打印示例（L0C数据通过DMA搬运到GM staging buffer后打印）：

```cpp
__gm__ int32_t* l0c_staging = (__gm__ int32_t*)(param->funcData->workspaceAddr);
AiCorePrintL0CTensor(param->ctx, (__cc__ int32_t*)l0cTensor.GetAddr(), 1024, 0, 32, 32, l0c_staging, "int32_l0c");
```

AiCoreLogF：

```cpp
AiCoreLogF(param->ctx, "GM address=%p", ((__gm__ float*)gmTensor.GetAddr()));
AiCoreLogF(param->ctx, "Shape=[%ld,%ld]", dim0, dim1);
AiCoreLogF(param->ctx, "INT8 input loaded");
```

### 注意事项

1. **L1/L0C staging buffer对齐**：l1_staging和l0c_staging地址必须32字节对齐，workspaceAddr默认满足要求。

2. **打印数量控制**：PRINT_BUFFER_SIZE当前为128KB（定义于`framework/src/interface/machine/device/tilefwk/aicpu_common.h`），若触发overflow warning，需增大该值后重新编译。

3. **FP8/HiFloat8支持平台**：仅Ascend 950PR（`__NPU_ARCH__ == 3510`）支持（见`SUPPORT_FP8_HF8_PRINT`宏定义）。

4. **AiCorePrintL1Tensor支持平台**：Ascend 950PR不支持；Atlas A2训练系列产品/Atlas A2推理系列产品、Atlas A3训练系列产品/Atlas A3推理系列产品支持（见`SUPPORT_L1_COPY`宏定义）。

5. **AIC (Cube核)中不能使用AiCorePrintUbTensor**：AIC (Cube核)的标量处理器(SP)没有到UB地址空间的物理通路，无法从UB标量读取数据。编译期已通过`static_assert`拦截，在AIC kernel中调用`AiCorePrintUbTensor`会触发编译报错：

   ```text
   error: static assertion failed: [AIC UB Print Error] AiCorePrintUbTensor is not supported on AIC (Cube) kernel.
   ```

   UB数据检查请在AIV (Vector核) kernel中完成，或在AIC中使用`AiCorePrintGmTensor`打印已搬到GM的数据。

6. **AiCoreLogF在AIC中打印UB数据值会触发运行时错误**：`AiCoreLogF`在AIC kernel中使用`%f`、`%d`等格式打印UB数据值时（如`((__ubuf__ float*)addr)[521]`），编译器会生成一条从UB地址空间的标量load指令，AIC SP不支持此操作，触发MPU error 271：

   ```text
   error from aicore error exception, core id is 0, error code = 271
   errorStr: The MPU address access is invalid
   ```

   `%p`打印地址值（不读取UB数据）是安全的。**正确做法**：AIC kernel中不直接读取UB数据值，将UB打印逻辑移到AIV kernel中。

7. **不可通过DMA将UB数据搬到GM再打印**：AIC (Cube核)上没有MTE3 DMA引擎（`copy_ubuf_to_gm`、`copy_ubuf_to_gm_align_v2`等intrinsic不支持cube target），`TStoreVec`（`OP_UB_COPY_OUT`）的`OpCoreType`为`AIV`，属于Vector核专用。在AIC kernel中调用这些接口会编译报错：

   ```text
   error: function type '...' of 'copy_ubuf_to_gm' does not support the given target feature
   ```

### 常见问题

### 1. 未看到打印输出

检查：ENABLE_AICORE_PRINT=1、已重新编译安装，已指定日志落盘路径，已设置日志级别为info级别（1），grep搜索文件正确。

### 2. L1/L0C Print对齐WARNING

确保l1_staging / l0c_staging地址32B对齐，workspaceAddr本身已对齐。

### 3. Overflow Warning

减少打印数量或增大PRINT_BUFFER_SIZE后重新编译。

### 4. FP8/HiFloat8无法打印

当前平台不支持（检查`SUPPORT_FP8_HF8_PRINT`宏；仅Ascend 950PR / `__NPU_ARCH__ == 3510`时为1）。

### 5. AiCorePrintL1Tensor找不到接口定义

当前平台不支持（检查SUPPORT_L1_COPY宏）。

### 6. ld.lld: error: undefined symbol

编译时出现`ld.lld: error: undefined symbol`链接错误，导致编译失败。

**原因**：`parallel_compile`配置值大于1时，CodeGen会并行编译多个子图；在此模式下，部分编译单元之间的符号依赖未正确处理，导致链接失败。

**解决方案**：修改`framework/src/interface/configs/tile_fwk_config.json`，将`parallel_compile`设为`1`（编译线程数为1，即串行编译）。注意：该配置项表示**并行编译线程数**，而非布尔开关（`1`表示单线程，`128`等为多线程并行）。修改后重新运行即可解决。

```json
"parallel_compile": 1
```

### 7. AIC kernel中调用AiCorePrintUbTensor编译报错

AIC (Cube核) kernel中使用`AiCorePrintUbTensor`时，编译器会触发`static_assert`：

```text
error: static assertion failed due to requirement '!std::is_same_v<float, float>':
  [AIC UB Print Error] AiCorePrintUbTensor is not supported on AIC (Cube) kernel.
  AIC Scalar Processor cannot scalar-load from UB address space.
  Please use AiCorePrintUbTensor in AIV (Vector) kernel instead,
  or use AiCorePrintGmTensor to print data that has been moved to GM.
```

**原因**：AIC (Cube核)的标量处理器(SP)没有到UB地址空间的物理通路，无法从UB标量读取数据。

**解决方案**：将`AiCorePrintUbTensor`调用移到AIV (Vector核) kernel中，或使用`AiCorePrintGmTensor`打印已搬到GM的数据。

### 8. AIC kernel中使用AiCoreLogF打印UB数据值触发error 271

在AIC kernel的CCE文件中使用如下代码：

```cpp
AiCoreLogF(param->ctx, "ubTensor val=%f", ((__ubuf__ float*)ubTensor.GetAddr())[521]);
```

运行时触发aicore error：

```text
error from aicore error exception, core id is 0, error code = 271
errorStr: The MPU address access is invalid
```

**原因**：`((__ubuf__ float*)addr)[521]`会生成一条从UB地址空间的标量load指令，AIC SP不支持此操作。**注意**：AIC kernel中从UB地址空间标量读取无法在编译期被`static_assert`拦截（因为`AiCoreLogF`的变参模板在参数表达式求值后，`__ubuf__`属性已丢失），也无法在运行时拦截（MPU error是硬件trap，无软件恢复机制）。

**解决方案**：

- 将UB数据打印逻辑移到AIV kernel中
- AIC kernel中仅使用`%p`打印UB地址值（不读取数据），这是安全的
- 检查AIC kernel的CCE代码，删除所有对UB地址空间做`[]`下标访问的表达式

### 9. AIC kernel中尝试TStoreVec / copy_ubuf_to_gm搬运UB数据编译报错

在AIC kernel中调用`TStoreVec`、`copy_ubuf_to_gm`、`copy_ubuf_to_gm_align_v2`等接口编译报错：

```text
error: function type 'void (__gm__ void *, __ubuf__ void *, ...)' of 'copy_ubuf_to_gm' does not support the given target feature
```

**原因**：Cube核上没有MTE3 DMA输出引擎，所有从UB源地址搬迁数据的intrinsic均不支持cube target。`TStoreVec`（`OP_UB_COPY_OUT`）的`OpCoreType`为`AIV`，是Vector核专用操作。

**解决方案**：此类操作只能在AIV (Vector核) kernel中使用，不要在AIC kernel中调用。需要打印UB数据时，在AIV中完成。