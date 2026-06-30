# CODEGEN组件常见问题处理

- **错误码范围**：F6XXXX
- 本文档说明CODEGEN组件的错误码定义、场景说明与排查建议。

---

## 错误码定义

相关错误码的枚举与码值统一定义在`framework/include/tilefwk/error_code.h`（见FwkErr、OperErr、GenCodeErr、CmpCodeErr等）。

---

## 日志分析建议

### 分析前准备

遇到CodeGen组件校验报错，或生成的Kernel代码不符合预期，可通过如下步骤进行日志收集并分析：

1. **设置日志级别为INFO**
   - 设置日志输出路径
   export ASCEND_PROCESS_LOG_PATH=*{用户指定日志路径}*
   - 设置日志级别为全局INFO级别
   export ASCEND_GLOBAL_LOG_LEVEL=1 // 0: DEBUG, 1: INFO, 2: WARN, 3: ERROR
   或指定CodeGen模块日志级别为INFO，如：
   export ASCEND_MODULE_LOG_LEVEL=CODEGEN=1

2. **设置并行编译数量为1**
   由于CodeGen模块通过并行编译多个子图方式节省编译时长，故为了防止输出日志乱序，定位问题时需要将并行编译改为串行，设置方法如下：
   - 修改tile_fwk_config.json中的parallel_compile为1
   - 重新编译并安装pyPTO包，参考[编译安装](../install/build_and_install.md)。

3. **再次执行用例，获取日志及kernel代码文件**
   日志路径一般为：*{用户指定日志路径}*/debug/plog/pypto-log***.log
   kernel代码文件路径一般为：   pyPTO工程路径或测试框架执行路径下，搜索kernel_aicore文件夹，文件夹内的TENSOR***.cpp即kernel代码文件。

4. **分析日志**
   - 对于FRAMEWORK（F60XXX）、OPERATION_ADAPTER（F61XXX）类错误，一般为上游数据异常导致，需要结合PASS日志分析
   - 其他类型错误需要结合上下文进行分析
<br>

### 分析样例

以**生成kernel代码中某个TileOP调用参数不符合预期**为例，参考步骤如下：

1. 按照[分析前准备](#分析前准备)，收集日志。
2. 在kernel代码中找到不符合预期的TileOp调用，例如：

   ```c++
   TAdd<LastUse3Dim<0, 1, 1>>(ubTensor_0, ubTensor_0, ubTensor_2);
   ```

3. 以上面TileOp调用代码为关键字在日志中进行搜索。
4. 找到日志后往上搜索出现的第一个”Op CodeGenNPU Start”关键字，即该TileOp生成的开始位置，由此往后以此检查日志信息是否符合预期。
5. 若怀疑和PASS传入的数据有关，则可以在"Op CodeGenNPU Start"关键字后搜索"Gen OP IS"关键字，后面包含了该Operation的Dump信息，样例如下：

   ```log
   Gen OP IS: <2 x 2 x 16 x 16 x DT_FP32 / sym_3_dim_0 x sym_3_dim_1 x sym_3_dim_2 x sym_3_dim_3 x DT_FP32> %152@5#(0)MEM_UB::MEM_UB = !10010 TILE_ADD(g:0, s:-1) %3@3#(0)MEM_UB::MEM_UB, %4@4#(0)MEM_UB::MEM_UB #IS_CUBE{0} #last_use{[0, 1, 1]}
   ```

   其中!10010即该OP的唯一标识码，可以此为关键字在PASS的图或日志中搜索获取相关信息，PASS定位指导详见[pass trouble shooting](./pass.md)

---

## 错误码说明

### F62014：GenCodeErr::SYMBOL_NOT_FOUND

**错误描述**
kernel代码中出现了变量使用时未定义错误，该类错误场景日志上下文一般会包含"UNDEFINED_VAR"关键字。

**可能原因**

- 子图中的Operation缺少need_alloc属性。CodeGen需要依赖子图Operation中的need_alloc属性生成变量定义语句，若该属性缺失，则会导致变量定义语句缺失从而报错。

**处理方式**

1. 可结合前文[分析样例](#分析样例)步骤，找到缺失属性的Operation联合PASS继续定位。

<br>

### F63001：GenCodeErr::COMPILE_CODE_FAILED

**错误描述**
kernel代码编译报错，导致编译失败。
如果bisheng编译器报错日志存在丢失或不完整，可能尝试如下方法复现：

1. 在日志中找到类似如下报错信息，包含报错的make命令：

   ```log
   Caught exception: 'ErrCode: F63001! Enum: CmpCodeErr::COMPILE_CODE_FAILED. kernel compilation failed, ret = 512
   compile cmd is:
   make -j112 -f /pypto/build/output/bin/output/output_20260609_193402_089005_981941_6466CEB1/kernel_aicore/Makefile_14_9752067179483835610.compile
   ```

2. 手工执行make命令

   ```bash
   make -j112 -f /pypto/build/output/bin/output/output_20260609_193402_089005_981941_6466CEB1/kernel_aicore/Makefile_14_9752067179483835610.compile
   ```

3. 获取完整的报错信息

**可能原因**

- 堆栈溢出，报错日志样例：

   ```log
   error: stack frame size (*****) exceeds limit (32768) in function
   ```

- PTO指令数据类型不匹配，报错日志样例：

   ```log
   /usr/local/Ascend/cann-9.0.0/include/pto/npu/a5/TStore.hpp:233:41: error: the 2nd parameter maybe need a type 'cc float *'
   copy_matrix_cc_to_gm(dstGlobalAddr, srcTileAddr, xmReg, xtReg);
   ```

   数据类型不匹配可能原因有：
    - 前端调用Operation接口参数传递错误，参考：[执行代码有pto相关报错](https://gitcode.com/cann/pypto/issues/705)
    - 使用了PTO-ISA不支持的数据类型。

- 生成的硬件指令和实际平台不匹配，报错日志样例：

   ```log
   error: function type 'void (__cbuf__ void *, __gm__ void *, unsigned char, unsigned short, unsigned short, unsigned short, unsigned short, unsigned int) noexcept' of 'copy_gm_to_cbuf' does not support the given target feature
      copy_gm_to_cbuf(dst, src, (uint8_t)0, nBurst, lenBurst, gmGap, l1Gap, (pad_t)0);
      ^
   ```

   指令和平台不匹配的原因有：
   - kernel代码编译参数为Vector，但是生成的kernel代码中包含了Cube相关指令，或者反之编译参数为Cube，但是生成的kernel代码中包含了Vector相关指令，导致bisheng编译器报错。
- 变量未定义，报错日志样例：

   ```log
   error: use of undeclared identifier 'sym_209_dim_0'; did you mean 'sym_65_dim_0'?
   UBTileTensorBF16Dim2_1 ubTensor_1((uint64_t)UB_S0_E512_T, (Shape2Dim(sym_209_dim_0, sym_209_dim_1)));
   ```

   此类变量用于运行时动态获取Shape、Offset大小，数据来源于Function::GetDynParamTable接口。

**处理方式**

1. 对于堆栈溢出问题，可参考：[算子编译报堆栈溢出错误](../tutorials/appendix/faq.md#算子编译报堆栈溢出错误)
2. 对于数据类型不支持问题，可重新分析算子计算流程，选用硬件支持的数据类型。
3. 对于硬件指令和实际平台不匹配问题，可联合PASS进一步分析：

   1. 确认CodeGen阶段前一个PASS处理后，一张Block子图中是否同时包含了Vector和Cube的Operation，CodeGen阶段获取的block子图必须是纯Vector或纯Cube子图。
   2. CodeGen使用Cube或Vector的编译参数依据为Function::IsCube()接口，需要PASS确认对cube或vector不同类型Block子图，该接口设置的值是否正确。

4. 对于变量未定义问题，联合PASS进一步分析从Function::GetDynParamTable接口获取的变量中缺失未定义变量的原因。
<br>

---

## 其他问题

### 二进制编译时长统计

CodeGen模块耗时可通过执行算子后在屏幕输出中观察Compiler Monitor提供的统计结果，样例如下：

```log
[Compiler Monitor] Stage: CodeGen(completed) | Stage elapsed: 1.2s | Total elapsed: 1.2s
[Compiler Monitor] Compilation finished 6/6 | Total functions: 6
[Compiler Monitor] Stage timing (aggregated by stage):
  CodeGen  1.2s   (sum over 6 functions)
  Pass     0.0s   (sum over 6 functions)
  Prepare  0.0s
[Compiler Monitor] Monitoring stopped | Total elapsed: 1.2s
```

- 单个kernel文件编译时长确认方法：
  1. 从pyPTO工程路径或测试框架路径下找到kernel_aicore文件夹及需要验证的kernel代码文件Tensor**.cpp，例如：
     {前置路径}/output/output_20260319_145742_163710_1702013_6466B4B5/**kernel_aicore/TENSOR_Step0_Unroll1_PATH0_hiddenfunc0_8_16874966534923480783_0_aiv.cpp**
  2. 打开kernel代码文件，到最底部找到编译该文件的bisheng命令并复制，例如：

     ```bash
     bisheng -c -O3 -g -x cce ... -o output/output_20260319_145742_163710_1702013_6466B4B5/kernel_aicore/ TENSOR_Step0_Unroll1_PATH0_hiddenfunc0_8_16874966534923480783_0_aiv.o output/ output_20260319_145742_163710_1702013_6466B4B5/kernel_aicore/ TENSOR_Step0_Unroll1_PATH0_hiddenfunc0_8_16874966534923480783_0_aiv.cpp
     ```

  3. cd {前置路径}
     确认当前在output文件夹上一层
  4. 执行刚刚复制的bisheng命令，确认可执行成功，若报bisheng命令找不到则参考:
  [prepare_environment](../../../.agents/skills/pypto-environment-setup/references/prepare_environment.md)  "CANN环境加载（通用模板）"章节，
  5. 利用系统自带如time、perf或其他shell命令结合bisheng命令统计时长，例如：

     ```bash
     time bisheng -c -O3 -g -x cce ... -o output/output_20260319_145742_163710_1702013_6466B4B5kernel_aicore/TENSOR_Step0_Unroll1_PATH0_hiddenfunc0_8_16874966534923480783_0_aiv.o outputoutput_20260319_145742_163710_1702013_6466B4B5/kernel_aicoreTENSOR_Step0_Unroll1_PATH0_hiddenfunc0_8_16874966534923480783_0_aiv.cpp
     ```

  6. 若确认编译时间较长，可查看生成的单kernel文件是否较长（例如超过5K行），若较长则可联合PASS考虑调整子图切分或其他手段减少生成的单kernel代码行数。
