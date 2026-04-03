# CODEGEN 组件错误码

- **范围**：F6XXXX
- 本文档说明 CODEGEN 组件的错误码定义、场景说明与排查建议。
---

## 错误码定义

相关错误码的统一定义，参见 `framework/src/codegen/utils/codegen_error.h` 文件。

---

## 排查建议

### 排查前准备

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
   - 重新编译并安装pypto包，参考[编译安装](../install/build_and_install.md#编译安装)

   ```bash
   cd pypto_project_path && python3 build_ci.py -f python3 --disable_auto_execute
   pip install build_out/pypto*.whl --force --no-deps
   cd -
   ```

3. **再次执行用例，获取日志及kernel代码文件**
   日志路径一般为：   *{用户指定日志路径}*/debug/plog/pypto-log***.log
   kernel代码文件路径一般为：   pypto工程路径或测试框架执行路径下，搜索kernel_aicore文件夹，文件夹内的TENSOR***.cpp即kernel代码文件。

4. **分析日志**
   - 对于FRAMEWORK（F60XXX）、OPERATION_ADAPTER（F61XXX）类错误，一般为上游数据异常导致，需要结合PASS日志分析
   - 其他类型错误需要结合上下文进行分析
<br>


### 场景举例

注意：所有场景日志分析均基于上述排查步骤中获取的日志为基础。

#### 生成kernel代码中某个TileOP调用参数不符合预期

1. 在kernel代码中找到不符合预期的TileOp调用，例如：

   ```c++
   TAdd<LastUse3Dim<0, 1, 1>>(ubTensor_0, ubTensor_0, ubTensor_2);
   ```

2. 以上面TileOp调用代码为关键字在日志中进行搜索。
3. 找到日志后往上搜索出现的第一个”Op CodeGenNPU Start”关键字，即该TileOp生成的开始位置，由此往后以此检查日志信息是否符合预期。
4. 若怀疑和PASS传入的数据有关，则可以在"Op CodeGenNPU Start"关键字后搜索"Gen OP IS"关键字，后面包含了该Operation的Dump信息，样例如下：

   ```c++
   Gen OP IS: <2 x 2 x 16 x 16 x DT_FP32 / sym_3_dim_0 x sym_3_dim_1 x sym_3_dim_2 x sym_3_dim_3 x DT_FP32> %152@5#(0)MEM_UB::MEM_UB = !10010 TILE_ADD(g:0, s:-1) %3@3#(0)MEM_UB::MEM_UB, %4@4#(0)MEM_UB::MEM_UB #IS_CUBE{0} #last_use{[0, 1, 1]}
   ```

   其中!10010即该OP的唯一标识码，可以此为关键字在PASS的图或日志中搜索获取相关信息，PASS定位指导详见[pass trouble shooting](./pass.md)
<br>


#### 错误码 F62014：SYMBOL_NOT_FOUND

该错误码表示kernel代码中调用了未定义的变量，常见错误场景如下：

##### Operation缺少need_alloc属性

该类错误场景日志上下文会包含"UNDEFINED_VAR"关键字。
CodeGen需要以来Operation中的need_alloc属性生成变量定义语句，若该属性缺失，则会导致变量定义语句缺失从而报错。
可结合前文[生成kernel代码中某个TileOP调用参数不符合预期](#生成kernel代码中某个TileOP调用参数不符合预期)步骤，找到缺失属性的Operation联合PASS继续定位。
<br>

#### 错误码 F63001：COMPILE_CODE_FAILED

kernel代码编译错误可能有多种原因导致，后续将根据不同场景完善排查指导。

##### 场景1: 堆栈溢出

报错关键字样例：

```log
error: stack frame size (*****) exceeds limit (32768) in function
```

可参考：  [算子编译报堆栈溢出错误](../tutorials/appendix/faq.md#算子编译报堆栈溢出错误)

##### 场景2: PTO指令数据类型不匹配

报错关键字"maybe need a type"，样例：

```log
/usr/local/Ascend/cann-9.0.0/include/pto/npu/a5/TStore.hpp:233:41: error: the 2nd parameter maybe need a type 'cc float *'
copy_matrix_cc_to_gm(dstGlobalAddr, srcTileAddr, xmReg, xtReg);
```

数据类型不匹配可能原因有：
- 前端调用Operation接口参数传递错误，参考：[执行代码有pto相关报错](https://gitcode.com/cann/pypto/issues/705)
- 使用了PTO-ISA不支持的数据类型，需要重新分析用例场景，使用硬件支持的数据类型

##### 场景3: 生成PTO指令和二进制编译参数指定的硬件平台不匹配

报错关键字"does not support the given target feature"，样例：

```log
error: function type 'void (__cbuf__ void *, __gm__ void *, unsigned char, unsigned short, unsigned short, unsigned short, unsigned short, unsigned int) noexcept' of 'copy_gm_to_cbuf' does not support the given target feature
    copy_gm_to_cbuf(dst, src, (uint8_t)0, nBurst, lenBurst, gmGap, l1Gap, (pad_t)0);
    ^
```

出现此类报错原因可能有：
- kernel代码编译参数为Vector，但是生成的kernel代码中包含了Cube相关指令，或者反之编译参数为Cube，但是生成的kernel代码中包含了Vector相关指令，导致bisheng编译器报错。
  该类问题一般为PASS将Vector和Cube的Operation在CodeGen阶段前仍然混合到一张子图导致，需要PASS进一步分析子图切分逻辑。CodeGen阶段看到的必须是独立的纯Vector或纯Cube子图。
- CodeGen使用Cube或Vector的编译参数依据为Function::IsCube()接口，需要PASS确认对不同子图，该接口设置的值是否正确。


##### 场景4: 变量未定义

1. 报错关键字包含sym_***：

```log
output/output_20260317_102613_935544_121641/kernel_aicore/TENSOR_loop_0_Unroll1_PATH0_hiddenfunc1_9_416851834981923603_3_aiv.cpp:16:70: error: use of undeclared identifier 'sym_209_dim_0'; did you mean 'sym_65_dim_0'?
UBTileTensorBF16Dim2_1 ubTensor_1((uint64_t)UB_S0_E512_T, (Shape2Dim(sym_209_dim_0, sym_209_dim_1)));
```

此类变量用于运行时动态获取Shape、Offset大小，数据来源于Function::GetDynParamTable接口，可在报错日志中往前搜索首个出现的"subprogram id"关键字，找到子图ID并告知PASS，由PASS继续分析变量缺失原因。
<br>


#### 二进制编译时长统计

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

由于当前CodeGen耗时主要为二进制编译，故专门将Top Function粒度的二进制编译时长记录于CodeGen模块的INFO级别日志中，可grep关键字"Top Function magic:"，该日志记录了该Top Function内的所有子图二进制编译时长总和（所有子图通过make并行编译），样例如下：

```log
[INFO ] PYPTO(726656):2026-03-19 10:59:35.426 [codegen_cloudnpu.cpp:760][CODEGEN]:Top Function magic: 8, hash: 16874966534923480783: Starting parallel compilation: 128 jobs, 1 tasks
[INFO ] PYPTO(726656):2026-03-19 10:59:36.135 [codegen_cloudnpu.cpp:768][CODEGEN]:Top Function magic: 8, hash: 16874966534923480783: Parallel compilation finished in 709.613831 ms

```
日志中记录了该Top Function内所有子图执行bisheng命令编译二进制的并发进程数量，以及总体耗时。

- 单个kernel文件编译时长确认方法：
  1. 从pypto工程路径或测试框架路径下找到kernel_aicore文件夹及需要验证的kernel代码文件Tensor**.cpp，例如：
     {前置路径}/output/output_20260319_145742_163710_1702013_6466B4B5/**kernel_aicore/TENSOR_Step0_Unroll1_PATH0_hiddenfunc0_8_16874966534923480783_0_aiv.cpp**
  2. 打开kernel代码文件，到最底部找到编译该文件的bisheng命令并复制，例如：

  ```bash
  bisheng -c -O3 -g -x cce ... -o output/output_20260319_145742_163710_1702013_6466B4B5/kernel_aicore/TENSOR_Step0_Unroll1_PATH0_hiddenfunc0_8_16874966534923480783_0_aiv.o output/output_20260319_145742_163710_1702013_6466B4B5/kernel_aicore/TENSOR_Step0_Unroll1_PATH0_hiddenfunc0_8_16874966534923480783_0_aiv.cpp
  ```
  3. cd {前置路径}
  确认当前在output文件夹上一层
  4. 执行刚刚复制的bisheng命令，确认可执行成功，若报bisheng命令找不到则参考:
  [prepare_environment](../../.agents/skills/pypto-environment-setup/references/prepare_environment.md)  "CANN 环境加载（通用模板）"章节
  . 利用系统自带如time、perf或其他shell命令结合bisheng命令统计时长，例如：

  ```bash
  time bisheng -c -O3 -g -x cce ... -o output/output_20260319_145742_163710_1702013_6466B4B5/kernel_aicore/TENSOR_Step0_Unroll1_PATH0_hiddenfunc0_8_16874966534923480783_0_aiv.o output/output_20260319_145742_163710_1702013_6466B4B5/kernel_aicore/TENSOR_Step0_Unroll1_PATH0_hiddenfunc0_8_16874966534923480783_0_aiv.cpp
  ```
