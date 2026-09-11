# F6XXXX

<!-- md-trans-meta sourceCommit=a560146b679ecdf0f7f2ac817828151d6a71b67d translatedAt=2026-08-11T09:08:02.167Z pushedAt=2026-08-26T11:03:10.616Z -->

## F62014 SYMBOL_NOT_FOUND

**Error Description**

A variable is undefined during the kernel code generation phase. The log context contains the `UNDEFINED_VAR` keyword.

**Possible Causes**

- The Operation in the subgraph is missing the `need_alloc` attribute. CodeGen relies on this attribute to generate variable definition statements, and its absence leads to missing variable definitions.

**Solution**

1. **Set the log level to INFO**:

   ```bash
   export ASCEND_PROCESS_LOG_PATH=<User-defined log path>
   export ASCEND_GLOBAL_LOG_LEVEL=1  # 0:DEBUG, 1:INFO, 2:WARN, 3:ERROR
   ```

2. **Switch from parallel compilation to serial compilation**: Modify the `"parallel_compile"` option in `tile_fwk_config.json` and change its value to **1** to enable serial codegen compilation. Then recompile and deploy.

3. **Run the use case and obtain the log and kernel code**:
   - Log: `{Log path}/debug/plog/pypto-log*.log`
   - Kernel code: Search for `TENSOR*.cpp` in the `kernel_aicore/` folder.

4. **Analyze logs:**
   - F60XXX / F61XXX errors → Generally caused by upstream data anomalies. Analyze them together with PASS logs.
   - Other types → Analyze them in context.

> **Example**: Take the case where TileOp call parameters in the kernel code do not meet expectations.

1. Collect logs by following the preceding steps.
2. Find the TileOp call that does not meet expectations, for example:

   ```c++
   TAdd<LastUse3Dim<0, 1, 1>>(ubTensor_0, ubTensor_0, ubTensor_2);
   ```
3. Search the logs using the preceding code as a keyword.
4. Search upward for the first `Op CodeGenNPU Start`, which is the starting point of this TileOp generation, and then check line by line going forward.
5. If you suspect a PASS data issue, search for `Gen OP IS` to obtain the Operation Dump information:
   ```log
   Gen OP IS: <2 x 2 x 16 x 16 x DT_FP32> %152@5#(0)MEM_UB = !10010 TILE_ADD(...) ...
   ```
   Here, `!10010` is the unique identifier of this OP, which can be searched in the PASS graph or logs. For details about PASS locating, see [pass.md](pass.md).

## F63001 COMPILE_CODE_FAILED

**Error Description**

Kernel code compilation failed.

> If the BiSheng Compiler error log is incomplete, you can locate the make command after `compile cmd is:` in the log and manually execute it to obtain the complete error information.

**Possible Causes**

- **Stack overflow** (`error: stack frame size exceeds limit`): The function stack frame exceeds the limit.
- **PTO instruction data type mismatch** (`the 2nd parameter maybe need a type`): The frontend API parameter is passed incorrectly, or a data type not supported by PTO-ISA is used.
- **Hardware instruction/platform mismatch** (`does not support the given target feature`): The compilation parameter is Vector but the code contains Cube instructions (or vice versa).
- **Undefined variable** (`use of undeclared identifier`): A runtime dynamic Shape/Offset variable is missing. The data originates from `Function::GetDynParamTable`.

**Solution**

1. Stack overflow: See [Stack Overflow Error During Operator Compilation](../faq/stack-overflow-compilation.md#stack-overflow-error-during-operator-compilation).
2. Unsupported data type: Re-analyze the operator computation flow and select a data type supported by the hardware.
3. Instruction-platform mismatch:
   - Check whether the block subgraph is purely Vector or purely Cube (CodeGen must not mix them).
   - Check whether the PASS setting for `Function::IsCube()` is correct.
4. Variable undefined: Troubleshoot with PASS to check whether the variable set returned by `Function::GetDynParamTable` has any omissions.


## Compilation Duration Statistics

### Overall Duration

After executing the operator, the screen outputs the `Compiler Monitor` statistics:

```log
[Compiler Monitor] Stage: CodeGen(completed) | Stage elapsed: 1.2s | Total elapsed: 1.2s
[Compiler Monitor] Compilation finished 6/6 | Total functions: 6
[Compiler Monitor] Stage timing (aggregated by stage):
  CodeGen  1.2s   (sum over 6 functions)
  Pass     0.0s   (sum over 6 functions)
  Prepare  0.0s
```

### Compilation Duration of a Single Kernel File

1. Locate the target `TENSOR*.cpp` file under `kernel_aicore/`, and copy the BiSheng compilation command from the bottom of the file.
2. `cd` to the parent directory of `output`, and run the command to verify it works.
3. Use `time` to measure the duration:

   ```bash
   time bisheng -c -O3 -g -x cce ... -o .../TENSOR_xxx.o .../TENSOR_xxx.cpp
   ```

4. If the compilation takes a long time and the single kernel file exceeds 5000 lines, consider adjusting the subgraph partitioning together with PASS.
