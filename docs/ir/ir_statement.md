# IR Statement

## OverView
整个 IR 的结构树大致为：
```text
program
  └── function
        ├── statement.op
        ├── statement.for
        │     ├── statement.if
        |     |     |   // then
        │     │     ├── statement.op
        │     │     ├── statement.yield
        |     |     |   // else
        │     │     ├── statement.op
        │     │     └── statement.yield
        │     ├── statement.op
        │     └── statement.yield
        ├── statement.op
        └── statement.return
```

示例1：
```text
program.module @main {
  program.entry @test_value
  attr arch = "PTOv2"
  attr enable_debug = true
  attr tile_default = { M=16, N=16, K=16 }
  func.func @test_value(%input_3: tensor<[%b_1, 128], fp32>, %scale1_4: fp32, %len_5: fp32, %output_8: tensor<[%b_1, 128], fp32>) {
    statement.op {
        %loop_tile_11 = tensor.view %input_3 : (tensor<[%b_1, 128], fp32>) -> tensor<[1, 128], fp32> {shape=[1, 128], offset=[0, 0]}
        %mul1_res_12 = tensor.mul %loop_tile_11, %scale1_4 : (tensor<[1, 128], fp32>, %scale1_4) -> tensor<[1, 128], fp32>
        %mul2_res_14 = tensor.mul %mul1_res_12, %const_pi_13 : (tensor<[1, 128], fp32>, 3.14) -> tensor<[1, 128], fp32>
        %output_15 = tensor.assemble %mul2_res_14, %output_8 : (tensor<[1, 128], fp32>, tensor<[%b_1, 128], fp32>) -> tensor<[1, 128], fp32> {offset=[0, 0]}
    }
    statement.return
  }
  // func.kind = control_flow
}
```

示例2：
```text
program.module @test_type_program {
  program.entry @test_type_complete
  attr arch = "PTOv2"
  attr enable_debug = true
  attr test_type = "tile_scalar_only"
  attr tile_default = { M=16, N=16, K=16 }
  func.func @test_type_complete(%input_tile_1: tile<[16, 32], [16, 32], fp32> #in, %scale_4: fp32 #in, %output_tile_5: tile<[16, 32], [16, 32], fp32> #out) -> (fp64) {
    statement.op {
        %const_2_9 = 2.0 : fp64
        %const_3_10 = 3.0 : fp64
        %tile_mul_11 = tile.OP_MUL %input_tile_1, %scale_4 : (tile<[16, 32], [16, 32], fp32>, fp32) -> tile<[16, 32], [16, 32], fp32>
        %tile_add_14 = tile.OP_ADD %tile_mul_11, %const_2_9 : (tile<[16, 32], [16, 32], fp32>, fp64) -> tile<[16, 32], [16, 32], fp32>
        %tile_sub_17 = tile.OP_SUB %tile_add_14, %const_3_10 : (tile<[16, 32], [16, 32], fp32>, fp64) -> tile<[16, 32], [16, 32], fp32>
        %tile_div_20 = tile.OP_DIV %tile_sub_17, %scale_4 : (tile<[16, 32], [16, 32], fp32>, fp32) -> tile<[16, 32], [16, 32], fp32>
        %output_tile_21 = tile.assemble %tile_div_20, %output_tile_5 : (tile<[16, 32], [16, 32], fp32>, tile<[16, 32], [16, 32], fp32>) -> tile<[16, 32], [16, 32], fp32> {offset=[0, 0]}
        %scalar1_23 = 10.5 : fp64
        %scalar2_24 = 5.2 : fp64
        %scalar_add_25 = tensor.OP_SCALAR_ADD %scalar1_23, %scalar2_24 : (fp64, fp64) -> fp64
        %scalar_mul_26 = tensor.OP_SCALAR_MUL %scalar_add_25, %const_2_9 : (fp64, fp64) -> fp64
    }
    statement.return %scalar_mul_26
  }
  // func.kind = kernel

}
```

## Program
Program 是用于编译和执行 PTO 应用程序的容器，pass 处理后生成的 Dataflow Function 和 Kernel Function 也将放入 Program 中。
### Syntax
```text
program.module @main {
  program.entry @test_value
  attr arch = "PTOv2"
  attr enable_debug = true
  attr tile_default = { M=16, N=16, K=16 }
  ...
}
```
Program 内包含 Function 和属性 attr。

### 数据结构
```cpp
class ProgramModule : public Object {
    std::shared_ptr<Function> programEntry_;
    std::vector<std::shared_ptr<Function>> functions_;
};
```

### Entry
Program 必须指定唯一的入口函数：
```text
program.entry @test_value
```

### Attr
可自定义属性字段。

### 约束
- 一个程序只有一个 `program.module`。
- 必须有唯一的 `program.entry`，且该函数必须是 `program.module` 内的 `func.func`。
- `program.module` 禁止嵌套。

## Function
表示一个SSA形式的函数抽象，它定义了函数名、函数参数以及函数类型。
### Syntax
```text
func.func @test_value(%input_3: tensor<[%b_1, 128], fp32>, %scale1_4: fp32, %len_5: fp32, %output_8: tensor<[%b_1, 128], fp32>) -> (int32) {
    // CompoundStatement
    statement.for{}
    statement.op{}
}
// func.kind = control_flow
```
- 函数名 `test_value`。
- 参数列表 `(%input_3: tensor<[%b_1, 128], fp32>, %scale1_4: fp32, %len_5: fp32, %output_8: tensor<[%b_1, 128], fp32>)`。
- 返回类型 `-> (int32)`。
- 函数类型 `// func.kind = control_flow`。
- 函数体包含 if/for 结构化控制流和线性op序列构成的基本块。

### 数据结构
```cpp
class Function : public Object {
    FunctionKind kind_;
    FunctionSignature signature_;
    // Scope holding function arguments (inputs)
    CompoundStatementPtr inputCompound_; 
    // Scope for Data objects and statements created in this function
    CompoundStatementPtr compound_;  
};
```

### 函数类型 func.kind
- ControlFlow：对应 host 执行的 control_flow function。
- DataFlow：对应 Root function。
- Kernel：对应 Leaf/Execute function。

### 函数体
Function 中有且只有一个 `CompoundStatement` 构成函数体，它管理 Function 中的作用域和其他 statement 语句（如 statement.if, statement.for, statement.op 和 statement.return）。

### Attr
可自定义属性字段。

### 约束
- 参数可以是 Tensor，Tile 或 Scalar

## Statement
Statement 表示了 For/If 控制流和线性的 Op 序列。主要有以下类别：
- `statement.compound`
- `statement.for`
- `statement.if`
- `statement.op`
- `statement.yield`
- `statement.return`

注：Statement 中可以看到其前序 Statement 中定义的值。但 for 和 if 内部 statement 定义的值对 for/if 后的 statement 是不可见的。

### CompoundStatement
它不在序列化的文本格式 text 上体现出来，只有内存中的数据结构。其完成作用管理的功能，作为 Function、ForStatement 和 IfStatement 的成员对象。其他的 Statement 记录在 CompoundStatement 中。

#### 数据结构
```cpp
class CompoundStatement : public Statement {
    // Pointer to parent scope (nullptr for root)
    std::weak_ptr<CompoundStatement> parent_{}; 
    // Statements in this scope                        
    std::vector<StatementPtr> statements_; 
    // Envtextonment table: variable name -> latest SSA Value 
    std::unordered_map<std::string, ValuePtr> envTable_; 
};
```

### ForStatement
表示循环结构，具有显式的循环迭代变量和循环传递值（loop-carried value）。

#### Syntax
```text
%res0, %res1 = statement.for %iv = %lb to %ub step %step
                iter_args(%acc0 = %init0 : T0, %acc1 = %init1 : T1, ...)
                [attributes]
{
    // 可以有 for/text/op statement 等
    statement.op {
        %new_acc0 = ...
        %new_acc1 = ...
    }
    statement.yield %new_acc0, %new_acc1
}
```
- 循环迭代变量、上下界和步长：`%iv = %lb to %ub step %step`。
- 循环传递值定义，即循环迭代中修改的变量： `iter_args`。`%acc0` 和 `%acc1` 为循环体中旧值的 SSA 值， `%new_acc0` 和 `%new_acc1` 为循环体中新计算值的 SSA 值，`%init0` 和 `%init1` 为循环开始前的初始值，`%res0` 和 `%res1` 为循环结束后的最终值。
- 结果值：`%res0`，`%res1`。

#### 数据结构
```cpp
class ForStatement : public Statement {
std::shared_ptr<Scalar> iterationVar_;
    std::shared_ptr<LoopRange> range_;
    std::vector<IterArg> iterArgs_;
    // Scope for Data objects and statements created in this loop body
    CompoundStatementPtr compound_;  
    // Result values of the for-statement
    std::vector<ValuePtr> results_; 
};
```

#### 约束
- 若存在 `iter_args`，则 `iter_args`、`results`、`yield` 中的值是一一对应的，数量和类型完全一致。且 `iter_args[i]`，`iter_args[i]`，`yield[i]` 表示用户前端定义的一个变量的不同时期的值。
- for 内部最后一个 statement 必须是 `yield`。

### IfStatement
表示 If-else 分支结构。

#### Syntax
```text
%r0, %r1 = statement.if %cond
{
  // then-region statements
  ...
  statement.yield %then_v0, %then_v1, ...
} else {
  // else-region statements
  ...
  statement.yield %else_v0, %else_v1, ...
}
```
- 判断条件：`%cond`
- 结果值：`%r0`, `%r1`

#### 数据结构
```cpp
class IfStatement : public Statement {
    std::string condition_;
    // Scope for Data objects and statements created in the then branch
    CompoundStatementPtr thenCompound_;  
    // Scope for Data objects and statements created in the else branch
    CompoundStatementPtr elseCompound_; 
    // Result values of the if-statement 
    std::vector<ValuePtr> results_;  
};
```

#### 约束
- `%cond` 必须为布尔类型的 Scalar。
- then 和 else 拥有独立的作用域，并且均以 `yield` 结束。
- IR 上一定有 then 块和 else 块，无论前端是否写有 else。
- then 和 else 中的 `yield` 数量与类型完全一致，且与 if 的 results 一致。若 then 和 else 中修改的变量不同，builder会自动补齐 `yield`。

### OpStatement
一组顺序的 Op 组成的基本语句块，其内部没有嵌套的控制流语句。

#### Syntax
```text
statement.op {
    // linear sequence of operations
    %v0 = ...
    %v1 = ...
    ...
}
```

#### 数据结构
```cpp
class OpStatement : public Statement {
    std::vector<OperationPtr> operations_;
};
```

#### 约束
- 所有 Op 必须位于 OpStatement 中。

### YieldStatement
statement.yield 是一个通用的区域终止符，用于将当前作用域的值返回到其父语句。

#### Syntax
```text
statement.yield %value0, %value1, ...
```

使用：
- statement.if 的 then/else 作用域结尾，将每个分支的修改结果回传给 statement.if。
- 在 statement.for 循环体结尾，用于提供更新后的循环传递值。

### ReturnStatement
表示函数执行结束，仅可返回可通过寄存器传递的值。输出 tensor 通过参数方式传递。

#### Syntax
```text
statement.return %value0, %value1, ...
```

#### 数据结构
```cpp
class ReturnStatement : public Statement {
    std::vector<ValuePtr> values_;
};
```

#### 约束
- 目前 ReturnStatement 必须且仅能在函数尾出现。