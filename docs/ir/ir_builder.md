# IRBuilder

## OverView

`IRBuilder` 是 PTO-IR 中用于构建中间表示（IR）的核心工具类。它提供了类型安全、结构化的 API 来构建程序模块（ProgramModule）、函数（Function）、语句（Statement）和操作（Operation）。

IRBuilder 采用显式上下文（Explicit Context）设计，所有构建状态存储在 `IRBuilderContext` 对象中，通过栈式作用域管理实现嵌套作用域。

IRBuilder 的使用流程：
```
IRBuilder
├── 创建模块和函数
│   ├── CreateFunction()
│   └── EnterFunctionBody()
├── 创建值对象
│   ├── CreateTensor()
│   ├── CreateScalar()
│   └── CreateConst()
├── 构建操作
│   ├── CreateBinaryOp() / CreateUnaryOp() 等
│   └── Emit()
└── 构建语句
    ├── CreateOpStmt()
    ├── CreateForStmt()
    ├── CreateIfStmt()
    ├── CreateReturn()
    └── CreateYield()
```

示例：
```cpp
// ===== Module =====
auto module = std::make_shared<ProgramModule>("main");
IRBuilder builder(module);
IRBuilderContext ctx;

// ===== Signature =====
FunctionSignature sig;

// tensor<[b, 128], fp32>
auto batch = std::make_shared<ScalarValue>(DataType::INT32, "b", ScalarValueKind::Symbolic);
std::vector<ScalarValuePtr> tensorShape = { batch, std::make_shared<ScalarValue>(int64_t(128)) };

auto inputTensor  = std::make_shared<TensorValue>(tensorShape, DataType::FP32, "input");
auto scale1       = std::make_shared<ScalarValue>(DataType::FP32, "scale1", ScalarValueKind::Symbolic);

auto result = std::make_shared<TensorValue>(tensorShape, DataType::FP32, "output");

sig.arguments = { inputTensor, scale1, result };

// ===== Function =====
auto func = builder.CreateFunction("test_value", FunctionKind::ControlFlow, sig, /*setAsEntry=*/true);

// enter func scope + create an initial block as insertion point
builder.EnterFunctionBody(ctx, func);

// mul1_res = mul(input, scale1)
auto mulVal1 = builder.CreateTensor(ctx, tensorShape, DataType::FP32, "mul1_res");
auto mulOp1 = builder.CreateBinaryOp(Opcode::OP_MUL, inputTensor, scale1, mulVal1);
builder.Emit(ctx, mulOp1);

auto pi = builder.CreateConst(ctx, 3.14, "const_pi");

// mul2_res = mul(mul1_res, pi)
auto mulVal2 = builder.CreateTensor(ctx, tensorShape, DataType::FP32, "output");
auto mulOp2 = builder.CreateBinaryOp(Opcode::OP_MUL, mulVal1, pi, mulVal2);
builder.Emit(ctx, mulOp2);

builder.CreateReturn(ctx, { });
ctx.PopScope();

```

## IRBuilder

IRBuilder 是无状态的工具类，所有构建状态存储在显式的 `IRBuilderContext` 对象中。通过栈式作用域管理（`PushScope`/`PopScope`）实现嵌套作用域。

### Syntax
```cpp
IRBuilder builder(module);
IRBuilderContext ctx;  // 显式上下文对象
auto func = builder.CreateFunction(name, kind, sig, setAsEntry);
builder.EnterFunctionBody(ctx, func);
// ... 构建IR ...
ctx.PopScope();  // 手动退出作用域
```

### 数据结构
```cpp
class IRBuilder {
    std::shared_ptr<ProgramModule> module_;  // 唯一的状态
};

struct IRBuilderContext {
    std::shared_ptr<Function> func{nullptr};
    std::shared_ptr<CompoundStatement> compound{nullptr};
    std::shared_ptr<OpStatement> activeOpStmt{nullptr};
    std::vector<ScopeFrame> scopeStack;  // 作用域栈
    
    void PushScope(CompoundStatementPtr newCompound, FunctionPtr newFunc = nullptr);
    void PopScope();
    void ResetInsertionPoint();
};
```

### 约束
- IRBuilder 必须关联一个 ProgramModule
- 所有构建方法都需要传入 `IRBuilderContext& ctx` 参数
- 使用显式的 `PushScope`/`PopScope` 管理作用域，确保作用域正确嵌套

## 模块和函数管理

### CreateFunction

创建新函数并添加到模块中。

#### Syntax
```cpp
auto func = builder.CreateFunction(
    "test_value",
    FunctionKind::ControlFlow,
    sig,
    /*setAsEntry=*/true
);
```

#### 数据结构
```cpp
std::shared_ptr<Function> CreateFunction(
    std::string name,
    FunctionKind kind,
    FunctionSignature sig,
    bool setAsEntry = false
);
```

**参数**：
- `name`: 函数名称
- `kind`: 函数类型（ControlFlow、DataFlow、Kernel）
- `sig`: 函数签名（参数和返回值）
- `setAsEntry`: 是否设置为程序入口点

**返回**：创建的 Function 对象

#### 约束
- 函数名在模块内必须唯一
- 函数签名中的参数类型必须是 Tensor、Tile 或 Scalar

### EnterFunctionBody

进入函数体作用域，将函数体的 CompoundStatement 推入作用域栈。

#### Syntax
```cpp

builder.EnterFunctionBody(ctx, func);
// 在函数体作用域内构建IR
// ...
ctx.PopScope();  // 手动退出作用域

```

#### 数据结构
```cpp
void EnterFunctionBody(IRBuilderContext& ctx, std::shared_ptr<Function> func);
```

#### 约束
- 进入作用域后必须手动调用 `ctx.PopScope()` 退出
- `EnterFunctionBody` 会调用 `ctx.PushScope()`，将当前状态保存到栈中并切换到函数体作用域

## 值创建

### CreateTensor

创建 Tensor 值对象，并自动添加到传入的作用域的环境表中。

#### Syntax
```cpp
std::vector<ScalarValuePtr> shape = { batch, std::make_shared<ScalarValue>(int64_t(128)) };
auto tensor = builder.CreateTensor(ctx, shape, DataType::FP32, "input");
```

#### 数据结构
```cpp
std::shared_ptr<TensorValue> CreateTensor(
    IRBuilderContext& ctx,
    const std::vector<ScalarValuePtr>& shape, 
    DataType dt, 
    std::string name = ""
);
```

#### 约束
- shape 中的 ScalarValue 可以是常量或符号值
- 必须在函数体作用域内调用（`ctx.compound` 不能为空）
- 创建的值会自动添加到当前 ctx 作用域的环境表中

### CreateScalar

创建符号标量（Symbolic Scalar）值对象，并自动添加到当前穿拖鞋、作用域的环境表中。

#### Syntax
```cpp
auto scalar = builder.CreateScalar(ctx, DataType::FP32, "scale1");
auto iv = builder.CreateScalar(ctx, DataType::INT32, "i");
```

#### 数据结构
```cpp
std::shared_ptr<ScalarValue> CreateScalar(
    IRBuilderContext& ctx,
    DataType dt, 
    std::string name = ""
);
```

#### 约束
- 创建的标量是符号值（Symbolic），运行时确定
- 必须在函数体作用域内调用（`ctx.compound` 不能为空）
- 创建的值会自动添加到当前 ctx 作用域的环境表中

### CreateConst

创建常量标量值对象，并自动添加到当前 ctx 作用域的环境表中。

#### Syntax
```cpp
auto const0 = builder.CreateConst(ctx, int64_t(0), "const_0");
auto const1 = builder.CreateConst(ctx, int64_t(1), "const_1");
auto pi = builder.CreateConst(ctx, 3.14, "const_pi");
```

#### 数据结构
```cpp
std::shared_ptr<ScalarValue> CreateConst(IRBuilderContext& ctx, int64_t v, std::string name = "");
std::shared_ptr<ScalarValue> CreateConst(IRBuilderContext& ctx, double v, std::string name = "");
```

#### 约束
- 支持 int64_t 和 double 类型的常量
- 类型自动推断（int64_t → INT64，double → FP64）
- 必须在函数体作用域内调用（`ctx.compound` 不能为空）
- 创建的值会自动添加到当前 ctx 作用域的环境表中

## 操作构建

### 操作创建和添加

操作通过 Schema 系统统一处理。使用 `CreateBinaryOp`、`CreateUnaryOp` 等方法创建操作对象，然后通过 `Emit` 方法添加到指定 ctx 的 OpStatement。

#### Syntax
```cpp
// 创建二元操作
auto result = builder.CreateTensor(ctx, tensorShape, DataType::FP32, "result");
auto mulOp = builder.CreateBinaryOp(Opcode::OP_MUL, input1, input2, result);
builder.Emit(ctx, mulOp);

// 创建一元操作
auto output = builder.CreateTensor(ctx, tensorShape, DataType::FP32, "output");
auto negOp = builder.CreateUnaryOp(Opcode::OP_NEG, input, output);
builder.Emit(ctx, negOp);
```

#### 数据结构
```cpp
// 通过 DEFOP 宏生成的操作创建方法
OperationPtr CreateBinaryOp(Opcode opcode, ValuePtr lhs, ValuePtr rhs, ValuePtr output);
OperationPtr CreateUnaryOp(Opcode opcode, ValuePtr input, ValuePtr output);
// ... 其他操作类型

// 添加操作到当前 OpStatement
OperationPtr Emit(IRBuilderContext& ctx, OperationPtr op);
```

**参数**：
- `opcode`: 操作码
- `lhs`/`rhs`/`input`: 输入值
- `output`: 输出值（必须预先创建）
- `ctx`: IRBuilder 上下文

**返回**：创建的 Operation 对象

#### 约束
- 所有操作必须通过 `Emit` 方法添加，才会添加到指定 ctx 的 OpStatement
- 输出值必须预先创建（通过 `CreateTensor`、`CreateScalar` 等）
- 如果当前没有活动的 OpStatement，`Emit` 会自动创建一个
- 输入值的类型必须符合操作的 Schema 定义
- 必须在函数体作用域内调用

## 语句构建

### CreateOpStmt

创建操作语句，作为操作的容器，并设置为当前活动的 OpStatement。

#### Syntax
```cpp
auto opStmt = builder.CreateOpStmt(ctx);
// 后续的 Emit 调用会将操作添加到这个 OpStatement 中
```

#### 数据结构
```cpp
OpStatementPtr CreateOpStmt(IRBuilderContext& ctx);
```

#### 约束
- 如果当前没有活动的 OpStatement，`Emit` 会自动创建一个
- OpStatement 中可以包含多个操作，按顺序执行
- 必须在函数体作用域内调用（`ctx.compound` 不能为空）

### CreateForStmt

创建 for 循环语句。

#### Syntax
```cpp
auto i = builder.CreateScalar(ctx, DataType::INT32, "i");
auto constant0 = builder.CreateConst(ctx, int64_t(0), "const_0");
auto constant1 = builder.CreateConst(ctx, int64_t(1), "const_1");
auto batch = ...; // 循环上界

auto fs = builder.CreateForStmt(ctx, i, constant0, batch, constant1);

builder.EnterForBody(ctx, fs);
// 循环体内的操作
// ...
ctx.PopScope();  // 退出循环体作用域
builder.ExitForStatement(ctx, fs);  // 处理循环携带变量
```

#### 数据结构
```cpp
ForStatementPtr CreateForStmt(
    IRBuilderContext& ctx,
    ScalarValuePtr iv,
    ScalarValuePtr start,
    ScalarValuePtr end,
    ScalarValuePtr step
);
```

**参数**：
- `ctx`: IRBuilder 上下文
- `iv`: 循环变量（induction variable）
- `start`: 起始值
- `end`: 结束值
- `step`: 步长

#### 约束
- 循环变量、起始值、结束值、步长都必须是 ScalarValue 类型
- 必须使用 `EnterForBody()` 进入循环体作用域
- 循环体末尾必须手动调用 `ctx.PopScope()` 退出作用域
- 退出循环体后必须调用 `ExitForStatement()` 处理循环携带变量
- `ExitForStatement()` 不会自动 pop scope，只负责处理循环携带变量和 yield
- IRBuilder 会自动识别循环携带变量并创建 iter_args

### CreateIfStmt

创建 if 条件语句。

#### Syntax
```cpp
auto cond = builder.CreateScalar(ctx, DataType::BOOL, "cond");
auto ifs = builder.CreateIfStmt(ctx, cond);
ValuePtr resIfX, resIfY;

builder.EnterIfThen(ctx, ifs);
resIfX = builder.CreateTensor(ctx, tensorShape, DataType::FP32, "outputX");
auto mulOpX = builder.CreateBinaryOp(Opcode::OP_MUL, resLoopX, scale1, resIfX);
builder.Emit(ctx, mulOpX);
ctx.PopScope();  // 退出 then 分支作用域

builder.EnterIfElse(ctx, ifs);
resIfY = builder.CreateTensor(ctx, tensorShape, DataType::FP32, "outputY");
auto mulOpY = builder.CreateBinaryOp(Opcode::OP_MUL, resLoopY, scale2, resIfY);
builder.Emit(ctx, mulOpY);
ctx.PopScope();  // 退出 else 分支作用域

builder.ExitIfStatement(ctx, ifs);  // 处理分支合并
```

#### 数据结构
```cpp
IfStatementPtr CreateIfStmt(IRBuilderContext& ctx, ScalarValuePtr cond);
```

**参数**：
- `ctx`: IRBuilder 上下文
- `cond`: 条件表达式（必须是 ScalarValue 类型）

#### 约束
- 条件必须是 ScalarValue 类型
- 必须使用 `EnterIfThen()` 和 `EnterIfElse()` 分别进入 then 和 else 分支
- 每个分支结束时必须手动调用 `ctx.PopScope()` 退出作用域
- 必须调用 `ExitIfStatement()` 完成 if 语句构建（处理分支合并）
- `ExitIfStatement()` 不会自动 pop scope，只负责处理分支合并和 yield
- IRBuilder 会自动在 then/else 分支末尾添加 yield 语句
- 若 then 和 else 中修改的变量不同，builder 会自动补齐 yield

### CreateReturn

创建返回语句。

#### Syntax
```cpp
builder.CreateReturn(ctx, { scalar });
```

#### 数据结构
```cpp
ReturnStatementPtr CreateReturn(IRBuilderContext& ctx, ValuePtrs values);
```

#### 约束
- 返回值必须是可通过寄存器传递的值（Scalar）
- 输出 tensor 通过参数方式传递，不在返回值中
- 目前 ReturnStatement 必须且仅能在函数尾出现
- 必须在函数体作用域内调用（`ctx.compound` 不能为空）

### CreateYield

创建 yield 语句（用于循环和条件分支的值传递）。

#### Syntax
```cpp
builder.CreateYield(ctx, { new_acc0, new_acc1 });
```

#### 数据结构
```cpp
YieldStatementPtr CreateYield(IRBuilderContext& ctx, ValuePtrs values);
```

#### 约束
- 在 for 循环体中，yield 用于提供更新后的循环传递值
- 在 if 的 then/else 分支中，yield 用于将分支结果回传给 if 语句
- IRBuilder 通常会在 `ExitForStatement()` 和 `ExitIfStatement()` 中自动添加 yield，一般不需要手动调用
- 必须在函数体作用域内调用（`ctx.compound` 不能为空）

## 作用域管理

IRBuilder 使用显式的栈式作用域管理，通过 `IRBuilderContext` 的 `PushScope`/`PopScope` 方法管理嵌套作用域。

### IRBuilderContext

`IRBuilderContext` 存储所有构建状态，包括当前函数、复合语句、活动操作语句和作用域栈。

#### Syntax
```cpp
IRBuilderContext ctx;  // 创建上下文对象

builder.EnterFunctionBody(ctx, func);  // 进入函数体作用域
// 在函数体作用域内
    
builder.EnterForBody(ctx, fs);  // 进入循环体作用域
// 在循环体作用域内
// ...
ctx.PopScope();  // 手动退出循环体作用域
    
ctx.PopScope();  // 手动退出函数体作用域

```

#### 作用域栈（Scope Stack）

`IRBuilderContext` 维护一个作用域栈（`scopeStack`），每个栈帧（`ScopeFrame`）保存：
- `compound`: 之前的复合语句 (scope)
- `func`: 之前的函数
- `activeOpStmt`: 之前的活跃的操作语句集合

#### 作用域层次
1. **函数输入作用域（InputCompound）**：存储函数参数
2. **函数体作用域（Compound）**：函数体的主作用域，是输入作用域的子作用域
3. **嵌套作用域**：For、If 等语句创建嵌套的作用域

#### 约束
- 作用域之间通过 `GetAncestorValues()` 查找父作用域中的变量，实现词法作用域语义
- 必须手动调用 `ctx.PopScope()` 退出作用域，确保作用域栈的正确性
- `EnterFunctionBody`、`EnterForBody`、`EnterIfThen`、`EnterIfElse` 只负责 push scope
- `ExitForStatement` 和 `ExitIfStatement` 不负责 pop scope，只处理合并逻辑
- 子作用域中的变量对父作用域不可见

### 环境表（Environment Table）

每个 CompoundStatement 维护一个环境表（EnvTable），用于存储作用域内的变量绑定：

- Key: 变量的 SSA 名称
- Value: 变量的值对象

环境表支持：
- `SetEnvVar()`: 设置变量绑定
- `GetEnvVar()`: 获取变量值（在当前 ctx 作用域）
- `GetAncestorValues()`: 获取所有祖先作用域中的变量

### 循环携带变量（Loop-Carried Variables）

在 `ExitForStatement()` 中，IRBuilder 会自动识别和处理循环携带变量。**注意**：`ExitForStatement()` 不会自动 pop scope，只负责处理循环携带变量的逻辑。

处理流程：
1. 查找在循环体中被修改的变量（通过比较循环前后的环境表）
2. 为每个循环携带变量创建 `iter_arg` 结构
3. 在循环体内创建对应的值对象（用于替换循环体中的初始值引用）
4. 在循环体末尾添加或更新 yield 语句
5. 调用 `ForStatement::BuildResult()` 构建循环结果
6. 更新父作用域环境表

**重要**：必须在调用 `ExitForStatement()` 之前已经通过 `ctx.PopScope()` 退出了循环体作用域。

### 条件分支合并（If Statement Merging）

在 `ExitIfStatement()` 中，IRBuilder 会自动处理条件分支的变量合并。**注意**：`ExitIfStatement()` 不会自动 pop scope，只负责处理分支合并的逻辑。

处理流程：
1. 识别在 then/else 分支中被修改的变量（通过比较分支前后的环境表）
2. 在 then/else 分支末尾添加或更新 yield 语句
3. 调用 `IfStatement::BuildResult()` 构建合并结果
4. 更新父作用域环境表

**重要**：必须在调用 `ExitIfStatement()` 之前已经通过 `ctx.PopScope()` 退出了 then 和 else 分支的作用域。
