# IR Type 和 Value

## OverView

PTO-IR 的类型系统（Type System）和值系统（Value System）是 IR 的核心组成部分。类型系统描述了数据的结构（如形状、数据类型），值系统表示具体的值对象（如标量、张量、Tile）。

类型与值的层次结构：

```
Type (基类)
├── ScalarType (标量类型)
├── TileType (Tile 类型)
└── TensorType (张量类型)

Value (基类)
├── Scalar (标量值)
├── Tile (Tile 值)
└── Tensor (张量值)
```

每个 Value 对象都包含一个 Type 对象，用于描述该值的类型信息：
- `Scalar` → `ScalarType`
- `Tile` → `TileType`
- `Tensor` → `TensorType`

示例：

```text
func.func @test_value(%input_3: tensor<[%b_1, 128], fp32>, %scale1_4: fp32) {
  statement.op {
    %const_0 = 0 : int64
    %const_pi_13 = 3.14 : fp64
    %loop_tile_11 = tensor.view %input_3 : (tensor<[%b_1, 128], fp32>) -> tensor<[1, 128], fp32>
    %mul1_res_12 = tensor.mul %loop_tile_11, %scale1_4 : (tensor<[1, 128], fp32>, fp32) -> tensor<[1, 128], fp32>
  }
}
```

## DataType

`DataType` 枚举定义了所有支持的基本数据类型。

### Syntax

```cpp
enum class DataType {
    BOOL,
    INT4, INT8, INT16, INT32, INT64,
    UINT8, UINT16, UINT32, UINT64,
    FP8, FP16, BF16, FP32, FP64,
    HF4, HF8,
    BOTTOM, UNKNOWN
};
```

### 约束

- 所有 Value 和 Type 必须关联一个有效的 DataType（不能是 BOTTOM 或 UNKNOWN，除非特殊情况）。

## Type 系统

Type 系统用于描述数据的结构信息。所有类型都继承自 `Type` 基类。

### Type

`Type` 是所有类型的基类，提供通用接口。

#### 数据结构

```cpp
class Type {
public:
    DataType GetDataType() const;
    static size_t GetDataTypeSize(DataType dataType);
    size_t GetDataTypeSize() const;
    virtual size_t GetTypeSize() const = 0;
    virtual void Print(std::ostream& os) const = 0;
};
```

#### 关键方法

- `GetDataType()`: 获取数据类型枚举值
- `GetDataTypeSize()`: 获取数据类型的字节大小（静态方法和实例方法）
- `GetTypeSize()`: 获取类型的总大小（标量 = 数据类型大小，Tile = 元素大小 × 元素数量）
- `Print()`: 打印类型信息

### ScalarType

`ScalarType` 表示标量类型（单个值）。

#### Syntax

```text
fp32
int32
bool
```

在 IR 中，标量类型直接显示为数据类型名称。

#### 数据结构

```cpp
class ScalarType : public Type {
public:
    explicit ScalarType(DataType dataType);
    size_t GetTypeSize() const override;  // 返回 GetDataTypeSize()
    void Print(std::ostream& os) const override;
};
```

#### 约束

- 类型大小等于数据类型大小
- 打印格式：数据类型名称（如 `fp32`、`int32`）

### TileType

`TileType` 表示 Tile 类型（固定形状的多维数组）。

#### Syntax

```ir
tile<[16, 32], fp32>
tile<[M, N], fp16>
```

在 IR 中，Tile 类型显示为 `tile<[shape], dtype>` 格式。

#### 数据结构

```cpp
class TileType : public Type {
public:
    TileType(DataType elementType, const std::vector<size_t>& shape);
    const std::vector<size_t>& GetShape() const;
    size_t GetTypeSize() const override;  // 元素大小 × 形状乘积
    void Print(std::ostream& os) const override;
};
```

#### 约束

- 包含静态形状信息（`std::vector<size_t>`）
- 类型大小 = 元素数据类型大小 × 所有维度的乘积
- 打印格式：`tile<[shape], dtype>`（如 `tile<[16, 32], fp32>`）

### TensorType

`TensorType` 表示张量类型（动态形状）。

#### Syntax

```text
tensor<[%b_1, 128], fp32>
tensor<[%M, %N, %K], fp16>
```

在 IR 中，Tensor 类型显示为 `tensor<[shape], dtype>` 格式，其中 shape 可以包含符号维度。

#### 数据结构

```cpp
class TensorType : public Type {
public:
    explicit TensorType(DataType dataType);
    size_t GetTypeSize() const override;  // 返回 GetDataTypeSize()
    void Print(std::ostream& os) const override;
};
```

#### 约束

- 形状信息存储在 Value 对象（Tensor）中，而不是 Type 中
- 类型大小等于数据类型大小（形状在运行时确定）
- 打印格式：数据类型名称（如 `fp32`），但完整的形状信息在 Value 中显示

## Value 系统

Value 系统用于表示 IR 中的具体值对象。所有值都继承自 `Value` 基类，遵循 SSA（静态单赋值）形式。

### ValueKind 枚举

`ValueKind` 枚举定义了值的种类：

```cpp
enum class ValueKind {
    Scalar,
    Tile,
    Tensor
};
```

### ScalarValueKind 枚举

`ScalarValueKind` 枚举定义了标量值的种类：

```cpp
enum class ScalarValueKind {
    Constant,    // 常量值（编译时已知）
    Symbolic     // 符号表达式（运行时确定）
};
```

### Value

`Value` 是所有值的基类。

#### Syntax

```text
%value_name_1 : type
%input_3 : tensor<[%b_1, 128], fp32>
%scale1_4 : fp32
```

在 IR 中，值显示为 `%{ssa_name} : {type}` 格式。

#### 数据结构

```cpp
class Value : public Object {
public:
    Value(ValueKind kind, TypePtr type, std::string name="");
    
    std::string GetSSAName() const;  // 获取 SSA 名称
    ValueKind GetValueKind() const;
    TypePtr GetType() const;
    DataType GetDataType() const;
    
    virtual void Print(std::ostream& os, int indent = 0) const = 0;
};
```

#### SSA 命名规则

- 如果名称为空：`%{id}`
- 如果名称非空：`%{name}_{id}`（如 `%input_3`、`%output_8`）

#### 约束

- 每个 Value 必须关联一个 Type 对象
- 所有值遵循 SSA 形式，即每个值只能被赋值一次

### Scalar

`Scalar` 表示标量值，支持常量值和符号值。

#### Syntax

```text
%const_0 = 0 : int64
%const_pi_13 = 3.14 : fp64
%scale1_4 : fp32
%batch_1 : int32
```

在 IR 中：
- 常量标量显示为 `%name = {value} : {type}` 格式
- 符号标量显示为 `%name : {type}` 格式

#### 数据结构

```cpp
class Scalar : public Value {
public:
    // 符号标量构造函数
    Scalar(DataType type, std::string name="", ScalarValueKind valueKind = ScalarValueKind::Symbolic);
    
    // 常量标量构造函数（类型自动推断）
    Scalar(bool value, std::string name="");
    Scalar(int value, std::string name="");      // DataType::INT32
    Scalar(int64_t value, std::string name="");  // DataType::INT64
    Scalar(double value, std::string name="");   // DataType::FP64
    Scalar(size_t value, std::string name="");   // DataType::UINT64
    
    ScalarValueKind GetScalarValueKind() const;
    ConstantType GetConstantValue() const;
    bool HasConstantValue() const;
    int64_t GetInt64Value() const;  // 获取常量值的 int64_t 表示
    
    void Print(std::ostream& os, int indent = 0) const override;
};
```

#### 约束

- 支持常量值和符号值
- 常量值使用 `std::variant<bool, int, int64_t, size_t, double>` 存储
- 打印规则：
  - 常量值：直接打印常量值（如 `2`、`3.14`、`10.5`）
  - 符号值：打印 SSA 名称（如 `%scale_66`）

### Tile

`Tile` 表示 Tile 值（固定形状的多维数组）。

#### Syntax

```text
%tile_0 : tile<[16, 32], fp32>
%tile_1 : tile<[M, N], fp16> {valid_shape=[%m, %n], strides=[32, 1], offset=0}
```

在 IR 中，Tile 值显示为 `%name : tile<[shape], dtype>` 格式，可能包含额外的属性信息。

#### 数据结构

```cpp
class Tile : public Value {
public:
    // 简化构造函数（只有形状和类型）
    Tile(std::vector<size_t> shape, DataType elementType, std::string name="");
    
    // 完整构造函数
    Tile(std::string name, std::vector<Scalar> validShapes, 
         std::vector<size_t> shape, std::vector<size_t> strides,
         Scalar startOffset, DataType elementType,
         std::shared_ptr<Memory> mem=nullptr);
    
    const std::vector<Scalar>& GetValidShape() const;
    const std::vector<size_t>& GetShape() const;  // 从 TileType 获取
    const std::vector<size_t>& GetStrides() const;
    Scalar GetStartOffset() const;
    const std::shared_ptr<Memory> GetMemory() const;
    
    void SetShape(const std::vector<size_t>& newShape);
    void SetStrides(const std::vector<size_t>& newStrides);
    void SetStartOffset(const Scalar newStartOffset);
    void SetMemory(const std::shared_ptr<Memory> newMem);
    
    void Print(std::ostream& os, int indent = 0) const override;
};
```

#### 约束

- 形状信息存储在 Type（TileType）中
- `validShapes_` 存储有效形状（Scalar 向量，支持符号维度）
- 支持步长（strides）、起始偏移（startOffset）、内存对象（Memory）
- 打印格式：`tile<[valid_shape], [tile_shape], dtype>`

### Tensor

`Tensor` 表示张量值（动态形状的多维数组）。

#### Syntax

```text
%input_3 : tensor<[%b_1, 128], fp32>
%output_8 : tensor<[%b_1, 128], fp32>
%tensor_0 : tensor<[%M, %N, %K], fp16>
```

在 IR 中，Tensor 值显示为 `%name : tensor<[shape], dtype>` 格式，其中 shape 可以包含符号维度。

#### 数据结构

```cpp
class Tensor : public Value {
public:
    // 从 Scalar 向量构造形状
    Tensor(const std::vector<Scalar>& shape, DataType type, 
           std::string name="", TileOpFormat format = TileOpFormat::TILEOP_ND);
    
    // 从整数向量构造形状（便捷构造函数）
    Tensor(DataType type, const std::vector<size_t>& shape,
           std::string name="", TileOpFormat format = TileOpFormat::TILEOP_ND);
    
    const std::vector<Scalar>& GetShape() const;
    TileOpFormat GetFormat() const;
    void SetFormat(TileOpFormat format);
    
    void Print(std::ostream& os, int indent) const override;
};
```

#### 约束

- 形状信息存储在 Value 对象中（`std::vector<Scalar>`），支持符号维度
- 形状不在 Type 中，因为形状在运行时确定
- 打印格式：`tensor<[shape], dtype>`（如 `tensor<[%b_1, 128], fp32>`）
