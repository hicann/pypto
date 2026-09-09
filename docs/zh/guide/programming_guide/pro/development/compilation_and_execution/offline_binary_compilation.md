# 离线二进制编译

PyPTO Pro Kernel可以接入算子工程的离线构建流程，生成随算子包发布的AI Core二进制。离线编译需要同时准备Device侧Kernel、Host侧Tiling和算子工程构建配置。

主要步骤如下：

1. 准备算子定义、InferShape、Host侧Tiling和调用接口。
2. 在`op_kernel`目录中实现PyPTO Pro Kernel。
3. 在`op_host/CMakeLists.txt`中启用PyPTO Pro Kernel。
4. 在Host侧填写TilingData，设置TilingKey、BlockDim和Workspace。
5. 运行算子工程构建脚本，生成算子安装包。

## 准备算子工程

与PyPTO Pro离线编译直接相关的目录如下，实际工程中的上级分类目录以算子工程为准：

```text
<operator_project>
├── build.sh
├── <op_class>
│   └── ${op_name}
│       ├── examples
│       │   └── test_aclnn_${op_name}.cpp
│       ├── op_host
│       │   ├── ${op_name}_def.cpp
│       │   ├── ${op_name}_infershape.cpp
│       │   ├── ${op_name}_tiling.cpp
│       │   └── CMakeLists.txt
│       ├── op_kernel
│       │   └── ${op_file}.py
│       └── op_graph
└── CMakeLists.txt
```

| 文件或目录 | 作用 |
| --- | --- |
| `${op_name}_def.cpp` | 定义算子的输入、输出、属性和数据类型。 |
| `${op_name}_infershape.cpp` | 推导输出shape和数据类型。 |
| `${op_name}_tiling.cpp` | 计算TilingData、TilingKey、BlockDim和Workspace。 |
| `${op_file}.py` | 实现Device侧Kernel。 |
| `test_aclnn_${op_name}.cpp` | 通过aclnn接口调用并验证算子。 |
| `op_graph` | 保存图模式所需的Graph Infer和算子原型注册等内容；仅使用aclnn时不需要。 |

## 使用PyPTO Pro实现Kernel

Kernel文件位于`op_kernel/${op_file}.py`，其中`${op_file}`不包含`.py`后缀，并与CMake中的名称一致。

离线交付的Kernel需要满足以下要求：

- 每个`op_kernel/${op_file}.py`中恰好定义一个使用`@pypto_pro.language.jit`声明的Kernel。
- 定义TilingKey，并通过`tiling_key`参数绑定到Kernel。
- 使用Python `dataclass`定义TilingData；Kernel中恰好包含一个该类型的参数，并且该参数必须位于参数列表末尾。
- Kernel函数名与算子Kernel入口名称一致。
- 业务输入、输出的参数名称与算子原型一致，形参顺序与Host侧下发顺序一致。
- Workspace位于所有业务输入、输出之后，并位于TilingData之前，即参数结尾为`workspace, tiling`。
- Kernel需要获取某个输入或输出的数据类型时，通过`datatype`声明对应参数。

下面展示`add_example`的Kernel结构，计算过程省略：

```python
from dataclasses import dataclass

import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField


@dataclass
class AddExampleTilingData:
    total_length: int
    tile_num: int


class AddExampleTilingKey:
    sch_mode = TilingKeyField(bits=1, values=[0, 1])


@pl.jit(
    tiling_key=AddExampleTilingKey,
    datatype={
        "x": "data_dtype",
        "y": "data_dtype",
        "z": "data_dtype",
    },
)
def add_example(
    x: pl.Ptr[pl.DT_UINT8],
    y: pl.Ptr[pl.DT_UINT8],
    z: pl.Ptr[pl.DT_UINT8],
    workspace: pl.Ptr[pl.DT_UINT8],
    tiling: AddExampleTilingData,
):
    x_tensor = pl.make_tensor(x, [tiling.total_length], [1], dtype=data_dtype)
    y_tensor = pl.make_tensor(y, [tiling.total_length], [1], dtype=data_dtype)
    z_tensor = pl.make_tensor(z, [tiling.total_length], [1], dtype=data_dtype)
    ...
```

`datatype`字典的key是Kernel参数名，value是Kernel中使用的dtype变量名。多个参数映射到同一变量时，实际数据类型必须一致。只声明Kernel内部需要读取数据类型的参数。

TilingData和TilingKey的字段规则参考[Tiling结果传输](../tiling/tiling_result_transfer.md)。Kernel参数和执行域的定义参考[Kernel核函数创建](../kernel_function.md)。

## 配置CMakeLists.txt

在`op_host/CMakeLists.txt`中调用`enable_pypto_kernel(<op_file>)`，并将该调用放在`add_modules_sources`或`add_modules_sources_with_soc`之前：

```cmake
enable_pypto_kernel(add_example)

add_modules_sources(
    OPTYPE add_example
    ACLNNTYPE aclnn
)
```

使用`add_modules_sources_with_soc`时配置方式相同：

```cmake
enable_pypto_kernel(add_example)

add_modules_sources_with_soc(
    OPTYPE add_example
    ACLNNTYPE aclnn
)
```

`add_example`必须与`op_kernel/add_example.py`的文件名一致。CMake配置阶段加载Kernel文件，生成Host和Device共同使用的TilingData与TilingKey头文件；后续构建阶段根据datatype和TilingKey生成Kernel实例。

## 实现Host侧Tiling

构建流程根据Kernel侧Python `dataclass`生成同名C++ Tiling类。Host侧直接填写生成的类型，不需要再次声明结构体：

```cpp
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    AddExampleTilingData *tiling =
        context->GetTilingData<AddExampleTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    tiling->total_length = total_length;
    tiling->tile_num = tile_num;

    uint64_t tiling_key = GET_TPL_TILING_KEY(0);
    context->SetTilingKey(tiling_key);
    context->SetBlockDim(block_dim);

    size_t *workspace_size = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace_size);
    workspace_size[0] = user_workspace_bytes;
    return ge::GRAPH_SUCCESS;
}
```

Host侧Tiling需要保证：

- TilingData字段名称、类型和顺序与Kernel侧定义一致。
- `GET_TPL_TILING_KEY(...)`的实参按照TilingKey字段定义顺序排列，并且属于合法组合。
- `context->SetBlockDim()`与Kernel的多核切分方式一致。
- Workspace大小覆盖Kernel使用的用户Workspace和相关接口需要的系统Workspace。

`GET_TPL_TILING_KEY(...)`接收各字段的实际候选值，并根据字段顺序和候选下标生成64-bit Key。`context->SetTilingKey()`接收打包后的Key，不能直接传入某个字段未经编码的值。

BlockDim的含义和计算方式参考[多核Tiling切分](../tiling/multi_core_tiling.md#在启动时设置逻辑block数block_dim)。需要系统Workspace时，通过相应平台接口查询所需大小后与用户Workspace相加，不要写死固定值。

## 编译算子二进制

编译前配置CANN和编译工具链环境，并确保构建使用的Python环境能够导入与当前源码配套的`pypto_pro`。在算子工程根目录执行：

```bash
bash build.sh --pkg --soc=${soc_version} --ops=add_example
```

编译多个算子时，使用英文逗号分隔名称：

```bash
bash build.sh --pkg --soc=${soc_version} --ops=add_example,other_op
```

指定自定义算子包名称时增加`--vendor_name`：

```bash
bash build.sh --pkg --soc=${soc_version} \
    --vendor_name=${vendor_name} \
    --ops=add_example
```

未指定`--vendor_name`时，自定义算子包名称默认为`custom`。

构建完成后，算子安装包生成在工程根目录的`build_out`中。安装算子包后可以通过aclnn调用；图模式还需要提供`op_graph`中的相关交付件。
