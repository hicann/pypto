# pypto.Tensor构造函数

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

创建Tensor对象。Tensor创建时为未初始化的随机值。

## 函数原型

```python
__init__(self, 
         shape=None, 
         dtype: Union[DataType, None] = None,
         name: str = "", 
         format: TileOpFormat = TileOpFormat.TILEOP_ND,
         data_ptr: Optional[int] = None, 
         device=None, 
         ori_shape=None
)
```

## 参数说明


| 参数名     | 输入/输出 | 说明                                                                 |
|------------|-----------|----------------------------------------------------------------------|
| shape      | 输入      | Tensor的形状，可以是以下类型：<br> - None：创建空Tensor<br> - List[int]：整数列表，指定各维度的大小<br> - List[Union[int, SymbolicScalar]]：包含整数或符号标量的列表，用于动态形状 |
| dtype      | 输入      | Tensor的数据类型。 |
| name       | 输入      | Tensor的名称。 |
| format     | 输入      | Tensor的格式，可选值包括：<br> - TileOpFormat.TILEOP_ND(默认)<br> - TileOpFormat.TILEOP_NZ |
| data_ptr   | 输入      | 数据指针，用于指定外部数据的内存地址，默认为None |
| device     | 输入      | 设备信息，默认为None |
| ori_shape  | 输入      | 原始形状，用于保存Tensor的原始形状信息，默认为None |

## 返回值说明

Tensor对象。

## 约束说明

无。

## 调用示例

```python
# 创建空Tensor 
empty_tensor = pypto.Tensor()  

# 创建指定形状和数据类型的Tensor 
tensor1 = pypto.Tensor(shape=(4, 4), dtype=pypto.DT_FP32) 
tensor2 = pypto.Tensor(shape=[8, 16, 32], dtype=pypto.DT_INT32)  

# 创建带名称的Tensor 
named_tensor = pypto.Tensor(shape=(4, 4),      
                            dtype=pypto.DT_FP32,      
                            name="input_tensor" )  

# 创建指定格式的Tensor 
sparse_tensor = pypto.Tensor(shape=(4, 32),      
                             dtype=pypto.DT_FP32,       
                             format=pypto.TileOpFormat.TILEOP_NZ )  

# 创建动态Shape的Tensor（使用符号化标量） 
dynamic_shape = [pypto.SymbolicScalar("N"), 4, 8] 
dynamic_tensor = pypto.Tensor(shape=dynamic_shape,      
                              dtype=pypto.DT_FP32 )  

# 使用 pypto.tensor 便捷函数创建（推荐方式） 
tensor3 = pypto.tensor((4, 4), pypto.DT_FP32) 
tensor4 = pypto.tensor((4, 4), pypto.DT_FP32, name="my_tensor")
```

