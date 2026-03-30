# pypto.distributed.my_symbolic_pe

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 推理系列产品 |    √     |
| Atlas A2 推理系列产品 |    √     |

## 功能说明

用于获取当前 pe。

## 函数原型

```python
my_symbolic_pe(group_name: str) -> SymbolicScalar
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| group_name   | 输入      |  集合通信操作所在通信域的名字，字符串长度: 1~128。<br> 支持的类型为：str 类型。|

## 返回值说明

返回当前 pe。

## 约束说明

无

## 调用示例

```python
my_pe = pypto.distributed.my_symbolic_pe(group_name="tp")
```
