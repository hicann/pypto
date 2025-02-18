# pypto.cond

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

定义一个if 条件操作，实现python中的if功能。

## 函数原型

```python
cond(scalar: SymInt)
```

## 参数说明


| 参数名 | 输入/输出 | 说明                                                                 |
|--------|-----------|----------------------------------------------------------------------|
| scalar | 输入      | 条件表达式，可以是整数或SymbolicScalar（符号标量），用于判断条件是否为真 |

## 返回值说明

pypto\_impl.RecordIfBranch: 返回一个条件分支对象，用于 Python 的 if 语句

## 约束说明

-   必须与 Python 的 if、elif、else 语句配合使用
-   条件表达式会被记录到计算图中
-   支持嵌套的条件语句

## 调用示例

```python
for s2_idx in pypto.loop(0, 10, 1, power_of_2(max_unroll_times), name="LOOP_L0_bIdx_mla_prolog", idx_name="b_idx"):
    if pypto.cond(pypto.is_loop_end(s2_idx, bn_per_batch)):
        ...
```

