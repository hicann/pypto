# pypto.set\_pass\_options

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

修改Pass优化参数信息。其主要功能是在编译流程中，针对特定的优化策略和具体的Pass，动态修改其运行时参数配置，从而实现精细化的控制和调试。

## 函数原型

```python
set_pass_options(*,
                     pg_skip_partition: Optional[bool] = None,
                     pg_upper_bound: Optional[int] = None,
                     pg_lower_bound: Optional[int] = None,
                     pg_parallel_lower_bound: Optional[int] = None,
                     mg_vec_parallel_lb: Optional[int] = None,
                     vec_nbuffer_mode: Optional[int] = None,
                     vec_nbuffer_setting: Optional[Dict[int, int]] = None,
                     cube_l1_reuse_mode: Optional[int] = None,
                     cube_l1_reuse_setting: Optional[Dict[int, int]] = None,
                     cube_nbuffer_mode: Optional[int] = None,
                     cube_nbuffer_setting: Optional[Dict[int, int]] = None,
                     mg_copyin_upper_bound: Optional[int] = None,
                     sg_set_scope: Optional[int] = None,
                     )
```

## 参数说明

| 参数名                  | 输入/输出 | 说明                                                                 |
|-------------------------|-----------|----------------------------------------------------------------------|
| pg_skip_partition       | 输入      | 含义：是否跳过子图切分过程。 <br> 说明：当值为True时，将完整的计算图作为单一子图，不进行切分。当值为False时，进行子图切分。 <br> 类型：bool <br> 取值范围：{True, False} <br> 默认值：False <br> 影响Pass范围： GraphPartition |
| pg_upper_bound          | 输入      | 含义：合图参数，用于配置子图大小上界。 <br> 说明：当子图大小达到上界不允许与其他子图合并。 <br> 类型：int <br> 取值范围：0~2147483647 <br> 默认值：10000 <br> 影响Pass范围： GraphPartition |
| pg_lower_bound          | 输入      | 含义：合图参数，用于配置子图大小下界。 <br> 说明：当子图大小小于下界时尝试与其他子图合并。 <br> 类型：int <br> 取值范围：0~2147483647 <br> 默认值：512 <br> 影响Pass范围： GraphPartition |
| pg_parallel_lower_bound | 输入      | 含义：合图参数，用于配置相同结构子图的最小并行度。 <br> 说明：当某个相同结构的子图数小于该值时不做合并。 <br> 类型：int <br> 取值范围：0~2147483647 <br> 默认值：20 <br> 影响Pass范围： GraphPartition |
| sg_set_scope            | 输入      | 含义：手动控制合图参数。 <br> 说明：将operation赋予特定的scopeId，若相邻的operation具有相同的非-1的scopeId，则会被强制合并在一个子图之中，并且这个子图不会与其他子图合并。该参数仅对存在上下游连接通路的operation生效，例如operation A的输出作为operation B的输入即构成此类连接通路。 <br> 类型：int <br> 取值范围：-1~2147483647 <br> 默认值：-1 <br> 影响Pass范围：GraphPartition <br> 配置建议：1）视图类Operation与其对应的计算类Operation应配置相同的scopeId。2）Reshape Operation较为特殊，部分场景会单独成子图，手动控制合图行为可能失效。|
| mg_vec_parallel_lb      | 输入      | 含义：合图参数，用于配置相同结构AIV子图的最小并行度。 <br> 说明：当某个相同结构的子图数小于该值时不做合并。 <br> 类型：int <br> 取值范围：1~48 <br> 默认值：48 <br> 影响Pass范围：NBufferMerge |
| vec_nbuffer_mode        | 输入      | 含义：合图参数，用于配置相同结构AIV子图合并策略。 <br> 说明：该参数适用于结构相同的AIV子图合并，避免同一结构子图数过大并增大核内流水调度可能性。 <br> 类型：int <br> 取值：<br> 0：不使能相同结构子图间合并逻辑。<br> 1：使能相同结构子图间合并，合并逻辑为依据mg_vec_parallel_lb自适应计算每个结构的合并数。<br> 2：所有结构相同子图都按用户设置vec_nbuffer_setting来做子图间的合并。 <br> 默认值：1 <br> 影响Pass范围： NBufferMerge |
| vec_nbuffer_setting     | 输入      | 含义：合图参数，用于配置相同结构AIV子图的合并数量。 <br> 说明：该参数适用于结构相同的AIV子图合并。 <br> 类型：dict[int, int] <br> 使用条件：<br> vec_nbuffer_mode = 0/1, vec_nbuffer_setting 设置为nullMap。<br> vec_nbuffer_mode = 2，用户手动设置vec_nbuffer_setting 。<br> 默认值：nullMap <br> 影响Pass范围： NBufferMerge |
| cube_l1_reuse_mode | 输入 | 含义：合图参数，用于配置重复搬运同一GM数据的子图合并策略。<br> 说明：该参数适用于含有CUBE计算的子图，避免同一数据被重复搬运次数过多。<br> 类型：int <br> 取值：<br> 0：不使能存在重复搬运的子图间合并逻辑。<br> 1：使能存在重复搬运的子图间合并，合并逻辑为依据cube核数自适应计算每组AIC子图的合并数。<br> 用户设置cube_l1_reuse_setting时，不论cube_l1_reuse_mode的设置，都按照用户设置的cube_l1_reuse_setting来做子图间的合并。<br> 默认值：1 <br>影响Pass范围： L1ReuseMerge |
| cube_l1_reuse_setting | 输入 | 含义：合图参数，用于配置重复搬运同一GM数据的子图合并数量。<br> 说明：该参数适用于含有CUBE计算的子图合并。 <br> 类型： dict[int, int] <br> 取值：<br> {-1, N}：key为-1时，value值N表示存在重复搬运的AIC子图的合并数量默认值为N。 <br> 默认值：nullMap <br> 影响Pass范围：L1ReuseMerge |
| cube_nbuffer_mode | 输入 | 含义：合图参数，用于配置相同结构AIC子图合并策略。 <br> 说明：该参数适用于结构相同的AIC子图合并，避免同一结构子图数过大并增大核内流水调度可能性。<br> 类型：int <br> 取值：<br> 0：不使能相同结构子图间合并逻辑。<br> 1：使能相同结构子图间合并，合并逻辑为依据cube核数自适应计算每个结构的合并数。<br> 用户设置cube_nbuffer_setting时，不论cube_nbuffer_mode的取值，都按照用户设置的cube_nbuffer_setting来做子图间的合并。 <br> 默认值：0 <br> 影响Pass范围：<span> L1ReuseMerge</span> |
| cube_nbuffer_setting    | 输入      | 含义：合图参数，用于配置相同结构AIC子图的合并数量。 <br> 说明：该参数适用于结构相同的AIC子图合并。 <br> 类型：dict[int, int] <br> 取值：<br> {-1, N}：key为-1时，value值N表示结构相同的AIC子图的合并数量默认值为N <br> 默认值：nullMap <br> 影响Pass范围： L1ReuseMerge |
| mg_copyin_upper_bound   | 输入      | 含义：合图参数，用于配置合图大小。 <br> 说明：该参数控制子图内搬运数据总量上界。当子图内数据搬运量大于该值则不再合并。 <br> 类型：int <br> 取值：0~2147483647 <br> 默认值：1048576 <br> 影响Pass范围：L1ReuseMerge|

## 返回值说明

无。

## 约束说明

- 设置时机：不要求在图编译开始前调用，可以在任何时候进行设置。
- 类型安全：必须确保传入的value的类型与参数定义的类型完全一致，否则可能导致未定义行为或运行时错误。
- 作用范围：参数设置是局部的，只会影响当前jit或者loop内的编译过程，若未设置，则继承上层作用域。

## 调用示例

```python
   pypto.set_pass_options(pg_skip_partition=False,
                       pg_upper_bound=10000,
                       pg_lower_bound=512,
                       pg_parallel_lower_bound=24,
                       mg_vec_parallel_lb=48,
                       vec_nbuffer_mode=1,
                       vec_nbuffer_setting={},
                       cube_l1_reuse_mode=0,
                       cube_l1_reuse_setting={},
                       cube_nbuffer_mode=0,
                       cube_nbuffer_setting={},
                       mg_copyin_upper_bound=1024 * 1024)
```