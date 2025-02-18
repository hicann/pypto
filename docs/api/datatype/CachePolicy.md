# CachePolicy

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

CachePolicy定义了Tensor的缓存策略，用于控制数据在各级缓存中的行为，优化内存访问性能和减少内存带宽消耗。

## 原型定义

```python
class CachePolicy(enum.Enum):
     PREFETCH = ...        # 预取策略，提前将数据加载到缓存中，减少访问延迟
     NONE_CACHEABLE = ...  # 不可缓存策略，数据不存储在缓存中，直接访问主存
```

