# TILE_FWK_DEVICE_ID

## 功能描述
指定PyPTO算子执行时使用的NPU设备卡号。框架和测试用例通过该变量确定目标设备。

- 类型：整数
- 取值范围：`0` ~ `N-1`（N为可用NPU数量）

## 配置示例
```bash
export TILE_FWK_DEVICE_ID=0
```

## 使用约束
- 指定的设备卡号必须为可用状态，可通过`npu-smi info`确认。
- 多进程场景下应避免多个进程使用同一设备卡号。

## 支持的型号
- Ascend 950PR
- Atlas A2训练系列产品 / Atlas A2推理系列产品
- Atlas A3训练系列产品 / Atlas A3推理系列产品
