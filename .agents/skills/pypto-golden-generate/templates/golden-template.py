#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

"""PyPTO {op} golden reference implementation.

模板说明：
  - 所有 {op} 占位符需替换为实际算子名称。
  - golden 是基于 torch + torch_npu 的 NPU 参考实现，计算在 NPU 上执行。
  - torch_npu 未安装时直接报错引导安装；仅无 NPU 硬件时回退 CPU。
  - 导出函数 {op}_golden() 供 test_{op}.py 调用。
  - 参考 examples/ 中的 golden 函数风格（activation、layernorm 等）。
"""

import os
import torch

_DEVICE = None
_HAS_NPU = False

try:
    import torch_npu
    _HAS_NPU = torch.npu.is_available() and torch.npu.device_count() > 0
except ImportError:
    raise ImportError(
        "torch_npu is not installed. Please install it first:\n"
        "  pip install torch_npu\n"
        "Or use the pypto-environment-setup skill to set up the full NPU environment."
    )

def _get_device(device_id: int | None = None) -> torch.device:
    global _DEVICE
    if _DEVICE is None:
        if not _HAS_NPU:
            _DEVICE = torch.device("cpu")
        else:
            if device_id is None:
                env_id = os.environ.get("ASCEND_DEVICE_ID")
                if env_id is not None:
                    device_id = int(env_id)
            _DEVICE = torch.device(f"npu:{device_id}") if device_id is not None else torch.device("npu")
    return _DEVICE


# ─────────────────────────────────────────────
# Golden 参考实现（NPU torch）
# ─────────────────────────────────────────────

def {op}_golden(x: torch.Tensor) -> torch.Tensor:
    """PyTorch 参考实现。

    根据算子规格中的数学公式实现。
    仅使用 torch 标准操作，不依赖 pypto。

    Args:
        x: 输入 tensor。
           根据实际算子需求调整参数列表（可多输入、可带 gamma/beta/eps 等参数）。

    Returns:
        计算结果 tensor。
        根据实际算子需求调整返回值（可多输出、可返回 tuple）。
    """
    device = _get_device()
    x = x.to(device)
    # TODO: 替换为实际 golden 逻辑
    # 示例（SiLU）:  return x * torch.sigmoid(x)
    # 示例（LayerNorm）:
    #   mean = x.mean(dim=-1, keepdim=True)
    #   var = x.var(dim=-1, keepdim=True, unbiased=False)
    #   normalized = (x - mean) / torch.sqrt(var + eps)
    #   return normalized * gamma + beta
    return x


# ==========================================
# 验证
# ==========================================

def _validate():
    """自动生成的验证函数 - 运行时动态生成验证报告"""

    device = _get_device()
    print("=" * 60)
    print("{op}_golden 验证报告")
    print("=" * 60)
    print(f"Device: {{device}}")

    # -- 1. 典型 case 验证（来自算子规格中的典型配置）--
    print("\n[典型 case 验证]")
    # TODO: 按算子规格中的典型配置生成验证
    #   注意: torch.randn 等创建 tensor 时必须带 device=device，使输入直接位于目标设备

    # -- 2. 泛化 case 验证（来自算子规格中的动态轴范围）--
    print("\n[泛化 case 验证]")
    # TODO: 按动态轴采样范围验证

    # -- 3. 值域检查（从公式推导）--
    print("\n[值域检查]")
    # TODO: 验证输出值域

    # -- 4. 数值稳定性检查 --
    print("\n[数值稳定性检查]")
    # TODO: 大值、小值、零值等极端输入

    # -- 5. API 对比（如适用）--
    print("\n[API 对比]")
    # TODO: 与 PyTorch 等价 API 对比
    #   注意: 对比双方 tensor 必须在相同 device 上

    print("\n" + "=" * 60)
    print("验证完成")
    print("=" * 60)


if __name__ == "__main__":
    _validate()
