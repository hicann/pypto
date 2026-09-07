#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Pipeline configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for preload pipeline transformation.

    Args:
        preload: How many iterations ahead an upstream stage runs. A larger value hides
                 more of the transfer/compute latency and usually performs better, at the
                 cost of a longer fill and drain — tune it per kernel.
                 Concretely it is the delay step between two consecutive stages on the
                 SAME core (see _compute_delays); the ctx ring-buffer depth follows from
                 the resulting delays (max_delay + 1), not from preload directly.

                 Zero pulls no stage ahead of any other, which leaves the serial loop with
                 cross-core sync inserted around each stage and nothing else changed. Start
                 there to confirm the serial kernel is correct, then raise it.

                 Must not be negative: a stage cannot run a negative number of iterations
                 ahead, and the delays that follow would index the ctx ring from before the
                 loop began.
    """

    preload: int = 2

    def __post_init__(self) -> None:
        if self.preload < 0:
            raise ValueError(
                f"pipeline: preload must be >= 0, got {self.preload}. It is how many "
                f"iterations ahead a stage runs, so a negative value has no meaning; use "
                f"preload=0 to keep the serial loop and only insert cross-core sync."
            )
