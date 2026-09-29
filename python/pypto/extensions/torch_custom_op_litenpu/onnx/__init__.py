# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""ONNX export for a pypto op: what the author declares, and everything applied to it.

``spec`` holds ``OnnxSymbolicSpec``, the passive config an author declares. ``export`` applies it:
it qualifies the op type, encodes the node meta and the declared attrs into ``g.op`` kwargs,
registers the ``torch.onnx`` symbolic, and keeps the recorded ai.onnx opset floor
(``recorded_onnx_opset_floor``) that an export driver reads back. ``node_reader`` reads pypto's node
meta back off ONNX nodes. All three are imported directly by their consumers.
"""
