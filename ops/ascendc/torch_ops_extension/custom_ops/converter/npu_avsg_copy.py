# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Any

import torch
import torchair
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor


def _avsg_copy(selection_k_rope: Tensor, selection_kv_cache: Tensor,
               full_k_rope: Tensor, full_kv_cache: Tensor, full_q_actual_seq: Tensor,
               stage_two_copy_mapping: Tensor, stage_two_copy_count: Tensor):
    return torchair.ge.custom_op(
        "AvsgCopy",
        inputs={
            "selection_k_rope": selection_k_rope,
            "selection_kv_cache": selection_kv_cache,
            "full_k_rope": full_k_rope,
            "full_kv_cache": full_kv_cache,
            "full_q_actual_seq": full_q_actual_seq,
            "stage_two_copy_mapping": stage_two_copy_mapping,
            "stage_two_copy_count": stage_two_copy_count,
        },
        outputs=["selection_k_rope", "selection_kv_cache"])


@register_fx_node_ge_converter(torch.ops.custom.npu_avsg_copy_functional.default)
def convert_npu_avsg_copy_functional(
    selection_k_rope: Tensor, selection_kv_cache: Tensor, full_k_rope: Tensor,
    full_kv_cache: Tensor, full_q_actual_seq: Tensor, stage_two_copy_mapping: Tensor,
    stage_two_copy_count: Tensor, *, meta_outputs: Any = None):
    return _avsg_copy(selection_k_rope, selection_kv_cache, full_k_rope, full_kv_cache,
                      full_q_actual_seq, stage_two_copy_mapping, stage_two_copy_count)


@register_fx_node_ge_converter(torch.ops.custom.npu_avsg_copy.default)
def convert_npu_avsg_copy(
    selection_k_rope: Tensor, selection_kv_cache: Tensor, full_k_rope: Tensor,
    full_kv_cache: Tensor, full_q_actual_seq: Tensor, stage_two_copy_mapping: Tensor,
    stage_two_copy_count: Tensor, *, meta_outputs: Any = None):
    _avsg_copy(selection_k_rope, selection_kv_cache, full_k_rope, full_kv_cache,
               full_q_actual_seq, stage_two_copy_mapping, stage_two_copy_count)
