# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
import torch_npu
import torchair
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, \
    BOOL, Support
from torchair._ge_concrete_graph.utils import dtype_promote
from torchair.ge import attr


# 为自定义算子注册converter，用于torch.compile 场景成图


# 注意： meta_outputs形参名为固定写法，若写错会影响ge节点的输出dtype与shape推导
@register_fx_node_ge_converter(torch.ops.custom.npu_avsg_match_functional.default)
def convert_npu_avsg_match_functional(
    selection_kv_block_table: Tensor,
    selection_kv_block_status: Tensor,
    selection_topk_indices: Tensor,
    full_kv_block_table: Tensor,
    full_kv_actual_seq: Tensor,
    full_q_actual_seq: Tensor,
    *, selection_topk_block_size: int = 64, kv_block_size: int = 128,
    meta_outputs: Any = None):
    out0, out1, out2, out3, out4, out5 = torchair.ge.custom_op(
        "AvsgMatch",
        inputs={"selection_kv_block_table": selection_kv_block_table,
                "selection_kv_block_status": selection_kv_block_status,
                "selection_topk_indices": selection_topk_indices,
                "full_kv_block_table": full_kv_block_table,
                "full_kv_actual_seq": full_kv_actual_seq,
                "full_q_actual_seq": full_q_actual_seq,
                },
        attrs={"selection_topk_block_size": attr.Int(selection_topk_block_size),
               "kv_block_size": attr.Int(kv_block_size)},
        outputs=['selection_kv_block_table', 'selection_kv_block_status',
                 'selection_kv_actual_seq', 'stage_two_copy_mapping', 'stage_two_copy_count',
                 'selection_topk_indices']
    )

    return out2, out3, out4, out0, out1, out5


# 注意： meta_outputs形参名为固定写法，若写错会影响ge节点的输出dtype与shape推导
@register_fx_node_ge_converter(torch.ops.custom.npu_avsg_match.default)
def convert_npu_avsg_match(
    selection_kv_block_table: Tensor,
    selection_kv_block_status: Tensor,
    selection_topk_indices: Tensor,
    full_kv_block_table: Tensor,
    full_kv_actual_seq: Tensor,
    full_q_actual_seq: Tensor,
    *, selection_topk_block_size: int = 64, kv_block_size: int = 128,
    meta_outputs: Any = None):
    out0, out1, out2, out3, out4, out5 = torchair.ge.custom_op(
        "AvsgMatch",
        inputs={"selection_kv_block_table": selection_kv_block_table,
                "selection_kv_block_status": selection_kv_block_status,
                "selection_topk_indices": selection_topk_indices,
                "full_kv_block_table": full_kv_block_table,
                "full_kv_actual_seq": full_kv_actual_seq,
                "full_q_actual_seq": full_q_actual_seq,
                },
        attrs={"selection_topk_block_size": attr.Int(selection_topk_block_size),
               "kv_block_size": attr.Int(kv_block_size)},
        outputs=['selection_kv_block_table', 'selection_kv_block_status',
                 'selection_kv_actual_seq', 'stage_two_copy_mapping', 'stage_two_copy_count',
                 'selection_topk_indices']
    )

    return out2, out3, out4
