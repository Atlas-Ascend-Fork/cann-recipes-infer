# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

def avsg_offload_enabled(model_config: dict | None = None) -> bool:
    if model_config is None:
        return False
    custom_params = model_config.get("custom_params", {})
    return bool(custom_params.get("enable_offload", False)) and custom_params.get("offload_strategy") == "avsg"

def run_avsg_offload(
    *,
    layer_idx: int,
    kv_block_size: int,
    topk_indices: torch.Tensor,
    actual_seq_qlen: torch.Tensor,
    full_kv_actual_seq: torch.Tensor,
    full_kv_cache: torch.Tensor,
    full_k_rope: torch.Tensor,
    block_table: torch.Tensor,
    selection_kv_cache: torch.Tensor,
    selection_k_rope: torch.Tensor,
    selection_kv_block_table: torch.Tensor,
    selection_kv_block_status: torch.Tensor,
    offload_cache,
):

    _topk_id = topk_indices + 0

    if offload_cache.is_dsa_shared_group_owner(layer_idx):
        topk_id, block_table_dsa, stage_two_copy_mapping, stage_two_copy_count = _process_avsg_match(
            _topk_id, block_table, actual_seq_qlen, full_kv_actual_seq,
            selection_kv_block_table, selection_kv_block_status, kv_block_size)

        offload_cache.stage_two_copy_mapping = stage_two_copy_mapping
        offload_cache.stage_two_copy_count = stage_two_copy_count
        offload_cache.topk_id = topk_id
        offload_cache.block_table_dsa = block_table_dsa

    kv_dsa, key_rope_dsa = _process_avsg_copy(full_kv_cache, full_k_rope, selection_kv_cache, selection_k_rope,
        offload_cache.stage_two_copy_mapping, offload_cache.stage_two_copy_count, actual_seq_qlen)

    return kv_dsa, key_rope_dsa, offload_cache.topk_id, offload_cache.block_table_dsa


def _process_avsg_match(topk_id,
        block_table, actual_seq_qlen, kv_lens,
        selection_kv_block_table, selection_kv_block_status, kv_block_size):

    selection_kv_block_table_npu = selection_kv_block_table
    selection_kv_block_status_npu = selection_kv_block_status

    # TND
    full_kv_block_table_npu = block_table
    full_kv_actual_seq_npu = kv_lens
    full_q_actual_seq_npu = actual_seq_qlen

    selection_kwargs = {
        "selection_kv_block_table": selection_kv_block_table_npu,
        "selection_kv_block_status": selection_kv_block_status_npu,
        "selection_topk_indices": topk_id,
        "full_kv_block_table": full_kv_block_table_npu,
        "full_kv_actual_seq": full_kv_actual_seq_npu,
        "full_q_actual_seq": full_q_actual_seq_npu,
        "selection_topk_block_size": 1,
        "kv_block_size": kv_block_size,
    }

    reused_num, stage_two_copy_mapping, stage_two_copy_count= torch.ops.custom.npu_avsg_match(**selection_kwargs)

    return topk_id, selection_kv_block_table_npu, stage_two_copy_mapping, stage_two_copy_count

def _process_avsg_copy(kv_cache, k_rope, selection_kv_cache, selection_k_rope, stage_two_copy_mapping, stage_two_copy_count, cu_q_lens):

    full_kv_cache_npu = kv_cache.squeeze(-2)
    full_k_rope_npu = k_rope.squeeze(-2)

    torch.ops.custom.npu_avsg_copy(
        selection_k_rope, selection_kv_cache, full_k_rope_npu,
        full_kv_cache_npu, cu_q_lens, stage_two_copy_mapping,
        stage_two_copy_count)

    return selection_kv_cache, selection_k_rope
