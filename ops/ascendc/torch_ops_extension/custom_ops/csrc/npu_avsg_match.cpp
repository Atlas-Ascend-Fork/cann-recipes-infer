/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <torch/library.h>
#include "ops_common.h"

namespace custom {
using namespace at_npu::native;

// npu tensor max size
const int SIZE = 8;
const int MAPPING_ALIGN_ENTRIES = 4;

// 工具函数，推导输出shape
std::tuple<at::Tensor, at::Tensor, at::Tensor> construct_avsg_match_output_tensor(
    const at::Tensor& full_kv_actual_seq, const at::Tensor& selection_topk_indices)
{
    c10::SmallVector<int64_t, SIZE> countShape;
    for (int64_t dim = 0; dim + 1 < selection_topk_indices.dim(); ++dim) {
        countShape.push_back(selection_topk_indices.size(dim));
    }
    c10::SmallVector<int64_t, SIZE> mappingShape = countShape;
    const int64_t topk = selection_topk_indices.size(-1);
    mappingShape.push_back((topk + MAPPING_ALIGN_ENTRIES - 1) /
        MAPPING_ALIGN_ENTRIES * MAPPING_ALIGN_ENTRIES);
    mappingShape.push_back(2);
    return std::make_tuple(
        at::empty(full_kv_actual_seq.sizes(), selection_topk_indices.options()),
        at::empty(mappingShape, selection_topk_indices.options()),
        at::empty(countShape, selection_topk_indices.options()));
}

// step2, 为NPU设备实现前向接口
std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_avsg_match_npu(
    const at::Tensor& selection_kv_block_table, const at::Tensor& selection_kv_block_status,
    const at::Tensor& selection_topk_indices, const at::Tensor& full_kv_block_table,
    const at::Tensor& full_kv_actual_seq, const at::Tensor& full_q_actual_seq,
    int64_t selection_topk_block_size, int64_t kv_block_size)
{
    // construct the output tensor
    auto outputs = construct_avsg_match_output_tensor(
        full_kv_actual_seq, selection_topk_indices);
    EXEC_NPU_CMD_V1(aclnnAvsgMatch, selection_kv_block_table,
        selection_kv_block_status, selection_topk_indices, full_kv_block_table,
        full_kv_actual_seq, full_q_actual_seq, selection_topk_block_size, kv_block_size,
        std::get<0>(outputs), std::get<1>(outputs), std::get<2>(outputs));
    return outputs;
}

// step3, 为META设备实现前向接口
std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_avsg_match_meta(
    const at::Tensor& selection_kv_block_table, const at::Tensor&,
    const at::Tensor& selection_topk_indices, const at::Tensor&, const at::Tensor& full_kv_actual_seq,
    const at::Tensor&, int64_t, int64_t)
{
    // construct the output tensor
    return construct_avsg_match_output_tensor(
        full_kv_actual_seq, selection_topk_indices);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
npu_avsg_match_functional_fuc(
    const at::Tensor& selection_kv_block_table, const at::Tensor& selection_kv_block_status,
    const at::Tensor& selection_topk_indices, const at::Tensor& full_kv_block_table,
    const at::Tensor& full_kv_actual_seq, const at::Tensor& full_q_actual_seq,
    int64_t selection_topk_block_size, int64_t kv_block_size)
{
    at::Tensor selection_kv_block_table_inplace = selection_kv_block_table.clone();
    at::Tensor selection_kv_block_status_inplace = selection_kv_block_status.clone();
    at::Tensor selection_topk_indices_inplace = selection_topk_indices.clone();
    auto outputs = npu_avsg_match_npu(selection_kv_block_table_inplace,
        selection_kv_block_status_inplace,
        selection_topk_indices_inplace, full_kv_block_table, full_kv_actual_seq, full_q_actual_seq,
        selection_topk_block_size, kv_block_size);
    return std::make_tuple(std::get<0>(outputs), std::get<1>(outputs), std::get<2>(outputs),
        selection_kv_block_table_inplace, selection_kv_block_status_inplace, selection_topk_indices_inplace);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
npu_avsg_match_functional_meta(
    const at::Tensor& selection_kv_block_table, const at::Tensor& selection_kv_block_status,
    const at::Tensor& selection_topk_indices, const at::Tensor& full_kv_block_table,
    const at::Tensor& full_kv_actual_seq, const at::Tensor& full_q_actual_seq,
    int64_t selection_topk_block_size, int64_t kv_block_size)
{
    auto outputs = npu_avsg_match_meta(selection_kv_block_table,
        selection_kv_block_status, selection_topk_indices, full_kv_block_table,
        full_kv_actual_seq, full_q_actual_seq, selection_topk_block_size, kv_block_size);
    return std::make_tuple(std::get<0>(outputs), std::get<1>(outputs), std::get<2>(outputs),
        at::empty_like(selection_kv_block_table), at::empty_like(selection_kv_block_status),
        at::empty_like(selection_topk_indices));
}
}

// step4, 为NPU设备注册前向实现
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_avsg_match", &custom::npu_avsg_match_npu);
    m.impl("npu_avsg_match_functional", &custom::npu_avsg_match_functional_fuc);
}

// step5, 为META设备注册前向实现
TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_avsg_match", &custom::npu_avsg_match_meta);
    m.impl("npu_avsg_match_functional", &custom::npu_avsg_match_functional_meta);
}
