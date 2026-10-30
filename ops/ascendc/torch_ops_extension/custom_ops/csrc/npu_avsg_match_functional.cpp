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
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/EmptyTensor.h>
#include <ATen/MemoryOverlap.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/npu_log.h"
#include <c10/util/Logging.h>
#include <ATen/Operators.h>
#include <ATen/NativeFunctions.h>
#include "torch_npu/csrc/core/npu/NPUException.h"

namespace at_npu {
namespace functionalization {

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_avsg_match_functional_npu(
    c10::DispatchKeySet dispatchKeySet, const at::Tensor& selection_kv_block_table,
    const at::Tensor& selection_kv_block_status, const at::Tensor& selection_topk_indices,
    const at::Tensor& full_kv_block_table, const at::Tensor& full_kv_actual_seq,
    const at::Tensor& full_q_actual_seq, int64_t selection_topk_block_size, int64_t kv_block_size)
{
    // 1.对所有入参中的tensor调用sync()
    at::functionalization::impl::sync(selection_kv_block_table);
    at::functionalization::impl::sync(selection_kv_block_status);
    at::functionalization::impl::sync(selection_topk_indices);
    at::functionalization::impl::sync(full_kv_block_table);
    at::functionalization::impl::sync(full_kv_actual_seq);
    at::functionalization::impl::sync(full_q_actual_seq);

    // 2.调用_from_functional_tensor解包所有需要调用非原地算子的入参
    auto selection_kv_block_table_unwarp = at::functionalization::impl::from_functional_tensor(selection_kv_block_table);
    auto selection_kv_block_status_unwarp = at::functionalization::impl::from_functional_tensor(selection_kv_block_status);
    auto selection_topk_indices_unwarp = at::functionalization::impl::from_functional_tensor(selection_topk_indices);
    auto full_kv_block_table_unwarp = at::functionalization::impl::from_functional_tensor(full_kv_block_table);
    auto full_kv_actual_seq_unwarp = at::functionalization::impl::from_functional_tensor(full_kv_actual_seq);
    auto full_q_actual_seq_unwarp = at::functionalization::impl::from_functional_tensor(full_q_actual_seq);
    // Grab the dispatcher entry corresponding out-of-place variant of the mutable op
    static auto gather_selection_handle = c10::Dispatcher::singleton()
        // specify namespace::op_name, op_overload_name
        // 这里的custom并不是c++的namespace,而是注册算子时的python模板名
        .findSchemaOrThrow("custom::npu_avsg_match_functional", "")
        // specify the C++ schema of the corresponding out-of-place op
        .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
            const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
            const at::Tensor&, const at::Tensor&, int64_t, int64_t)>();
    // Redispath to the out-of-place op when mutable op is called by user
    at::Tensor tmp_input1;
    at::Tensor tmp_input2;
    at::Tensor tmp_input3;
    at::Tensor tmp_input4;
    at::Tensor tmp_input5;
    at::Tensor tmp_input6;
    {
        at::AutoDispatchSkipFunctionalize guard;
        // 3.调用非原地算子，得到临时输出
        std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> gather_result =
            gather_selection_handle.call(selection_kv_block_table_unwarp,
            selection_kv_block_status_unwarp, selection_topk_indices_unwarp, full_kv_block_table_unwarp,
            full_kv_actual_seq_unwarp, full_q_actual_seq_unwarp, selection_topk_block_size, kv_block_size);
        tmp_input1 = std::get<0>(gather_result);
        tmp_input2 = std::get<1>(gather_result);
        tmp_input3 = std::get<2>(gather_result);
        tmp_input4 = std::get<3>(gather_result);
        tmp_input5 = std::get<4>(gather_result);
        tmp_input6 = std::get<5>(gather_result);
    }

    // 4.将临时输出拷贝给输入，达到原地修改的目的
    at::functionalization::impl::replace_(selection_kv_block_table, tmp_input4);
    at::functionalization::impl::replace_(selection_kv_block_status, tmp_input5);
    at::functionalization::impl::replace_(selection_topk_indices, tmp_input6);
    at::functionalization::impl::commit_update(selection_kv_block_table);
    at::functionalization::impl::commit_update(selection_kv_block_status);
    at::functionalization::impl::commit_update(selection_topk_indices);

    // 5.再次调用sync()
    at::functionalization::impl::sync(selection_kv_block_table);
    at::functionalization::impl::sync(selection_kv_block_status);
    at::functionalization::impl::sync(selection_topk_indices);
    at::functionalization::impl::sync(full_kv_block_table);
    at::functionalization::impl::sync(full_kv_actual_seq);
    at::functionalization::impl::sync(full_q_actual_seq);
    return std::make_tuple(tmp_input1, tmp_input2, tmp_input3);
}

} // namespace functionalization

namespace {
// 注册Functionalize
TORCH_LIBRARY_IMPL(custom, Functionalize, m) {
    m.impl("npu_avsg_match", TORCH_FN(functionalization::npu_avsg_match_functional_npu));
}
} // namespace

} // ns at_npu
