/** Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved. */
#include <torch/library.h>
#include "ops_common.h"

namespace custom {
void npu_avsg_copy_npu(const at::Tensor& selection_k_rope, const at::Tensor& selection_kv_cache,
    const at::Tensor& full_k_rope, const at::Tensor& full_kv_cache,
    const at::Tensor& full_q_actual_seq, const at::Tensor& stage_two_copy_mapping,
    const at::Tensor& stage_two_copy_count)
{
    EXEC_NPU_CMD_V1(aclnnAvsgCopy, selection_k_rope, selection_kv_cache, full_k_rope,
        full_kv_cache, full_q_actual_seq, stage_two_copy_mapping, stage_two_copy_count);
}

void npu_avsg_copy_meta(const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&) {}

std::tuple<at::Tensor, at::Tensor> npu_avsg_copy_functional_npu(
    const at::Tensor& selection_k_rope, const at::Tensor& selection_kv_cache,
    const at::Tensor& full_k_rope, const at::Tensor& full_kv_cache,
    const at::Tensor& full_q_actual_seq, const at::Tensor& stage_two_copy_mapping,
    const at::Tensor& stage_two_copy_count)
{
    at::Tensor kRope = selection_k_rope.clone();
    at::Tensor kvCache = selection_kv_cache.clone();
    npu_avsg_copy_npu(kRope, kvCache, full_k_rope, full_kv_cache, full_q_actual_seq,
        stage_two_copy_mapping, stage_two_copy_count);
    return std::make_tuple(kRope, kvCache);
}

std::tuple<at::Tensor, at::Tensor> npu_avsg_copy_functional_meta(
    const at::Tensor& selection_k_rope, const at::Tensor& selection_kv_cache,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&)
{
    return std::make_tuple(at::empty_like(selection_k_rope), at::empty_like(selection_kv_cache));
}
} // namespace custom

TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_avsg_copy", &custom::npu_avsg_copy_npu);
    m.impl("npu_avsg_copy_functional", &custom::npu_avsg_copy_functional_npu);
}

TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_avsg_copy", &custom::npu_avsg_copy_meta);
    m.impl("npu_avsg_copy_functional", &custom::npu_avsg_copy_functional_meta);
}
