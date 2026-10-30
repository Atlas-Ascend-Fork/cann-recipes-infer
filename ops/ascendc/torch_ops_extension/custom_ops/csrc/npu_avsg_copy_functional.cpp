/** Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved. */
#include <torch/library.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/core/LegacyTypeDispatch.h>

namespace at_npu {
namespace functionalization {
void npu_avsg_copy_functionalize(c10::DispatchKeySet,
    const at::Tensor& selection_k_rope, const at::Tensor& selection_kv_cache,
    const at::Tensor& full_k_rope, const at::Tensor& full_kv_cache,
    const at::Tensor& full_q_actual_seq, const at::Tensor& stage_two_copy_mapping,
    const at::Tensor& stage_two_copy_count)
{
    at::functionalization::impl::sync(selection_k_rope);
    at::functionalization::impl::sync(selection_kv_cache);
    at::functionalization::impl::sync(full_k_rope);
    at::functionalization::impl::sync(full_kv_cache);
    at::functionalization::impl::sync(full_q_actual_seq);
    auto kRope = at::functionalization::impl::from_functional_tensor(selection_k_rope);
    auto kvCache = at::functionalization::impl::from_functional_tensor(selection_kv_cache);
    auto fullKRope = at::functionalization::impl::from_functional_tensor(full_k_rope);
    auto fullKvCache = at::functionalization::impl::from_functional_tensor(full_kv_cache);
    auto fullQActualSeq = at::functionalization::impl::from_functional_tensor(full_q_actual_seq);
    static auto handle = c10::Dispatcher::singleton()
        .findSchemaOrThrow("custom::npu_avsg_copy_functional", "")
        .typed<std::tuple<at::Tensor, at::Tensor>(const at::Tensor&, const at::Tensor&,
            const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
            const at::Tensor&)>();
    std::tuple<at::Tensor, at::Tensor> result;
    {
        at::AutoDispatchSkipFunctionalize guard;
        result = handle.call(kRope, kvCache, fullKRope, fullKvCache,
            fullQActualSeq, stage_two_copy_mapping, stage_two_copy_count);
    }
    at::functionalization::impl::replace_(selection_k_rope, std::get<0>(result));
    at::functionalization::impl::replace_(selection_kv_cache, std::get<1>(result));
    at::functionalization::impl::commit_update(selection_k_rope);
    at::functionalization::impl::commit_update(selection_kv_cache);
    at::functionalization::impl::sync(selection_k_rope);
    at::functionalization::impl::sync(selection_kv_cache);
    at::functionalization::impl::sync(full_k_rope);
    at::functionalization::impl::sync(full_kv_cache);
    at::functionalization::impl::sync(full_q_actual_seq);
}
} // namespace functionalization

TORCH_LIBRARY_IMPL(custom, Functionalize, m) {
    m.impl("npu_avsg_copy", TORCH_FN(functionalization::npu_avsg_copy_functionalize));
}
} // namespace at_npu
