/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 */

#include "avsg_copy.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(AvsgCopy);

namespace {
const aclTensor* WrapViewTensor(const aclTensor* input, aclOpExecutor* executor)
{
    if (input == nullptr) {
        return nullptr;
    }
    return executor->CreateView(input, input->GetViewShape(), input->GetStorageShape(),
        input->GetViewStrides(), input->GetViewOffset());
}
}

AvsgCopyOutputs AvsgCopy(
    aclTensor* selectionKRope,
    aclTensor* selectionKvCache,
    const aclTensor* fullKRope,
    const aclTensor* fullKvCache,
    const aclTensor* fullQActualSeq,
    const aclTensor* stageTwoCopyMapping,
    const aclTensor* stageTwoCopyCount,
    aclOpExecutor* executor)
{
    L0_DFX(AvsgCopy, selectionKRope, selectionKvCache, fullKRope, fullKvCache,
        fullQActualSeq, stageTwoCopyMapping, stageTwoCopyCount);
    AvsgCopyOutputs nullResult = {nullptr, nullptr};
    const aclTensor* wrappedSelectionKRope = WrapViewTensor(selectionKRope, executor);
    const aclTensor* wrappedSelectionKvCache = WrapViewTensor(selectionKvCache, executor);
    const aclTensor* wrappedFullKRope = WrapViewTensor(fullKRope, executor);
    const aclTensor* wrappedFullKvCache = WrapViewTensor(fullKvCache, executor);
    const aclTensor* wrappedFullQActualSeq = WrapViewTensor(fullQActualSeq, executor);
    const aclTensor* wrappedCopyMapping = WrapViewTensor(stageTwoCopyMapping, executor);
    const aclTensor* wrappedCopyCount = WrapViewTensor(stageTwoCopyCount, executor);
    OP_CHECK(wrappedSelectionKRope != nullptr && wrappedSelectionKvCache != nullptr &&
        wrappedFullKvCache != nullptr && wrappedFullQActualSeq != nullptr &&
        wrappedCopyMapping != nullptr && wrappedCopyCount != nullptr,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "WrapViewTensor failed."), return nullResult);

    auto ret = INFER_SHAPE(AvsgCopy,
        OP_INPUT(wrappedSelectionKRope, wrappedSelectionKvCache, wrappedFullKRope,
            wrappedFullKvCache, wrappedFullQActualSeq, wrappedCopyMapping, wrappedCopyCount),
        OP_OUTPUT(selectionKRope, selectionKvCache));
    OP_CHECK(ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AvsgCopy InferShape failed."), return nullResult);
    auto launchRet = ADD_TO_LAUNCHER_LIST_AICORE(AvsgCopy,
        OP_INPUT(wrappedSelectionKRope, wrappedSelectionKvCache, wrappedFullKRope,
            wrappedFullKvCache, wrappedFullQActualSeq, wrappedCopyMapping, wrappedCopyCount),
        OP_OUTPUT(selectionKRope, selectionKvCache));
    OP_CHECK(launchRet == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AvsgCopy launch failed."), return nullResult);
    return {selectionKRope, selectionKvCache};
}

} // namespace l0op
