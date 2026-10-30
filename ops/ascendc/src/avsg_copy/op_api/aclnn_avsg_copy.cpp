/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 */

#include "aclnn_avsg_copy.h"
#include "avsg_copy.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

extern "C" {

aclnnStatus aclnnAvsgCopyGetWorkspaceSize(
    aclTensor* selectionKRope,
    aclTensor* selectionKvCache,
    const aclTensor* fullKRope,
    const aclTensor* fullKvCache,
    const aclTensor* fullQActualSeq,
    const aclTensor* stageTwoCopyMapping,
    const aclTensor* stageTwoCopyCount,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    OP_CHECK_NULL(selectionKRope, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(selectionKvCache, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(fullKvCache, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(fullQActualSeq, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(stageTwoCopyMapping, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(stageTwoCopyCount, return ACLNN_ERR_PARAM_NULLPTR);

    UniqueExecutor uniqueExecutor = CREATE_EXECUTOR();
    aclOpExecutor* executorInst = uniqueExecutor.get();
    CHECK_RET(executorInst != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto results = l0op::AvsgCopy(selectionKRope, selectionKvCache, fullKRope, fullKvCache,
        fullQActualSeq, stageTwoCopyMapping, stageTwoCopyCount, executorInst);
    CHECK_RET(results.outSelectionKRope != nullptr && results.outSelectionKvCache != nullptr,
        ACLNN_ERR_INNER_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAvsgCopy(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAvsgCopy);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

} // extern "C"
