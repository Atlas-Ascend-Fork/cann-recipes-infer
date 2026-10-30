/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 */

#include "aclnn/aclnn_base.h"
#include "aclnn_avsg_match.h"
#include "avsg_match.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;
// 标准双 API 模式	aclnnXxxGetWorkspaceSize() + aclnnXxx()
#ifdef __cplusplus
extern "C" {
#endif

static bool CheckInputsNotNull(
    const aclTensor* selectionKvBlockTable,
    const aclTensor* selectionKvBlockStatus,
    const aclTensor* selectionTopkIndices,
    const aclTensor* fullKvBlockTable,
    const aclTensor* fullKvActualSeq,
    const aclTensor* fullQActualSeq,
    const aclTensor* outSelectionKvActualSeq,
    const aclTensor* outStageTwoCopyMapping,
    const aclTensor* outStageTwoCopyCount)
{
    OP_CHECK_NULL(selectionKvBlockTable, return false);
    OP_CHECK_NULL(selectionKvBlockStatus, return false);
    OP_CHECK_NULL(selectionTopkIndices, return false);
    OP_CHECK_NULL(fullKvBlockTable, return false);
    OP_CHECK_NULL(fullKvActualSeq, return false);
    OP_CHECK_NULL(fullQActualSeq, return false);
    OP_CHECK_NULL(outSelectionKvActualSeq, return false);
    OP_CHECK_NULL(outStageTwoCopyMapping, return false);
    OP_CHECK_NULL(outStageTwoCopyCount, return false);
    return true;
}

aclnnStatus aclnnAvsgMatchGetWorkspaceSize(
    aclTensor* selectionKvBlockTable,
    aclTensor* selectionKvBlockStatus,
    aclTensor* selectionTopkIndices,
    const aclTensor* fullKvBlockTable,
    const aclTensor* fullKvActualSeq,
    const aclTensor* fullQActualSeq,
    int64_t selectionTopkBlockSize,
    int64_t kvBlockSize,
    aclTensor* outSelectionKvActualSeq,
    aclTensor* outStageTwoCopyMapping,
    aclTensor* outStageTwoCopyCount,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{

    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    CHECK_RET(
        CheckInputsNotNull(
            selectionKvBlockTable, selectionKvBlockStatus, selectionTopkIndices, fullKvBlockTable,
            fullKvActualSeq, fullQActualSeq, outSelectionKvActualSeq, outStageTwoCopyMapping,
            outStageTwoCopyCount),
        ACLNN_ERR_PARAM_NULLPTR);

    OP_CHECK(selectionTopkBlockSize > 0 && kvBlockSize > 0,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "block sizes must be positive."), return ACLNN_ERR_PARAM_INVALID);

    UniqueExecutor uniqueExecutor = CREATE_EXECUTOR();
    aclOpExecutor* executorInst = uniqueExecutor.get();
    CHECK_RET(executorInst != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 调用l0op算子构建计算图 不执行 具体执行在CommonOpExecutorRun
    auto results = l0op::AvsgMatch(
        selectionKvBlockTable, selectionKvBlockStatus,
        selectionTopkIndices, fullKvBlockTable, fullKvActualSeq, fullQActualSeq,
        selectionTopkBlockSize, kvBlockSize, executorInst);
    CHECK_RET(results.outSelectionKvActualSeq != nullptr && results.stageTwoCopyMapping != nullptr &&
        results.stageTwoCopyCount != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // ViewCopy 是 l0op内置的算子，用于在两个张量之间拷贝数据
    auto vc0 = l0op::ViewCopy(results.outSelectionKvActualSeq, outSelectionKvActualSeq, executorInst);
    CHECK_RET(vc0 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto vc1 = l0op::ViewCopy(results.stageTwoCopyMapping, outStageTwoCopyMapping, executorInst);
    CHECK_RET(vc1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto vc2 = l0op::ViewCopy(results.stageTwoCopyCount, outStageTwoCopyCount, executorInst);
    CHECK_RET(vc2 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAvsgMatch(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAvsgMatch);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream); // CommonOpExecutorRun 是框架提供的通用执行函数
}

#ifdef __cplusplus
}
#endif
