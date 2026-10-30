/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 */

#ifndef OP_API_INC_AVSG_MATCH_H_
#define OP_API_INC_AVSG_MATCH_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnAvsgMatchGetWorkspaceSize(
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
    aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnAvsgMatch(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_AVSG_MATCH_H_
