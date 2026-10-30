/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 */

#ifndef OP_API_INC_AVSG_COPY_H_
#define OP_API_INC_AVSG_COPY_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnAvsgCopyGetWorkspaceSize(
    aclTensor* selectionKRope,
    aclTensor* selectionKvCache,
    const aclTensor* fullKRope,
    const aclTensor* fullKvCache,
    const aclTensor* fullQActualSeq,
    const aclTensor* stageTwoCopyMapping,
    const aclTensor* stageTwoCopyCount,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnAvsgCopy(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif
#endif // OP_API_INC_AVSG_COPY_H_
