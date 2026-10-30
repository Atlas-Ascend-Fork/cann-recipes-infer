/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 */

#ifndef OP_API_INC_LEVEL0_OP_AVSG_MATCH_H_
#define OP_API_INC_LEVEL0_OP_AVSG_MATCH_H_

#include "opdev/op_executor.h"

namespace l0op {

struct AvsgMatchOutputs {
    aclTensor* outSelectionKvBlockTable;
    aclTensor* outSelectionKvBlockStatus;
    aclTensor* outSelectionKvActualSeq;
    aclTensor* stageTwoCopyMapping;
    aclTensor* stageTwoCopyCount;
    aclTensor* outSelectionTopkIndices;
};

AvsgMatchOutputs AvsgMatch(
    aclTensor* selectionKvBlockTable,
    aclTensor* selectionKvBlockStatus,
    aclTensor* selectionTopkIndices,
    const aclTensor* fullKvBlockTable,
    const aclTensor* fullKvActualSeq,
    const aclTensor* fullQActualSeq,
    int64_t selectionTopkBlockSize,
    int64_t kvBlockSize,
    aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_OP_AVSG_MATCH_H_
