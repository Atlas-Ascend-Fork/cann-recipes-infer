/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 */

#ifndef OP_API_INC_LEVEL0_OP_AVSG_COPY_H_
#define OP_API_INC_LEVEL0_OP_AVSG_COPY_H_

#include "opdev/op_executor.h"

namespace l0op {

struct AvsgCopyOutputs {
    aclTensor* outSelectionKRope;
    aclTensor* outSelectionKvCache;
};

AvsgCopyOutputs AvsgCopy(
    aclTensor* selectionKRope,
    aclTensor* selectionKvCache,
    const aclTensor* fullKRope,
    const aclTensor* fullKvCache,
    const aclTensor* fullQActualSeq,
    const aclTensor* stageTwoCopyMapping,
    const aclTensor* stageTwoCopyCount,
    aclOpExecutor* executor);

} // namespace l0op
#endif // OP_API_INC_LEVEL0_OP_AVSG_COPY_H_
