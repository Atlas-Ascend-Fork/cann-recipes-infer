/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 */

#include <register/op_impl_registry.h>
#include "error/ops_error.h"

namespace ops {
static ge::graphStatus InferShape4AvsgCopy(gert::InferShapeContext* context)
{
    const gert::Shape* selectionKRopeShape = context->GetInputShape(0);
    const gert::Shape* selectionKvCacheShape = context->GetInputShape(1);
    OPS_LOG_E_IF_NULL(context, selectionKRopeShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, selectionKvCacheShape, return ge::GRAPH_FAILED);
    *context->GetOutputShape(0) = *selectionKRopeShape;
    *context->GetOutputShape(1) = *selectionKvCacheShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype4AvsgCopy(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, context->GetInputDataType(1));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AvsgCopy)
    .InferShape(InferShape4AvsgCopy)
    .InferDataType(InferDtype4AvsgCopy);
} // namespace ops
