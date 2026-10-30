/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 */

/*!
 * \file avsg_match.cc
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "error/ops_error.h"

using namespace ge;
namespace ops {
const int32_t INDEX_INPUT_0 = 0;
const int32_t INDEX_INPUT_1 = 1;
const int32_t INDEX_INPUT_2 = 2;
const int32_t INDEX_INPUT_4 = 4;
const int32_t INDEX_OUTPUT_0 = 0;
const int32_t INDEX_OUTPUT_1 = 1;
const int32_t INDEX_OUTPUT_2 = 2;
const int32_t INDEX_OUTPUT_3 = 3;
const int32_t INDEX_OUTPUT_4 = 4;
const int32_t INDEX_OUTPUT_5 = 5;
const int64_t COPY_MAPPING_FIELDS = 2;
const int64_t COPY_MAPPING_ALIGN_ENTRIES = 4;

static ge::graphStatus InferShape4AvsgMatch(gert::InferShapeContext* context)
{
    const gert::Shape* selectionKvBlockTableShape = context->GetInputShape(INDEX_INPUT_0);
    const gert::Shape* selectionKvBlocStatusShape = context->GetInputShape(INDEX_INPUT_1);
    const gert::Shape* selectionTopkIndicesShape = context->GetInputShape(INDEX_INPUT_2);
    const gert::Shape* fullKvActualSeqShape = context->GetInputShape(INDEX_INPUT_4);
    OPS_LOG_E_IF_NULL(context, selectionKvBlockTableShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, selectionKvBlocStatusShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, selectionTopkIndicesShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, fullKvActualSeqShape, return ge::GRAPH_FAILED);

    *context->GetOutputShape(INDEX_OUTPUT_0) = *selectionKvBlockTableShape;
    *context->GetOutputShape(INDEX_OUTPUT_1) = *selectionKvBlocStatusShape;
    *context->GetOutputShape(INDEX_OUTPUT_2) = *fullKvActualSeqShape;
    *context->GetOutputShape(INDEX_OUTPUT_5) = *selectionTopkIndicesShape;

    const size_t topkRank = selectionTopkIndicesShape->GetDimNum();
    OPS_ERR_IF(topkRank < 1,
        OPS_LOG_E(context->GetNodeName(), "selection_topk_indices rank must be positive."),
        return ge::GRAPH_FAILED);
    gert::Shape* copyCountShape = context->GetOutputShape(INDEX_OUTPUT_4);
    *copyCountShape = *selectionTopkIndicesShape;
    copyCountShape->SetDimNum(topkRank - 1);

    gert::Shape* copyMappingShape = context->GetOutputShape(INDEX_OUTPUT_3);
    *copyMappingShape = *selectionTopkIndicesShape;
    const int64_t topk = selectionTopkIndicesShape->GetDim(topkRank - 1);
    const int64_t alignedTopk = (topk + COPY_MAPPING_ALIGN_ENTRIES - 1) /
        COPY_MAPPING_ALIGN_ENTRIES * COPY_MAPPING_ALIGN_ENTRIES;
    copyMappingShape->SetDim(topkRank - 1, alignedTopk);
    copyMappingShape->AppendDim(COPY_MAPPING_FIELDS);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype4AvsgMatch(gert::InferDataTypeContext* context)
{
    for (int32_t outputIdx = INDEX_OUTPUT_0; outputIdx <= INDEX_OUTPUT_5; ++outputIdx) {
        context->SetOutputDataType(outputIdx, ge::DT_INT32);
    }
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AvsgMatch)
    .InferShape(InferShape4AvsgMatch)
    .InferDataType(InferDtype4AvsgMatch);
}  // namespace ops
