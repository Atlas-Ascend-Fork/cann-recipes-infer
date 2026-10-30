/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 */

#include "avsg_match.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(AvsgMatch); //注册算子类型到框架，使得框架能识别这个算子名

namespace {
constexpr int64_t MAPPING_FIELDS = 2;
constexpr int64_t MAPPING_ALIGN_ENTRIES = 4;

// 封装带有stride的view shape，确保 tiling 层能获取到正确的 view 信息
static const aclTensor* WrapViewTensor(const aclTensor* input, aclOpExecutor* executor)
{
    if (input == nullptr) {
        return nullptr;
    }
    aclTensor* newTensor = executor->CreateView(  // 只是创建一个新的张量描述符（元数据），共享底层内存，不涉及数据拷贝
        input,
        input->GetViewShape(),      // 视图形状（逻辑形状）
        input->GetStorageShape(),   // 底层存储形状
        input->GetViewStrides(),    // 步幅（非连续的关键信息）
        input->GetViewOffset()      // 偏移量
    );
    return newTensor;
}

void BuildIntermediateShapes(const aclTensor* selectionTopkIndices,
    gert::Shape& mappingShape, gert::Shape& countShape)
{
    op::Shape topkShape = selectionTopkIndices->GetViewShape();
    const size_t rank = topkShape.GetDimNum();
    for (size_t dim = 0; dim + 1 < rank; ++dim) {
        mappingShape.AppendDim(topkShape.GetDim(dim));
        countShape.AppendDim(topkShape.GetDim(dim));
    }
    const int64_t topk = topkShape.GetDim(rank - 1);
    mappingShape.AppendDim((topk + MAPPING_ALIGN_ENTRIES - 1) / MAPPING_ALIGN_ENTRIES * MAPPING_ALIGN_ENTRIES);
    mappingShape.AppendDim(MAPPING_FIELDS);
}
}

AvsgMatchOutputs AvsgMatch(
    aclTensor* selectionKvBlockTable,
    aclTensor* selectionKvBlockStatus,
    aclTensor* selectionTopkIndices,
    const aclTensor* fullKvBlockTable,
    const aclTensor* fullKvActualSeq,
    const aclTensor* fullQActualSeq,
    int64_t selectionTopkBlockSize,
    int64_t kvBlockSize,
    aclOpExecutor* executor)
{
    L0_DFX(AvsgMatch, selectionKvBlockTable, selectionKvBlockStatus, selectionTopkIndices,
        fullKvBlockTable, fullKvActualSeq, fullQActualSeq, selectionTopkBlockSize, kvBlockSize);
    AvsgMatchOutputs nullResult = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};

    // 使用 CreateView 封装 view 信息，让 tiling 层能获取到正确的 view shape 和 stride
    const aclTensor* wrappedSelKvBlkT = WrapViewTensor(selectionKvBlockTable, executor);
    const aclTensor* wrappedSelKvBlkSt = WrapViewTensor(selectionKvBlockStatus, executor);
    const aclTensor* wrappedSelTopKIn = WrapViewTensor(selectionTopkIndices, executor);
    const aclTensor* wrappedFulKvBlkT = WrapViewTensor(fullKvBlockTable, executor);
    const aclTensor* wrappedFulKvActSeq = WrapViewTensor(fullKvActualSeq, executor);
    const aclTensor* wrappedFulQActSeq = WrapViewTensor(fullQActualSeq, executor);

    OP_CHECK(wrappedSelKvBlkT != nullptr && wrappedSelKvBlkSt != nullptr &&
             wrappedSelTopKIn != nullptr && wrappedFulKvBlkT != nullptr &&
             wrappedFulKvActSeq != nullptr && wrappedFulQActSeq != nullptr,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "WrapViewTensor failed."),
        return nullResult);

    // outSelKvActualSeq should have shape [batchsize], same as fullKvActualSeq
    op::Shape fulKvActSeqViewShape = fullKvActualSeq->GetViewShape();
    gert::Shape outSelKvActualSeqShape;
    outSelKvActualSeqShape.AppendDim(fulKvActSeqViewShape.GetDimNum() > 0 ? fulKvActSeqViewShape.GetDim(0) : 0);
    gert::Shape mappingShape;
    gert::Shape countShape;
    BuildIntermediateShapes(selectionTopkIndices, mappingShape, countShape);
    auto outSelKvActualSeq = executor->AllocTensor(outSelKvActualSeqShape, fullKvActualSeq->GetDataType());
    auto stageTwoCopyMapping = executor->AllocTensor(mappingShape, selectionTopkIndices->GetDataType());
    auto stageTwoCopyCount = executor->AllocTensor(countShape, selectionTopkIndices->GetDataType());

    // INFER_SHAPE 宏的作用：
    //     调用 op_host 层的 avsg_match_proto.cpp
    //     进行形状推导，验证输入输出的形状兼容性
    //     传递属性给 tiling 阶段
    // CANN约定：INFER_SHAPE(算子名, ...)  →  触发 op_host/*_proto.cpp 中的形状推导
    // 注意：形状和 stride 信息通过 wrapped tensor 传递，tiling 层使用 GetOriginShape() 和 GetInputStride() 获取
    auto ret = INFER_SHAPE(AvsgMatch,
        OP_INPUT(wrappedSelKvBlkT, wrappedSelKvBlkSt, wrappedSelTopKIn, wrappedFulKvBlkT,
                 wrappedFulKvActSeq, wrappedFulQActSeq),
        OP_ATTR(selectionTopkBlockSize, kvBlockSize),
        OP_OUTPUT(selectionKvBlockTable, selectionKvBlockStatus, outSelKvActualSeq,
                  stageTwoCopyMapping, stageTwoCopyCount, selectionTopkIndices));
    OP_CHECK(ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AvsgMatch InferShape failed."),
        return nullResult);

    // ADD_TO_LAUNCHER_LIST_AICORE 宏的作用：
    //     将算子加入 AICore 执行队列
    //     触发 tiling 计算（调用 avsg_match_tiling.cpp）
    //     最终启动 kernel（调用 op_kernel/avsg_match.cpp）
    // CANN约定：ADD_TO_LAUNCHER_LIST_AICORE(算子名, ...)  →  触发 op_host/*_tiling.cpp 中的 tiling 计算
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(AvsgMatch,
        OP_INPUT(wrappedSelKvBlkT, wrappedSelKvBlkSt, wrappedSelTopKIn, wrappedFulKvBlkT,
                 wrappedFulKvActSeq, wrappedFulQActSeq),
        OP_ATTR(selectionTopkBlockSize, kvBlockSize),
        OP_OUTPUT(selectionKvBlockTable, selectionKvBlockStatus, outSelKvActualSeq,
                  stageTwoCopyMapping, stageTwoCopyCount, selectionTopkIndices));
    OP_CHECK(retAicore == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AvsgMatch ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullResult);

    return {selectionKvBlockTable, selectionKvBlockStatus, outSelKvActualSeq,
        stageTwoCopyMapping, stageTwoCopyCount, selectionTopkIndices};
}

} // namespace l0op
