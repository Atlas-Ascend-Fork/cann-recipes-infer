/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 */


#include "avsg_copy_tiling.h"
#include <algorithm>
#include <graph/utils/type_utils.h>

namespace optiling {
namespace {
constexpr int32_t SEL_K_ROPE_IDX = 0;
constexpr int32_t SEL_KV_CACHE_IDX = 1;
constexpr int32_t FULL_K_ROPE_IDX = 2;
constexpr int32_t FULL_KV_CACHE_IDX = 3;
constexpr int32_t FULL_Q_ACTSEQ_IDX = 4;
constexpr int32_t COPY_MAPPING_IDX = 5;
constexpr int32_t COPY_COUNT_IDX = 6;
constexpr int64_t MAX_K_ROPE_DIM = 64;
constexpr int64_t MAX_KV_CACHE_DIM = 1024;
constexpr int64_t DOUBLE_BUFFER_COUNT = 2;

template <typename T>
T CeilDiv(T value, T divisor)
{
    return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
}

template <typename T>
T CeilAlign(T value, T alignment)
{
    return alignment == 0 ? 0 : CeilDiv(value, alignment) * alignment;
}
}

ge::graphStatus AvsgCopyTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OPS_ERR_IF(platformInfo == nullptr,
        OPS_LOG_E(context_->GetNodeName(), "get platformInfo nullptr."), return ge::GRAPH_FAILED);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum_ = platform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ubSize_ = static_cast<int64_t>(ubSize);
    OPS_ERR_IF(coreNum_ <= 0 || ubSize_ <= 0,
        OPS_LOG_E(context_->GetNodeName(), "invalid AIV core or UB size."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvsgCopyTiling::GetShapeAndDtypeInfo()
{
    auto selKRope = context_->GetRequiredInputShape(SEL_K_ROPE_IDX);
    auto selKvCache = context_->GetRequiredInputShape(SEL_KV_CACHE_IDX);
    auto fullKRope = context_->GetOptionalInputShape(FULL_K_ROPE_IDX);
    auto fullKvCache = context_->GetRequiredInputShape(FULL_KV_CACHE_IDX);
    auto fullQActualSeq = context_->GetRequiredInputShape(FULL_Q_ACTSEQ_IDX);
    auto copyMapping = context_->GetRequiredInputShape(COPY_MAPPING_IDX);
    auto copyCount = context_->GetRequiredInputShape(COPY_COUNT_IDX);
    OPS_ERR_IF(selKRope == nullptr || selKvCache == nullptr || fullKvCache == nullptr ||
        fullQActualSeq == nullptr || copyMapping == nullptr || copyCount == nullptr,
        OPS_LOG_E(context_->GetNodeName(), "required input shape is nullptr."), return ge::GRAPH_FAILED);

    gert::Shape selKRopeShape = selKRope->GetOriginShape();
    gert::Shape selKvCacheShape = selKvCache->GetOriginShape();
    gert::Shape fullKvCacheShape = fullKvCache->GetOriginShape();
    gert::Shape fullQShape = fullQActualSeq->GetOriginShape();
    gert::Shape mappingShape = copyMapping->GetOriginShape();
    gert::Shape countShape = copyCount->GetOriginShape();
    OPS_ERR_IF(selKRopeShape.GetDimNum() != 3 || selKvCacheShape.GetDimNum() != 3 ||
        fullKvCacheShape.GetDimNum() != 3 || fullQShape.GetDimNum() != 1,
        OPS_LOG_E(context_->GetNodeName(), "cache/rope or full_q_actual_seq rank is invalid."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(selKvCacheShape.GetDim(1) != fullKvCacheShape.GetDim(1),
        OPS_LOG_E(context_->GetNodeName(), "selection/full kv block sizes must be equal."),
        return ge::GRAPH_FAILED);
    const int64_t blockSize = fullKvCacheShape.GetDim(1);
    OPS_ERR_IF(blockSize <= 0 || (blockSize & (blockSize - 1)) != 0,
        OPS_LOG_E(context_->GetNodeName(), "KV block size must be a power of two."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(selKvCacheShape.GetDim(2) != fullKvCacheShape.GetDim(2) ||
        selKvCacheShape.GetDim(2) > MAX_KV_CACHE_DIM,
        OPS_LOG_E(context_->GetNodeName(), "selection/full KV dimensions are incompatible."),
        return ge::GRAPH_FAILED);

    const bool hasKRope = fullKRope != nullptr;
    int64_t ropeDim = selKRopeShape.GetDim(2);
    if (hasKRope) {
        gert::Shape fullKRopeShape = fullKRope->GetOriginShape();
        OPS_ERR_IF(fullKRopeShape.GetDimNum() != 3 ||
            fullKRopeShape.GetDim(0) != fullKvCacheShape.GetDim(0) ||
            fullKRopeShape.GetDim(1) != blockSize || fullKRopeShape.GetDim(2) != ropeDim ||
            selKRopeShape.GetDim(0) != selKvCacheShape.GetDim(0) ||
            selKRopeShape.GetDim(1) != blockSize || ropeDim > MAX_K_ROPE_DIM,
            OPS_LOG_E(context_->GetNodeName(), "selection/full K-Rope shapes are incompatible."),
            return ge::GRAPH_FAILED);
    }

    const size_t countRank = countShape.GetDimNum();
    OPS_ERR_IF(countRank < 1 || mappingShape.GetDimNum() != countRank + 2 ||
        mappingShape.GetDim(countRank + 1) != 2,
        OPS_LOG_E(context_->GetNodeName(), "copy mapping/count ranks are incompatible."),
        return ge::GRAPH_FAILED);
    int64_t bshCapacity = 1;
    for (size_t dim = 0; dim < countRank; ++dim) {
        OPS_ERR_IF(mappingShape.GetDim(dim) != countShape.GetDim(dim),
            OPS_LOG_E(context_->GetNodeName(), "copy mapping/count leading dimensions differ."),
            return ge::GRAPH_FAILED);
        bshCapacity *= countShape.GetDim(dim);
    }
    const int64_t mappingEntriesPerBsh = mappingShape.GetDim(countRank);
    OPS_ERR_IF(mappingEntriesPerBsh <= 0,
        OPS_LOG_E(context_->GetNodeName(), "mapping entries per BSH must be positive."),
        return ge::GRAPH_FAILED);

    auto selKvDesc = context_->GetRequiredInputDesc(SEL_KV_CACHE_IDX);
    auto fullKvDesc = context_->GetRequiredInputDesc(FULL_KV_CACHE_IDX);
    auto selKRopeDesc = context_->GetRequiredInputDesc(SEL_K_ROPE_IDX);
    auto fullQDesc = context_->GetRequiredInputDesc(FULL_Q_ACTSEQ_IDX);
    auto mappingDesc = context_->GetRequiredInputDesc(COPY_MAPPING_IDX);
    auto countDesc = context_->GetRequiredInputDesc(COPY_COUNT_IDX);
    OPS_ERR_IF(selKvDesc == nullptr || fullKvDesc == nullptr || selKRopeDesc == nullptr ||
        fullQDesc == nullptr || mappingDesc == nullptr || countDesc == nullptr,
        OPS_LOG_E(context_->GetNodeName(), "required input desc is nullptr."), return ge::GRAPH_FAILED);
    cacheDtype_ = selKvDesc->GetDataType();
    OPS_ERR_IF((cacheDtype_ != ge::DT_FLOAT16 && cacheDtype_ != ge::DT_BF16 && cacheDtype_ != ge::DT_INT8) ||
        fullKvDesc->GetDataType() != cacheDtype_ || selKRopeDesc->GetDataType() != cacheDtype_ ||
        fullQDesc->GetDataType() != ge::DT_INT32 || mappingDesc->GetDataType() != ge::DT_INT32 ||
        countDesc->GetDataType() != ge::DT_INT32,
        OPS_LOG_E(context_->GetNodeName(), "input dtypes are incompatible."), return ge::GRAPH_FAILED);
    if (hasKRope) {
        auto fullKRopeDesc = context_->GetOptionalInputDesc(FULL_K_ROPE_IDX);
        OPS_ERR_IF(fullKRopeDesc == nullptr || fullKRopeDesc->GetDataType() != cacheDtype_,
            OPS_LOG_E(context_->GetNodeName(), "full K-Rope dtype is incompatible."), return ge::GRAPH_FAILED);
    }

    const gert::Stride* kvStride = context_->GetRequiredInputStride(FULL_KV_CACHE_IDX);
    int64_t kvStride0 = 0;
    int64_t kvStride1 = 0;
    if (kvStride != nullptr && kvStride->GetDimNum() >= 2) {
        kvStride0 = kvStride->GetStride(0);
        kvStride1 = kvStride->GetStride(1);
    }
    if (kvStride0 <= 0 || kvStride1 <= 0) {
        kvStride1 = fullKvCacheShape.GetDim(2);
        kvStride0 = blockSize * kvStride1;
    }
    int64_t ropeStride0 = 0;
    int64_t ropeStride1 = 0;
    if (hasKRope) {
        const gert::Stride* ropeStride = context_->GetOptionalInputStride(FULL_K_ROPE_IDX);
        if (ropeStride != nullptr && ropeStride->GetDimNum() >= 2) {
            ropeStride0 = ropeStride->GetStride(0);
            ropeStride1 = ropeStride->GetStride(1);
        }
        if (ropeStride0 <= 0 || ropeStride1 <= 0) {
            ropeStride1 = ropeDim;
            ropeStride0 = blockSize * ropeStride1;
        }
    }

    tilingData_.set_bshCapacity(bshCapacity);
    tilingData_.set_mappingEntriesPerBsh(mappingEntriesPerBsh);
    tilingData_.set_batchsize(fullQShape.GetDim(0));
    tilingData_.set_headnum(countShape.GetDim(countRank - 1));
    tilingData_.set_fullKvBlockSize(blockSize);
    tilingData_.set_fullKvBlockSizeLog2(static_cast<int32_t>(__builtin_ctzll(static_cast<uint64_t>(blockSize))));
    tilingData_.set_kRopeDim(ropeDim);
    tilingData_.set_kvCacheDim(fullKvCacheShape.GetDim(2));
    tilingData_.set_fullKvCacheStride0(kvStride0);
    tilingData_.set_fullKvCacheStride1(kvStride1);
    tilingData_.set_fullKRopeStride0(ropeStride0);
    tilingData_.set_fullKRopeStride1(ropeStride1);
    tilingData_.set_hasKRope(hasKRope ? 1 : 0);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvsgCopyTiling::DoOpTiling()
{
    const int64_t totalPotentialEntries =
        tilingData_.get_bshCapacity() * tilingData_.get_mappingEntriesPerBsh();
    if (totalPotentialEntries == 0 || tilingData_.get_batchsize() == 0) {
        tilingData_.set_usedCoreNum(0);
        context_->SetBlockDim(1);
        return ge::GRAPH_SUCCESS;
    }
    const int64_t usedCoreNum = std::min(coreNum_, totalPotentialEntries);
    tilingData_.set_usedCoreNum(usedCoreNum);
    context_->SetBlockDim(usedCoreNum);

    const int64_t dtypeBytes = ge::GetSizeByDataType(cacheDtype_);
    const int64_t kvCacheUbSize = CeilAlign(
        tilingData_.get_kvCacheDim() * dtypeBytes, ubBlockSize_);
    const int64_t kRopeUbSize = CeilAlign(
        tilingData_.get_kRopeDim() * dtypeBytes, ubBlockSize_);
    const int64_t countBytes = CeilAlign(
        tilingData_.get_bshCapacity() * static_cast<int64_t>(sizeof(int32_t)), ubBlockSize_);
    const int64_t prefixBytes = CeilAlign(
        (tilingData_.get_bshCapacity() + 1) * static_cast<int64_t>(sizeof(int64_t)), ubBlockSize_);
    const int64_t mappingBytes = CeilAlign(
        tilingData_.get_mappingEntriesPerBsh() * 2 * static_cast<int64_t>(sizeof(int32_t)), ubBlockSize_);
    const int64_t workBufSize = countBytes + std::max(prefixBytes, mappingBytes);
    const int64_t copyBufferSize = (kvCacheUbSize + kRopeUbSize) * DOUBLE_BUFFER_COUNT;
    OPS_ERR_IF(workBufSize + copyBufferSize > ubSize_,
        OPS_LOG_E(context_->GetNodeName(), "AvsgCopy UB is insufficient: need %ld, have %ld.",
            workBufSize + copyBufferSize, ubSize_), return ge::GRAPH_FAILED);
    tilingData_.set_kvCacheUbSize(kvCacheUbSize);
    tilingData_.set_kRopeUbSize(kRopeUbSize);
    tilingData_.set_workBufSize(workBufSize);
    tilingData_.set_buffNum(DOUBLE_BUFFER_COUNT);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvsgCopyTiling::PostTiling()
{
    context_->SetTilingKey(0);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OPS_ERR_IF(workspaces == nullptr,
        OPS_LOG_E(context_->GetNodeName(), "get workspaces nullptr."), return ge::GRAPH_FAILED);
    auto platform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    workspaces[0] = platform.GetLibApiWorkSpaceSize();
    OPS_ERR_IF(context_->GetRawTilingData() == nullptr,
        OPS_LOG_E(context_->GetNodeName(), "get tiling data nullptr."), return ge::GRAPH_FAILED);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvsgCopyTiling::RunTiling()
{
    OPS_ERR_IF(context_ == nullptr, OPS_LOG_E("AvsgCopy", "context is nullptr."), return ge::GRAPH_FAILED);
    if (GetShapeAndDtypeInfo() != ge::GRAPH_SUCCESS || GetPlatformInfo() != ge::GRAPH_SUCCESS ||
        DoOpTiling() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return PostTiling();
}

ge::graphStatus Tiling4AvsgCopy(gert::TilingContext* context)
{
    AvsgCopyTiling tiling(context);
    return tiling.RunTiling();
}

ge::graphStatus TilingPrepare4AvsgCopy(gert::TilingParseContext*)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AvsgCopy)
    .Tiling(Tiling4AvsgCopy)
    .TilingParse<AvsgCopyCompileInfo>(TilingPrepare4AvsgCopy);
} // namespace optiling
