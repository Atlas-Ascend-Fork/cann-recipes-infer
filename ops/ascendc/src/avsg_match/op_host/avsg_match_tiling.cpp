/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 */

/*!
 * \file avsg_match_tiling.cpp
 * \brief
 */

#include "avsg_match_tiling.h"

namespace optiling {

constexpr int32_t SEL_KV_BLOCK_TABLE_IDX = 0;
constexpr int32_t SEL_KV_BLOCK_STAT_IDX = 1;
constexpr int32_t SEL_TOPK_INDICES_IDX = 2;
constexpr int32_t FULL_KV_BLOCK_TABLE_IDX = 3;
constexpr int32_t FULL_KV_ACTSEQ_IDX = 4;
constexpr int32_t FULL_Q_ACTSEQ_IDX = 5;
constexpr int64_t DEFAULT_TOPK_BLOCK_SIZE = 64;
constexpr size_t CONST1 = 1;
constexpr size_t CONST2 = 2;
constexpr size_t CONST3 = 3;
constexpr size_t CONST4 = 4;

constexpr int32_t MAX_TOPK_NUM = 2048;
constexpr int32_t TOPK_SPLIT_NUM = 32;

template <typename T>
static inline T CeilDiv(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd)));
}

template <typename T>
static inline T CeilAlign(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd)) * (rnd));
}

ge::graphStatus AvsgMatchTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OPS_ERR_IF(platformInfo == nullptr, OPS_LOG_E(context_->GetNodeName(), "get platformInfo nullptr."),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    OPS_ERR_IF(
        coreNum_ <= 0, OPS_LOG_E(context_->GetNodeName(), "coreNum must be greater than 0."),
        return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    ubSize_ = static_cast<int64_t>(ubSizePlatForm);
    OPS_ERR_IF(
        ubSize_ <= 0, OPS_LOG_E(context_->GetNodeName(), "ubSize must be greater than 0."),
        return ge::GRAPH_FAILED);

    ubBlockSize_ = 32; // UB allocation alignment in bytes.

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvsgMatchTiling::GetInputAttrs()
{
    auto attrs = context_->GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OPS_LOG_E(context_->GetNodeName(), "get attrs nullptr."),
        return ge::GRAPH_FAILED);
    const int64_t* attrBlockSizePtr = attrs->GetAttrPointer<int64_t>(0);
    if (attrBlockSizePtr == nullptr || *attrBlockSizePtr <= 0) {
        selTopKBlockSize_ = DEFAULT_TOPK_BLOCK_SIZE;
    } else {
        selTopKBlockSize_ = *attrBlockSizePtr;
    }
    const int64_t* kvBlockSizePtr = attrs->GetAttrPointer<int64_t>(1);
    OPS_ERR_IF(kvBlockSizePtr == nullptr || *kvBlockSizePtr <= 0,
        OPS_LOG_E(context_->GetNodeName(), "kv_block_size must be positive."), return ge::GRAPH_FAILED);
    kvBlockSize_ = *kvBlockSizePtr;
    OPS_ERR_IF((kvBlockSize_ & (kvBlockSize_ - 1)) != 0 || kvBlockSize_ % selTopKBlockSize_ != 0,
        OPS_LOG_E(context_->GetNodeName(), "kv_block_size must be a power of two and a multiple of selection_topk_block_size."),
        return ge::GRAPH_FAILED);
    tilingData_.set_selTopKBlockSize(selTopKBlockSize_);
    tilingData_.set_fullKvBlockSize(kvBlockSize_);
    tilingData_.set_fullKvBlockSizeLog2(static_cast<int32_t>(__builtin_ctzll(static_cast<uint64_t>(kvBlockSize_))));
    tilingData_.set_selKvBlockSize(kvBlockSize_);
    tilingData_.set_selKvBlockSizeLog2(__builtin_ctzll(static_cast<uint64_t>(kvBlockSize_)));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvsgMatchTiling::GetSelBlockTable()
{
    // selection_kv_block_table: [batchsize, s_maxblocknum], shared by queries in each batch.
    auto selKvBlkTIn = context_->GetRequiredInputShape(SEL_KV_BLOCK_TABLE_IDX);
    OPS_ERR_IF(selKvBlkTIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get selKvBlkTIn nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape selKvBlkTInShape = selKvBlkTIn->GetOriginShape();
    size_t dimsN = selKvBlkTInShape.GetDimNum();
    OPS_ERR_IF((dimsN != CONST2),
        OPS_LOG_E(context_->GetNodeName(), "selection_kv_block_table dim:%lu should be 2.", dimsN),
        return ge::GRAPH_FAILED);
    tilingData_.set_selMaxBlockNum(selKvBlkTInShape.GetDim(1));
    selKvBlockTableRow_ = selKvBlkTInShape.GetDim(0);

    // selection_kv_block_status: [batchsize, statesize * 2], two int32 words per entry.
    auto selKvBlkStIn = context_->GetRequiredInputShape(SEL_KV_BLOCK_STAT_IDX);
    OPS_ERR_IF(selKvBlkStIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get selKvBlkStIn nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape selKvBlkStShape = selKvBlkStIn->GetOriginShape();
    dimsN = selKvBlkStShape.GetDimNum();
    OPS_ERR_IF((dimsN != CONST2),
        OPS_LOG_E(context_->GetNodeName(), "selection_kv_block_status dim:%lu should be 2.", dimsN),
        return ge::GRAPH_FAILED);
    stateSize_ = selKvBlkStShape.GetDim(CONST1) / CONST2;
    // Hash slot extraction and probing use statesize - 1 as a bit mask.
    OPS_ERR_IF(stateSize_ <= 0 || (stateSize_ & (stateSize_ - 1)) != 0,
        OPS_LOG_E(context_->GetNodeName(), "statesize:%ld must be a power of two.", stateSize_),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((stateSize_ != tilingData_.get_selKvBlockSize() * selKvBlkTInShape.GetDim(1)),
        OPS_LOG_E(context_->GetNodeName(), "selection_kv_block_status last dim:%ld should be %ld.",
            stateSize_, tilingData_.get_selKvBlockSize() * selKvBlkTInShape.GetDim(1)),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvsgMatchTiling::GetTopkIndices()
{
    // selection_topk_indices: [batchsize, seq, headnum, topk].  BSND or TND
    auto selTopKIn = context_->GetRequiredInputShape(SEL_TOPK_INDICES_IDX);
    OPS_ERR_IF(selTopKIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get selTopKIn nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape selTopKInShape = selTopKIn->GetOriginShape();
    size_t dimsN = selTopKInShape.GetDimNum();
    OPS_ERR_IF((dimsN != CONST4 && dimsN != CONST3),
        OPS_LOG_E(context_->GetNodeName(), "selection_topk_indices dim:%lu should be 3 or 4.", dimsN),
        return ge::GRAPH_FAILED);
    if (dimsN == CONST4) {
        t_ = selTopKInShape.GetDim(0) * selTopKInShape.GetDim(CONST1);
        headnum_ = selTopKInShape.GetDim(CONST2);
        topk_ = selTopKInShape.GetDim(CONST3);
    } else {
        t_ = selTopKInShape.GetDim(0);
        headnum_ = selTopKInShape.GetDim(CONST1);
        topk_ = selTopKInShape.GetDim(CONST2);
    }
    OPS_ERR_IF((topk_ > TOPK_SPLIT_NUM && selTopKBlockSize_ != 1),
        OPS_LOG_E(context_->GetNodeName(), "topk:%ld is more than %d, only support block size 1 but got :%ld",
            topk_, TOPK_SPLIT_NUM, selTopKBlockSize_), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvsgMatchTiling::CheckSelInfo()
{
    OPS_ERR_IF((topk_ > MAX_TOPK_NUM),
        OPS_LOG_E(context_->GetNodeName(), "selection_topk_indices topk_:%ld should <= %d.", topk_, MAX_TOPK_NUM),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((headnum_ != 1),
        OPS_LOG_E(context_->GetNodeName(), "selection_topk_indices headnum_:%ld only supports 1.", headnum_),
        return ge::GRAPH_FAILED);
    // Get batchSize_ from full_q_actual_seq dim0
    auto fulQSeqIn = context_->GetRequiredInputShape(FULL_Q_ACTSEQ_IDX);
    OPS_ERR_IF(fulQSeqIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get fulQSeqIn nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape fulQSeqInShape = fulQSeqIn->GetOriginShape();
    batchSize_ = fulQSeqInShape.GetDim(0);
    // full_kv_actual_seq dim0 must match full_q_actual_seq dim0
    auto fulKvSeqIn = context_->GetRequiredInputShape(FULL_KV_ACTSEQ_IDX);
    OPS_ERR_IF(fulKvSeqIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get fulKvSeqIn nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape fulKvSeqInShape = fulKvSeqIn->GetOriginShape();
    OPS_ERR_IF((fulKvSeqInShape.GetDim(0) != batchSize_),
        OPS_LOG_E(context_->GetNodeName(), "full_kv_actual_seq dim0:%ld should == full_q_actual_seq dim0:%ld.",
            fulKvSeqInShape.GetDim(0), batchSize_), return ge::GRAPH_FAILED);
    OPS_ERR_IF((t_ < batchSize_),
        OPS_LOG_E(context_->GetNodeName(), "t_:%ld should >= batchSize_:%ld.", t_, batchSize_),
        return ge::GRAPH_FAILED);
    // selection_kv_block_table dim0 must >= batchSize_
    OPS_ERR_IF((selKvBlockTableRow_ < batchSize_),
        OPS_LOG_E(context_->GetNodeName(), "selection_kv_block_table dim0:%ld need >= BatchSize:%ld.",
            selKvBlockTableRow_, batchSize_), return ge::GRAPH_FAILED);
    // s_maxblocknum >= ceil(topk * selection_topk_block_size / kv_block_size)
    int64_t minBlockNum = (topk_ * selTopKBlockSize_ + tilingData_.get_selKvBlockSize() - 1) /
        tilingData_.get_selKvBlockSize();
    OPS_ERR_IF((tilingData_.get_selMaxBlockNum() < minBlockNum),
        OPS_LOG_E(context_->GetNodeName(),
            "s_maxblocknum:%ld is too small, topk_:%ld selTopKBlockSize_:%ld s_block_size:%ld",
            tilingData_.get_selMaxBlockNum(), topk_, selTopKBlockSize_, tilingData_.get_selKvBlockSize()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvsgMatchTiling::GetFullKvBlkTable()
{
    // [batchsize, f_maxblocknum]
    auto fulKvBlkTIn = context_->GetRequiredInputShape(FULL_KV_BLOCK_TABLE_IDX);
    OPS_ERR_IF(fulKvBlkTIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get fulKvBlkTIn nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape fulKvBlkTInShape = fulKvBlkTIn->GetOriginShape();
    size_t dimsN = fulKvBlkTInShape.GetDimNum();
    OPS_ERR_IF((dimsN != CONST2),
        OPS_LOG_E(context_->GetNodeName(), "full_kv_block_table dim:%lu should be 2.", dimsN),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((fulKvBlkTInShape.GetDim(0) != batchSize_),
        OPS_LOG_E(context_->GetNodeName(), "full_kv_block_table dim0:%ld should be batchSize:%ld.",
            fulKvBlkTInShape.GetDim(0), batchSize_), return ge::GRAPH_FAILED);
    // The packed cache word reserves tokenId 0xFFFFFF for the invalid sentinel.
    // Validate the addressable token range on host, before any device packing.
    constexpr int64_t MAX_TOKEN_COUNT = 0x00FFFFFF;
    OPS_ERR_IF(fulKvBlkTInShape.GetDim(1) > MAX_TOKEN_COUNT / kvBlockSize_,
        OPS_LOG_E(context_->GetNodeName(), "full KV capacity exceeds the packed 24-bit token range."),
        return ge::GRAPH_FAILED);
    tilingData_.set_fullMaxBlockNum(fulKvBlkTInShape.GetDim(1));
    tilingData_.set_batchsize(batchSize_);
    tilingData_.set_t(t_);
    tilingData_.set_headnum(headnum_);
    tilingData_.set_topk(topk_);
    tilingData_.set_statesize(stateSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvsgMatchTiling::GetSeqLenIn()
{
    // full_kv_actual_seq: [batchsize]
    auto fulKvSeqIn = context_->GetRequiredInputShape(FULL_KV_ACTSEQ_IDX);
    OPS_ERR_IF(fulKvSeqIn == nullptr || fulKvSeqIn->GetOriginShape().GetDimNum() != 1,
        OPS_LOG_E(context_->GetNodeName(), "full_kv_actual_seq dim should be 1."), return ge::GRAPH_FAILED);
    // full_q_actual_seq: [batchsize]
    auto fulQSeqIn = context_->GetRequiredInputShape(FULL_Q_ACTSEQ_IDX);
    OPS_ERR_IF(fulQSeqIn == nullptr || fulQSeqIn->GetOriginShape().GetDimNum() != 1,
        OPS_LOG_E(context_->GetNodeName(), "full_q_actual_seq dim should be 1."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvsgMatchTiling::GetInputDtypeInfo()
{
    for (int32_t inputIdx = 0; inputIdx <= FULL_Q_ACTSEQ_IDX; ++inputIdx) {
        auto inputDesc = context_->GetRequiredInputDesc(inputIdx);
        OPS_ERR_IF(inputDesc == nullptr || inputDesc->GetDataType() != ge::DT_INT32,
            OPS_LOG_E(context_->GetNodeName(), "kv cache idx dtype should be int32."), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvsgMatchTiling::GetShapeAttrsInfo()
{
    OPS_ERR_IF(context_ == nullptr, OPS_LOG_E("AvsgMatch", "context can not be nullptr."),
        return ge::GRAPH_FAILED);
    if (GetInputAttrs() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetSelBlockTable() != ge::GRAPH_SUCCESS || GetTopkIndices() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckSelInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetFullKvBlkTable() != ge::GRAPH_SUCCESS || GetSeqLenIn() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // dtype校验
    if (GetInputDtypeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvsgMatchTiling::DoOpTiling()
{
    tilingData_.set_ubSize(ubSize_);
    if (batchSize_ == 0) {
        tilingData_.set_stageOneUsedCoreNum(0);
        context_->SetBlockDim(1);
        return ge::GRAPH_SUCCESS;
    }
    int64_t bsCoreFactor = CeilDiv(batchSize_, static_cast<int64_t>(coreNum_));
    int64_t bsCoreNum = CeilDiv(batchSize_, bsCoreFactor);
    tilingData_.set_stageOneUsedCoreNum(bsCoreNum);
    bsCoreFactor = CeilDiv(batchSize_, bsCoreNum);
    int64_t tailCoreBsFactor = batchSize_ - (bsCoreNum - 1) * bsCoreFactor;
    tilingData_.set_stageOneMainCoreBsLoopNum(bsCoreFactor);
    tilingData_.set_stageOneTailCoreBsLoopNum(tailCoreBsFactor);

    const int64_t topkUb = CeilAlign(topk_ * static_cast<int64_t>(sizeof(int32_t)), ubBlockSize_);
    const int64_t statusUb = CeilAlign(stateSize_ * 2 * static_cast<int64_t>(sizeof(int32_t)), ubBlockSize_);
    const int64_t seqUb = CeilAlign(bsCoreFactor * static_cast<int64_t>(sizeof(int32_t)), ubBlockSize_);
    OPS_ERR_IF(topkUb + statusUb + seqUb >= ubSize_,
        OPS_LOG_E(context_->GetNodeName(), "StageOne UB is insufficient."), return ge::GRAPH_FAILED);
    context_->SetBlockDim(bsCoreNum);
    tilingKey_ = 0;
    return ge::GRAPH_SUCCESS;
}

uint64_t AvsgMatchTiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus AvsgMatchTiling::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OPS_ERR_IF(workspaces == nullptr,
        OPS_LOG_E(context_->GetNodeName(), "get workspaces nullptr."), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc:: PlatformAscendC(context_->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    workspaces[0] = sysWorkspaceSize;
    OPS_ERR_IF(context_->GetRawTilingData() == nullptr,
        OPS_LOG_E(context_->GetNodeName(), "get tiling data nullptr."), return ge::GRAPH_FAILED);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvsgMatchTiling::RunTiling()
{
    ge::graphStatus ret = GetShapeAttrsInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = GetPlatformInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = DoOpTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    return PostTiling();
}

ge::graphStatus Tiling4AvsgMatch(gert::TilingContext* context)
{
    AvsgMatchTiling tiling(context);
    return tiling.RunTiling();
}

ge::graphStatus TilingPrepare4AvsgMatch(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AvsgMatch)
    .Tiling(Tiling4AvsgMatch)
    .TilingParse<AvsgMatchCompileInfo>(TilingPrepare4AvsgMatch);
} // namespace optiling
