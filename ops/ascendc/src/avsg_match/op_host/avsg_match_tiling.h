/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 */

/*!
 * \file avsg_match_tiling.h
 * \brief
 */

#ifndef AVSG_MATCH_TILING_H_
#define AVSG_MATCH_TILING_H_


#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "error/ops_error.h"
#include "platform/platform_info.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(AvsgMatchTilingData)
TILING_DATA_FIELD_DEF(int64_t, stageOneUsedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, stageOneMainCoreBsLoopNum);
TILING_DATA_FIELD_DEF(int64_t, stageOneTailCoreBsLoopNum);
TILING_DATA_FIELD_DEF(int64_t, selTopKBlockSize);
TILING_DATA_FIELD_DEF(int64_t, fullKvBlockSize);
TILING_DATA_FIELD_DEF(int32_t, fullKvBlockSizeLog2);
TILING_DATA_FIELD_DEF(int64_t, selKvBlockSize);
TILING_DATA_FIELD_DEF(int64_t, selKvBlockSizeLog2);
TILING_DATA_FIELD_DEF(int64_t, fullMaxBlockNum);
TILING_DATA_FIELD_DEF(int64_t, selMaxBlockNum);
TILING_DATA_FIELD_DEF(int64_t, batchsize);
TILING_DATA_FIELD_DEF(int64_t, t);
TILING_DATA_FIELD_DEF(int64_t, headnum);
TILING_DATA_FIELD_DEF(int64_t, topk);
TILING_DATA_FIELD_DEF(int64_t, statesize);
TILING_DATA_FIELD_DEF(int64_t, ubSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AvsgMatch, AvsgMatchTilingData)

struct AvsgMatchCompileInfo {
};

class AvsgMatchTiling
{
public:
    explicit AvsgMatchTiling(gert::TilingContext* context) : context_(context)
    {}
    ~AvsgMatchTiling()
    {}
    ge::graphStatus RunTiling();

protected:
    ge::graphStatus DoOpTiling();
    uint64_t GetTilingKey() const;
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus GetShapeAttrsInfo();
    ge::graphStatus PostTiling();

private:
    ge::graphStatus GetInputAttrs();
    ge::graphStatus GetSelBlockTable();
    ge::graphStatus GetTopkIndices();
    ge::graphStatus CheckSelInfo();
    ge::graphStatus GetFullKvBlkTable();
    ge::graphStatus GetSeqLenIn();
    ge::graphStatus GetInputDtypeInfo();

private:
    AvsgMatchTilingData tilingData_;
    int64_t selTopKBlockSize_ = 0;
    int64_t kvBlockSize_ = 0;

    int64_t coreNum_ = 0;
    int64_t ubSize_ = 0;
    int64_t ubBlockSize_ = 0;

    int64_t t_ = 0;
    int64_t batchSize_ = 0;
    int64_t headnum_ = 0;
    int64_t topk_ = 0;
    int64_t stateSize_ = 0;
    int64_t selKvBlockTableRow_ = 0;

    gert::TilingContext *context_ = nullptr;
    uint64_t tilingKey_{0};
};

} // namespace optiling

#endif // AVSG_MATCH_TILING_H_
