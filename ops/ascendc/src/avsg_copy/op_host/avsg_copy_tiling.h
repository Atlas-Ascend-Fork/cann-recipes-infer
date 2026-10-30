/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 */

#ifndef AVSG_COPY_TILING_H_
#define AVSG_COPY_TILING_H_

#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "error/ops_error.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(AvsgCopyTilingData)
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, bshCapacity);
TILING_DATA_FIELD_DEF(int64_t, mappingEntriesPerBsh);
TILING_DATA_FIELD_DEF(int64_t, batchsize);
TILING_DATA_FIELD_DEF(int64_t, headnum);
TILING_DATA_FIELD_DEF(int64_t, fullKvBlockSize);
TILING_DATA_FIELD_DEF(int32_t, fullKvBlockSizeLog2);
TILING_DATA_FIELD_DEF(int64_t, kRopeDim);
TILING_DATA_FIELD_DEF(int64_t, kvCacheDim);
TILING_DATA_FIELD_DEF(int64_t, kRopeUbSize);
TILING_DATA_FIELD_DEF(int64_t, kvCacheUbSize);
TILING_DATA_FIELD_DEF(int64_t, workBufSize);
TILING_DATA_FIELD_DEF(int64_t, buffNum);
TILING_DATA_FIELD_DEF(int64_t, fullKvCacheStride0);
TILING_DATA_FIELD_DEF(int64_t, fullKvCacheStride1);
TILING_DATA_FIELD_DEF(int64_t, fullKRopeStride0);
TILING_DATA_FIELD_DEF(int64_t, fullKRopeStride1);
TILING_DATA_FIELD_DEF(int32_t, hasKRope);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AvsgCopy, AvsgCopyTilingData)
struct AvsgCopyCompileInfo {};

class AvsgCopyTiling {
public:
    explicit AvsgCopyTiling(gert::TilingContext* context) : context_(context) {}
    ge::graphStatus RunTiling();

private:
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus GetShapeAndDtypeInfo();
    ge::graphStatus DoOpTiling();
    ge::graphStatus PostTiling();

    AvsgCopyTilingData tilingData_;
    gert::TilingContext* context_ = nullptr;
    int64_t coreNum_ = 0;
    int64_t ubSize_ = 0;
    int64_t ubBlockSize_ = 32;
    ge::DataType cacheDtype_ = ge::DT_UNDEFINED;
};

} // namespace optiling
#endif // AVSG_COPY_TILING_H_
