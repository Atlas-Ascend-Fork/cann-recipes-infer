/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 */

#include "kernel_operator.h"
#include "avsg_copy.h"

using namespace AscendC;
using namespace AvsgCopyNs;

extern "C" __global__ __aicore__ void avsg_copy(
    GM_ADDR selection_k_rope, GM_ADDR selection_kv_cache, GM_ADDR full_k_rope,
    GM_ADDR full_kv_cache, GM_ADDR full_q_actual_seq, GM_ADDR stage_two_copy_mapping,
    GM_ADDR stage_two_copy_count, GM_ADDR selection_k_rope_out, GM_ADDR selection_kv_cache_out,
    GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }

    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    AvsgCopy<DTYPE_FULL_KV_CACHE> op(&pipe, &tilingData);
    op.Init(selection_k_rope_out, selection_kv_cache_out, full_k_rope, full_kv_cache,
        full_q_actual_seq, stage_two_copy_mapping, stage_two_copy_count);
    op.Process();
}
