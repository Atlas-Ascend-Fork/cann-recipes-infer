/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 */

/*!
 * \file avsg_match.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "avsg_match_split_bs_reuse_vec.h"

using namespace AscendC;
using namespace AvsgMatchNs;
extern "C" __global__ __aicore__ void avsg_match(
    GM_ADDR selection_kv_block_table, GM_ADDR selection_kv_block_status, GM_ADDR selection_topk_indices,
    GM_ADDR full_kv_block_table, GM_ADDR full_kv_actual_seq, GM_ADDR full_q_actual_seq,
    GM_ADDR selection_kv_block_table_out, GM_ADDR selection_kv_block_status_out,
    GM_ADDR selection_kv_actual_seq, GM_ADDR stage_two_copy_mapping, GM_ADDR stage_two_copy_count,
    GM_ADDR selection_topk_indices_out, GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }

    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);

    AvsgMatchSplitBsReuseVec op(&pipe, &tilingData);
    op.Init(selection_kv_block_table, selection_kv_block_status, selection_topk_indices,
        full_kv_block_table, full_kv_actual_seq, full_q_actual_seq, selection_kv_actual_seq,
        stage_two_copy_mapping, stage_two_copy_count);
    op.Process();
}
