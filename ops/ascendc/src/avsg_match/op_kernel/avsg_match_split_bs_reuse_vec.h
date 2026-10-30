/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 */

/*!
 * \file avsg_match_split_bs_reuse_vec.h
 * \brief
 */
#ifndef AVSG_MATCH_SPLIT_BS_REUSE_VEC_H
#define AVSG_MATCH_SPLIT_BS_REUSE_VEC_H

#include "kernel_operator.h"
#include <cstdint>

namespace AvsgMatchNs {
using namespace AscendC;

constexpr int32_t SORT_ELEMENTS_PER_REPEAT = 32;
constexpr int32_t INVALID_SLOT = -1;
constexpr int32_t UB_BLOCK_ALIGN_BYTES = 32;

template <typename U>
__aicore__ inline U CeilAlign(U a, U b)
{
    return (a + b - 1) / b * b;
}

// ============================================================
// TempBufferCursor: UB 临时缓冲区分配器
// ============================================================
struct TempBufferCursor {
    LocalTensor<int32_t> buf;
    int64_t remainingBytes;

    __aicore__ inline TempBufferCursor(LocalTensor<int32_t> buffer, int64_t bytes)
        : buf(buffer), remainingBytes(bytes) {}

    // Allocate count elements with 32-byte UB alignment.
    template<typename T>
    __aicore__ inline LocalTensor<T> Alloc(int64_t count) {
        int64_t bytes = count * sizeof(T);
        bytes = CeilAlign(bytes, static_cast<int64_t>(UB_BLOCK_ALIGN_BYTES));
        ASSERT_MSG(bytes <= remainingBytes,
                   "TempBufferCursor: insufficient bytes, need %ld, have %ld \n",
                   bytes, remainingBytes);
        auto allocated = buf.template ReinterpretCast<T>();
        Advance(bytes);
        return allocated;
    }

    // 创建快照，子函数可安全修改，不影响原 cursor
    __aicore__ inline TempBufferCursor Fork() const {
        return TempBufferCursor(buf, remainingBytes);
    }

    // Reuse scratch at the current cursor without advancing it.
    __aicore__ inline LocalTensor<int32_t>& Current() { return buf; }
    __aicore__ inline const LocalTensor<int32_t>& Current() const { return buf; }

private:
    __aicore__ inline void Advance(int64_t bytes) {
        // 向上对齐到 int32 边界，保证后续分配地址对齐
        int64_t alignedBytes = (bytes + sizeof(int32_t) - 1) / sizeof(int32_t) * sizeof(int32_t);
        buf = buf[alignedBytes / sizeof(int32_t)];
        remainingBytes -= alignedBytes;
    }
};

class AvsgMatchSplitBsReuseVec
{
public:
    __aicore__ inline AvsgMatchSplitBsReuseVec(TPipe* pipe, const AvsgMatchTilingData* tiling)
        : pipe_(pipe), tiling_(tiling)
    {}

    __aicore__ inline void Init(
        GM_ADDR selection_kv_block_table, GM_ADDR selection_kv_block_status, GM_ADDR selection_topk_indices,
        GM_ADDR full_kv_block_table, GM_ADDR full_kv_actual_seq, GM_ADDR full_q_actual_seq,
        GM_ADDR selection_kv_actual_seq, GM_ADDR stage_two_copy_mapping, GM_ADDR stage_two_copy_count)
    {

        blkIdx_ = GetBlockIdx();
        if (blkIdx_ >= tiling_->stageOneUsedCoreNum) {
            return;
        }

        // stage1
        stageOneLoopNum_ = (blkIdx_ == tiling_->stageOneUsedCoreNum - 1) ? tiling_->stageOneTailCoreBsLoopNum : tiling_->stageOneMainCoreBsLoopNum;

        // Apply Local Buff
        selTopKIdxUb_ = CeilAlign(
            static_cast<int64_t>(tiling_->topk * sizeof(int32_t)),
            static_cast<int64_t>(UB_BLOCK_ALIGN_BYTES));
        pipe_->InitBuffer(selTopKIdxQue_, 1, selTopKIdxUb_);

        selKvActSeqUb_ = CeilAlign(static_cast<int64_t>(stageOneLoopNum_ * sizeof(int32_t)), static_cast<int64_t>(UB_BLOCK_ALIGN_BYTES));
        // One batch of status: [statesize, 2] int32 words.
        selBlockStatUb_ = CeilAlign(
            static_cast<int64_t>(tiling_->statesize * sizeof(uint64_t)),
            static_cast<int64_t>(UB_BLOCK_ALIGN_BYTES));
        offsetOutputUb_ = CeilAlign(
            static_cast<int64_t>(tiling_->topk  * sizeof(uint64_t)),
            static_cast<int64_t>(UB_BLOCK_ALIGN_BYTES));

        int64_t totalUb = CeilAlign(static_cast<int64_t> (tiling_->ubSize - UB_BLOCK_ALIGN_BYTES), static_cast<int64_t>(UB_BLOCK_ALIGN_BYTES));
        tmpbuffUb_ = CeilAlign(static_cast<int64_t>(totalUb - selTopKIdxUb_ - selKvActSeqUb_), static_cast<int64_t>(UB_BLOCK_ALIGN_BYTES)); // Bytes

        pipe_->InitBuffer(workBuf_, selKvActSeqUb_ + tmpbuffUb_); // Per-core sequence output followed by reusable status and scratch buffers.

        // For GM_Address
        // [batchsize, statesize * 2], initialized to -1 for an empty cache.
        selKvBlockStatusGm_.SetGlobalBuffer(
            (__gm__ int32_t*)selection_kv_block_status + blkIdx_ * tiling_->stageOneMainCoreBsLoopNum * tiling_->statesize * HASH_ENTRY_FIELDS);
        // [batchsize, seq, headnum, topk] token粒度=groupid * selection_topk_block_size
        selTopKIndicesGm_.SetGlobalBuffer((__gm__ int32_t*)selection_topk_indices);

        // [batchsize] 每个batch实际的seq长度
        fullKvActualSeqGm_.SetGlobalBuffer((__gm__ int32_t*)full_kv_actual_seq + blkIdx_ * tiling_->stageOneMainCoreBsLoopNum);
        // [batchsize] Cumulative query lengths (TND batch boundaries).
        fullQActualSeqGm_.SetGlobalBuffer((__gm__ int32_t*)full_q_actual_seq);

        // [batchsize] Reused token counts accumulated across queries.
        selKvActualSeqGm_.SetGlobalBuffer((__gm__ int32_t*)selection_kv_actual_seq + blkIdx_ * tiling_->stageOneMainCoreBsLoopNum);
        // [query_token, head, aligned_topk, 2], each valid entry is {src_token, dst_token}.
        stageOneCopyMappingGM_.SetGlobalBuffer((__gm__ int32_t*)stage_two_copy_mapping);
        // [query_token, head], only the active prefix selected by full_q_actual_seq is consumed.
        mismatchNumArrayGm_.SetGlobalBuffer((__gm__ int32_t*)stage_two_copy_count);

        selKvBlockTableGm_.SetGlobalBuffer((__gm__ int32_t*)selection_kv_block_table);
        fullKvBlockTableGm_.SetGlobalBuffer((__gm__ int32_t*)full_kv_block_table);
    }

    __aicore__ inline void Process()
    {
        if (blkIdx_ >= tiling_->stageOneUsedCoreNum) {
            return;
        }

        StageOne();
    }

private:
    template <HardEvent event>
    __aicore__ inline void SetWaitFlag(HardEvent evt)
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
        SetFlag<event>(eventId);
        WaitFlag<event>(eventId);
    }

    // ============================================================
    // 哈希表字段访问器
    // ============================================================
    // word0: signed record[31:24] | unsigned tokenId[23:0].
    // word1: fingerprint | cacheSlot; independent of the cache record.
    // Token 0xFFFFFF is the invalid sentinel; decode it back to -1.
    __aicore__ inline void ExtractRecord(LocalTensor<int32_t>& dst,
        LocalTensor<int32_t>& packedIdx, int32_t nums)
    {
        AscendC::ShiftRight(dst, packedIdx, RECORD_SHIFT_BITS, nums);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StripRecord(LocalTensor<int32_t>& dst,
        LocalTensor<int32_t>& packedIdx, int32_t nums)
    {
        AscendC::ShiftLeft(dst, packedIdx, RECORD_BIT_WIDTH, nums);
        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<uint32_t> tmp = dst.template ReinterpretCast<uint32_t>();
        AscendC::ShiftRight(tmp, tmp, static_cast<uint32_t>(RECORD_BIT_WIDTH), nums);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void PackEntry(LocalTensor<int32_t>& index,
        LocalTensor<int32_t>& record, int32_t nums)
    {
        AscendC::ShiftLeft(record, record, RECORD_SHIFT_BITS, nums);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Or(index.template ReinterpretCast<uint16_t>(),
            index.template ReinterpretCast<uint16_t>(),
            record.template ReinterpretCast<uint16_t>(),
            nums * sizeof(int32_t) / sizeof(uint16_t));
        AscendC::PipeBarrier<PIPE_V>();
    }

    // ============================================================
    // 模运算
    // ============================================================

    // 矢量重载 — TempBufferCursor 入参 → PIPE_V
    __aicore__ inline void BitwiseModPowerOfTwo(LocalTensor<int32_t>& vector,
        int32_t modValue, TempBufferCursor cursor, int32_t nums)
    {
        LocalTensor<int32_t> modMaskLocal = cursor.Alloc<int32_t>(nums);
        AscendC::Duplicate(modMaskLocal, static_cast<int32_t>(modValue - 1), nums);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::And(vector.template ReinterpretCast<uint16_t>(),
            vector.template ReinterpretCast<uint16_t>(),
            modMaskLocal.template ReinterpretCast<uint16_t>(),
            nums * sizeof(int32_t) / sizeof(uint16_t));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ExtractTopkIndex(LocalTensor<int32_t>& dst,
        LocalTensor<int32_t>& packedIdx, TempBufferCursor cursor, int32_t nums)
    {
        LocalTensor<int32_t> sign = cursor.Alloc<int32_t>(nums);
        LocalTensor<int32_t> stateMask = cursor.Alloc<int32_t>(nums);
        AscendC::Duplicate(
            stateMask, static_cast<int32_t>(tiling_->statesize - 1), nums);
        AscendC::PipeBarrier<PIPE_V>();
        StripRecord(dst, packedIdx, nums);
        AscendC::ShiftRight(sign, dst, 31, nums);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::And(dst.template ReinterpretCast<uint16_t>(),
            dst.template ReinterpretCast<uint16_t>(),
            stateMask.template ReinterpretCast<uint16_t>(),
            nums * sizeof(int32_t) / sizeof(uint16_t));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Or(dst.template ReinterpretCast<uint16_t>(),
            dst.template ReinterpretCast<uint16_t>(),
            sign.template ReinterpretCast<uint16_t>(),
            nums * sizeof(int32_t) / sizeof(uint16_t));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeFingerprintFromTopkId(
        LocalTensor<int32_t>& result, LocalTensor<int32_t>& topkId,
        TempBufferCursor cursor, bool copyInput, int32_t nums)
    {
        if (copyInput) {
            AscendC::Adds(result, topkId, 0, nums);
            AscendC::PipeBarrier<PIPE_V>();
        }
        int32_t fingerprintMod = HASH_INDEX_SIGN_BIT / tiling_->statesize;
        BitwiseModPowerOfTwo(result, fingerprintMod, cursor.Fork(), nums);
        AscendC::Muls(result, result,
            static_cast<int32_t>(tiling_->statesize), nums);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void EncodeFingerprintIntoPackedIndex(
        LocalTensor<int32_t>& packedIdx, LocalTensor<int32_t>& topkId,
        TempBufferCursor cursor, int32_t nums)
    {
        LocalTensor<int32_t> fingerprint = cursor.Alloc<int32_t>(nums);
        ComputeFingerprintFromTopkId(
            fingerprint, topkId, cursor.Fork(), true, nums);
        AscendC::Or(packedIdx.template ReinterpretCast<uint16_t>(),
            packedIdx.template ReinterpretCast<uint16_t>(),
            fingerprint.template ReinterpretCast<uint16_t>(),
            nums * sizeof(int32_t) / sizeof(uint16_t));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ExtractStoredFingerprint(
        LocalTensor<int32_t>& fingerprint, LocalTensor<int32_t>& packedEntry,
        TempBufferCursor cursor, int32_t nums)
    {
        LocalTensor<int32_t> slotPart = cursor.Alloc<int32_t>(nums);
        StripRecord(fingerprint, packedEntry, nums);
        AscendC::Adds(slotPart, fingerprint, 0, nums);
        AscendC::PipeBarrier<PIPE_V>();
        BitwiseModPowerOfTwo(slotPart, tiling_->statesize, cursor.Fork(), nums);
        AscendC::Sub(fingerprint, fingerprint, slotPart, nums);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void GetOutdatedMask(LocalTensor<uint8_t>& outdatedMask,
        LocalTensor<int32_t>& packedEntry,
        LocalTensor<int32_t>& hashMapping, TempBufferCursor cursor, int32_t nums)
    {
        LocalTensor<int32_t> storedFingerprint = cursor.Alloc<int32_t>(nums);
        LocalTensor<int32_t> expectedFingerprint = cursor.Alloc<int32_t>(nums);

        ExtractStoredFingerprint(
            storedFingerprint, packedEntry, cursor.Fork(), nums);

        ExtractTopkIndex(
            expectedFingerprint, packedEntry, cursor.Fork(), nums);
        GatherRecordIdFromHash(expectedFingerprint, expectedFingerprint, hashMapping, cursor.Fork(), nums);
        ComputeFingerprintFromTopkId(expectedFingerprint, expectedFingerprint,
            cursor.Fork(), false, nums);
        CompareNe(outdatedMask, storedFingerprint, expectedFingerprint, nums);
    }

    // ============================================================
    // 哈希位置计算
    // ============================================================

    __aicore__ inline void ComputeHashPosition(LocalTensor<int32_t>& hashId,
        LocalTensor<int32_t>& topkId, TempBufferCursor cursor, int32_t nums)
    {
        AscendC::Muls(hashId, topkId, HASH_PRIME_MULTIPLIER, nums);
        AscendC::PipeBarrier<PIPE_V>();
        BitwiseModPowerOfTwo(hashId, tiling_->statesize, cursor.Fork(), nums);
    }

    // ============================================================
    // 哈希步长计算
    // ============================================================

    __aicore__ inline void ComputeHashStride(LocalTensor<int32_t>& stride,
        LocalTensor<int32_t>& topkId, TempBufferCursor cursor, int32_t nums)
    {
        AscendC::Muls(stride, topkId, HASH_STRIDE_MULTIPLIER, nums);
        AscendC::PipeBarrier<PIPE_V>();
        BitwiseModPowerOfTwo(stride, tiling_->statesize, cursor.Fork(), nums);
        // Clear the lowest bit of the masked hash stride.
        AscendC::ShiftRight(stride.template ReinterpretCast<uint32_t>(),
            stride.template ReinterpretCast<uint32_t>(), static_cast<uint32_t>(1), nums);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftLeft(stride.template ReinterpretCast<uint32_t>(),
            stride.template ReinterpretCast<uint32_t>(), static_cast<uint32_t>(1), nums);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(stride, stride, 1, nums);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline int64_t GetActualQSeqLen(int64_t bsIdx)
    {
        return fullQActualSeqGm_.GetValue(bsIdx);
    }

    __aicore__ inline void CopyInTopKIndices(int64_t globalOffset)
    {
        int64_t gmOffset = globalOffset * tiling_->topk;
        LocalTensor<int32_t> selTopKIdxLocal = selTopKIdxQue_.AllocTensor<int32_t>();
        DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
        DataCopyExtParams dataCopyParams{
            static_cast<uint16_t>(1), static_cast<uint32_t>(tiling_->topk * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(selTopKIdxLocal, selTopKIndicesGm_[gmOffset], dataCopyParams, dataCopyPadParams);
        selTopKIdxQue_.EnQue(selTopKIdxLocal);
    }

    __aicore__ inline void CopyInBlockStatus(int64_t loopIdx, LocalTensor<int32_t>& selBlockStatLocal)
    {
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
        DataCopyExtParams dataCopyParamsSt{
            static_cast<uint16_t>(1), static_cast<uint32_t>(selBlockStatUb_), 0, 0, 0};
        DataCopyPad(selBlockStatLocal, selKvBlockStatusGm_[loopIdx * selBlockStatUb_/sizeof(int32_t)], dataCopyParamsSt, dataCopyPadParams);
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    }

    __aicore__ inline void LoadBlockTables(TempBufferCursor& cursor, int32_t globalBatchId,
                                           LocalTensor<int32_t>& localFullBlkTable,
                                           LocalTensor<int32_t>& localSelBlkTable)
    {
        int64_t fullBlkTableSize = CeilAlign(tiling_->fullMaxBlockNum, static_cast<int64_t>(UB_BLOCK_ALIGN_BYTES));
        int64_t selBlkTableSize = CeilAlign(tiling_->selMaxBlockNum, static_cast<int64_t>(UB_BLOCK_ALIGN_BYTES));
        localFullBlkTable = cursor.Alloc<int32_t>(fullBlkTableSize);
        localSelBlkTable = cursor.Alloc<int32_t>(selBlkTableSize);

        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyExtParams fullCopyParams{
            static_cast<uint16_t>(1),
            static_cast<uint32_t>(tiling_->fullMaxBlockNum * sizeof(int32_t)),
            0, 0, 0};
        DataCopyPad(localFullBlkTable,
                    fullKvBlockTableGm_[globalBatchId * tiling_->fullMaxBlockNum],
                    fullCopyParams, padParams);
        DataCopyExtParams selCopyParams{
            static_cast<uint16_t>(1),
            static_cast<uint32_t>(tiling_->selMaxBlockNum * sizeof(int32_t)),
            0, 0, 0};
        DataCopyPad(localSelBlkTable,
                    selKvBlockTableGm_[globalBatchId * tiling_->selMaxBlockNum],
                    selCopyParams, padParams);
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    }

    __aicore__ inline void GatherRecordIndexFromHash(LocalTensor<int32_t>& hashRecordIdx, LocalTensor<int32_t>& hashIdx, LocalTensor<int32_t>& hashMapping, TempBufferCursor cursor, int32_t nums){

        // [idx* 2 + 1]
        LocalTensor<int32_t> topkidindices = cursor.Alloc<int32_t>(nums);

        // hashIdx: >0, byte offset = hashIdx * 8 + 4 (entry stride=8B, record_index at +4)
        AscendC::Maxs(topkidindices, hashIdx, 0, nums);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(topkidindices, topkidindices, static_cast<int32_t>(HASH_ENTRY_FIELDS * sizeof(int32_t)), nums);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(topkidindices, topkidindices, static_cast<int32_t>(sizeof(int32_t)), nums);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(hashRecordIdx, hashMapping, topkidindices.template ReinterpretCast<uint32_t>(), (uint32_t)0, nums);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void GatherPackedTokenFromHash(LocalTensor<int32_t>& hashRecordId, LocalTensor<int32_t>& hashIdx, LocalTensor<int32_t>& hashMapping, TempBufferCursor cursor, int32_t nums){

        LocalTensor<int32_t> topkidindices = cursor.Alloc<int32_t>(nums);
        AscendC::Maxs(topkidindices, hashIdx, 0, nums);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(topkidindices, topkidindices, static_cast<int32_t> (HASH_ENTRY_FIELDS * sizeof(int32_t)), nums);
        AscendC::PipeBarrier<PIPE_V>(); // 偏移的单位为Bytes, 每个元素占4个Byte,还得乘4
        AscendC::Gather(hashRecordId, hashMapping, topkidindices.template ReinterpretCast<uint32_t>(), (uint32_t)0, nums);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void DecodeTokenId(LocalTensor<int32_t>& dst,
        LocalTensor<int32_t>& packedToken, int32_t nums)
    {
        // ((low24 + 1) & 0xFFFFFF) - 1 preserves unsigned token IDs
        // and maps the reserved 0xFFFFFF code to -1 without scalar work.
        StripRecord(dst, packedToken, nums);
        AscendC::Adds(dst, dst, 1, nums);
        AscendC::PipeBarrier<PIPE_V>();
        StripRecord(dst, dst, nums);
        AscendC::Adds(dst, dst, -1, nums);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void GatherRecordIdFromHash(LocalTensor<int32_t>& tokenId,
        LocalTensor<int32_t>& slot, LocalTensor<int32_t>& table,
        TempBufferCursor cursor, int32_t nums)
    {
        GatherPackedTokenFromHash(tokenId, slot, table, cursor, nums);
        DecodeTokenId(tokenId, tokenId, nums);
    }

    __aicore__ inline void CompareEq(LocalTensor<uint8_t>& result, LocalTensor<int32_t>& hashRecordId, LocalTensor<int32_t>& selTopKIdxLocal, int32_t nums){
        AscendC::Duplicate(result.template ReinterpretCast<int32_t>(), 0, nums/AscendC::AscendCUtils::GetBitSize(sizeof(int32_t)));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Compare(result, hashRecordId, selTopKIdxLocal, AscendC::CMPMODE::EQ, nums);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CompareNe(LocalTensor<uint8_t>& result, LocalTensor<int32_t>& hashRecordId, LocalTensor<int32_t>& selTopKIdxLocal, int32_t nums){
        CompareEq(result, hashRecordId, selTopKIdxLocal, nums);
        AscendC::Not(result.template ReinterpretCast<uint16_t>(), result.template ReinterpretCast<uint16_t>(), nums/AscendC::AscendCUtils::GetBitSize(sizeof(uint16_t)));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CompareEq(LocalTensor<uint8_t>& result, LocalTensor<int32_t>& hashRecordId, int32_t src1Scalar, int32_t nums){
        AscendC::Duplicate(result.template ReinterpretCast<int32_t>(), 0, nums/AscendC::AscendCUtils::GetBitSize(sizeof(int32_t)));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::CompareScalar(result, hashRecordId, src1Scalar, AscendC::CMPMODE::EQ, nums);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CompareNe(LocalTensor<uint8_t>& result, LocalTensor<int32_t>& hashRecordId, int32_t src1Scalar, int32_t nums){
        CompareEq(result, hashRecordId, src1Scalar, nums);
        AscendC::Not(result.template ReinterpretCast<uint16_t>(), result.template ReinterpretCast<uint16_t>(), nums/AscendC::AscendCUtils::GetBitSize(sizeof(uint16_t)));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void isNegative(LocalTensor<uint8_t>& result, LocalTensor<int32_t>& value, TempBufferCursor cursor, bool reuseBuf, int32_t nums){
        LocalTensor<int32_t> vector = cursor.Alloc<int32_t>(nums);
        LocalTensor<int32_t> src = reuseBuf? value : cursor.Alloc<int32_t>(nums);
        AscendC::ShiftRight(src, value, (AscendC::AscendCUtils::GetBitSize(sizeof(int32_t)) - 1), nums);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(vector, INVALID_SLOT, nums);
        AscendC::PipeBarrier<PIPE_V>();
        CompareEq(result, src, vector, nums);
        AscendC::PipeBarrier<PIPE_V>();
    }

    struct HashMaskType {
        bool match;
        bool empty;
    };

    __aicore__ inline void ComputeHashMatchMask(LocalTensor<uint8_t> result, LocalTensor<int32_t>& hashIdx, LocalTensor<int32_t>& hashMapping, LocalTensor<int32_t>& selTopKIdxLocal, TempBufferCursor cursor, int32_t nums){
        LocalTensor<int32_t> hashRecordId = cursor.Alloc<int32_t>(nums);
        LocalTensor<int32_t> gatherResult = cursor.Alloc<int32_t>(nums);
        AscendC::Duplicate(hashRecordId, INVALID_SLOT, nums);
        AscendC::Duplicate(gatherResult, INVALID_SLOT, nums);
        AscendC::PipeBarrier<PIPE_V>();

        GatherRecordIndexFromHash(hashRecordId, hashIdx, hashMapping, cursor.Fork(), nums);
        ExtractTopkIndex(
            hashRecordId, hashRecordId, cursor.Fork(), nums);
        CompareNe(result, hashRecordId, gatherResult, nums);

        LocalTensor<uint8_t> mask = cursor.Alloc<uint8_t>(nums);
        GatherRecordIdFromHash(hashRecordId, hashRecordId, hashMapping, cursor.Fork(), nums);
        CompareEq(mask, hashRecordId, selTopKIdxLocal, nums);

        // Combine the valid-slot mask with the token-match mask.
        AscendC::And(result.template ReinterpretCast<uint16_t>(), result.template ReinterpretCast<uint16_t>(), mask.template ReinterpretCast<uint16_t>(), nums / AscendC::AscendCUtils::GetBitSize(sizeof(uint16_t)));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeHashEmptyMask(LocalTensor<uint8_t> result, LocalTensor<int32_t>& hashIdx, LocalTensor<int32_t>& hashMapping, TempBufferCursor cursor, int32_t nums){
        LocalTensor<int32_t> recordLocal = cursor.Alloc<int32_t>(nums);
        GatherRecordIndexFromHash(recordLocal, hashIdx, hashMapping, cursor.Fork(), nums);

        LocalTensor<uint8_t> mask = cursor.Alloc<uint8_t>(nums);

        GetOutdatedMask(result, recordLocal, hashMapping, cursor.Fork(), nums);

        ExtractTopkIndex(recordLocal, recordLocal, cursor.Fork(), nums);
        isNegative(mask, recordLocal, cursor.Fork(), false, nums); // Index 是否小于0， 小于0也为空，
        AscendC::Or(result.template ReinterpretCast<uint16_t>(), result.template ReinterpretCast<uint16_t>(), mask.template ReinterpretCast<uint16_t>(), nums / AscendC::AscendCUtils::GetBitSize(sizeof(uint16_t)));
        AscendC::PipeBarrier<PIPE_V>();

        GatherPackedTokenFromHash(recordLocal, recordLocal, hashMapping, cursor.Fork(), nums); // Cache lifetime is in word0.
        ExtractRecord(recordLocal, recordLocal, nums);
        isNegative(mask, recordLocal, cursor.Fork(), false, nums);
        AscendC::Or(result.template ReinterpretCast<uint16_t>(), result.template ReinterpretCast<uint16_t>(), mask.template ReinterpretCast<uint16_t>(), nums / AscendC::AscendCUtils::GetBitSize(sizeof(uint16_t)));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void IgnoreInvalidTopk(LocalTensor<uint8_t> result, LocalTensor<int32_t>& selTopKIdxLocal, TempBufferCursor cursor, int32_t nums){
        LocalTensor<uint8_t> validMask = cursor.Alloc<uint8_t>(nums);
        CompareNe(validMask, selTopKIdxLocal, INVALID_SLOT, nums);
        AscendC::And(result.template ReinterpretCast<uint16_t>(), result.template ReinterpretCast<uint16_t>(), validMask.template ReinterpretCast<uint16_t>(), nums / AscendC::AscendCUtils::GetBitSize(sizeof(uint16_t)));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeHashMask(LocalTensor<uint8_t> result, LocalTensor<int32_t>& hashIdx, LocalTensor<int32_t>& hashMapping, LocalTensor<int32_t>& selTopKIdxLocal, TempBufferCursor cursor, int32_t nums, HashMaskType maskType){
        if (maskType.match) {
            ComputeHashMatchMask(result, hashIdx, hashMapping, selTopKIdxLocal, cursor, nums);
        } else if (maskType.empty) {
            ComputeHashEmptyMask(result, hashIdx, hashMapping, cursor, nums);
        }
    }

    __aicore__ inline void AdvanceHashProbe(LocalTensor<int32_t>& hashId, LocalTensor<uint8_t> mask, LocalTensor<int32_t>& stride, TempBufferCursor cursor, int32_t modValue, int32_t nums){

        // Skip first nums int32 (for external use)
        cursor.Alloc<int32_t>(nums);

        // offset = zeros; offset[!mask] = stride[!mask]
        LocalTensor<float> offset = cursor.Alloc<float>(nums);
        AscendC::Duplicate(offset, static_cast<float> (0), nums);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Select(offset, mask, offset, stride.template ReinterpretCast<float>(), AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, nums); // 1 选择offset， 0 选择floatStride
        AscendC::PipeBarrier<PIPE_V>();

        // Select already preserved the int32 stride bit pattern. Reinterpret it
        // directly; a numeric float-to-int cast collapses every positive stride
        // to one and silently turns double hashing into linear probing.
        LocalTensor<int32_t> int32offset = offset.template ReinterpretCast<int32_t>();
        AscendC::Add(hashId, hashId, int32offset, nums);
        AscendC::PipeBarrier<PIPE_V>();

        BitwiseModPowerOfTwo(hashId, modValue, cursor.Fork(), nums);
    }

    template <typename T>
    __aicore__ inline void Scatter(LocalTensor<T>& value, LocalTensor<int32_t>& indicesLocal,
        LocalTensor<T>& dstLocal, uint64_t validNum)
    {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        if (validNum == 0) {
            return;
        }
        constexpr uint32_t shift = sizeof(T) == sizeof(int64_t) ? 3 : 2;
        auto offsets = indicesLocal.template ReinterpretCast<uint32_t>();
        AscendC::ShiftLeft(offsets, offsets, shift, static_cast<int32_t>(validNum));
        AscendC::PipeBarrier<PIPE_V>();
        // Submit destinations directly to SIMD Scatter in this architecture branch.
        AscendC::Scatter(dstLocal, value, offsets, 0, static_cast<uint32_t>(validNum));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftRight(offsets, offsets, shift, static_cast<int32_t>(validNum));
        AscendC::PipeBarrier<PIPE_V>();
#else
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        for (uint32_t i = 0; i < validNum; ++i) {
            dstLocal.SetValue(indicesLocal.GetValue(i), value.GetValue(i));
        }
        SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
#endif
    }

    __aicore__ inline void ProcessMissMaskAndUpdateMatchTopkId(LocalTensor<uint8_t> result, LocalTensor<int32_t>& selTopKIdLocal, LocalTensor<int32_t>& hashMapping, TempBufferCursor cursor, uint64_t& mismatchNum, uint64_t& matchNum, int32_t totalNum){

        HashMaskType maskType{true, false};  // match mode
        LocalTensor<int32_t> hashLocal = cursor.Alloc<int32_t>(totalNum);
        LocalTensor<uint8_t> matchMask = cursor.Alloc<uint8_t>(totalNum);
        ProbeHashTable(hashLocal, matchMask, selTopKIdLocal, hashMapping, cursor.Fork(), totalNum, maskType);

        LocalTensor<int32_t> recordIndex = cursor.Alloc<int32_t>(chunkSize_);
        // Update HashStatus By Match Mask
        uint64_t validNum = 0;
        GatherByBitMask(recordIndex, matchMask, hashLocal, validNum, totalNum);
        GatherRecordIndexFromHash(recordIndex, hashLocal, hashMapping, cursor.Fork(), totalNum); // Reused Ub_ chunkTopkId
        ExtractTopkIndex(recordIndex, recordIndex, cursor.Fork(), totalNum);

        // Update Match SelTopkId — V pipe Select replaces S pipe Scatter
        LocalTensor<int32_t> tmpBuf = cursor.Alloc<int32_t>(totalNum);
        LocalTensor<int32_t> matchIndices= recordIndex; //Reused Ub_
        AscendC::Select(selTopKIdLocal.template ReinterpretCast<float>(), matchMask, recordIndex.template ReinterpretCast<float>(), selTopKIdLocal.template ReinterpretCast<float>(), AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, totalNum);
        AscendC::PipeBarrier<PIPE_V>();
        matchNum += validNum;

        // GetMissMatchMask
        AscendC::Not(result.template ReinterpretCast<uint16_t>(), matchMask.template ReinterpretCast<uint16_t>(), totalNum/AscendC::AscendCUtils::GetBitSize(sizeof(uint16_t)));
        AscendC::PipeBarrier<PIPE_V>();
        IgnoreInvalidTopk(result, selTopKIdLocal, cursor.Fork(), totalNum);
        PipeBarrier<PIPE_ALL>();
        AscendC::CreateVecIndex(tmpBuf, 0, totalNum);
        PipeBarrier<PIPE_V>();
        GatherByBitMask(matchIndices, result, tmpBuf, mismatchNum, totalNum);
    }

    __aicore__ inline void RefreshMatchStatus(LocalTensor<int32_t>& hashMapping, TempBufferCursor cursor, int32_t globalTOffset, int32_t numSeqs){

        HashMaskType maskType{true, false};  // match mode
        for(int32_t seqId = 0; seqId < numSeqs; seqId++){
            TempBufferCursor chunkCursor = cursor.Fork();
            int32_t globalIndex = (globalTOffset+seqId) * tiling_->headnum;
            CopyInTopKIndices(globalIndex);
            LocalTensor<int32_t> selTopKIdLocal = selTopKIdxQue_.DeQue<int32_t>();

            LocalTensor<uint8_t> matchMask = chunkCursor.Alloc<uint8_t>(tiling_->topk);

            LocalTensor<int32_t> chunkHashLocal = chunkCursor.Alloc<int32_t>(tiling_->topk);
            LocalTensor<int32_t> recordIndex = chunkCursor.Alloc<int32_t>(tiling_->topk);
            ProbeHashTable(chunkHashLocal, matchMask, selTopKIdLocal, hashMapping, chunkCursor.Fork(), tiling_->topk, maskType);
            // Update HashStatus By Match Mask
            uint64_t validNum = 0;
            GatherRecordIndexFromHash(recordIndex, chunkHashLocal, hashMapping, chunkCursor.Fork(), tiling_->topk); // Reused Ub_ chunkTopkId
            // Get Indices
            LocalTensor<int32_t> matchTopkIdx = chunkCursor.Alloc<int32_t>(tiling_->topk);
            GatherByBitMask(matchTopkIdx, matchMask, recordIndex, validNum, tiling_->topk);
            ExtractTopkIndex(matchTopkIdx, matchTopkIdx,
                chunkCursor.Fork(), validNum);
            RefreshRecord(matchTopkIdx, hashMapping, chunkCursor.Fork(), static_cast<int32_t>(MAX_RECORD_LIFETIME), validNum);
            selTopKIdxQue_.FreeTensor(selTopKIdLocal);
        }

    }

    __aicore__ inline void ProbeHashTable(LocalTensor<int32_t>& hashId, LocalTensor<uint8_t> mask, LocalTensor<int32_t>& selTopKIdLocal, LocalTensor<int32_t>& hashMapping, TempBufferCursor cursor, int32_t nums, HashMaskType maskType){

        int32_t mod_value = tiling_->statesize;
        ComputeHashPosition(hashId, selTopKIdLocal, cursor.Fork(), nums);
        LocalTensor<int32_t> stride = cursor.Alloc<int32_t>(nums);
        ComputeHashStride(stride, selTopKIdLocal, cursor.Fork(), nums);
        // Update hashIdx
        for (int64_t retry = 0; retry < MAX_PROBE_RETRIES; ++retry) {
            ComputeHashMask(mask, hashId, hashMapping, selTopKIdLocal, cursor.Fork(), nums, maskType);
            AdvanceHashProbe(hashId, mask, stride, cursor.Fork(), mod_value, nums);
        }
        ComputeHashMask(mask, hashId, hashMapping, selTopKIdLocal, cursor.Fork(), nums, maskType);

        IgnoreInvalidTopk(mask, selTopKIdLocal, cursor.Fork(), nums);
    }

    __aicore__ inline void GatherByBitMask(LocalTensor<int32_t>& result, LocalTensor<uint8_t>& mask, LocalTensor<int32_t>& src, uint64_t& validNum, int32_t totalNum){
        uint8_t src0BlockStride = 1; // 每个Block内的元素偏移
        uint16_t repeatTimes = 1; //总共totalNum个元素，共需要处理多少次
        uint16_t src0RepeatStride = 1; // 64个元素的 int32_t 是256字节，每个blocksize是32， 256个字节就是占了8个blocks，
        uint8_t src1RepeatStride = 1;//每次处理64个元素, pattern占了64位， 一共是两个blocks
        AscendC::GatherMask(result, src, mask.template ReinterpretCast<uint32_t>(), true,  static_cast<uint32_t>(totalNum), { src0BlockStride, repeatTimes, src0RepeatStride, src1RepeatStride}, validNum);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void DecrementRecordCounters(LocalTensor<int32_t>& statusLocal, TempBufferCursor cursor, int32_t decayValue, int64_t totalNum){
        int64_t chunkNum = totalNum/chunkSize_;
        for(int64_t chunkId = 0; chunkId < chunkNum; chunkId++){
            DecrementRecordCountersPerChunk(statusLocal, cursor.Fork(), decayValue, chunkId, chunkSize_);
        }
    }

    __aicore__ inline void DecrementRecordCountersPerChunk(LocalTensor<int32_t>& statusLocal, TempBufferCursor cursor, int32_t decayValue, int64_t chunkId, int64_t totalNum){

        uint64_t stride = sizeof(int64_t) / sizeof(int32_t);
        totalNum *= stride;
        LocalTensor<int32_t> result = statusLocal[chunkId * totalNum];
        LocalTensor<int32_t> value = cursor.Alloc<int32_t>(totalNum);
        LocalTensor<int32_t> record = cursor.Alloc<int32_t>(totalNum);
        AscendC::PipeBarrier<PIPE_ALL>();
        ArithProgression<int32_t>(value, 0, 1, totalNum);
        AscendC::PipeBarrier<PIPE_V>();
        BitwiseModPowerOfTwo(value, HASH_ENTRY_FIELDS, cursor.Fork(), totalNum); // 0,1,0,1,0,1,0,1,0,1,0,1,0,1,.....,0,1

        AscendC::Adds(value, value, -1, totalNum); // -1,0 selects cache words only
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(value, value, -decayValue, totalNum);
        AscendC::PipeBarrier<PIPE_V>();

        ExtractRecord(record, result, totalNum);
        StripRecord(result, result, totalNum);

        AscendC::Sub(record, record, value, totalNum);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Maxs(record, record, INVALID_SLOT-1, totalNum);
        AscendC::PipeBarrier<PIPE_V>();

        PackEntry(result, record, totalNum);

    }

    __aicore__ inline void InterleaveTensors(LocalTensor<int32_t>& result, LocalTensor<int32_t>& topkId, LocalTensor<int32_t>& emptyIdx, TempBufferCursor cursor, uint64_t uniqueNum, uint64_t totalNum){
        // TODO(kuangwenwei): 后续可以作均匀切分，使得更加均匀
        // Scatter: Gather+Selection
        // UpdateCopyMapping
        uint64_t numTensors = 2;
        uniqueNum *= numTensors;
        totalNum *= numTensors;
        AscendC::LocalTensor<int32_t> indexLocal = cursor.Alloc<int32_t>(totalNum);
        PipeBarrier<PIPE_ALL>();
        AscendC::CreateVecIndex(indexLocal, 0, totalNum);
        PipeBarrier<PIPE_V>();
        BitwiseModPowerOfTwo(indexLocal, HASH_ENTRY_FIELDS, cursor.Fork(), totalNum); // 0,1,0,1,0,1,0,1,0,1,0,1,0,1,.....,0,1
        AscendC::LocalTensor<int32_t> tmpbuffIndex = cursor.Alloc<int32_t>(totalNum);
        AscendC::Duplicate(tmpbuffIndex, 1, totalNum);
        PipeBarrier<PIPE_V>();
        LocalTensor<uint8_t> mask = cursor.Alloc<uint8_t>(totalNum / AscendC::AscendCUtils::GetBitSize(sizeof(uint8_t)));
        CompareEq(mask, indexLocal, tmpbuffIndex, totalNum); // cur != prev
        PipeBarrier<PIPE_ALL>();
        AscendC::CreateVecIndex(indexLocal, 0, uniqueNum);
        PipeBarrier<PIPE_V>();
        AscendC::ShiftRight(indexLocal, indexLocal, 1, uniqueNum); //Divs_(2)
        PipeBarrier<PIPE_V>();
        AscendC::Muls(indexLocal, indexLocal, static_cast<int32_t>(sizeof(int32_t)), uniqueNum);
        PipeBarrier<PIPE_V>();
        AscendC::Gather(result, topkId, indexLocal.template ReinterpretCast<uint32_t>(), (uint32_t)0, uniqueNum);
        PipeBarrier<PIPE_V>();
        AscendC::Gather(tmpbuffIndex, emptyIdx, indexLocal.template ReinterpretCast<uint32_t>(), (uint32_t)0, uniqueNum);
        PipeBarrier<PIPE_V>();
        AscendC::Adds(indexLocal, tmpbuffIndex, 0, uniqueNum);
        PipeBarrier<PIPE_V>();

        AscendC::Select(result.template ReinterpretCast<float>(), mask, indexLocal.template ReinterpretCast<float>(), result.template ReinterpretCast<float>(), AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,  uniqueNum);   // true: tmpbuffIndex, false: result
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void RefreshRecord(LocalTensor<int32_t>& emptyIdx, LocalTensor<int32_t>& statusLocal, TempBufferCursor cursor, int32_t refreshValue, uint64_t totalNum){
        LocalTensor<int32_t> value = cursor.Alloc<int32_t>(CeilAlign(static_cast<int32_t>(totalNum), SORT_ELEMENTS_PER_REPEAT));
        LocalTensor<int32_t> key = cursor.Alloc<int32_t>(CeilAlign(static_cast<int32_t>(totalNum), SORT_ELEMENTS_PER_REPEAT));
        GatherPackedTokenFromHash(value, emptyIdx, statusLocal, cursor.Fork(), totalNum);
        StripRecord(value, value, totalNum);
        AscendC::Adds(value, value, refreshValue << RECORD_SHIFT_BITS, totalNum); // 刷新record
        // Cache word offset: slot * 2
        AscendC::Muls(key, emptyIdx, static_cast<int32_t>(HASH_ENTRY_FIELDS), totalNum);
        AscendC::PipeBarrier<PIPE_V>();
        Scatter(value, key, statusLocal, totalNum);
    }

    __aicore__ inline void UpdateHashStatus(LocalTensor<int32_t>& topkId, LocalTensor<int32_t>& emptyIdx, LocalTensor<int32_t>& statusLocal, TempBufferCursor cursor, uint64_t uniqueNum, uint64_t totalNum){
        // 计算hash
        LocalTensor<int32_t> hashLocal = cursor.Alloc<int32_t>(totalNum);
        HashMaskType maskType{false, true};  // empty-slot mode
        LocalTensor<uint8_t> emptyMask = cursor.Alloc<uint8_t>(totalNum);
        ProbeHashTable(hashLocal, emptyMask, topkId, statusLocal, cursor.Fork(), totalNum, maskType); //找要更新的hashId(uniqueNum)中找到可注入的pos

        // 构建InterLevel
        uint64_t insertNum = 0;
        LocalTensor<int32_t> insertId = cursor.Alloc<int32_t>(totalNum);
        LocalTensor<int32_t> insertIdx = cursor.Alloc<int32_t>(totalNum);

        // Cache update touches only word0, leaving the bucket word intact.
        StripRecord(insertIdx, topkId, uniqueNum);
        AscendC::Duplicate(insertId, static_cast<int32_t>(MAX_RECORD_LIFETIME), uniqueNum);
        PipeBarrier<PIPE_V>();
        PackEntry(insertIdx, insertId, uniqueNum);
        AscendC::Muls(insertId, emptyIdx, static_cast<int32_t>(HASH_ENTRY_FIELDS), uniqueNum);
        PipeBarrier<PIPE_V>();
        Scatter(insertIdx, insertId, statusLocal, uniqueNum);

        LocalTensor<int32_t> insertTopkId = cursor.Alloc<int32_t>(totalNum);
        GatherByBitMask(insertTopkId, emptyMask, topkId, insertNum, uniqueNum);
        GatherByBitMask(insertIdx, emptyMask, emptyIdx, insertNum, uniqueNum);
        EncodeFingerprintIntoPackedIndex(insertIdx, insertTopkId, cursor.Fork(), insertNum);
        GatherByBitMask(insertId, emptyMask, hashLocal, insertNum, uniqueNum);
        AscendC::Muls(insertId, insertId, static_cast<int32_t>(HASH_ENTRY_FIELDS), insertNum);
        PipeBarrier<PIPE_V>();
        AscendC::Adds(insertId, insertId, 1, insertNum);
        PipeBarrier<PIPE_V>();
        Scatter(insertIdx, insertId, statusLocal, insertNum);

    }

    __aicore__ inline void CalEmptyIdx(LocalTensor<int32_t>& result, LocalTensor<int32_t>& statusLocal, TempBufferCursor cursor, uint64_t requiredNum, uint64_t totalNum){

        int32_t maskSize = totalNum / AscendC::AscendCUtils::GetBitSize(sizeof(uint8_t));
        LocalTensor<uint8_t> emptyMask = cursor.Alloc<uint8_t>(maskSize); // 在totalNum中找空闲的位置 totalNum <= tiling_->stateSize
        uint64_t emptyNum = 0;

        LocalTensor<int32_t> records = cursor.Alloc<int32_t>(totalNum); // 在totalNum中找空闲的位置 totalNum <= tiling_->stateSize
        LocalTensor<int32_t> tmpBuf = cursor.Current(); // 在totalNum中找空闲的位置 totalNum <= tiling_->stateSize

        PipeBarrier<PIPE_ALL>();
        AscendC::CreateVecIndex(records, 0, totalNum);
        PipeBarrier<PIPE_V>();
        while(emptyNum < requiredNum){

            GatherRecordIdFromHash(records, records, statusLocal, cursor.Fork(), totalNum);
            isNegative(emptyMask, records, cursor.Fork(), true, totalNum);
            PipeBarrier<PIPE_ALL>();
            AscendC::CreateVecIndex(records, 0, totalNum);
            PipeBarrier<PIPE_V>();
            GatherByBitMask(tmpBuf, emptyMask, records, emptyNum, totalNum);
            if(emptyNum >= requiredNum) break;

            GatherPackedTokenFromHash(records, records, statusLocal, cursor.Fork(), totalNum);
            ExtractRecord(records, records, totalNum);
            isNegative(emptyMask, records, cursor.Fork(), true, totalNum);

            PipeBarrier<PIPE_ALL>();
            AscendC::CreateVecIndex(records, 0, totalNum);
            PipeBarrier<PIPE_V>();
            GatherByBitMask(tmpBuf, emptyMask, records, emptyNum, totalNum);

            if (emptyNum < requiredNum) DecrementRecordCounters(statusLocal, cursor.Fork(), 1, totalNum);
        }

        AscendC::Adds(result, tmpBuf, 0, requiredNum);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void BlockTableLookupVec(
        LocalTensor<int32_t>& idxLocal,
        LocalTensor<int32_t>& localBlkTable,
        LocalTensor<int32_t>& blockTableIdx,
        LocalTensor<int32_t>& blockNumIdx,
        TempBufferCursor cursor,
        int32_t blockSize,
        int32_t blockSizeLog2,
        int32_t maxBlockNum,
        uint64_t count)
    {
        AscendC::Muls(idxLocal, idxLocal, static_cast<int32_t>(tiling_->selTopKBlockSize), count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftRight(blockTableIdx, idxLocal, blockSizeLog2, count);
        AscendC::PipeBarrier<PIPE_V>();
        BitwiseModPowerOfTwo(idxLocal, blockSize, cursor.Fork(), count);

        AscendC::Maxs(blockTableIdx, blockTableIdx, 0, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mins(blockTableIdx, blockTableIdx, maxBlockNum - 1, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(blockTableIdx, blockTableIdx, static_cast<int32_t>(sizeof(int32_t)), count);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Gather(blockNumIdx, localBlkTable,
                        blockTableIdx.template ReinterpretCast<uint32_t>(),
                        static_cast<uint32_t>(0), count);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Muls(blockNumIdx, blockNumIdx, blockSize, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(idxLocal, idxLocal, blockNumIdx, count);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeCopyMappingVec(
        LocalTensor<int32_t>& missTopkId,
        LocalTensor<int32_t>& emptyIdxLocal,
        LocalTensor<int32_t>& localFullBlkTable,
        LocalTensor<int32_t>& localSelBlkTable,
        TempBufferCursor cursor,
        uint64_t mismatchNum,
        int64_t totalNum)
    {
        LocalTensor<int32_t> blockTableIdx = cursor.Alloc<int32_t>(totalNum);
        LocalTensor<int32_t> blockNumIdx = cursor.Alloc<int32_t>(totalNum);

        BlockTableLookupVec(missTopkId, localFullBlkTable, blockTableIdx, blockNumIdx, cursor,
            static_cast<int32_t>(tiling_->fullKvBlockSize),
            static_cast<int32_t>(tiling_->fullKvBlockSizeLog2),
            static_cast<int32_t>(tiling_->fullMaxBlockNum),
            mismatchNum);

        BlockTableLookupVec(emptyIdxLocal, localSelBlkTable, blockTableIdx, blockNumIdx, cursor,
            static_cast<int32_t>(tiling_->selKvBlockSize),
            static_cast<int32_t>(tiling_->selKvBlockSizeLog2),
            static_cast<int32_t>(tiling_->selMaxBlockNum),
            mismatchNum);
    }

    __aicore__ inline void ProcessMisMatchTopkId(LocalTensor<uint8_t>& mismatchMask, LocalTensor<int32_t>& selTopKIdLocal, LocalTensor<int32_t>& statusLocal, LocalTensor<int32_t>& emptyIdxLocal, LocalTensor<int32_t>& localFullBlkTable, LocalTensor<int32_t>& localSelBlkTable, TempBufferCursor cursor, int64_t bshIdx, uint64_t mismatchNum, int64_t totalNum){

        LocalTensor<int32_t> missTopkId = cursor.Alloc<int32_t>(totalNum);
        LocalTensor<int32_t> missTopkIdx = cursor.Current();
        PipeBarrier<PIPE_ALL>();
        AscendC::CreateVecIndex(missTopkId, 0, totalNum);
        PipeBarrier<PIPE_V>();

        GatherByBitMask(missTopkIdx, mismatchMask, missTopkId, mismatchNum, totalNum);
        GatherByBitMask(missTopkId, mismatchMask, selTopKIdLocal, mismatchNum, totalNum);

        // UpdateSelTopkIdx
        Scatter(emptyIdxLocal, missTopkIdx, selTopKIdLocal, mismatchNum); // missTopKIdx用完释放
        // UpdateHashMapping
        UpdateHashStatus(missTopkId, emptyIdxLocal, statusLocal, cursor.Fork(), mismatchNum, totalNum);

        ComputeCopyMappingVec(missTopkId, emptyIdxLocal, localFullBlkTable, localSelBlkTable, cursor.Fork(), mismatchNum, totalNum);

        // Export physical token indices using block sizes supplied by host tiling.
        LocalTensor<int32_t> topkIdToIdxMap = cursor.Alloc<int32_t>(offsetOutputUb_ / sizeof(int32_t));
        AscendC::Duplicate(topkIdToIdxMap, INVALID_SLOT, offsetOutputUb_ / sizeof(int32_t));
        PipeBarrier<PIPE_V>();

        InterleaveTensors(topkIdToIdxMap, missTopkId, emptyIdxLocal, cursor.Fork(), mismatchNum, totalNum);

        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        CopyOutStageOneCopyMapping(bshIdx, topkIdToIdxMap);
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    }

    __aicore__ inline void clearMtpToken(LocalTensor<int32_t>& statusLocal, int64_t curReqKvLength, TempBufferCursor cursor, int64_t totalNum){

        int32_t chunkSize = chunkSize_;
        int64_t chunkNum = totalNum/chunkSize;

        for(int64_t chunkId = 0; chunkId < chunkNum; chunkId++){
            LocalTensor<int32_t> tensor = statusLocal[chunkId * chunkSize * HASH_ENTRY_FIELDS];
            clearMtpTokenPerChunk(tensor, curReqKvLength, cursor.Fork(), chunkSize);
        }
    }
    __aicore__ inline void clearMtpTokenPerChunk(LocalTensor<int32_t>& statusLocal, int64_t curReqKvLength, TempBufferCursor cursor, int64_t totalNum){
        LocalTensor<int32_t> topkId = cursor.Alloc<int32_t>(totalNum);
        LocalTensor<int32_t> topkIdx = cursor.Alloc<int32_t>(totalNum);

        PipeBarrier<PIPE_ALL>();
        AscendC::CreateVecIndex(topkId, 0, totalNum);
        PipeBarrier<PIPE_V>();

        GatherRecordIndexFromHash(topkIdx, topkId, statusLocal, cursor.Fork(), totalNum);
        GatherPackedTokenFromHash(topkId, topkId, statusLocal, cursor.Fork(), totalNum);
        DecodeTokenId(statusLocal, topkId, totalNum);
        AscendC::Adds(statusLocal, statusLocal, static_cast<int32_t>(-curReqKvLength), totalNum);
        AscendC::PipeBarrier<PIPE_V>();

        TempBufferCursor tmpCursor = cursor.Fork();
        LocalTensor<uint8_t> tmpMask = cursor.Alloc<uint8_t>(totalNum);
        isNegative(tmpMask, statusLocal, cursor.Fork(), true, totalNum);

        // Keep valid packed tokens. Clear MTP tokens to record=0, token=-1,
        // matching the previous lifetime semantics without touching word1.
        AscendC::Duplicate(statusLocal, HASH_INDEX_MASK, totalNum);
        PipeBarrier<PIPE_V>();
        AscendC::Select(topkId.template ReinterpretCast<float>(), tmpMask,
            topkId.template ReinterpretCast<float>(), statusLocal.template ReinterpretCast<float>(),
            AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, totalNum);
        PipeBarrier<PIPE_V>();
        InterleaveTensors(statusLocal, topkId, topkIdx, tmpCursor, totalNum, totalNum);

    }

    __aicore__ inline void StageOne(){

        LocalTensor<int32_t> selKvActSeqLocal = workBuf_.Get<int32_t>();
        freeBufLocal_ = selKvActSeqLocal[selKvActSeqUb_ / sizeof(int32_t)];
        TempBufferCursor rootCursor(freeBufLocal_, tmpbuffUb_); // Per-core scratch cursor.

        AscendC::Duplicate(selKvActSeqLocal, 0, selKvActSeqUb_ / sizeof(int32_t));
        AscendC::Duplicate(freeBufLocal_, 0, tmpbuffUb_ / sizeof(int32_t));
        PipeBarrier<PIPE_ALL>();

        // 在 batch 循环前，一次性将当前核负责的元素范围
        int32_t firstBatchId = blkIdx_ * tiling_->stageOneMainCoreBsLoopNum;
        int32_t lastBatchId = firstBatchId + stageOneLoopNum_ - 1;
        int64_t gmStart = (firstBatchId == 0) ? 0 : GetActualQSeqLen(firstBatchId - 1) * tiling_->headnum;
        int64_t gmEnd = GetActualQSeqLen(lastBatchId) * tiling_->headnum;
        LocalTensor<int32_t> mismatchNumLocal = rootCursor.Alloc<int32_t>(gmEnd - gmStart);

        int64_t curFullKvSeqLen = 0;
        int64_t curFullQSeqLen = 0;

        for (int64_t loopIdx = 0; loopIdx < stageOneLoopNum_; loopIdx++) {

            curFullKvSeqLen = fullKvActualSeqGm_.GetValue(loopIdx);

            int32_t globalBatchId = (blkIdx_ * tiling_->stageOneMainCoreBsLoopNum + loopIdx);

            int64_t cumQSeqLen = GetActualQSeqLen(globalBatchId);
            int64_t tOffset = (globalBatchId == 0) ? 0 : GetActualQSeqLen(globalBatchId - 1);
            curFullQSeqLen = cumQSeqLen - tOffset;
            int64_t curFullKvSeqModify = curFullKvSeqLen - curFullQSeqLen;

            TempBufferCursor cursor = rootCursor.Fork(); // 每个batch重新创建cursor

            InitStageOneCopyMapping(cursor, tOffset, (globalBatchId == tiling_->batchsize-1) ? tiling_->t - tOffset : curFullQSeqLen);
            SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);

            if (curFullKvSeqLen <= 0){
                selKvActSeqLocal.SetValue(loopIdx, 0);
                continue;
            }

            LocalTensor<int32_t> statusLocal = cursor.Alloc<int32_t>(selBlockStatUb_ / sizeof(int32_t));
            CopyInBlockStatus(loopIdx, statusLocal);
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);

            DecrementRecordCounters(statusLocal, cursor.Fork(), 1, tiling_->statesize);

            RefreshMatchStatus(statusLocal, cursor.Fork(), tOffset, curFullQSeqLen);

            // block table 按 batch 索引，对本 batch 所有 seq/head 相同，循环外加载一次即可
            LocalTensor<int32_t> localFullBlkTable, localSelBlkTable;
            LoadBlockTables(cursor, globalBatchId, localFullBlkTable, localSelBlkTable);

            uint64_t reusedNum = 0;
            for (int64_t seqIdx = 0; seqIdx < curFullQSeqLen; seqIdx++) {
                for (int64_t headIdx = 0; headIdx < tiling_->headnum; headIdx++) {
                    TempBufferCursor seqCursor = cursor.Fork();

                    int32_t globalIndex = (tOffset + seqIdx) * tiling_->headnum + headIdx;
                    CopyInTopKIndices(globalIndex);
                    LocalTensor<int32_t> selTopKIdLocal = selTopKIdxQue_.DeQue<int32_t>();

                    uint64_t mismatchNum = 0;
                    LocalTensor<uint8_t> mismatchMask = seqCursor.Alloc<uint8_t>(tiling_->topk);
                    ProcessMissMaskAndUpdateMatchTopkId(mismatchMask, selTopKIdLocal, statusLocal, seqCursor.Fork(), mismatchNum, reusedNum, tiling_->topk);

                    if (mismatchNum > 0) {
                        LocalTensor<int32_t> emptyIdxLocal = seqCursor.Current();
                        if (curFullKvSeqLen > poolSize_){
                            CalEmptyIdx(emptyIdxLocal, statusLocal, seqCursor.Fork(), mismatchNum, tiling_->statesize);
                            emptyIdxLocal = seqCursor.Alloc<int32_t>(tiling_->topk);
                        }
                        else{
                            emptyIdxLocal = seqCursor.Alloc<int32_t>(tiling_->topk);
                            GatherByBitMask(emptyIdxLocal, mismatchMask, selTopKIdLocal, mismatchNum, tiling_->topk);
                        }

                        ProcessMisMatchTopkId(mismatchMask, selTopKIdLocal, statusLocal, emptyIdxLocal, localFullBlkTable, localSelBlkTable, seqCursor.Fork(), globalIndex, mismatchNum, tiling_->topk);
                    }
                    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
                    mismatchNumLocal.SetValue(globalIndex - gmStart, static_cast<int32_t>(mismatchNum));
                    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                    CopyOutTopKIndices(globalIndex, selTopKIdLocal);
                    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
                    selTopKIdxQue_.FreeTensor(selTopKIdLocal);
                }
            }
            selKvActSeqLocal.SetValue(loopIdx, reusedNum);
            clearMtpToken(statusLocal, curFullKvSeqModify+1, cursor.Fork(), tiling_->statesize);
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            CopyOutState(loopIdx, statusLocal);
            SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
        }
        CopyOutMismatchNum(gmStart, mismatchNumLocal, (gmEnd - gmStart) * sizeof(int32_t));
        CopyOutSelKvActSeq(selKvActSeqLocal);
    }

    __aicore__ inline void CopyOutState(int64_t loopIdx, LocalTensor<int32_t>& writeOutBlockStatLocal){
        DataCopyExtParams dataCopyParBlkStat{
        static_cast<uint16_t>(1), static_cast<uint32_t>(selBlockStatUb_), 0, 0, 0};
        DataCopyPad(selKvBlockStatusGm_[loopIdx * selBlockStatUb_ / sizeof(int32_t)], writeOutBlockStatLocal, dataCopyParBlkStat);
    }

    __aicore__ inline void CopyOutSelKvActSeq(LocalTensor<int32_t>& selKvActSeqLocal){
        DataCopyExtParams dataCopyParSeqLen{
            static_cast<uint16_t>(1), static_cast<uint32_t>(stageOneLoopNum_ * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(selKvActualSeqGm_, selKvActSeqLocal, dataCopyParSeqLen);
    }

    __aicore__ inline void CopyOutMismatchNum(int64_t gmStart, LocalTensor<int32_t>& mismatchNumUb, int64_t ubBytes){
        DataCopyExtParams dataCopyParams{
            static_cast<uint16_t>(1), static_cast<uint32_t>(ubBytes), 0, 0, 0};
        DataCopyPad(mismatchNumArrayGm_[gmStart], mismatchNumUb, dataCopyParams);
    }

    __aicore__ inline void CopyOutStageOneCopyMapping(int64_t bshIdx, LocalTensor<int32_t>& stageOneAddressMappingLocal){
        int64_t gmOffset = bshIdx * offsetOutputUb_ / sizeof(int32_t);
        DataCopyExtParams dataCopyParAddrMapping{
        static_cast<uint16_t>(1), static_cast<uint32_t>(offsetOutputUb_), 0, 0, 0};
        DataCopyPad(stageOneCopyMappingGM_[gmOffset], stageOneAddressMappingLocal, dataCopyParAddrMapping);
    }

    __aicore__ inline void InitStageOneCopyMapping(TempBufferCursor cursor, int64_t tOffset, int64_t qlen){
        int64_t nums = offsetOutputUb_ / sizeof(int32_t);
        int64_t gmOffset = tOffset * tiling_->headnum * nums;
        LocalTensor<int32_t> topkIdToIdxMap = cursor.Alloc<int32_t>(nums);
        AscendC::Duplicate(topkIdToIdxMap, INVALID_SLOT, nums);
        DataCopyExtParams dataCopyParAddrMapping{
        static_cast<uint16_t>(1), static_cast<uint32_t>(nums * sizeof(int32_t)), 0, 0, 0};
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        for(int32_t idx = 0; idx<qlen; idx++){
            DataCopyPad(stageOneCopyMappingGM_[gmOffset + idx * tiling_->headnum * nums], topkIdToIdxMap, dataCopyParAddrMapping);
        }
    }

    __aicore__ inline void CopyOutTopKIndices(int64_t globalOffset, LocalTensor<int32_t>& selTopKIdLocal){
        int64_t gmOffset = globalOffset * tiling_->topk;
        DataCopyExtParams dataCopyTopkIdx{
            static_cast<uint16_t>(1), static_cast<uint32_t>(tiling_->topk * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(selTopKIndicesGm_[gmOffset], selTopKIdLocal, dataCopyTopkIdx);

    }

private:
    TPipe* pipe_;
    const AvsgMatchTilingData* tiling_;

    // Stage One
    int32_t stageOneLoopNum_ = 0;

    int32_t blkIdx_ = -1;

    int32_t selTopKIdxUb_ = 0;
    int32_t selBlockStatUb_ = 0;
    int32_t selKvActSeqUb_ = 0;
    int32_t tmpbuffUb_ = 0;
    int32_t offsetOutputUb_ = 0;
    int32_t chunkSize_ = tiling_->topk;
    int32_t poolSize_ = tiling_->selMaxBlockNum * tiling_->selKvBlockSize;

    constexpr static int64_t HASH_ENTRY_FIELDS = 2;
    constexpr static int64_t MAX_PROBE_RETRIES = 3;
    constexpr static int64_t MAX_RECORD_LIFETIME = 8; // (0~127)
    constexpr static int32_t HASH_PRIME_MULTIPLIER = 2039;   // 素数，用于扰乱 topkId → hashId 映射
    constexpr static int32_t HASH_STRIDE_MULTIPLIER = 7;      // Hash stride multiplier before masking and low-bit clearing.
    constexpr static int32_t HASH_INDEX_MASK = 0x00FFFFFF;
    constexpr static int32_t HASH_INDEX_SIGN_BIT = 0x00800000;

    // 哈希表 entry 位域常量 — 统一管理 record 位宽与偏移
    // entry: [record:8 | tokenId:24], [fingerprint | cacheSlot].
    // Reinitialize persisted tables when migrating from the old layout.
    int32_t RECORD_BIT_WIDTH = AscendC::AscendCUtils::GetBitSize(sizeof(int8_t));    // 8
    int32_t RECORD_SHIFT_BITS = AscendC::AscendCUtils::GetBitSize(sizeof(int32_t))   // 32
                                - RECORD_BIT_WIDTH;                                    // = 24

    GlobalTensor<int32_t> selKvBlockTableGm_;
    GlobalTensor<int32_t> selKvBlockStatusGm_;
    GlobalTensor<int32_t> selTopKIndicesGm_;
    GlobalTensor<int32_t> fullKvBlockTableGm_;
    GlobalTensor<int32_t> fullKvActualSeqGm_;
    GlobalTensor<int32_t> fullQActualSeqGm_;
    GlobalTensor<int32_t> selKvActualSeqGm_;

    GlobalTensor<int32_t> stageOneCopyMappingGM_;
    GlobalTensor<int32_t> mismatchNumArrayGm_;

    LocalTensor<int32_t> freeBufLocal_;

    TQue<QuePosition::VECIN, 1> selTopKIdxQue_;
    TBuf<QuePosition::VECCALC> workBuf_;

};

} // namespace AvsgMatchNs
#endif // AVSG_MATCH_SPLIT_BS_REUSE_VEC_H
