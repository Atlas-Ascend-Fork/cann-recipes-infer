/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 */

#ifndef AVSG_COPY_H_
#define AVSG_COPY_H_

#include "kernel_operator.h"
#include <cstdint>

namespace AvsgCopyNs {
using namespace AscendC;

constexpr int32_t UB_BLOCK_ALIGN_BYTES = 32;

template <typename U>
__aicore__ inline U CeilAlign(U value, U alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

struct TempBufferCursor {
    LocalTensor<int32_t> buf;
    int64_t remainingBytes;

    __aicore__ inline TempBufferCursor(LocalTensor<int32_t> buffer, int64_t bytes)
        : buf(buffer), remainingBytes(bytes) {}

    template <typename T>
    __aicore__ inline LocalTensor<T> Alloc(int64_t count)
    {
        int64_t bytes = CeilAlign(count * static_cast<int64_t>(sizeof(T)),
            static_cast<int64_t>(UB_BLOCK_ALIGN_BYTES));
        ASSERT_MSG(bytes <= remainingBytes, "AvsgCopy TempBufferCursor overflow.");
        LocalTensor<T> allocated = buf.template ReinterpretCast<T>();
        buf = buf[bytes / sizeof(int32_t)];
        remainingBytes -= bytes;
        return allocated;
    }

    __aicore__ inline TempBufferCursor Fork() const
    {
        return TempBufferCursor(buf, remainingBytes);
    }
};

template <typename T>
class AvsgCopy {
public:
    __aicore__ inline AvsgCopy(TPipe* pipe, const AvsgCopyTilingData* tiling)
        : pipe_(pipe), tiling_(tiling) {}

    __aicore__ inline void Init(
        GM_ADDR selection_k_rope, GM_ADDR selection_kv_cache, GM_ADDR full_k_rope,
        GM_ADDR full_kv_cache, GM_ADDR full_q_actual_seq, GM_ADDR stage_two_copy_mapping,
        GM_ADDR stage_two_copy_count)
    {
        blkIdx_ = GetBlockIdx();
        if (blkIdx_ >= tiling_->usedCoreNum) {
            return;
        }
        kRopeUbOffset_ = tiling_->kvCacheUbSize / sizeof(T);
        pipe_->InitBuffer(kvCacheQue_, tiling_->buffNum,
            tiling_->kvCacheUbSize + tiling_->kRopeUbSize);
        pipe_->InitBuffer(workBuf_, tiling_->workBufSize);

        selKRopeGm_.SetGlobalBuffer((__gm__ T*)selection_k_rope);
        selKvCacheGm_.SetGlobalBuffer((__gm__ T*)selection_kv_cache);
        if (tiling_->hasKRope) {
            fullKRopeGm_.SetGlobalBuffer((__gm__ T*)full_k_rope);
        }
        fullKvCacheGm_.SetGlobalBuffer((__gm__ T*)full_kv_cache);
        fullQActualSeqGm_.SetGlobalBuffer((__gm__ int32_t*)full_q_actual_seq);
        stageTwoCopyMappingGm_.SetGlobalBuffer((__gm__ int32_t*)stage_two_copy_mapping);
        stageTwoCopyCountGm_.SetGlobalBuffer((__gm__ int32_t*)stage_two_copy_count);
    }

    __aicore__ inline void Process()
    {
        if (blkIdx_ >= tiling_->usedCoreNum) {
            return;
        }
        StageTwo();
    }

private:
    template <HardEvent event>
    __aicore__ inline void SetWaitFlag(HardEvent evt)
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
        AscendC::SetFlag<event>(eventId);
        AscendC::WaitFlag<event>(eventId);
    }

    struct StageTwoCopyRange {
        int64_t bshIdxStart = 0;
        int64_t bshIdxEnd = 0;
        int64_t firstBshStartEntry = 0;
        LocalTensor<int32_t> bshEndLocal;
    };

    __aicore__ inline void BuildStageTwoCopyRange(
        StageTwoCopyRange& copyRange, TempBufferCursor cursor, int64_t bshCount)
    {
        LocalTensor<int32_t> copyCountLocal = copyRange.bshEndLocal;
        LocalTensor<int64_t> prefixSumLocal = cursor.Alloc<int64_t>(bshCount + 1);
        DataCopyExtParams countCopyParams{static_cast<uint16_t>(1),
            static_cast<uint32_t>(bshCount * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> countPadParams{false, 0, 0, 0};
        DataCopyPad(copyCountLocal, stageTwoCopyCountGm_, countCopyParams, countPadParams);
        SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

        int64_t totalValidEntries = 0;
        prefixSumLocal.SetValue(0, 0);
        for (int64_t i = 0; i < bshCount; ++i) {
            totalValidEntries += copyCountLocal.GetValue(i);
            prefixSumLocal.SetValue(i + 1, totalValidEntries);
        }
        if (totalValidEntries == 0) {
            return;
        }

        const int64_t baseEntries = totalValidEntries / tiling_->usedCoreNum;
        const int64_t extraEntries = totalValidEntries % tiling_->usedCoreNum;
        const int64_t myStart = blkIdx_ * baseEntries +
            (blkIdx_ < extraEntries ? blkIdx_ : extraEntries);
        const int64_t myEnd = myStart + baseEntries + (blkIdx_ < extraEntries ? 1 : 0);
        if (myStart >= myEnd) {
            return;
        }

        int64_t lo = 0;
        int64_t hi = bshCount;
        while (lo < hi) {
            const int64_t mid = (lo + hi) / 2;
            if (prefixSumLocal.GetValue(mid + 1) <= myStart) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        copyRange.bshIdxStart = lo;
        copyRange.firstBshStartEntry = myStart - prefixSumLocal.GetValue(lo);

        hi = bshCount;
        while (lo < hi) {
            const int64_t mid = (lo + hi) / 2;
            if (prefixSumLocal.GetValue(mid + 1) < myEnd) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        copyRange.bshIdxEnd = lo + 1;
        copyCountLocal.SetValue(lo, static_cast<int32_t>(myEnd - prefixSumLocal.GetValue(lo)));
    }

    __aicore__ inline void LoadOwnedBshCopyMapping(
        const StageTwoCopyRange& copyRange, int64_t bshId,
        LocalTensor<int32_t>& mappingLocal, int64_t& loopNum)
    {
        const int64_t startEntry = bshId == copyRange.bshIdxStart ? copyRange.firstBshStartEntry : 0;
        const int64_t endEntry = copyRange.bshEndLocal.GetValue(bshId);
        loopNum = endEntry - startEntry;
        if (loopNum <= 0) {
            loopNum = 0;
            return;
        }
        const int64_t entryOffset = bshId * tiling_->mappingEntriesPerBsh + startEntry;
        DataCopyExtParams mappingCopyParams{static_cast<uint16_t>(1),
            static_cast<uint32_t>(loopNum * 2 * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> mappingPadParams{false, 0, 0, 0};
        DataCopyPad(mappingLocal, stageTwoCopyMappingGm_[entryOffset * 2],
            mappingCopyParams, mappingPadParams);
        SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    }

    __aicore__ inline void CopyFromFullKv(int32_t src, int32_t dst)
    {
        const int64_t blockIdx = static_cast<int64_t>(src) >> tiling_->fullKvBlockSizeLog2;
        const int64_t blockOffset = static_cast<int64_t>(src) & (tiling_->fullKvBlockSize - 1);
        const int64_t fullKvCacheAddr = blockIdx * tiling_->fullKvCacheStride0 +
            blockOffset * tiling_->fullKvCacheStride1;
        const int64_t fullKRopeAddr = blockIdx * tiling_->fullKRopeStride0 +
            blockOffset * tiling_->fullKRopeStride1;
        const int64_t selKvCacheAddr = static_cast<int64_t>(dst) * tiling_->kvCacheDim;
        const int64_t selKRopeAddr = static_cast<int64_t>(dst) * tiling_->kRopeDim;

        LocalTensor<T> inTensor = kvCacheQue_.AllocTensor<T>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams kvCopyParams{static_cast<uint16_t>(1),
            static_cast<uint32_t>(tiling_->kvCacheDim * sizeof(T)), 0, 0, 0};
        DataCopyPad(inTensor, fullKvCacheGm_[fullKvCacheAddr], kvCopyParams, padParams);
        DataCopyExtParams ropeCopyParams{static_cast<uint16_t>(1),
            static_cast<uint32_t>(tiling_->kRopeDim * sizeof(T)), 0, 0, 0};
        if (tiling_->hasKRope) {
            DataCopyPad(inTensor[kRopeUbOffset_], fullKRopeGm_[fullKRopeAddr], ropeCopyParams, padParams);
        }
        kvCacheQue_.EnQue(inTensor);
        inTensor = kvCacheQue_.DeQue<T>();
        DataCopyPad(selKvCacheGm_[selKvCacheAddr], inTensor, kvCopyParams);
        if (tiling_->hasKRope) {
            DataCopyPad(selKRopeGm_[selKRopeAddr], inTensor[kRopeUbOffset_], ropeCopyParams);
        }
        kvCacheQue_.FreeTensor(inTensor);
    }

    __aicore__ inline void StageTwo()
    {
        const int64_t bshCount = static_cast<int64_t>(
            fullQActualSeqGm_.GetValue(tiling_->batchsize - 1)) * tiling_->headnum;
        if (bshCount <= 0 || bshCount > tiling_->bshCapacity) {
            return;
        }
        TempBufferCursor cursor(workBuf_.Get<int32_t>(), tiling_->workBufSize);
        StageTwoCopyRange copyRange;
        copyRange.bshEndLocal = cursor.Alloc<int32_t>(tiling_->bshCapacity);
        BuildStageTwoCopyRange(copyRange, cursor.Fork(), bshCount);
        if (copyRange.bshIdxStart >= copyRange.bshIdxEnd) {
            return;
        }
        LocalTensor<int32_t> mappingLocal = cursor.Alloc<int32_t>(
            tiling_->mappingEntriesPerBsh * 2);
        int64_t loopNum = 0;
        for (int64_t bshId = copyRange.bshIdxStart; bshId < copyRange.bshIdxEnd; ++bshId) {
            LoadOwnedBshCopyMapping(copyRange, bshId, mappingLocal, loopNum);
            for (int64_t loopIdx = 0; loopIdx < loopNum; ++loopIdx) {
                CopyFromFullKv(mappingLocal.GetValue(loopIdx * 2),
                    mappingLocal.GetValue(loopIdx * 2 + 1));
            }
        }
    }

    TPipe* pipe_;
    const AvsgCopyTilingData* tiling_;
    int32_t blkIdx_ = -1;
    int64_t kRopeUbOffset_ = 0;
    GlobalTensor<T> selKRopeGm_;
    GlobalTensor<T> selKvCacheGm_;
    GlobalTensor<T> fullKRopeGm_;
    GlobalTensor<T> fullKvCacheGm_;
    GlobalTensor<int32_t> fullQActualSeqGm_;
    GlobalTensor<int32_t> stageTwoCopyMappingGm_;
    GlobalTensor<int32_t> stageTwoCopyCountGm_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> kvCacheQue_;
    TBuf<QuePosition::VECCALC> workBuf_;
};

} // namespace AvsgCopyNs
#endif // AVSG_COPY_H_
