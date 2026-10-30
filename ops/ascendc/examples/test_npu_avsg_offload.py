# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details.

"""128-step TND KV correctness test for AvsgMatch + AvsgCopy.

Run after installing the custom operators and torch extension:
    python test_npu_avsg_offload.py
All input indices and CPU golden values are prepared before the first operator
call. Each step's result is snapshotted before the next step reuses the cache.
"""

import numpy as np
import torch
import torch_npu
import torchair as tng
import torchair._contrib.custom_torch_ops
import custom_ops

from torch_npu.testing.testcase import TestCase, run_tests
from torchair.configs.compiler_config import CompilerConfig


DEVICE_ID = 0
STEPS = 128
BLOCK_SIZE = 128
TOPK = 512
STATE_SIZE = 4096
KV_DIM = 512
ROPE_DIM = 64
MAX_SEQ_LEN = 8192
SEED = 42


def random_step_indices(rng, initial_kv_lengths, q_lengths, steps=STEPS):
    """Generate all steps up front; -1 denotes unused topk positions."""
    cumulative_q = np.cumsum(q_lengths, dtype=np.int32)
    indices = np.full((steps, int(cumulative_q[-1]), 1, TOPK), -1, dtype=np.int32)
    kv_lengths = initial_kv_lengths[None, :] + np.arange(steps, dtype=np.int32)[:, None]
    for step in range(steps):
        row = 0
        for batch, q_length in enumerate(q_lengths):
            for query in range(int(q_length)):
                # Earlier queries in an MTP group cannot see later KV tokens.
                visible_kv = int(kv_lengths[step, batch] - q_length + query + 1)
                count = min(TOPK, visible_kv)
                indices[step, row, 0, :count] = rng.choice(visible_kv, count, replace=False)
                row += 1
    return indices, kv_lengths, cumulative_q


def gather_paged_kv(cache, block_table, indices, q_lengths, valid_mask):
    """Gather in original TND/topk order, independently of match's hash state."""
    block_size = cache.shape[1]
    token_ids = indices[:, 0, :].astype(np.int64)
    mask = valid_mask[:, 0, :]
    batch_ids = np.repeat(np.arange(len(q_lengths)), q_lengths)
    rows = np.broadcast_to(batch_ids[:, None], token_ids.shape)[mask]
    tokens = token_ids[mask]
    block_ids = block_table[rows, tokens // block_size].astype(np.int64)
    physical_tokens = block_ids * block_size + tokens % block_size
    return cache.reshape(-1, cache.shape[-1])[physical_tokens].copy()


def to_npu(array, dtype=None, swapped=False):
    tensor = torch.from_numpy(array.copy()).to(device=f"npu:{DEVICE_ID}", dtype=dtype)
    if not swapped:
        return tensor
    offloaded = torch_npu.empty_with_swapped_memory(
        tensor.shape, dtype=tensor.dtype, device=tensor.device)
    offloaded.copy_(tensor)
    return offloaded


class Net(torch.nn.Module):
    def forward(self, selection_k_rope, selection_kv_cache, selection_kv_block_table,
                selection_kv_block_status, selection_topk_indices, full_k_rope,
                full_kv_cache, full_kv_block_table, full_kv_actual_seq, full_q_actual_seq):
        _, mapping, count = torch.ops.custom.npu_avsg_match(
            selection_kv_block_table, selection_kv_block_status, selection_topk_indices,
            full_kv_block_table, full_kv_actual_seq, full_q_actual_seq,
            selection_topk_block_size=1, kv_block_size=full_kv_cache.shape[1])
        torch.ops.custom.npu_avsg_copy(
            selection_k_rope, selection_kv_cache, full_k_rope, full_kv_cache,
            full_q_actual_seq, mapping, count)
        return selection_k_rope, selection_kv_cache, selection_kv_block_table, selection_topk_indices


class TestCustomAvsgOffload(TestCase):
    def _run_multistep(self, api_impl_mode, dtype, is_offload, block_size=BLOCK_SIZE,
                       max_seq_len=MAX_SEQ_LEN):
        torch_npu.npu.set_device(DEVICE_ID)
        torch.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
        torch.manual_seed(SEED)
        torch_npu.npu.manual_seed_all(SEED)
        rng = np.random.default_rng(SEED)

        # Uneven TND lengths exercise cumulative Q offsets.
        q_lengths = np.array([1, 2], dtype=np.int32)
        initial_kv_lengths = np.full(len(q_lengths), max_seq_len - STEPS, dtype=np.int32)
        batch_size = len(q_lengths)
        full_blocks_per_batch = max_seq_len // block_size
        selection_blocks_per_batch = STATE_SIZE // block_size
        full_blocks = batch_size * full_blocks_per_batch
        selection_blocks = batch_size * selection_blocks_per_batch
        full_table = rng.permutation(full_blocks).astype(np.int32).reshape(batch_size, -1)
        selection_table = rng.permutation(selection_blocks).astype(np.int32).reshape(batch_size, -1)

        # Values are exactly representable in FP16 and BF16, permitting exact
        # copy comparisons while random channel vectors distinguish tokens.
        full_kv = rng.integers(-128, 128, (full_blocks, block_size, KV_DIM)).astype(np.float16) / 16
        full_rope = rng.integers(-128, 128, (full_blocks, block_size, ROPE_DIM)).astype(np.float16) / 16

        # Phase 1: generate every step and its independent golden before NPU execution.
        all_indices, all_kv_lengths, cumulative_q = random_step_indices(
            rng, initial_kv_lengths, q_lengths)
        all_masks = all_indices >= 0
        golden = [
            (gather_paged_kv(full_kv, full_table, indices, q_lengths, mask),
             gather_paged_kv(full_rope, full_table, indices, q_lengths, mask))
            for indices, mask in zip(all_indices, all_masks)
        ]

        full_kv_npu = to_npu(full_kv, dtype, swapped=is_offload)
        full_rope_npu = to_npu(full_rope, dtype, swapped=is_offload)
        full_table_npu = to_npu(full_table)
        selection_table_npu = to_npu(selection_table)
        status_npu = to_npu(np.full((batch_size, STATE_SIZE * 2), -1, dtype=np.int32))
        selection_kv_npu = torch.zeros(
            (selection_blocks, block_size, KV_DIM), dtype=dtype, device=f"npu:{DEVICE_ID}")
        selection_rope_npu = torch.zeros(
            (selection_blocks, block_size, ROPE_DIM), dtype=dtype, device=f"npu:{DEVICE_ID}")
        cumulative_q_npu = to_npu(cumulative_q)
        # Match rewrites indices in place; never share their storage with golden inputs.
        step_indices_npu = [to_npu(indices) for indices in all_indices]
        step_kv_lengths_npu = [to_npu(lengths) for lengths in all_kv_lengths]

        model = Net().npu().eval()
        if api_impl_mode == "dynamo":
            config = CompilerConfig()
            config.experimental_config.keep_inference_input_mutations = True
            model = torch.compile(model, backend=tng.get_npu_backend(compiler_config=config),
                                  dynamic=True, fullgraph=True)

        # Phase 2: keep cache/status across all calls. Do not warm up on this
        # state: the first execution must observe the cold, all--1 status.
        results = []
        with torch.no_grad():
            for step in range(STEPS):
                rope_out, kv_out, table_out, indices_out = model(
                    selection_rope_npu, selection_kv_npu, selection_table_npu, status_npu,
                    step_indices_npu[step], full_rope_npu, full_kv_npu, full_table_npu,
                    step_kv_lengths_npu[step], cumulative_q_npu)
                torch.npu.synchronize()
                indices_host = indices_out.cpu().numpy().copy()
                mask = all_masks[step]
                # Check remapped slots before indexing so negative slots cannot
                # silently select the last entry through NumPy's indexing rules.
                self.assertTrue(np.all(indices_host[mask] >= 0), f"step={step}: negative valid slot")
                self.assertTrue(np.all(indices_host[mask] < STATE_SIZE), f"step={step}: slot overflow")
                self.assertTrue(np.all(indices_host[~mask] == -1), f"step={step}: padding changed")
                table_host = table_out.cpu().numpy()
                self.assertTrue(np.all((table_host >= 0) & (table_host < selection_blocks)),
                                f"step={step}: invalid physical block")
                # Snapshot now: the next call can evict these slots. float()
                # permits NumPy conversion for BF16 without changing values.
                results.append((
                    gather_paged_kv(kv_out.float().cpu().numpy(), table_host,
                                    indices_host, q_lengths, mask),
                    gather_paged_kv(rope_out.float().cpu().numpy(), table_host,
                                    indices_host, q_lengths, mask)))

        # Phase 3: compare all steps, preserving original token order.
        for step, (expected, actual) in enumerate(zip(golden, results)):
            for name, reference, result in zip(("KV", "RoPE"), expected, actual):
                with self.subTest(mode=api_impl_mode, step=step, cache=name):
                    torch.testing.assert_close(
                        torch.from_numpy(result), torch.from_numpy(reference).float(),
                        rtol=0, atol=0,
                        msg=f"{api_impl_mode}, offload={is_offload}, step={step}: {name} differs")

    def test_avsg_offload_tnd_multistep_eager(self):
        self._run_multistep("eager", torch.float16, is_offload=True)

    def test_avsg_offload_tnd_multistep_eager_seq16384(self):
        self._run_multistep("eager", torch.float16, is_offload=True, max_seq_len=16384)

    def test_avsg_offload_tnd_multistep_eager_block256(self):
        self._run_multistep("eager", torch.float16, is_offload=True, block_size=256)

    def test_avsg_offload_tnd_multistep_graph(self):
        self._run_multistep("dynamo", torch.bfloat16, is_offload=True)


if __name__ == "__main__":
    run_tests()
