# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
import torch_npu
import torchair
import custom_ops
import numpy as np
import torch.nn as nn
from torch_npu.testing.testcase import TestCase, run_tests

np.random.seed(21)  # 固定随机种子
np.set_printoptions(suppress=True)

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))


import numpy as np
import argparse
import random
import torch.nn.functional as F

def softmax_func(x, axis=None):
    if "float16" in x.dtype.name:
        x = x.astype(np.float32)
    x_max = x.max(axis=axis, keepdims=True)
    x_sub = x - x_max
    y = np.exp(x_sub)
    x_sum = y.sum(axis=axis, keepdims=True)
    ans = y / x_sum
    return ans, x_max, x_sum

def softplus(x, beta=1.0):
    return np.log(1 + np.exp(beta * x)) / beta

def moe_gating_top_k_numpy(x: np.ndarray, input_ids: np.ndarray, tid2eid: np.ndarray, bias: np.ndarray, k: int, k_group: int = 1, group_count: int = 1,
                           group_select_mode: int = 0, renorm: int = 0, norm_type: int = 0, y2_flag: bool = False,
                           routed_scaling_factor: float = 1.0, eps: float = 1e-20,
                           additional_bias: np.ndarray = None, additional_token_mask: np.ndarray = None) -> tuple:
    ori_dtype = x.dtype
    x = x.float().numpy()
    if bias is not None:
        bias = bias.astype("float32")
    if additional_bias is not None:
        additional_bias = additional_bias.astype("float32")

    # 归一化
    if norm_type == 0:  # softmax
        x, _, _ = softmax_func(x, -1)
    elif norm_type == 1:  # sigmoid
        x = 1 / (1 + np.exp(-x))
    else:
        x = softplus(x)
        x = np.sqrt(x)

    original_x = x

    # 添加偏置, additional_token_mask为true的行使用additional_bias替换bias
    use_additional_bias = additional_bias is not None and additional_token_mask is not None
    if use_additional_bias and bias is not None:
        x = x + np.where(additional_token_mask.astype(bool)[:, None], additional_bias[None, :], bias[None, :])
    elif use_additional_bias:
        x = x + additional_token_mask.astype(bool)[:, None] * additional_bias[None, :]
    elif bias is not None:
        x = x + bias

    if tid2eid is not None and input_ids is not None:
        indices = tid2eid[input_ids]
    else:
        # 选择top-k专家
        indices = np.argsort(-x, axis=-1, kind='stable')[:, :k]
    y = np.take_along_axis(original_x, indices, axis=1)

    if norm_type != 0:
        y /= (np.sum(y, axis=-1, keepdims=True) + eps)

    # 应用缩放因子
    y *= routed_scaling_factor

    return torch.from_numpy(y).to(ori_dtype), torch.from_numpy(indices.astype(np.int32))


class TestCustomMoeGatingTopK(TestCase):
    def test_moe_gating_top_k_different_dtypes(self):
        """测试不同数据类型的MOE gating top_k算子"""
        torch_npu.npu.set_device(int(DEVICE_ID))

        # 设置参数
        batch_size = 16
        expert_count = 256
        k = 6
        kGroup = 1
        groupCount = 1
        routedScalingFactor = 1.0
        eps = 1e-6
        groupSelectMode = 0
        renorm = 0
        outFlag = False
        N = 100

        # 测试不同数据类型
        dtypes = [torch.float16, torch.bfloat16, torch.float32]
        normTypes = [1, 2] # 测试sigmoid及softplus

        for dtype in dtypes:
            for normType in normTypes:
                print(f"Testing dtype: {dtype}, norm_type: {normType}")

                # 创建输入数据
                np.random.seed(42)
                x = torch.tensor(np.random.uniform(-2, 2, (batch_size, expert_count))).to(dtype)
                input_ids = torch.tensor(np.random.uniform(0, N, (batch_size, ))).to(torch.int64)
                tid2eid = torch.tensor(np.random.uniform(0, expert_count, (N, k))).to(torch.int32)

                # CPU参考结果
                cpu_yOut, cpu_expertIdxOut = moe_gating_top_k_numpy(
                    x, input_ids.numpy(), tid2eid.numpy(), None, k=k, k_group=kGroup, group_count=groupCount,
                    routed_scaling_factor=routedScalingFactor, eps=eps,
                    group_select_mode=groupSelectMode, renorm=renorm, norm_type=normType, y2_flag=outFlag
                )

                # 转换到NPU
                x_npu = x.to("npu:%s" % DEVICE_ID)
                input_ids_npu = input_ids.to("npu:%s" % DEVICE_ID)
                tid2eid_npu = tid2eid.to("npu:%s" % DEVICE_ID)

                # NPU计算
                npu_yOut, npu_expertIdxOut, _ = torch.ops.custom.npu_moe_gating_top_k(
                    x_npu, k, bias=None, input_ids=input_ids_npu, tid2eid=tid2eid_npu, k_group=kGroup, group_count=groupCount,
                    routed_scaling_factor=routedScalingFactor, eps=eps,
                    group_select_mode=groupSelectMode, renorm=renorm, norm_type=normType, out_flag=outFlag
                )

                # # 转换回CPU进行比较
                npu_yOut_cpu = npu_yOut.cpu().float().numpy()
                npu_expertIdxOut_cpu = npu_expertIdxOut.cpu().int().numpy()

                cpu_yOut = cpu_yOut.float().numpy()

                # # 验证结果
                yOut_close = np.allclose(npu_yOut_cpu, cpu_yOut, rtol=0.01, atol=0.001, equal_nan=True)
                expertIdxOut_equal = np.array_equal(npu_expertIdxOut_cpu, cpu_expertIdxOut)

                print(f"  yOut close: {yOut_close}, expertIdxOut equal: {expertIdxOut_equal}")

                self.assertTrue(yOut_close, f"yOut precision compare fail for dtype {dtype}")
                self.assertTrue(expertIdxOut_equal, f"expertIdxOut compare fail for dtype {dtype}")

    def test_moe_gating_top_k_384_experts_topk6(self):
        """测试 384 专家 + topk 6 场景"""
        torch_npu.npu.set_device(int(DEVICE_ID))

        # 设置参数 - 384专家 + topk 6
        batch_size = 16
        expert_count = 384
        k = 6
        kGroup = 1
        groupCount = 1
        routedScalingFactor = 1.0
        eps = 1e-6
        groupSelectMode = 0
        renorm = 0
        outFlag = False
        N = 100

        # 测试不同数据类型
        dtypes = [torch.float16, torch.bfloat16]
        normTypes = [1, 2]

        for dtype in dtypes:
            for normType in normTypes:
                print(f"Testing 384 experts: dtype: {dtype}, norm_type: {normType}")

                # 创建输入数据
                np.random.seed(42)
                x = torch.tensor(np.random.uniform(-2, 2, (batch_size, expert_count))).to(dtype)

                # CPU参考结果
                cpu_yOut, cpu_expertIdxOut = moe_gating_top_k_numpy(
                    x, None, None, None, k=k, k_group=kGroup, group_count=groupCount,
                    routed_scaling_factor=routedScalingFactor, eps=eps,
                    group_select_mode=groupSelectMode, renorm=renorm, norm_type=normType, y2_flag=outFlag
                )

                # 转换到NPU
                x_npu = x.to("npu:%s" % DEVICE_ID)

                # NPU计算
                npu_yOut, npu_expertIdxOut, _ = torch.ops.custom.npu_moe_gating_top_k(
                    x_npu, k, bias=None, input_ids=None, tid2eid=None, k_group=kGroup, group_count=groupCount,
                    routed_scaling_factor=routedScalingFactor, eps=eps,
                    group_select_mode=groupSelectMode, renorm=renorm, norm_type=normType, out_flag=outFlag
                )

                # 转换回CPU进行比较
                npu_yOut_cpu = npu_yOut.cpu().float().numpy()
                npu_expertIdxOut_cpu = npu_expertIdxOut.cpu().int().numpy()
                cpu_yOut = cpu_yOut.float().numpy()

                # 验证结果
                yOut_close = np.allclose(npu_yOut_cpu, cpu_yOut, rtol=0.01, atol=0.001, equal_nan=True)
                expertIdxOut_equal = np.array_equal(npu_expertIdxOut_cpu, cpu_expertIdxOut)

                print(f"  yOut close: {yOut_close}, expertIdxOut equal: {expertIdxOut_equal}")

                self.assertTrue(yOut_close, f"yOut precision compare fail for 384 experts dtype {dtype}")
                self.assertTrue(expertIdxOut_equal, f"expertIdxOut compare fail for 384 experts dtype {dtype}")

    def test_moe_gating_top_k_different_dtypes_graph(self):
        """测试不同数据类型的MOE gating top_k算子"""
        torch_npu.npu.set_device(int(DEVICE_ID))

        class Network(nn.Module):
            def __init__(self):
                super(Network, self).__init__()

            def forward(self, x_npu, k, bias, input_ids, tid2eid, k_group, group_count, routed_scaling_factor, eps, group_select_mode, renorm,
                        norm_type, out_flag):
                npu_yOut, npu_expertIdxOut, y2_out = torch.ops.custom.npu_moe_gating_top_k(
                    x_npu, k, bias=bias, input_ids=input_ids, tid2eid=tid2eid, k_group=k_group, group_count=group_count,
                    routed_scaling_factor=routed_scaling_factor, eps=eps, group_select_mode=group_select_mode, renorm=renorm,
                    norm_type=norm_type, out_flag=out_flag
                )
                return npu_yOut, npu_expertIdxOut, y2_out

        # 设置参数
        batch_size = 16
        expert_count = 256
        k = 6
        kGroup = 1
        groupCount = 1
        routedScalingFactor = 1.0
        eps = 1e-6
        groupSelectMode = 0
        renorm = 0
        outFlag = False
        N = 100

        # 测试不同数据类型
        dtypes = [torch.float16, torch.bfloat16, torch.float32]
        normTypes = [1, 2]

        for dtype in dtypes:
            for normType in normTypes:
                print(f"Testing dtype: {dtype}, norm_type: {normType}")

                # 创建输入数据
                np.random.seed(42)
                x = torch.tensor(np.random.uniform(-2, 2, (batch_size, expert_count))).to(dtype)
                input_ids = torch.tensor(np.random.uniform(0, N, (batch_size, ))).to(torch.int64)
                tid2eid = torch.tensor(np.random.uniform(0, expert_count, (N, k))).to(torch.int32)

                # CPU参考结果
                cpu_yOut, cpu_expertIdxOut = moe_gating_top_k_numpy(
                    x, input_ids.numpy(), tid2eid.numpy(), None, k=k, k_group=kGroup, group_count=groupCount,
                    routed_scaling_factor=routedScalingFactor, eps=eps,
                    group_select_mode=groupSelectMode, renorm=renorm, norm_type=normType, y2_flag=outFlag
                )

                # 转换到NPU
                x_npu = x.to("npu:%s" % DEVICE_ID)
                input_ids_npu = input_ids.to("npu:%s" % DEVICE_ID)
                tid2eid_npu = tid2eid.to("npu:%s" % DEVICE_ID)

                # NPU计算
                npu_yOut, npu_expertIdxOut, _ = torch.ops.custom.npu_moe_gating_top_k(
                    x_npu, k, bias=None, input_ids=input_ids_npu, tid2eid=tid2eid_npu, k_group=kGroup, group_count=groupCount,
                    routed_scaling_factor=routedScalingFactor, eps=eps,
                    group_select_mode=groupSelectMode, renorm=renorm, norm_type=normType, out_flag=outFlag
                )

                npu_mode = Network().to("npu:%s" % DEVICE_ID)
                from torchair.configs.compiler_config import CompilerConfig
                config = CompilerConfig()
                config.mode = "reduce-overhead"
                npu_backend = torchair.get_npu_backend(compiler_config=config)

                npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
                npu_yOut, npu_expertIdxOut, _ = npu_mode(x_npu, k, None, input_ids_npu, tid2eid_npu, kGroup, groupCount,
                                                        routedScalingFactor, eps, groupSelectMode, renorm, normType, outFlag)

                # 转换回CPU进行比较
                npu_yOut_cpu = npu_yOut.cpu().float().numpy()
                npu_expertIdxOut_cpu = npu_expertIdxOut.cpu().int().numpy()

                cpu_yOut = cpu_yOut.float().numpy()

                # # 验证结果
                yOut_close = np.allclose(npu_yOut_cpu, cpu_yOut, rtol=0.01, atol=0.001, equal_nan=True)
                expertIdxOut_equal = np.array_equal(npu_expertIdxOut_cpu, cpu_expertIdxOut)

                print(f"  yOut close: {yOut_close}, expertIdxOut equal: {expertIdxOut_equal}")

                self.assertTrue(yOut_close, f"yOut precision compare fail for dtype {dtype}")
                self.assertTrue(expertIdxOut_equal, f"expertIdxOut compare fail for dtype {dtype}")

    def test_moe_gating_top_k_additional_bias(self):
        """测试可选additional_bias和additional_token_mask输入"""
        torch_npu.npu.set_device(int(DEVICE_ID))

        # 设置参数
        batch_size = 16
        expert_count = 256
        k = 6
        kGroup = 1
        groupCount = 1
        routedScalingFactor = 1.0
        eps = 1e-6
        groupSelectMode = 0
        renorm = 0
        outFlag = False

        dtypes = [torch.float16, torch.bfloat16, torch.float32]
        normTypes = [0, 1, 2]  # softmax / sigmoid / softplus

        for dtype in dtypes:
            for normType in normTypes:
                print(f"Testing additional bias: dtype: {dtype}, norm_type: {normType}")

                # 创建输入数据
                np.random.seed(42)
                x = torch.tensor(np.random.uniform(-2, 2, (batch_size, expert_count))).to(dtype)
                bias = torch.tensor(np.random.uniform(-0.5, 0.5, (expert_count,))).to(dtype)
                additional_bias = torch.tensor(np.random.uniform(-0.5, 0.5, (expert_count,))).to(dtype)
                # 一半token标记为使用additional_bias
                additional_token_mask = torch.zeros(batch_size, dtype=torch.bool)
                additional_token_mask[::2] = True

                # CPU参考结果(bf16不支持numpy(), 先转fp32, 保持量化值)
                cpu_yOut, cpu_expertIdxOut = moe_gating_top_k_numpy(
                    x, None, None, bias.float().numpy(), k=k, k_group=kGroup, group_count=groupCount,
                    routed_scaling_factor=routedScalingFactor, eps=eps,
                    group_select_mode=groupSelectMode, renorm=renorm, norm_type=normType, y2_flag=outFlag,
                    additional_bias=additional_bias.float().numpy(), additional_token_mask=additional_token_mask.numpy()
                )

                # 转换到NPU
                x_npu = x.to("npu:%s" % DEVICE_ID)
                bias_npu = bias.to("npu:%s" % DEVICE_ID)
                additional_bias_npu = additional_bias.to("npu:%s" % DEVICE_ID)
                additional_token_mask_npu = additional_token_mask.to("npu:%s" % DEVICE_ID)

                # NPU计算
                npu_yOut, npu_expertIdxOut, _ = torch.ops.custom.npu_moe_gating_top_k(
                    x_npu, k, bias=bias_npu, input_ids=None, tid2eid=None,
                    additional_bias=additional_bias_npu, additional_token_mask=additional_token_mask_npu,
                    k_group=kGroup, group_count=groupCount,
                    routed_scaling_factor=routedScalingFactor, eps=eps,
                    group_select_mode=groupSelectMode, renorm=renorm, norm_type=normType, out_flag=outFlag
                )

                # 转换回CPU进行比较
                npu_yOut_cpu = npu_yOut.cpu().float().numpy()
                npu_expertIdxOut_cpu = npu_expertIdxOut.cpu().int().numpy()
                cpu_yOut = cpu_yOut.float().numpy()

                # 验证结果
                yOut_close = np.allclose(npu_yOut_cpu, cpu_yOut, rtol=0.01, atol=0.001, equal_nan=True)
                expertIdxOut_equal = np.array_equal(npu_expertIdxOut_cpu, cpu_expertIdxOut)

                print(f"  yOut close: {yOut_close}, expertIdxOut equal: {expertIdxOut_equal}")

                self.assertTrue(yOut_close, f"yOut precision compare fail for dtype {dtype} with additional bias")
                self.assertTrue(expertIdxOut_equal, f"expertIdxOut compare fail for dtype {dtype} with additional bias")

    def test_moe_gating_top_k_mask(self):
        """测试可选token_mask输入: token_mask为False的行expert_idx输出k个-1"""
        torch_npu.npu.set_device(int(DEVICE_ID))

        # 设置参数
        batch_size = 16
        expert_count = 256
        k = 6
        kGroup = 1
        groupCount = 1
        routedScalingFactor = 1.0
        eps = 1e-6
        groupSelectMode = 0
        renorm = 0
        outFlag = False
        N = 100

        # 一半token被mask掉
        mask = torch.ones(batch_size, dtype=torch.bool)
        mask[::2] = False

        for use_hash in [False, True]:
            # 创建输入数据
            np.random.seed(42)
            x = torch.tensor(np.random.uniform(-2, 2, (batch_size, expert_count))).to(torch.float32)
            bias = torch.tensor(np.random.uniform(-0.5, 0.5, (expert_count,))).to(torch.float32)
            if use_hash:
                input_ids = torch.tensor(np.random.uniform(0, N, (batch_size,))).to(torch.int64)
                # token_mask为False的行填入非法input_ids, 验证不会访问tid2eid越界
                input_ids[~mask] = -1
                tid2eid = torch.tensor(np.random.uniform(0, expert_count, (N, k))).to(torch.int32)
            else:
                input_ids = None
                tid2eid = None

            # CPU参考结果(仅计算mask为True的行)
            cpu_yOut, cpu_expertIdxOut = moe_gating_top_k_numpy(
                x, input_ids.numpy() if use_hash else None, tid2eid.numpy() if use_hash else None, bias.numpy(),
                k=k, k_group=kGroup, group_count=groupCount,
                routed_scaling_factor=routedScalingFactor, eps=eps,
                group_select_mode=groupSelectMode, renorm=renorm, norm_type=1, y2_flag=outFlag
            )

            # 转换到NPU
            x_npu = x.to("npu:%s" % DEVICE_ID)
            bias_npu = bias.to("npu:%s" % DEVICE_ID)
            mask_npu = mask.to("npu:%s" % DEVICE_ID)
            input_ids_npu = input_ids.to("npu:%s" % DEVICE_ID) if use_hash else None
            tid2eid_npu = tid2eid.to("npu:%s" % DEVICE_ID) if use_hash else None

            # NPU计算
            npu_yOut, npu_expertIdxOut, _ = torch.ops.custom.npu_moe_gating_top_k(
                x_npu, k, bias=bias_npu, input_ids=input_ids_npu, tid2eid=tid2eid_npu,
                token_mask=mask_npu, k_group=kGroup, group_count=groupCount,
                routed_scaling_factor=routedScalingFactor, eps=eps,
                group_select_mode=groupSelectMode, renorm=renorm, norm_type=1, out_flag=outFlag
            )

            npu_yOut_cpu = npu_yOut.cpu().float().numpy()
            npu_expertIdxOut_cpu = npu_expertIdxOut.cpu().int().numpy()

            # 验证token_mask为False的行expert_idx全为-1
            masked_rows = ~mask.numpy()
            masked_all_neg_one = np.all(npu_expertIdxOut_cpu[masked_rows] == -1)
            self.assertTrue(masked_all_neg_one, f"token_masked rows expert_idx should be all -1, hash={use_hash}")

            # 验证token_mask为True的行与CPU参考结果一致
            valid_rows = mask.numpy()
            yOut_close = np.allclose(
                npu_yOut_cpu[valid_rows], cpu_yOut.float().numpy()[valid_rows],
                rtol=0.01, atol=0.001, equal_nan=True)
            expertIdxOut_equal = np.array_equal(
                npu_expertIdxOut_cpu[valid_rows], cpu_expertIdxOut.numpy()[valid_rows])
            print(f"  hash={use_hash}: yOut close: {yOut_close}, expertIdxOut equal: {expertIdxOut_equal}")
            self.assertTrue(yOut_close, f"yOut precision compare fail for valid rows, hash={use_hash}")
            self.assertTrue(expertIdxOut_equal, f"expertIdxOut compare fail for valid rows, hash={use_hash}")

    def test_moe_gating_top_k_token_mask_dtypes(self):
        """多dtype下的token_mask用例: 覆盖fp16/bf16/fp32的WriteMaskedRow填充"""
        torch_npu.npu.set_device(int(DEVICE_ID))

        batch_size = 32
        expert_count = 256
        k = 6
        N = 100

        mask = torch.ones(batch_size, dtype=torch.bool)
        mask[::2] = False

        for k in [6, 16]:  # k>8覆盖yOutQueue_的UB按uint16视图清零的边界
            for dtype in [torch.float16, torch.bfloat16, torch.float32]:
                for use_hash in [False, True]:
                    np.random.seed(99)
                    x = torch.tensor(np.random.uniform(-2, 2, (batch_size, expert_count))).to(dtype)
                    bias = torch.tensor(np.random.uniform(-0.5, 0.5, (expert_count,))).to(dtype)
                    if use_hash:
                        input_ids = torch.tensor(np.random.uniform(0, N, (batch_size,))).to(torch.int64)
                        tid2eid = torch.tensor(np.random.uniform(0, expert_count, (N, k))).to(torch.int32)
                        # token_mask为False的行填入非法input_ids, 验证不会访问tid2eid越界
                        input_ids_masked = input_ids.clone()
                        input_ids_masked[~mask] = -1
                    else:
                        input_ids = None
                        input_ids_masked = None
                        tid2eid = None

                # 无token_mask参照结果(必须使用合法input_ids)
                y_ref, e_ref, _ = torch.ops.custom.npu_moe_gating_top_k(
                    x.to("npu:%s" % DEVICE_ID), k, bias=bias.to("npu:%s" % DEVICE_ID),
                    input_ids=input_ids.to("npu:%s" % DEVICE_ID) if use_hash else None,
                    tid2eid=tid2eid.to("npu:%s" % DEVICE_ID) if use_hash else None,
                    norm_type=1)
                # 混合token_mask
                y, e, _ = torch.ops.custom.npu_moe_gating_top_k(
                    x.to("npu:%s" % DEVICE_ID), k, bias=bias.to("npu:%s" % DEVICE_ID),
                    input_ids=input_ids_masked.to("npu:%s" % DEVICE_ID) if use_hash else None,
                    tid2eid=tid2eid.to("npu:%s" % DEVICE_ID) if use_hash else None,
                    token_mask=mask.to("npu:%s" % DEVICE_ID), norm_type=1)

                m = mask.numpy()
                y_cpu = y.cpu().float().numpy()
                y_ref_cpu = y_ref.cpu().float().numpy()
                e_cpu = e.cpu().int().numpy()
                e_ref_cpu = e_ref.cpu().int().numpy()

                self.assertTrue(np.all(e_cpu[~m] == -1), f"masked rows expert_idx should be -1, dtype={dtype}, hash={use_hash}")
                self.assertTrue(np.all(y_cpu[~m] == 0), f"masked rows y should be 0, dtype={dtype}, hash={use_hash}")
                self.assertTrue(np.array_equal(e_cpu[m], e_ref_cpu[m]), f"valid rows expert_idx mismatch, dtype={dtype}, hash={use_hash}")
                self.assertTrue(np.allclose(y_cpu[m], y_ref_cpu[m], rtol=0.01, atol=0.001),
                                f"valid rows y mismatch, dtype={dtype}, hash={use_hash}")
                print(f"  dtype={dtype}, hash={use_hash}: masked(-1/0) and valid rows -> PASS")

    def test_moe_gating_top_k_token_mask_grouped(self):
        """分组路径 + out_flag=True 的token_mask用例（覆盖generalized/regbase分组分支）"""
        torch_npu.npu.set_device(int(DEVICE_ID))

        batch_size = 32
        expert_count = 256
        k = 6
        kGroup = 4
        groupCount = 8
        routedScalingFactor = 2.0
        eps = 1e-6
        groupSelectMode = 1  # top2 sum
        renorm = 0
        outFlag = True

        mask = torch.ones(batch_size, dtype=torch.bool)
        mask[1::2] = False

        for k in [6, 16]:  # k>8覆盖yOutQueue_的UB按uint16视图清零的边界
            for dtype in [torch.float16, torch.bfloat16, torch.float32]:
                for normType in [0, 1]:  # softmax / sigmoid
                    np.random.seed(123)
                    x = torch.tensor(np.random.uniform(-2, 2, (batch_size, expert_count))).to(dtype)
                    bias = torch.tensor(np.random.uniform(-0.5, 0.5, (expert_count,))).to(dtype)
                    x_npu = x.to("npu:%s" % DEVICE_ID)
                    bias_npu = bias.to("npu:%s" % DEVICE_ID)
                    mask_npu = mask.to("npu:%s" % DEVICE_ID)

                # 无token_mask参照(out_flag=True以校验out行为)
                y_ref, e_ref, out_ref = torch.ops.custom.npu_moe_gating_top_k(
                    x_npu, k, bias=bias_npu, k_group=kGroup, group_count=groupCount,
                    routed_scaling_factor=routedScalingFactor, eps=eps,
                    group_select_mode=groupSelectMode, renorm=renorm,
                    norm_type=normType, out_flag=outFlag)
                # 混合token_mask
                y, e, out = torch.ops.custom.npu_moe_gating_top_k(
                    x_npu, k, bias=bias_npu, token_mask=mask_npu, k_group=kGroup, group_count=groupCount,
                    routed_scaling_factor=routedScalingFactor, eps=eps,
                    group_select_mode=groupSelectMode, renorm=renorm,
                    norm_type=normType, out_flag=outFlag)

                m = mask.numpy()
                y_cpu = y.cpu().float().numpy()
                y_ref_cpu = y_ref.cpu().float().numpy()
                e_cpu = e.cpu().int().numpy()
                e_ref_cpu = e_ref.cpu().int().numpy()
                out_cpu = out.cpu().float().numpy()
                out_ref_cpu = out_ref.cpu().float().numpy()

                self.assertTrue(np.all(e_cpu[~m] == -1), f"masked expert_idx should be -1, dtype={dtype}, norm={normType}")
                self.assertTrue(np.all(y_cpu[~m] == 0), f"masked y should be 0, dtype={dtype}, norm={normType}")
                self.assertTrue(np.all(out_cpu[~m] == 0), f"masked out should be 0, dtype={dtype}, norm={normType}")
                self.assertTrue(np.array_equal(e_cpu[m], e_ref_cpu[m]), f"valid expert_idx mismatch, dtype={dtype}, norm={normType}")
                self.assertTrue(np.allclose(y_cpu[m], y_ref_cpu[m], rtol=0.01, atol=0.001),
                                f"valid y mismatch, dtype={dtype}, norm={normType}")
                self.assertTrue(np.allclose(out_cpu[m], out_ref_cpu[m], rtol=0.01, atol=0.001),
                                f"valid out mismatch, dtype={dtype}, norm={normType}")
                print(f"  grouped dtype={dtype}, norm={normType}: masked(-1/0/0) and valid rows -> PASS")

if __name__ == "__main__":
    run_tests()
