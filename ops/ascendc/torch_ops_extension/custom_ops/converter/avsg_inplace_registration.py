# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""AVSG reinplace registration without DSA-specific multistream overrides."""

import importlib
import inspect

import torch


_GRAPH_PASS_MODULES = (
    "npugraph_ex._acl_concrete_graph.graph_pass",
    "torch_npu.dynamo.npugraph_ex._acl_concrete_graph.graph_pass",
    "torchair._acl_concrete_graph.graph_pass",
    "torch_npu.dynamo.torchair._acl_concrete_graph.graph_pass",
)


def register_avsg_inplace_pairs():
    """Register both AVSG pairs; keep the backend's normal alias/stream checks.

    Match returns (seq, mapping, count, next_table, next_status, next_indices).
    Its last three outputs correspond to inputs 0, 1 and 2; copy returns
    (next_rope, next_cache), corresponding to inputs 0 and 1.
    """
    pairs = (
        (torch.ops.custom.npu_avsg_match_functional.default,
         torch.ops.custom.npu_avsg_match.default, [0, 1, 2]),
        (torch.ops.custom.npu_avsg_copy_functional.default,
         torch.ops.custom.npu_avsg_copy.default, [0, 1]),
    )
    modules = {}
    errors = []
    for module_name in _GRAPH_PASS_MODULES:
        try:
            graph_pass = importlib.import_module(module_name)
        except ImportError as error:
            errors.append(f"{module_name}: {error}")
            continue
        if id(graph_pass) in modules:
            continue
        required = ("inplaceable_npu_ops", "InplaceableNpuOp",
                    "check_multi_stream_for_multi_reinplace")
        missing = [name for name in required if not hasattr(graph_pass, name)]
        if missing:
            raise RuntimeError(f"unsupported AVSG reinplace backend {module_name}: missing {missing}")
        if not isinstance(graph_pass.inplaceable_npu_ops, dict):
            raise RuntimeError(f"unsupported AVSG reinplace registry in {module_name}")
        for function, expected in (
            (graph_pass.InplaceableNpuOp, ("inplace_op", "mutated_arg", "extra_check")),
            (graph_pass.check_multi_stream_for_multi_reinplace, ("node",)),
        ):
            if tuple(inspect.signature(function).parameters)[:len(expected)] != expected:
                raise RuntimeError(f"unsupported AVSG reinplace API in {module_name}: {function}")
        modules[id(graph_pass)] = graph_pass
    if not modules:
        raise RuntimeError("no compatible TorchAir module for AVSG reinplace: " + "; ".join(errors))
    for graph_pass in modules.values():
        for functional_op, inplace_op, mutated_args in pairs:
            graph_pass.inplaceable_npu_ops[functional_op] = graph_pass.InplaceableNpuOp(
                inplace_op=inplace_op,
                mutated_arg=mutated_args,
                extra_check=graph_pass.check_multi_stream_for_multi_reinplace,
            )
    return len(modules)
