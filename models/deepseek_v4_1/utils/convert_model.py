# coding=utf-8
# Adapted from
# https://gitee.com/ascend/ModelZoo-PyTorch/blob/master/MindIE/LLM/DeepSeek/DeepSeek-V2/NPU_inference/fp8_cast_bf16.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import sys
import shutil
from argparse import ArgumentParser
from glob import glob
import numpy as np
import torch
import torch_npu
from safetensors.torch import load_file, save_file
from tqdm import tqdm

CUR_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.realpath(os.path.join(CUR_DIR, "../../../"))
sys.path.append(ROOT_DIR)

from convert_config import generate_quant_config, generate_ignore_item

NUM_BITS_4 = 4
NUM_BITS_8 = 8
BLOCK = 32
MX_KEEP_SUBSTRINGS = (".shared_experts.", ".experts.", ".engram.embed.", ".indexer.wq_b")

def expand_scale(scale: torch.Tensor) -> torch.Tensor:
    """
    Expands a 32x32-block FP8 scale to 1x32-group format by repeating each
    row 32 times along dim 0.

    Args:
    scale (torch.Tensor): Scale tensor of shape (N // 32, K // 32) and
        dtype float8_e8m0fnu.

    Returns:
    torch.Tensor: Expanded scale tensor of shape (N, K // 32).
    """
    return scale.view(torch.uint8).repeat_interleave(BLOCK, dim=0).view(scale.dtype)


def is_hif8_target(base_name: str) -> bool:
    return not any(sub in base_name for sub in MX_KEEP_SUBSTRINGS)


def e8m0_to_float(scale: torch.Tensor) -> torch.Tensor:
    """
    Decode an E8M0 scale (uint8 biased exponent) to FP32: value = 2 ** (b - 127).
    """
    exponent = scale.view(torch.uint8).to(torch.int32) - 127
    return torch.ldexp(torch.ones(exponent.shape, dtype=torch.float32), exponent)


def dequant_block_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Undo the checkpoint's block-FP8 quantization: (N, K) e4m3 + (N/32, K/32) e8m0 -> BF16.
    """
    rows, cols = weight.shape
    scale_f32 = e8m0_to_float(scale)
    scale_f32 = scale_f32.repeat_interleave(BLOCK, dim=0).repeat_interleave(BLOCK, dim=1)
    return (weight.to(torch.float32) * scale_f32[:rows, :cols]).to(torch.bfloat16)


def hif8_weight_quant(tensor: torch.Tensor, num_groups: int = 1):
    """
    Offline per-output-channel HiF8 weight quantization. Returns (uint8 weight, FP32 scale).
    """
    rows, cols = tensor.shape
    rank = rows // num_groups
    if num_groups > 1:
        tensor = tensor.view(num_groups, rank, cols).transpose(0, 1).reshape(rank, num_groups * cols)
    quantized, scale = torch_npu.npu_dynamic_quant(tensor.npu(), dst_type=torch_npu.hifloat8)
    quantized = quantized.cpu().view(torch.uint8)
    scale = scale.cpu().to(torch.float32)
    if num_groups > 1:
        quantized = quantized.view(rank, num_groups, cols).transpose(0, 1).reshape(rows, cols)
        scale = scale.repeat(num_groups)
    return quantized.contiguous(), scale.unsqueeze(-1)


def nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def copy_py_json(src, target):
    for root, _, files in os.walk(src):
        for file in files:
            if file.endswith(('.py', '.json', '.jinja')):
                src_path = os.path.join(root, file)
                rel_dir = os.path.relpath(root, src)
                dst_dir = os.path.join(target, rel_dir)
                os.makedirs(dst_dir, exist_ok=True)
                dst_path = os.path.join(dst_dir, file)
                shutil.copy2(src_path, dst_path)


def convert_quantization_config(config, hif8=False):
    """
    Rewrite config['quantization_config'] in place: pop the fp8 quantization_config,
    generate a compressed-tensors config and set it at the top level (where
    model_worker and get_quant_config read it). text_config / vision_config
    are left untouched.
    Returns (old_config, new_config).
    """
    fields = config.get("text_config") or config
    num_layers = fields.get("num_hidden_layers")
    compress_ratios = fields.get("compress_ratios")
    kv_source_layers = fields.get("kv_source_layers", fields.get("kv_source_layer_ids"))
    index_source_layers = fields.get("index_source_layers", fields.get("index_source_layer_ids"))
    if not kv_source_layers or not index_source_layers:
        sys.exit("config.json: kv_source_layers / index_source_layers are required "
                 "to generate the quantization ignore list")
    if num_layers is None or compress_ratios is None:
        sys.exit("config.json: num_hidden_layers / compress_ratios are required")
    if len(compress_ratios) < num_layers:
        sys.exit(
            f"config.json: compress_ratios has {len(compress_ratios)} entries, "
            f"but num_hidden_layers is {num_layers}"
        )
    cache_scheme = {"kv_cache_scheme": {"num_bits": NUM_BITS_8, "type": "float"},
                    "comp_cache_scheme": {"num_bits": NUM_BITS_4, "type": "float"},
                    "li_cache_scheme": {"num_bits": NUM_BITS_4, "type": "float"}}

    old_quant = config.pop("quantization_config", None)

    quant_ignore_layers = generate_ignore_item(num_layers, compress_ratios,
                                               kv_source_layers, index_source_layers)
    quantization_config = generate_quant_config(
        cache_scheme, quant_ignore_layers, w4a8=True, is_mx=True, hif8=hif8)
    quantization_config["weight_block_size"] = [128, 128] if hif8 else [1, BLOCK]
    config["quantization_config"] = quantization_config
    return old_quant, quantization_config


def main(fp8_path, output_path, hif8=False):
    """
    Expands 32x32-block FP8 weights to 1x32-group format and rewrites the quantization config.

    This function reads the checkpoint under fp8_path, expands every F8_E8M0 scale
    of shape (N // 32, K // 32) to shape (N, K // 32) by repeating each scale row 32 times,
    and saves the converted checkpoint to output_path. It also replaces the fp8
    quantization_config of config.json with a compressed-tensors config generated
    via convert_config.py, preserving the nested text_config / vision_config structure.

    Args:
    fp8_path (str): The path to the directory containing the input checkpoint
        shards, model.safetensors.index.json and config.json.
    output_path (str): The path to the directory where the converted checkpoint
        is saved.

    Notes:
    - The function assumes the scales are stored in safetensor files.
    - Routed experts and engram embed (already 1x32) and all bf16/fp32 tensors
      are copied byte-identically.
    """
    os.makedirs(output_path, exist_ok=True)
    model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")
    config_file = os.path.join(fp8_path, "config.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    with open(config_file, "r") as f:
        config = json.load(f)

    old_quant, new_quant = convert_quantization_config(config, hif8=hif8)
    o_groups = int((config.get("text_config") or config).get("o_groups", 8))

    grand_trans = 0
    grand_hif = 0
    grand_added = 0
    safetensor_files = sorted(glob(os.path.join(fp8_path, "*.safetensors")))
    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device="cpu")
        new_state_dict = {}
        rewritten_weights = set()
        for weight_name, weight in current_state_dict.items():
            if (weight_name.endswith(".scale")
                    and weight.dtype == torch.float8_e8m0fnu
                    and weight.dim() == 2):
                base = weight_name[:-len(".scale")]
                weight_key = base + ".weight"
                if weight_key in current_state_dict:
                    w = current_state_dict[weight_key]
                    if (w.dtype == torch.float8_e4m3fn and w.dim() == 2
                            and w.shape[0] % BLOCK == 0 and w.shape[1] % BLOCK == 0
                            and weight.shape[0] == w.shape[0] // BLOCK
                            and weight.shape[1] == w.shape[1] // BLOCK):
                        if hif8 and is_hif8_target(base):
                            quant_weight, quant_scale = hif8_weight_quant(
                                dequant_block_fp8(w, weight),
                                num_groups=o_groups if base.endswith(".attn.wo_a") else 1)
                            new_state_dict[weight_key] = quant_weight
                            new_state_dict[weight_name] = quant_scale
                            grand_added += (nbytes(quant_weight) - nbytes(w)
                                            + nbytes(quant_scale) - nbytes(weight))
                            grand_hif += 1
                            rewritten_weights.add(weight_key)
                            continue
                        new_scale = expand_scale(weight)
                        new_state_dict[weight_name] = new_scale
                        grand_trans += 1
                        grand_added += new_scale.numel() - weight.numel()
                        continue
            if weight_name in rewritten_weights:
                continue
            new_state_dict[weight_name] = weight

        new_safetensor_file = os.path.join(output_path, file_name)
        save_file(new_state_dict, new_safetensor_file, metadata={'format': 'pt'})

    copy_py_json(fp8_path, output_path)

    new_model_index_file = os.path.join(output_path, "model.safetensors.index.json")
    new_config_file = os.path.join(output_path, "config.json")
    metadata = dict(model_index.get("metadata") or {})
    if "total_size" in metadata:
        metadata["total_size"] += grand_added
    with open(new_model_index_file, "w") as f:
        json.dump({"metadata": metadata, "weight_map": model_index["weight_map"]}, f, indent=2)
    with open(new_config_file, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Done. scales expanded: {grand_trans}, linears requantized to HiF8: {grand_hif} "
          f"(+{grand_added / 2**30:.3f}GiB)")
    print(f"output: {output_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_fp8_hf_path", type=str, required=True)
    parser.add_argument("--output_hf_path", type=str, required=True)
    parser.add_argument("--hif8", action="store_true",
                        help="requantize the attention-side linears and the Engram wkv to "
                             "per-channel HiF8 W8A8; the MoE experts, the Engram table and "
                             "the Indexer wq_b keep their MX weights")
    args = parser.parse_args()

    main(args.input_fp8_hf_path, args.output_hf_path, hif8=args.hif8)
