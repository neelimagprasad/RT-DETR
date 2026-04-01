#!/usr/bin/env python3
"""
Convert Hugging Face Docling Heron-101 weights to this repo's RT-DETRv2 checkpoint format.

HF repo: docling-project/docling-layout-heron-101
Files: model.safetensors + config.json (Transformers RTDetrV2ForObjectDetection)

This script builds this repo's RT-DETRv2-R101 (hidden_dim=384) model and maps as many weights as possible:
  - backbone stem + ResNet stages
  - decoder_input_proj -> decoder.input_proj
  - enc_output / enc_score_head / enc_bbox_head
  - decoder class/bbox heads
  - decoder layers (including packing q/k/v projections into MultiheadAttention in_proj_*)
  - denoising_class_embed

Output is a .pth that you can pass to tools/train.py using -t.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple
import re

import torch
from safetensors import safe_open


def _pack_qkv(
    q_w: torch.Tensor,
    k_w: torch.Tensor,
    v_w: torch.Tensor,
    q_b: torch.Tensor,
    k_b: torch.Tensor,
    v_b: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    in_w = torch.cat([q_w, k_w, v_w], dim=0)
    in_b = torch.cat([q_b, k_b, v_b], dim=0)
    return in_w, in_b


def _load_hf_safetensors(path: Path) -> Dict[str, torch.Tensor]:
    tensors: Dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors


def _maybe_pad_embedding(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    # Handle (num_classes) vs (num_classes+1) edge cases.
    if dst.shape == src.shape:
        return src
    if len(dst.shape) == 2 and len(src.shape) == 2 and dst.shape[1] == src.shape[1]:
        out = dst.clone()
        n = min(dst.shape[0], src.shape[0])
        out[:n] = src[:n]
        return out
    return src


def _map_backbone_key(hf_key: str) -> str:
    """Map HF RT-DETR ResNet backbone keys to this repo's PResNet keys."""
    if not hf_key.startswith("model.backbone.model."):
        return ""

    mapped = hf_key
    mapped = mapped.replace("model.backbone.model.embedder.embedder.0.", "backbone.conv1.conv1_1.")
    mapped = mapped.replace("model.backbone.model.embedder.embedder.1.", "backbone.conv1.conv1_2.")
    mapped = mapped.replace("model.backbone.model.embedder.embedder.2.", "backbone.conv1.conv1_3.")
    mapped = mapped.replace(".convolution.", ".conv.")
    mapped = mapped.replace(".normalization.", ".norm.")
    mapped = mapped.replace("model.backbone.model.encoder.stages.", "backbone.res_layers.")
    mapped = re.sub(r"backbone\.res_layers\.(\d+)\.layers\.(\d+)\.layer\.0\.", r"backbone.res_layers.\1.blocks.\2.branch2a.", mapped)
    mapped = re.sub(r"backbone\.res_layers\.(\d+)\.layers\.(\d+)\.layer\.1\.", r"backbone.res_layers.\1.blocks.\2.branch2b.", mapped)
    mapped = re.sub(r"backbone\.res_layers\.(\d+)\.layers\.(\d+)\.layer\.2\.", r"backbone.res_layers.\1.blocks.\2.branch2c.", mapped)

    # Stage 2/3/4 stride-2 shortcuts are avgpool + ConvNormLayer in a Sequential.
    mapped = re.sub(
        r"backbone\.res_layers\.(\d+)\.layers\.(\d+)\.shortcut\.1\.conv\.",
        r"backbone.res_layers.\1.blocks.\2.short.conv.conv.",
        mapped,
    )
    mapped = re.sub(
        r"backbone\.res_layers\.(\d+)\.layers\.(\d+)\.shortcut\.1\.norm\.",
        r"backbone.res_layers.\1.blocks.\2.short.conv.norm.",
        mapped,
    )

    # Stage 1 shortcut is a direct ConvNormLayer.
    mapped = re.sub(
        r"backbone\.res_layers\.(\d+)\.layers\.(\d+)\.shortcut\.conv\.",
        r"backbone.res_layers.\1.blocks.\2.short.conv.",
        mapped,
    )
    mapped = re.sub(
        r"backbone\.res_layers\.(\d+)\.layers\.(\d+)\.shortcut\.norm\.",
        r"backbone.res_layers.\1.blocks.\2.short.norm.",
        mapped,
    )

    return mapped


def convert(*, cfg_path: str, hf_safetensors: Path, output_pth: Path) -> None:
    # Build this repo model (RT-DETRv2-R101/384) using YAMLConfig.
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from src.core import YAMLConfig
    from src.misc import dist_utils

    cfg = YAMLConfig(cfg_path, device="cpu")
    model = dist_utils.de_parallel(cfg.model)
    dst_sd = model.state_dict()

    hf = _load_hf_safetensors(hf_safetensors)

    mapped: Dict[str, torch.Tensor] = {}

    # --- backbone ---
    for hf_key, tensor in hf.items():
        dst_key = _map_backbone_key(hf_key)
        if dst_key:
            mapped[dst_key] = tensor

    # --- input proj (3 levels) ---
    for i in range(3):
        mapped[f"decoder.input_proj.{i}.conv.weight"] = hf[f"model.decoder_input_proj.{i}.0.weight"]
        for bn_key in ["weight", "bias", "running_mean", "running_var", "num_batches_tracked"]:
            mapped[f"decoder.input_proj.{i}.norm.{bn_key}"] = hf[f"model.decoder_input_proj.{i}.1.{bn_key}"]

    # --- query pos head ---
    for j in range(2):
        mapped[f"decoder.query_pos_head.layers.{j}.weight"] = hf[f"model.decoder.query_pos_head.layers.{j}.weight"]
        mapped[f"decoder.query_pos_head.layers.{j}.bias"] = hf[f"model.decoder.query_pos_head.layers.{j}.bias"]

    # --- enc output (proj + norm) ---
    mapped["decoder.enc_output.proj.weight"] = hf["model.enc_output.0.weight"]
    mapped["decoder.enc_output.proj.bias"] = hf["model.enc_output.0.bias"]
    mapped["decoder.enc_output.norm.weight"] = hf["model.enc_output.1.weight"]
    mapped["decoder.enc_output.norm.bias"] = hf["model.enc_output.1.bias"]

    # --- encoder heads ---
    mapped["decoder.enc_score_head.weight"] = hf["model.enc_score_head.weight"]
    mapped["decoder.enc_score_head.bias"] = hf["model.enc_score_head.bias"]

    for j in range(3):
        mapped[f"decoder.enc_bbox_head.layers.{j}.weight"] = hf[f"model.enc_bbox_head.layers.{j}.weight"]
        mapped[f"decoder.enc_bbox_head.layers.{j}.bias"] = hf[f"model.enc_bbox_head.layers.{j}.bias"]

    # --- decoder heads ---
    for i in range(6):
        mapped[f"decoder.dec_score_head.{i}.weight"] = hf[f"model.decoder.class_embed.{i}.weight"]
        mapped[f"decoder.dec_score_head.{i}.bias"] = hf[f"model.decoder.class_embed.{i}.bias"]
        for j in range(3):
            mapped[f"decoder.dec_bbox_head.{i}.layers.{j}.weight"] = hf[f"model.decoder.bbox_embed.{i}.layers.{j}.weight"]
            mapped[f"decoder.dec_bbox_head.{i}.layers.{j}.bias"] = hf[f"model.decoder.bbox_embed.{i}.layers.{j}.bias"]

    # --- denoising embedding ---
    if "decoder.denoising_class_embed.weight" in dst_sd and "model.denoising_class_embed.weight" in hf:
        mapped["decoder.denoising_class_embed.weight"] = _maybe_pad_embedding(
            dst_sd["decoder.denoising_class_embed.weight"], hf["model.denoising_class_embed.weight"]
        )

    # --- decoder layers ---
    for i in range(6):
        # cross attention (HF calls it encoder_attn)
        mapped[f"decoder.decoder.layers.{i}.cross_attn.num_points_scale"] = hf[
            f"model.decoder.layers.{i}.encoder_attn.n_points_scale"
        ]
        for k in ["sampling_offsets", "attention_weights", "value_proj", "output_proj"]:
            mapped[f"decoder.decoder.layers.{i}.cross_attn.{k}.weight"] = hf[
                f"model.decoder.layers.{i}.encoder_attn.{k}.weight"
            ]
            mapped[f"decoder.decoder.layers.{i}.cross_attn.{k}.bias"] = hf[
                f"model.decoder.layers.{i}.encoder_attn.{k}.bias"
            ]

        # norms / ffn
        mapped[f"decoder.decoder.layers.{i}.norm1.weight"] = hf[f"model.decoder.layers.{i}.self_attn_layer_norm.weight"]
        mapped[f"decoder.decoder.layers.{i}.norm1.bias"] = hf[f"model.decoder.layers.{i}.self_attn_layer_norm.bias"]
        mapped[f"decoder.decoder.layers.{i}.norm2.weight"] = hf[
            f"model.decoder.layers.{i}.encoder_attn_layer_norm.weight"
        ]
        mapped[f"decoder.decoder.layers.{i}.norm2.bias"] = hf[f"model.decoder.layers.{i}.encoder_attn_layer_norm.bias"]
        mapped[f"decoder.decoder.layers.{i}.linear1.weight"] = hf[f"model.decoder.layers.{i}.fc1.weight"]
        mapped[f"decoder.decoder.layers.{i}.linear1.bias"] = hf[f"model.decoder.layers.{i}.fc1.bias"]
        mapped[f"decoder.decoder.layers.{i}.linear2.weight"] = hf[f"model.decoder.layers.{i}.fc2.weight"]
        mapped[f"decoder.decoder.layers.{i}.linear2.bias"] = hf[f"model.decoder.layers.{i}.fc2.bias"]
        mapped[f"decoder.decoder.layers.{i}.norm3.weight"] = hf[f"model.decoder.layers.{i}.final_layer_norm.weight"]
        mapped[f"decoder.decoder.layers.{i}.norm3.bias"] = hf[f"model.decoder.layers.{i}.final_layer_norm.bias"]

        # self attention: pack q/k/v into in_proj
        q_w = hf[f"model.decoder.layers.{i}.self_attn.q_proj.weight"]
        k_w = hf[f"model.decoder.layers.{i}.self_attn.k_proj.weight"]
        v_w = hf[f"model.decoder.layers.{i}.self_attn.v_proj.weight"]
        q_b = hf[f"model.decoder.layers.{i}.self_attn.q_proj.bias"]
        k_b = hf[f"model.decoder.layers.{i}.self_attn.k_proj.bias"]
        v_b = hf[f"model.decoder.layers.{i}.self_attn.v_proj.bias"]
        in_w, in_b = _pack_qkv(q_w, k_w, v_w, q_b, k_b, v_b)
        mapped[f"decoder.decoder.layers.{i}.self_attn.in_proj_weight"] = in_w
        mapped[f"decoder.decoder.layers.{i}.self_attn.in_proj_bias"] = in_b
        mapped[f"decoder.decoder.layers.{i}.self_attn.out_proj.weight"] = hf[
            f"model.decoder.layers.{i}.self_attn.out_proj.weight"
        ]
        mapped[f"decoder.decoder.layers.{i}.self_attn.out_proj.bias"] = hf[
            f"model.decoder.layers.{i}.self_attn.out_proj.bias"
        ]

    # Validate shapes against destination state_dict
    ok = 0
    bad = []
    for k, v in mapped.items():
        if k not in dst_sd:
            bad.append((k, "missing_in_dst"))
            continue
        if tuple(dst_sd[k].shape) != tuple(v.shape):
            bad.append((k, f"shape_mismatch dst={tuple(dst_sd[k].shape)} src={tuple(v.shape)}"))
            continue
        ok += 1

    print(f"[convert] mapped_ok={ok} mapped_total={len(mapped)} dst_total={len(dst_sd)}")
    if bad:
        print(f"[convert] skipped_or_mismatched={len(bad)} (showing up to 30)")
        for item in bad[:30]:
            print(" -", item[0], item[1])

    # Create a minimal checkpoint: only include the mapped tensors that match.
    out_sd = {k: v for k, v in mapped.items() if k in dst_sd and tuple(dst_sd[k].shape) == tuple(v.shape)}
    output_pth.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": out_sd}, output_pth)
    print(f"[saved] {output_pth}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=str,
        default="rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r101vd_bbox_data_heron101.yml",
        help="Config to build this repo model (must be R101/384 + num_classes=17).",
    )
    p.add_argument("--repo-id", type=str, default="docling-project/docling-layout-heron-101")
    p.add_argument("--filename", type=str, default="model.safetensors")
    p.add_argument(
        "--output",
        type=str,
        default="rtdetrv2_pytorch/weights/heron101_converted.pth",
        help="Output .pth to pass as -t for training.",
    )
    p.add_argument(
        "--hf-cache-dir",
        type=str,
        default="",
        help="Optional Hugging Face cache dir (leave empty for default).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"huggingface_hub is required to download weights: {e}")

    cache_dir = args.hf_cache_dir or None
    hf_path = Path(
        hf_hub_download(repo_id=args.repo_id, filename=args.filename, cache_dir=cache_dir)
    ).resolve()

    convert(
        cfg_path=args.config,
        hf_safetensors=hf_path,
        output_pth=Path(args.output).expanduser().resolve(),
    )


if __name__ == "__main__":
    main()

