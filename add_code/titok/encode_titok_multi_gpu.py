#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-GPU TiTok Tokenization Script

This script encodes images to TiTok tokens using multiple GPUs with PyTorch DDP.
Tokens are saved as JSON files with shape (32, 1).

Example usage:

1. Single GPU:
   python encode_titok_multi_gpu.py \
       --a_dir /path/to/data \
       --ckpt_dir /path/to/titok_checkpoint \
       --batch_size 64 \
       --num_workers 8

2. Multi-GPU with torchrun:
   torchrun --nproc_per_node=8 encode_titok_multi_gpu.py \
       --a_dir /path/to/data \
       --ckpt_dir /path/to/titok_checkpoint \
       --batch_size 64 \
       --num_workers 8 \
       --check_every 500 \
       --skip_existing

3. Multi-node with torchrun:
   torchrun --nnodes=2 --nproc_per_node=8 \
       --master_addr=<master_ip> --master_port=<port> \
       encode_titok_multi_gpu.py \
       --a_dir /path/to/data \
       --ckpt_dir /path/to/titok_checkpoint \
       --batch_size 64

Directory structure:
  a_dir/
    ├── images/
    │   ├── episode_000001/
    │   │   ├── frame_001.png
    │   │   └── ...
    │   └── episode_000002/
    └── TiToken/  (output directory, created automatically)
        ├── episode_000001/
        │   ├── frame_001.json
        │   └── ...
        └── episode_000002/
"""

import os
import json
import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from omegaconf import OmegaConf

from modeling.titok import TiTok


class ShardedPngDataset(Dataset):
    """
    Dataset for sharded PNG images.
    
    Args:
        items: list of (global_idx:int, img_path:Path, rel_path:str, out_json:Path)
        target_size: optional (width, height) tuple for resizing images
    """
    def __init__(self, items, target_size=None):
        self.items = items
        self.target_size = target_size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        global_idx, img_path, rel, out_json = self.items[i]
        img = Image.open(img_path).convert("RGB")

        if self.target_size is not None:
            if img.size != self.target_size:
                img = img.resize(self.target_size, resample=Image.BICUBIC)

        arr = np.asarray(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1)  # (3,H,W)
        return x, int(global_idx), rel, str(out_json)


def collate_fn(batch):
    xs, gids, rels, outs = zip(*batch)
    return torch.stack(xs, 0), list(gids), list(rels), list(outs)


def save_tokens_as_32x1_json(tokens_32: torch.Tensor, out_json: str):
    """
    Save tokens as JSON with shape (32, 1).
    
    Args:
        tokens_32: token tensor of shape (32,)
        out_json: output JSON file path
    """
    # tokens_32: (32,)
    data = [[int(v)] for v in tokens_32.detach().cpu().reshape(-1).tolist()]  # (32, 1)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


@torch.no_grad()
def save_check_recon(model: TiTok, token_idx_1x1xT: torch.Tensor, check_path: Path, amp: bool, amp_dtype):
    """
    Decode tokens and save reconstructed image for quality checking.
    
    Args:
        model: TiTok model
        token_idx_1x1xT: token indices of shape (1, 1, T) on CUDA
        check_path: output path for reconstructed image
        amp: whether to use automatic mixed precision
        amp_dtype: data type for AMP (e.g., torch.float16)
    """
    # token_idx_1x1xT: (1, 1, T) on cuda
    if amp:
        with torch.autocast("cuda", dtype=amp_dtype, enabled=True):
            recon = model.decode_tokens(token_idx_1x1xT)
    else:
        recon = model.decode_tokens(token_idx_1x1xT)

    recon = torch.clamp(recon, 0.0, 1.0)
    recon = (recon * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
    check_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(recon).save(str(check_path))


def init_dist():
    """
    Initialize distributed training. torchrun automatically sets environment variables.
    
    Returns:
        rank: global rank of current process
        world_size: total number of processes
        local_rank: local rank on current node
    """
    # torchrun automatically sets these environment variables
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    return rank, world_size, local_rank


def infer_target_size_from_config(config):
    """
    Infer target image size from config. Returns None if not found.
    
    Args:
        config: OmegaConf config object
        
    Returns:
        (width, height) tuple or None
    """
    # Try to detect image size flexibly; return None if not found
    for key in ["image_size", "img_size", "resolution"]:
        try:
            v = OmegaConf.select(config, f"model.{key}")
        except Exception:
            v = None
        if v is None:
            continue
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return (int(v[0]), int(v[1]))
        if isinstance(v, int):
            return (int(v), int(v))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a_dir", type=str, required=True, help="Data directory (contains images/ folder)")
    ap.add_argument("--ckpt_dir", type=str, required=True, help="Local checkpoint directory (contains config.yaml and pytorch_model.bin)")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--check_every", type=int, default=500)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--no_amp", action="store_true")
    args = ap.parse_args()

    rank, world_size, local_rank = init_dist()
    torch.cuda.set_device(local_rank)
    device = "cuda"

    amp = not args.no_amp
    amp_dtype = torch.float16  # You can also use bf16 if your GPU supports it

    A_DIR = Path(args.a_dir)
    images_root = A_DIR / "images"
    out_root = A_DIR / "TiToken"
    check_dir = Path(args.ckpt_dir) / "check"

    # ---- Load model (each process loads its own copy to its GPU; normal for inference) ----
    config_path = Path(args.ckpt_dir) / "config.yaml"
    bin_path = Path(args.ckpt_dir) / "pytorch_model.bin"

    config = OmegaConf.load(str(config_path))
    model = TiTok(config)
    model.load_pretrained_weight(str(bin_path))
    model.eval()
    model.requires_grad_(False)
    model = model.to(device)

    assert getattr(model, "quantize_mode", None) == "vq", "Only VQ mode TiTok tokenizer is supported"

    target_size = infer_target_size_from_config(config)

    # ---- Build global file list (sorted) ----
    all_pngs = sorted(images_root.rglob("*.png"))
    if rank == 0:
        print(f"[Scan] Found {len(all_pngs)} pngs under {images_root}")
        print(f"[Dist] world_size={world_size}")

    # ---- Shard by global index ----
    items = []
    for global_idx, p in enumerate(all_pngs):
        if (global_idx % world_size) != rank:
            continue
        rel = str(p.relative_to(images_root))  # e.g., episode_xxx/frame_001.png
        out_json = (out_root / rel).with_suffix(".json")
        if args.skip_existing and out_json.exists():
            continue
        items.append((global_idx, p, rel, out_json))

    if rank == 0:
        # Other ranks can also print their workload (optional)
        pass
    print(f"[Rank {rank}] to process {len(items)} images on cuda:{local_rank}", flush=True)

    if len(items) == 0:
        if world_size > 1:
            torch.distributed.barrier()
        if rank == 0:
            print("[Done] Nothing to do.")
        return

    ds = ShardedPngDataset(items, target_size=target_size)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
        collate_fn=collate_fn,
        drop_last=False,
    )

    processed = 0

    for xs, gids, rels, out_jsons in dl:
        xs = xs.to(device, non_blocking=True)

        with torch.no_grad():
            if amp:
                with torch.autocast("cuda", dtype=amp_dtype, enabled=True):
                    _, enc_dict = model.encode(xs)
            else:
                _, enc_dict = model.encode(xs)

        token_indices = enc_dict["min_encoding_indices"]  # Expected shape: (B, 1, 32)
        if token_indices.dim() == 2:
            token_indices = token_indices.unsqueeze(1)

        B = token_indices.shape[0]

        # Write JSON + maybe check reconstruction
        for i in range(B):
            # Shape: (32,)
            t32 = token_indices[i, 0, :]
            save_tokens_as_32x1_json(t32, out_jsons[i])

            global_idx = gids[i]
            if args.check_every > 0 and (global_idx % args.check_every == 0):
                check_idx = global_idx // args.check_every
                check_path = check_dir / f"{check_idx}.png"
                # Shape: (1, 1, 32)
                save_check_recon(model, token_indices[i:i+1], check_path, amp=amp, amp_dtype=amp_dtype)

        processed += B
        if processed % 1000 == 0:
            print(f"[Rank {rank}] processed={processed}", flush=True)

    # ---- Summarize results ----
    total = processed
    if world_size > 1:
        t = torch.tensor([processed], device=device, dtype=torch.long)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        total = int(t.item())
        torch.distributed.barrier()

    if rank == 0:
        print(f"[Done] total processed across all ranks = {total}")
        print(f"[Done] tokens under: {out_root}")
        print(f"[Done] checks under: {check_dir}")


if __name__ == "__main__":
    main()
