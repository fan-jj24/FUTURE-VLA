#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build MS-Swift Training Data with Fast TiToken Processing

This script processes episode JSONL files in-place:
1. Tokenizes actions using the Fast action tokenizer (AutoProcessor)
2. Replaces <Act> placeholders with tokenized action sequences
3. Loads TiToken visual tokens from pre-generated JSON files
4. Replaces <Rec> blocks with concatenated TiToken sequences (32 frames)
5. Converts relative image paths to absolute paths
6. Normalizes actions with configurable parameters

The script uses multiprocessing for efficient parallel processing of episodes.

Directory Structure Requirements:
  A/
    ├── data/
    │   ├── episode_000001.jsonl
    │   ├── episode_000002.jsonl
    │   └── ...
    ├── images/
    │   ├── episode_000001/
    │   │   ├── frame_001.png
    │   │   └── ...
    │   └── episode_000002/
    ├── TiToken/
    │   ├── episode_000001/
    │   │   ├── frame_001.json  (contains 32x1 token array)
    │   │   └── ...
    │   └── episode_000002/
    └── fast_tokenizer/  (local directory with AutoProcessor)

Input JSONL Format (per line):
  {
    "messages": [...],
    "images": ["images/episode_xxx/frame_001.png", ...],
    "Recimage": ["images/episode_xxx/frame_001.png", ...],  # 32 future frames
    "actions": [[d0,d1,...,d6], ...]  # (16, 7) array
  }

Output JSONL Format (per line):
  {
    "messages": [
      {
        "role": "user",
        "content": "Task: ... <Act>tok1 tok2 ... tokN</Act>"
      },
      {
        "role": "assistant", 
        "content": "<Rec>t0 t1 ... t31</Rec><Rec>t0 t1 ... t31</Rec>..."
      }
    ],
    "images": ["/absolute/path/to/image1.png", ...]
  }

Example Usage:

1. Basic usage with default settings:
   python build_ms_swift_fast_titoken.py \\
       --A /path/to/A \\
       --fast_repo /path/to/fast_tokenizer

2. Parallel processing with 16 workers:
   python build_ms_swift_fast_titoken.py \\
       --A /path/to/A \\
       --fast_repo /path/to/fast_tokenizer \\
       --workers 16 \\
       --batch_size 256

3. Custom pattern and strict mode (fail if TiToken missing):
   python build_ms_swift_fast_titoken.py \\
       --A /path/to/A \\
       --fast_repo /path/to/fast_tokenizer \\
       --pattern "data/train_episode_*.jsonl" \\
       --strict

4. Enable backup before processing:
   python build_ms_swift_fast_titoken.py \\
       --A /path/to/A \\
       --fast_repo /path/to/fast_tokenizer \\
       --backup_suffix .bak

5. Custom normalization parameters:
   python build_ms_swift_fast_titoken.py \\
       --A /path/to/A \\
       --fast_repo /path/to/fast_tokenizer \\
       --rpy_std_mult 10.0 \\
       --gripper_raw

Normalization Details:
  - Base action shape: (16, 7) where each row is [xyz, rpy, gripper]
  - Default mean/std are pre-computed from training data
  - --rpy_std_mult: Multiplier for rotation (rpy) standard deviation (default: 5.0)
  - --gripper_raw: If set, gripper normalization is disabled (mean=0, std=1)

Performance Tips:
  - Use --workers equal to CPU cores for optimal performance
  - Increase --batch_size if you have more RAM (reduces tokenizer overhead)
  - Use --backup_suffix "" to skip backup if you have original data saved elsewhere
  - TiToken JSON files are cached per worker (LRU cache size: 20000)
"""

import os
import re
import glob
import json
import shutil
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from multiprocessing import get_context
from collections import OrderedDict

import numpy as np
from transformers import AutoProcessor

# -------------------------
# Optional fast json
# -------------------------
def _maybe_orjson():
    try:
        import orjson  # type: ignore
        return orjson
    except Exception:
        return None

ORJSON = _maybe_orjson()

def json_loads(line: str) -> Any:
    if ORJSON is not None:
        return ORJSON.loads(line)
    return json.loads(line)

def dump_line(obj: Any) -> bytes:
    """Always return a single JSON line ending with '\n'."""
    if ORJSON is not None:
        return ORJSON.dumps(obj) + b"\n"
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


# -------------------------
# Worker globals
# -------------------------
PROC = None
MEAN = None
STD = None
EPS = 0
A_ROOT = None

# regex
ACT_RE = re.compile(r"<Act>\s*<action>\s*</Act>", flags=re.IGNORECASE)
REC_BLOCK_RE = re.compile(r"<Rec>.*?</Rec>", flags=re.DOTALL)

def init_worker(a_root: str, fast_repo: str, rpy_std_mult: float, gripper_raw: bool):
    """
    One processor per process; set normalization config.
    """
    global PROC, MEAN, STD, A_ROOT

    # avoid thread explosion
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    A_ROOT = os.path.abspath(a_root)
    PROC = AutoProcessor.from_pretrained(
        fast_repo,
        trust_remote_code=True,
        local_files_only=True,
    )

    base_mean = np.array(
        [
            0.06278137117624283,
            0.0868409126996994,
            -0.09037282317876816,
            0.0005407406715676188,
            0.005643361248075962,
            -0.005229088477790356,
            -0.04964079707860947,
        ],
        dtype=np.float32,
    )
    base_std = np.array(
        [
            0.3355240225791931,
            0.3784470558166504,
            0.44472837448120117,
            0.03924351558089256,
            0.06339313089847565,
            0.07797032594680786,
            0.9987710118293762,
        ],
        dtype=np.float32,
    )
    # xyz: unchanged; rpy: std * mult
    mean = base_mean.copy()
    std = base_std.copy()
    std[3:6] *= float(rpy_std_mult)

    # gripper raw: mean=0,std=1
    if gripper_raw:
        mean[6] = 0.0
        std[6] = 1.0

    MEAN = mean
    STD = std
    
# -------------------------
# Utilities
# -------------------------
def fix_task_text(s: str) -> str:
    # Task: ...\n\n  -> Task: .... . ...
    if "\n\n" in s and "Task:" in s:
        i = s.find("\n\n")
        s = s[:i].rstrip() + ". " + s[i + 2 :].lstrip()

    # Based on the observation -> Based on the images
    s = s.replace("Based on the observation,", "Based on the images,")

    # Type2: prefix "If we execute" with "Based on the images,"
    if "If we execute" in s and "Based on the images, If we execute" not in s:
        s = s.replace("If we execute", "Based on the images, If we execute", 1)

    return s


def extract_token_ids(batch_enc: Any) -> List[List[int]]:
    """
    Robustly extract token ids per sample from processor output.
    """
    if isinstance(batch_enc, dict):
        key = None
        for k in ("input_ids", "token_ids", "ids"):
            if k in batch_enc:
                key = k
                break
        if key is None:
            raise KeyError(f"Cannot find token ids key in processor output keys={list(batch_enc.keys())}")

        ids = batch_enc[key]
        if hasattr(ids, "detach"):
            ids = ids.detach().cpu().numpy()
        ids = np.asarray(ids)

        am = batch_enc.get("attention_mask", None)
        if am is not None:
            if hasattr(am, "detach"):
                am = am.detach().cpu().numpy()
            am = np.asarray(am)
            lens = am.sum(axis=-1).astype(np.int64)
            out: List[List[int]] = []
            for i in range(ids.shape[0]):
                L = int(lens[i])
                out.append(ids[i, :L].astype(np.int64).tolist())
            return out

        if ids.ndim == 1:
            return [ids.astype(np.int64).tolist()]
        return [ids[i].astype(np.int64).tolist() for i in range(ids.shape[0])]

    if isinstance(batch_enc, list):
        if len(batch_enc) == 0:
            return []
        if isinstance(batch_enc[0], int):
            return [list(map(int, batch_enc))]
        if isinstance(batch_enc[0], list):
            return [list(map(int, x)) for x in batch_enc]

    raise TypeError(f"Unsupported processor output type: {type(batch_enc)}")


def act_replace_all(s: str, act_str: str) -> str:
    return ACT_RE.sub(f"<Act>{act_str}</Act>", s)


class LRUCache:
    def __init__(self, maxsize: int = 20000):
        self.maxsize = maxsize
        self._d: OrderedDict[str, str] = OrderedDict()

    def get(self, key: str) -> Optional[str]:
        v = self._d.get(key, None)
        if v is not None:
            self._d.move_to_end(key)
        return v

    def put(self, key: str, value: str) -> None:
        self._d[key] = value
        self._d.move_to_end(key)
        if len(self._d) > self.maxsize:
            self._d.popitem(last=False)


def titoken_json_path_from_image_rel(rel_img: str, a_root: str) -> str:
    """
    images/episode_xxx/xxx.png -> A/TiToken/episode_xxx/xxx.json
    """
    rel = rel_img.replace("\\", "/")
    if rel.startswith("images/"):
        rel = rel[len("images/"):]
    rel_json = os.path.splitext(rel)[0] + ".json"
    return os.path.join(a_root, "TiToken", rel_json)


def load_titoken_str_for_image(rel_img: str, a_root: str, cache: LRUCache, strict: bool) -> str:
    """
    Load (32,1) list json and return "t0 t1 ... t31"
    """
    jp = titoken_json_path_from_image_rel(rel_img, a_root)
    cached = cache.get(jp)
    if cached is not None:
        return cached

    if not os.path.exists(jp):
        if strict:
            raise FileNotFoundError(jp)
        cache.put(jp, "")
        return ""

    with open(jp, "rb") as f:
        if ORJSON is not None:
            arr = ORJSON.loads(f.read())
        else:
            arr = json.loads(f.read().decode("utf-8"))

    toks: List[int] = []
    for x in arr:
        if isinstance(x, list) and len(x) > 0:
            toks.append(int(x[0]))
        else:
            toks.append(int(x))

    s = " ".join(str(t) for t in toks)
    cache.put(jp, s)
    return s


def build_rec_segments(recimage_list: List[str], a_root: str, cache: LRUCache, strict: bool) -> str:
    """
    32 Recimage -> "<Rec>...</Rec><Rec>...</Rec>..."
    """
    segs = []
    for rel in recimage_list:
        tok_str = load_titoken_str_for_image(rel, a_root, cache, strict)
        segs.append(f"<Rec>{tok_str}</Rec>")
    return "".join(segs)


def replace_rec_block_once(s: str, rec_concat: str) -> str:
    # Replace the first <Rec> ... </Rec> block with concatenated <Rec>...</Rec> segments
    return REC_BLOCK_RE.sub(rec_concat, s, count=1)


def abs_image_paths(images: List[str], a_root: str) -> List[str]:
    out = []
    for p in images:
        if os.path.isabs(p):
            out.append(p)
        else:
            out.append(os.path.join(a_root, p))
    return out


# -------------------------
# Episode processing (in-place)
# -------------------------
def process_one_episode_inplace(in_fp: str, batch_size: int, strict: bool, backup_suffix: str) -> Dict[str, int]:
    """
    Read one episode jsonl, rewrite it in-place via temp + atomic replace.
    """
    assert A_ROOT is not None and PROC is not None and MEAN is not None and STD is not None

    in_fp = str(in_fp)
    tmp_fp = in_fp + ".tmp"

    n_in = 0
    n_out = 0
    n_skipped = 0

    cache = LRUCache(maxsize=20000)

    # optional backup
    if backup_suffix:
        bak_fp = in_fp + backup_suffix
        if not os.path.exists(bak_fp):
            shutil.copy2(in_fp, bak_fp)

    # ensure tmp removed
    if os.path.exists(tmp_fp):
        os.remove(tmp_fp)

    batch_items: List[Dict[str, Any]] = []
    batch_actions: List[np.ndarray] = []

    mean = MEAN[None, None, :]
    std = (STD + EPS)[None, None, :]

    with open(tmp_fp, "wb") as w:
        def flush():
            nonlocal n_out, n_skipped
            if not batch_items:
                return

            actions = np.stack(batch_actions, axis=0).astype(np.float32)  # (B,16,7)
            z = (actions - mean) / std

            enc = PROC(z)
            ids_list = extract_token_ids(enc)
            if len(ids_list) != len(batch_items):
                raise RuntimeError(f"Processor batch size mismatch: {len(ids_list)} vs {len(batch_items)}")

            for item, ids in zip(batch_items, ids_list):
                act_str = " ".join(str(x) for x in ids)

                messages = item["messages"]
                images_rel = item["images"]
                rec_rel = item["recimage"]

                # build Rec segments (32 segments)
                rec_concat = build_rec_segments(rec_rel, A_ROOT, cache, strict)

                # edit messages
                for m in messages:
                    c = m.get("content", "")
                    c = fix_task_text(c)
                    c = act_replace_all(c, act_str)
                    if m.get("role") == "assistant":
                        c = replace_rec_block_once(c, rec_concat)
                    m["content"] = c

                out_obj = {
                    "messages": messages,
                    "images": abs_image_paths(images_rel, A_ROOT),
                }
                w.write(dump_line(out_obj))
                n_out += 1

            batch_items.clear()
            batch_actions.clear()

        with open(in_fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_in += 1
                try:
                    obj = json_loads(line)
                except Exception:
                    n_skipped += 1
                    continue

                if "messages" not in obj or "images" not in obj or "Recimage" not in obj or "actions" not in obj:
                    n_skipped += 1
                    continue

                try:
                    assert len(obj["actions"]) == 16
                    assert len(obj["actions"][0]) == 7
                    a = np.asarray(obj["actions"], dtype=np.float32)
                except Exception:
                    n_skipped += 1
                    continue
                if a.shape != (16, 7):
                    n_skipped += 1
                    continue

                batch_items.append(
                    {
                        "messages": obj["messages"],
                        "images": obj["images"],
                        "recimage": obj["Recimage"],
                    }
                )
                batch_actions.append(a)

                if len(batch_items) >= batch_size:
                    flush()

        flush()

    os.replace(tmp_fp, in_fp)
    return {"in": n_in, "out": n_out, "skipped": n_skipped}


def worker_entry(args: Tuple[str, int, bool, str]) -> Dict[str, Any]:
    in_fp, batch_size, strict, backup_suffix = args
    st = process_one_episode_inplace(in_fp, batch_size, strict, backup_suffix)
    st["file"] = os.path.basename(in_fp)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--A", type=str, required=True, help="Root A directory")
    ap.add_argument("--fast_repo", type=str, required=True, help="Local fast repo path (pure dir)")
    ap.add_argument("--pattern", type=str, default="data/episode_*.jsonl")
    ap.add_argument("--workers", type=int, default=max(os.cpu_count(), 1))
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--strict", action="store_true", help="Fail if TiToken json missing")
    ap.add_argument("--backup_suffix", type=str, default="", help="'' to disable backup")

    # normalization knobs
    ap.add_argument("--rpy_std_mult", type=float, default=5.0, help="Multiply std for d3..d5")
    ap.add_argument("--gripper_raw", action="store_true", help="Set gripper mean=0,std=1 (no normalization)")

    args = ap.parse_args()

    a_root = os.path.abspath(args.A)

    in_glob = os.path.join(a_root, args.pattern)
    in_files = sorted(glob.glob(in_glob))
    if not in_files:
        raise RuntimeError(f"No files matched: {in_glob}")

    tasks = [(fp, int(args.batch_size), bool(args.strict), str(args.backup_suffix)) for fp in in_files]

    ctx = get_context("spawn")
    with ctx.Pool(
        processes=int(args.workers),
        initializer=init_worker,
        initargs=(a_root, args.fast_repo, float(args.rpy_std_mult), bool(args.gripper_raw)),
    ) as pool:
        total_in = total_out = total_skipped = 0
        for st in pool.imap_unordered(worker_entry, tasks, chunksize=1):
            total_in += st["in"]
            total_out += st["out"]
            total_skipped += st["skipped"]
            print(f"[{st['file']}] in={st['in']} out={st['out']} skipped={st['skipped']}", flush=True)

    print(f"\nDONE. episodes={len(in_files)} total_in={total_in} total_out={total_out} total_skipped={total_skipped}")
    print("All files rewritten in-place under:", os.path.join(a_root, "data"))


if __name__ == "__main__":
    main()
