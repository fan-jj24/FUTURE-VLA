#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parallel export ms-swift style jsonl from robot parquet episodes.

This script converts robot trajectory data stored in parquet format into MS-Swift
compatible JSONL training data with parallel processing support.

Input structure:
A/
  data/chunk-***/episode_******.parquet
  meta/episodes.jsonl

Output structure (out_dir):
out_dir/
  data/episode_******.jsonl
  images/episode_******/A_image_episode_******_frame_******.png
  images/episode_******/A_wrist_image_episode_******_frame_******.png

JSONL (one file per episode):
- each line = one sample (sliding window at time t)
- images: 32 paths (input 16 frames, each frame: image + wrist)
- Recimage: 32 paths (future 16 frames, each frame: image + wrist)
- actions: (16, 7) with special tail padding:
    pad_action[:6] = 0.0, pad_action[6] = last_action[6]
- NO episode_index field in output json

Example Usage:

1. Basic usage with default settings:
   python swift_trans_pall_libero.py \\
       --dataset_root /path/to/LIBERO_data \\
       --name LIBERO \\
       --out_dir /path/to/output

2. Use specific QA template (type 1 only):
   python swift_trans_pall_libero.py \\
       --dataset_root /path/to/LIBERO_data \\
       --name LIBERO \\
       --out_dir /path/to/output \\
       --qa_mode 1

3. Parallel processing with 16 workers:
   python swift_trans_pall_libero.py \\
       --dataset_root /path/to/LIBERO_data \\
       --name LIBERO \\
       --out_dir /path/to/output \\
       --workers 16 \\
       --overwrite_images

4. Debug mode (process only first 10 episodes):
   python swift_trans_pall_libero.py \\
       --dataset_root /path/to/LIBERO_data \\
       --name LIBERO \\
       --out_dir /path/to/output \\
       --limit 10 \\
       --dry_run

5. Random QA template selection:
   python swift_trans_pall_libero.py \\
       --dataset_root /path/to/LIBERO_data \\
       --name LIBERO \\
       --out_dir /path/to/output \\
       --qa_mode random

QA Template Modes:
  - '1': Only use template 1 (observation -> action + future)
  - '2': Only use template 2 (action -> future state)
  - 'both': Generate both templates for each timestep (double the data)
  - 'alternate': Alternate between template 1 and 2
  - 'random': Randomly select template 1 or 2 for each timestep

Input Data Requirements:
  - Parquet files must contain: image, wrist_image, actions, frame_index, episode_index
  - meta/episodes.jsonl should contain: episode_index, tasks, length
  - Actions should be 7-dimensional: [x, y, z, roll, pitch, yaw, gripper]

Output Data Format:
  Each JSONL line contains:
  {
    "messages": [
      {"role": "user", "content": "<image> * 32 + Task description"},
      {"role": "assistant", "content": "Action tokens + <Rec> future states"}
    ],
    "images": ["images/episode_xxx/frame_001.png", ...],  # 32 paths
    "Recimage": ["images/episode_xxx/frame_017.png", ...],  # 32 paths
    "actions": [[x,y,z,r,p,y,g], ...]  # (16, 7) array
  }

Performance Tips:
  - Use --workers equal to CPU cores for optimal performance
  - Use --overwrite_images if images need to be regenerated
  - Use --dry_run to test configuration without writing files
  - Missing wrist images will be replaced with main camera images
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from concurrent.futures import ProcessPoolExecutor, as_completed


# ----------------------------
# Small utils
# ----------------------------
def log(msg: str) -> None:
    print(msg, flush=True)


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def ensure_len(vec: Any, n: int, pad_value: float = 0.0) -> List[float]:
    """Ensure list length == n by trunc/pad."""
    if vec is None:
        return [pad_value] * n
    if isinstance(vec, np.ndarray):
        vec = vec.tolist()
    if not isinstance(vec, (list, tuple)):
        return [pad_value] * n
    vec = list(vec)
    if len(vec) >= n:
        return [float(v) for v in vec[:n]]
    return [float(v) for v in vec] + [pad_value] * (n - len(vec))


def load_image_from_cell(cell: Any, dataset_root: Path) -> Optional[Image.Image]:
    """
    cell may be:
      - dict-like with keys 'bytes' and/or 'path'
      - numpy array
      - raw bytes
    """
    if cell is None:
        return None

    if isinstance(cell, dict):
        b = cell.get("bytes", None)
        p = cell.get("path", None)
        if b is not None:
            try:
                return Image.open(BytesIO(b)).convert("RGB")
            except Exception:
                return None
        if p:
            try:
                pth = Path(p)
                if not pth.is_absolute():
                    pth = dataset_root / pth
                if pth.exists():
                    return Image.open(pth).convert("RGB")
            except Exception:
                return None
        return None

    if isinstance(cell, (bytes, bytearray)):
        try:
            return Image.open(BytesIO(cell)).convert("RGB")
        except Exception:
            return None

    if isinstance(cell, np.ndarray):
        try:
            arr = cell.astype(np.uint8) if cell.dtype != np.uint8 else cell
            return Image.fromarray(arr).convert("RGB")
        except Exception:
            return None

    # last resort: mapping-like
    try:
        if hasattr(cell, "get"):
            b = cell.get("bytes", None)
            p = cell.get("path", None)
            if b is not None:
                return Image.open(BytesIO(b)).convert("RGB")
            if p:
                pth = Path(p)
                if not pth.is_absolute():
                    pth = dataset_root / pth
                if pth.exists():
                    return Image.open(pth).convert("RGB")
    except Exception:
        pass

    return None


def build_instruction_map(meta_episodes_jsonl: Path) -> Dict[int, str]:
    """
    episodes.jsonl line example:
    {"episode_index": 0, "tasks": ["..."], "length": 214}
    """
    mp: Dict[int, str] = {}
    if not meta_episodes_jsonl.exists():
        log(f"[WARN] meta file not found: {meta_episodes_jsonl}")
        return mp

    with meta_episodes_jsonl.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                epi = safe_int(obj.get("episode_index", -1), -1)
                tasks = obj.get("tasks", [])
                if isinstance(tasks, str):
                    inst = tasks
                elif isinstance(tasks, list) and tasks:
                    inst = " / ".join([str(t) for t in tasks if t is not None])
                else:
                    inst = ""
                if epi >= 0:
                    mp[epi] = inst
            except Exception as e:
                log(f"[WARN] meta parse failed at line {ln}: {e}")
    return mp


def make_message_templates(instruction: str) -> Tuple[str, str, str, str]:
    # user侧仍然用 <image>；assistant侧未来图用 <Recimage>
    imgs32 = "<image>" * 32
    recimgs32 = "<Recimage>" * 32

    user1 = f"{imgs32}Task: {instruction}\n\nBased on the observation, what action should be executed?"
    asst1 = f"The robot should execute<Act><action></Act>, which will lead to the next state<Rec>{recimgs32}</Rec>."

    user2 = f"{imgs32}Task: {instruction}\n\nIf we execute the action<Act><action></Act>, what will be the resulting state?"
    asst2 = f"After executing<Act><action></Act>, the system will transition to the state<Rec>{recimgs32}</Rec>."

    return user1, asst1, user2, asst2


def select_qa_type(t: int, mode: str) -> int:
    """
    return 1 or 2 or 3 (both)
    """
    mode = mode.lower()
    if mode == "1":
        return 1
    if mode == "2":
        return 2
    if mode == "both":
        return 3
    if mode == "random":
        return 1 if (t * 2654435761) % 2 == 0 else 2
    return 1 if (t % 2 == 0) else 2  # alternate default


def discover_parquets(dataset_root: Path) -> List[Path]:
    return sorted(dataset_root.glob("data/chunk-*/episode_*.parquet"))


# ----------------------------
# Worker globals (per process)
# ----------------------------
G_DATASET_ROOT: Optional[Path] = None
G_OUT_DIR: Optional[Path] = None
G_NAME: str = ""
G_QA_MODE: str = "both"
G_OVERWRITE_IMAGES: bool = False
G_DRY_RUN: bool = False
G_INSTRUCTION_MAP: Dict[int, str] = {}


def _init_worker(
    dataset_root: str,
    out_dir: str,
    name: str,
    meta_file: str,
    qa_mode: str,
    overwrite_images: bool,
    dry_run: bool,
) -> None:
    global G_DATASET_ROOT, G_OUT_DIR, G_NAME, G_QA_MODE, G_OVERWRITE_IMAGES, G_DRY_RUN, G_INSTRUCTION_MAP
    G_DATASET_ROOT = Path(dataset_root)
    G_OUT_DIR = Path(out_dir)
    G_NAME = name
    G_QA_MODE = qa_mode
    G_OVERWRITE_IMAGES = overwrite_images
    G_DRY_RUN = dry_run
    G_INSTRUCTION_MAP = build_instruction_map(Path(meta_file))


def _process_one_episode(parquet_path_str: str) -> Tuple[str, int]:
    """
    Return (episode_stem, num_samples_written)
    """
    assert G_DATASET_ROOT is not None and G_OUT_DIR is not None

    parquet_path = Path(parquet_path_str)
    episode_stem = parquet_path.stem  # episode_001234

    data_dir = G_OUT_DIR / "data"
    images_dir = G_OUT_DIR / "images" / episode_stem
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(parquet_path)
    needed_cols = ["image", "wrist_image", "actions", "frame_index", "episode_index"]
    for c in needed_cols:
        if c not in df.columns:
            raise RuntimeError(f"Missing column {c} in {parquet_path}")

    df = df.sort_values("frame_index").reset_index(drop=True)

    epi_index = safe_int(df["episode_index"].iloc[0], -1)
    instruction = G_INSTRUCTION_MAP.get(epi_index, "")
    if not instruction:
        # 并行时尽量少打印；需要的话你可以打开
        log(f"[WARN] instruction not found for episode_index={epi_index} ({episode_stem})")
        instruction = ""

    user1, asst1, user2, asst2 = make_message_templates(instruction)

    frames: List[Dict[str, Any]] = []
    missing_wrist = 0
    missing_image = 0

    for i in range(len(df)):
        row = df.iloc[i]
        frame_idx = safe_int(row["frame_index"], i)

        img = load_image_from_cell(row["image"], G_DATASET_ROOT)
        wrist = load_image_from_cell(row["wrist_image"], G_DATASET_ROOT)

        if img is None:
            missing_image += 1

        if wrist is None:
            missing_wrist += 1
            wrist = img  # wrist缺失时退化到img，保证列表长度恒定

        act = ensure_len(row["actions"], 7, pad_value=0.0)

        img_name = f"{G_NAME}_image_{episode_stem}_frame_{frame_idx:06d}.png"
        wrist_name = f"{G_NAME}_wrist_image_{episode_stem}_frame_{frame_idx:06d}.png"
        img_path = images_dir / img_name
        wrist_path = images_dir / wrist_name

        if not G_DRY_RUN:
            if img is not None and (G_OVERWRITE_IMAGES or (not img_path.exists())):
                img.save(img_path)
            if wrist is not None and (G_OVERWRITE_IMAGES or (not wrist_path.exists())):
                wrist.save(wrist_path)

        frames.append(
            {
                "frame_index": frame_idx,
                "image_path": img_path,
                "wrist_path": wrist_path,
                "action": act,
            }
        )

    T = len(frames)
    # 你想看缺失情况就打开这行（并行会有点刷屏）
    log(f"[EP {episode_stem}] frames={T}, missing_image={missing_image}, missing_wrist={missing_wrist}")

    def clamp_idx(x: int) -> int:
        if x < 0:
            return 0
        if x >= T:
            return T - 1
        return x

    out_jsonl = data_dir / f"{episode_stem}.jsonl"
    if G_DRY_RUN:
        return episode_stem, 0

    # action pad：前6维0，第7维=最后一帧第7维
    last_a7 = 0.0
    if T > 0:
        last_a7 = float(frames[-1]["action"][6])
    pad_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, last_a7]

    written = 0
    with out_jsonl.open("w", encoding="utf-8") as f:
        for t in range(T):
            # 输入窗口：t-15..t（头部不足重复0）
            in_ids = [clamp_idx(k) for k in range(t - 15, t + 1)]
            # 未来窗口：t+1..t+16（尾部不足重复last）
            out_ids = [clamp_idx(k) for k in range(t + 1, t + 17)]

            # actions chunk：t..t+15（尾部不足用特殊 pad_action）
            actions_chunk: List[List[float]] = []
            for k in range(t, t + 16):
                if k < T:
                    actions_chunk.append(frames[k]["action"])
                else:
                    actions_chunk.append(pad_action)

            # images / Recimage 各 32 个
            images_in: List[str] = []
            images_out: List[str] = []

            for k in in_ids:
                images_in.append(frames[k]["image_path"].relative_to(G_OUT_DIR).as_posix())
                images_in.append(frames[k]["wrist_path"].relative_to(G_OUT_DIR).as_posix())

            for k in out_ids:
                images_out.append(frames[k]["image_path"].relative_to(G_OUT_DIR).as_posix())
                images_out.append(frames[k]["wrist_path"].relative_to(G_OUT_DIR).as_posix())

            qa_type = select_qa_type(t, G_QA_MODE)

            def dump_one(u: str, a: str) -> None:
                nonlocal written
                obj = {
                    "messages": [
                        {"role": "user", "content": u},
                        {"role": "assistant", "content": a},
                    ],
                    "images": images_in,       # 32
                    "Recimage": images_out,    # 32
                    "actions": actions_chunk,  # (16,7)
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                written += 1

            if qa_type == 1:
                dump_one(user1, asst1)
            elif qa_type == 2:
                dump_one(user2, asst2)
            else:
                dump_one(user1, asst1)
                dump_one(user2, asst2)

    return episode_stem, written


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, required=True, help="Path to A/")
    ap.add_argument("--name", type=str, required=True, help="Dataset outer name A (used in image filename prefix)")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory")
    ap.add_argument(
        "--qa_mode",
        type=str,
        default="both",
        help="QA template mode: 1 | 2 | alternate | both | random (default: both)",
    )
    ap.add_argument("--overwrite_images", action="store_true", help="Overwrite existing png files")
    ap.add_argument("--limit", type=int, default=-1, help="Only process first N episodes (debug)")
    ap.add_argument("--dry_run", action="store_true", help="Do not write files")
    ap.add_argument("--workers", type=int, default=0, help="Num parallel workers (0=cpu_count)")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_file = (dataset_root / "meta" / "episodes.jsonl").resolve()

    parquets = discover_parquets(dataset_root)
    if args.limit > 0:
        parquets = parquets[: args.limit]

    workers = args.workers if args.workers and args.workers > 0 else (os.cpu_count() or 8)

    log(f"[INFO] dataset_root={dataset_root}")
    log(f"[INFO] out_dir={out_dir}")
    log(f"[INFO] meta_file={meta_file}")
    log(f"[INFO] found {len(parquets)} parquet episodes")
    log(f"[INFO] qa_mode={args.qa_mode}")
    log(f"[INFO] workers={workers}")

    # 并行跑：每个 episode 一个 task
    ok = 0
    failed = 0
    total_written = 0

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(
            str(dataset_root),
            str(out_dir),
            args.name,
            str(meta_file),
            args.qa_mode,
            args.overwrite_images,
            args.dry_run,
        ),
    ) as ex:
        futures = [ex.submit(_process_one_episode, str(p)) for p in parquets]

        for fu in as_completed(futures):
            try:
                episode_stem, written = fu.result()
                ok += 1
                total_written += written
                log(f"[OK] {episode_stem}: wrote {written} samples")
            except Exception as e:
                failed += 1
                log(f"[FAIL] {repr(e)}")

    log(f"\n[DONE] ok={ok}, failed={failed}, total_samples={total_written}")


if __name__ == "__main__":
    main()
