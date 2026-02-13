#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Convert local image episode folders to WebDataset: one tar per episode (parallel processing)

import argparse
import os
import sys
import time
from pathlib import Path
from PIL import Image
import webdataset as wds
from multiprocessing import Pool, cpu_count


def _lanczos():
    # Pillow 10+ compatibility
    try:
        return Image.Resampling.LANCZOS
    except Exception:
        return Image.LANCZOS


def collect_episode_dirs(input_dir: Path) -> list[Path]:
    # Scan one level: input_dir/episode_*/
    eps = [d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")]
    eps.sort(key=lambda p: p.name)
    return eps


def collect_images_in_episode(ep_dir: Path, exts_lower: set[str]) -> list[Path]:
    # Scan one level: no subdirectories
    files = []
    for p in ep_dir.iterdir():
        if p.is_file() and p.suffix.lower() in exts_lower:
            files.append(p)
    # Sort by filename (frame_000001 zero-padding maintains frame order)
    files.sort(key=lambda p: p.name)
    return files


def process_one_episode(args):
    """
    Worker: one episode -> one tar
    """
    ep_dir_str, out_tar_str, extensions, resize_to, overwrite = args
    ep_dir = Path(ep_dir_str)
    out_tar = Path(out_tar_str)

    try:
        if out_tar.exists() and not overwrite:
            return (ep_dir.name, 0, f"skip (exists): {out_tar.name}")

        exts_lower = {e.lower() for e in extensions}
        img_files = collect_images_in_episode(ep_dir, exts_lower)
        if not img_files:
            return (ep_dir.name, 0, f"no images in {ep_dir.name}")

        out_tar.parent.mkdir(parents=True, exist_ok=True)

        # Write to temporary file, then atomically replace to avoid partial tar
        tmp_tar = out_tar.with_suffix(out_tar.suffix + ".tmp")
        if tmp_tar.exists():
            tmp_tar.unlink()

        with wds.TarWriter(str(tmp_tar)) as sink:
            for i, img_path in enumerate(img_files):
                img = Image.open(img_path).convert("RGB")
                if resize_to:
                    img = img.resize((resize_to, resize_to), _lanczos())

                sample = {
                    "__key__": "%08d" % i,  # 8-digit key for alignment
                    "jpg": img,             # Write as jpg
                }
                sink.write(sample)

        # Atomic replace
        os.replace(tmp_tar, out_tar)
        return (ep_dir.name, len(img_files), None)

    except Exception as e:
        return (ep_dir.name, 0, f"{repr(e)}")


def convert_local_images_to_wds(
    input_dir,
    output_dir,
    max_samples_per_shard=1000,  # Not used in "one tar per episode" mode, kept for signature compatibility
    split_name="train",
    has_labels=False,            # Episode-based packing does not support has_labels by default
    extensions=None,
    resize_to=256,
    num_workers=None,
    overwrite=False,
):
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']

    if has_labels:
        raise ValueError("Current mode is 'one tar per episode', does not support --has_labels.")

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    episode_dirs = collect_episode_dirs(input_path)
    print(f"[INFO] input_dir={input_path}", flush=True)
    print(f"[INFO] output_dir={output_path}", flush=True)
    print(f"[INFO] episodes={len(episode_dirs)}", flush=True)
    print(f"[INFO] split_name={split_name}", flush=True)
    if resize_to:
        print(f"[INFO] resize_to={resize_to}x{resize_to} (may upscale)", flush=True)
    else:
        print(f"[INFO] resize_to=None (keep original)", flush=True)

    if num_workers is None:
        num_workers = cpu_count()
    num_workers = max(1, int(num_workers))
    print(f"[INFO] num_workers={num_workers}", flush=True)

    # Tasks: one tar per episode
    tasks = []
    for ep_dir in episode_dirs:
        out_tar = output_path / f"{split_name}-{ep_dir.name}.tar"
        tasks.append((str(ep_dir), str(out_tar), extensions, resize_to, overwrite))

    if not tasks:
        print("[INFO] no episode dirs found.", flush=True)
        return

    now = time.time()
    ok = 0
    fail = 0
    skipped = 0
    total_imgs = 0

    print("[INFO] start processing episodes...", flush=True)

    with Pool(processes=num_workers) as pool:
        processed = 0
        # Episode-level tasks, chunksize=1 for better load balancing
        for ep_name, nimgs, err in pool.imap_unordered(process_one_episode, tasks, chunksize=1):
            processed += 1
            if err is None:
                ok += 1
                total_imgs += nimgs
                if processed % 10 == 0 or processed == len(tasks):
                    print(f"[OK] {processed}/{len(tasks)} latest={ep_name} imgs={nimgs} total_imgs={total_imgs}", flush=True)
            else:
                # Skip is a type of "non-failure"
                if err.startswith("skip (exists)"):
                    skipped += 1
                else:
                    fail += 1
                print(f"[WARN] {ep_name}: {err}", file=sys.stderr, flush=True)

    dt = time.time() - now
    print(f"\n[DONE] ok={ok}, skipped={skipped}, fail={fail}, episodes_total={len(tasks)}, total_imgs={total_imgs}", flush=True)
    print(f"[DONE] time={dt:.2f}s ({dt/60:.2f} min)", flush=True)

    # List generated tars
    tar_files = sorted(output_path.glob(f"{split_name}-episode_*.tar"))
    print(f"[DONE] generated {len(tar_files)} tar files under: {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate episodes into WebDataset: one tar per episode (parallel processing)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

python convert_local_images_to_wds.py \\
    --input_dir /path/to/episodes \\
    --output_dir /path/to/output \\
    --split_name train \\
    --resize_to 256 \\
    --num_workers 8 \\
    --overwrite
        """
    )
    
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Input directory containing episode_******/ folders")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Output directory for generated *.tar files")
    parser.add_argument("--split_name", type=str, default="train", 
                        help="Output tar prefix (default: train)")
    parser.add_argument("--extensions", nargs="+", default=None,
                        help="List of supported image extensions, e.g., .jpg .png (default: common formats)")
    parser.add_argument("--resize_to", type=int, default=256,
                        help="Target image size (square). Set to 0 for no resizing (default: 256)")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of parallel workers, defaults to cpu_count()")
    parser.add_argument("--overwrite", action="store_true", 
                        help="Overwrite existing tar files")
    parser.add_argument("--has_labels", action="store_true", 
                        help="(Not supported in this mode) Parameter kept for legacy script compatibility")

    args = parser.parse_args()

    if args.resize_to == 0:
        args.resize_to = None

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return

    convert_local_images_to_wds(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        split_name=args.split_name,
        has_labels=args.has_labels,
        extensions=args.extensions,
        resize_to=args.resize_to,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
