import os
import re
import argparse


def split_train_val(
    data_dir,
    file_pattern,
    train_output,
    val_output,
    val_ratio=0.01
):
    """
    Split tar files into train and validation sets
    
    Args:
        data_dir: Directory containing tar files
        file_pattern: Regex pattern to match files (e.g., r"^train-episode_(\d+)\.tar$")
        train_output: Output path for train file list
        val_output: Output path for validation file list
        val_ratio: Validation set ratio (default: 0.01 for 1%)
    """
    # Compile regex pattern
    pat = re.compile(file_pattern)
    
    # Collect all matching files
    all_paths = []
    for name in os.listdir(data_dir):
        m = pat.match(name)
        if m:
            # Extract episode ID (first capture group) and full path
            episode_id = int(m.group(1))
            full_path = os.path.join(data_dir, name)
            all_paths.append((episode_id, full_path))
    
    # Sort by episode ID for reproducibility
    all_paths.sort(key=lambda x: x[0])
    
    total_files = len(all_paths)
    if total_files == 0:
        print(f"Error: No files matching pattern '{file_pattern}' found in {data_dir}")
        return
    
    print(f"Found {total_files} files matching pattern")
    
    # Calculate validation set size (ceiling of val_ratio, ensure at least 1)
    val_count = max(1, int((total_files * val_ratio) + 0.999999))  # ceiling
    
    # Uniform sampling: select every 'step' file
    step = max(1, total_files // val_count)
    
    # Generate validation indices
    val_idx = set(range(0, total_files, step))
    # Trim to exact percentage (may be slightly more due to rounding)
    val_idx = set(sorted(val_idx)[:val_count])
    
    # Split into train and validation
    train_files, val_files = [], []
    for i, (_, path) in enumerate(all_paths):
        if i in val_idx:
            val_files.append(path)
        else:
            train_files.append(path)
    
    print(f"\nSplit summary:")
    print(f"  Training set: {len(train_files)} files")
    print(f"  Validation set: {len(val_files)} files")
    print(f"  Validation ratio: {len(val_files) / total_files * 100:.2f}% (target≈{val_ratio * 100:.1f}%)")
    print(f"  Sampling step: {step}")
    
    # Save to files
    with open(train_output, "w") as f:
        for p in train_files:
            f.write(p + "\n")
    
    with open(val_output, "w") as f:
        for p in val_files:
            f.write(p + "\n")
    
    print(f"\nFile lists saved to:")
    print(f"  Train: {train_output}")
    print(f"  Val: {val_output}")


def main():
    parser = argparse.ArgumentParser(
        description='Split tar files into train and validation sets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

python generate.py \\
    --data_dir /path/to/data \\
    --file_pattern "^train-episode_(\\d+)\\.tar$" \\
    --train_output ./train_files.txt \\
    --val_output ./val_files.txt \\
    --val_ratio 0.01
        """
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing tar files"
    )
    parser.add_argument(
        "--file_pattern",
        type=str,
        default=r"^train-episode_(\d+)\.tar$",
        help="Regex pattern to match files. Must contain at least one capture group for episode ID (default: '^train-episode_(\\d+)\\.tar$')"
    )
    parser.add_argument(
        "--train_output",
        type=str,
        default="train_files.txt",
        help="Output path for train file list (default: train_files.txt)"
    )
    parser.add_argument(
        "--val_output",
        type=str,
        default="val_files.txt",
        help="Output path for validation file list (default: val_files.txt)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.01,
        help="Validation set ratio, e.g., 0.01 for 1%% (default: 0.01)"
    )
    
    args = parser.parse_args()
    
    # Validate data directory
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory does not exist: {args.data_dir}")
        return
    
    if not os.path.isdir(args.data_dir):
        print(f"Error: {args.data_dir} is not a directory")
        return
    
    # Validate val_ratio
    if args.val_ratio <= 0 or args.val_ratio >= 1:
        print(f"Error: val_ratio must be between 0 and 1, got {args.val_ratio}")
        return
    
    # Split dataset
    split_train_val(
        data_dir=args.data_dir,
        file_pattern=args.file_pattern,
        train_output=args.train_output,
        val_output=args.val_output,
        val_ratio=args.val_ratio
    )


if __name__ == "__main__":
    main()
