import json
import shutil
import argparse
import random
from pathlib import Path

def setup_al_structure(target_path: Path):
    """
    Creates the required directory structure for the Active Learning pipeline.
    """
    paths = [
        "train/good",
        "test/good",
        "test/reject",
        "ground_truth/reject",
        "unlabeled_pool/images",
        "active_pool/images",
        "active_pool/masks"
    ]
    for p in paths:
        (target_path / p).mkdir(parents=True, exist_ok=True)

def get_random_files(src_dir: Path, num_samples: int, exclude_set: set = None) -> list[Path]:
    """
    Retrieves a randomized list of .bmp files from a directory, 
    optionally excluding already processed files.
    """
    if not src_dir.exists():
        print(f"[WARNING] Source directory not found: {src_dir}")
        return []
    
    # Collect all .bmp files
    all_files = [f for f in src_dir.glob("*.bmp") if f.is_file()]
    
    # Remove files that have already been allocated to test/good or test/reject
    if exclude_set:
        all_files = [f for f in all_files if f not in exclude_set]
        
    # Shuffle the list randomly
    random.shuffle(all_files)
    
    # Return the requested number of samples, or all if num_samples is None
    if num_samples is not None and num_samples > 0:
        return all_files[:num_samples]
    
    return all_files

def copy_files(file_paths: list[Path], dst_dir: Path):
    """
    Copies a list of files to the target directory.
    """
    for file_path in file_paths:
        shutil.copy2(file_path, dst_dir / file_path.name)

def main():
    parser = argparse.ArgumentParser(description="Dataset Reorganizer for Active Learning")
    parser.add_argument("src", type=str, help="Source dataset root path")
    parser.add_argument("dst", type=str, help="Destination Active Learning dataset path")
    parser.add_argument("--config", type=str, default="config.json", help="JSON config file")
    
    args = parser.parse_args()
    src_root = Path(args.src)
    dst_root = Path(args.dst)
    
    with open(args.config, 'r') as f:
        config = json.load(f)
        
    # Ensure reproducibility across different runs
    random.seed(config.get("random_seed", 42))
    
    setup_al_structure(dst_root)
    
    # Memory set to track files allocated to supervised test sets
    used_test_files = set()
    
    # ---------------------------------------------------------
    # TRAIN GOOD (Can mix multiple sources, e.g. good + dust)
    # ---------------------------------------------------------
    for train_cfg in config["train_good"]:
        train_src = src_root / train_cfg["source_dir"]
        train_files = get_random_files(train_src, train_cfg["num_samples"])
        copy_files(train_files, dst_root / "train" / "good")
        print(f"-> Copied {len(train_files)} samples from {train_cfg['source_dir']} to train/good")
    
    # ---------------------------------------------------------
    # TEST GOOD (Can mix multiple sources, e.g. good + dust)
    # ---------------------------------------------------------
    for test_good_cfg in config["test_good"]:
        test_good_src = src_root / test_good_cfg["source_dir"]
        test_good_files = get_random_files(test_good_src, test_good_cfg["num_samples"])
        
        used_test_files.update(test_good_files) # Track to prevent leakage into unlabeled_pool
        copy_files(test_good_files, dst_root / "test" / "good")
        print(f"-> Copied {len(test_good_files)} samples from {test_good_cfg['source_dir']} to test/good")
    
    # ---------------------------------------------------------
    # TEST REJECT (Defects)
    # ---------------------------------------------------------
    for reject_cfg in config["test_reject"]:
        cat_src = src_root / reject_cfg["source_dir"]
        cat_files = get_random_files(cat_src, reject_cfg["num_samples"])
        
        used_test_files.update(cat_files) # Track to prevent leakage into unlabeled_pool
        copy_files(cat_files, dst_root / "test" / "reject")
        print(f"-> Copied {len(cat_files)} {reject_cfg['name']} samples to test/reject")
        
    # ---------------------------------------------------------
    # UNLABELED POOL (Active Learning source)
    # ---------------------------------------------------------
    pool_files = []
    
    # Iterate exclusively through the test directories defined in config
    for pool_src_str in config["unlabeled_pool"]["source_dirs"]:
        pool_src = src_root / pool_src_str
        
        # Get all remaining files, explicitly passing the exclusion set
        remaining_files = get_random_files(pool_src, num_samples=None, exclude_set=used_test_files)
        pool_files.extend(remaining_files)
        
    # Final global shuffle to mix different defect types and good samples in the pool
    random.shuffle(pool_files)
    
    if not config["unlabeled_pool"].get("take_all_remaining", True):
        max_pool = config["unlabeled_pool"].get("budget_al", len(pool_files))
        pool_files = pool_files[:max_pool]
        
    copy_files(pool_files, dst_root / "unlabeled_pool" / "images")
    print(f"-> Unlabeled pool populated with {len(pool_files)} unseen images from test folders.")
    print(f"\n[COMPLETE] Active Learning Dataset ready in: {dst_root}")

if __name__ == "__main__":
    main()