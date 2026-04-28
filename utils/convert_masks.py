import os
import glob
import cv2
import numpy as np
import argparse


def convert_masks(input_dir: str, output_dir: str):
    """
    Converts low_intenisty mask images from MVTEC DL Tool into visibile binary masks and save them in BMP format.
    """
    os.makedirs(output_dir, exist_ok=True)

    mask_paths = []
    valid_extensions = [".bmp", ".BMP"]

    for ext in valid_extensions:
        search_pattern = os.path.join(input_dir, f"*{ext}")
        mask_paths.extend(glob.glob(search_pattern))
    
    if not mask_paths:
        print(f"[WARNING] No mask files found in '{input_dir}' with specified extensions.")
        return
    
    for path in mask_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"[WARNING] Could not read mask image: {path}")
            continue

        visible_mask = np.where(img > 0, 255, 0).astype(np.uint8)
        filename = os.path.basename(path)
        filename_no_ext = os.path.splitext(filename)[0]
        out_path = os.path.join(output_dir, f"{filename_no_ext}.bmp")
        cv2.imwrite(out_path, visible_mask)
        print(f"Processed {filename} -> {os.path.basename(out_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Argument for converting anomaly images pipeline")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Insert manually the input dir path"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Insert manually the output dir path"
    )
    args = parser.parse_args()
    convert_masks(input_dir=parser.get("input_dir", "dataset/active_pool/masks"), output_dir=parser.get("output_dir", "dataset/active_pool/masks"))