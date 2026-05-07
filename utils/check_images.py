from pathlib import Path
from typing import Union, List

def check_unique_images(input_folder: Union[str, Path], folders_to_check: List[Union[str, Path]]) -> None:
    """
    Checks if .bmp images from the input folder exist in a given list of folders
    using set-based lookup for higher performance.
    """
    input_dir = Path(input_folder)
    
    existing_files = {}
    for folder in folders_to_check:
        folder_path = Path(folder)
        if folder_path.exists():
            for file in folder_path.glob("*.bmp"):
                existing_files[file.name] = folder_path

    duplicates_found = 0
    for img in input_dir.glob("*.bmp"):
        if img.name in existing_files:
            print(f"[CONFLICT] Image '{img.name}' already exists in: {existing_files[img.name]}")
            duplicates_found += 1

    if duplicates_found == 0:
        print("[SUCCESS] All images are unique and can be used!")
    else:
        print(f"[WARNING] Found {duplicates_found} duplicate(s). Check the list above!")



if __name__ == "__main__":
    input_dir = "D:/emanuele/Code/SuperSimpleNet/dataset/unlabeled_pool"
    output_folders_to_check = [
        "D:/emanuele/Code/SuperSimpleNet/dataset/train/good",
        "D:/emanuele/Code/SuperSimpleNet/dataset/test/good",
        "D:/emanuele/Code/SuperSimpleNet/dataset/test/reject",
        "D:/emanuele/Code/SuperSimpleNet/dataset/unlabeled_pool_0",
        "D:/emanuele/Code/SuperSimpleNet/dataset/active_pool/images"
    ]

    check_unique_images(input_dir, output_folders_to_check)