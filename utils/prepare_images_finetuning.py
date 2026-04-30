from pathlib import Path
import shutil
import os
import argparse

def create_dataset_from_txt(txt_file_path: str, output_dir: str, input_dir: str):
    txt_path = Path(txt_file_path)
    dest_path = Path(output_dir)
    input_path = Path(input_dir)

    if not txt_path.is_file():
        print(f"[-] Error: file '{txt_file_path}' does not exist.")
        return
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    image_paths = [p.strip() for p in content.split(';') if p.strip()]

    succes_count = 0
    error_count = 0

    print(f"[*] Found {len(image_paths)} paths. Starting copy process to '{output_dir}'...")

    for image_path_str in image_paths:
        img_path = Path(image_path_str)
        source_path = input_path / img_path.name

        if img_path.is_file():
            dst_path = dest_path / img_path.name

            shutil.copy2(source_path, dst_path)
            succes_count += 1

        else:
            print(f"[!] Warning: File not found or invalid path -> {image_path_str}")
            error_count += 1

    print(f"\n Process completed!")
    print(f"Successfully copied: {succes_count}")
    print(f"Errors: {error_count}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy images from an input directory to an output directory based on a semicolon-separated TXT file.")

    parser.add_argument("txt_file", type=str, help="Path to the TXT file containing image names/paths separated by ';'")
    parser.add_argument("input_dir", type=str, help="Directory where the source images are located")
    parser.add_argument("output_dir", type=str, help="Directory where images will be copied")

    args = parser.parse_args()

    create_dataset_from_txt(
        txt_file_path=args.txt_file,
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )