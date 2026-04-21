import os
import glob
import cv2

def convert_bmp_to_png(dataset_path: str) -> None:
    """
    Finds all .bmp files in the specified directory and its subdirectories,
    converts them to .png format, and removes the original .bmp files.
    """
    # Create a recursive search pattern for .bmp files
    search_pattern = os.path.join(dataset_path, "**", "*.bmp")
    bmp_files = glob.glob(search_pattern, recursive=True)

    if not bmp_files:
        print("No .bmp files found. The dataset is ready.")
        return

    converted_count = 0
    for bmp_file in bmp_files:
        # Read the image using OpenCV
        img = cv2.imread(bmp_file)
        
        if img is not None:
            # Generate the new .png file path
            png_file = os.path.splitext(bmp_file)[0] + ".png"
            
            # Save the image as .png and delete the old .bmp file
            cv2.imwrite(png_file, img)
            os.remove(bmp_file)
            
            converted_count += 1
            print(f"Converted: {os.path.basename(bmp_file)} -> .png")
        else:
            print(f"Warning: Could not read {bmp_file}")

    print(f"\nConversion complete. Total files converted: {converted_count}")

if __name__ == "__main__":
    # Point this to the root of your custom dataset
    DATASET_ROOT = "./dataset/first_test/train"
    convert_bmp_to_png(DATASET_ROOT)