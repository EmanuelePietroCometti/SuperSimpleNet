import os
from anomalib.data.mvtec import make_mvtec_dataset

def debug_anomalib():
    """
    Directly calls the anomalib dataset builder to debug why it returns 0 samples.
    """
    # Define your exact path here (adjust if your root is different)
    # E.g., if first_test is inside 'dataset', use "./dataset/first_test"
    dataset_path = "./dataset/first_test"
    
    print(f"Checking absolute path: {os.path.abspath(dataset_path)}\n")
    
    try:
        # Load the dataset dataframe without enforcing the 'train' split yet
        # We pass split=None to see EVERY image anomalib manages to detect
        df_all = make_mvtec_dataset(dataset_path, split=None)
        
        print(f"Total images found by anomalib in all folders: {len(df_all)}")
        
        if len(df_all) > 0:
            print("\nBreakdown by split (train vs test):")
            print(df_all['split'].value_counts())
            
            # If anomalib found them but labeled them wrong (e.g. 'Train' instead of 'train')
            # this will reveal the bug.
            train_samples = df_all[df_all['split'] == 'train']
            print(f"\nImages officially recognized for 'train' split: {len(train_samples)}")
            
            if len(train_samples) == 0:
                print("\nWARNING: Images found, but none are marked as 'train'.")
                print("Check if your 'train' folder has capital letters (e.g., 'Train').")
                
        else:
            print("\nCRITICAL: Anomalib found 0 images overall.")
            print("This means the root path is wrong, or files are not .png")
            
    except Exception as e:
        print(f"Anomalib crashed with error: {e}")

if __name__ == "__main__":
    debug_anomalib()