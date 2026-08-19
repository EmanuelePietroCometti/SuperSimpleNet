from pathlib import Path
import albumentations as A
import pandas as pd
from anomalib.data.utils import Split, InputNormalizationMethod
from pandas import DataFrame

from datamodules.base import Supervision
from datamodules.base.datamodule import SSNDataModule
from datamodules.base.dataset import SSNDataset


class MVTECDataset(SSNDataset):
    """
    Dataset class for MVTec AD dataset

    Args:
        root (Path): path to root of dataset
        category (str): one of 15 categories
        transform (A.Compose): transforms used for preprocessing
        split (Split): either train or test split
        debug (bool): debug flag for some debug printing
        supervision (Supervision): flag to signal the level of supervision for the dataset
    """

    def __init__(
        self,
        root: Path,
        category: str,
        transform: A.Compose,
        split: Split,
        normal_flips: bool = False,
        supervision: Supervision = Supervision.UNSUPERVISED,
        debug: bool = False,
    ) -> None:
        super().__init__(
            transform=transform,
            root=root,
            split=split,
            flips=False,
            normal_flips=normal_flips,
            supervision=supervision,
            debug=debug,
        )
        self.root_category = Path(root) / Path(category)

    def make_dataset(self) -> tuple[DataFrame, DataFrame]:
        """
        Custom parser to bypass Anomalib's hardcoded MVTec logic.
        Handles custom datasets with .bmp or .png files, flexible mask naming, 
        and custom ground_truth subfolders (train/test).
        """
        samples_list = []
        # Safely extract the string representation of the split (e.g., "train" or "test")
        split_str = self.split.value if hasattr(self.split, "value") else str(self.split)
        split_dir = self.root_category / split_str
        
        # Look for both .bmp and .png images to ensure compatibility
        valid_extensions = ("*.bmp", "*.png")
        image_paths = []
        for ext in valid_extensions:
            image_paths.extend(list(split_dir.rglob(ext)))
            
        for img_path in image_paths:
            label = img_path.parent.name
            mask_path = ""
        
            # Search for masks if the image is anomalous, regardless of the train/test split
            if label != "good":
                # CUSTOM path (yours): ground_truth / train / category
                mask_dir_custom = self.root_category / "ground_truth" / split_str / label
                # STANDARD MVTec path: ground_truth / category
                mask_dir_standard = self.root_category / "ground_truth" / label
                
                # Flexible list of possible mask filenames across both directories
                possible_masks = [
                    # Search first in your custom structure
                    mask_dir_custom / f"{img_path.stem}_mask.bmp",
                    mask_dir_custom / f"{img_path.stem}_mask.png",
                    mask_dir_custom / f"{img_path.stem}.bmp",
                    mask_dir_custom / f"{img_path.stem}.png",
                    # Fallback to standard MVTec structure (for future compatibility)
                    mask_dir_standard / f"{img_path.stem}_mask.bmp",
                    mask_dir_standard / f"{img_path.stem}_mask.png",
                    mask_dir_standard / f"{img_path.stem}.bmp",
                    mask_dir_standard / f"{img_path.stem}.png"
                ]
                
                mask_found = False
                for p_mask in possible_masks:
                    if p_mask.exists():
                        mask_path = str(p_mask)
                        mask_found = True
                        break
                        
                if not mask_found:
                    # Provide explicit feedback if a mask is missing to prevent silent failures
                    raise FileNotFoundError(
                        f"\n[!] Mask not found!\n"
                        f"Image: {img_path.name}\n"
                        f"Searched in:\n"
                        f"  1. {mask_dir_custom}\n"
                        f"  2. {mask_dir_standard}\n"
                        f"Ensure that the mask exists and is named correctly (.bmp or .png)"
                    )
            
            # Construct the row dictionary required by the DataLoader
            samples_list.append({
                "path": str(self.root_category),
                "split": split_str,
                "label": label,
                "image_path": str(img_path),
                "mask_path": mask_path,
                "label_index": 0 if label == "good" else 1
            })
            
        if not samples_list:
            raise RuntimeError(f"No images found in {split_dir}")
            
        samples_df = pd.DataFrame(samples_list)
        
        normal_samples = samples_df[samples_df["label_index"] == 0].reset_index(drop=True)
        anomalous_samples = samples_df[samples_df["label_index"] == 1].reset_index(drop=True)
        
        return normal_samples, anomalous_samples


class MVTec(SSNDataModule):
    """
    Datamodule for MVTec AD
    """

    def __init__(
        self,
        root: Path | str,
        category: str,
        image_size: tuple[int, int] | None = None,
        normalization: str | InputNormalizationMethod = InputNormalizationMethod.IMAGENET,
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        num_workers: int = 0,
        seed: int | None = None,
        normal_flips: bool = False,
        debug: bool = False,
        supervision: Supervision = Supervision.UNSUPERVISED
    ) -> None:
        print(f"Resolution set to: {image_size}")

        super().__init__(
            root=root,
            supervision=supervision,
            image_size=image_size,
            normalization=normalization,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            seed=seed,
            flips=False,
        )

        self.train_data = MVTECDataset(
            category=category,
            transform=self.transform_train,
            split=Split.TRAIN,
            root=root,
            debug=debug,
            normal_flips=normal_flips,
            supervision=supervision,
        )
        self.test_data = MVTECDataset(
            category=category,
            transform=self.transform_eval,
            split=Split.TEST,
            root=root,
            debug=debug,
            supervision=supervision,
        )