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
    """

    def __init__(
        self,
        root: Path,
        category: str,
        transform: A.Compose,
        split: Split,
        normal_flips: bool = False,
        debug: bool = False,
    ) -> None:
        super().__init__(
            transform=transform,
            root=root,
            split=split,
            flips=False,
            normal_flips=normal_flips,
            supervision=Supervision.UNSUPERVISED,
            debug=debug,
        )
        self.root_category = Path(root) / Path(category)

    def make_dataset(self) -> tuple[DataFrame, DataFrame]:
        """
        Custom parser to bypass Anomalib's hardcoded MVTec logic.
        Handles custom datasets with .bmp files and flexible mask naming.
        """
        samples_list = []
        # Safely extract the string representation of the split (e.g., "train" or "test")
        split_str = self.split.value if hasattr(self.split, "value") else str(self.split)
        split_dir = self.root_category / split_str
        
        # Look for all .bmp images in the specified directory
        for img_path in split_dir.rglob("*.bmp"):
            label = img_path.parent.name
            mask_path = ""
        
            # Match masks (ONLY for the anomalous test set)
            if split_str == "test" and label != "good":
                mask_dir = self.root_category / "ground_truth" / label
                
                # Search for the exact .bmp file with the _mask suffix
                mask_file = mask_dir / f"{img_path.stem}_mask.bmp"
                
                if mask_file.exists():
                    mask_path = str(mask_file)
                else:
                    # Provide explicit feedback if a mask is missing to prevent silent failures
                    raise FileNotFoundError(
                        f"Mask not found!\n"
                        f"Image: {img_path.name}\n"
                        f"Expected mask: {mask_file}"
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
            raise RuntimeError(f"No .bmp images found in {split_dir}")
            
        # Return the DataFrame expected by the datamodule
        samples_df = pd.DataFrame(samples_list)
        return samples_df, pd.DataFrame()


class MVTec(SSNDataModule):
    """
    Datamodule for MVTec AD

    Args:
        root (Path): path to root of dataset
        category (str): one of 15 categories
        image_size ( int | tuple[int, int] | None): image size in format of (h, w)
        normalization (str | InputNormalizationMethod): normalization method for images, defaults to imagenet
        train_batch_size (int): batch size used in training
        eval_batch_size (int): batch size used in test / inference
        num_workers (int): number of dataloader workers. Must be <= 1 for supervised
        seed (int | None): seed
        debug (bool): debug flag for some debug printing
    """

    def __init__(
        self,
        root: Path | str,
        category: str,
        image_size: tuple[int, int] | None = None,
        normalization: str
        | InputNormalizationMethod = InputNormalizationMethod.IMAGENET,
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        num_workers: int = 0,
        seed: int | None = None,
        normal_flips: bool = False,
        debug: bool = False,
    ) -> None:
        print(f"Resolution set to: {image_size}")

        super().__init__(
            root=root,
            supervision=Supervision.UNSUPERVISED,
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
        )
        self.test_data = MVTECDataset(
            category=category,
            transform=self.transform_eval,
            split=Split.TEST,
            root=root,
            debug=debug,
        )
