# datamodules/active_learning.py
import torch
import albumentations as A
from pathlib import Path
from pandas import DataFrame
from anomalib.data.utils import Split

from datamodules.base import Supervision
from datamodules.base.datamodule import SSNDataModule
from datamodules.base.dataset import SSNDataset

class ActiveLearningDataset(SSNDataset):
    """
    Dataset class supporting both unsupervised and supervised active learning modes.
    - Unsupervised: Loads only normal images. Forces empty masks ("").
    - Supervised: Loads normal images + user-provided misclassified images with masks.
    """
    def __init__(self, root: Path, transform: A.Compose, split: Split, mode: str, **kwargs):
        self.mode = mode
        # Enable flips/augmentations ONLY for supervised fine-tuning to prevent overfitting
        flips = True if mode == "sup" else False
        super().__init__(
            transform=transform,
            root=root,
            split=split,
            flips=flips,
            normal_flips=False,
            supervision=Supervision.FULLY_SUPERVISED if mode == "sup" else Supervision.UNSUPERVISED,
            **kwargs
        )
    
    def make_dataset(self) -> tuple[DataFrame, DataFrame]:
        normal_samples = []
        anomalous_samples = []
        
        if self.split == Split.TRAIN:
            # ALWAYS load 'good' images for training. Pass "" to prevent mask loading.
            good_dir = self.root / "train" / "good"
            if good_dir.exists():
                for img_path in good_dir.glob("*.*"):
                    normal_samples.append([str(self.root), img_path.stem, self.split.value, str(img_path), "", 0, True])
            
            # IF in supervised mode, load the difficult samples dynamically added by the Active Sampler
            if self.mode == "sup":
                misc_img_dir = self.root / "active_pool" / "images"
                misc_mask_dir = self.root / "active_pool" / "masks"
                
                if misc_img_dir.exists() and misc_mask_dir.exists():
                    for img_path in misc_img_dir.glob("*.*"):
                        mask_path = misc_mask_dir / f"{img_path.stem}.bmp"
                        # Apply the mask if it exists, otherwise pass empty string
                        mask_str = str(mask_path) if mask_path.exists() else ""
                        anomalous_samples.append([str(self.root), img_path.stem, self.split.value, str(img_path), mask_str, 1, True])
        else:
            # TEST SPLIT: Load good and reject for evaluation metrics
            good_dir = self.root / "test" / "good"
            if good_dir.exists():
                for img_path in good_dir.glob("*.*"):
                    normal_samples.append([str(self.root), img_path.stem, self.split.value, str(img_path), "", 0, True])
            
            reject_dir = self.root / "test" / "reject"
            gt_dir = self.root / "ground_truth" / "reject"
            if reject_dir.exists():
                for img_path in reject_dir.glob("*.*"):
                    mask_path = gt_dir / f"{img_path.stem}_mask.bmp"
                    mask_str = str(mask_path) if mask_path.exists() else ""
                    anomalous_samples.append([str(self.root), img_path.stem, self.split.value, str(img_path), mask_str, 1, True])

        cols = ["path", "sample_id", "split", "image_path", "mask_path", "label_index", "is_segmented"]
        return DataFrame(normal_samples, columns=cols), DataFrame(anomalous_samples, columns=cols)


class ActiveLearningDataModule(SSNDataModule):
    """
    DataModule orchestrating the dataset loading based on the active learning mode.
    """
    def __init__(self, root: Path | str, mode: str = "unsup", image_size: tuple[int, int] = (256, 256), **kwargs):
        super().__init__(
            root=root,
            supervision=Supervision.FULLY_SUPERVISED if mode == "sup" else Supervision.UNSUPERVISED,
            image_size=image_size,
            **kwargs
        )
        self.mode = mode
        
        # Distance transform and mask dilation are required ONLY for supervised training masks
        dt = (3, 2) if mode == "sup" else None
        dilate = 7 if mode == "sup" else None

        self.train_data = ActiveLearningDataset(
            root=self.root, transform=self.transform_train, split=Split.TRAIN, mode=self.mode, dt=dt, dilate=dilate
        )
        self.test_data = ActiveLearningDataset(
            root=self.root, transform=self.transform_eval, split=Split.TEST, mode=self.mode
        )