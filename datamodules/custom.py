from pathlib import Path
import albumentations as A
from pandas import DataFrame
from anomalib.data.utils import Split

from datamodules.base import Supervision
from datamodules.base.datamodule import SSNDataModule
from datamodules.base.dataset import SSNDataset

class CustomUnsupervisedDataset(SSNDataset):
    """
    Dataset for unsupervised training. Loads normal data for training and 
    normal/anomalous data for testing without requiring masks during train.
    """
    def __init__(self, root: Path, transform: A.Compose, split: Split, **kwargs):
        super().__init__(
            transform=transform,
            root=root,
            split=split,
            flips=False,
            normal_flips=False,
            supervision=Supervision.UNSUPERVISED,
            **kwargs
        )
    
    def make_dataset(self) -> tuple[DataFrame, DataFrame]:
        normal_samples = []
        anomalous_samples = []
        
        if self.split == Split.TRAIN:
            # Load only from train/good for unsupervised training
            good_dir = self.root / "train" / "good"
            if good_dir.exists():
                for img_path in good_dir.glob("*.*"):
                    normal_samples.append([
                        str(self.root), img_path.stem, self.split.value, str(img_path), "", 0, True
                    ])
        else:
            # Load test/good and test/reject for evaluation
            good_dir = self.root / "test" / "good"
            if good_dir.exists():
                for img_path in good_dir.glob("*.*"):
                    normal_samples.append([
                        str(self.root), img_path.stem, self.split.value, str(img_path), "", 0, True
                    ])
            
            reject_dir = self.root / "test" / "reject"
            gt_dir = self.root / "ground_truth" / "reject"
            if reject_dir.exists():
                for img_path in reject_dir.glob("*.*"):
                    # Handle mask if present (for metric evaluation only)
                    mask_path = gt_dir / f"{img_path.stem}_mask.png" 
                    mask_str = str(mask_path) if mask_path.exists() else ""
                    anomalous_samples.append([
                        str(self.root), img_path.stem, self.split.value, str(img_path), mask_str, 1, True
                    ])

        cols = ["path", "sample_id", "split", "image_path", "mask_path", "label_index", "is_segmented"]
        return DataFrame(normal_samples, columns=cols), DataFrame(anomalous_samples, columns=cols)


class CustomSupervisedDataset(SSNDataset):
    """
    Dataset for supervised fine-tuning. Loads hard negatives from the 
    'misclassified' folder and balances them with normal samples from 'train/good'.
    """
    def __init__(self, root: Path, transform: A.Compose, split: Split, **kwargs):
        super().__init__(
            transform=transform,
            root=root,
            split=split,
            flips=True,
            normal_flips=False,
            supervision=Supervision.FULLY_SUPERVISED,
            **kwargs
        )
    
    def make_dataset(self) -> tuple[DataFrame, DataFrame]:
        normal_samples = []
        anomalous_samples = []
        
        # Load user-provided misclassified anomalies
        misc_img_dir = self.root / "misclassified" / "images"
        misc_mask_dir = self.root / "misclassified" / "masks"
        
        if misc_img_dir.exists() and misc_mask_dir.exists():
            for img_path in misc_img_dir.glob("*.*"):
                mask_path = misc_mask_dir / f"{img_path.stem}.png" # Assumes mask has same name as image
                if mask_path.exists():
                    anomalous_samples.append([
                        str(self.root), img_path.stem, self.split.value, str(img_path), str(mask_path), 1, True
                    ])
                    
        # Load normal samples to allow the dataset to balance positive/negative samples
        good_dir = self.root / "train" / "good"
        if good_dir.exists():
            for img_path in good_dir.glob("*.*"):
                normal_samples.append([
                    str(self.root), img_path.stem, self.split.value, str(img_path), "", 0, True
                ])

        cols = ["path", "sample_id", "split", "image_path", "mask_path", "label_index", "is_segmented"]
        return DataFrame(normal_samples, columns=cols), DataFrame(anomalous_samples, columns=cols)


class CustomDataModule(SSNDataModule):
    """
    DataModule that switches between Unsupervised and Supervised logic.
    """
    def __init__(self, root: Path | str, mode: str = "unsup", image_size: tuple[int, int] = (256, 256), **kwargs):
        super().__init__(
            root=root,
            supervision=Supervision.UNSUPERVISED if mode == "unsup" else Supervision.FULLY_SUPERVISED,
            image_size=image_size,
            **kwargs
        )
        self.mode = mode

        if self.mode == "unsup":
            self.train_data = CustomUnsupervisedDataset(root=self.root, transform=self.transform_train, split=Split.TRAIN)
            self.test_data = CustomUnsupervisedDataset(root=self.root, transform=self.transform_eval, split=Split.TEST)
        else:
            # Add Distance Transform and Dilation parameters for supervised training masks
            self.train_data = CustomSupervisedDataset(root=self.root, transform=self.transform_train, split=Split.TRAIN, dt=(3, 2), dilate=7)
            # Evaluate on standard test set to measure the improvement
            self.test_data = CustomUnsupervisedDataset(root=self.root, transform=self.transform_eval, split=Split.TEST)