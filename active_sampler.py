import torch 
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model.supersimplenet import SuperSimpleNet

class UnlabelledPoolDataset(Dataset):
    """
    Dataset for scanning unannotated raw images.
    """
    def __init__(self, pool_dir: Path, transform: A.Compose):
        self.image_paths = list(pool_dir.glob("*.*"))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        img_array = np.array(image)
        transformed = self.transform(image=img_array)
        return transformed["image"], str(image_path)
    
@torch.no_grad()
def extract_active_samples(
    model_weights_path: str,
    unlabeled_pool_dir: str,
    output_misclassified_dir: str,
    budget: int = 20,
    threshold: float = 0.5,
):
    """
    Executes the Hybrid Active Learning logic (Uncertainty + Coreset Diversity)
    to select the best hard-negative samples for human annotation.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = {
        "backbone": "wide_resnet50_2", 
        "layers": ["layer1", "layer2", "layer3"], 
        "image_size": (256, 256)
    }

    model = SuperSimpleNet(
        image_size=config["image_size"],
        config=config
    ).to(device)
    
    model.load_model(model_weights_path)
    model.eval()

    # Corrected: A.Compose needs a capital 'C'
    transform = A.Compose([
        A.Resize(*config["image_size"]),
        A.Normalize(),
        ToTensorV2()
    ])

    pool_dataset = UnlabelledPoolDataset(pool_dir=Path(unlabeled_pool_dir), transform=transform)
    if len(pool_dataset) == 0:
        print("No unlabelled samples found in the pool directory.")
        return
    
    pool_loader = DataLoader(pool_dataset, batch_size=16, shuffle=False, num_workers=4)

    all_paths = []
    all_scores = []
    all_features = []

    print(f"Scanning {len(pool_dataset)} unannotated images...")
    for images, paths in pool_loader:
        images = images.to(device)
        anomaly_map, score = model(images)

        pooled_features = torch.nn.functional.adaptive_avg_pool2d(anomaly_map, (1, 1)).squeeze()

        if pooled_features.dim() == 1:
            pooled_features = pooled_features.unsqueeze(0)

        all_paths.extend(paths)
        all_scores.extend(torch.sigmoid(score).cpu().numpy())
        all_features.extend(pooled_features.cpu().numpy())

    all_scores = np.array(all_scores)
    all_features = np.array(all_features)

    # Uncertainty calculation: 1.0 means most uncertain (closest to threshold)
    uncertainity_scores = 1.0 - np.abs(all_scores - threshold) * 2.0

    candidate_multiplier = 3
    num_candidates = min(len(all_scores), budget * candidate_multiplier)
    candidate_indices = np.argsort(uncertainity_scores)[::-1][:num_candidates]

    candidate_features = all_features[candidate_indices]
    candidate_paths = [all_paths[i] for i in candidate_indices]

    print(f"Extracting top {budget} diverse samples using KMeans clustering...")
    actual_budget = min(budget, len(candidate_features))

    kmeans = KMeans(n_clusters=actual_budget, random_state=42, n_init=10)
    kmeans.fit(candidate_features)

    selected_paths = []
    for i in range(actual_budget):
        cluster_center = kmeans.cluster_centers_[i]
        distances = np.linalg.norm(candidate_features - cluster_center, axis=1)
        closest_index = np.argmin(distances)
        selected_paths.append(candidate_paths[closest_index])

    out_dir = Path(output_misclassified_dir) / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Selected samples copied to {out_dir}...")
    for path in selected_paths:
        file_path = Path(path)
        dest_path = out_dir / f"{file_path.stem}.bmp"

        img = Image.open(file_path).convert("RGB")
        img.save(dest_path, format="BMP")
        print(f" -> {dest_path.name}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python active_sampler.py <weights_path.pt> <unlabeled_pool_dir> <output_active_pool_dir>")
        sys.exit(1)
        
    extract_active_samples(sys.argv[1], sys.argv[2], sys.argv[3])