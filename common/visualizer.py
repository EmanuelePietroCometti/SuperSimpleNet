from pathlib import Path

import cv2
import numpy as np
from anomalib.data.utils import read_image
from matplotlib import pyplot as plt

from tqdm import tqdm


class Visualizer:
    def __init__(self, save_path: Path):
        self.save_path = save_path

    def visualize(self, results: dict, y_pred: np.ndarray, best_threshold: float):
        for image_path, mask_path, anomaly_map, score, seg_score, label, pred_lbl in tqdm(
            zip(
                results["image_path"],
                results["mask_path"],
                results["anomaly_map"],
                results["score"],
                results["seg_score"],
                results["label"],
                y_pred
            ),
            total=len(results["label"]),
        ):
            anomaly_map = anomaly_map.squeeze()
            h, w = anomaly_map.shape
            image = read_image(image_path, image_size=(h, w))
            # if no mask path -> all zeros
            if mask_path:
                gt_mask = read_image(mask_path, image_size=(h, w)).squeeze()
            else:
                gt_mask = np.zeros_like(anomaly_map)
            # images are normed to [0, 1] so we cut off at dinamic threshold
            pred_mask = anomaly_map >= best_threshold

            gt_label = "Anomalous" if label.item() else "Normal"
            pred_label_str = "Anomalous" if pred_lbl else "Normal"

            true_lbl = int(label.item())
            if true_lbl == 1 and pred_lbl == 1:
                category = "TP"
            elif true_lbl == 0 and pred_lbl == 0:
                category = "TN"
            elif true_lbl == 0 and pred_lbl == 1:
                category = "FP"
            elif true_lbl == 1 and pred_lbl == 0:
                category = "FN"

            # adjust height and width as 256 as baseline
            fig_h = h / 256 * 2
            fig_w = w / 256 * 12

            fig, plots = plt.subplots(1, 5, figsize=(fig_w, fig_h))
            for s_plt in plots:
                s_plt.axis("off")

            fig.tight_layout()

            plots[0].imshow(image)
            plots[0].title.set_text("Image")

            plots[1].imshow(gt_mask)
            plots[1].title.set_text(f"Ground truth.\n{gt_label}")

            plots[2].imshow(pred_mask)
            plots[2].title.set_text(f"Predicted mask\n(Th: {best_threshold:.3f})")

            plots[3].imshow(anomaly_map)
            plots[3].title.set_text(f"Anomaly map\nNorm.")

            plots[4].imshow(anomaly_map, vmax=1, vmin=0)
            plots[4].title.set_text(
                f"Anomaly map.\nScore: {round(score.item(), 4)}\nSScore: {round(seg_score.item(), 4)}"
            )

            # subdir name is parent's name
            subdir_name = Path(image_path).parent.name
            plot_name = f"{Path(image_path).stem}.png"

            # besides parent name also add anomalous or normal
            total_path = self.save_path / category / subdir_name
            total_path.mkdir(exist_ok=True, parents=True)

            plt.savefig(total_path / plot_name, bbox_inches="tight")

            ano_maps_dir = total_path / "anomaly_maps"
            ano_maps_dir.mkdir(exist_ok=True, parents=True)

            cv2.imwrite(str(ano_maps_dir / plot_name), anomaly_map.numpy() * 255)

            plt.close("all")
