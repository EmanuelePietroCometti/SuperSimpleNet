import copy
import json
from pathlib import Path

import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import precision_recall_curve, confusion_matrix
from sklearn.metrics import roc_curve, precision_score, recall_score, f1_score
import numpy as np

import torch
from anomalib.utils.metrics import AUROC, AUPRO
from torchmetrics import Metric, AveragePrecision
from pytorch_lightning import LightningDataModule
from tqdm import tqdm
import pandas as pd

from common.results_writer import ResultsWriter
from common.visualizer import Visualizer
from datamodules import sensum, ksdd2
from datamodules.base import Supervision
from datamodules.mvtec import MVTec
from datamodules.visa import Visa
from datamodules.ksdd2 import KSDD2
from datamodules.sensum import Sensum
from model.supersimplenet import SuperSimpleNet
import argparse

@torch.no_grad()
def eval(
    model: SuperSimpleNet,
    datamodule: LightningDataModule,
    device: str,
    image_metrics: dict[str, Metric],
    pixel_metrics: dict[str, Metric],
    normalize: bool = True,
    image_save_path: Path = None,
    score_save_path: Path = None,
):
    model.to(device)
    model.eval()

    test_loader = datamodule.test_dataloader()
    results = {
        "anomaly_map": [],
        "gt_mask": [],
        "score": [],
        "seg_score": [],
        "label": [],
        "image_path": [],
        "mask_path": [],
    }
    
    for batch in tqdm(test_loader, position=0, leave=True):
        image_batch = batch["image"].to(device)
        anomaly_map, anomaly_score = model.forward(image_batch)

        anomaly_map = anomaly_map.detach().cpu()
        anomaly_score = anomaly_score.detach().cpu()

        results["anomaly_map"].append(anomaly_map)
        results["gt_mask"].append(batch["mask"].detach().cpu())

        results["score"].append(torch.sigmoid(anomaly_score))
        results["seg_score"].append(
            anomaly_map.reshape(anomaly_map.shape[0], -1).max(dim=1).values
        )
        results["label"].append(batch["label"].detach().cpu())

        results["image_path"].extend(batch["image_path"])
        results["mask_path"].extend(batch["mask_path"])

    # Global concatenation
    results["anomaly_map"] = torch.cat(results["anomaly_map"])
    results["score"] = torch.cat(results["score"])
    results["seg_score"] = torch.cat(results["seg_score"])
    results["gt_mask"] = torch.cat(results["gt_mask"])
    results["label"] = torch.cat(results["label"])

    # 1. Clean NaNs to safeguard the raw metrics
    am_raw = torch.nan_to_num(results["anomaly_map"], nan=0.0)
    score_raw = torch.nan_to_num(results["score"], nan=0.0)
    seg_score_raw = torch.nan_to_num(results["seg_score"], nan=0.0)
    
    results_dict = {}

    # ==========================================
    # --- SKLEARN RAW METRICS COMPUTATION ---
    # ==========================================
    from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score, f1_score, average_precision_score
    import numpy as np

    y_true = results["label"].numpy()
    y_scores = score_raw.numpy()
    seg_scores = seg_score_raw.numpy()

    # Image-level Metrics
    results_dict["I-AUROC"] = float(roc_auc_score(y_true, y_scores))
    results_dict["AP-det"] = float(average_precision_score(y_true, y_scores))
    results_dict["seg-I-AUROC"] = float(roc_auc_score(y_true, seg_scores))
    results_dict["seg-AP-det"] = float(average_precision_score(y_true, seg_scores))

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    best_threshold = float(thresholds[np.argmax(tpr - fpr)])
    y_pred = (y_scores >= best_threshold).astype(int)

    results_dict["Precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    results_dict["Recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    results_dict["F1-score"] = float(f1_score(y_true, y_pred, zero_division=0))

    # Pixel-level Metrics (Subsampled by 10 to prevent system RAM overflow)
    p_true = results["gt_mask"].flatten().numpy()
    p_true = (p_true >= 0.5).astype(int)
    p_scores = am_raw.flatten().numpy()

    step = 10 
    p_true_sub = p_true[::step]
    p_scores_sub = p_scores[::step]

    results_dict["P-AUROC"] = float(roc_auc_score(p_true_sub, p_scores_sub))
    results_dict["AP-loc"] = float(average_precision_score(p_true_sub, p_scores_sub))

    fpr_p, tpr_p, thresholds_p = roc_curve(p_true_sub, p_scores_sub)
    best_pixel_threshold = float(thresholds_p[np.argmax(tpr_p - fpr_p)])
    p_pred_sub = (p_scores_sub >= best_pixel_threshold).astype(int)

    results_dict["Pixel-F1"] = float(f1_score(p_true_sub, p_pred_sub, zero_division=0))

    # AUPRO (Requires spatial structure, utilizing TorchMetrics on GPU)
    if "AUPRO" in pixel_metrics:
        metric = pixel_metrics["AUPRO"]
        metric.to(device)
        try:
            # Enforce strictly binary masks for TorchMetrics validation
            binary_mask = (results["gt_mask"] >= 0.5).type(torch.long).to(device)
            metric.update(am_raw.to(device), binary_mask)
            results_dict["AUPRO"] = metric.compute().item()
        except Exception as e:
            print(f"\n[!] Error computing AUPRO: {e}")
            results_dict["AUPRO"] = 0.0
        finally:
            metric.to("cpu")

    # ==========================================
    # --- LOGGING & EXPORT ---
    # ==========================================
    print("\n--- EVALUATION RESULTS ---")
    for name, value in results_dict.items():
        if isinstance(value, float):
            print(f"{name}: {value:.4f} ", end="")
        else:
            print(f"{name}: {value} ", end="")
    print(f"| Opt-Th: {best_threshold:.4f}\n")

    if image_save_path:
        print("Visualizing Anomaly Maps...")
        # Normalization applied ONLY for image generation/visualizer
        eps = 1e-8
        vis_anomaly_map = (am_raw - am_raw.flatten().min()) / (am_raw.flatten().max() - am_raw.flatten().min() + eps)
        
        vis_results = results.copy()
        vis_results["anomaly_map"] = vis_anomaly_map
        
        visualizer = Visualizer(image_save_path)
        visualizer.visualize(
            vis_results,
            y_pred=y_pred,
            best_threshold=best_threshold,
            best_pixel_threshold=best_pixel_threshold
        )

    # JSON Export
    score_dict = {}
    if score_save_path:
        for img_path, score, seg_score, label in zip(
            results["image_path"],
            score_raw,
            seg_score_raw,
            results["label"],
        ):
            img_path = Path(img_path)
            anomaly_type = img_path.parent.name
            if anomaly_type not in score_dict:
                score_dict[anomaly_type] = {"good": {}, "bad": {}}

            kind = "bad" if label == 1 else "good"
            score_dict[anomaly_type][kind][img_path.stem] = {
                "score": score.item(),
                "seg_score": seg_score.item(),
            }

        score_save_path.mkdir(exist_ok=True, parents=True)
        with open(score_save_path / "scores.json", "w") as f:
            json.dump(score_dict, f)
        with open(score_save_path / "metrics.json", "w") as f:
            json.dump(results_dict, f, indent=4)

    return results_dict

def get_sensum(config):
    data = []
    for category in [sensum.Category.Capsule, sensum.Category.Softgel]:
        for fold_num in range(3):
            datamodule = Sensum(
                root=Path(config["datasets_folder"]) / "SensumSODF",
                fold=sensum.FixedFoldNumber(fold_num),
                category=category,
                image_size=sensum.get_default_resolution(category),
                train_batch_size=config["batch"],
                eval_batch_size=config["batch"],
                num_workers=config["num_workers"],
                supervision=Supervision.MIXED_SUPERVISION,
                ratio_segmented=sensum.RatioSegmented.M100,
                seed=config["seed"],
                flips=False,
            )
            datamodule.setup()
            data.append((f"{category.value}_{fold_num}", datamodule))
    return data


def get_ksdd2(config):
    datamodule = KSDD2(
        root=Path(config["datasets_folder"]) / "KolektorSDD2",
        image_size=ksdd2.get_default_resolution(),
        train_batch_size=config["batch"],
        eval_batch_size=config["batch"],
        num_workers=config["num_workers"],
        supervision=Supervision.MIXED_SUPERVISION,
        num_segmented=ksdd2.NumSegmented.N246,
        seed=config["seed"],
        flips=False,
    )
    datamodule.setup()

    return [("ksdd2", datamodule)]


def get_mvtec(config):
    data = []

    categories = [
        "screw",
        "pill",
        "capsule",
        "carpet",
        "grid",
        "tile",
        "wood",
        "zipper",
        "cable",
        "toothbrush",
        "transistor",
        "metal_nut",
        "bottle",
        "hazelnut",
        "leather",
    ]

    for category in categories:
        datamodule = MVTec(
            root=Path(config["datasets_folder"]) / "mvtec",
            category=category,
            image_size=(256, 256),
            train_batch_size=config["batch"],
            eval_batch_size=config["batch"],
            num_workers=config["num_workers"],
            seed=config["seed"],
        )
        datamodule.setup()
        data.append((category, datamodule))

    return data


def get_visa(config):
    data = []

    categories = [
        "candle",
        "capsules",
        "cashew",
        "chewinggum",
        "fryum",
        "macaroni1",
        "macaroni2",
        "pcb1",
        "pcb2",
        "pcb3",
        "pcb4",
        "pipe_fryum",
    ]

    for category in categories:
        datamodule = Visa(
            root=Path(config["datasets_folder"]) / "visa",
            category=category,
            image_size=(256, 256),
            train_batch_size=config["batch"],
            eval_batch_size=config["batch"],
            num_workers=config["num_workers"],
            seed=config["seed"],
        )
        datamodule.setup()

        data.append((category, datamodule))

    return data


def get_avg(df):
    cat_avg = df.groupby("category").mean(numeric_only=True)
    total_avg = df.mean(axis=0, numeric_only=True).to_frame().T
    total_avg.index = ["total"]
    combined = pd.concat([cat_avg, total_avg], axis=0)

    return combined


def get_std(df):
    # take std of cat mean - this covers standard splits as well as CV for sensum
    cat_std = (
        df.groupby(["run_id", "category"])
        .mean(numeric_only=True)
        .reset_index()
        .groupby("category")
        .std()
    )

    total_std = df.groupby("run_id").mean(numeric_only=True).std(numeric_only=True)
    total_std = total_std.to_frame().T
    total_std.index = ["total"]

    combined = pd.concat([cat_std, total_std], axis=0)

    return combined


def merge_csvs(dataset, run_ids, ratio, base_path):
    # read all csv and merge into one
    joined = None
    for run_id in run_ids:
        file = base_path / str(run_id) / ratio / dataset / ("last.csv")
        print(file)
        df = pd.read_csv(file)
        if joined is None:
            joined = df
        else:
            joined = pd.concat([joined, df], axis=0)

    return joined


def get_stats(dataset, run_ids, ratio, base_path):
    joined = merge_csvs(dataset, run_ids, ratio, base_path)

    comb_avg = get_avg(joined)
    comb_std = get_std(joined)

    return comb_avg, comb_std


def generate_result_json(run_ids, datasets, ratios, res_path):
    """
    Generate json with mean and std for all passed datasets and run_ids.

    Args:
        run_ids: list of run_ids
        datasets: list of datasets
        ratios: list of ratios for each datasets
        res_path: root path of results (csvs)

    """
    res_json = {"avg": {}, "std": {}}

    for dataset, ratio in zip(datasets, ratios):
        if ratio not in res_json["avg"]:
            res_json["avg"][ratio] = {}
            res_json["std"][ratio] = {}

        avg, std = get_stats(dataset, run_ids, ratio, res_path)
        avg = avg.drop(columns=["run_id"])
        std = std.drop(columns=["run_id"])

        res_json["avg"][ratio][dataset] = avg.to_dict()
        if len(run_ids) > 1:
            res_json["std"][ratio][dataset] = std.to_dict()

    Path("./res_json").mkdir(exist_ok=True, parents=True)
    with open("./res_json/ssn.json", "w") as f:
        json.dump(res_json, f)

def compute_confusion_matrix(results, image_save_path):
    """Chiamata solo a fine training, non durante la validazione."""
    print("Generating Confusion Matrix and Separating Images...")
    y_true = results["label"].numpy()
    y_scores = results["score"].numpy()

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_threshold_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else thresholds[-1]

    y_pred = (y_scores >= best_threshold).astype(int)

    classification_dir = image_save_path / "classification"
    classification_dir.mkdir(exist_ok=True, parents=True)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Good (0)", "Anomaly (1)"],
                yticklabels=["Good (0)", "Anomaly (1)"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix (Threshold: {best_threshold:.3f})")
    plt.tight_layout()
    plt.savefig(classification_dir / "confusion_matrix.png")
    plt.close()

    for cat in ["TP", "TN", "FP", "FN"]:
        (classification_dir / cat).mkdir(exist_ok=True, parents=True)

    for img_path_str, true_lbl, pred_lbl in zip(results["image_path"], y_true, y_pred):
        img_path = Path(img_path_str)
        if true_lbl == 1 and pred_lbl == 1:
            dest_folder = "TP"
        elif true_lbl == 0 and pred_lbl == 0:
            dest_folder = "TN"
        elif true_lbl == 0 and pred_lbl == 1:
            dest_folder = "FP"
        else:
            dest_folder = "FN"
        shutil.copy(img_path, classification_dir / dest_folder / img_path.name)

    print(f"Classification separated in: {classification_dir}")


def run_eval(datasets, ratios, run_id, res_path):
    """
    Evaluate the performance for given datasets for checkpoints with run_id.

    Args:
        datasets: list of dataset names
        ratios: ratio of labeled images for each dataset passed
        run_id: run_id of checkpoints to be used
        res_path: path to where  results csv will be saved
    """
    config = {
        "weights_path": Path(r"./weights"),
        "datasets_folder": Path(r"./datasets"),
        "results_save_path": res_path,
        "image_save_path": None,  # set to save images
        "score_save_path": None,  # set to save scores
        "seed": 42,
        "batch": 8,
        "num_workers": 0,
        "run_id": str(run_id),
        # "ratio": "1", # configured below in the loop if using extended version
        "adapt_cls_feat": False,  # (JIMS extension) cls features are not adapted
        # "adapt_cls_feat": True,
    }
    data_functions = {
        "sensum": get_sensum,
        "ksdd2": get_ksdd2,
        "mvtec": get_mvtec,
        "visa": get_visa,
    }

    for dataset, ratio in zip(datasets, ratios):
        config["ratio"] = ratio

        data_list = data_functions[dataset](config)

        results_writer = ResultsWriter(
            metrics=[
                "AP-det",
                "AP-loc",
                "P-AUROC",
                "I-AUROC",
                "AUPRO",
                "seg-AP-det",
                "seg-I-AUROC",
                "run_id",
            ]
        )

        for cat, datamodule in data_list:
            print("Evaluating", f"{dataset}-{cat}")
            weight_path = (
                config["weights_path"]
                / config["run_id"]
                / dataset
                / cat
                / config["ratio"]
                / "weights.pt"
            )
            model = SuperSimpleNet(image_size=datamodule.image_size, config=config)
            model.load_model(weight_path)

            image_metrics = {
                "I-AUROC": AUROC(),
                "AP-det": AveragePrecision(num_classes=1),
            }
            pixel_metrics = {
                "P-AUROC": AUROC(),
                "AP-loc": AveragePrecision(num_classes=1),
                "AUPRO": AUPRO(),  # aupro calculation can be slow, and it requires some gpu memory
            }

            results = eval(
                model=model,
                datamodule=datamodule,
                device="cuda",
                image_metrics=image_metrics,
                pixel_metrics=pixel_metrics,
                normalize=True,
                score_save_path=config["score_save_path"]
                / config["run_id"]
                / dataset
                / cat
                / config["ratio"]
                if config["score_save_path"]
                else None,
                image_save_path=config["image_save_path"]
                / config["run_id"]
                / dataset
                / cat
                / config["ratio"]
                if config["image_save_path"]
                else None,
            )
            results["run_id"] = config["run_id"]

            if config["image_save_path"]:
                save_path = config["image_save_path"] / config["run_id"] / dataset / cat / config["ratio"]
                compute_confusion_matrix(results, save_path)

            if dataset == "sensum":
                # for sensum remove fold num when saving
                res_cat = cat[:-2]
            else:
                res_cat = cat
            results_writer.add_result(
                category=res_cat,
                last=results,
            )
            results_writer.save(
                Path(config["results_save_path"])
                / config["run_id"]
                / config["ratio"]
                / dataset
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SSN on a specific checkpoint")
    
    # Positional argument for the weights file (exactly as passed from Colab)
    parser.add_argument("weights_path", type=str, help="Path to the trained weights file (.pt/.pth)")
    
    # Dataset and Paths
    parser.add_argument("--dataset", type=str, default="mvtec", help="Dataset name")
    parser.add_argument("--category", type=str, required=True, help="Defect category (e.g., custom_no_dust)")
    parser.add_argument("--datasets_folder", type=str, required=True, help="Root folder of datasets")
    parser.add_argument("--data_path", type=str, required=False, help="Alias for datasets_folder")
    parser.add_argument("--results_save_path", type=str, default="./results", help="Where to save eval outputs")
    
    # Architecture and DataLoader
    parser.add_argument("--image_size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--backbone", type=str, default="wide_resnet50_2")
    parser.add_argument("--layers", type=str, nargs="+", default=["layer2", "layer3"])
    parser.add_argument("--patch_size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=456654, help="Global random seed") 
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Variable name compatibility handling
    if args.data_path and not args.datasets_folder:
        args.datasets_folder = args.data_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Configuration dictionary construction
    config = {
        "dataset": args.dataset,
        "category": args.category,
        "datasets_folder": Path(args.datasets_folder),
        "results_save_path": Path(args.results_save_path),
        "image_size": tuple(args.image_size),
        "batch": args.batch,
        "num_workers": args.num_workers,
        "backbone": args.backbone,
        "layers": args.layers,
        "patch_size": args.patch_size,
        "seed": args.seed,
        "adapt_cls_feat": True,
    }

    print(f"\n--- Starting Evaluation for {args.dataset} - {args.category} ---")
    print(f"Loading weights from: {args.weights_path}")

    # Model initialization and weights loading
    model = SuperSimpleNet(image_size=config["image_size"], config=config)
    model.load_model(Path(args.weights_path))

    # DataModule initialization
    if args.dataset == "mvtec":
        datamodule = MVTec(
            root=config["datasets_folder"],
            category=config["category"],
            image_size=config["image_size"],
            train_batch_size=config["batch"],
            eval_batch_size=config["batch"],
            num_workers=config["num_workers"],
            seed=config["seed"],
            supervision=Supervision.MIXED_SUPERVISION 
        )
    elif args.dataset == "visa":
         datamodule = Visa(
            root=config["datasets_folder"] / "visa",
            category=config["category"],
            image_size=config["image_size"],
            train_batch_size=config["batch"],
            eval_batch_size=config["batch"],
            num_workers=config["num_workers"],
            seed=config["seed"]
        )
    else:
         raise NotImplementedError(f"Dataset {args.dataset} evaluation not implemented via CLI.")
         
    datamodule.setup()

    # Metrics Setup
    image_metrics = {
        "I-AUROC": AUROC(),
        "AP-det": AveragePrecision(num_classes=1),
    }
    pixel_metrics = {
        "P-AUROC": AUROC(),
        "AP-loc": AveragePrecision(num_classes=1),
        "AUPRO": AUPRO(),
    }

    # Definition of save paths for visualizations and scores
    image_save_path = config["results_save_path"] / "visual_eval" / config["dataset"] / config["category"]
    score_save_path = config["results_save_path"] / "scores_eval" / config["dataset"] / config["category"]

    # Core Evaluation Start
    results = eval(
        model=model,
        datamodule=datamodule,
        device=device,
        image_metrics=image_metrics,
        pixel_metrics=pixel_metrics,
        normalize=True,
        image_save_path=image_save_path,
        score_save_path=score_save_path
    )
    
    print(f"Evaluation completed. Results saved in {args.results_save_path}")

if __name__ == "__main__":
    main()