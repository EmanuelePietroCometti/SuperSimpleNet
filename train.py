import os
import sys

from datamodules.base import Supervision

LOG_WANDB = False

import copy
import json
from pathlib import Path

if LOG_WANDB:
    import wandb

from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule, seed_everything

from torchmetrics.classification import BinaryAveragePrecision, BinaryAUROC
from sklearn.metrics import precision_score, recall_score, f1_score
from torchmetrics import Metric
from anomalib.utils.metrics import AUPRO

from datamodules import ksdd2, sensum
from datamodules.ksdd2 import KSDD2, NumSegmented
from datamodules.sensum import Sensum, RatioSegmented
from datamodules.mvtec import MVTec
from datamodules.visa import Visa

from model.supersimplenet import SuperSimpleNet

from common.visualizer import Visualizer
from common.results_writer import ResultsWriter
from common.loss import focal_loss
from torchvision.transforms import v2
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix
import argparse
from common.calibration import piecewise_min_max_calibration


def train(
    model: SuperSimpleNet,
    epochs: int,
    datamodule: LightningDataModule,
    device: str,
    config: dict,
    image_metrics: dict[str, Metric],
    pixel_metrics: dict[str, Metric],
    th: float = 0.5,
    clip_grad: bool = True,
    eval_step_size: int = 4,
):
    model.to(device)
    optimizer, scheduler = model.get_optimizers()

    WEIGHT_AUROC = 0.70
    WEIGHT_AUPRO = 0.30
    best_combined_score = -1.0
    best_model_weights = None

    gpu_transforms = v2.Compose([
        v2.Resize(size=config["image_size"], antialias=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]).to(device)

    model.train()
    train_loader = datamodule.train_dataloader()
    test_loader = datamodule.test_dataloader()

    print("\n" + "="*40)
    print("DATASET DEBUG INFO")
    try:
        n_normal = len(datamodule.train_data._normal_samples)
        n_anomalous = len(datamodule.train_data._anomalous_samples)
        print(f"Normal images loaded: {n_normal}")
        print(f"Anomalous images loaded: {n_anomalous}")
        print(f"Total batches per epoch: {len(train_loader)} (Batch size: {config['batch']})")
        print(f"Total images processed per epoch: {len(train_loader) * config['batch']}")
    except Exception as e:
        print(f"Unable to read dataset information directly: {e}")
    print("="*40 + "\n")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        with tqdm(
            total=len(train_loader),
            desc=str(epoch) + "/" + str(epochs),
            miniters=int(1),
            unit="batch",
        ) as prog_bar:
            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()

                image_batch = batch["image"].to(device, non_blocking=True)

                # best downsampling proposed by DestSeg
                mask = batch["mask"].to(device, dtype=torch.float32, non_blocking=True)

                image_batch = gpu_transforms(image_batch)

                mask = F.interpolate(
                    mask,
                    size=(model.fh, model.fw),
                    mode="bilinear",
                    align_corners=True,
                )
                mask = torch.where(
                    mask < 0.5, torch.zeros_like(mask), torch.ones_like(mask)
                )

                label = batch["label"].to(device, dtype=torch.float32, non_blocking=True)
                is_segmented = batch["is_segmented"].to(device, dtype=torch.float32, non_blocking=True)

                anomaly_map, score, mask, label = model.forward(
                    image_batch, mask, label
                )

                seg_focal = focal_loss(torch.sigmoid(anomaly_map), mask, reduction=None)

                # use this shape to apply weights from distance transform if enabled
                seg_l1 = torch.zeros_like(anomaly_map)

                # adjusted truncated l1: mask + flipped sign (ano->pos, good->neg)
                normal_scores = anomaly_map[mask == 0]
                seg_l1[mask == 0] = torch.clip(normal_scores + th, min=0)

                anomalous_scores = anomaly_map[mask > 0]
                seg_l1[mask > 0] = torch.clip(-anomalous_scores + th, min=0)

                if "loss_mask" in batch:
                    loss_mask = batch["loss_mask"].to(device, dtype=torch.float32, non_blocking=True)

                    # resize loss_mask to fit the loss
                    loss_mask = F.interpolate(
                        loss_mask,
                        size=seg_focal.shape[-2:],
                        mode="bilinear",
                        align_corners=True,
                    )

                    # due to feat. duplication stack mask and multiply to get weighted loss
                    loss_mask = torch.cat((loss_mask, loss_mask))
                    seg_focal *= loss_mask
                    seg_l1 *= loss_mask

                # due to feat. duplication
                is_segmented = torch.cat((is_segmented, is_segmented)).type(torch.bool)

                bad_loss = seg_l1[is_segmented][mask[is_segmented] > 0]
                good_loss = seg_l1[is_segmented][mask[is_segmented] == 0]
                focal_val = seg_focal[is_segmented]

                if len(good_loss):
                    good_loss = good_loss.mean()
                else:
                    good_loss = 0
                if len(bad_loss):
                    bad_loss = bad_loss.mean()
                else:
                    bad_loss = 0
                if len(focal_val):
                    focal_val = focal_val.mean()
                else:
                    focal_val = 0

                # seg loss is combination of trunc l1 and focal (separately avg each l1 part due to unbalanced pixels)
                seg_loss = good_loss + bad_loss + focal_val

                loss = seg_loss + focal_loss(torch.sigmoid(score), label)

                loss.backward()

                if clip_grad:
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1).item()
                else:
                    norm = 0.0

                optimizer.step()
                
                loss_val = loss.detach().item()
                total_loss += loss_val

                output = {
                    "batch_loss": round(loss_val, 5),
                    "avg_loss": round(total_loss / (i + 1), 5),
                    "norm": norm,
                }

                prog_bar.set_postfix(**output)
                prog_bar.update(1)

            if (epoch + 1) % eval_step_size == 0:
                results = test(
                    model=model,
                    test_loader=test_loader,
                    device=device,
                    config=config,
                    image_metrics=image_metrics,
                    pixel_metrics=pixel_metrics,
                    normalize=True
                )

                current_auroc = results.get("I-AUROC", 0.0)
                current_aupro = results.get("AUPRO", 0.0)

                combined_score = (current_auroc * WEIGHT_AUROC) + (current_aupro * WEIGHT_AUPRO)

                if combined_score > best_combined_score:
                    print(f"New best model found! Score: {best_combined_score:.4f} -> {combined_score:.4f}")
                    print(f"I-AUROC: {current_auroc:.4f} | AUPRO: {current_aupro:.4f}")
                    best_combined_score = combined_score
                    best_model_weights = copy.deepcopy(model.state_dict())
                if LOG_WANDB:
                    wandb.log({**results, **output})
            else:
                if LOG_WANDB:
                    wandb.log(output)
        scheduler.step()
    
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        print(f"Training finished! Best model's weights loaded! Best combined score: {best_combined_score:.4f}")

    return results


@torch.no_grad()
def test(
    model: SuperSimpleNet,
    test_loader,
    device: str,
    config: dict,
    image_metrics: dict[str, Metric],
    pixel_metrics: dict[str, Metric],
    normalize: bool = True,
    image_save_path: Path = None,
    score_save_path: Path = None,
):
    model.to(device)
    model.eval()

    gpu_transforms = v2.Compose([
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]).to(device)

    seg_image_metrics = {}

    for m_name, metric in image_metrics.items():
        metric.to(device)
        metric.reset()
        seg_image_metrics[f"seg-{m_name}"] = copy.deepcopy(metric).to(device)

    for metric in pixel_metrics.values():
        metric.to(device)
        metric.reset()

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
        image_batch = batch["image"].to(device, non_blocking=True)
        mask_batch = batch["mask"].to(device, dtype=torch.float32, non_blocking=True)

        image_batch = gpu_transforms(image_batch)

        anomaly_map, anomaly_score = model.forward(image_batch)

        anomaly_map_sig = torch.sigmoid(anomaly_map).detach()
        anomaly_score_sig = torch.sigmoid(anomaly_score).detach()
        seg_score = anomaly_map.detach().reshape(anomaly_map.shape[0], -1).max(dim=1).values
        seg_score_sig = torch.sigmoid(seg_score)

        label_long = batch["label"].to(device, dtype=torch.long, non_blocking=True)
        mask_long = (mask_batch >= 0.5).to(dtype=torch.long)

        for metric in image_metrics.values():
            metric.update(anomaly_score_sig, label_long)

        for metric in seg_image_metrics.values():
            metric.update(seg_score_sig, label_long)

        for name, metric in pixel_metrics.items():
            try:
                am_clean = torch.nan_to_num(anomaly_map_sig, nan=0.0)
                metric.update(am_clean, mask_long)
            except RuntimeError:
                pass
        
        results["anomaly_map"].append(anomaly_map_sig.cpu())
        results["score"].append(anomaly_score_sig.cpu())
        results["seg_score"].append(seg_score.cpu())
        
        results["gt_mask"].append(mask_batch.detach().cpu())
        results["label"].append(batch["label"].detach().cpu())

        results["image_path"].extend(batch["image_path"])
        results["mask_path"].extend(batch["mask_path"])

    # Base Metrics Calculation
    results_dict = {}
    for name, metric in image_metrics.items():
        results_dict[name] = metric.compute().item()

    for name, metric in seg_image_metrics.items():
        results_dict[name] = metric.compute().item()

    for name, metric in pixel_metrics.items():
        try:
            results_dict[name] = metric.compute().item()
        except RuntimeError:
            results_dict[name] = 0

    # Global Concatenation
    results["anomaly_map"] = torch.cat(results["anomaly_map"])
    results["score"] = torch.cat(results["score"])
    results["seg_score"] = torch.cat(results["seg_score"])
    results["gt_mask"] = torch.cat(results["gt_mask"])
    results["label"] = torch.cat(results["label"])

    # Calculate RAW Thresholds BEFORE Calibration
    y_true = results["label"].numpy()
    y_scores_raw = results["score"].numpy()

    # Calculate optimal threshold for images
    fpr, tpr, thresholds = roc_curve(y_true, y_scores_raw)
    raw_img_th = thresholds[np.argmax(tpr - fpr)]

    # Calculate optimal threshold for pixels (subsampled to prevent RAM overflow)
    p_true = results["gt_mask"].flatten().numpy()
    p_true = (p_true >= 0.5).astype(int)
    p_scores_raw = results["anomaly_map"].flatten().numpy()
    fpr_p, tpr_p, thresholds_p = roc_curve(p_true[::10], p_scores_raw[::10])
    raw_pix_th = thresholds_p[np.argmax(tpr_p - fpr_p)]

    # Apply Piecewise Min-Max Calibration
    if normalize:
        print("Applying Piecewise Min-Max Calibration (Target TH: 0.5)...")
        results["score"] = piecewise_min_max_calibration(results["score"], raw_img_th)
        results["seg_score"] = piecewise_min_max_calibration(results["seg_score"], raw_img_th)
        results["anomaly_map"] = piecewise_min_max_calibration(results["anomaly_map"], raw_pix_th)

        # After calibration, the optimal thresholds are mathematically forced to 0.5
        best_threshold = 0.5
        best_pixel_threshold = 0.5
    else:
        best_threshold = raw_img_th
        best_pixel_threshold = raw_pix_th

    # Calculate Final Metrics using Calibrated Scores
    y_scores_calibrated = results["score"].numpy()
    y_pred = (y_scores_calibrated >= best_threshold).astype(int)

    results_dict["Precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    results_dict["Recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    results_dict["F1-score"] = float(f1_score(y_true, y_pred, zero_division=0))
    
    p_true_sub = p_true[::10] 
    p_scores_cal_sub = results["anomaly_map"].flatten().numpy()[::10]
    p_pred_sub = (p_scores_cal_sub >= best_pixel_threshold).astype(int)
    results_dict["Pixel-F1"] = float(f1_score(p_true_sub, p_pred_sub, zero_division=0))

    for name, value in results_dict.items():
        print(f"{name}: {value:.4f} ", end="")
    print(f"| Opt-Th: {best_threshold:.4f}")

    # Visualization and Storage
    if image_save_path:
        print("Saving Confusion Matrix plot and Separating Images...")
        
        classification_dir = image_save_path.parent / "classification_results"
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

        print("Calculating Optimal Pixel Threshold...")
        p_true = results["gt_mask"].flatten().numpy()
        p_true = (p_true >= 0.5).astype(int)
        p_scores = results["anomaly_map"].flatten().numpy()
        
        # Subsampling [::10] prevents RAM overflow on massive pixel arrays
        fpr_p, tpr_p, thresholds_p = roc_curve(p_true[::10], p_scores[::10])
        youden_j_p = tpr_p - fpr_p
        best_pixel_threshold_idx = np.argmax(youden_j_p)
        best_pixel_threshold = thresholds_p[best_pixel_threshold_idx]
        print(f"Optimal Pixel threshold (Youden): {best_pixel_threshold:.4f}")

        print("Visualizing...")
        visualizer = Visualizer(classification_dir)
        visualizer.visualize(
            results, 
            y_pred=y_pred, 
            best_threshold=best_threshold, 
            best_pixel_threshold=best_pixel_threshold
        )

    if score_save_path:
        score_dict = {}
        for img_path, score, seg_score, label in zip(
            results["image_path"],
            results["score"],
            results["seg_score"],
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


def train_and_eval(model, datamodule, config, device):
    if LOG_WANDB:
        os.environ["WANDB__SERVICE_WAIT"] = "300"
        wandb.init(project=config["wandb_project"], config=config, name=config["name"])

    image_metrics = {
        "I-AUROC": BinaryAUROC(thresholds=100),
        "AP-det": BinaryAveragePrecision(thresholds=100)
    }
    pixel_metrics = {
        "P-AUROC": BinaryAUROC(thresholds=100),
        "AUPRO": AUPRO(),
        "AP-loc": BinaryAveragePrecision(thresholds=100),
    }

    train(
        model=model,
        epochs=config["epochs"],
        datamodule=datamodule,
        device=device,
        config=config,
        image_metrics=image_metrics,
        pixel_metrics=pixel_metrics,
        clip_grad=config["clip_grad"],
        eval_step_size=config["eval_step_size"],
        th=config["th"]
    )
    if LOG_WANDB:
        wandb.finish()

    try:
        model.save_model(
            Path(config["results_save_path"])
            / config["setup_name"]
            / "checkpoints"
            / config["dataset"]
            / config["category"]
            / str(config["ratio"]),
        )
    except Exception as e:
        print("Error saving checkpoint" + str(e))

    results = test(
        model=model,
        test_loader=datamodule.test_dataloader(),
        device=device,
        config=config,
        image_metrics=image_metrics,
        pixel_metrics=pixel_metrics,
        normalize=True,
        image_save_path=Path(config["results_save_path"])
        / config["setup_name"]
        / "visual"
        / config["dataset"]
        / config["category"]
        / str(config["ratio"]),
        score_save_path=Path(config["results_save_path"])
        / config["setup_name"]
        / "scores"
        / config["dataset"]
        / config["category"]
        / str(config["ratio"]),
    )

    return results


def main_mvtec(device, config, supervision=Supervision.UNSUPERVISED):
    config = copy.deepcopy(config)
    config["dataset"] = "mvtec"
    config["ratio"] = 1

    # Dynamically extract the category from the parsed CLI arguments
    category = config["category"]
    print(f"Training on {category}")

    config["name"] = f"{category}_{config['setup_name']}"

    # Enforce deterministic behavior for reproducibility
    seed_everything(config["seed"], workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    results_writer = ResultsWriter(
        metrics=[
            "AP-det",
            "AP-loc",
            "P-AUROC",
            "I-AUROC",
            "AUPRO",
            "seg-AP-det",
            "seg-I-AUROC",
        ]
    )

    model = SuperSimpleNet(image_size=config["image_size"], config=config)

    datamodule = MVTec(
        root=Path(config["datasets_folder"]), 
        category=category,
        image_size=config["image_size"],
        train_batch_size=config["batch"],
        eval_batch_size=config["batch"],
        num_workers=config["num_workers"],
        seed=config["seed"],
        supervision=supervision
    )
    datamodule.setup()

    results = train_and_eval(
        model=model, datamodule=datamodule, config=config, device=device
    )

    results_writer.add_result(
        category=category,
        last=results,
    )
    
    results_writer.save(
        Path(config["results_save_path"])
        / config["setup_name"]
        / config["dataset"]
        / str(config["ratio"])
    )


def main_visa(device, config):
    config = copy.deepcopy(config)
    config["dataset"] = "visa"
    config["ratio"] = 1

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

    results_writer = ResultsWriter(
        metrics=[
            "AP-det",
            "AP-loc",
            "P-AUROC",
            "I-AUROC",
            "AUPRO",
            "seg-AP-det",
            "seg-I-AUROC",
        ]
    )

    for category in categories:
        print(f"Training on {category}")

        config["category"] = category
        config["name"] = f"{category}_{config['setup_name']}"

        seed_everything(config["seed"], workers=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        model = SuperSimpleNet(image_size=config["image_size"], config=config)

        datamodule = Visa(
            root=Path(config["datasets_folder"]) / "visa",
            category=category,
            image_size=config["image_size"],
            train_batch_size=config["batch"],
            eval_batch_size=config["batch"],
            num_workers=config["num_workers"],
            seed=config["seed"],
        )
        datamodule.setup()

        results = train_and_eval(
            model=model, datamodule=datamodule, config=config, device=device
        )

        results_writer.add_result(
            category=category,
            last=results,
        )
        results_writer.save(
            Path(config["results_save_path"])
            / config["setup_name"]
            / config["dataset"]
            / str(config["ratio"])
        )


def main_ksdd2(device, config, supervision):
    config = copy.deepcopy(config)
    config["dataset"] = "ksdd2"
    config["category"] = "ksdd2"
    config["name"] = f"ksdd2_{config['setup_name']}"

    results_writer = ResultsWriter(
        metrics=[
            "AP-det",
            "AP-loc",
            "P-AUROC",
            "I-AUROC",
            "AUPRO",
            "seg-AP-det",
            "seg-I-AUROC",
            "ratio",
        ]
    )

    seed_everything(config["seed"], workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = SuperSimpleNet(image_size=ksdd2.get_default_resolution(), config=config)

    datamodule = KSDD2(
        root=Path(config["datasets_folder"]) / "KolektorSDD2",
        supervision=supervision,
        image_size=ksdd2.get_default_resolution(),
        train_batch_size=config["batch"],
        eval_batch_size=config["batch"],
        num_workers=config["num_workers"],
        num_segmented=NumSegmented(config["ratio"]),
        seed=config["seed"],
        flips=config["flips"],
        dt=config["dt"],
        dilate=config["dilate"],
    )
    datamodule.setup()

    results = train_and_eval(
        model=model, datamodule=datamodule, config=config, device=device
    )

    results["ratio"] = config["ratio"]
    results_writer.add_result(
        category="ksdd2",
        last=results,
    )
    results_writer.save(
        Path(config["results_save_path"])
        / config["setup_name"]
        / config["dataset"]
        / str(config["ratio"])
    )


def main_sensum(device, config, supervision):
    config = copy.deepcopy(config)
    config["dataset"] = "sensum"

    results_writer = ResultsWriter(
        metrics=[
            "AP-det",
            "AP-loc",
            "P-AUROC",
            "I-AUROC",
            "AUPRO",
            "seg-AP-det",
            "seg-I-AUROC",
            "fold",
            "ratio",
        ]
    )

    for category in [sensum.Category.Capsule, sensum.Category.Softgel]:
        print(f"Training on {category.value}")

        seed_everything(config["seed"], workers=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        for fold_num in range(3):
            config["category"] = f"{category.value}_{fold_num}"
            config["name"] = f"{category.value}_{config['setup_name']}_{fold_num}"
            config["fold"] = fold_num

            model = SuperSimpleNet(
                image_size=sensum.get_default_resolution(category), config=config
            )

            datamodule = Sensum(
                root=Path(config["datasets_folder"]) / "SensumSODF",
                supervision=supervision,
                fold=sensum.FixedFoldNumber(fold_num),
                category=category,
                image_size=sensum.get_default_resolution(category),
                train_batch_size=config["batch"],
                eval_batch_size=config["batch"],
                num_workers=config["num_workers"],
                ratio_segmented=sensum.RatioSegmented(config["ratio"]),
                seed=config["seed"],
                flips=config["flips"],
                dt=config["dt"],
                dilate=config["dilate"],
            )
            datamodule.setup()

            results = train_and_eval(
                model=model, datamodule=datamodule, config=config, device=device
            )

            # also log fold as a separate column
            results["fold"] = fold_num
            results["ratio"] = config["ratio"]
            results_writer.add_result(
                category=f"{category.value}",
                last=results,
            )
            results_writer.save(
                Path(config["results_save_path"])
                / config["setup_name"]
                / config["dataset"]
                / str(config["ratio"])
            )


def run_unsup(data_name, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = {
        "dataset": args.dataset,
        "category": args.category,
        "name": f"{args.category}_ssn",
        "wandb_project": args.wandb_project,
        "datasets_folder": Path(args.datasets_folder),
        "num_workers": args.num_workers,
        "setup_name": args.setup_name,
        "backbone": args.backbone,
        "layers": args.layers,
        "patch_size": args.patch_size,
        "noise": True,
        "perlin": True,
        "no_anomaly": "empty",
        "bad": True,
        "overlap": True,
        "adapt_cls_feat": True,
        "noise_std": args.noise_std,
        "perlin_thr": args.perlin_thr,
        "image_size": tuple(args.image_size),
        "seed": args.seed,
        "batch": args.batch,
        "epochs": args.epochs,
        "flips": True,
        "seg_lr": args.seg_lr,
        "dec_lr": args.dec_lr,
        "adapt_lr": args.adapt_lr,
        "gamma": args.gamma,
        "stop_grad": True,
        "clip_grad": False,
        "eval_step_size": args.eval_step_size,
        "results_save_path": Path(args.results_save_path),
        "th": args.th
    }
    
    if data_name == "visa":
        config["perlin_thr"] = 0.6
        main_visa(device=device, config=config)
    if data_name == "mvtec":
        config["perlin_thr"] = 0.2
        main_mvtec(device=device, config=config, supervision=Supervision.UNSUPERVISED)


def run_sup(data_name, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = {
        "dataset": args.dataset,
        "category": args.category,
        "name": f"{args.category}_ssn",
        "wandb_project": args.wandb_project,
        "datasets_folder": Path(args.datasets_folder),
        "num_workers": args.num_workers,
        "setup_name": args.setup_name,
        "dt": args.dt,
        "dilate": args.dilate,
        "backbone": args.backbone,
        "layers": args.layers,
        "patch_size": args.patch_size,
        "noise": True,
        "perlin": True,
        "no_anomaly": "empty",
        "bad": True,
        "overlap": False,
        "adapt_cls_feat": True,
        "noise_std": args.noise_std,
        "perlin_thr": args.perlin_thr,
        "image_size": tuple(args.image_size),
        "seed": args.seed,
        "batch": args.batch,
        "epochs": args.epochs,
        "flips": True,
        "seg_lr": args.seg_lr,
        "dec_lr": args.dec_lr,
        "adapt_lr": args.adapt_lr,
        "gamma": args.gamma,
        "stop_grad": False,
        "clip_grad": True,
        "eval_step_size": args.eval_step_size,
        "results_save_path": Path(args.results_save_path),
        "th": args.th
    }
    
    if data_name == "sensum":
        config["ratio"] = RatioSegmented.M100.value
        if float(config["ratio"]) == 0:
            config["perlin_thr"] = 0.2
        main_sensum(
            device=device, config=config, supervision=Supervision.MIXED_SUPERVISION
        )
        
    if data_name == "ksdd2":
        config["ratio"] = NumSegmented.N246.value
        if float(config["ratio"]) == 0:
            config["perlin_thr"] = 0.2
        main_ksdd2(
            device=device, config=config, supervision=Supervision.MIXED_SUPERVISION
        )

    if data_name == "mvtec":
        config["perlin_thr"] = 0.2
        main_mvtec(
            device=device,
            config=config,
            supervision=Supervision.MIXED_SUPERVISION
        )


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Configuration Parser for SSN")
    
    # Argomenti Base (Mancanti prima)
    parser.add_argument("--dataset", type=str, default="mvtec", help="Dataset type")
    parser.add_argument("--category", type=str, required=True, help="Category of the dataset")
    parser.add_argument("--mode", type=str, default="sup", help="Supervision paradigm")
    
    # Compatibilità percorsi (Risolve il crash con Colab)
    parser.add_argument("--datasets_folder", type=str, required=True, help="Path to the datasets directory")
    parser.add_argument("--data_path", type=str, required=False, help="Alias for datasets_folder")
    
    parser.add_argument("--wandb_project", type=str, default="ssn", help="Name of the Weights & Biases project")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of background workers for the DataLoader")
    parser.add_argument("--setup_name", type=str, required=True, help="Name of the current experimental setup")
    parser.add_argument("--results_save_path", type=str, default="./results", help="Directory path where results will be saved")
    
    parser.add_argument("--dt", type=int, nargs=2, default=[3, 2], help="Distance transform parameters")
    parser.add_argument("--dilate", type=int, default=7, help="Dilation factor for the mask")
    parser.add_argument("--image_size", type=int, nargs=2, default=[512, 512], help="Image resolution")
    parser.add_argument("--noise_std", type=float, default=0.015, help="Standard deviation for the applied noise")
    parser.add_argument("--perlin_thr", type=float, default=0.2, help="Threshold for Perlin noise generation")
    
    parser.add_argument("--backbone", type=str, default="wide_resnet50_2", help="Backbone architecture")
    parser.add_argument("--layers", type=str, nargs="+", default=["layer2", "layer3"], help="List of backbone layers")
    parser.add_argument("--patch_size", type=int, default=3, help="Patch size applied in the processing step")
    
    parser.add_argument("--seed", type=int, default=456654, help="Global random seed")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=150, help="Total number of training epochs")
    
    parser.add_argument("--seg_lr", type=float, default=0.0002, help="Learning rate for the segmentation module")
    parser.add_argument("--dec_lr", type=float, default=0.0002, help="Learning rate for the decoder module")
    parser.add_argument("--adapt_lr", type=float, default=0.0001, help="Learning rate for the adaptation module")
    
    parser.add_argument("--gamma", type=float, default=0.4, help="Gamma decay parameter")
    parser.add_argument("--eval_step_size", type=int, default=5, help="Number of steps/epochs between evaluations")
    parser.add_argument("--th", type=float, default=0.5, help="Threshold applied for binary classification/segmentation")
    
    return parser

def main():
    parser = get_args_parser()
    args = parser.parse_args()

    if hasattr(args, 'data_path') and args.data_path and not args.datasets_folder:
        args.datasets_folder = args.data_path

    print(f"Starting training pipeline in {args.mode.upper()} mode on dataset: {args.dataset}")

    if args.mode == "unsup":
        run_unsup(args.dataset, args)
    elif args.mode == "sup":
        run_sup(args.dataset, args)

if __name__ == "__main__":
    main()