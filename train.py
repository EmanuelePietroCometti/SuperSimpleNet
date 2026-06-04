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
from sklearn.metrics import precision_recall_curve, confusion_matrix


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

    gpu_transforms = v2.Compose([
        v2.Resize(size=config["image_size"], antialias=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]).to(device)

    model.train()
    train_loader = datamodule.train_dataloader()
    test_loader = datamodule.test_dataloader()
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
                if LOG_WANDB:
                    wandb.log({**results, **output})
            else:
                if LOG_WANDB:
                    wandb.log(output)
        scheduler.step()

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
        mask_long = mask_batch.to(dtype=torch.long)

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

    for name, value in results_dict.items():
        print(f"{name}: {value} ", end="")
    print()

    if image_save_path or score_save_path:
        results["anomaly_map"] = torch.cat(results["anomaly_map"])
        results["score"] = torch.cat(results["score"])
        results["seg_score"] = torch.cat(results["seg_score"])
        results["gt_mask"] = torch.cat(results["gt_mask"])
        results["label"] = torch.cat(results["label"])

        if normalize:
            am_min, am_max = results["anomaly_map"].min(), results["anomaly_map"].max()
            results["anomaly_map"] = (results["anomaly_map"] - am_min) / (am_max - am_min + 1e-8)

            s_min, s_max = results["score"].min(), results["score"].max()
            results["score"] = (results["score"] - s_min) / (s_max - s_min + 1e-8)

            ss_min, ss_max = results["seg_score"].min(), results["seg_score"].max()
            results["seg_score"] = (results["seg_score"] - ss_min) / (ss_max - ss_min + 1e-8)

        if image_save_path:
            print("Visualizing")
            visualizer = Visualizer(image_save_path)
            visualizer.visualize(results)

            print("Generating Confusion Matrix and Separating Images...")
            y_true = results["label"].numpy()
            y_scores = results["score"].numpy()

            precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            best_threshold_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else thresholds[-1]

            y_pred = (y_scores >= best_threshold).astype(int)

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
                elif true_lbl == 1 and pred_lbl == 0:
                    dest_folder = "FN"

                shutil.copy(img_path, classification_dir / dest_folder / img_path.name)
                
            print(f"Classification separated in: {classification_dir}")

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

    return results_dict


def train_and_eval(model, datamodule, config, device):
    if LOG_WANDB:
        os.environ["WANDB__SERVICE_WAIT"] = "300"
        wandb.init(project=config["wandb_project"], config=config, name=config["name"])

    image_metrics = {
        "I-AUROC": BinaryAUROC(thresholds=100),
        "AP-det": BinaryAveragePrecision(thresholds=100),
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


def main_mvtec(device, config):
    config = copy.deepcopy(config)
    config["dataset"] = "mvtec"
    config["ratio"] = 1

    """"screw",
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
        "carpet","""

    categories = [
        "reda"
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

        # deterministic
        seed_everything(config["seed"], workers=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        model = SuperSimpleNet(image_size=config["image_size"], config=config)

        datamodule = MVTec(
            root=Path(config["datasets_folder"]) / "mvtec",
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


def run_unsup(data_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = {
        "wandb_project": "ssn",
        "datasets_folder": Path("./datasets"),
        "num_workers": 8,
        "setup_name": "superSimpleNet",
        "backbone": "wide_resnet50_2",
        "layers": ["layer2", "layer3"],
        "patch_size": 3,
        "noise": True,
        "perlin": True,
        "no_anomaly": "empty",
        "bad": True,
        "overlap": True,  # makes no difference, just faster if false to avoid computation
        "adapt_cls_feat": True,  # (JIMS extension) cls features are not adapted
        "noise_std": 0.035,
        "perlin_thr": 0.6,
        "image_size": (256, 256),
        "seed": 42,
        "batch": 4,
        "epochs": 300,
        "flips": True,  # makes no difference, just faster if false to avoid computation
        "seg_lr": 0.0002,
        "dec_lr": 0.0002,
        "adapt_lr": 0.0001,
        "gamma": 0.4,
        "stop_grad": True,
        "clip_grad": False,
        "eval_step_size": 20,
        "results_save_path": Path("./results"),
    }
    if data_name == "visa":
        config["perlin_thr"] = 0.6
        main_visa(device=device, config=config)
    if data_name == "mvtec":
        config["perlin_thr"] = 0.6
        main_mvtec(device=device, config=config)


def run_sup(data_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = {
        "wandb_project": "ssn",
        "datasets_folder": Path("./datasets"),
        "num_workers": 1,
        "setup_name": "superSimpleNet",
        "dt": (3, 2),   # distance transform
        "dilate": 2,    # dilate mask
        "backbone": "wide_resnet50_2",
        "layers": ["layer2", "layer3"],
        "patch_size": 3,
        "noise": True,
        "perlin": True,
        "no_anomaly": "empty",
        "bad": True,
        "overlap": False,
        "adapt_cls_feat": True,  # (JIMS extension) cls features are not adapted
        "noise_std": 0.035,
        "perlin_thr": 0.6,
        "seed": 456654,
        "batch": 4,
        "epochs": 300,
        "flips": True,
        "seg_lr": 0.0002,
        "dec_lr": 0.0002,
        "adapt_lr": 0.0001,
        "gamma": 0.4,
        "stop_grad": False,
        "clip_grad": True,
        "eval_step_size": 20,
        "results_save_path": Path("./results"),
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


def main():
    run_unsup(sys.argv[1])
    run_sup(sys.argv[1])


if __name__ == "__main__":
    main()
