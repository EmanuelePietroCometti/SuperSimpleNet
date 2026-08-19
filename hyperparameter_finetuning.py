import argparse
import optuna
import torch
from pathlib import Path

from datamodules.base import Supervision
from train import train_and_eval
from model.supersimplenet import SuperSimpleNet
from datamodules.mvtec import MVTec
from pytorch_lightning import seed_everything

def parse_args() -> argparse.Namespace:
    """
    Parses command line arguments for the Optuna hyperparameter tuning script.
    """
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Tuning for SSN")
    
    # Core Data Arguments
    parser.add_argument("--dataset", type=str, default="mvtec", help="Dataset type")
    parser.add_argument("--category", type=str, required=True, help="Category of the dataset")
    parser.add_argument("--datasets_folder", type=str, required=True, help="Path to the datasets directory")
    parser.add_argument("--data_path", type=str, required=False, help="Alias for datasets_folder")
    
    # Run Arguments
    parser.add_argument("--setup_name", type=str, default="optuna_search", help="Prefix for the trial runs")
    parser.add_argument("--results_save_path", type=str, default="./results_optuna", help="Directory for saving results")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs per trial")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=1, help="DataLoader workers")
    parser.add_argument("--n_trials", type=int, default=30, help="Number of Optuna trials to run")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")

    return parser.parse_args()

def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    """
    Optuna objective function for SuperSimpleNet hyperparameter optimization.
    Returns a combined score of Image F1 and Pixel F1 to be maximized.
    """
    noise_std = trial.suggest_float("noise_std", 0.05, 0.35)
    perlin_thr = trial.suggest_float("perlin_thr", 0.1, 0.8)
    
    seg_lr = trial.suggest_float("seg_lr", 1e-6, 2e-3, log=True)
    dec_lr = trial.suggest_float("dec_lr", 1e-6, 2e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.2, 0.8)
    
    patch_size = trial.suggest_int("patch_size", 1, 7, step=2)
    layer_choice = trial.suggest_categorical("layers_choice", ["layer1_2", "layer2_3", "layer1_2_3"])
    
    if layer_choice == "layer1_2":
        layers = ["layer1", "layer2"]
    elif layer_choice == "layer2_3":
        layers = ["layer2", "layer3"]
    else:
        layers = ["layer1", "layer2", "layer3"]
        
    backbone = trial.suggest_categorical("backbone", ["wide_resnet50_2", "resnet50", "resnet34", "resnet18"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = {
        "wandb_project": "ssn_optuna",
        "datasets_folder": Path(args.datasets_folder),
        "num_workers": args.num_workers,
        "setup_name": f"{args.setup_name}_trial_{trial.number}",
        "dataset": args.dataset,
        "category": args.category,
        "ratio": 1,
        "dt": (3, 2),
        "dilate": 7,
        "backbone": backbone,
        "layers": layers,
        "patch_size": patch_size,
        "noise": True,
        "perlin": True,
        "no_anomaly": "empty",
        "bad": True,
        "overlap": False,
        "adapt_cls_feat": True,
        "noise_std": noise_std,
        "perlin_thr": perlin_thr,
        "image_size": (256, 256),
        "seed": args.seed,
        "batch": args.batch,
        "epochs": args.epochs,
        "flips": True,
        "seg_lr": seg_lr,
        "dec_lr": dec_lr,
        "adapt_lr": 0.0001,
        "gamma": gamma,
        "stop_grad": False,
        "clip_grad": True,
        "eval_step_size": 5,
        "results_save_path": Path(args.results_save_path),
        "th": 0.5,
        "name": f"optuna_run_{trial.number}"
    }

    seed_everything(config["seed"], workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = SuperSimpleNet(image_size=config["image_size"], config=config)
    
    datamodule = MVTec(
        root=Path(config["datasets_folder"]),
        category=config["category"],
        image_size=config["image_size"],
        train_batch_size=config["batch"],
        eval_batch_size=config["batch"],
        num_workers=config["num_workers"],
        seed=config["seed"],
        supervision=Supervision.MIXED_SUPERVISION
    )
    datamodule.setup()

    results = train_and_eval(model=model, datamodule=datamodule, config=config, device=device)

    # Retrieve the computed F1 scores
    image_f1 = results.get("F1-score", 0.0)
    pixel_f1 = results.get("Pixel-F1", 0.0)
    
    # Calculate the combined objective score
    combined_f1 = (image_f1 + pixel_f1) / 2.0

    print(f"\n[Trial {trial.number}] Target Metrics -> Image-F1: {image_f1:.4f} | Pixel-F1: {pixel_f1:.4f} | Combined: {combined_f1:.4f}\n")

    return combined_f1

def main():
    args = parse_args()
    
    if args.data_path and not args.datasets_folder:
        args.datasets_folder = args.data_path
        
    print(f"Starting Hyperparameter Optimization for {args.dataset} - {args.category}")
    print(f"Results will be saved to: {args.results_save_path}")

    db_url = "sqlite:///supersimplenet_tuning.db"
    
    study = optuna.create_study(
        direction="maximize", 
        storage=db_url, 
        study_name="ssn_optimization", 
        load_if_exists=True
    )
    
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)
    
    print("\n--- OPTIMIZATION COMPLETE ---")
    best_trial = study.best_trial
    print(f"Best Combined F1-score: {best_trial.value:.4f}")
    print("Best Parameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()