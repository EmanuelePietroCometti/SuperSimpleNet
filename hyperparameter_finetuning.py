import optuna
import optuna_dashboard
import torch
from pathlib import Path
from datamodules.base import Supervision
from train import train_and_eval
from model.supersimplenet import SuperSimpleNet
from datamodules.mvtec import MVTec
from pytorch_lightning import seed_everything

def objective(trial):
    """
    Optuna objective function for SuperSimpleNet hyperparameter optimization.
    Returns the Image-AUROC (I-AUROC) metric to be maximized.
    """
    # Define the hyperparameter search space
    noise_std = trial.suggest_float("noise_std", 0.05, 0.35)
    perlin_thr = trial.suggest_float("perlin_thr", 0.1, 0.8)
    
    # Learning rates are best searched on a logarithmic scale
    seg_lr = trial.suggest_float("seg_lr", 1e-6, 2e-3, log=True)
    dec_lr = trial.suggest_float("dec_lr", 1e-6, 2e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.2, 0.8)
    
    # Patch size for pooling should ideally be an odd number (1, 3, 5, 7)
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
    
    # Setup the configuration dictionary dynamically for this trial
    config = {
        "wandb_project": "ssn_optuna",
        "datasets_folder": Path("./datasets"),
        "num_workers": 1,
        "setup_name": f"optuna_trial_{trial.number}",
        "dataset": "mvtec",
        "category": "reda",
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
        "seed": 42,
        "batch": 4,
        "epochs": 20,
        "flips": True,
        "seg_lr": seg_lr,
        "dec_lr": dec_lr,
        "adapt_lr": 0.0001,
        "gamma": gamma,
        "stop_grad": False,
        "clip_grad": True,
        "eval_step_size": 5,
        "results_save_path": Path("./results_optuna"),
        "th": 0.5,
        "name": f"optuna_run_{trial.number}"
    }

    # Make the run deterministic for reproducibility
    seed_everything(config["seed"], workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Instantiate Model and DataModule directly
    model = SuperSimpleNet(image_size=config["image_size"], config=config)
    datamodule = MVTec(
        root=Path(config["datasets_folder"]) / "mvtec",
        category=config["category"],
        image_size=config["image_size"],
        train_batch_size=config["batch"],
        eval_batch_size=config["batch"],
        num_workers=config["num_workers"],
        seed=config["seed"],
        supervision=Supervision.MIXED_SUPERVISION
    )
    datamodule.setup()

    # Execute Training and Evaluation pipeline
    results = train_and_eval(model=model, datamodule=datamodule, config=config, device=device)

    # Return the target metric for Optuna to evaluate the trial's success
    # If the model collapses and returns NaN or no value, return 0
    return results.get("I-AUROC", 0.0)


if __name__ == "__main__":
    db_url = "sqlite:///supersimplenet_tuning.db"
    # Define a study that aims to maximize the returned objective (I-AUROC)
    study = optuna.create_study(direction="maximize", storage=db_url, study_name="ssn_optimization", load_if_exists=True)
    
    # Execute the hyperparameter search
    study.optimize(objective, n_trials=30)
    
    # Print out the best hyperparameters found
    print("\n--- OPTIMIZATION COMPLETE ---")
    best_trial = study.best_trial
    print(f"Best I-AUROC: {best_trial.value}")
    print("Best Parameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")