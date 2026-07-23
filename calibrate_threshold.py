"""
calibrate_threshold.py — Calibrazione della soglia per un export contratto 3.0.

Esegue l'ONNX ESPORTATO (non il modello PyTorch) sul test set etichettato e
scrive nei metadati del file .onnx i valori che il runtime C++ legge all'avvio:

    calibrated_threshold        soglia immagine sullo score finale del grafo
                                (sigmoid in-graph -> stessa scala di
                                eval.py raw_img_th, PRE piecewise calibration)
    calibrated_threshold_pixel  soglia pixel sulla mappa finale (contorni NG)
    calibration_map_min/max     scala FISSA per la visualizzazione heatmap
                                (niente min-max per-immagine: un patch pulito
                                deve restare blu)
    render_lut_rgb_b64          colormap jet condivisa col runtime C++ (LUT
                                256x3 uint8 RGB, base64)
    verified = true             solo se la calibrazione e' andata a buon fine

Calibrare sull'ONNX invece che sul PyTorch chiude il cerchio: "verified"
certifica l'artefatto che va in produzione, inclusi blur/sigmoid in-graph.
Il preprocessing usato qui e' il datamodule canonico del training
(v2.Resize antialias=True, RGB [0,1]): e' la catena di riferimento con cui
la pipeline C++ deve dimostrare parita'.

La soglia e' scelta con il criterio di Youden (argmax TPR-FPR) sulla ROC,
identico a eval.py:111. Gli score del grafo sono gia' passati per la sigmoid,
quindi la soglia vive sulla stessa scala: nessuna ritrasformazione.

Usage
-----
    python calibrate_threshold.py model.onnx --dataset mvtec --category grid \\
        --datasets_folder ./datasets
    python calibrate_threshold.py model.onnx ... --dry_run   # solo report
"""

import argparse
import base64
import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve
from tqdm import tqdm

from export_common import EXPORT_CONTRACT, INPUT_NAME, OUTPUT_NAMES, read_metadata, update_metadata

# Stesso sottocampionamento di eval.py:118: i punteggi pixel di un test set
# intero non stanno in RAM, e per una soglia ROC il decimato e' statisticamente
# equivalente.
PIXEL_SUBSAMPLE_STEP = 10


def validate_model(onnx_path: Path) -> dict[str, str]:
    """Rifiuta subito i modelli con cui una calibrazione non ha senso."""
    meta = read_metadata(onnx_path)
    contract = meta.get("export_contract", meta.get("anomaly_export_contract", "<assente>"))
    if contract != EXPORT_CONTRACT:
        raise SystemExit(
            f"ERROR: contratto di export '{contract}' non supportato (atteso {EXPORT_CONTRACT}).\n"
            f"Ri-esporta il modello con l'export_onnx.py corrente."
        )
    if meta.get("weights_source") == "random_self_test":
        raise SystemExit(
            "ERROR: il modello e' stato esportato con --self_test (pesi random).\n"
            "Calibrare una soglia su output casuali non ha significato: esporta il checkpoint reale."
        )
    if meta.get("architecture") != "supersimplenet":
        raise SystemExit(
            f"ERROR: architettura '{meta.get('architecture')}' — questo script usa i datamodule di "
            f"SuperSimpleNet; calibra il modello nel repo della sua architettura."
        )
    if meta.get("score_semantics") != "final" or meta.get("normalization") != "in_graph":
        raise SystemExit(
            "ERROR: metadati incoerenti con il contratto 3.0 "
            f"(score_semantics={meta.get('score_semantics')}, normalization={meta.get('normalization')})."
        )
    return meta


def build_datamodule(args, image_size: tuple[int, int]):
    # Stessa costruzione di eval.py::main — la catena di preprocessing del
    # datamodule (Resize antialias=True, RGB [0,1]) E' il riferimento canonico.
    # Import lazy: i datamodule tirano dentro l'ambiente di training completo
    # (anomalib, albumentations); i guard sui metadati devono funzionare anche
    # dove quell'ambiente non c'e'.
    from datamodules.base import Supervision
    from datamodules.mvtec import MVTec
    from datamodules.visa import Visa

    if args.dataset == "mvtec":
        datamodule = MVTec(
            root=Path(args.datasets_folder),
            category=args.category,
            image_size=image_size,
            train_batch_size=args.batch,
            eval_batch_size=args.batch,
            num_workers=args.num_workers,
            seed=args.seed,
            supervision=Supervision.MIXED_SUPERVISION,
        )
    elif args.dataset == "visa":
        datamodule = Visa(
            root=Path(args.datasets_folder) / "visa",
            category=args.category,
            image_size=image_size,
            train_batch_size=args.batch,
            eval_batch_size=args.batch,
            num_workers=args.num_workers,
            seed=args.seed,
        )
    else:
        raise SystemExit(f"ERROR: dataset '{args.dataset}' non supportato (mvtec | visa).")
    datamodule.setup()
    return datamodule


def run_onnx_on_testset(onnx_path: Path, datamodule) -> dict:
    import onnxruntime as ort

    available = ort.get_available_providers()
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    print(f"ONNX Runtime providers: {sess.get_providers()}")

    scores, labels = [], []
    pix_scores, pix_true = [], []
    global_min, global_max = np.inf, -np.inf

    for batch in tqdm(datamodule.test_dataloader(), desc="Calibrating", leave=True):
        x = batch["image"].numpy().astype(np.float32)  # RGB [0,1]: input contratto 3.0
        anomaly_map, anomaly_score = sess.run(OUTPUT_NAMES, {INPUT_NAME: x})

        scores.append(anomaly_score)
        labels.append(batch["label"].numpy())
        global_min = min(global_min, float(anomaly_map.min()))
        global_max = max(global_max, float(anomaly_map.max()))

        mask = (batch["mask"].numpy().reshape(-1) >= 0.5).astype(np.int64)
        pix_true.append(mask[::PIXEL_SUBSAMPLE_STEP])
        pix_scores.append(anomaly_map.reshape(-1)[::PIXEL_SUBSAMPLE_STEP])

    return {
        "scores": np.concatenate(scores),
        "labels": np.concatenate(labels).astype(np.int64),
        "pix_scores": np.concatenate(pix_scores),
        "pix_true": np.concatenate(pix_true),
        "global_min": global_min,
        "global_max": global_max,
    }


def youden_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    return float(thresholds[np.argmax(tpr - fpr)])


def jet_lut_rgb_b64() -> str:
    """LUT 256x3 uint8 (RGB) da cv2.COLORMAP_JET, serializzata base64.

    È la colormap condivisa del contratto di rendering (render_reference.py
    [D-Q3]): incorporandola nei metadati, render_reference.py e il runtime C++
    consumano gli STESSI 768 byte -> nessuna divergenza jet-matplotlib vs
    jet-OpenCV. cv2 restituisce BGR: invertiamo l'ultimo asse per avere RGB.
    """
    ramp = np.arange(256, dtype=np.uint8).reshape(256, 1)
    bgr = cv2.applyColorMap(ramp, cv2.COLORMAP_JET).reshape(256, 3)
    rgb = np.ascontiguousarray(bgr[:, ::-1])  # BGR -> RGB
    return base64.b64encode(rgb.tobytes()).decode("ascii")


def main():
    p = argparse.ArgumentParser(description="Calibra la soglia di un export ONNX (contratto 3.0)")
    p.add_argument("onnx_path", type=str, help="Modello .onnx esportato da export_onnx.py")
    p.add_argument("--dataset", type=str, default="mvtec", choices=["mvtec", "visa"])
    p.add_argument("--category", type=str, required=True)
    p.add_argument("--datasets_folder", type=str, required=True)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_auroc", type=float, default=0.5,
                   help="I-AUROC minimo per accettare la calibrazione: sotto questo valore i "
                        "metadati NON vengono scritti (un modello peggiore del caso non e' "
                        "calibrabile, e' rotto).")
    p.add_argument("--dry_run", action="store_true",
                   help="Calcola e stampa il report senza modificare il file .onnx")
    args = p.parse_args()

    onnx_path = Path(args.onnx_path)
    if not onnx_path.is_file():
        raise SystemExit(f"ERROR: {onnx_path} non esiste.")

    meta = validate_model(onnx_path)
    image_size = (int(meta["input_height"]), int(meta["input_width"]))
    print(f"\n--- Calibrating {onnx_path.name} "
          f"({meta['architecture']}, {image_size[0]}x{image_size[1]}, "
          f"weights: {meta.get('weights_source')}) ---")

    datamodule = build_datamodule(args, image_size)
    r = run_onnx_on_testset(onnx_path, datamodule)

    n_good = int((r["labels"] == 0).sum())
    n_bad = int((r["labels"] == 1).sum())
    if n_good == 0 or n_bad == 0:
        raise SystemExit(
            f"ERROR: il test set ha {n_good} good e {n_bad} bad — servono entrambe le classi "
            f"per una curva ROC. Controlla dataset/category."
        )

    # --- Soglia immagine (verdetto OK/NG) ---
    i_auroc = float(roc_auc_score(r["labels"], r["scores"]))
    ap_det = float(average_precision_score(r["labels"], r["scores"]))
    threshold = youden_threshold(r["labels"], r["scores"])
    y_pred = (r["scores"] >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(r["labels"], y_pred, labels=[0, 1]).ravel()

    # --- Soglia pixel (contorni difetto sulla mappa finale) ---
    if r["pix_true"].any():
        p_auroc = float(roc_auc_score(r["pix_true"], r["pix_scores"]))
        pixel_threshold = youden_threshold(r["pix_true"], r["pix_scores"])
    else:
        # Nessun pixel anomalo nelle GT mask: la soglia pixel non e' definibile.
        p_auroc, pixel_threshold = None, None

    print(f"\nimages: {n_good} good / {n_bad} bad")
    print(f"I-AUROC = {i_auroc:.4f}   AP-det = {ap_det:.4f}")
    print(f"calibrated_threshold       = {threshold:.6f}  (scala: sigmoid score del grafo)")
    print(f"confusion @ threshold      : TP={tp} FN={fn} FP={fp} TN={tn}")
    if pixel_threshold is not None:
        print(f"P-AUROC = {p_auroc:.4f}")
        print(f"calibrated_threshold_pixel = {pixel_threshold:.6f}")
    else:
        print("calibrated_threshold_pixel : non definibile (nessun pixel anomalo nelle mask)")
    print(f"calibration_map_min/max    = {r['global_min']:.6f} / {r['global_max']:.6f}")

    if i_auroc < args.min_auroc:
        raise SystemExit(
            f"\nERROR: I-AUROC {i_auroc:.4f} < --min_auroc {args.min_auroc}: metadati NON scritti.\n"
            f"Un AUROC cosi' basso indica un modello/preprocessing rotto, non una soglia da calibrare."
        )
    if i_auroc < 0.8:
        print(f"\n[WARN] I-AUROC {i_auroc:.4f} < 0.8: la soglia esiste ma il modello discrimina male.")

    if args.dry_run:
        print("\n[dry run] nessuna modifica al file .onnx.")
        return

    updates = {
        "calibrated_threshold": f"{threshold:.6f}",
        # Chiavi del contratto di rendering (render_reference.py [D-Q1]):
        # range GLOBALE FISSO della MAPPA finale per la normalizzazione display.
        # Rinominano calibration_global_min/max (che avevano semantica ambigua
        # score-vs-mappa tra i repo, CONTRACT.md §4-Q1).
        "calibration_map_min": f"{r['global_min']:.6f}",
        "calibration_map_max": f"{r['global_max']:.6f}",
        # LUT jet condivisa (256x3 uint8 RGB, base64) consumata identica da
        # render_reference.py e dal runtime C++ [D-Q3].
        "render_lut_rgb_b64": jet_lut_rgb_b64(),
        "calibration_dataset": f"{args.dataset}/{args.category}",
        "calibration_num_good": str(n_good),
        "calibration_num_bad": str(n_bad),
        "calibration_i_auroc": f"{i_auroc:.4f}",
        "calibration_date": datetime.date.today().isoformat(),
        "verified": "true",
    }
    if pixel_threshold is not None:
        # Rinominata da calibrated_pixel_threshold [D-Q6].
        updates["calibrated_threshold_pixel"] = f"{pixel_threshold:.6f}"
        updates["calibration_p_auroc"] = f"{p_auroc:.4f}"

    update_metadata(onnx_path, updates)
    print(f"\n[OK] metadati aggiornati in {onnx_path} (verified=true).")


if __name__ == "__main__":
    main()
