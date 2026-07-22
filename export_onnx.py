"""
export_onnx.py — Export SuperSimpleNet nel contratto ONNX 3.0 condiviso.

Contratto 3.0 (vedi export_common.py — identico per le 4 architetture)
----------------------------------------------------------------------
    Input   "image"         : float32 [B, 3, H, W]
            RGB in [0,1], gia' alla risoluzione del modello.
            NON normalizzato: la normalizzazione ImageNet e' un nodo del
            grafo (InGraphNormalize), quindi non puo' essere dimenticata
            o applicata due volte dall'host — il bug che aveva fatto
            crollare l'AUROC da ~0.91 a ~0.51 in eval.py.
    Output  "anomaly_map"   : float32 [B, 1, H, W]
            mappa FINALE: upsample bilineare + blur gaussiano canonico
            (kernel=25, sigma=4, identico a training/eval) applicati
            in-graph. Nessun min-max, nessuna soglia.
    Output  "anomaly_score" : float32 [B]
            numero FINALE: sigmoid(fc_score logit), lo stesso valore su
            cui eval.py:74 calcola soglie e metriche. Direttamente
            confrontabile con metadata["calibrated_threshold"].

Solo l'asse batch e' dinamico; H/W sono fissati alla risoluzione di training.
Il grafo e' sempre fp32: fp16/int8 sono responsabilita' del runtime e
richiedono ri-calibrazione della soglia.

Nota sulla soglia — piecewise_min_max_calibration (eval.py:131) e' una
trasformazione monotona applicata DOPO la sigmoid: la soglia da scrivere nei
metadati ("calibrated_threshold") deve essere quella sulla scala sigmoid
PRE-calibrazione (raw_img_th in eval.py:111), che e' esattamente la scala
dello score emesso da questo grafo.

Il training non viene toccato: si esporta un wrapper attorno a un modello in
eval(), quindi forward() di training e velocita' GPU restano invariati.

Usage
-----
    python export_onnx.py weights.pt --image_size 512 512 -o model.onnx
    python export_onnx.py --self_test          # pesi random, nessun checkpoint
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from export_common import (
    ExportWrapper,
    build_metadata,
    export,
    resolve_output_path,
    verify,
)
from model.supersimplenet import SuperSimpleNet


class ExportGaussianBlur(nn.Module):
    """Clone ONNX-esportabile del GaussianBlur di torchvision.

    torchvision.transforms.GaussianBlur costruisce il kernel a runtime da
    tensori (linspace/exp), quindi l'exporter classico vede una convoluzione
    con kernel di forma ignota e fallisce ("Unsupported: ONNX export of
    convolution for kernel of unknown shape"). Qui il kernel — stessa
    matematica di _get_gaussian_kernel1d/2d — e' precalcolato come buffer,
    cosi' diventa un initializer costante del grafo. Pad reflect + conv2d
    replicano esattamente il percorso di torchvision; l'equivalenza numerica
    e' verificata con un assert in _swap_blur_for_export, non assunta.
    """

    def __init__(self, kernel_size: int, sigma: float, channels: int = 1):
        super().__init__()
        ksize_half = (kernel_size - 1) * 0.5
        x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
        pdf = torch.exp(-0.5 * (x / sigma).pow(2))
        kernel1d = pdf / pdf.sum()
        kernel2d = torch.mm(kernel1d[:, None], kernel1d[None, :])
        self.register_buffer(
            "weight", kernel2d.expand(channels, 1, kernel_size, kernel_size).clone()
        )
        self.channels = channels
        self.padding = [kernel_size // 2] * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self.padding, mode="reflect")
        return F.conv2d(x, self.weight, groups=self.channels)


def _swap_blur_for_export(model: SuperSimpleNet, device: str) -> None:
    """Sostituisce il blur di torchvision con il clone esportabile, dopo aver
    dimostrato che i due producono lo stesso output sul device di export."""
    tv_blur = model.anomaly_map_generator.blur
    kernel_size = tv_blur.kernel_size[0]
    sigma = float(tv_blur.sigma[0])
    export_blur = ExportGaussianBlur(kernel_size, sigma, channels=1).to(device)

    with torch.no_grad():
        probe = torch.rand(2, 1, 64, 64, device=device)
        ref, rep = tv_blur(probe), export_blur(probe)
    if not torch.allclose(ref, rep, atol=1e-6):
        raise RuntimeError(
            f"ExportGaussianBlur diverge dal GaussianBlur torchvision "
            f"(|d|max={(ref - rep).abs().max().item():.2e}): export bloccato."
        )
    model.anomaly_map_generator.blur = export_blur


class SuperSimpleNetExport(ExportWrapper):
    """Blur e sigmoid restano NEL grafo (contratto 3.0).

    Il blur e' AnomalyMapGenerator(sigma=4) -> kernel_size = 2*ceil(3*4)+1 = 25,
    identico a quello usato in training/eval: non va sostituito con Identity
    (in-graph tramite ExportGaussianBlur, verificato equivalente).
    La sigmoid replica eval.py:74 (`torch.sigmoid(anomaly_score)`), cosi' lo
    score esportato e' lo stesso numero su cui eval.py calcola la soglia.
    """

    def __init__(self, model: SuperSimpleNet):
        super().__init__()
        self.model = model

    def core(self, x: torch.Tensor):
        anomaly_map, logit = self.model(x)
        return anomaly_map, torch.sigmoid(logit)


def build_export_model(config: dict, weights_path: Path | None, device: str) -> SuperSimpleNetExport:
    """Build a SuperSimpleNet, load weights (unless None => random self-test),
    put it in eval mode, and wrap it for export. AnomalyMapGenerator (upsample
    + blur) resta nel grafo: nel contratto 3.0 la mappa esportata e' finale."""
    model = SuperSimpleNet(image_size=config["image_size"], config=config)
    if weights_path is not None:
        model.load_model(weights_path)
    model.to(device).eval()
    _swap_blur_for_export(model, device)

    wrapper = SuperSimpleNetExport(model).to(device).eval()
    for p in wrapper.parameters():
        p.requires_grad_(False)
    return wrapper


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export SuperSimpleNet (contratto ONNX 3.0)")
    p.add_argument("weights_path", type=str, nargs="?", default=None,
                   help="Path to trained weights.pt (omit with --self_test)")
    p.add_argument("-o", "--output", type=str, default=None, help="Output .onnx path")
    p.add_argument("--image_size", type=int, nargs=2, default=[512, 512])
    p.add_argument("--backbone", type=str, default="wide_resnet50_2")
    p.add_argument("--layers", type=str, nargs="+", default=["layer2", "layer3"])
    p.add_argument("--patch_size", type=int, default=3)
    p.add_argument("--self_test", action="store_true",
                   help="Export with random weights and run the parity check (no checkpoint needed)")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = {
        "image_size": tuple(args.image_size),
        "backbone": args.backbone,
        "layers": args.layers,
        "patch_size": args.patch_size,
        "adapt_cls_feat": True,
    }

    if args.self_test:
        if args.weights_path is not None:
            # Refuse rather than silently exporting random weights: an untrained
            # model produces garbage output that masquerades as a pipeline bug
            # (this exact mistake shipped a random-weights SK-RD4AD model that
            # scored ~1 on every image).
            raise SystemExit(
                "ERROR: --self_test and a weights_path are mutually exclusive.\n"
                "  - To export your TRAINED model:  python export_onnx.py <weights.pt> [...]\n"
                "  - To test the export pipeline with random weights:  python export_onnx.py --self_test"
            )
        weights = None
    elif args.weights_path is None:
        raise SystemExit("Provide a weights_path, or pass --self_test to export with random weights.")
    else:
        weights = Path(args.weights_path)

    wrapper = build_export_model(config, weights, device)

    onnx_path = resolve_output_path(args.output, weights, "supersimplenet_selftest")

    # kernel_size = 2*ceil(3*sigma)+1 = 25 con sigma=4: e' il blur reale di
    # AnomalyMapGenerator, riportato nei metadati a solo scopo di
    # documentazione (map_semantics=final_blurred: il runtime non ri-sfoca).
    metadata = build_metadata(
        architecture="supersimplenet",
        image_size=config["image_size"],
        blur_kernel_size=25,
        blur_sigma=4.0,
        dynamic_crop=False,
        weights_path=weights,
        resize_mode="bilinear_antialias",  # datamodules/base/datamodule.py:65
        extra={
            "backbone": args.backbone,
            "layers": ",".join(args.layers),
            "patch_size": args.patch_size,
        },
    )

    print(f"\n--- Exporting SuperSimpleNet (contract 3.0, weights: {metadata['weights_source']}) -> {onnx_path} ---")
    export(wrapper, config["image_size"], onnx_path, device, metadata)
    print(f"[OK] fp32 export: {onnx_path}")
    print(f"     input 'image'         : float32 [B,3,{config['image_size'][0]},{config['image_size'][1]}] "
          f"RGB [0,1], normalizzazione in-graph")
    print( "     output 'anomaly_map'  : float32 [B,1,H,W] (finale: upsample + blur in-graph)")
    print( "     output 'anomaly_score': float32 [B] (finale: sigmoid in-graph)")
    print( "     NOTA: verified=false — eseguire la calibrazione della soglia per "
           "scrivere calibrated_threshold/calibration_global_min/max e verified=true.")

    if args.self_test:
        verify(wrapper, onnx_path, config["image_size"], device)


if __name__ == "__main__":
    main()
