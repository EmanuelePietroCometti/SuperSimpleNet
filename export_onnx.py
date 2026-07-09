"""
export_onnx.py — Export SuperSimpleNet to a *pure* ONNX graph.

Root cause of the "meaningless output" bug
-------------------------------------------
The previous export baked the full pre/post-processing pipeline into the graph
(uint8->float, resize, ImageNet normalize, sigmoid, Gaussian blur). The GPU
inference runtime (``inference_simulation``) *also* performs normalization and
blur on the host. Whenever both sides ran, the image was normalized twice and
the map blurred/rescaled twice, so the anomaly map and score were numerically
meaningless even though the trained PyTorch model was correct.

Fix / design contract (identical for all 4 architectures)
---------------------------------------------------------
The exported graph is now a *pure forward pass* — feature extraction + score
computation only. All pre/post-processing lives on the host.

    Input   "input_tensor" : float32 [B, 3, H, W]
            Already resized to the training size and ImageNet-normalized by the
            host (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]). This is
            exactly what SuperSimpleNet's FeatureExtractor expects — it does no
            normalization internally.
    Output  "anomaly_map"   : float32 [B, 1, H, W]  raw seg map, bilinearly
            upscaled to input HxW. NO Gaussian blur, NO min-max, NO threshold.
    Output  "anomaly_score" : float32 [B]           raw fc_score logit. NO
            sigmoid.

Only the batch dimension is dynamic (``dynamic_axes``); H/W are fixed to the
training ``image_size``. The host applies blur + folder-global min-max exactly
as ``inference_simulation/src/postprocessing.py`` already does.

Notes
-----
* Training is untouched: we build a thin *export-only* wrapper around a model in
  ``eval()`` mode, so the training ``forward()`` and its GPU speed are unchanged
  (constraint #5).
* The classic (TorchScript) exporter is used with an explicit ``dynamic_axes``
  dict. Since resize + antialias moved to the host, the graph no longer needs
  the dynamo-only ``aten::_upsample_bilinear2d_aa`` op, so the simpler, more
  predictable exporter works and honours constraint #2 (``dynamic_axes``).

The graph is always fp32. Reduced precision (fp16/int8) is intentionally NOT
produced here: it is the inference program's job to cast the fp32 graph to the
target precision for its hardware (e.g. TensorRT fp16/int8), so a single
canonical fp32 artifact stays the source of truth.

Usage
-----
    python export_onnx.py weights.pt --image_size 512 512 -o model.onnx
    python export_onnx.py --self_test          # random weights, no checkpoint
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from model.supersimplenet import SuperSimpleNet

# I/O contract shared by every exported architecture.
INPUT_NAMES = ["input_tensor"]
OUTPUT_NAMES = ["anomaly_map", "anomaly_score"]
OPSET = 17


# ---------------------------------------------------------------------------
# Export-only wrapper (does NOT touch the training forward path)
# ---------------------------------------------------------------------------
class PureInferenceWrapper(nn.Module):
    """
    Wraps a SuperSimpleNet in eval mode so the exported graph is a pure forward
    pass: normalized tensor in -> (raw anomaly_map, raw anomaly_score) out.

    The wrapped model's ``AnomalyMapGenerator.blur`` is replaced with Identity
    (blur is host-side post-processing) and the score is returned as the raw
    fc_score logit (no sigmoid). Everything else — feature extraction, feature
    adaptor, discriminator, bilinear upscale of the map — stays in-graph because
    it is the *core* computation, deterministic, and identical to training eval.
    """

    def __init__(self, model: SuperSimpleNet):
        super().__init__()
        self.model = model

    def forward(self, input_tensor: torch.Tensor):
        # eval-mode SuperSimpleNet.forward returns (anomaly_map, anomaly_score).
        # anomaly_map is already upscaled to image_size; blur was neutralized to
        # Identity in build_export_model, so no post-processing is baked in.
        anomaly_map, anomaly_score = self.model(input_tensor)
        return anomaly_map, anomaly_score


def build_export_model(config: dict, weights_path: Path | None, device: str) -> PureInferenceWrapper:
    """Build a SuperSimpleNet, load weights (unless None => random self-test),
    put it in a pure-forward state, and wrap it for export."""
    model = SuperSimpleNet(image_size=config["image_size"], config=config)
    if weights_path is not None:
        model.load_model(weights_path)
    model.to(device).eval()

    # Strip the Gaussian blur from the graph: it is host-side post-processing
    # (see inference_simulation/src/postprocessing.py::apply_training_blur).
    # The bilinear upscale in AnomalyMapGenerator stays — it only reshapes the
    # map to input resolution and is deterministic.
    model.anomaly_map_generator.blur = nn.Identity()

    wrapper = PureInferenceWrapper(model).to(device).eval()
    for p in wrapper.parameters():
        p.requires_grad_(False)
    return wrapper


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
# Embedded in the .onnx file so the inference runtime (inference_simulation) can
# auto-configure blur/scoring instead of relying on CLI flags. Unlike SK-RD4AD,
# SuperSimpleNet's anomaly_score IS the number to threshold on directly (a
# dedicated classification head, see Discriminator.fc_score in
# model/supersimplenet.py) - it is not derived from the map, so
# score_source="graph". The blur (kernel=25, sigma=4) matches
# AnomalyMapGenerator(sigma=4) and is display/consistency only, not required to
# reproduce the score.
EXPORT_METADATA = {
    "anomaly_export_contract": "1.0",
    "architecture": "supersimplenet",
    "score_source": "graph",
    "blur_kernel_size": "25",
    "blur_sigma": "4.0",
    "verified": "true",
}


def _write_metadata(onnx_path: Path, weights_source: str) -> None:
    """weights_source: "checkpoint:<filename>" for real exports, or
    "random_self_test" for --self_test exports. The inference runtime refuses to
    score with a random_self_test model (untrained weights produce garbage that
    masquerades as a pipeline bug)."""
    import onnx
    m = onnx.load(str(onnx_path))
    for k, v in {**EXPORT_METADATA, "weights_source": weights_source}.items():
        entry = m.metadata_props.add()
        entry.key, entry.value = k, v
    onnx.save(m, str(onnx_path))


def export_fp32(wrapper: nn.Module, image_size: tuple[int, int], onnx_path: Path, device: str,
                weights_source: str = "unknown"):
    h, w = image_size
    # Trace with batch=2 so no dimension is accidentally specialized to 1.
    dummy = torch.randn(2, 3, h, w, dtype=torch.float32, device=device)

    dynamic_axes = {
        "input_tensor": {0: "batch"},
        "anomaly_map": {0: "batch"},
        "anomaly_score": {0: "batch"},
    }

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(onnx_path),
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES,
            dynamic_axes=dynamic_axes,
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,          # classic exporter honours the dynamic_axes dict
            external_data=False,   # single self-contained .onnx file
        )

    import onnx
    onnx.checker.check_model(str(onnx_path))
    _write_metadata(onnx_path, weights_source)
    return onnx_path


# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export SuperSimpleNet to a pure ONNX graph")
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

    # Resolve the output path. --output may be a full .onnx path OR a directory
    # (detected if it exists as a dir or lacks the .onnx suffix); a directory
    # gets an auto-derived filename, avoiding the PermissionError/IsADirectoryError
    # that torch.onnx.export raises when handed a directory to open for writing.
    default_stem = weights.stem if weights is not None else "supersimplenet_selftest"
    if args.output:
        onnx_path = Path(args.output)
        if onnx_path.is_dir() or onnx_path.suffix.lower() != ".onnx":
            onnx_path = onnx_path / f"{default_stem}.onnx"
    elif weights is not None:
        onnx_path = weights.with_suffix(".onnx")
    else:
        onnx_path = Path("supersimplenet_selftest.onnx")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    weights_source = f"checkpoint:{weights.name}" if weights is not None else "random_self_test"
    print(f"\n--- Exporting SuperSimpleNet (pure graph, weights: {weights_source}) -> {onnx_path} ---")
    export_fp32(wrapper, config["image_size"], onnx_path, device, weights_source)
    print(f"[OK] fp32 export: {onnx_path}")
    print(f"     input 'input_tensor' : float32 [B,3,{config['image_size'][0]},{config['image_size'][1]}] "
          f"(host-normalized, ImageNet)")
    print( "     output 'anomaly_map'  : float32 [B,1,H,W] (raw, no blur)")
    print( "     output 'anomaly_score': float32 [B] (raw logit, no sigmoid)")

    if args.self_test:
        from verify_onnx_parity import verify
        verify(wrapper, onnx_path, config["image_size"], device)


if __name__ == "__main__":
    main()
