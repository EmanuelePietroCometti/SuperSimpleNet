"""
verify_onnx_parity.py — PyTorch<->ONNX parity test for the SuperSimpleNet
contract-3.0 export.

Thin CLI wrapper around export_common.verify: feeds the *same* [0,1] dummy
tensor (the contract-3.0 input domain — the graph normalizes internally)
through the PyTorch wrapper and the ONNXRuntime session and asserts parity on
both outputs, at batch 1 and 4 to prove the dynamic batch axis is real.

Run standalone (random weights, no checkpoint needed):
    python verify_onnx_parity.py
Or against a real export:
    python verify_onnx_parity.py --weights weights.pt --onnx model.onnx
"""

import argparse
from pathlib import Path

import torch

from export_common import build_metadata, export, verify
from export_onnx import build_export_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default=None)
    p.add_argument("--onnx", type=str, default=None)
    p.add_argument("--image_size", type=int, nargs=2, default=[512, 512])
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = {
        "image_size": tuple(args.image_size),
        "backbone": "wide_resnet50_2",
        "layers": ["layer2", "layer3"],
        "patch_size": 3,
        "adapt_cls_feat": True,
    }
    weights = Path(args.weights) if args.weights else None
    wrapper = build_export_model(config, weights, device)

    if args.onnx:
        onnx_path = Path(args.onnx)
    else:
        onnx_path = Path("supersimplenet_selftest.onnx")
        metadata = build_metadata(
            architecture="supersimplenet",
            image_size=config["image_size"],
            blur_kernel_size=25,
            blur_sigma=4.0,
            weights_path=weights,
        )
        export(wrapper, config["image_size"], onnx_path, device, metadata)

    verify(wrapper, onnx_path, config["image_size"], device)


if __name__ == "__main__":
    main()
