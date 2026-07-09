"""
verify_onnx_parity.py — Strict PyTorch<->ONNX mathematical parity test for the
SuperSimpleNet pure export.

Feeds the *same* normalized dummy tensor through the PyTorch wrapper and the
ONNXRuntime session and asserts exact parity with
``np.testing.assert_allclose(atol=1e-5)`` on both outputs. Also checks that a
*different* batch size works, proving the dynamic batch axis is real.

Run standalone (random weights, no checkpoint needed):
    python verify_onnx_parity.py
Or against a real export:
    python verify_onnx_parity.py --weights weights.pt --onnx model.onnx
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort

from export_onnx import build_export_model, export_fp32, INPUT_NAMES, OUTPUT_NAMES


def _run_torch(wrapper, x_np):
    with torch.no_grad():
        m, s = wrapper(torch.from_numpy(x_np).to(next(wrapper.parameters()).device))
    return m.cpu().numpy(), s.cpu().numpy()


def _run_onnx(sess, x_np):
    m, s = sess.run(OUTPUT_NAMES, {INPUT_NAMES[0]: x_np})
    return m, s


def verify(wrapper, onnx_path, image_size, device, atol=1e-5):
    h, w = image_size
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)

    print("\n--- PyTorch vs ONNX parity (atol=1e-5) ---")
    for batch in (1, 4):  # batch=1 and batch>1 prove the dynamic axis works
        x = rng.standard_normal((batch, 3, h, w)).astype(np.float32)
        tm, ts = _run_torch(wrapper, x)
        om, os_ = _run_onnx(sess, x)

        np.testing.assert_allclose(om, tm, atol=atol, rtol=0,
                                   err_msg=f"anomaly_map mismatch (batch={batch})")
        np.testing.assert_allclose(os_, ts, atol=atol, rtol=0,
                                   err_msg=f"anomaly_score mismatch (batch={batch})")
        print(f"  batch={batch}: map |Δ|max={np.abs(om-tm).max():.2e}  "
              f"score |Δ|max={np.abs(os_-ts).max():.2e}  OK")
    print("[PASS] Exact mathematical parity confirmed.")


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
        export_fp32(wrapper, config["image_size"], onnx_path, device)

    verify(wrapper, onnx_path, config["image_size"], device)


if __name__ == "__main__":
    main()
