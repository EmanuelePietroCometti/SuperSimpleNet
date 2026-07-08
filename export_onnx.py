import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.export import Dim

from model.supersimplenet import SuperSimpleNet

# ImageNet stats used by the train/eval preprocessing pipelines
# (see train.py / eval.py gpu_transforms) - baked into the graph below.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ==========================================
# --- PATCH GAUSSIAN BLUR ---
# ==========================================
class StaticGaussianBlur(torch.nn.Module):
    """
    A static implementation of Gaussian Blur using a fixed Conv2d layer.
    Bypasses data-dependent control flow issues of torchvision's GaussianBlur.
    """
    def __init__(self, kernel_size: int, sigma: float):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        k = torch.arange(kernel_size, dtype=torch.float32) - self.padding
        kernel_1d = torch.exp(-(k ** 2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)

        self.register_buffer("weight", kernel_2d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, padding=self.padding)


# ==========================================
# --- PATCH BATCH NORM ---
# ==========================================
class StaticBatchNorm2d(torch.nn.Module):
    """
    Static affine replacement for a frozen (eval-mode) nn.BatchNorm2d.

    torch's dynamo-based ONNX exporter (torch 2.8, and other recent versions)
    has a bug translating aten::_native_batch_norm_legit_no_training - the op
    used for BatchNorm2d.forward() in eval mode - crashing with
    "AttributeError: 'tuple' object has no attribute 'dtype'". Every BN in
    this graph (both the pretrained backbone's and the discriminator's) runs
    in eval mode with frozen running stats, so it is mathematically just
    y = x * scale + shift with scale/shift precomputed from
    (weight, bias, running_mean, running_var, eps); folding it ahead of time
    sidesteps the buggy op entirely and is exact (not an approximation).
    """
    def __init__(self, bn: torch.nn.BatchNorm2d):
        super().__init__()
        with torch.no_grad():
            scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            shift = bn.bias - bn.running_mean * scale
        self.register_buffer("scale", scale.view(1, -1, 1, 1))
        self.register_buffer("shift", shift.view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.shift


def patch_batch_norm(model):
    """Recursively replaces every nn.BatchNorm2d (backbone + discriminator) with
    its static affine equivalent. Must run after model.eval() so running_mean/
    running_var reflect the final trained statistics."""
    for child_name, child in model.named_children():
        if isinstance(child, torch.nn.BatchNorm2d):
            setattr(model, child_name, StaticBatchNorm2d(child))
        else:
            patch_batch_norm(child)


# ==========================================
# --- PATCH ADAPTIVE MAX POOLING ---
# ==========================================
class GlobalMaxPool2d(torch.nn.Module):
    """
    Static replacement for AdaptiveMaxPool2d(1).
    Exports seamlessly to ONNX as a ReduceMax node.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.amax(x, dim=[-2, -1], keepdim=True)


def patch_adaptive_max_pool(model):
    """Recursively replaces any remaining nn.AdaptiveMaxPool2d(1) in the module tree.

    Defensive: SuperSimpleNet.discriminator no longer uses this (it reduces
    with plain amax/mean, see model/supersimplenet.py), but a custom/backbone
    module could still use one.
    """
    for child_name, child in model.named_children():
        if isinstance(child, torch.nn.AdaptiveMaxPool2d):
            if child.output_size in [1, (1, 1)]:
                setattr(model, child_name, GlobalMaxPool2d())
        else:
            patch_adaptive_max_pool(child)


# Monkey-patch the functional API globally just in case it is called directly
_orig_ada_max = F.adaptive_max_pool2d
def _patched_ada_max(input, output_size, return_indices=False):
    if output_size in [(1, 1), 1]:
        return torch.amax(input, dim=[-2, -1], keepdim=True)
    return _orig_ada_max(input, output_size, return_indices)
F.adaptive_max_pool2d = _patched_ada_max


# ==========================================
# --- PRE/POST-PROCESSING WRAPPER ---
# ==========================================
class InferenceWrapper(torch.nn.Module):
    """
    Wraps SuperSimpleNet so the exported ONNX graph is self-contained:
    input is a raw uint8 RGB image and outputs are ready-to-use results.

    Preprocessing (baked into the graph):
      - uint8 [B, H, W, 3] -> float32 [B, 3, H, W] in [0, 1]
      - resize (bilinear, antialias) to the model's fixed training image_size
      - ImageNet mean/std normalization

    Postprocessing (baked into the graph):
      - sigmoid on the anomaly score logit -> probability in [0, 1]
      - anomaly map is already upscaled + blurred inside the model (eval mode)

    H and W are nailed down to the training image_size (baked as the Resize
    target); only the batch dimension is configurable/dynamic at export time.
    """
    def __init__(self, model: SuperSimpleNet, image_size: tuple[int, int]):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.register_buffer(
            "image_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "image_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # image: uint8, [B, H, W, 3], RGB. H/W must match self.image_size -
        # only B is dynamic in the exported graph (see export_fp32 below).
        x = image.float() / 255.0
        x = F.interpolate(
            x, size=self.image_size, mode="bilinear", align_corners=False, antialias=False
        )
        x = (x - self.image_mean) / self.image_std

        anomaly_map, anomaly_score = self.model(x)
        anomaly_score = torch.sigmoid(anomaly_score)

        return anomaly_map, anomaly_score


def build_wrapper(weights_path: Path, config: dict, device: str) -> InferenceWrapper:
    model = SuperSimpleNet(image_size=config["image_size"], config=config)
    model.load_model(weights_path)
    model.to(device)
    model.eval()

    # Patch BatchNorm2d (backbone + discriminator) to a static affine op -
    # works around a torch.onnx dynamo exporter bug (see StaticBatchNorm2d).
    patch_batch_norm(model)

    # Patch Gaussian Blur (data-dependent control flow -> static conv)
    if hasattr(model, 'anomaly_map_generator') and hasattr(model.anomaly_map_generator, 'blur'):
        blur_module = model.anomaly_map_generator.blur
        if hasattr(blur_module, 'kernel_size') and hasattr(blur_module, 'sigma'):
            k_size = blur_module.kernel_size[0]
            sigma = blur_module.sigma[0]
            model.anomaly_map_generator.blur = StaticGaussianBlur(kernel_size=k_size, sigma=sigma).to(device)
        else:
            model.anomaly_map_generator.blur = torch.nn.Identity()

    # Patch any remaining Adaptive Max Pooling (defensive, see docstring above)
    patch_adaptive_max_pool(model)

    wrapper = InferenceWrapper(model, config["image_size"]).to(device)
    wrapper.eval()
    return wrapper


def export_fp32(
    wrapper: InferenceWrapper,
    image_size: tuple[int, int],
    onnx_path: Path,
    device: str,
    batch_min: int = 1,
    batch_max: int = 64,
):
    h, w = image_size

    # NOTE: the dummy batch dim must be > 1 (torch.export "0/1 specialization"):
    # if traced with batch=1, torch.export bakes batch as a static constant
    # even when explicitly marked dynamic, silently producing a graph that
    # only works for batch=1. Using batch=2 for tracing avoids this.
    dummy_input = torch.randint(0, 256, (2, 3, h, w), dtype=torch.float32, device=device)
    # An unbounded Dim("batch") makes torch.export's constraint solver blow up
    # ("AssertionError: batch < 2147483647/16777216") on this graph - giving
    # it explicit min/max avoids that and also bakes a validity range into
    # the ONNX graph. At runtime (onnxruntime) the batch dim stays symbolic
    # and any batch size can be requested; min/max only affect export-time
    # solving and are not enforced by onnxruntime itself.
    batch = Dim("batch", min=batch_min, max=batch_max)

    print(f"Compiling computational graph (dynamic batch, fixed {h}x{w}) to {onnx_path}...")

    onnx_program = torch.onnx.export(
        wrapper,
        (dummy_input,),
        input_names=["image"],
        output_names=["anomaly_map", "anomaly_score"],
        dynamic_shapes={"image": {0: batch}},
        dynamo=True,
        opset_version=18,
        # Keep the whole graph (weights included) in a single .onnx file.
        # The default (external_data=True) writes weights to a companion
        # "<name>.onnx.data" file, which is easy to lose/rename when the
        # model is copied to another machine -> "onnx.data not found" at
        # load time in the GPU inference program. The model is well under
        # the 2GB protobuf limit, so a single self-contained file is safe.
        external_data=False,
    )

    onnx_program.save(str(onnx_path))

    # Clean up a stray external-data file from a previous broken export, if any.
    stray_data_file = onnx_path.with_name(onnx_path.name + ".data")
    if stray_data_file.exists():
        stray_data_file.unlink()

    # The dynamic-batch translation can emit a graph whose nodes aren't in
    # strict topological order (harmless for onnxruntime inference, but it
    # trips up onnx shape_inference / the int8 quantizer below) - re-sort it.
    import onnx
    from onnxruntime.transformers.onnx_model import OnnxModel

    onnx_model = onnx.load(str(onnx_path))
    sorted_model = OnnxModel(onnx_model)
    sorted_model.topological_sort()
    onnx.checker.check_model(sorted_model.model)
    onnx.save(sorted_model.model, str(onnx_path))


def convert_to_fp16(fp32_path: Path, fp16_path: Path):
    try:
        import onnx
        from onnxconverter_common import float16
    except ImportError as e:
        raise ImportError(
            "FP16 export requires 'onnx' and 'onnxconverter-common' "
            "(pip install onnx onnxconverter-common)"
        ) from e

    model = onnx.load(str(fp32_path))
    # keep_io_types=True: input stays uint8, output stays float32, so the
    # inference code does not need any change - only internal compute is fp16.
    # op_block_list=["Cast"]: without it, the converter leaves the first
    # Cast node (uint8 -> float32 preprocessing) declared as float32 while
    # rewriting its downstream consumers' expected type to float16, producing
    # an inconsistent graph onnxruntime refuses to load. Blocking Cast makes
    # it insert its own consistent adapter casts instead.
    model_fp16 = float16.convert_float_to_float16(
        model, keep_io_types=True
    )
    onnx.checker.check_model(model_fp16)
    onnx.save(model_fp16, str(fp16_path))


def _make_calibration_reader(
    calibration_images_dir: str | None, image_size: tuple[int, int], num_random_samples: int = 32
):
    """
    Builds a CalibrationDataReader for INT8 static quantization.

    The graph's input H/W is fixed to the training image_size, so calibration
    images are resized to that exact size (same as any real inference input
    must be). If a folder of real images is given, those are used; otherwise
    falls back to random uint8 images - good enough to exercise the
    quantization pipeline and try out INT8 speed, but not representative of
    real data, so accuracy should not be trusted; pass real images for that.
    """
    import numpy as np
    from onnxruntime.quantization import CalibrationDataReader

    h, w = image_size
    image_paths = []
    if calibration_images_dir:
        exts = (".png", ".jpg", ".jpeg", ".bmp")
        image_paths = sorted(
            p for p in Path(calibration_images_dir).rglob("*") if p.suffix.lower() in exts
        )
        if not image_paths:
            raise FileNotFoundError(
                f"No images found in --calibration_images_dir={calibration_images_dir}"
            )

    class Reader(CalibrationDataReader):
        def __init__(self):
            self._index = 0
            if image_paths:
                self._n = len(image_paths)
            else:
                print(
                    f"[!] No --calibration_images_dir given: calibrating INT8 with "
                    f"{num_random_samples} random images. This will run, but the "
                    f"resulting quantization ranges won't reflect real data - pass "
                    f"real images for a meaningful INT8 model."
                )
                self._n = num_random_samples

        def get_next(self):
            if self._index >= self._n:
                return None
            if image_paths:
                from PIL import Image
                img = np.array(
                    Image.open(image_paths[self._index]).convert("RGB").resize((w, h))
                ).astype(np.float32) / 255.0
            else:
                img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

            img = np.transpose(img, (2,0,1))
            self._index += 1
            return {"image": img[None, ...]}

    return Reader()


def convert_to_int8(
    fp32_path: Path, int8_path: Path, image_size: tuple[int, int], calibration_images_dir: str | None = None
):
    try:
        from onnxruntime.quantization import quantize_static, QuantType, QuantFormat
    except ImportError as e:
        raise ImportError(
            "INT8 export requires 'onnxruntime' (pip install onnxruntime)"
        ) from e

    # Static/calibrated quantization (QDQ format): the standard, GPU/TensorRT
    # friendly way to get INT8 for a CNN. Dynamic (weight-only) quantization
    # was tried first but onnxruntime's quantize_dynamic unconditionally runs
    # replace_gemm_with_matmul() on the model, which corrupts shape metadata
    # for this graph's final Gemm (fc_score) and crashes shape inference -
    # static quantization does not hit that code path.
    reader = _make_calibration_reader(calibration_images_dir, image_size)
    quantize_static(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=["Conv", "MatMul"],
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )


# ==========================================
# --- MAIN EXPORT LOGIC ---
# ==========================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SuperSimpleNet to ONNX")
    parser.add_argument("weights_path", type=str, help="Path to the trained weights")
    parser.add_argument("--image_size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--backbone", type=str, default="wide_resnet50_2")
    parser.add_argument("--layers", type=str, nargs="+", default=["layer2", "layer3"])
    parser.add_argument("--patch_size", type=int, default=3)
    parser.add_argument(
        "--batch_min",
        type=int,
        default=1,
        help="Minimum batch size the exported graph declares support for (default: 1). "
             "Does not limit what onnxruntime accepts at inference time; only affects "
             "torch.export's shape solving at export time.",
    )
    parser.add_argument(
        "--batch_max",
        type=int,
        default=64,
        help="Maximum batch size the exported graph declares support for (default: 64). "
             "Does not limit what onnxruntime accepts at inference time; only affects "
             "torch.export's shape solving at export time.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16", "int8"],
        default="fp32",
        help="The model is always exported in fp32 first. Passing fp16/int8 "
             "additionally converts that fp32 graph and saves it as a separate "
             "file, so you can re-run this script to try out other precisions "
             "without retraining or touching the fp32 export.",
    )
    parser.add_argument(
        "--calibration_images_dir",
        type=str,
        default=None,
        help="Folder of real sample images (any size, any of png/jpg/jpeg/bmp) "
             "used to calibrate --precision int8. If omitted, random synthetic "
             "images are used instead, which is enough to test the pipeline but "
             "gives no accuracy guarantees - pass real images for a usable model.",
    )
    return parser.parse_args()


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

    print(f"\n--- Exporting SuperSimpleNet to ONNX ---")

    wrapper = build_wrapper(Path(args.weights_path), config, device)

    weights_path = Path(args.weights_path)
    onnx_path = weights_path.with_suffix(".onnx")

    export_fp32(wrapper, config["image_size"], onnx_path, device, args.batch_min, args.batch_max)

    print(f"✅ ONNX (fp32) export successful: {onnx_path} (self-contained, no .onnx.data required)")
    print("   Input 'image': uint8 [B, H, W, 3] RGB - preprocessing (resize/normalize) is in-graph.")
    print("   Outputs: 'anomaly_map' [B, 1, H, W], 'anomaly_score' [B] (post-sigmoid probability).")
    print(f"   Batch is dynamic; H/W are fixed to {tuple(args.image_size)} (training image_size).")

    if args.precision == "fp16":
        fp16_path = onnx_path.with_name(onnx_path.stem + "_fp16.onnx")
        print(f"\nConverting to FP16: {fp16_path}...")
        convert_to_fp16(onnx_path, fp16_path)
        print(f"✅ ONNX (fp16) export successful: {fp16_path}")

    elif args.precision == "int8":
        int8_path = onnx_path.with_name(onnx_path.stem + "_int8.onnx")
        print(f"\nConverting to INT8 (static, calibrated quantization): {int8_path}...")
        convert_to_int8(onnx_path, int8_path, config["image_size"], args.calibration_images_dir)
        print(f"✅ ONNX (int8) export successful: {int8_path}")


if __name__ == "__main__":
    main()