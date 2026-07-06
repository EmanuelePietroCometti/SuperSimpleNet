import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from model.supersimplenet import SuperSimpleNet

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
        
        # Generate 1D and 2D Gaussian kernels
        k = torch.arange(kernel_size, dtype=torch.float32) - self.padding
        kernel_1d = torch.exp(-(k ** 2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
        
        # Register as static ONNX weight
        self.register_buffer("weight", kernel_2d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, padding=self.padding)


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
    """Recursively replaces nn.AdaptiveMaxPool2d in the module tree."""
    for child_name, child in model.named_children():
        if isinstance(child, torch.nn.AdaptiveMaxPool2d):
            if child.output_size == 1 or child.output_size == (1, 1):
                setattr(model, child_name, GlobalMaxPool2d())
        else:
            patch_adaptive_max_pool(child)

# Monkey-patch the functional API globally just in case it is called directly
_orig_ada_max = F.adaptive_max_pool2d
def _patched_ada_max(input, output_size, return_indices=False):
    if output_size == (1, 1) or output_size == 1:
        return torch.amax(input, dim=[-2, -1], keepdim=True)
    return _orig_ada_max(input, output_size, return_indices)
F.adaptive_max_pool2d = _patched_ada_max


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
    
    model = SuperSimpleNet(image_size=config["image_size"], config=config)
    model.load_model(Path(args.weights_path))
    model.to(device)
    model.eval()

    # --- Apply Network Patches ---
    
    # Patch Gaussian Blur
    if hasattr(model, 'anomaly_map_generator') and hasattr(model.anomaly_map_generator, 'blur'):
        blur_module = model.anomaly_map_generator.blur
        if hasattr(blur_module, 'kernel_size') and hasattr(blur_module, 'sigma'):
            k_size = blur_module.kernel_size[0]
            sigma = blur_module.sigma[0]
            model.anomaly_map_generator.blur = StaticGaussianBlur(kernel_size=k_size, sigma=sigma).to(device)
        else:
            model.anomaly_map_generator.blur = torch.nn.Identity()

    # Patch Adaptive Max Pooling
    patch_adaptive_max_pool(model)

    # --- Export ---
    dummy_input = torch.randn(1, 3, args.image_size[0], args.image_size[1], device=device)
    weights_path = Path(args.weights_path)
    onnx_path = weights_path.with_suffix(".onnx")

    print(f"Compiling strictly static computational graph to {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=14,                              
        do_constant_folding=True,                      
        input_names=["input"],                         
        output_names=["anomaly_map", "anomaly_score"], 
        dynamic_axes={                                 
            "input": {0: "batch_size"},
            "anomaly_map": {0: "batch_size"},
            "anomaly_score": {0: "batch_size"}
        }
    )

    print(f"✅ ONNX Export successful! Ready for C++ TensorRT engine.")

if __name__ == "__main__":
    main()