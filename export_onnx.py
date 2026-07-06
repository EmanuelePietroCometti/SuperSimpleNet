import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from model.supersimplenet import SuperSimpleNet

class StaticGaussianBlur(torch.nn.Module):
    """
    A static implementation of Gaussian Blur using a fixed Conv2d layer.
    This bypasses the data-dependent control flow issues of torchvision's 
    GaussianBlur during ONNX/Dynamo export, making it TensorRT-friendly.
    """
    def __init__(self, kernel_size: int, sigma: float):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # Generate a 1D Gaussian kernel
        k = torch.arange(kernel_size, dtype=torch.float32) - self.padding
        kernel_1d = torch.exp(-(k ** 2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Create a 2D kernel via outer product of the 1D kernels
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        
        # Reshape for depthwise conv2d: (out_channels, in_channels/groups, H, W)
        # The anomaly map strictly has 1 channel.
        kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
        
        # Register as a buffer so it is exported as a constant weight in ONNX
        self.register_buffer("weight", kernel_2d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the static convolution to exactly replicate the blur
        return F.conv2d(x, self.weight, padding=self.padding)


def parse_args() -> argparse.Namespace:
    """
    Parses command line arguments for the ONNX export script.
    """
    parser = argparse.ArgumentParser(description="Export SuperSimpleNet to ONNX for C++/TensorRT Inference")
    
    # Positional argument for the exact weights file
    parser.add_argument("weights_path", type=str, help="Path to the trained weights file (.pt/.pth)")
    
    # Architecture and input constraints (must perfectly match training phase)
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
    print(f"Target resolution: {config['image_size']}")
    print(f"Loading weights from: {args.weights_path}")

    # Initialize the Model and load state dict
    model = SuperSimpleNet(image_size=config["image_size"], config=config)
    model.load_model(Path(args.weights_path))
    model.to(device)
    model.eval()

    # Patch Dynamic GaussianBlur with StaticGaussianBlur
    if hasattr(model, 'blur') and model.blur is not None:
        try:
            # Extract the original parameters from torchvision's GaussianBlur
            k_size = model.blur.kernel_size[0]
            sigma = model.blur.sigma[0]
            print(f"Patching dynamic GaussianBlur -> StaticGaussianBlur (kernel={k_size}, sigma={sigma})")
            
            # Inject the static module
            model.blur = StaticGaussianBlur(kernel_size=k_size, sigma=sigma).to(device)
        except Exception as e:
            print(f"Warning: Could not extract blur parameters ({e}). Bypassing internal blur.")
            # Fallback: remove the blur from the graph; C++ engine must handle it via cv::GaussianBlur
            model.blur = torch.nn.Identity() 

    # Create a Dummy Tensor representing a single image batch
    dummy_input = torch.randn(1, 3, args.image_size[0], args.image_size[1], device=device)

    # Define Output Path
    weights_path = Path(args.weights_path)
    onnx_path = weights_path.with_suffix(".onnx")

    # Execute ONNX Export
    print(f"Compiling and exporting computational graph to {onnx_path}...")
    
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=14,                              # Safest baseline for TensorRT compatibility
        do_constant_folding=True,                      # Essential optimization for inference speed
        input_names=["input"],                         # Named input node for C++ retrieval
        output_names=["anomaly_map", "anomaly_score"], # Named output nodes for C++ retrieval
        dynamic_axes={                                 # Enables batch size flexibility during inference
            "input": {0: "batch_size"},
            "anomaly_map": {0: "batch_size"},
            "anomaly_score": {0: "batch_size"}
        }
    )

    print(f"ONNX Export successful! The model is ready for high-performance integration.")

if __name__ == "__main__":
    main()