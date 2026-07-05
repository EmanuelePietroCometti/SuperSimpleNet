import argparse
import torch
from pathlib import Path
from model.supersimplenet import SuperSimpleNet

def parse_args():
    parser = argparse.ArgumentParser(description="Export SuperSimpleNet to ONNX for C++/TensorRT Inference")
    
    # Positional argument for the weights file
    parser.add_argument("weights_path", type=str, help="Path to the trained weights file (.pt/.pth)")
    
    # Architecture and input (must exactly match the training configuration)
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
        "adapt_cls_feat": True, # Kept to True as per default setup
    }

    print(f"\n--- Exporting SuperSimpleNet to ONNX ---")
    print(f"Target resolution: {config['image_size']}")
    print(f"Loading weights from: {args.weights_path}")

    # Model initialization
    model = SuperSimpleNet(image_size=config["image_size"], config=config)
    model.load_model(Path(args.weights_path))
    model.to(device)
    model.eval()

    # Dummy tensor creation
    # Shape: (Batch_Size, Channels, Height, Width)
    dummy_input = torch.randn(1, 3, args.image_size[0], args.image_size[1], device=device)

    # Output path definition
    weights_path = Path(args.weights_path)
    onnx_path = weights_path.with_suffix(".onnx")

    # ONNX Export Optimized for TensorRT
    print(f"Compiling and exporting computational graph to {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=14,                  # Stable version for TensorRT
        do_constant_folding=True,          # Pre-inference optimization
        input_names=["input"],             # Input node name exposed to C++
        output_names=["anomaly_map", "anomaly_score"], # Output node names exposed to C++
        dynamic_axes={                     # Allows dynamic batch sizes during inference if needed
            "input": {0: "batch_size"},
            "anomaly_map": {0: "batch_size"},
            "anomaly_score": {0: "batch_size"}
        }
    )

    print(f"ONNX Export successful! Ready for high-performance inference.")

if __name__ == "__main__":
    main()