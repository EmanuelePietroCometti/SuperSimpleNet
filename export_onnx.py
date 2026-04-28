import torch
import sys
import types
import torch.nn as nn
from pathlib import Path
from model.supersimplenet import SuperSimpleNet

def safe_discriminator_forward(self, seg_features, cls_features):
    """
    ONNX-safe forward pass for the Discriminator.
    Replaces multi-dimensional squeeze() with flatten() to prevent tracing errors.
    """
    map_out = self.seg(seg_features)

    map_dec_copy = map_out
    if self.stop_grad:
        map_dec_copy = map_dec_copy.detach()
        
    mask_cat = torch.cat((cls_features, map_dec_copy), dim=1)
    dec_out = self.dec_head(mask_cat)

    dec_max = self.dec_max_pool(dec_out)
    dec_avg = self.dec_avg_pool(dec_out)

    map_max = self.map_max_pool(map_out)
    if self.stop_grad:
        map_max = map_max.detach()

    map_avg = self.map_avg_pool(map_out)
    if self.stop_grad:
        map_avg = map_avg.detach()

    dec_cat = torch.cat((dec_max, dec_avg, map_max, map_avg), dim=1).flatten(1)
    score = self.fc_score(dec_cat).flatten()

    return map_out, score


def export_model_to_onnx(weights_path: str, output_dir: str):
    """
    Loads the weights of a SuperSimpleNet model and exports it to ONNX format,
    optimizing it for production inference.
    """
    print("\n--- STARTING ONNX EXPORT ---")
    device = "cpu"
    
    config = {
        "backbone": "wide_resnet50_2", 
        "layers": ["layer1", "layer2", "layer3"], 
        "patch_size": 3,
        "image_size": (256, 256)
    }
    
    try:
        model = SuperSimpleNet(image_size=config["image_size"], config=config).to(device)
    except Exception as e:
        print(f"[ERROR] Unable to initialize the model: {e}")
        return

    weights_file = Path(weights_path)
    if not weights_file.exists():
        print(f"[ERROR] Weights file does not exist: {weights_file}")
        return
        
    print(f"Loading weights from: {weights_file}")
    model.load_model(weights_file)
    model.eval()

    if hasattr(model, 'anomaly_map_generator') and hasattr(model.anomaly_map_generator, 'blur'):
        model.anomaly_map_generator.blur = nn.Identity()
        print("[-] GaussianBlur disabled for export (apply it in post-processing).")

    model.discriminator.forward = types.MethodType(safe_discriminator_forward, model.discriminator)
    print("[-] Discriminator patched for ONNX compatibility.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_file_path = out_dir / "supersimplenet_production.onnx"
    
    dummy_input = torch.randn(1, 3, config["image_size"][0], config["image_size"][1], device=device)
    
    try:
        torch.onnx.export(
            model,                       
            dummy_input,                 
            str(onnx_file_path),         
            export_params=True,          
            opset_version=14,
            do_constant_folding=True,    
            input_names=['input_image'], 
            output_names=['anomaly_map', 'anomaly_score']
        )
        print(f"[SUCCESS] Model successfully exported to: {onnx_file_path}")
    except Exception as e:
        print(f"[ERROR] ONNX export failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python export_onnx.py <weights_path.pt> <output_directory>")
        sys.exit(1)
        
    export_model_to_onnx(sys.argv[1], sys.argv[2])