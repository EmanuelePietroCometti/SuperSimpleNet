# export_onnx.py
import torch
import sys
from pathlib import Path
from model.supersimplenet import SuperSimpleNet

def export_model_to_onnx(weights_path: str, output_dir: str):
    """
    Carica i pesi di un modello SuperSimpleNet e lo esporta in formato ONNX.
    """
    print("\n--- AVVIO ESPORTAZIONE ONNX ---")
    device = "cpu"
    config = {
        "backbone": "wide_resnet50_2", 
        "layers": ["layer1", "layer2", "layer3"], 
        "image_size": (256, 256)
    }
    
    try:
        model = SuperSimpleNet(image_size=config["image_size"], config=config).to(device)
    except Exception as e:
        print(f"[ERRORE] Impossibile inizializzare il modello: {e}")
        return

    weights_file = Path(weights_path)
    if not weights_file.exists():
        print(f"[ERRORE] Il file dei pesi non esiste: {weights_file}")
        return
        
    print(f"Caricamento pesi da: {weights_file}")
    model.load_model(weights_file)
    model.eval()

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
            opset_version=11,            
            do_constant_folding=True,    
            input_names=['input_image'], 
            output_names=['anomaly_map', 'anomaly_score'], 
            dynamic_axes={
                'input_image': {0: 'batch_size'},
                'anomaly_map': {0: 'batch_size'},
                'anomaly_score': {0: 'batch_size'}
            }
        )
        print(f"[SUCCESS] Modello esportato correttamente in: {onnx_file_path}")
    except Exception as e:
        print(f"[ERRORE] Esportazione ONNX fallita: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python export_onnx.py <percorso_pesi.pt> <cartella_di_output>")
        sys.exit(1)
        
    export_model_to_onnx(sys.argv[1], sys.argv[2])