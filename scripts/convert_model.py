import torch
import sys
sys.path.append('.')
from mobile_sam import sam_model_registry, SamPredictor

print("Loading PyTorch model...")
checkpoint = torch.load("../web/public/models/mobile_sam.pt", map_location=torch.device('cpu'))
mobile_sam = sam_model_registry["vit_t"]()
mobile_sam.load_state_dict(checkpoint)
mobile_sam.eval()

print("Converting to ONNX...")
dummy_input = torch.randn(1, 3, 1024, 1024)
torch.onnx.export(
    mobile_sam,
    dummy_input,
    "../web/public/models/mobile_sam.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print("Model converted successfully!") 