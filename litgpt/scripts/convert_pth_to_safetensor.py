import torch
from pathlib import Path
from safetensors.torch import save_file



def convert_pth_to_safetensors(pth_file_path: Path, safetensors_file_path: Path):
    print(f"Loading PyTorch state dict from: {pth_file_path}")
    

    try:
        state_dict = torch.load(pth_file_path / "model.pth", map_location=torch.device('cpu'))
    except Exception as e:
        print(f"Error loading .pth file: {e}")
        return

    # If the loaded object is a nn.Module, get its state_dict
    if isinstance(state_dict, torch.nn.Module):
        state_dict = state_dict.state_dict()


    print(f"Saving to SafeTensors format at: {safetensors_file_path}")

    save_file(state_dict, safetensors_file_path)
    
    print("Conversion complete!")


