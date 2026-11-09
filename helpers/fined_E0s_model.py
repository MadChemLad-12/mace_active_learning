
import torch, json
from configs.constants import FOUNDATION_MODEL_PATH
try:
    model = torch.load(FOUNDATION_MODEL_PATH, map_location='cpu', weights_only=False)
    
    # Extract the atomic numbers using the confirmed model attribute
    zs = model.atomic_numbers
    if isinstance(zs, torch.Tensor):
        zs = zs.tolist()
        
    # Extract raw E0 array
    e0_tensors = model.atomic_energies_fn.atomic_energies
    e0s_dict = {int(z): float(e0_tensors[i]) for i, z in enumerate(zs)}
    
    # Save directly to the target configuration file destination
    output_path = 'configs/foundation_e0s.json'
    with open(output_path, 'w') as f:
        json.dump(e0s_dict, f, indent=4)
    print(f'Successfully compiled foundation E0 parameters to: {output_path}')
except Exception as e:
    print(f'ERROR generating foundation baseline map: {e}')
    exit(1)
