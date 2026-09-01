import torch
ckpt = torch.load("mace_V2_active_learning.model", map_location="cpu", weights_only=False)
print(type(ckpt))
print([a for a in dir(ckpt) if "head" in a.lower()])
if hasattr(ckpt, "heads"):
    print("heads:", ckpt.heads)
print("atomic_energies shape:", ckpt.atomic_energies_fn.atomic_energies.shape)