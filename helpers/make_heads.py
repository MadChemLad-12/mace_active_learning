import json, sys, torch
train, replay, e0_json, model_path = sys.argv[1:5]

def load_json(p):
    return {int(k): float(v) for k, v in json.load(open(p)).items() if k != "Atomic Number"}

m = torch.load(model_path, map_location="cpu", weights_only=False)
zs = [int(z) for z in m.atomic_numbers]
heads = list(m.heads)
row = m.atomic_energies_fn.atomic_energies.detach().reshape(len(heads), len(zs))[heads.index("pt_head")]
keep = set(zs) if (len(sys.argv) > 5 and sys.argv[5] == "all") else {1, 6, 8, 9, 16, 78}
default_e0 = load_json(e0_json) 
default_e0 = load_json(e0_json)                      # your six DFT E0s
if mode_all:
    default_e0 = {z: default_e0.get(z, 0.0) for z in zs}   # zeros for the rest
    pt_e0 = {z: float(row[i]) for i, z in enumerate(zs)}

cfg = {
  "Default": {"train_file": train, "E0s": str(load_json(e0_json)),
              "energy_key": "REF_energy", "forces_key": "REF_forces", "stress_key": "REF_stress"},
  "pt_head": {"train_file": replay, "E0s": str(pt_e0),
              "energy_key": "energy", "forces_key": "forces", "stress_key": "stress"},
}
print(repr(cfg))