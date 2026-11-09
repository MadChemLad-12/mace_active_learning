import json, sys, torch, os

train, replay, e0_json, model_path = sys.argv[1:5]
mode_all = len(sys.argv) > 5 and sys.argv[5] == "all"

def load_json(p):
    with open(p) as f:
        return {int(k): float(v) for k, v in json.load(f).items() if k != "Atomic Number"}

m = torch.load(model_path, map_location="cpu", weights_only=False)
zs = [int(z) for z in m.atomic_numbers]
heads = list(m.heads)
row = m.atomic_energies_fn.atomic_energies.detach().reshape(len(heads), len(zs))[heads.index("pt_head")]

six = load_json(e0_json)                       # your DFT E0s
if mode_all:
    default_e0 = {z: six.get(z, 0.0) for z in zs}          # zeros for the rest
    pt_e0 = {z: float(row[i]) for i, z in enumerate(zs)}
else:
    default_e0 = six
    pt_e0 = {z: float(row[i]) for i, z in enumerate(zs) if z in six}

vf = float(os.getenv("VALIDATION_FRACTION", "0.1"))

cfg = {
    "Default": {"train_file": train, "E0s": str(default_e0),
                "energy_key": "REF_energy", "forces_key": "REF_forces", "stress_key": "REF_stress",
                "test_file": os.environ["EVAL_CONFIGS"], "valid_fraction": vf,
                "config_type_weights": str(json.loads(os.environ["CONFIG_TYPE_WEIGHTS"]))},
    "pt_head": {"train_file": replay, "E0s": str(pt_e0),
                "energy_key": "energy", "forces_key": "forces", "stress_key": "stress",
                "valid_fraction": vf},
}
print(repr(cfg))