"""
Pre-flight OOM check for MACE training batches.

Loads your actual foundation model, builds a real batch from structures
you specify (or your single largest structures repeated to worst-case
batch composition), runs one forward + backward pass, and reports peak
GPU memory. Compare against your GPU's total memory to know BEFORE
submitting whether a given (BATCH_SIZE, MAX_COUNT) combination is safe.

Usage:
    python oom_preflight.py --pool master_train_pool.xyz \
        --model mace-mp-0b3-medium-float32.model \
        --batch_size 4 --max_count 350
"""
import argparse
import sys
import ase.io
import torch
from mace.calculators import MACECalculator
from mace import data
from mace.tools import torch_geometric


def build_worst_case_batch(pool_path, batch_size, max_count, r_max, z_table):
    frames = ase.io.read(pool_path, ":")
    # Filter to allowed size, same as check_residuals.py's MAX_COUNT gate
    frames = [a for a in frames if len(a) <= max_count]
    frames.sort(key=len, reverse=True)
    worst_case = frames[:batch_size]

    print(f"Worst-case batch: {[len(a) for a in worst_case]} atoms "
          f"(total={sum(len(a) for a in worst_case)})")

    configs = [data.Configuration(
        atomic_numbers=a.numbers,
        positions=a.positions,
        properties={},
        property_weights={},
        cell=a.cell[:] if a.cell is not None else None,
        pbc=a.pbc,
    ) for a in worst_case]

    atomic_data = [
        data.AtomicData.from_config(c, z_table=z_table, cutoff=r_max)
        for c in configs
    ]
    loader = torch_geometric.dataloader.DataLoader(
        dataset=atomic_data, batch_size=len(atomic_data), shuffle=False
    )
    return next(iter(loader))


def profile_batch(model_path, pool_path, batch_size, max_count, r_max, device="cuda", threshold=0.85) -> bool:
    calc = MACECalculator(model_paths=model_path, device=device, default_dtype="float32")
    model = calc.models[0]
    model.train()  # backward pass needs training-mode graph retention

    # Use the model's OWN z_table (e.g. mace-mp foundation models cover ~89 elements)
    batch = build_worst_case_batch(pool_path, batch_size, max_count, r_max, calc.z_table)

    torch.cuda.reset_peak_memory_stats()
    batch = batch.to(device)

    out = model(batch.to_dict(), training=True)
    loss = out["energy"].sum() + out["forces"].sum()
    loss.backward()

    torch.cuda.synchronize()
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    is_safe = peak_gb < (threshold * total_gb)

    print(f"\nPeak GPU memory used: {peak_gb:.2f} GB")
    print(f"GPU total memory:     {total_gb:.2f} GB")
    print(f"Headroom:             {total_gb - peak_gb:.2f} GB "
          f"({'SAFE' if is_safe else 'RISKY — reduce batch_size or max_count'})")
    
    return is_safe


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_count", type=int, default=350)
    p.add_argument("--r_max", type=float, default=5.0)
    args = p.parse_args()

    is_safe = profile_batch(
        args.model, args.pool, args.batch_size, args.max_count, args.r_max
    )
    
    # Exit 0 if safe (passes shell 'if'), exit 1 if unsafe (fails shell 'if')
    sys.exit(0 if is_safe else 1)