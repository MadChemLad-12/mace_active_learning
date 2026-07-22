#!/usr/bin/env python3
"""
plotloss.py -- Plot MACE training loss and RMSE curves from mace_train.log
============================================================================

Parses the multi-head MACE training log and produces figures for each head:
  1. loss_curve_{head}.png     -- loss value vs epoch (Stage 1 and Stage 2)
  2. rmse_curve_{head}.png     -- RMSE_F and RMSE_E vs epoch

Usage:
    python plotloss.py                          # Processes all heads found in the log
    python plotloss.py --head pt_head           # Only plots the 'pt_head' metrics
    python plotloss.py --log mace_train.log --out my_figures/
"""

import argparse
import re
import sys
from pathlib import Path


def parse_log(log_path: str, target_head: str = None) -> dict:
    """
    Parse MACE training log into structured data for found heads.
    If target_head is provided, it filters out all other heads.
    """
    heads_data = {}
    
    # Robust pattern to handle timestamps, variable spacing, and trailing units/stresses
    epoch_line_pat = re.compile(
        r'Epoch\s+(\d+):\s+head:\s+([\w.-]+),\s+loss=([\d.]+),\s*RMSE_E_per_atom=\s*([\d.]+)\s*meV,\s*RMSE_F=\s*([\d.]+)'
    )
    
    initial_pat = re.compile(
        r'Initial:.*?RMSE_E_per_atom=\s*([\d.]+)\s*meV.*?RMSE_F=\s*([\d.]+)\s*meV'
    )
    
    lines = Path(log_path).read_text(errors='replace').split('\n')
    full_text = '\n'.join(lines)

    # Attempt to capture initial metrics globally if present
    initial_e, initial_f = None, None
    m_init = initial_pat.search(full_text)
    if m_init:
        initial_e = float(m_init.group(1))
        initial_f = float(m_init.group(2))

    for line in lines:
        # Parse epoch data
        m = epoch_line_pat.search(line)
        if m:
            epoch_val = int(m.group(1))
            head_val  = m.group(2)
            loss_val  = float(m.group(3))
            e_val     = float(m.group(4))
            f_val     = float(m.group(5))

            # Filter out if a specific head was requested via CLI
            if target_head and head_val != target_head:
                continue

            # Initialize tracking nested dictionary for new heads on the fly
            if head_val not in heads_data:
                heads_data[head_val] = {
                    'epochs':         [],
                    'loss':           [],
                    'rmse_f':         [],
                    'rmse_e':         [],
                    'stage2_epoch':   None,
                    'best_epoch':     None,
                    'initial_rmse_f': initial_f,
                    'initial_rmse_e': initial_e,
                }
            
            # Watch out for Stage 2 markers triggering explicitly inside individual heads
            if 'Changing loss based on Stage Two' in line and not heads_data[head_val]['stage2_epoch']:
                heads_data[head_val]['stage2_epoch'] = epoch_val

            heads_data[head_val]['epochs'].append(epoch_val)
            heads_data[head_val]['loss'].append(loss_val)
            heads_data[head_val]['rmse_e'].append(e_val)
            heads_data[head_val]['rmse_f'].append(f_val)

    # Find best epoch (lowest RMSE_F) for each collected head
    for head, data in heads_data.items():
        if data['rmse_f']:
            best_idx = data['rmse_f'].index(min(data['rmse_f']))
            data['best_epoch'] = data['epochs'][best_idx]
            
    return heads_data


def print_summary_table(head_name: str, data: dict):
    """Prints a structured performance report with scientific accuracy benchmarks."""
    epochs = data['epochs']
    best_ep = data['best_epoch']
    min_f = min(data['rmse_f'])
    final_f = data['rmse_f'][-1]
    min_e = min(data['rmse_e'])
    final_e = data['rmse_e'][-1]

    print("\n" + "═"*65)
    print(f" TRAINING SUMMARY REPORT: HEAD [{head_name}]")
    print("═"*65)
    print(f"  Total Completed Epochs:  {max(epochs)}")
    print(f"  Best Force Checkpoint:   {min_f:.2f} meV/Å  (Epoch {best_ep})")
    print(f"  Final Force Metric:      {final_f:.2f} meV/Å  (Epoch {max(epochs)})")
    print(f"  Best Energy Checkpoint:  {min_e:.2f} meV/atom")
    print(f"  Final Energy Metric:     {final_e:.2f} meV/atom")
    
    if data['initial_rmse_f']:
        f_impr = (data['initial_rmse_f'] - min_f) / data['initial_rmse_f'] * 100
        print(f"  Force Error Reduction:   {f_impr:.1f}% ({data['initial_rmse_f']:.1f} -> {min_f:.1f} meV/Å)")
    print("─"*65)
    
    # Target Evaluation Framework
    print(" ACCURACY BENCHMARK ASSESSMENT:")
    print("─"*65)
    
    # 1. Force targets Evaluation
    print(f"  • Current Best Force RMSE: {min_f:.1f} meV/Å")
    if min_f > 100:
        print("    [!] TARGET FAIL: Poor structural resolution. Bad for geometries.")
    elif 50 < min_f <= 100:
        print("    [✓] ACCEPTABLE: Suitable for bulk relaxation, rough screening.")
    elif 25 <= min_f <= 50:
        print("    [✓] GOOD: Reliable for complex interfaces & standard pathways.")
    else:
        print("    [★] EXCELLENT: Publication quality. High accuracy for surface reactions/NEB.")

    # 2. Energy targets Evaluation
    print(f"  • Current Best Energy RMSE: {min_e:.1f} meV/atom")
    if min_e > 10:
        print("    [!] TARGET FAIL: Far from chemical accuracy (~43 meV/mol). High risk of false phases.")
    elif 3 < min_e <= 10:
        print("    [✓] ACCEPTABLE: Decent for relative differences, watch out for fine barriers.")
    else:
        print("    [★] EXCELLENT: Ideal for precise thermodynamics and surface coverage phase diagrams.")
    print("═"*65 + "\n")


def make_figures(head_name: str, data: dict, out_dir: Path):
    """Generate and save both figures for a specific target training head."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("ERROR: matplotlib not installed. Run: pip install matplotlib")
        sys.exit(1)

    if not data['epochs']:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    epochs  = data['epochs']
    stage2  = data['stage2_epoch']
    best_ep = data['best_epoch']

    # Colour scheme
    col_force  = '#2196F3'
    col_energy = '#FF5722'
    col_loss   = '#4CAF50'
    col_stage2 = '#9C27B0'
    col_best   = '#F44336'

    def add_stage_annotations(ax):
        if stage2 and stage2 in epochs:
            ax.axvline(stage2, color=col_stage2, linestyle='--', linewidth=1.5, alpha=0.7)
            ax.axvspan(stage2, max(epochs), alpha=0.05, color=col_stage2)
        if best_ep is not None:
            ax.axvline(best_ep, color=col_best, linestyle=':', linewidth=2, alpha=0.9)

    # =========================================================================
    # FIGURE 1: RMSE CURVES
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f'MACE Training Progress — Head: {head_name}', fontsize=14, fontweight='bold')

    # Top panel: RMSE_F
    ax1.plot(epochs, data['rmse_f'], color=col_force, linewidth=2, label='RMSE_F (validation)', zorder=3)

    if data['initial_rmse_f']:
        ax1.axhline(data['initial_rmse_f'], color=col_force, linestyle=':', alpha=0.4, label='Initial RMSE_F')

    # Scientific Targets for Forces
    for thresh, label in [(100, '100 meV/Å (Acceptable Screening)'), 
                          (50, '50 meV/Å (Good Geometry)'), 
                          (25, '25 meV/Å (Publication Quality)')]:
        ax1.axhline(thresh, color='gray', linestyle='-.', alpha=0.4, linewidth=0.8)
        ax1.text(max(epochs) * 0.98, thresh + (thresh*0.05), label, ha='right', va='bottom', fontsize=7, color='gray')

    add_stage_annotations(ax1)

    if best_ep is not None and data['rmse_f']:
        best_val = min(data['rmse_f'])
        ax1.annotate(f"Best: {best_val:.1f} meV/Å\n(ep {best_ep})",
                    xy=(best_ep, best_val), xytext=(best_ep + max(epochs)*0.03, best_val * 1.3),
                    arrowprops=dict(arrowstyle='->', color=col_best), fontsize=9, color=col_best)

    ax1.set_ylabel('RMSE_F (meV/Å)', fontsize=11)
    ax1.set_yscale('log')
    ax1.set_ylim(bottom=max(5, min(data['rmse_f']) * 0.5))
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Bottom panel: RMSE_E
    ax2.plot(epochs, data['rmse_e'], color=col_energy, linewidth=2, label='RMSE_E/atom (validation)', zorder=3)

    if data['initial_rmse_e']:
        ax2.axhline(data['initial_rmse_e'], color=col_energy, linestyle=':', alpha=0.4, label='Initial RMSE_E')

    # Scientific Targets for Energy
    for thresh, label in [(10, '10 meV/atom (Acceptable Screening)'), 
                          (3, '3 meV/atom (Chemical Accuracy Target)')]:
        ax2.axhline(thresh, color='gray', linestyle='-.', alpha=0.4, linewidth=0.8)
        ax2.text(max(epochs) * 0.98, thresh + (thresh*0.05), label, ha='right', va='bottom', fontsize=7, color='gray')

    add_stage_annotations(ax2)

    ax2.set_ylabel('RMSE_E (meV/atom)', fontsize=11)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_yscale('log')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    rmse_path = out_dir / f'rmse_curve_{head_name}.png'
    plt.savefig(rmse_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {rmse_path}")

    # =========================================================================
    # FIGURE 2: LOSS CURVE
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(f'MACE Training Loss — Head: {head_name}', fontsize=14, fontweight='bold')

    ax.plot(epochs, data['loss'], color=col_loss, linewidth=2, label='Training loss')
    if best_ep:
        ax.axvline(best_ep, color=col_best, linestyle=':', linewidth=2, label='Best Checkpoint')
    if stage2:
        ax.axvline(stage2, color=col_stage2, linestyle='--', linewidth=1.5, label='Stage 2 split')

    ax.set_ylabel('Loss (dimensionless)', fontsize=11)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_yscale('log')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    loss_path = out_dir / f'loss_curve_{head_name}.png'
    plt.savefig(loss_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {loss_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--log', default='mace_train.log', help='Path to MACE training log')
    parser.add_argument('--out', default='.', help='Output directory for figures')
    parser.add_argument('--head', default=None, help='Select specific head to analyze (default: process all heads found)')
    args = parser.parse_args()

    log_path = args.log
    if not Path(log_path).exists():
        print(f"ERROR: Log file not found: {log_path}")
        sys.exit(1)

    if args.head:
        print(f"Scanning and Parsing content from: {log_path} specifically for head: {args.head}")
    else:
        print(f"Scanning and Parsing content from: {log_path} (All heads mode)")
        
    all_heads_data = parse_log(log_path, target_head=args.head)

    if not all_heads_data:
        print(f"ERROR: No matching target head entries parsed successfully.")
        sys.exit(1)

    out_dir = Path(args.out)
    for head_name, data in all_heads_data.items():
        print(f"\nProcessing visual rendering for Head Layer -> [{head_name}] ({len(data['epochs'])} epochs found)")
        make_figures(head_name, data, out_dir)
        print_summary_table(head_name, data)
        
    print("Execution complete.")


if __name__ == '__main__':
    main()