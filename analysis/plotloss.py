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
import numpy as np


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

    # NOTE: captures the head name too — initial RMSE must be tracked
    # per-head, not globally. A global "first Initial: line wins" approach
    # silently assigns one head's initial error to every other head, which
    # corrupts the "Force/Energy Error Reduction %" figures for all heads
    # after the first one in the log.
    initial_pat = re.compile(
        r'Initial:\s+head:\s+([\w.-]+),.*?RMSE_E_per_atom=\s*([\d.]+)\s*meV.*?RMSE_F=\s*([\d.]+)\s*meV'
    )

    lines = Path(log_path).read_text(errors='replace').split('\n')

    # Per-head initial metrics
    initial_by_head = {}
    for line in lines:
        m_init = initial_pat.search(line)
        if m_init:
            h, e_val, f_val = m_init.group(1), float(m_init.group(2)), float(m_init.group(3))
            if h not in initial_by_head:
                initial_by_head[h] = (e_val, f_val)

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
                init_e, init_f = initial_by_head.get(head_val, (None, None))
                heads_data[head_val] = {
                    'epochs':         [],
                    'loss':           [],
                    'rmse_f':         [],
                    'rmse_e':         [],
                    'stage2_epoch':   None,
                    'best_epoch':     None,
                    'initial_rmse_f': init_f,
                    'initial_rmse_e': init_e,
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


def analyze_trends(head_name: str, data: dict, window_frac: float = 0.5) -> dict:
    """
    Fit a simple linear trend (slope via least squares) to RMSE_E and RMSE_F
    over the most recent `window_frac` fraction of epochs, to distinguish
    "still noisily improving" from "steadily degrading" — the pattern that
    matters most for deciding what to change before the next active
    learning round (e.g. pt_head degrading while Default improves signals
    a replay-size or loss-weight problem, not a training bug).

    Returns a dict with slopes (units/epoch), direction labels, and the
    fraction of epoch-to-epoch steps that got worse (noise indicator).
    """
    epochs = data['epochs']
    n = len(epochs)
    if n < 6:
        return {'insufficient_data': True}

    window_start = int(n * (1 - window_frac))
    ep_window = np.array(epochs[window_start:])
    e_window = np.array(data['rmse_e'][window_start:])
    f_window = np.array(data['rmse_f'][window_start:])

    def slope(x, y):
        # least-squares linear fit: y = m*x + c, return m
        A = np.vstack([x, np.ones_like(x)]).T
        m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        return m

    def worsening_fraction(y):
        diffs = np.diff(y)
        return float(np.mean(diffs > 0)) if len(diffs) else 0.0

    e_slope = slope(ep_window, e_window)
    f_slope = slope(ep_window, f_window)

    return {
        'insufficient_data': False,
        'window_epochs': (int(ep_window[0]), int(ep_window[-1])),
        'e_slope_per_epoch': e_slope,
        'f_slope_per_epoch': f_slope,
        'e_direction': 'DEGRADING' if e_slope > 0 else 'improving',
        'f_direction': 'DEGRADING' if f_slope > 0 else 'improving',
        'e_worsening_fraction': worsening_fraction(e_window),
        'f_worsening_fraction': worsening_fraction(f_window),
        'e_total_change': float(e_window[-1] - e_window[0]),
        'f_total_change': float(f_window[-1] - f_window[0]),
    }


def compare_heads_and_recommend(all_heads_data: dict, all_trends: dict) -> list:
    """
    Cross-head comparison producing concrete, actionable suggestions for
    the NEXT active learning round — not just per-head benchmark status.
    Looks specifically for the divergent-heads pattern (one head steadily
    degrading while another improves), which is the earliest signal of
    replay/forgetting issues in multihead fine-tuning, well before it
    becomes a catastrophic Stage 1/SWA-destroying problem.
    """
    recs = []
    degrading_heads = []
    improving_heads = []

    for head, trend in all_trends.items():
        if trend.get('insufficient_data'):
            continue
        is_degrading = (
            trend['e_direction'] == 'DEGRADING' and trend['e_worsening_fraction'] > 0.55
        ) or (
            trend['f_direction'] == 'DEGRADING' and trend['f_worsening_fraction'] > 0.55
        )
        (degrading_heads if is_degrading else improving_heads).append(head)

    if degrading_heads and improving_heads:
        recs.append(
            f"[DIVERGENCE] Head(s) {degrading_heads} degrading while "
            f"{improving_heads} improve. This is the classic replay/forgetting "
            f"signature seen in earlier rounds (Stage 1/SWA E0 inconsistency). "
            f"Before next round: consider increasing --num_samples_pt, raising "
            f"the degrading head's relative loss weight, or checking whether "
            f"the SWA phase (--start_swa) reverses this trend before assuming "
            f"it's a real problem."
        )

    for head in degrading_heads:
        t = all_trends[head]
        recs.append(
            f"[{head}] Energy RMSE trending {t['e_direction']} "
            f"({t['e_slope_per_epoch']:+.3f} meV/epoch over epochs "
            f"{t['window_epochs'][0]}-{t['window_epochs'][1]}, "
            f"worsened in {t['e_worsening_fraction']*100:.0f}% of steps). "
            f"If this continues past --start_swa, treat as an early forgetting "
            f"signal rather than normal noise."
        )

    for head, trend in all_trends.items():
        if trend.get('insufficient_data'):
            continue
        if trend['f_direction'] == 'improving' and trend['f_worsening_fraction'] < 0.3:
            recs.append(
                f"[{head}] Force RMSE improving cleanly "
                f"({trend['f_slope_per_epoch']:+.3f} meV/Å/epoch) — no action needed."
            )

    if not recs:
        recs.append("No strong divergence or degradation trend detected in the analyzed window.")

    return recs


def print_summary_table(head_name: str, data: dict, trend: dict = None):
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

    # Recent-window trend (distinguishes noisy-but-improving from steadily
    # degrading — the latter is the early forgetting/replay-imbalance signal)
    if trend and not trend.get('insufficient_data'):
        print(" RECENT TREND (most recent window):")
        print("─"*65)
        ep_lo, ep_hi = trend['window_epochs']
        print(f"  • Window: epochs {ep_lo}-{ep_hi}")
        print(f"  • Energy RMSE:  {trend['e_direction']:<10s} "
              f"({trend['e_slope_per_epoch']:+.3f} meV/epoch, "
              f"net {trend['e_total_change']:+.2f} meV, "
              f"worsened {trend['e_worsening_fraction']*100:.0f}% of steps)")
        print(f"  • Force RMSE:   {trend['f_direction']:<10s} "
              f"({trend['f_slope_per_epoch']:+.3f} meV/Å/epoch, "
              f"net {trend['f_total_change']:+.2f} meV/Å, "
              f"worsened {trend['f_worsening_fraction']*100:.0f}% of steps)")
        if trend['e_direction'] == 'DEGRADING' or trend['f_direction'] == 'DEGRADING':
            print("  [!] Steady degradation detected in this window — see")
            print("      cross-head recommendations at the end of output.")
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
    all_trends = {}
    for head_name, data in all_heads_data.items():
        print(f"\nProcessing visual rendering for Head Layer -> [{head_name}] ({len(data['epochs'])} epochs found)")
        make_figures(head_name, data, out_dir)
        trend = analyze_trends(head_name, data)
        all_trends[head_name] = trend
        print_summary_table(head_name, data, trend)

    # Cross-head comparison — only meaningful with 2+ heads (multihead
    # fine-tuning runs), and this is where the actionable "what to change
    # before next round" guidance lives.
    if len(all_heads_data) > 1:
        print("\n" + "═"*65)
        print(" NEXT-ROUND RECOMMENDATIONS (cross-head analysis)")
        print("═"*65)
        for rec in compare_heads_and_recommend(all_heads_data, all_trends):
            print(f"  • {rec}")
        print("═"*65 + "\n")

    print("Execution complete.")


if __name__ == '__main__':
    main()