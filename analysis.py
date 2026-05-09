"""
从 LAMMPS log 中提取降温阶段的密度-温度数据，
用分段线性拟合计算 Tg，支持多个组成对比。
"""

import sys, os
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def parse_cooling_block(lines, start):
    """
    Look for cooling data in a list of log lines starting from 'start'.
    Returns (temps, densities, end_index) or (None, None, end).
    """
    n = len(lines)
    i = start
    while i < n and not lines[i].strip().startswith("Step"):
        i += 1
    if i >= n:
        return None, None, n
    i += 1
    temps, densities = [], []
    while i < n and "Loop time" not in lines[i] and lines[i].strip():
        parts = lines[i].strip().split()
        if len(parts) >= 4:
            try:
                t = float(parts[1])
                d = float(parts[2])
                temps.append(t)
                densities.append(d)
            except ValueError:
                pass
        i += 1
    # skip blank/CJK lines
    while i < n and not lines[i].strip():
        i += 1
    return np.array(temps), np.array(densities), i


def parse_log(filename):
    """Parse LAMMPS log, return cooling data (temps decreasing from ~500K)."""
    with open(filename) as f:
        text = f.read()

    lines = text.split("\n")
    # Find the cooling section: the last thermo block where temp decreases
    # Scan backwards through thermo blocks
    all_blocks = []
    i = 0
    while i < len(lines):
        # skip empty/CJK
        if not lines[i].strip() or not lines[i][0].isdigit() and "Step" not in lines[i]:
            i += 1
            continue
        temps, densities, i_next = parse_cooling_block(lines, i)
        if temps is not None and len(temps) > 2:
            all_blocks.append((temps, densities))
        i = i_next

    if not all_blocks:
        print(f"[ERROR] No thermo data found in {filename}")
        return None, None

    # Pick the block where temperature spans the widest range (the cooling run)
    best = max(all_blocks, key=lambda b: b[0].max() - b[0].min())
    temps, densities = best

    # Only keep the cooling portion (temp decreasing from high T)
    # Find the highest temperature point
    peak = np.argmax(temps)
    temps = temps[peak:]
    densities = densities[peak:]

    return np.array(temps), np.array(densities)


def dedup(temps, densities):
    """Average points at the same temperature."""
    uniq = {}
    for t, d in zip(temps, densities):
        key = round(t, 1)
        uniq.setdefault(key, []).append(d)
    t_u = np.array(sorted(uniq.keys()))
    d_u = np.array([np.mean(uniq[k]) for k in sorted(uniq.keys())])
    return t_u, d_u


def fit_segments(x, y, break_idx):
    """Two-segment linear regression, return (seg1, seg2, Tg)."""
    s1, i1, r1, _, _ = stats.linregress(x[:break_idx], y[:break_idx])
    s2, i2, r2, _, _ = stats.linregress(x[break_idx:], y[break_idx:])
    tg = (i2 - i1) / (s1 - s2)
    return (s1, i1, r1**2), (s2, i2, r2**2), tg


def find_best_split(x, y, min_pts=8):
    """Find split index that maximizes sum of R² (each segment ≥ min_pts)."""
    n = len(x)
    if n < 2 * min_pts:
        return n // 2
    best_r2 = -np.inf
    best_idx = n // 2
    for i in range(min_pts, n - min_pts + 1):
        try:
            (_, _, r2_1), (_, _, r2_2), _ = fit_segments(x, y, i)
            s = r2_1 + r2_2
            if s > best_r2:
                best_r2 = s
                best_idx = i
        except Exception:
            continue
    return best_idx


def analyze_one(log_path, label=None):
    """Analyze a single log file, return (Tg, temps, densities, seg1, seg2, best_idx)."""
    name = label or os.path.basename(os.path.dirname(log_path))
    print(f"\n{'='*50}")
    print(f"Analyzing: {name}")
    print(f"  Log: {log_path}")

    temps, densities = parse_log(log_path)
    if temps is None or len(temps) < 5:
        print(f"  [SKIP] Not enough data")
        return None

    print(f"  Raw data points: {len(temps)}")

    temps, densities = dedup(temps, densities)
    print(f"  After dedup: {len(temps)} points")
    print(f"  T range: {temps.min():.0f} – {temps.max():.0f} K")

    best_idx = find_best_split(temps, densities)
    seg1, seg2, tg = fit_segments(temps, densities, best_idx)

    (s1, i1, r2_1), (s2, i2, r2_2) = seg1, seg2
    print(f"\n  === Results ===")
    print(f"  Split: {best_idx} rubber + {len(temps)-best_idx} glass")
    print(f"  Rubber: ρ = {s1:.6f}T + {i1:.4f}  (R²={r2_1:.4f})")
    print(f"  Glass:  ρ = {s2:.6f}T + {i2:.4f}  (R²={r2_2:.4f})")
    print(f"  Tg = {tg:.1f} K")

    return {"name": name, "tg": tg, "temps": temps, "densities": densities,
            "seg1": seg1, "seg2": seg2, "best_idx": best_idx, "spec_vol": 1.0 / densities}


def plot_comparison(results):
    """Plot all compositions on the same figure."""
    n = len(results)
    if n == 0:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    _ = fig  # suppress unused warning

    colors = plt.cm.viridis(np.linspace(0, 0.85, n))

    for res, c in zip(results, colors):
        t = res["temps"]
        d = res["densities"]
        sv = 1.0 / d  # specific volume
        tg = res["tg"]
        (s1, i1, _), (s2, i2, _) = res["seg1"], res["seg2"]
        idx = res["best_idx"]

        t_fit = np.linspace(t.min() - 5, t.max() + 5, 200)

        # Density plot
        ax1.scatter(t, d, c=[c], s=15, alpha=0.7, zorder=5)
        ax1.plot(t_fit, s1 * t_fit + i1, "-", c=c, lw=1,
                 label=f"{res['name']} (Tg={tg:.0f}K)")
        ax1.plot(t_fit, s2 * t_fit + i2, "-", c=c, lw=1)
        ax1.axvline(tg, c=c, ls=":", lw=0.8, alpha=0.5)

        # Specific volume plot
        (sv_s1, sv_i1, _), (sv_s2, sv_i2, _), sv_tg = fit_segments(t, sv, idx)
        ax2.scatter(t, sv, c=[c], s=15, alpha=0.7, zorder=5)
        ax2.plot(t_fit, sv_s1 * t_fit + sv_i1, "-", c=c, lw=1,
                 label=f"{res['name']} (Tg={sv_tg:.0f}K)")
        ax2.plot(t_fit, sv_s2 * t_fit + sv_i2, "-", c=c, lw=1)
        ax2.axvline(sv_tg, c=c, ls=":", lw=0.8, alpha=0.5)

    ax1.set(xlabel="Temperature (K)", ylabel="Density (g/cm³)",
            title="Density vs Temperature")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set(xlabel="Temperature (K)", ylabel="Specific Volume (cm³/g)",
            title="Specific Volume vs Temperature (classic)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    figpath = os.path.join(os.path.dirname(__file__) or ".", "Tg_comparison.png")
    plt.savefig(figpath, dpi=150)
    print(f"\nComparison plot saved: {figpath}")
    plt.close()

    # Print summary table
    print(f"\n{'='*50}")
    print(f"{'Composition':<15} {'Tg (density)':<15} {'Tg (spec.vol)':<15}")
    print(f"{'-'*45}")
    for res in results:
        t = res["temps"]
        sv = 1.0 / res["densities"]
        (_, _, _), (_, _, _), tg_v = fit_segments(t, sv, res["best_idx"])
        print(f"{res['name']:<15} {res['tg']:<15.1f} {tg_v:<15.1f}")


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else "log.lammps"

    if not os.path.exists(log_path):
        print(f"Log not found: {log_path}")
        print("Usage: python analysis.py [log.lammps]")
        sys.exit(1)

    name = os.path.splitext(os.path.basename(log_path))[0]
    res = analyze_one(log_path, label=name)
    if res:
        plot_comparison([res])


if __name__ == "__main__":
    main()
