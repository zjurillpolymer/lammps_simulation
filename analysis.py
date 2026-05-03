"""
从 log.lammps 中提取降温数据，计算玻璃化转变温度 Tg。
方法: 对密度-温度曲线做分段线性拟合，交点即为 Tg。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

LOG_FILE = "log.lammps"


def parse_log(filename):
    """提取降温阶段 (Step Temp Density Volume) 的 thermo 数据。"""
    temps, densities = [], []
    thermo_block = False
    cooling_section = False

    with open(filename) as f:
        lines = f.readlines()

    for line in lines:
        if "降温阶段" in line or "逐级降温" in line:
            cooling_section = True

        if cooling_section and line.strip().startswith("Step") and "Density" in line:
            thermo_block = True
            continue

        if thermo_block and "Loop time" in line:
            thermo_block = False
            continue

        if thermo_block and cooling_section:
            parts = line.strip().split()
            if len(parts) == 4:
                try:
                    temps.append(float(parts[1]))
                    densities.append(float(parts[2]))
                except ValueError:
                    continue

    return np.array(temps), np.array(densities)


def dedup(temps, densities):
    """相同温度取平均 (log 中上一 block 末态 = 下一 block 初态)。"""
    uniq = {}
    for t, d in zip(temps, densities):
        key = round(t, 1)
        uniq.setdefault(key, []).append(d)
    t_u = np.array(sorted(uniq.keys()))
    d_u = np.array([np.mean(uniq[k]) for k in sorted(uniq.keys())])
    return t_u, d_u


def fit_segments(x, y, break_idx):
    """两段线性回归，返回参数和交点 Tg。"""
    slope1, intercept1, r1, _, _ = stats.linregress(x[:break_idx], y[:break_idx])
    slope2, intercept2, r2, _, _ = stats.linregress(x[break_idx:], y[break_idx:])
    tg = (intercept2 - intercept1) / (slope1 - slope2)
    return (slope1, intercept1, r1**2), (slope2, intercept2, r2**2), tg


def find_best_split(x, y):
    """遍历分割点，选两段 R² 之和最大的。"""
    n = len(x)
    best_r2 = -np.inf
    best_idx = n // 2
    for i in range(2, n - 1):
        (_, _, r2_1), (_, _, r2_2), _ = fit_segments(x, y, i)
        s = r2_1 + r2_2
        if s > best_r2:
            best_r2 = s
            best_idx = i
    return best_idx


def plot_results(temps, densities, spec_vol, seg1, seg2, tg, tg_v, best_idx):
    """绘制密度-T 和比容-T 图。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    t_fit = np.linspace(temps.min() - 10, temps.max() + 10, 200)

    # 左图: 密度 vs 温度
    (s1, i1, r2_1) = seg1
    (s2, i2, r2_2) = seg2
    ax1.scatter(temps, densities, c="C0", zorder=5, label="Data")
    ax1.plot(t_fit, s1 * t_fit + i1, "--", c="C1", label=f"Rubber (R²={r2_1:.3f})")
    ax1.plot(t_fit, s2 * t_fit + i2, "--", c="C2", label=f"Glass (R²={r2_2:.3f})")
    ax1.axvline(tg, c="C3", ls=":", lw=1.5, label=f"Tg = {tg:.0f} K")
    ax1.set(xlabel="Temperature (K)", ylabel="Density (g/cm³)")
    ax1.legend(title=f"Split: {best_idx}+{len(temps)-best_idx}")
    ax1.set_title("Density vs Temperature")

    # 右图: 比容 vs 温度 (经典方法)
    (s1, i1, r2_1), (s2, i2, r2_2), tg_v = fit_segments(temps, spec_vol, best_idx)
    ax2.scatter(temps, spec_vol, c="C0", zorder=5, label="Data")
    ax2.plot(t_fit, s1 * t_fit + i1, "--", c="C1", label=f"Rubber (R²={r2_1:.3f})")
    ax2.plot(t_fit, s2 * t_fit + i2, "--", c="C2", label=f"Glass (R²={r2_2:.3f})")
    ax2.axvline(tg_v, c="C3", ls=":", lw=1.5, label=f"Tg = {tg_v:.0f} K")
    ax2.set(xlabel="Temperature (K)", ylabel="Specific Volume (cm³/g)")
    ax2.legend(title=f"Split: {best_idx}+{len(temps)-best_idx}")
    ax2.set_title("Specific Volume vs Temperature")

    plt.tight_layout()
    plt.savefig("Tg_analysis.png", dpi=150)
    print("图片已保存至 Tg_analysis.png")


def main():
    temps, densities = parse_log(LOG_FILE)
    print(f"提取到 {len(temps)} 个原始数据点")
    if len(temps) < 4:
        print("数据点不足，无法拟合。")
        return

    temps, densities = dedup(temps, densities)
    # 排除 500K 预平衡点（不是降温扫描的一部分）
    mask = temps < 450
    temps, densities = temps[mask], densities[mask]
    spec_vol = 1.0 / densities

    print(f"去重后 {len(temps)} 个点 (已排除 500K 预平衡点):")
    for t, d in zip(temps, densities):
        print(f"  T = {t:8.2f} K,  ρ = {d:.6f} g/cm³")

    best_idx = find_best_split(temps, densities)
    n = len(temps)
    print(f"\n最佳分割: 前 {best_idx} 点 (高温) + {n - best_idx} 点 (低温)")

    seg1, seg2, tg = fit_segments(temps, densities, best_idx)
    (s1, i1, r2_1), (s2, i2, r2_2) = seg1, seg2
    print(f"\n=== 拟合结果 (密度-温度) ===")
    print(f"高温段: ρ = {s1:.6f} T + {i1:.4f}  (R² = {r2_1:.4f})")
    print(f"低温段: ρ = {s2:.6f} T + {i2:.4f}  (R² = {r2_2:.4f})")
    print(f"Tg = {tg:.1f} K")

    # 比容法
    (_, _, _), (_, _, _), tg_v = fit_segments(temps, spec_vol, best_idx)
    print(f"Tg (比容法) = {tg_v:.1f} K")

    plot_results(temps, densities, spec_vol, seg1, seg2, tg, tg_v, best_idx)


if __name__ == "__main__":
    main()
