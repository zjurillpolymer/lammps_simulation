"""
Analyze Li+ Brownian motion trajectory in silica glass.
Reads trajectory_li.lammpstrj and computes MSD, visualizes path.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_lammps_trajectory(filename):
    """Read LAMMPS custom dump trajectory."""
    frames = []
    with open(filename) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("ITEM: TIMESTEP"):
            timestep = int(lines[i + 1].strip())
            # skip to box bounds
            while i < len(lines) and not lines[i].startswith("ITEM: ATOMS"):
                i += 1
            # if ITEM: ATOMS line - in LAMMPS custom dump, the header
            # has id type x y z in this order
            atoms = []
            i += 1
            while i < len(lines) and not lines[i].startswith("ITEM:"):
                parts = lines[i].strip().split()
                if len(parts) >= 5:
                    atoms.append({
                        'id': int(parts[0]),
                        'type': int(parts[1]),
                        'x': float(parts[2]),
                        'y': float(parts[3]),
                        'z': float(parts[4]),
                    })
                i += 1
            frames.append({'step': timestep, 'atoms': atoms})
        else:
            i += 1
    return frames

def compute_msd_file(filename, dt_ps=0.01):
    """Compute MSD of all atoms in trajectory file."""
    frames = read_lammps_trajectory(filename)
    if len(frames) < 2:
        print("Not enough frames for MSD")
        return

    positions = []
    times = []
    for fr in frames:
        pos = np.array([[a['x'], a['y'], a['z']] for a in fr['atoms']])
        if len(pos) > 0:
            positions.append(pos.mean(axis=0))  # average over Li atoms (should be 1)
            times.append(fr['step'] * dt_ps)

    positions = np.array(positions)
    times = np.array(times)

    # Single-reference (if only 1 atom) the MSD is just squared displacement from t0
    msd = np.sum((positions - positions[0])**2, axis=1)

    # Fit D from slope of MSD vs time (linear regime)
    half = len(times) // 2
    coeffs = np.polyfit(times[half:], msd[half:], deg=1)
    D = coeffs[0] / 6.0  # MSD = 6Dt in 3D

    print(f"Total frames: {len(frames)}")
    print(f"Total time: {times[-1]:.2f} ps")
    print(f"Final MSD: {msd[-1]:.2f} A^2")
    print(f"Diffusion coefficient: {D:.4f} A^2/ps = {D*1e4:.4f} cm^2/s")

    # ---- Plot MSD ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(times, msd, 'b-', linewidth=1.5)
    ax1.set_xlabel("Time (ps)")
    ax1.set_ylabel("MSD (A^2)")
    ax1.set_title("Li+ Mean Squared Displacement in SiO2 at 1500K")
    ax1.grid(True, alpha=0.3)

    # ---- Plot 3D trajectory ----
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    # Color from start (blue) to end (red)
    colors = plt.cm.plasma(np.linspace(0, 1, len(positions)))
    for i in range(len(positions) - 1):
        ax2.plot(positions[i:i+2, 0], positions[i:i+2, 1], positions[i:i+2, 2],
                 color=colors[i], alpha=0.7, linewidth=1)
    ax2.scatter(positions[0,0], positions[0,1], positions[0,2],
                color='green', s=80, marker='o', label='Start')
    ax2.scatter(positions[-1,0], positions[-1,1], positions[-1,2],
                color='red', s=80, marker='x', label='End')
    ax2.set_xlabel("X (A)")
    ax2.set_ylabel("Y (A)")
    ax2.set_zlabel("Z (A)")
    ax2.set_title(f"Li+ Trajectory (D = {D:.3f} A^2/ps)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("li_brownian_motion.png", dpi=150)
    print("Saved plot to li_brownian_motion.png")
    plt.show()

if __name__ == "__main__":
    import sys
    fname = sys.argv[1] if len(sys.argv) > 1 else "trajectory_li.lammpstrj"
    compute_msd_file(fname, dt_ps=0.001)  # 1 fs step in metal units
