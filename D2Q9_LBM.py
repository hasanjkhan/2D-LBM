import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

# Load the segmented image (black = pore, white = solid)
img = imread("no vugs S1 inverted.png") 

# Convert to binary domain matrix
# Fluid = 1, Solid = 0
domain = (img == 0).astype(int)
ny, nx = domain.shape
#!/usr/bin/env python3

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Physics parameters
NU = 0.10            # kinematic viscosity (lattice units)
FORCE_X = 1.0e-6     # constant body force in +x direction (left -> right)
FORCE_Y = 0.0

# LBM iteration controls
STEPS_MAX      = 1000
MIN_STEPS      = 1000
CHECK_EVERY    = 100
PRINT_EVERY    = 200
CONV_TOL       = 1e-6

# Binarization
PIXEL_SOLID_THRESHOLD = 128  # >= threshold -> fluid (white pore); < threshold -> solid (black)
THRESHOLD_AUTO = True        # apply Otsu if the frame looks non-binary

# Slicing
N_SLICES = 10  # includes first and last frames

def choose_root_dir():
    try:
        from tkinter import Tk, filedialog
        root = Tk(); root.withdraw()
        d = filedialog.askdirectory(title="Select TOP-LEVEL folder containing experiment subfolders")
        root.destroy()
        if not d:
            raise SystemExit("No folder selected. Exiting.")
        return Path(d)
    except Exception as e:
        raise SystemExit(f"Could not open folder dialog. "
                         f"Run as: python {Path(sys.argv[0]).name} <root_dir>\nDetails: {e}")

def find_cropped_avis(root: Path):
    out = []
    for p in root.rglob("*.avi"):
        if "cropped" in p.name.lower():
            out.append(p)
    return sorted(out)

def read_meta(meta_path: Path):
    """Returns dict with Q_ul_min, C_pct, experiment_duration_min, M_total_mg, rho_g_per_ml."""
    df = pd.read_csv(meta_path)
    if df.empty:
        raise ValueError(f"{meta_path}: empty")
    row = df.iloc[0]
    need = ["Q_ul_min","C_pct","experiment_duration_min","M_total_mg","rho_g_per_ml"]
    missing = [k for k in need if k not in df.columns]
    if missing:
        raise ValueError(f"{meta_path}: missing columns {missing}")
    return {k: float(row[k]) for k in need}

def to_binary(im_gray: np.ndarray) -> np.ndarray:
    """Return bool mask: True == Fluid (white)."""
    u = im_gray
    uniq = np.unique(u)
    if uniq.size <= 4 and np.isin(uniq, [0,1,255]).all():
        pores = (u >= 1)
    else:
        if THRESHOLD_AUTO:
            _, th = cv2.threshold(u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, th = cv2.threshold(u, PIXEL_SOLID_THRESHOLD, 255, cv2.THRESH_BINARY)
        pores = (th == 255)
    return pores

def grab_frame_gray(cap: cv2.VideoCapture, idx: int):
    """Seek to absolute frame index and return grayscale uint8 image."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        return None
    if frame.ndim == 3:
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        g = frame
    return g

# LBM core (D2Q9 with BGK & Guo force)
c = np.array([[0, 0],
              [1, 0], [0, 1], [-1, 0], [0, -1],
              [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.int32)
w = np.array([4/9,
              1/9, 1/9, 1/9, 1/9,
              1/36, 1/36, 1/36, 1/36], dtype=np.float64)
cs2 = 1.0/3.0
inv_cs2 = 1.0/cs2
tau = 0.5 + NU/cs2
omega = 1.0 / tau
opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)

def feq(rho, ux, uy):
    cu = (c[:, 0, None, None] * ux + c[:, 1, None, None] * uy)
    u2 = ux**2 + uy**2
    return (w[:, None, None] * rho *
            (1.0 + (cu * inv_cs2) + 0.5 * (cu**2) * (inv_cs2**2) - 0.5 * (u2 * inv_cs2)))

def forcing_term(ux, uy, Fx, Fy):
    e_dot_F = c[:, 0, None, None]*Fx + c[:, 1, None, None]*Fy
    e_dot_u = c[:, 0, None, None]*ux + c[:, 1, None, None]*uy
    u_dot_F = ux*Fx + uy*Fy
    return w[:, None, None] * (1.0 - 0.5*omega) * (
        (e_dot_F * inv_cs2) + (e_dot_u * e_dot_F) * (inv_cs2**2) - (u_dot_F) * inv_cs2
    )

def stream_periodic_xy(f):
    f_streamed = np.empty_like(f)
    for i, (cx, cy) in enumerate(c):
        f_streamed[i] = np.roll(np.roll(f[i], shift=cx, axis=1), shift=cy, axis=0)
    return f_streamed

def apply_bounce_back(f, solid_mask):
    for i in range(9):
        j = opp[i]
        fi = f[i]
        fj = f[j]
        tmp = fi[solid_mask].copy()
        fi[solid_mask] = fj[solid_mask]
        fj[solid_mask] = tmp

def lbm_permeability_from_mask(fluid_mask: np.ndarray, Fx=FORCE_X, Fy=FORCE_Y):
    ny, nx = fluid_mask.shape

    # enforce top/bottom walls as solid
    fluid_eff = fluid_mask.copy()
    fluid_eff[0, :]  = False
    fluid_eff[-1, :] = False

    solid = ~fluid_eff

    rho = np.ones((ny, nx), dtype=np.float64)
    ux  = np.zeros((ny, nx), dtype=np.float64)
    uy  = np.zeros((ny, nx), dtype=np.float64)
    f   = feq(rho, ux, uy)  # init at rest

    prev_q = None
    steps_used = STEPS_MAX

    for step in range(1, STEPS_MAX + 1):
        # macroscopic
        rho = np.sum(f, axis=0)
        ux  = (np.sum(f * c[:, 0, None, None], axis=0) + 0.5*Fx) / rho
        uy  = (np.sum(f * c[:, 1, None, None], axis=0) + 0.5*Fy) / rho
        ux[solid] = 0.0; uy[solid] = 0.0

        # collide + force
        feq_ = feq(rho, ux, uy)
        S    = forcing_term(ux, uy, Fx, Fy)
        f    = (1.0 - omega) * f + omega * feq_ + S

        # stream (periodic)
        f    = stream_periodic_xy(f)

        # bounce-back on solids (grains + top/bottom walls)
        apply_bounce_back(f, solid)

        if step % PRINT_EVERY == 0 or step == 1:
            q = float(np.mean(ux))  # Darcy flux
            print(f"  step {step:5d}  q={q:.3e}")

        if step % CHECK_EVERY == 0 and step >= MIN_STEPS:
            q = float(np.mean(ux))
            if prev_q is not None:
                rel = abs(q - prev_q) / max(abs(prev_q), 1e-30)
                if rel < CONV_TOL:
                    steps_used = step
                    break
            prev_q = q

    # final fields
    rho = np.sum(f, axis=0)
    ux  = (np.sum(f * c[:, 0, None, None], axis=0) + 0.5*Fx) / rho
    uy  = (np.sum(f * c[:, 1, None, None], axis=0) + 0.5*Fy) / rho
    ux[solid] = 0.0
    uy[solid] = 0.0

    q_mean = float(np.mean(ux))
    K = NU * q_mean / max(Fx, 1e-30)
    porosity = float(fluid_eff.mean())
    return K, q_mean, porosity, steps_used, ux, uy, rho, fluid_eff

# Injected Volume Ratio (IVR)
def ivr_percent_vector(n_points: int, meta: dict):
    Q_ul_min = meta["Q_ul_min"]
    C_pct    = meta["C_pct"]
    T_min    = meta["experiment_duration_min"]
    M_total  = meta["M_total_mg"]
    rho      = meta["rho_g_per_ml"]


    Q_mL_s = (Q_ul_min / 1000.0) / 60.0
    C_mg_mL = C_pct * 10.0 * rho

    t = np.linspace(0.0, T_min * 60.0, n_points)
    mg_rate = Q_mL_s * C_mg_mL   # mg/s
    mg_inj  = np.minimum(t * mg_rate, M_total)
    ivr = (mg_inj / M_total) * 100.0
    return ivr

# PNG helper
def save_velocity_png(speed, out_path: Path, vmin: float, vmax: float):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(speed, origin='upper', cmap='turbo', vmin=vmin, vmax=vmax)

    divider = make_axes_locatable(ax)
    # top arrow axis (flow →)
    top_ax = divider.append_axes("top", size="6%", pad=0.6)
    top_ax.set_xlim(0, 1); top_ax.set_ylim(0, 1); top_ax.axis('off')
    top_ax.annotate('', xy=(0.9, 0.5), xytext=(0.1, 0.5),
                    xycoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', linewidth=3, color='black'))
    top_ax.text(0.5, 0.9, "Flow direction", color='black',
                ha='center', va='bottom', fontsize=10, transform=top_ax.transAxes)

    # colorbar on right
    cax = divider.append_axes("right", size="4.5%", pad=0.25)
    fig.colorbar(im, cax=cax, label='|u| (lattice units)')

    ax.set_title('Velocity magnitude (converged)')
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)

def process_video(video_path: Path):
    print(f"\n=== {video_path} ===")
    meta_path = video_path.with_name("meta.csv")
    if not meta_path.exists():
        print(f"    Skipping: meta.csv not found next to video.")
        return

    try:
        meta = read_meta(meta_path)
    except Exception as e:
        print(f"    Skipping: bad meta.csv ({e})")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("    Could not open video.")
        return

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames < 2:
        print("    Not enough frames.")
        cap.release()
        return

    # indices of 10 evenly spaced frames including first/last
    idxs = np.unique(np.linspace(0, n_frames - 1, N_SLICES, dtype=int))
    if len(idxs) < N_SLICES:
        idxs = np.linspace(0, n_frames - 1, N_SLICES, dtype=int)

    Ks, qs, phis = [], [], []
    # for velocity deliverables
    ux_list, uy_list, rho_list, speed_list, fluid_list = [], [], [], [], []
    stats_rows = []
    prof_rows  = []

    out_dir = video_path.parent
    maps_dir = out_dir / "velocity_maps"
    maps_dir.mkdir(exist_ok=True)

    # IVR time axis
    ivr_percent = ivr_percent_vector(len(idxs), meta)
    ivr_ratio   = ivr_percent / 100.0

    frame_hw = None

    for slice_idx, fi in enumerate(idxs, start=1):
        g = grab_frame_gray(cap, int(fi))
        if g is None:
            print(f"Frame {fi} could not be read; skipping.")
            Ks.append(np.nan); qs.append(np.nan); phis.append(np.nan)
            continue

        if frame_hw is None:
            frame_hw = (g.shape[0], g.shape[1])
        pores = to_binary(g)

        print(f"  Slice {slice_idx:02d}/{len(idxs)} : frame {fi}  -- running LBM at {g.shape[1]}x{g.shape[0]}")
        K, q, phi, steps_used, ux, uy, rho, fluid_eff = lbm_permeability_from_mask(pores, Fx=FORCE_X, Fy=FORCE_Y)
        speed = np.sqrt(ux**2 + uy**2)

        # store
        Ks.append(K); qs.append(q); phis.append(phi)
        ux_list.append(ux.astype(np.float32))
        uy_list.append(uy.astype(np.float32))
        rho_list.append(rho.astype(np.float32))
        speed_list.append(speed.astype(np.float32))
        fluid_list.append(fluid_eff.astype(np.uint8))

        m = fluid_eff
        sp = speed[m]
        ux_f = ux[m]

        def pctl(a, p):
            return float(np.percentile(a, p)) if a.size else np.nan

        stats_rows.append(dict(
            slice_idx=slice_idx, frame_index=int(fi),
            IVR_percent=float(ivr_percent[slice_idx-1]),
            IVR_ratio=float(ivr_ratio[slice_idx-1]),
            q_mean=float(q), phi=float(phi),
            speed_mean=float(np.mean(sp)) if sp.size else np.nan,
            speed_median=float(np.median(sp)) if sp.size else np.nan,
            speed_p90=pctl(sp, 90), speed_p99=pctl(sp, 99), speed_max=float(np.max(sp)) if sp.size else np.nan,
            ux_mean=float(np.mean(ux_f)) if ux_f.size else np.nan,
            ux_median=float(np.median(ux_f)) if ux_f.size else np.nan,
            ux_p90=pctl(ux_f, 90), ux_p99=pctl(ux_f, 99),
            ux_min=float(np.min(ux_f)) if ux_f.size else np.nan,
            ux_max=float(np.max(ux_f)) if ux_f.size else np.nan
        ))

        # ---------- 1D profiles (means across the orthogonal direction, over fluid only) ----------
        m_float = fluid_eff.astype(np.float32)
        # along x (columns): mean over y where fluid
        cnt_x = m_float.sum(axis=0)                         # (W,)
        sum_ux_x = (ux * m_float).sum(axis=0)
        sum_sp_x = (speed * m_float).sum(axis=0)
        ux_x = np.divide(sum_ux_x, cnt_x, out=np.zeros_like(sum_ux_x), where=cnt_x > 0)
        sp_x = np.divide(sum_sp_x, cnt_x, out=np.zeros_like(sum_sp_x), where=cnt_x > 0)
        W = ux_x.size
        for x in range(W):
            prof_rows.append(dict(
                slice_idx=slice_idx, frame_index=int(fi),
                IVR_ratio=float(ivr_ratio[slice_idx-1]),
                axis='x', pos_idx=x, pos_norm_0_1=x/(W-1 if W>1 else 1),
                ux_mean=float(ux_x[x]), speed_mean=float(sp_x[x])
            ))
        # along y (rows): mean over x where fluid
        cnt_y = m_float.sum(axis=1)                         # (H,)
        sum_ux_y = (ux * m_float).sum(axis=1)
        sum_sp_y = (speed * m_float).sum(axis=1)
        ux_y = np.divide(sum_ux_y, cnt_y, out=np.zeros_like(sum_ux_y), where=cnt_y > 0)
        sp_y = np.divide(sum_sp_y, cnt_y, out=np.zeros_like(sum_sp_y), where=cnt_y > 0)
        H = ux_y.size
        for y in range(H):
            prof_rows.append(dict(
                slice_idx=slice_idx, frame_index=int(fi),
                IVR_ratio=float(ivr_ratio[slice_idx-1]),
                axis='y', pos_idx=y, pos_norm_0_1=y/(H-1 if H>1 else 1),
                ux_mean=float(ux_y[y]), speed_mean=float(sp_y[y])
            ))

    cap.release()

    Ks   = np.array(Ks, dtype=float)
    qs   = np.array(qs, dtype=float)
    phis = np.array(phis, dtype=float)

    # Normalize permeability
    Ki = Ks[0]
    K_over_Ki = Ks / Ki if np.isfinite(Ki) and abs(Ki) > 0 else np.full_like(Ks, np.nan)

    # Save normalized permeability output
    df_perm = pd.DataFrame({
        "frame_index": idxs,
        "IVR_percent": ivr_percent,
        "IVR_ratio": ivr_ratio,
        "K_lattice": Ks,
        "K_over_Ki": K_over_Ki,
        "q_mean": qs,
        "porosity_fraction": phis
    })
    csv_perm = out_dir / "permeability_profile.csv"
    df_perm.to_csv(csv_perm, index=False)
    print(f"   Saved: {csv_perm}")

    # plot K/Ki vs IVR
    png_path = out_dir / "K_over_Ki_vs_IVR.png"
    plt.figure(figsize=(8,5))
    plt.plot(ivr_ratio, K_over_Ki, linewidth=2.0)
    plt.xlabel("IVR (ratio)")
    plt.ylabel("Relative permeability (K/Ki)")
    plt.title("Change in permeability vs IVR")
    plt.grid(True, alpha=0.4)
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(png_path, dpi=180)
    plt.close()
    print(f"   Saved: {png_path}")

    # save velocity
    df_stats = pd.DataFrame(stats_rows)
    csv_stats = out_dir / "velocities_stats.csv"
    df_stats.to_csv(csv_stats, index=False)
    print(f"   Saved: {csv_stats}")

    # save velocity profiles
    df_prof = pd.DataFrame(prof_rows)
    csv_prof = out_dir / "velocities_profiles.csv"
    df_prof.to_csv(csv_prof, index=False)
    print(f"   Saved: {csv_prof}")

    all_vals = []
    for sp, fl in zip(speed_list, fluid_list):
        m = fl.astype(bool)
        if np.any(m):
            all_vals.append(sp[m])
    if all_vals:
        concat = np.concatenate(all_vals)
        robust_max = float(np.percentile(concat, 99.0))
        if robust_max <= 0 or not np.isfinite(robust_max):
            robust_max = float(np.max(concat)) if concat.size else 1.0
    else:
        robust_max = 1.0
    vmin, vmax = 0.0, robust_max

    with open(maps_dir / "vscale.txt", "w") as f:
        f.write(f"vmin={vmin}\nvmax={vmax}\npercentile=99\n")

    # ---------- Save velocity PNGs ----------
    for k, (sp, fi) in enumerate(zip(speed_list, idxs), start=1):
        out_png = maps_dir / f"vel_slice_{k:02d}_f{int(fi)}_ivr{ivr_ratio[k-1]:.2f}.png"
        save_velocity_png(sp, out_png, vmin=vmin, vmax=vmax)
    print(f"   Saved: {maps_dir}/vel_slice_*.png")

    # Flow field output
    ux_slices    = np.stack(ux_list, axis=0)   if ux_list else np.empty((0,))
    uy_slices    = np.stack(uy_list, axis=0)   if uy_list else np.empty((0,))
    speed_slices = np.stack(speed_list, axis=0)if speed_list else np.empty((0,))
    rho_slices   = np.stack(rho_list, axis=0)  if rho_list else np.empty((0,))
    fluid_mask   = fluid_list[0] if fluid_list else None  # same geometry across slices

    npz_path = out_dir / "velocity_fields.npz"
    np.savez_compressed(
        npz_path,
        ux_slices=ux_slices, uy_slices=uy_slices, speed_slices=speed_slices, rho_slices=rho_slices,
        fluid_mask=(fluid_mask if fluid_mask is not None else np.array([], dtype=np.uint8)),
        frame_indices=np.array(idxs, dtype=np.int32),
        ivr_ratio=np.array(ivr_ratio, dtype=np.float32),
        ivr_percent=np.array(ivr_percent, dtype=np.float32),
        vmin=np.float32(vmin), vmax=np.float32(vmax), robust_percentile=np.float32(99.0),
        nu=np.float32(NU), Fx=np.float32(FORCE_X), Fy=np.float32(FORCE_Y)
    )
    print(f"   Saved: {npz_path}")

def main():
    if len(sys.argv) > 1:
        root = Path(sys.argv[1]).expanduser().resolve()
    else:
        root = choose_root_dir()

    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    avis = find_cropped_avis(root)
    if not avis:
        print("No '*cropped*.avi' videos found under:", root)
        return

    print(f"Found {len(avis)} video(s).")
    for v in avis:
        process_video(v)

if __name__ == "__main__":
    main()
# LBM parameters
omega = 1.0
tau = 1 / omega
nu = (tau - 0.5) / 3
force = 1e-5  # Body force in y-direction

# D2Q9 Lattice parameters
c = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
              [1, 1], [-1, 1], [-1, -1], [1, -1]])
w = np.array([4/9] + [1/9]*4 + [1/36]*4)
opp = [0, 3, 4, 1, 2, 7, 8, 5, 6]

# Initialize fields
f = np.zeros((9, ny, nx))
rho = np.ones((ny, nx))
u = np.zeros((ny, nx, 2))

# Initialize f with equilibrium
for i in range(9):
    f[i] = w[i]

# Main LBM loop
n_steps = 10000
for step in range(n_steps):
    rho = np.sum(f, axis=0)
    u = np.tensordot(f, c, axes=(0, 0)) / rho[..., None]
    u[:, :, 1] += force / (2 * rho)

    for i in range(9):
        cu = 3 * (c[i, 0]*u[:, :, 0] + c[i, 1]*u[:, :, 1])
        feq = w[i] * rho * (1 + cu + 0.5*cu**2 - 1.5*(u[:, :, 0]**2 + u[:, :, 1]**2))
        f[i] += -omega * (f[i] - feq)

    for i in range(9):
        f[i][domain == 0] = f[opp[i]][domain == 0]  # bounce-back for solids

    for i in range(9):
        f[i] = np.roll(np.roll(f[i], c[i, 0], axis=1), c[i, 1], axis=0)

    u[:, :, 1] += force / (2 * rho)

    if step % 2000 == 0:
        print(f"Step {step}/{n_steps}")

# Step 5: Calculate permeability
avg_u_y = np.mean(u[:, :, 1][domain == 1])
permeability = avg_u_y / force * nu

print(f"\n Average Vertical Velocity (u_y): {avg_u_y:.3e}")
print(f" Estimated Permeability: {permeability:.3e} (LU²)")

# Visualize vertical velocity field
plt.imshow(u[:, :, 1] * domain, cmap='jet')
plt.colorbar(label="Velocity (u_y)")
plt.title("LBM Vertical Flow Velocity Field")
plt.axis('off')
plt.tight_layout()
plt.show()