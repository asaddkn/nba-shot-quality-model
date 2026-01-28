# make_sick_heatmap.py
# Shot-level overperformance heatmap: (made - xFG) averaged by location, then smoothed.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.patches import Circle, Rectangle, Arc

# =========================
# Config
# =========================
INPUT_CSV = "shai_scored_full_league_xfg.csv"   # needs: LOC_X, LOC_Y, SHOT_MADE_FLAG, xFG
OUT_PNG   = "shai_shot_level_overperformance_heatmap.png"

BINS = 100                 # 60–100
SIGMA = 6               # 2.5–6 (higher = smoother)
KEEP_DENSITY_PCTL = 10    # keep top (100-KEEP)% density: 60–85
VMAX = 0.05               # shot-level overperformance is larger scale than FG% diff (try 0.15–0.35)

X_RANGE = (-250, 250)
Y_RANGE = (-60, 300)

TITLE = "Shai Gilgeous-Alexander — Shot-level Overperformance (Made − xFG), 2025–26"

# =========================
# Court drawing
# =========================
def draw_half_court(ax):
    # Hoop + backboard
    ax.add_patch(Circle((0, 0), 7.5, fill=False, lw=2))
    ax.plot([-30, 30], [-7.5, -7.5], lw=2)

    # Paint
    ax.add_patch(Rectangle((-80, -47.5), 160, 190, fill=False, lw=2))
    ax.add_patch(Rectangle((-60, -47.5), 120, 190, fill=False, lw=2))

    # Free-throw circle (top half)
    ax.add_patch(Arc((0, 142.5), 120, 120, theta1=0, theta2=180, lw=2))

    # Restricted area
    ax.add_patch(Arc((0, 0), 80, 80, theta1=0, theta2=180, lw=2))

    # Three-point line: corners + arc
    ax.plot([-220, -220], [-47.5, 92.5], lw=2)
    ax.plot([ 220,  220], [-47.5, 92.5], lw=2)
    ax.add_patch(Arc((0, 0), 475, 475, theta1=22, theta2=158, lw=2))

    # Half-court arc (optional)
    ax.add_patch(Arc((0, 422.5), 120, 120, theta1=180, theta2=360, lw=2))


# =========================
# Load + validate
# =========================
df = pd.read_csv(INPUT_CSV)

required = {"LOC_X", "LOC_Y", "SHOT_MADE_FLAG", "xFG"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in {INPUT_CSV}: {missing}")

for c in ["LOC_X", "LOC_Y", "SHOT_MADE_FLAG", "xFG"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["LOC_X", "LOC_Y", "SHOT_MADE_FLAG", "xFG"]).copy()

print("Rows:", len(df))
print("Make rate:", df["SHOT_MADE_FLAG"].mean())
print("Mean xFG:", df["xFG"].mean())

# =========================
# Shot-level overperformance per shot
# =========================
# made=1 → + (1 - xFG), missed=0 → - xFG
df["over"] = df["SHOT_MADE_FLAG"] - df["xFG"]

# =========================
# Bin into a grid: average overperformance per location
# =========================
over_sum, xedges, yedges = np.histogram2d(
    df["LOC_X"], df["LOC_Y"],
    bins=BINS,
    range=[X_RANGE, Y_RANGE],
    weights=df["over"]
)

cnt, _, _ = np.histogram2d(
    df["LOC_X"], df["LOC_Y"],
    bins=[xedges, yedges]
)

over_avg = over_sum / (cnt + 1e-9)

# =========================
# Smooth AFTER averaging (this avoids the "whole court goes red" effect)
# =========================
over_s = gaussian_filter(over_avg, sigma=SIGMA)
cnt_s  = gaussian_filter(cnt, sigma=SIGMA)

# =========================
# Mask low-density areas (percentile-based, robust)
# =========================
density = cnt_s / (cnt_s.max() + 1e-9)
thr = np.percentile(density, KEEP_DENSITY_PCTL)  # keep top density
over_s[density < thr] = np.nan

# =========================
# Plot
# =========================
fig, ax = plt.subplots(figsize=(7, 7), dpi=140)
ax.set_facecolor("white")

im = ax.imshow(
    over_s.T,
    origin="lower",
    extent=[X_RANGE[0], X_RANGE[1], Y_RANGE[0], Y_RANGE[1]],
    cmap="RdBu_r",
    vmin=-VMAX,
    vmax=VMAX,
    interpolation="bicubic",
)

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Small horizontal colour scale inside the court
cax = inset_axes(ax,
                 width="30%",
                 height="3%",
                 loc='lower right',
                 borderpad=2)

cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
cbar.ax.tick_params(labelsize=8)
cbar.set_label("Made − xFG", fontsize=9)

draw_half_court(ax)

ax.set_xlim(*X_RANGE)
ax.set_ylim(*Y_RANGE)
ax.axis("off")


ax.set_title(TITLE, fontsize=12, pad=8)

plt.tight_layout()
plt.subplots_adjust(left=0, right=1, top=0.92, bottom=0)
plt.savefig(OUT_PNG, bbox_inches="tight", dpi=300)
plt.show()

print(f"Saved: {OUT_PNG}")
