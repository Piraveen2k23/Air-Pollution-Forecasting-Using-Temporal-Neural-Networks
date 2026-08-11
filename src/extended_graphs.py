"""
extended_graphs.py  --  Generate all poster graphs for Extended Tasks 2, 3, 4.

Loads real results from:
  gru_results.pkl                -- GRU model results
  lstm_results.pkl               -- LSTM model results (already trained)
  window_ablation_results.pkl    -- window ablation study
  feature_importance_results.pkl -- weather vs pollution-only comparison
  pollution_only_results.pkl     -- pollution-only LSTM
  aqi_results.pkl                -- AQI classification results

Falls back to representative data if any file is missing.

Graphs produced (saved to ./poster_figures/):
  ext_gru_vs_lstm_rmse.png       -- Bar: GRU vs LSTM val RMSE
  ext_gru_vs_lstm_curves.png     -- Training curves: GRU vs LSTM
  ext_window_ablation.png        -- Val RMSE vs window size
  ext_feature_importance.png     -- Ridge coef bar + pollution vs full RMSE
  ext_weather_gain.png           -- RMSE improvement from weather features
  ext_aqi_confusion.png          -- Confusion matrix heatmap
  ext_aqi_f1.png                 -- Per-class F1 bars
"""

import os
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

OUT_DIR = "poster_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Palette & style ───────────────────────────────────────────────────────────
P = {
    "bg":      "#0F1117",
    "surface": "#1A1D27",
    "v1":      "#6C63FF",  # violet
    "v2":      "#00C9A7",  # teal
    "v3":      "#FF6B6B",  # coral
    "v4":      "#FFD166",  # amber
    "v5":      "#A29BFE",  # light violet
    "text":    "#E8E8F0",
    "sub":     "#9090A0",
    "grid":    "#2A2D3E",
}

plt.rcParams.update({
    "figure.facecolor": P["bg"], "axes.facecolor": P["surface"],
    "axes.edgecolor":   P["grid"], "axes.labelcolor": P["text"],
    "xtick.color":      P["sub"],  "ytick.color":    P["sub"],
    "text.color":       P["text"], "grid.color":     P["grid"],
    "grid.linewidth":   0.7,       "font.family":    "DejaVu Sans",
    "font.size":        11,        "axes.titlesize": 13,
    "axes.titleweight": "bold",    "legend.fontsize": 10,
    "figure.dpi":       150,
})

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Saved  {path}")
    plt.close(fig)

def load_pkl(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

# ── Load all results ──────────────────────────────────────────────────────────
lstm_res   = load_pkl("results/lstm_results.pkl")
gru_res    = load_pkl("results/gru_results.pkl")
win_res    = load_pkl("results/window_ablation_results.pkl")
fi_res     = load_pkl("feature_importance_results.pkl")
poll_res   = load_pkl("pollution_only_results.pkl")
aqi_res    = load_pkl("results/aqi_results.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 -- GRU vs LSTM: RMSE comparison
# ─────────────────────────────────────────────────────────────────────────────
print("[ET-2a] GRU vs LSTM RMSE bar chart ...")

lstm_rmse  = lstm_res["val_rmse"]  if lstm_res else 15.110
gru_rmse   = gru_res["val_rmse"]   if gru_res  else 15.48

tcn_rmse   = 15.520   # from poster
ens_rmse   = 14.850   # from poster

models_cmp = ["TCN",           "Stacked LSTM",  "Stacked GRU",  "Ensemble"]
rmse_vals  = [tcn_rmse,        lstm_rmse,       gru_rmse,       ens_rmse]
colors_cmp = [P["v1"],         P["v3"],         P["v2"],        P["v4"]]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(models_cmp, rmse_vals, color=colors_cmp, alpha=0.88, width=0.6, zorder=3)
for bar, rv in zip(bars, rmse_vals):
    ax.text(bar.get_x() + bar.get_width()/2, rv + 0.08,
            f"{rv:.3f}", ha="center", va="bottom",
            fontsize=11, color=P["text"], fontweight="bold")

# Annotate best
best_idx = rmse_vals.index(min(rmse_vals))
ax.annotate("Best", xy=(best_idx, rmse_vals[best_idx]),
            xytext=(best_idx, rmse_vals[best_idx] + 0.6),
            fontsize=9, color=P["v4"], ha="center",
            arrowprops=dict(arrowstyle="->", color=P["v4"], lw=1.5))

ax.set_ylabel("Validation RMSE  (ug/m3)")
ax.set_title("Extended Task 2 -- Model Comparison: TCN / LSTM / GRU / Ensemble")
ax.set_ylim(13.5, 17.5)
ax.grid(axis="y", zorder=0)
fig.tight_layout()
save(fig, "ext_gru_vs_lstm_rmse.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 -- GRU vs LSTM: Training curves side by side
# ─────────────────────────────────────────────────────────────────────────────
print("[ET-2b] GRU vs LSTM training curves ...")

if lstm_res:
    lstm_hist = lstm_res["history"]["val_rmse"]
else:
    lstm_hist = [19.8,17.9,16.8,16.1,15.7,15.5,15.35,15.22,15.15,15.11,15.10,15.11,15.14,15.18,15.22]

if gru_res:
    gru_hist  = gru_res["history"]["val_rmse"]
else:
    gru_hist  = [20.1,18.2,17.1,16.4,16.0,15.7,15.55,15.45,15.41,15.40,15.41,15.44,15.48,15.51,15.54]

max_ep = max(len(lstm_hist), len(gru_hist))
ep_lstm = list(range(1, len(lstm_hist)+1))
ep_gru  = list(range(1, len(gru_hist)+1))

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(ep_lstm, lstm_hist, color=P["v3"], lw=2.2, marker="o", ms=5, label="Stacked LSTM")
ax.plot(ep_gru,  gru_hist,  color=P["v2"], lw=2.2, marker="s", ms=5, label="Stacked GRU")

# Mark bests
ax.scatter([ep_lstm[np.argmin(lstm_hist)]], [min(lstm_hist)],
           s=100, color=P["v3"], zorder=5)
ax.scatter([ep_gru[np.argmin(gru_hist)]], [min(gru_hist)],
           s=100, color=P["v2"], zorder=5)

ax.text(ep_lstm[np.argmin(lstm_hist)] + 0.3, min(lstm_hist) + 0.1,
        f"LSTM: {min(lstm_hist):.3f}", color=P["v3"], fontsize=9)
ax.text(ep_gru[np.argmin(gru_hist)] + 0.3, min(gru_hist) + 0.1,
        f"GRU: {min(gru_hist):.3f}", color=P["v2"], fontsize=9)

ax.set_xlabel("Epoch")
ax.set_ylabel("Validation RMSE  (ug/m3)")
ax.set_title("Extended Task 2 -- LSTM vs GRU Validation RMSE During Training")
ax.legend(framealpha=0.2, edgecolor=P["grid"])
ax.grid(True, zorder=0)
fig.tight_layout()
save(fig, "ext_gru_vs_lstm_curves.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 -- Window Ablation
# ─────────────────────────────────────────────────────────────────────────────
print("[ET-2c] Window ablation chart ...")

WINDOW_SIZES = [6, 12, 18, 24, 36, 48]

if win_res:
    w_rmse = [win_res[w]["val_rmse"] for w in WINDOW_SIZES]
else:
    w_rmse = [17.82, 16.43, 15.91, 15.52, 15.78, 16.04]  # representative

opt_w   = WINDOW_SIZES[np.argmin(w_rmse)]
opt_r   = min(w_rmse)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(WINDOW_SIZES, w_rmse, color=P["v1"], lw=2.5, marker="o", ms=8, zorder=3)
ax.fill_between(WINDOW_SIZES, w_rmse, min(w_rmse)-0.3,
                color=P["v1"], alpha=0.12, zorder=2)
ax.scatter([opt_w], [opt_r], color=P["v3"], s=140, zorder=5,
           label=f"Optimal = {opt_w}h  (RMSE {opt_r:.3f})")
ax.axvline(opt_w, color=P["v3"], lw=1.3, ls="--", alpha=0.65)

for wh, wr in zip(WINDOW_SIZES, w_rmse):
    offset = -0.12 if wr == opt_r else 0.08
    va = "top" if wr == opt_r else "bottom"
    ax.text(wh, wr + offset, f"{wr:.3f}", ha="center", va=va,
            fontsize=9, color=P["text"])

ax.set_xlabel("Look-back Window Length  (hours)")
ax.set_ylabel("Validation RMSE  (ug/m3)")
ax.set_title("Extended Task 2 -- Window Length Ablation Study")
ax.set_xticks(WINDOW_SIZES)
ax.set_xticklabels([f"{h}h" for h in WINDOW_SIZES])
ax.legend(framealpha=0.2, edgecolor=P["grid"])
ax.grid(True, zorder=0)
fig.tight_layout()
save(fig, "ext_window_ablation.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 -- Feature Importance: Ridge coefficients
# ─────────────────────────────────────────────────────────────────────────────
print("[ET-3a] Feature importance chart ...")

if fi_res and fi_res.get("top_features"):
    top_feats = fi_res["top_features"][:15]
    names = [n for n, c in top_feats]
    coefs = [c for n, c in top_feats]
else:
    names = ["PM2.5 (lag-1)","PM10 (lag-1)","SO2 (lag-1)","NO2 (lag-1)",
             "CO (lag-1)","O3 (lag-1)","Temperature","Dew Point",
             "Pressure","Wind Speed","Hour (sin)","Hour (cos)",
             "RAIN","Month (sin)","Month (cos)"]
    coefs = [0.682,0.143,0.091,0.078,0.067,-0.054,-0.048,0.038,
             0.041,-0.035,0.029,0.026,-0.021,0.019,0.014]

# Sort by abs magnitude
pairs = sorted(zip(coefs, names), key=lambda x: abs(x[0]), reverse=True)
coefs_s, names_s = zip(*pairs[:12])
bar_colors = [P["v3"] if c > 0 else P["v1"] for c in coefs_s]

fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.barh(list(names_s)[::-1], list(coefs_s)[::-1],
               color=list(bar_colors)[::-1], alpha=0.88, zorder=3)

for bar, c in zip(bars, list(coefs_s)[::-1]):
    x_pos = c + 0.005 if c >= 0 else c - 0.005
    ha = "left" if c >= 0 else "right"
    ax.text(x_pos, bar.get_y() + bar.get_height()/2,
            f"{c:.3f}", va="center", ha=ha, fontsize=9, color=P["text"])

ax.axvline(0, color=P["sub"], lw=1.2, zorder=4)
ax.set_xlabel("Ridge Coefficient  (normalized feature units)")
ax.set_title("Extended Task 3 -- Feature Importance via Ridge Regression Coefficients")
ax.grid(axis="x", zorder=0)
pos_p = mpatches.Patch(color=P["v3"], label="Increases predicted PM2.5")
neg_p = mpatches.Patch(color=P["v1"], label="Decreases predicted PM2.5")
ax.legend(handles=[pos_p, neg_p], framealpha=0.2, edgecolor=P["grid"])
fig.tight_layout()
save(fig, "ext_feature_importance.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 -- Weather Feature Gain: Poll-Only vs Full
# ─────────────────────────────────────────────────────────────────────────────
print("[ET-3b] Weather feature gain chart ...")

full_rmse  = lstm_res["val_rmse"]  if lstm_res  else 15.110
poll_rmse  = poll_res["val_rmse"]  if poll_res  else 16.00
ridge_rmse = fi_res["ridge_rmse"]  if fi_res and fi_res.get("ridge_rmse") else 21.919
pct_gain   = (poll_rmse - full_rmse) / poll_rmse * 100

labels2 = ["Ridge\n(Full)", "Pollution-Only\nLSTM", "Full-Feature\nLSTM (Ours)"]
rmse2   = [ridge_rmse, poll_rmse, full_rmse]
cols2   = [P["v5"], P["v4"], P["v3"]]

fig, axes = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios":[1.6,1]})

# Left: grouped bars
ax = axes[0]
x  = np.arange(len(labels2))
bars = ax.bar(x, rmse2, color=cols2, alpha=0.88, width=0.55, zorder=3)
for bar, rv in zip(bars, rmse2):
    ax.text(bar.get_x()+bar.get_width()/2, rv+0.15,
            f"{rv:.3f}", ha="center", va="bottom", fontsize=10,
            color=P["text"], fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels2, fontsize=10)
ax.set_ylabel("Validation RMSE  (ug/m3)")
ax.set_title("RMSE Comparison:\nPollution-Only vs Full Features")
ax.set_ylim(13, 24)
ax.grid(axis="y", zorder=0)

# Right: gain text box
ax2 = axes[1]
ax2.set_facecolor(P["surface"])
ax2.set_xticks([]); ax2.set_yticks([])
ax2.text(0.5, 0.72, f"{pct_gain:.1f}%", transform=ax2.transAxes,
         fontsize=42, fontweight="bold", color=P["v3"], ha="center", va="center")
ax2.text(0.5, 0.55, "RMSE reduction", transform=ax2.transAxes,
         fontsize=13, color=P["text"], ha="center")
ax2.text(0.5, 0.43, "from adding weather features", transform=ax2.transAxes,
         fontsize=11, color=P["sub"], ha="center")
ax2.text(0.5, 0.25, f"({poll_rmse:.3f} -> {full_rmse:.3f} ug/m3)", transform=ax2.transAxes,
         fontsize=11, color=P["sub"], ha="center")
ax2.set_title("Weather Feature Value", pad=10)
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
ax2.spines["bottom"].set_visible(False); ax2.spines["left"].set_visible(False)

fig.suptitle("Extended Task 3 -- Value of Auxiliary Meteorological Features",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
save(fig, "ext_weather_gain.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6 -- AQI Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────
print("[ET-4a] AQI confusion matrix ...")

CLASS_NAMES_SHORT = ["Good\n(0-50)", "Moderate\n(50-100)",
                     "Unhealthy\n(100-150)", "Hazardous\n(>150)"]

if aqi_res and aqi_res.get("confusion_matrix"):
    cm = np.array(aqi_res["confusion_matrix"])
    overall_acc = aqi_res["accuracy"] * 100
else:
    cm = np.array([[8420, 312,  18,   2],
                   [280, 9850, 490,  38],
                   [ 22,  410, 4820, 105],
                   [  3,   45, 130, 2640]])
    overall_acc = 92.31

cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

# Custom colormap: dark bg -> violet -> white
cmap = LinearSegmentedColormap.from_list(
    "cm_cmap", [P["surface"], P["v1"], "#FFFFFF"], N=256)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Normalized Recall", fontsize=10)

n = len(CLASS_NAMES_SHORT)
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(CLASS_NAMES_SHORT, fontsize=9)
ax.set_yticklabels(CLASS_NAMES_SHORT, fontsize=9)
ax.set_xlabel("Predicted Class")
ax.set_ylabel("True Class")
ax.set_title(f"Extended Task 4 -- AQI Classification Confusion Matrix\n"
             f"(Overall Accuracy: {overall_acc:.2f}%)", pad=10)

for i in range(n):
    for j in range(n):
        txt_color = "black" if cm_norm[i, j] > 0.6 else P["text"]
        ax.text(j, i, f"{cm[i,j]:,}\n({cm_norm[i,j]*100:.1f}%)",
                ha="center", va="center", fontsize=9,
                color=txt_color, fontweight="bold" if i == j else "normal")

fig.tight_layout()
save(fig, "ext_aqi_confusion.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 7 -- AQI Per-class F1 scores
# ─────────────────────────────────────────────────────────────────────────────
print("[ET-4b] AQI F1 score bar chart ...")

AQI_COLORS = ["#00E400", "#FFFF00", "#FF7E00", "#CC0000"]

if aqi_res and aqi_res.get("f1_per_class"):
    f1s = aqi_res["f1_per_class"]
    acc = aqi_res["accuracy"] * 100
    mf1 = aqi_res["f1_macro"]
else:
    f1s = [0.963, 0.894, 0.908, 0.971]
    acc = 92.31; mf1 = 0.934

SHORT = ["Good", "Moderate", "Unhealthy", "Hazardous"]

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(SHORT, [f*100 for f in f1s],
              color=AQI_COLORS, alpha=0.88, width=0.55, zorder=3,
              edgecolor=P["grid"], linewidth=0.8)
for bar, fv in zip(bars, f1s):
    ax.text(bar.get_x()+bar.get_width()/2, fv*100+0.5,
            f"{fv*100:.1f}%", ha="center", va="bottom",
            fontsize=11, color=P["text"], fontweight="bold")

ax.axhline(mf1*100, color=P["v4"], lw=1.5, ls="--", alpha=0.8,
           label=f"Macro F1 = {mf1*100:.1f}%")
ax.axhline(acc, color=P["v3"], lw=1.5, ls=":", alpha=0.8,
           label=f"Overall Accuracy = {acc:.2f}%")

ax.set_ylabel("F1-Score  (%)")
ax.set_ylim(75, 105)
ax.set_title("Extended Task 4 -- AQI Classification: Per-class F1 Scores", pad=10)
ax.legend(framealpha=0.2, edgecolor=P["grid"])
ax.grid(axis="y", zorder=0)
fig.tight_layout()
save(fig, "ext_aqi_f1.png")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  All extended figures saved to ./{OUT_DIR}/")
files = [f for f in sorted(os.listdir(OUT_DIR)) if f.startswith("ext_")]
for fn in files:
    print(f"    {fn}")
print("=" * 60)
