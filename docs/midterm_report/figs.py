import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
GREY, DARK, ACC = "#9e9e9e", "#505050", "#1f5fbf"
GPU_USD_H = 1.147208384   # g2-standard-16 on-demand, GCP accelerator-optimized pricing page
PRICE = {"llama": 0.18e-6, "qwen": 0.30e-6}   # Together AI list price, USD/token (in = out)

# results/cuad-25/runs.jsonl
R = {  # plan: {strategy: (time_s, switch_time_s, {model: (prompt, completion)}, f1)}
 "all_llama": {"naive": (4374.1, 0.0, {"llama": (443444, 71796)}, .6526),
               "batched": (557.5, 0.0, {"llama": (443444, 72249)}, .6453)},
 "all_qwen": {"naive": (4055.5, 0.0, {"qwen": (445356, 70143)}, .6633),
              "batched": (352.5, 0.0, {"qwen": (445356, 70008)}, .6653)},
 "mixed": {"naive": (5129.7, 973.6, {"llama": (223322, 36128), "qwen": (221078, 33110)}, .6539),
           "batched": (447.6, 3.1, {"llama": (223322, 36056), "qwen": (221078, 33187)}, .6565)},
 "moa": {"naive": (11526.2, 1949.6, {"llama": (426405, 90318), "qwen": (445356, 69496)}, .5552),
         "batched": (1255.8, 9.2, {"llama": (426467, 90430), "qwen": (445356, 70004)}, .5502)},
}
api = lambda u: sum((p + c) * PRICE[m] for m, (p, c) in u.items())
gpu = lambda t: t / 3600 * GPU_USD_H

# Figure 1: the same four plans ranked by API cost and by GPU cost (batched)
plans = sorted(R, key=lambda p: api(R[p]["batched"][2]))
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.0), sharey=True)
for ax, (title, val) in zip(axes, [("API cost (USD)", lambda p: api(R[p]["batched"][2])),
                                   ("GPU cost, batched (USD)", lambda p: gpu(R[p]["batched"][0]))]):
    vals = [val(p) for p in plans]
    best = min(range(len(vals)), key=vals.__getitem__)
    ax.barh(range(len(plans)), vals, color=[ACC if i == best else GREY for i in range(len(plans))], height=0.6)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.02, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_xlim(0, max(vals) * 1.25)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
axes[0].set_yticks(range(len(plans)), [f"{p} (F1 {R[p]['batched'][3]:.2f})" for p in plans])
axes[0].invert_yaxis()
fig.tight_layout()
fig.savefig("fig/ranking.pdf")

# Figure 2: naive vs batched GPU time per plan, switch time marked inside naive
order = ["all_qwen", "mixed", "all_llama", "moa"]
fig, ax = plt.subplots(figsize=(6.3, 2.5))
y = range(len(order))
for i, p in enumerate(order):
    tn, sw, _, _ = R[p]["naive"]; tb = R[p]["batched"][0]
    ax.barh(i - 0.18, tn / 60, height=0.34, color=GREY)
    if sw:
        ax.barh(i - 0.18, sw / 60, height=0.34, color=DARK)
    ax.barh(i + 0.18, tb / 60, height=0.34, color=ACC)
    ax.text(tn / 60 + 2, i - 0.18, f"{tn/60:.0f} min", va="center", fontsize=8)
    ax.text(tb / 60 + 2, i + 0.18, f"{tb/60:.1f} min  ({tn/tb:.1f}x faster)", va="center", fontsize=8)
ax.set_yticks(list(y), order)
ax.invert_yaxis()
ax.set_xlabel("GPU time (minutes)")
ax.set_xlim(0, 230)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=GREY, label="naive"), Patch(color=DARK, label="naive: model switches"),
                   Patch(color=ACC, label="batched")], frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
fig.tight_layout()
fig.savefig("fig/strategies.pdf")

for p in order:
    print(p, round(api(R[p]["batched"][2]), 4), round(gpu(R[p]["batched"][0]), 4), round(gpu(R[p]["naive"][0]), 4),
          round(R[p]["naive"][0] / R[p]["batched"][0], 2), round(R[p]["naive"][1] / R[p]["naive"][0] * 100, 1))
