"""Scalabilite forte et faible du CG multi-GPU : medianes, tableaux, figures.

Usage : python analysis/analyse.py results/scaling_N*.csv
"""
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
(ROOT / "figures").mkdir(exist_ok=True)
MODES = ["aware-overlap", "aware-sync", "staged-sync"]
COL = {"aware-overlap": "#1b6e5a", "aware-sync": "#d9822b", "staged-sync": "#b5452c"}

rows = [r for f in sys.argv[1:] for r in csv.DictReader(open(f))]
g = defaultdict(list)
for r in rows:
    g[(r["kind"], r["mode"], int(r["ranks"]))].append(r)
med = {k: {f: st.median(float(r[f]) for r in v) for f in ["time_s", "time_per_iter_us", "t_halo_s", "t_allreduce_s"]}
       | {"iters": v[0]["iters"], "relres": v[0]["true_relres"], "n": len(v)} for k, v in g.items()}
P = sorted({k[2] for k in med})

md = ["## Scalabilite forte : Poisson 3D 512^3 (134 M d'inconnues), 1140 iterations\n",
      "| GPU | " + " | ".join(f"{m} : temps (s), acceleration, efficacite" for m in MODES) + " |",
      "|---" * (1 + len(MODES)) + "|"]
for p in P:
    cells = []
    for m in MODES:
        t1, t = med[("strong", m, 1)]["time_s"], med[("strong", m, p)]["time_s"]
        cells.append(f"{t:.3f} ; {t1 / t:.1f}x ; {100 * t1 / t / p:.0f} %")
    md.append(f"| {p} | " + " | ".join(cells) + " |")
md += ["", "## Scalabilite faible : 256 x 256 x 256P (16,8 M d'inconnues par GPU), temps par iteration\n",
       "| GPU | " + " | ".join(f"{m} : us/iter, efficacite" for m in MODES) + " |", "|---" * (1 + len(MODES)) + "|"]
for p in P:
    cells = []
    for m in MODES:
        t1, t = med[("weak", m, 1)]["time_per_iter_us"], med[("weak", m, p)]["time_per_iter_us"]
        cells.append(f"{t:.0f} ; {100 * t1 / t:.0f} %")
    md.append(f"| {p} | " + " | ".join(cells) + " |")
md += ["", "## Ou part le temps (scalabilite forte, rang le plus lent) : attente du halo et Allreduce (s)\n",
       "| GPU | " + " | ".join(f"{m} : halo / allreduce / total" for m in MODES) + " |", "|---" * (1 + len(MODES)) + "|"]
for p in P:
    md.append(f"| {p} | " + " | ".join(
        f"{med[('strong', m, p)]['t_halo_s']:.3f} / {med[('strong', m, p)]['t_allreduce_s']:.3f} / {med[('strong', m, p)]['time_s']:.3f}"
        for m in MODES) + " |")
md.append("")
(ROOT / "results" / "summary.md").write_text("\n".join(md))
print("\n".join(md))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
for m in MODES:
    t1 = med[("strong", m, 1)]["time_s"]
    ax[0].plot(P, [t1 / med[("strong", m, p)]["time_s"] for p in P], "o-", color=COL[m], label=m)
    w1 = med[("weak", m, 1)]["time_per_iter_us"]
    ax[1].plot(P, [100 * w1 / med[("weak", m, p)]["time_per_iter_us"] for p in P], "o-", color=COL[m], label=m)
ax[0].plot(P, P, "k--", lw=0.8, label="ideal")
ax[0].set_xscale("log", base=2); ax[0].set_yscale("log", base=2)
ax[0].set_xticks(P); ax[0].set_xticklabels(P); ax[0].set_yticks(P); ax[0].set_yticklabels(P)
ax[0].set_title("Forte : 512^3, acceleration"); ax[0].set_xlabel("GPU (4 par noeud)")
ax[1].set_xscale("log", base=2); ax[1].set_xticks(P); ax[1].set_xticklabels(P)
ax[1].set_ylim(0, 105); ax[1].set_title("Faible : 16,8 M / GPU, efficacite (%)"); ax[1].set_xlabel("GPU (4 par noeud)")
for a in ax:
    a.grid(alpha=0.3); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(ROOT / "figures" / "scaling.png", dpi=140)
