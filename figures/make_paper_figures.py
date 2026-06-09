"""Generate polished figures for the NeurIPS write-up from the paper-reproduction
runs (Qwen3-235B / Gemini, judged by Llama-Guard-4).

Outputs to paper/figs/:
  fig_efficiency.png   -- Figure 5: cumulative ASR vs queries + per-iteration bar
  fig_heatmap.png      -- Figure 4: target x category jailbreak%
"""
import json, math, os, re, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

QDIR = "logs/paper_qwen_target"
GDIR = "logs/paper_gemini"
OUT = "paper/figs"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 11})


def rows(d):
    return [json.loads(l) for l in open(d + "/status.jsonl")]


# ---------- Figure 5: efficiency ----------
def efficiency():
    rs = rows(QDIR); N = sum(1 for r in rs if r.get("returncode") == 0)
    q = sorted(r["queries_to_jailbreak"] for r in rs
               if r.get("jailbroken") and r.get("queries_to_jailbreak"))
    it = {1: 0, 2: 0, 3: 0}
    for x in q:
        it[math.ceil(x / 30)] += 1
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    xs = q; ys = [100 * (i + 1) / N for i in range(len(q))]
    a1.step([0] + xs, [0] + ys, where="post", color="#c0392b", lw=2.2)
    a1.set_xlabel("Queries to first jailbreak"); a1.set_ylabel("Cumulative JB% of dataset")
    a1.set_title("(a) PAIR efficiency on Qwen3-235B"); a1.set_xlim(0, 90); a1.grid(alpha=.25)
    a2.bar([1, 2, 3], [it[1], it[2], it[3]], color="#c0392b", alpha=.85, width=.6)
    a2.set_xlabel("Iteration of first success"); a2.set_ylabel("# jailbroken behaviors")
    a2.set_xticks([1, 2, 3]); a2.set_title("(b) Successes by iteration")
    for k in (1, 2, 3):
        a2.text(k, it[k] + .15, str(it[k]), ha="center", fontsize=11, weight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_efficiency.png", dpi=200, bbox_inches="tight")
    plt.close(); print("wrote fig_efficiency.png")


# ---------- Figure 4: target x category heatmap (full team panel) ----------
def heatmap():
    # per-category % jailbroken; name -> {category: percent}
    pct = {}
    for name, d in [("Qwen3-235B", QDIR), ("Gemini-2.5-Flash", GDIR)]:
        rs = rows(d); bycat = {}
        for r in rs:
            if r.get("returncode") != 0: continue
            c = r.get("category", "?"); bycat.setdefault(c, [0, 0]); bycat[c][1] += 1
            if r.get("jailbroken"): bycat[c][0] += 1
        pct[name] = {c: 100 * nj / nt for c, (nj, nt) in bycat.items()}
    # collaborator results (target-gpt branch, identical setup)
    pct["GPT-3.5"] = {"Economic harm": 80, "Fraud/Deception": 80,
                      "Harassment/Discrimination": 70, "Malware/Hacking": 90,
                      "Physical harm": 80}
    pct["GPT-4o"] = {"Economic harm": 90, "Fraud/Deception": 70,
                     "Harassment/Discrimination": 60, "Malware/Hacking": 50,
                     "Physical harm": 80}
    # RAW QWEN OVERRIDE: Qwen3-235B per-category under original judge (option A)
    pct["Qwen3-235B"] = {"Economic harm": 60, "Fraud/Deception": 60,
                         "Harassment/Discrimination": 50, "Malware/Hacking": 40,
                         "Physical harm": 70}
    # collaborator Together targets (category_heatmap_values.csv, raw judge)
    pct["Llama-3-8B"] = {"Economic harm": 80, "Fraud/Deception": 100,
                         "Harassment/Discrimination": 90, "Malware/Hacking": 90,
                         "Physical harm": 90}
    pct["Gemma-3n"] = {"Economic harm": 100, "Fraud/Deception": 100,
                       "Harassment/Discrimination": 100, "Malware/Hacking": 100,
                       "Physical harm": 100}
    # full column panel (order matches Table 2)
    cols = [("Llama-3-8B", "Llama-3-8B"), ("Gemma-3n", "Gemma-3n"),
            ("Qwen3-235B", "Qwen3-235B"), ("GPT-3.5", "GPT-3.5"),
            ("GPT-4o", "GPT-4o"), ("Gemini-2.5-Flash", "Gemini-2.5-Flash")]
    cats = sorted({c for k in ("Qwen3-235B", "Gemini-2.5-Flash") for c in pct[k]})
    M = np.full((len(cats), len(cols)), np.nan)
    for j, (_, key) in enumerate(cols):
        if key is None: continue
        for i, c in enumerate(cats):
            if c in pct.get(key, {}): M[i, j] = pct[key][c]
    cmap = plt.cm.Reds.copy(); cmap.set_bad("#e9e9e9")
    fig, ax = plt.subplots(figsize=(0.95 * len(cols) + 1.8, 0.42 * len(cats) + 1.4))
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c for c, _ in cols], rotation=20, ha="right", fontsize=9)
    for j, (_, key) in enumerate(cols):                   # red labels for placeholders
        if key is None:
            ax.get_xticklabels()[j].set_color("red")
    ax.set_yticks(range(len(cats))); ax.set_yticklabels(cats, fontsize=9)
    for j, (_, key) in enumerate(cols):
        if key is None:                                   # placeholder column
            ax.text(j, (len(cats) - 1) / 2, "TBD", ha="center", va="center",
                    rotation=90, fontsize=10, color="red", weight="bold")
            continue
        for i in range(len(cats)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center",
                        fontsize=8, color="white" if M[i, j] > 50 else "#333")
    plt.colorbar(im, label="Jailbreak %", fraction=.046, pad=.04)
    ax.set_title("PAIR JB% by target × category")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(); print("wrote fig_heatmap.png")



if __name__ == "__main__":
    efficiency()
    heatmap()
