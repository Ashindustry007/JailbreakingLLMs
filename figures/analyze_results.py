"""Harvest extra paper-style results from existing PAIR run logs (zero extra API).

Produces, from logs/*/status.jsonl (+ per-behavior .log files):
  - results_table2.csv  : per-target ASR + queries (Table 2 analog, open/closed tagged)
  - figure4_heatmap.csv : target x category jailbreak% (Figure 4 analog)
  - figure5_iters.csv   : distribution of which iteration produced the jailbreak,
                          and cumulative ASR vs queries (Figure 5 / efficiency analog)
  - table4_efficiency.csv: avg seconds/behavior, seconds/jailbreak, peak memory
  - RESULTS_HARVEST.md  : a readable summary of all of the above
PNG plots are written too if matplotlib is importable.
"""
from __future__ import annotations
import glob, json, math, os, re
from datetime import datetime
from collections import defaultdict

OUT = "results_harvest"
os.makedirs(OUT, exist_ok=True)

# Open/closed source tagging for the models in this study.
SOURCE = {
    "api-llama-4-scout": "open", "api-gemma-4-26b": "open", "api-gpt-oss-120b": "open",
    "moonshotai.kimi-k2.5": "open", "api-deepseek-v4-flash": "open",
    "claude-sonnet-4-6": "closed", "mistral.mistral-large-3-675b-instruct": "closed",
    "api-mistral-small-3.2-2506": "closed",
}
MEM_RE = re.compile(r"Memory after:\s*([\d.]+)\s*MB")

def iso(t):
    try: return datetime.fromisoformat(t)
    except Exception: return None

def load_dir(d):
    s = os.path.join(d, "status.jsonl")
    if not os.path.isfile(s): return None
    rows = []
    for line in open(s):
        try: rows.append(json.loads(line))
        except Exception: pass
    # keep only valid completed behaviors (rc==0, not budget/judge tainted)
    done = [r for r in rows if r.get("returncode") == 0
            and not r.get("budget_exceeded") and not r.get("judge_failed")]
    return rows, done

def peak_mem(rows):
    best = 0.0
    for r in rows:
        lf = r.get("log_file")
        if lf and os.path.isfile(lf):
            for m in MEM_RE.finditer(open(lf, errors="replace").read()):
                best = max(best, float(m.group(1)))
    return best

def main():
    # Unified 20-stream view: use the target_* runs (incl. target_llama4_20str).
    # The original 30-stream llama (logs/pair_ucsd_variant) is kept on disk as a
    # larger-sample reference but excluded here so Table 2 is one consistent setup.
    dirs = sorted(glob.glob("logs/target_*"))
    table2, heat, iters_rows, table4 = [], [], [], []
    cum = {}  # target -> list of queries_to_jailbreak
    for d in dirs:
        loaded = load_dir(d)
        if not loaded: continue
        rows, done = loaded
        if not done: continue
        tgt = done[-1]["target_model"]; jdg = done[-1]["judge_model"]
        ns = done[-1]["n_streams"]; ni = done[-1]["n_iterations"]
        jb = [r for r in done if r.get("jailbroken") is True]
        q = [r["queries_to_jailbreak"] for r in jb if r.get("queries_to_jailbreak")]
        asr = 100*len(jb)/len(done)
        table2.append(dict(target=tgt, source=SOURCE.get(tgt, "?"), judge=jdg,
                           budget=f"{ns}x{ni}", n=len(done), jailbroken=len(jb),
                           asr_pct=round(asr,1),
                           mean_queries=round(sum(q)/len(q),1) if q else None,
                           median_queries=(sorted(q)[len(q)//2] if q else None)))
        # Figure 4: per category JB%
        bycat = defaultdict(lambda: [0,0])
        for r in done:
            c = r.get("category","?"); bycat[c][1]+=1
            if r.get("jailbroken"): bycat[c][0]+=1
        for c,(nj,nt) in sorted(bycat.items()):
            heat.append(dict(target=tgt, category=c, jb=nj, total=nt,
                             jb_pct=round(100*nj/nt,1)))
        # Figure 5: which iteration produced each jailbreak (iter = ceil(q/n_streams))
        idist = defaultdict(int)
        for x in q:
            it = math.ceil(x/ns); idist[it]+=1
        for it in range(1, ni+1):
            iters_rows.append(dict(target=tgt, iteration=it, successes=idist.get(it,0)))
        cum[tgt] = sorted(q)
        # Table 4: efficiency (time per behavior / per jailbreak, peak mem)
        secs = [ (iso(r["finished_at"])-iso(r["started_at"])).total_seconds()
                 for r in done if iso(r.get("started_at")) and iso(r.get("finished_at")) ]
        jb_secs = [ (iso(r["finished_at"])-iso(r["started_at"])).total_seconds()
                    for r in jb if iso(r.get("started_at")) and iso(r.get("finished_at")) ]
        table4.append(dict(target=tgt,
                           avg_sec_per_behavior=round(sum(secs)/len(secs),1) if secs else None,
                           avg_sec_per_jailbreak=round(sum(jb_secs)/len(jb_secs),1) if jb_secs else None,
                           peak_mem_MB=round(peak_mem(rows),1)))

    def write_csv(path, rows, cols):
        with open(path,"w") as f:
            f.write(",".join(cols)+"\n")
            for r in rows:
                f.write(",".join(str(r.get(c,"")) for c in cols)+"\n")

    write_csv(f"{OUT}/results_table2.csv", table2,
              ["target","source","judge","budget","n","jailbroken","asr_pct","mean_queries","median_queries"])
    write_csv(f"{OUT}/figure4_heatmap.csv", heat, ["target","category","jb","total","jb_pct"])
    write_csv(f"{OUT}/figure5_iters.csv", iters_rows, ["target","iteration","successes"])
    write_csv(f"{OUT}/table4_efficiency.csv", table4,
              ["target","avg_sec_per_behavior","avg_sec_per_jailbreak","peak_mem_MB"])

    # Markdown summary
    with open(f"{OUT}/RESULTS_HARVEST.md","w") as f:
        f.write("# Harvested PAIR results (from existing run logs)\n\n")
        f.write("## Table 2 analog — direct attacks (ASR + queries)\n\n")
        f.write("| Target | Source | Judge | Budget | N | JB | ASR% | mean q | median q |\n|---|---|---|---|---|---|---|---|---|\n")
        for r in table2:
            f.write(f"| {r['target']} | {r['source']} | {r['judge']} | {r['budget']} | {r['n']} | {r['jailbroken']} | {r['asr_pct']} | {r['mean_queries']} | {r['median_queries']} |\n")
        f.write("\n## Table 4 analog — efficiency\n\n| Target | sec/behavior | sec/jailbreak | peak mem MB |\n|---|---|---|---|\n")
        for r in table4:
            f.write(f"| {r['target']} | {r['avg_sec_per_behavior']} | {r['avg_sec_per_jailbreak']} | {r['peak_mem_MB']} |\n")
        f.write("\n## Figure 5 analog — jailbreaks by iteration\n\n")
        f.write("CSV: figure5_iters.csv (successes per iteration, per target)\n")
        f.write("\n## Figure 4 analog — target x category JB%\n\nCSV: figure4_heatmap.csv\n")
    print(f"Wrote CSVs + RESULTS_HARVEST.md to {OUT}/")
    print(f"Targets analyzed: {len(table2)}")

    # Optional plots
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Figure 4 heatmap
        cats = sorted({h['category'] for h in heat}); tgts=[r['target'] for r in table2]
        if cats and tgts:
            import numpy as np
            M=np.full((len(cats),len(tgts)), np.nan)
            for h in heat:
                if h['target'] in tgts:
                    M[cats.index(h['category']), tgts.index(h['target'])]=h['jb_pct']
            fig,ax=plt.subplots(figsize=(1.6*len(tgts)+3, 0.5*len(cats)+2))
            im=ax.imshow(M, cmap="Reds", aspect="auto", vmin=0, vmax=100)
            ax.set_xticks(range(len(tgts))); ax.set_xticklabels([t.split('.')[-1][:14] for t in tgts], rotation=30, ha="right")
            ax.set_yticks(range(len(cats))); ax.set_yticklabels(cats)
            plt.colorbar(im,label="JB%"); ax.set_title("Figure 4 analog: PAIR JB% by target x category")
            plt.tight_layout(); plt.savefig(f"{OUT}/figure4_heatmap.png", dpi=130); plt.close()
        # Figure 5 cumulative ASR vs queries
        n_by_tgt = {r['target']: r['n'] for r in table2}
        fig,ax=plt.subplots(figsize=(7,4.5))
        for tgt,qs in cum.items():
            if not qs: continue
            N = n_by_tgt.get(tgt) or len(qs)
            xs = sorted(qs)
            ys = [100*(i+1)/N for i in range(len(xs))]
            ax.step(xs, ys, where="post", label=tgt.split('.')[-1][:16])
        ax.set_xlabel("Queries to jailbreak"); ax.set_ylabel("Cumulative JB% of dataset")
        ax.set_title("Figure 5 analog: efficiency (ASR vs queries)"); ax.legend(fontsize=7)
        plt.tight_layout(); plt.savefig(f"{OUT}/figure5_efficiency.png", dpi=130); plt.close()
        print("Wrote figure4_heatmap.png + figure5_efficiency.png")
    except Exception as e:
        print(f"(plots skipped: {e})")

if __name__ == "__main__":
    main()
