# Figures

Plotting code for the report figures. Reads the PAIR run logs and result
summaries.

* `make_paper_figures.py` — `efficiency()` (cumulative jailbreaks vs queries +
  per-iteration counts) and `heatmap()` (target x category jailbreak %).
* `analyze_results.py` — tabular summaries of a run directory.

```bash
python figures/make_paper_figures.py
```
