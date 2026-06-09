# Defenses: SmoothLLM and perplexity filter

Takes the prompts that jailbroke the undefended target and replays them through
two defenses applied to the same model:

* **SmoothLLM** — N perturbed copies (random character swaps), majority vote.
* **Perplexity filter** — block prompts whose GPT-2 perplexity exceeds the max
  among the benign JailbreakBench goals.

**Finding** (`results/defended_performance.csv`, Qwen3-235B, original judge):

| Defense | JB% | Drop |
|---------|-----|------|
| None | 56% | - |
| SmoothLLM | 14% | 75% |
| Perplexity filter | 56% | 0% |

SmoothLLM cuts ASR sharply; the perplexity filter does nothing, because PAIR's
prompts are fluent natural language, not high-perplexity token noise.

## Run

```bash
export TOGETHER_API_KEY=...
python experiments/defense/defend_eval.py
```
