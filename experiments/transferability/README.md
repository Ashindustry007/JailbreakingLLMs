# Transferability

Replays the jailbreak prompts that broke a source model against each downstream
target, with no PAIR loop (one query + one judge call per prompt). Tests whether
PAIR's *semantic* prompts port across models.

**Finding** (`results/transfer_matrix.csv`): Qwen3-235B's prompts transfer
strongly to weaker targets (94% to GPT-3.5, 82% to Gemma-3n) but 0% to Gemini,
which also resists direct attack. Transfer is high among similarly-trained
models and low against a robust target.

## Run

```bash
export TOGETHER_API_KEY=...   GEMINI_API_KEY=...
python experiments/transferability/transfer_paper.py
```
