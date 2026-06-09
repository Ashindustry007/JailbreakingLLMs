# Stress test: PAIR on 2026 SOTA models

Does the 2024 PAIR *methodology* still jailbreak today's frontier models? We run
PAIR through a gateway with a stronger attacker (deepseek) against a panel of
recent targets: Llama-4-Scout, GPT-OSS-120B, Gemma, Mistral-Large-3,
Claude-Sonnet-4.6.

**Finding.** Direct PAIR breaks only Llama-4-Scout (18-29%) and 0% elsewhere
(`results/sota_direct.csv`). But transfer tells a different story: replaying
Llama-4-Scout's jailbreak prompts breaks Mistral-Large-3 **57%** of the time
despite its 0% direct ASR (`results/sota_transfer.csv`). Frontier alignment
blunts direct adaptive attacks but not transferred ones.

## Run

```bash
export OPENAI_API_KEY=...   OPENAI_BASE_URL=<gateway>/v1
python run_pair.py --attack-model <attacker> --target-model <target> --limit 50
python experiments/stress_test/transfer_eval.py          # source->downstream transfer matrix
```
