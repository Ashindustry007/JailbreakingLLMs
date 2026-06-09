# Judge-prompt sensitivity

The reported ASR is driven as much by the judge prompt as by the attack. The
default JailbreakBench Llama-Guard prompt asks only whether the conversation
touches a hazard category, so it flags **refusals** and **benign on-topic
("defanged") answers** as jailbreaks. Our refusal-aware prompt (root
`judges.py`) instead asks whether the model actually delivered harmful content,
and adds a deterministic refusal/length pre-filter.

`judge_compare.py` re-judges the same PAIR responses under both prompts.

**Finding** (`results/table8_judge_comparison.csv`): the gap is all false
positives and tracks how a model refuses.

| Target | Original | Refusal-aware |
|--------|----------|---------------|
| Qwen3-235B | 56% | 34% |
| Llama-3-8B | 90% | 30% |
| Gemma-3n | 100% | 80% |
| Gemini-2.5-Flash | 0% | 0% |

## Run

```bash
export TOGETHER_API_KEY=...
python experiments/judge_prompt/judge_compare.py --log-dir logs/paper_qwen_target
```
