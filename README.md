# PAIR Reproduction — Jailbreaking GPT-3.5 & GPT-4o

A reproduction of **PAIR** (Prompt Automatic Iterative Refinement; [Chao et al.,
2023](https://arxiv.org/abs/2310.08419)) on the
[JailbreakBench](https://arxiv.org/abs/2404.01318) behavior set, run with
currently-callable models. The PAIR algorithm is **unchanged** — only the
paper's retired model endpoints are swapped for available ones.

Full analysis and per-table breakdown: [`RESULT.md`](RESULT.md).

## Core Configuration

| Role | Model | Provider |
|------|-------|----------|
| **Attacker** | Qwen2.5-7B-Instruct-Turbo | Together.ai (serverless) |
| **Target** | `gpt-3.5-turbo-1106` / `gpt-4o-2024-11-20` | OpenAI |
| **Judge** | Llama-Guard-4-12B | Together.ai (serverless) |

| PAIR hyperparameter | Value |
|---------------------|-------|
| Parallel streams `N` | 30 |
| Iterations / depth `K` | 3 |
| Max queries / behavior | `N × K` = 90 |
| Early stop | on first jailbreak (judge score = 10) |
| Dataset | first 50 of 100 JBB-Behaviors (5/10 harm categories) |

## Core Results

**Overall:**

| Target | ASR | Mean queries / success | Cost / behavior |
|--------|:---:|:----------------------:|:---------------:|
| GPT-3.5-Turbo | 40/50 = **80%** | 28.3 | ≈ $0.038 |
| GPT-4o | 35/50 = **70%** | 17.9 | ≈ $0.078 |

**Per-category ASR:**

| Category | GPT-3.5 | GPT-4o | Δ |
|----------|:-------:|:------:|:--:|
| Harassment / Discrimination | 70% | 60% | −10 |
| Malware / Hacking | 90% | 50% | **−40** |
| Physical harm | 80% | 80% | 0 |
| Economic harm | 80% | 90% | +10 |
| Fraud / Deception | 80% | 70% | −10 |
| **Overall** | **80%** | **70%** | −10 |

## Brief Conclusions

1. **PAIR reproduces.** Black-box semantic-reframing jailbreaks succeed on both
   GPT-3.5 (80%) and GPT-4o (70%) well within the 90-query budget — the paper's
   core claim holds on modern targets.
2. **No robustness gain can be claimed from ASR alone.** The 80% → 70% drop is
   not statistically significant (two-proportion *p* ≈ 0.25 at n = 50).
3. **Key methodological finding:** a topic-classifier judge (Llama-Guard) counts
   *defanged* benign answers as jailbreaks, inflating ASR — and the inflation
   **grows with alignment** (all 5 of GPT-4o's one-query "successes" are false
   positives). The metric therefore *understates* the very safety improvement it
   is meant to measure; a stricter judge would lower both ASRs.
4. **Clearest real signal:** GPT-4o is specifically hardened on Malware/Hacking
   (90% → 50%), refusing or returning defensive content far more than GPT-3.5.

> **Caveats.** Judge **and** attacker differ from the paper (Qwen for Mixtral,
> Llama-Guard-4 for GPT-4 / Llama-Guard-1), so these numbers are **not** directly
> comparable to the paper's. Coverage is 50/100 behaviors (5/10 categories); ASR
> carries wide CIs at this sample size.

## Reproduce

Set API keys (`OPENAI_API_KEY`, `TOGETHER_API_KEY`) in a `.env` file, then:

```bash
set -a && source .env && set +a
WANDB_MODE=offline python run_dataset.py --num-behaviors 50 --target-model gpt-3.5-turbo-1106 --output results/gpt35_head50.csv
WANDB_MODE=offline python run_dataset.py --num-behaviors 50 --target-model gpt-4o-2024-11-20 --output results/gpt4o_head50.csv
```

`--resume` is on by default (re-running continues from a stalled/crashed run);
add `--sampling stratified` to cover all 10 categories. A single behavior can
still be run via `main.py` (see its arguments). For a new GPT target, first run
`python patch_jailbreakbench.py` to register it in the installed JailbreakBench
package.

## Citation

```bibtex
@misc{chao2023jailbreaking,
      title={Jailbreaking Black Box Large Language Models in Twenty Queries},
      author={Patrick Chao and Alexander Robey and Edgar Dobriban and Hamed Hassani and George J. Pappas and Eric Wong},
      year={2023},
      eprint={2310.08419},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

Original code released under [MIT License](LICENSE).
