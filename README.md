# PAIR Reproduction — Jailbreaking GPT-3.5 & GPT-4o

A reproduction of **PAIR** (Prompt Automatic Iterative Refinement; [Chao et al.,
2023](https://arxiv.org/abs/2310.08419)) on the
[JailbreakBench](https://arxiv.org/abs/2404.01318) behavior set, run with
currently-callable models. The PAIR algorithm is **unchanged** — only the
paper's retired model endpoints are swapped for available ones.

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

**Table 2 — Attack success: PAIR vs JBC baseline** (first 50 behaviors, same
Llama-Guard-4 judge). PAIR is our iterative attack; JBC = 9 static
jailbreakchat templates applied once each (no iteration).

| Method | Metric | GPT-3.5-Turbo | GPT-4o |
|--------|--------|:-------------:|:------:|
| **PAIR** (ours) | Jailbreak % (ASR) | **80%** | **70%** |
|  | Queries / success | 28.3 | 17.9 |
| **JBC** (9 templates) | Avg. Jailbreak % | 7.8% | **0%** |
|  | ASR (any template) | 66% | **0%** |

**Per-category ASR (PAIR):**

| Category | GPT-3.5 | GPT-4o | Δ |
|----------|:-------:|:------:|:--:|
| Harassment / Discrimination | 70% | 60% | −10 |
| Malware / Hacking | 90% | 50% | **−40** |
| Physical harm | 80% | 80% | 0 |
| Economic harm | 80% | 90% | +10 |
| Fraud / Deception | 80% | 70% | −10 |
| **Overall** | **80%** | **70%** | −10 |

**Table 4 — Efficiency (PAIR):** all inference is remote API, so memory is N/A.

| Target | Running time / behavior | Cost / behavior |
|--------|:----------------------:|:---------------:|
| GPT-3.5-Turbo | ≈ 12 s | ≈ $0.038 |
| GPT-4o | ≈ 21 s | ≈ $0.078 |

**Table 3 — Transferability:** % of a source model's successful jailbreak
prompts that *also* jailbreak a downstream model when replayed once (no
iteration). Same-model cell omitted, per the paper.

| Method | Source (original target) | → GPT-3.5 | → GPT-4o |
|--------|--------------------------|:---------:|:--------:|
| PAIR (ours) | GPT-4o | **45.7%** (16/35) | — |

**Table 5 — Defended performance:** replay each model's jailbreak prompts on the
same model behind a defense; JB% over 50 behaviors (drop vs None in parens).

| Defense | GPT-3.5 | GPT-4o |
|---------|:-------:|:------:|
| None | 80.0% | 70.0% |
| SmoothLLM (N=10, q=10%, majority vote) | 44.0% (↓45.0%) | 30.0% (↓57.1%) |
| Perplexity filter (GPT-2) | 80.0% (↓0.0%) | 70.0% (↓0.0%) |

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
5. **PAIR ≫ static templates (JBC).** The 9 jailbreakchat templates reach only
   7.8% Avg on GPT-3.5 and **0% on GPT-4o**, vs PAIR's 80% / 70%. Adaptivity —
   refining against the target's refusals — is what defeats alignment; fixed
   public templates have aged out of modern models (GPT-4o blocks all 9).
6. **Jailbreaks transfer downstream.** 45.7% of the prompts crafted against
   GPT-4o still jailbreak GPT-3.5 unchanged — prompts tuned on a stronger model
   carry over to a weaker one without any re-optimization, echoing the paper's
   transferability finding (GPT-4 → GPT-3.5 = 65% there).
7. **Perplexity filtering is useless against PAIR; SmoothLLM helps.** The
   perplexity filter blocks **0** prompts (↓0.0%) — PAIR's jailbreaks are fluent
   natural language (GPT-2 perplexity 18–52), invisible to a filter built for
   GCG's gibberish. SmoothLLM (random char swaps + majority vote) cuts ASR
   45–57%. Semantic jailbreaks require a perturbation- or model-level defense.

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
