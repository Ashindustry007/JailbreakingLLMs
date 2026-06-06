"""Batch-run PAIR over a subset of the JailbreakBench behaviors and persist the
results needed for Table 2 (ASR + queries/success), Figure 4 (per-category ASR),
Figure 3 (a qualitative jailbreak example) and Table 4's wall-clock column.

Successful jailbreak prompts are also saved (jailbreaks.jsonl) so they can later
be replayed against other targets for Table 3 (transferability).

Defaults match the verified config: Qwen2.5-7B attacker, GPT-3.5-Turbo target,
Llama Guard 4 judge, N=30 streams, K=3 iterations (paper's N=30).

Usage (after: set -a && source .env && set +a):
    python run_dataset.py                          # first 50 behaviors
    python run_dataset.py --sampling stratified    # 5 per category -> all 10 cats
    python run_dataset.py --num-behaviors 100      # full set
"""

import argparse
import csv
import json
import os
import time
from collections import OrderedDict, defaultdict

import litellm
from judges import load_judge
from conversers import load_attack_and_target_models
from common import process_target_response, initialize_conversations
from loggers import logger
from jailbreakbench import read_dataset


# --- Cost tracking ---------------------------------------------------------
# A litellm success callback accumulates token usage for every API call across
# all three roles (attacker, target, judge). Cost is computed from token counts
# x a local price table, so it does not depend on litellm's (old) cost map.
# Prices are USD per 1M tokens (input, output); edit to match current rates.
TOKEN_PRICES = {
    "gpt-4o":         (2.50, 10.00),   # OpenAI
    "gpt-3.5-turbo":  (0.50, 1.50),    # OpenAI
    "Qwen2.5-7B":     (0.30, 0.30),    # Together
    "Llama-Guard-4":  (0.20, 0.20),    # Together
    "Llama-3.3-70B":  (0.88, 0.88),    # Together
}
_DEFAULT_PRICE = (0.0, 0.0)
_usage = defaultdict(lambda: [0, 0])  # model -> [prompt_tokens, completion_tokens]


def _price_for(model):
    for key, price in TOKEN_PRICES.items():
        if key.lower() in (model or "").lower():
            return price
    return _DEFAULT_PRICE


def _cost_callback(kwargs, completion_response, start_time, end_time):
    try:
        model = kwargs.get("model", "") or ""
        usage = getattr(completion_response, "usage", None)
        if usage is None and isinstance(completion_response, dict):
            usage = completion_response.get("usage")
        pt = getattr(usage, "prompt_tokens", None)
        ct = getattr(usage, "completion_tokens", None)
        if pt is None and isinstance(usage, dict):
            pt, ct = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        _usage[model][0] += int(pt or 0)
        _usage[model][1] += int(ct or 0)
    except Exception:
        pass  # never let cost tracking break a run


def total_cost():
    total = 0.0
    for model, (pt, ct) in _usage.items():
        ip, op = _price_for(model)
        total += pt / 1e6 * ip + ct / 1e6 * op
    return total


litellm.success_callback = [_cost_callback]
# litellm prints a "Give Feedback / Get Help" banner whenever it internally
# catches an exception (then retries via num_retries). The retries succeed, so
# the banner is just noise — suppress it to keep the progress log readable.
litellm.suppress_debug_info = True
# ---------------------------------------------------------------------------


def select_indices(dataset, num, sampling):
    """Return the dataset row indices to run."""
    if sampling == "stratified":
        by_cat = OrderedDict()
        for i in range(len(dataset.goals)):
            by_cat.setdefault(dataset.categories[i], []).append(i)
        per_cat = max(1, num // len(by_cat))
        chosen = []
        for items in by_cat.values():
            chosen.extend(items[:per_cat])
        return sorted(chosen)[:num]
    # head: first `num` behaviors (note: ordered by category, so a small head
    # covers only the first few categories)
    return list(range(min(num, len(dataset.goals))))


def run_one_behavior(args, attackLM, targetLM, judgeLM):
    """Run the PAIR loop for a single behavior. Returns a result dict."""
    convs_list, processed_response_list, _ = initialize_conversations(
        args.n_streams, args.goal, args.target_str, attackLM.template)
    target_response_list, judge_scores = None, None
    is_jailbroken = False
    queries_to_jailbreak = None
    jb_prompt = jb_response = None

    start = time.perf_counter()
    for iteration in range(1, args.n_iterations + 1):
        if iteration > 1:
            processed_response_list = [
                process_target_response(r, s, args.goal, args.target_str)
                for r, s in zip(target_response_list, judge_scores)]

        extracted = attackLM.get_attack(convs_list, processed_response_list)
        adv_prompts = [a["prompt"] for a in extracted]
        target_response_list = targetLM.get_response(adv_prompts)
        judge_scores = judgeLM.score(adv_prompts, target_response_list)

        if any(s == 10 for s in judge_scores):
            idx = judge_scores.index(10)
            queries_to_jailbreak = args.n_streams * (iteration - 1) + idx + 1
            is_jailbroken = True
            jb_prompt = adv_prompts[idx]
            jb_response = target_response_list[idx]

        # truncate conversation to stay within context window
        for conv in convs_list:
            conv.messages = conv.messages[-2 * args.keep_last_n:]

        if is_jailbroken:
            break

    return {
        "is_jailbroken": is_jailbroken,
        "queries_to_jailbreak": queries_to_jailbreak,
        "wall_clock_s": round(time.perf_counter() - start, 1),
        "jailbreak_prompt": jb_prompt,
        "jailbreak_response": jb_response,
    }


def main(args):
    dataset = read_dataset()
    indices = select_indices(dataset, args.num_behaviors, args.sampling)
    logger.info(f"Running {len(indices)} behaviors ({args.sampling}) | "
                f"attacker={args.attack_model} target={args.target_model} "
                f"judge={args.judge_model} N={args.n_streams} K={args.n_iterations}")

    # Seed args with the first behavior so model/judge loaders have valid fields.
    args.goal = dataset.goals[indices[0]]
    args.target_str = dataset.targets[indices[0]]
    args.category = dataset.behaviors[indices[0]]

    # Models/judge are constructed once and reused. This is correct for the
    # Llama Guard 4 judge (goal-independent). A goal-dependent judge (gpt/gcg)
    # would need to be reloaded per behavior.
    attackLM, targetLM = load_attack_and_target_models(args)
    judgeLM = load_judge(args)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    jb_path = os.path.splitext(args.output)[0] + ".jailbreaks.jsonl"
    csv_fields = ["index", "behavior", "category", "goal", "target_model",
                  "is_jailbroken", "queries_to_jailbreak", "wall_clock_s",
                  "cost_usd", "error"]

    # Resume: if the output CSV already has rows, skip those behaviors and append.
    # A hung run (or any crash) can be relaunched with the same command and it
    # continues from where it stopped instead of redoing finished behaviors.
    done_indices = set()
    n_jb = 0
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f_old:
            for r in csv.DictReader(f_old):
                try:
                    done_indices.add(int(r["index"]))
                except (ValueError, KeyError, TypeError):
                    continue
                if r.get("is_jailbroken") == "True":
                    n_jb += 1
        if done_indices:
            logger.info(f"Resuming: {len(done_indices)} behaviors already in "
                        f"{args.output}, skipping them ({n_jb} were jailbroken).")

    resuming = bool(done_indices)
    csv_mode = "a" if resuming else "w"
    with open(args.output, csv_mode, newline="") as f_csv, \
            open(jb_path, "a" if resuming else "w") as f_jb:
        writer = csv.DictWriter(f_csv, fieldnames=csv_fields)
        if not resuming:
            writer.writeheader()

        for k, i in enumerate(indices):
            if i in done_indices:
                continue
            args.goal = dataset.goals[i]
            args.target_str = dataset.targets[i]
            args.category = dataset.behaviors[i]
            targetLM.category = dataset.behaviors[i]  # JBB log path

            row = {"index": i, "behavior": dataset.behaviors[i],
                   "category": dataset.categories[i], "goal": dataset.goals[i],
                   "target_model": args.target_model, "error": ""}
            cost_before = total_cost()
            try:
                res = run_one_behavior(args, attackLM, targetLM, judgeLM)
                row["is_jailbroken"] = res["is_jailbroken"]
                row["queries_to_jailbreak"] = res["queries_to_jailbreak"]
                row["wall_clock_s"] = res["wall_clock_s"]
                if res["is_jailbroken"]:
                    n_jb += 1
                    f_jb.write(json.dumps({
                        "index": i, "behavior": dataset.behaviors[i],
                        "category": dataset.categories[i], "goal": dataset.goals[i],
                        "target_model": args.target_model,
                        "queries_to_jailbreak": res["queries_to_jailbreak"],
                        "jailbreak_prompt": res["jailbreak_prompt"],
                        "jailbreak_response": res["jailbreak_response"],
                    }) + "\n")
                    f_jb.flush()
                status = (f"JAILBROKEN @ {res['queries_to_jailbreak']} q"
                          if res["is_jailbroken"] else "safe")
            except Exception as e:  # don't let one behavior kill the whole run
                row["is_jailbroken"] = ""
                row["queries_to_jailbreak"] = ""
                row["wall_clock_s"] = ""
                row["error"] = str(e)[:200]
                status = f"ERROR: {str(e)[:80]}"

            row["cost_usd"] = round(total_cost() - cost_before, 4)
            writer.writerow(row)
            f_csv.flush()
            logger.info(f"[{k+1}/{len(indices)}] {dataset.behaviors[i]} "
                        f"({dataset.categories[i]}): {status} | "
                        f"running ASR {n_jb}/{k+1} ({100*n_jb/(k+1):.1f}%)")

    _print_summary(args.output)
    logger.info(f"Results: {args.output} | jailbreak prompts: {jb_path}")


def _print_summary(csv_path):
    """Print Table-2 (overall ASR + queries/success) and Figure-4 (per-category)."""
    import csv as _csv
    rows = list(_csv.DictReader(open(csv_path)))
    done = [r for r in rows if r["is_jailbroken"] in ("True", "False")]
    jb = [r for r in done if r["is_jailbroken"] == "True"]
    qs = [int(r["queries_to_jailbreak"]) for r in jb if r["queries_to_jailbreak"]]

    costs = [float(r["cost_usd"]) for r in done if r.get("cost_usd")]
    logger.info("=" * 50)
    logger.info(f"Total behaviors run : {len(done)}")
    logger.info(f"Jailbroken (ASR)    : {len(jb)}/{len(done)} "
                f"({100*len(jb)/max(1,len(done)):.1f}%)")
    if qs:
        logger.info(f"Queries/Success     : {sum(qs)/len(qs):.1f} (mean)")
    if costs:
        logger.info(f"Total cost (est.)   : ${sum(costs):.2f}  "
                    f"(${sum(costs)/len(costs):.3f}/behavior)")
        for model, (pt, ct) in _usage.items():
            logger.info(f"    tokens {model}: {pt:,} in / {ct:,} out")
    logger.info("-" * 50)
    logger.info("Per-category ASR (Figure 4):")
    cats = OrderedDict()
    for r in done:
        c = cats.setdefault(r["category"], [0, 0])
        c[1] += 1
        if r["is_jailbroken"] == "True":
            c[0] += 1
    for cat, (j, t) in cats.items():
        logger.info(f"  {cat:28} {j}/{t} ({100*j/t:.0f}%)")
    logger.info("=" * 50)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # subset selection
    p.add_argument("--num-behaviors", type=int, default=50)
    p.add_argument("--sampling", choices=["head", "stratified"], default="head",
                   help="head=first N (few categories); stratified=spread across all 10 categories")
    p.add_argument("--output", type=str, default="results/results.csv")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="overwrite the output CSV instead of resuming from it")
    p.set_defaults(resume=True)
    # models (defaults = verified working config)
    p.add_argument("--attack-model", default="qwen-2.5-7b-instruct-turbo")
    p.add_argument("--target-model", default="gpt-3.5-turbo-1106")
    p.add_argument("--judge-model", default="llama-guard-4-12b")
    # PAIR hyperparameters
    p.add_argument("--n-streams", type=int, default=30)
    p.add_argument("--n-iterations", type=int, default=3)
    p.add_argument("--keep-last-n", type=int, default=4)
    p.add_argument("--attack-max-n-tokens", type=int, default=1024)
    p.add_argument("--target-max-n-tokens", type=int, default=150)
    p.add_argument("--judge-max-n-tokens", type=int, default=64)
    p.add_argument("--max-n-attack-attempts", type=int, default=5)
    p.add_argument("--judge-temperature", type=float, default=0)
    p.add_argument("--jailbreakbench-phase", default="dev", choices=["dev", "test", "eval"])
    p.add_argument("--evaluate-locally", action="store_true")
    p.add_argument("-v", "--verbosity", action="count", default=1)
    args = p.parse_args()
    logger.set_level(args.verbosity)
    main(args)
