"""Summarize PAIR runs into tables and lightweight SVG figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean


PRICE_PER_M_TOKENS = {
    "qwen-2.5-7b-instruct-turbo": {"input": 0.30, "output": 0.30},
    "gemma-3n-e4b-it": {"input": 0.06, "output": 0.12},
    "llama-3-8b-instruct-lite": {"input": 0.14, "output": 0.14},
    "llama-guard-4-12b": {"input": 0.20, "output": 0.20},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("logs/pair_together_variant"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/pair_together_variant"))
    parser.add_argument("--attack-max-n-tokens", type=int, default=1024)
    parser.add_argument("--target-max-n-tokens", type=int, default=150)
    parser.add_argument("--judge-max-n-tokens", type=int, default=64)
    return parser.parse_args()


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_records(input_root: Path) -> list[dict]:
    latest: dict[tuple[str, int], dict] = {}
    for status_path in input_root.glob("**/status.jsonl"):
        with status_path.open(encoding="utf-8") as status_file:
            for line in status_file:
                if not line.strip():
                    continue
                record = json.loads(line)
                record["_status_path"] = str(status_path)
                key = (str(status_path), int(record["index"]))
                latest[key] = record
    return list(latest.values())


def target_label(record: dict) -> str:
    return str(record.get("target_model", "unknown"))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_table2(records: list[dict]) -> list[dict]:
    rows = []
    by_target: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record.get("returncode") == 0:
            by_target[target_label(record)].append(record)

    for target, target_records in sorted(by_target.items()):
        successes = [r for r in target_records if r.get("jailbroken") is True]
        queries = [
            int(r["queries_to_jailbreak"])
            for r in successes
            if r.get("queries_to_jailbreak") is not None
        ]
        rows.append(
            {
                "target_model": target,
                "completed_behaviors": len(target_records),
                "jailbroken_behaviors": len(successes),
                "jailbreak_percent": round(100 * len(successes) / len(target_records), 2)
                if target_records
                else 0,
                "queries_per_success": round(mean(queries), 2) if queries else "",
            }
        )
    return rows


def summarize_table4(records: list[dict], args: argparse.Namespace) -> list[dict]:
    rows = []
    by_target: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record.get("returncode") == 0:
            by_target[target_label(record)].append(record)

    for target, target_records in sorted(by_target.items()):
        attack_model = str(target_records[0].get("attack_model", ""))
        judge_model = str(target_records[0].get("judge_model", ""))
        n_streams = int(target_records[0].get("n_streams", 0))
        n_iterations = int(target_records[0].get("n_iterations", 0))
        query_budget = n_streams * n_iterations
        query_counts = [
            int(r["queries_to_jailbreak"])
            if r.get("queries_to_jailbreak") is not None
            else query_budget
            for r in target_records
        ]

        wall_seconds = []
        for record in target_records:
            started = parse_time(record.get("started_at"))
            finished = parse_time(record.get("finished_at"))
            if started and finished:
                wall_seconds.append((finished - started).total_seconds())

        attack_price = PRICE_PER_M_TOKENS.get(attack_model, {}).get("output", 0)
        target_price = PRICE_PER_M_TOKENS.get(target, {}).get("output", 0)
        judge_price = PRICE_PER_M_TOKENS.get(judge_model, {}).get("output", 0)
        output_cap_cost = sum(query_counts) * (
            args.attack_max_n_tokens * attack_price
            + args.target_max_n_tokens * target_price
            + args.judge_max_n_tokens * judge_price
        ) / 1_000_000

        rows.append(
            {
                "target_model": target,
                "completed_behaviors": len(target_records),
                "total_target_queries_est": sum(query_counts),
                "mean_queries_est": round(mean(query_counts), 2) if query_counts else "",
                "total_wall_seconds": round(sum(wall_seconds), 2),
                "mean_wall_seconds": round(mean(wall_seconds), 2) if wall_seconds else "",
                "output_cap_cost_usd_est": round(output_cap_cost, 4),
                "cost_note": "Uses output token caps only; input tokens/provider billing may differ.",
            }
        )
    return rows


def category_rates(records: list[dict]) -> tuple[list[str], list[str], dict[tuple[str, str], float]]:
    categories = sorted({str(r.get("category", "unknown")) for r in records if r.get("returncode") == 0})
    targets = sorted({target_label(r) for r in records if r.get("returncode") == 0})
    rates = {}
    for category in categories:
        for target in targets:
            cells = [
                r
                for r in records
                if r.get("returncode") == 0
                and str(r.get("category", "unknown")) == category
                and target_label(r) == target
            ]
            rates[(category, target)] = (
                100 * sum(1 for r in cells if r.get("jailbroken") is True) / len(cells)
                if cells
                else math.nan
            )
    return categories, targets, rates


def color_for_rate(value: float) -> str:
    if math.isnan(value):
        return "#eeeeee"
    intensity = int(255 - 180 * value / 100)
    return f"rgb(255,{intensity},{intensity})"


def write_heatmap_svg(path: Path, categories: list[str], targets: list[str], rates: dict) -> None:
    cell_w, cell_h = 160, 28
    left, top = 260, 80
    width = left + cell_w * len(targets) + 30
    height = top + cell_h * len(categories) + 40
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="30" font-family="Arial" font-size="18" font-weight="bold">Figure 4-style Jailbreak % by Category</text>',
    ]
    for j, target in enumerate(targets):
        x = left + j * cell_w + 5
        lines.append(
            f'<text x="{x}" y="{top - 12}" font-family="Arial" font-size="11">{target}</text>'
        )
    for i, category in enumerate(categories):
        y = top + i * cell_h
        lines.append(
            f'<text x="10" y="{y + 18}" font-family="Arial" font-size="12">{category}</text>'
        )
        for j, target in enumerate(targets):
            x = left + j * cell_w
            value = rates[(category, target)]
            label = "" if math.isnan(value) else f"{value:.0f}%"
            lines.append(
                f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" '
                f'fill="{color_for_rate(value)}" stroke="#cccccc"/>'
            )
            lines.append(
                f'<text x="{x + cell_w / 2}" y="{y + 18}" text-anchor="middle" '
                f'font-family="Arial" font-size="12">{label}</text>'
            )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_iteration_counts(log_path: Path) -> list[tuple[int, int, int]]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    iteration_matches = list(re.finditer(r"SUMMARY STATISTICS for Iteration (\d+)", text))
    counts = []
    for match in iteration_matches:
        iteration = int(match.group(1))
        segment = text[match.end() : match.end() + 300]
        count_match = re.search(r"Number of New Jailbreaks: (\d+)/(\d+)", segment)
        if count_match:
            counts.append((iteration, int(count_match.group(1)), int(count_match.group(2))))
    return counts


def summarize_figure5(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        log_file = record.get("log_file")
        if not log_file:
            continue
        log_path = Path(log_file)
        if not log_path.is_absolute():
            status_path = Path(record.get("_status_path", "."))
            log_path = status_path.parent / log_path.name
        if not log_path.exists():
            continue
        for iteration, new_jailbreaks, streams in parse_iteration_counts(log_path):
            rows.append(
                {
                    "target_model": target_label(record),
                    "behavior_index": record.get("index"),
                    "iteration": iteration,
                    "new_jailbreaks": new_jailbreaks,
                    "streams": streams,
                }
            )
    return rows


def resolve_log_path(record: dict) -> Path | None:
    log_file = record.get("log_file")
    if not log_file:
        return None
    log_path = Path(log_file)
    if not log_path.is_absolute():
        status_path = Path(record.get("_status_path", "."))
        log_path = status_path.parent / log_path.name
    return log_path if log_path.exists() else None


def write_figure3_example(path: Path, records: list[dict]) -> None:
    for record in records:
        if record.get("returncode") != 0 or record.get("jailbroken") is not True:
            continue
        log_path = resolve_log_path(record)
        if not log_path:
            continue
        text = log_path.read_text(encoding="utf-8", errors="replace")
        match = re.search(
            r"Example Jailbreak PROMPT:\n\n(?P<prompt>.*?)\n\n+Example Jailbreak RESPONSE:\n\n(?P<response>.*?)(?:\n\n\n|wandb:|Memory before:)",
            text,
            flags=re.DOTALL,
        )
        if not match:
            continue
        prompt = match.group("prompt").strip()
        response = match.group("response").strip()
        path.write_text(
            "\n".join(
                [
                    "# Figure 3-style Example Jailbreak",
                    "",
                    f"- target_model: `{target_label(record)}`",
                    f"- behavior_index: `{record.get('index')}`",
                    f"- category: `{record.get('category')}`",
                    "",
                    "## Prompt",
                    "",
                    "```text",
                    prompt,
                    "```",
                    "",
                    "## Response",
                    "",
                    "```text",
                    response,
                    "```",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return

    path.write_text(
        "# Figure 3-style Example Jailbreak\n\nNo successful jailbreak example was found in the completed logs.\n",
        encoding="utf-8",
    )


def write_bar_svg(path: Path, rows: list[dict]) -> None:
    totals: dict[tuple[str, int], int] = defaultdict(int)
    targets = sorted({r["target_model"] for r in rows})
    iterations = sorted({int(r["iteration"]) for r in rows})
    for row in rows:
        totals[(row["target_model"], int(row["iteration"]))] += int(row["new_jailbreaks"])

    cell_w, bar_w, gap = 120, 28, 8
    left, top, chart_h = 120, 60, 180
    width = left + len(targets) * len(iterations) * (bar_w + gap) + 80
    height = top + chart_h + 80
    max_value = max(totals.values(), default=1)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="30" font-family="Arial" font-size="18" font-weight="bold">Figure 5-style New Jailbreaks by Iteration</text>',
    ]
    x = left
    for target in targets:
        lines.append(f'<text x="{x}" y="{top + chart_h + 35}" font-family="Arial" font-size="11">{target}</text>')
        for iteration in iterations:
            value = totals[(target, iteration)]
            h = chart_h * value / max_value if max_value else 0
            y = top + chart_h - h
            lines.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{h}" fill="#7aa6c2"/>')
            lines.append(f'<text x="{x + bar_w / 2}" y="{y - 4}" text-anchor="middle" font-family="Arial" font-size="10">{value}</text>')
            lines.append(f'<text x="{x + bar_w / 2}" y="{top + chart_h + 15}" text-anchor="middle" font-family="Arial" font-size="10">K{iteration}</text>')
            x += bar_w + gap
        x += cell_w // 2
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown_summary(path: Path, table2_rows: list[dict], table4_rows: list[dict]) -> None:
    lines = ["# PAIR Together Variant Summary", ""]
    lines.append("## Table 2-style Metrics")
    for row in table2_rows:
        lines.append(
            f"- `{row['target_model']}`: JB% {row['jailbreak_percent']} "
            f"({row['jailbroken_behaviors']}/{row['completed_behaviors']}), "
            f"Queries/Success {row['queries_per_success'] or 'n/a'}"
        )
    lines.extend(["", "## Table 4-style Runtime/Cost Estimates"])
    for row in table4_rows:
        lines.append(
            f"- `{row['target_model']}`: wall {row['total_wall_seconds']}s total, "
            f"estimated output-cap cost ${row['output_cap_cost_usd_est']}"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "- Figure 4 heatmap is generated from `category` and `jailbroken` fields in status files.",
            "- Figure 5 is a log-parsed approximation from per-iteration summary lines, not the original paper's full K=1..12 ablation unless those runs are produced.",
            "- Cost estimates use configured max output token caps and public price assumptions; they are not exact provider invoices.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    records = load_records(args.input_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    table2_rows = summarize_table2(records)
    table4_rows = summarize_table4(records, args)
    figure5_rows = summarize_figure5(records)
    categories, targets, rates = category_rates(records)

    write_csv(
        args.output_dir / "table2_summary.csv",
        table2_rows,
        ["target_model", "completed_behaviors", "jailbroken_behaviors", "jailbreak_percent", "queries_per_success"],
    )
    write_csv(
        args.output_dir / "table4_cost_runtime_estimates.csv",
        table4_rows,
        [
            "target_model",
            "completed_behaviors",
            "total_target_queries_est",
            "mean_queries_est",
            "total_wall_seconds",
            "mean_wall_seconds",
            "output_cap_cost_usd_est",
            "cost_note",
        ],
    )
    write_csv(
        args.output_dir / "figure5_iteration_counts.csv",
        figure5_rows,
        ["target_model", "behavior_index", "iteration", "new_jailbreaks", "streams"],
    )
    write_heatmap_svg(args.output_dir / "figure4_category_heatmap.svg", categories, targets, rates)
    write_bar_svg(args.output_dir / "figure5_iteration_counts.svg", figure5_rows)
    write_figure3_example(args.output_dir / "figure3_example_jailbreak.md", records)
    write_markdown_summary(args.output_dir / "summary.md", table2_rows, table4_rows)
    print(f"Wrote summary artifacts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
