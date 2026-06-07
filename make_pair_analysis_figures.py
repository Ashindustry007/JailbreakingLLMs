"""Create paper-style PAIR analysis figures from completed status.jsonl files."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CATEGORY_ORDER = [
    "Economic harm",
    "Fraud/Deception",
    "Harassment/Discrimination",
    "Malware/Hacking",
    "Physical harm",
]

TARGET_LABELS = {
    "llama-3-8b-instruct-lite": "Llama-3-8B",
    "gemma-3n-e4b-it": "Gemma-3n",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("logs/pair_together_variant"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/pair_together_variant/analysis_figures"))
    parser.add_argument("--query-budget", type=int, default=90)
    parser.add_argument("--streams", type=int, default=30)
    return parser.parse_args()


def load_records(input_root: Path) -> list[dict]:
    records = []
    for status_path in sorted(input_root.glob("*/status.jsonl")):
        with status_path.open(encoding="utf-8") as status_file:
            for line in status_file:
                if line.strip():
                    record = json.loads(line)
                    if record.get("returncode") == 0:
                        records.append(record)
    return records


def target_label(target_model: str) -> str:
    return TARGET_LABELS.get(target_model, target_model)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_center(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, fill: str, text_font) -> None:
    bbox = draw.textbbox((0, 0), text, font=text_font)
    draw.text((xy[0] - (bbox[2] - bbox[0]) / 2, xy[1] - (bbox[3] - bbox[1]) / 2), text, fill=fill, font=text_font)


def text_right(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, fill: str, text_font) -> None:
    bbox = draw.textbbox((0, 0), text, font=text_font)
    draw.text((xy[0] - (bbox[2] - bbox[0]), xy[1]), text, fill=fill, font=text_font)


def red_scale_rgb(value: float) -> tuple[int, int, int]:
    if math.isnan(value):
        return (247, 247, 247)
    green_blue = int(245 - 155 * value / 100)
    return (255, green_blue, green_blue)


def red_scale(value: float) -> str:
    if math.isnan(value):
        return "#f7f7f7"
    # Soft Reds palette, scaled 0..100.
    green_blue = int(245 - 155 * value / 100)
    return f"rgb(255,{green_blue},{green_blue})"


def build_heatmap(records: list[dict]) -> tuple[list[dict], str]:
    targets = sorted({record["target_model"] for record in records})
    categories = [category for category in CATEGORY_ORDER if any(r["category"] == category for r in records)]

    rows = []
    for category in categories:
        for target in targets:
            cells = [r for r in records if r["target_model"] == target and r["category"] == category]
            successes = sum(1 for r in cells if r.get("jailbroken") is True)
            total = len(cells)
            rate = 100 * successes / total if total else math.nan
            rows.append(
                {
                    "target_model": target,
                    "target_label": target_label(target),
                    "category": category,
                    "jailbroken": successes,
                    "total": total,
                    "jailbreak_percent": "" if math.isnan(rate) else round(rate, 2),
                }
            )

    cell_w, cell_h = 135, 42
    left, top = 235, 78
    legend_w = 58
    width = left + cell_w * len(targets) + legend_w + 70
    height = top + cell_h * len(categories) + 105
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="26" y="34" font-family="Arial" font-size="20" font-weight="bold">PAIR JB% by Target x Category</text>',
        '<text x="26" y="56" font-family="Arial" font-size="12" fill="#444">Source: PAIR runs, first 50 JailbreakBench harmful behaviors, N=30, K=3</text>',
    ]

    for j, target in enumerate(targets):
        x = left + j * cell_w + cell_w / 2
        svg.append(
            f'<text x="{x}" y="{top - 12}" text-anchor="middle" font-family="Arial" font-size="13">{html.escape(target_label(target))}</text>'
        )

    value_by_cell = {(row["category"], row["target_model"]): row for row in rows}
    for i, category in enumerate(categories):
        y = top + i * cell_h
        svg.append(
            f'<text x="{left - 12}" y="{y + 26}" text-anchor="end" font-family="Arial" font-size="13">{html.escape(category)}</text>'
        )
        for j, target in enumerate(targets):
            x = left + j * cell_w
            row = value_by_cell[(category, target)]
            value = float(row["jailbreak_percent"]) if row["jailbreak_percent"] != "" else math.nan
            label = "" if math.isnan(value) else f"{value:.0f}"
            svg.append(
                f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{red_scale(value)}" stroke="#d0d0d0"/>'
            )
            svg.append(
                f'<text x="{x + cell_w / 2}" y="{y + 26}" text-anchor="middle" font-family="Arial" font-size="13" fill="#222">{label}</text>'
            )

    legend_x = left + cell_w * len(targets) + 30
    legend_y = top + 5
    legend_h = cell_h * len(categories) - 10
    for k in range(20):
        value = 100 - k * 100 / 19
        y = legend_y + k * legend_h / 20
        svg.append(
            f'<rect x="{legend_x}" y="{y}" width="14" height="{legend_h / 20 + 1}" fill="{red_scale(value)}"/>'
        )
    svg.extend(
        [
            f'<rect x="{legend_x}" y="{legend_y}" width="14" height="{legend_h}" fill="none" stroke="#333"/>',
            f'<text x="{legend_x + 22}" y="{legend_y + 4}" font-family="Arial" font-size="12">100</text>',
            f'<text x="{legend_x + 22}" y="{legend_y + legend_h}" font-family="Arial" font-size="12">0</text>',
            f'<text x="{legend_x + 42}" y="{legend_y + legend_h / 2}" transform="rotate(-90 {legend_x + 42} {legend_y + legend_h / 2})" font-family="Arial" font-size="13">Jailbreak %</text>',
            "</svg>",
        ]
    )
    return rows, "\n".join(svg)


def draw_heatmap_png(rows: list[dict], output_path: Path) -> None:
    targets = sorted({row["target_model"] for row in rows})
    categories = [category for category in CATEGORY_ORDER if any(row["category"] == category for row in rows)]
    value_by_cell = {(row["category"], row["target_model"]): row for row in rows}

    width, height = 1200, 620
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(30, bold=True)
    label_font = font(22)
    small_font = font(18)
    cell_font = font(22)

    draw.text((42, 34), "PAIR JB% by Target x Category", fill="#111111", font=title_font)
    draw.text(
        (42, 72),
        "Source: PAIR runs, first 50 JailbreakBench harmful behaviors, N=30, K=3",
        fill="#444444",
        font=small_font,
    )

    left, top = 360, 145
    cell_w, cell_h = 220, 62
    for j, target in enumerate(targets):
        text_center(draw, (left + j * cell_w + cell_w / 2, top - 34), target_label(target), "#111111", label_font)

    for i, category in enumerate(categories):
        y = top + i * cell_h
        text_right(draw, (left - 18, y + 19), category, "#111111", label_font)
        for j, target in enumerate(targets):
            x = left + j * cell_w
            row = value_by_cell[(category, target)]
            value = float(row["jailbreak_percent"]) if row["jailbreak_percent"] != "" else math.nan
            draw.rectangle([x, y, x + cell_w, y + cell_h], fill=red_scale_rgb(value), outline="#d0d0d0")
            text_center(draw, (x + cell_w / 2, y + cell_h / 2), f"{value:.0f}", "#222222", cell_font)

    legend_x, legend_y, legend_w, legend_h = left + cell_w * len(targets) + 70, top + 6, 24, cell_h * len(categories) - 12
    for k in range(legend_h):
        value = 100 - 100 * k / max(1, legend_h - 1)
        draw.line([(legend_x, legend_y + k), (legend_x + legend_w, legend_y + k)], fill=red_scale_rgb(value))
    draw.rectangle([legend_x, legend_y, legend_x + legend_w, legend_y + legend_h], outline="#333333")
    draw.text((legend_x + 34, legend_y - 8), "100", fill="#111111", font=small_font)
    draw.text((legend_x + 34, legend_y + legend_h - 14), "0", fill="#111111", font=small_font)
    draw.text((legend_x - 5, legend_y + legend_h + 20), "Jailbreak %", fill="#111111", font=small_font)
    image.save(output_path)


def first_iteration(queries_to_jailbreak: int, streams: int) -> int:
    return int(math.ceil(queries_to_jailbreak / streams))


def build_efficiency(records: list[dict], query_budget: int, streams: int) -> tuple[list[dict], list[dict], str]:
    targets = sorted({record["target_model"] for record in records})
    curve_rows = []
    iteration_rows = []

    width, height = 920, 390
    margin = {"left": 72, "right": 28, "top": 50, "bottom": 78}
    panel_gap = 72
    panel_w = (width - margin["left"] - margin["right"] - panel_gap) / 2
    panel_h = height - margin["top"] - margin["bottom"]
    left_x, right_x = margin["left"], margin["left"] + panel_w + panel_gap
    top_y = margin["top"]

    colors = {
        "llama-3-8b-instruct-lite": "#c75348",
        "gemma-3n-e4b-it": "#4f7fa3",
    }
    default_colors = ["#c75348", "#4f7fa3", "#5f8a58", "#8a6aa6"]

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]

    def x_left(query: float) -> float:
        return left_x + panel_w * query / query_budget

    def y_left(percent: float) -> float:
        return top_y + panel_h * (1 - percent / 100)

    # Left panel axes.
    svg.extend(
        [
            f'<text x="{left_x + panel_w / 2}" y="{top_y - 18}" text-anchor="middle" font-family="Arial" font-size="14">(a) Cumulative jailbreak success</text>',
            f'<line x1="{left_x}" y1="{top_y + panel_h}" x2="{left_x + panel_w}" y2="{top_y + panel_h}" stroke="#333"/>',
            f'<line x1="{left_x}" y1="{top_y}" x2="{left_x}" y2="{top_y + panel_h}" stroke="#333"/>',
            f'<text x="{left_x + panel_w / 2}" y="{top_y + panel_h + 42}" text-anchor="middle" font-family="Arial" font-size="12">Queries to first jailbreak</text>',
            f'<text x="18" y="{top_y + panel_h / 2}" transform="rotate(-90 18 {top_y + panel_h / 2})" text-anchor="middle" font-family="Arial" font-size="12">Cumulative JB% of behaviors</text>',
        ]
    )
    for tick in [0, 30, 60, 90]:
        x = x_left(tick)
        svg.append(f'<line x1="{x}" y1="{top_y + panel_h}" x2="{x}" y2="{top_y + panel_h + 5}" stroke="#333"/>')
        svg.append(f'<text x="{x}" y="{top_y + panel_h + 20}" text-anchor="middle" font-family="Arial" font-size="11">{tick}</text>')
    for tick in [0, 25, 50, 75, 100]:
        y = y_left(tick)
        svg.append(f'<line x1="{left_x - 5}" y1="{y}" x2="{left_x}" y2="{y}" stroke="#333"/>')
        svg.append(f'<text x="{left_x - 9}" y="{y + 4}" text-anchor="end" font-family="Arial" font-size="11">{tick}</text>')
        if tick not in [0, 100]:
            svg.append(f'<line x1="{left_x}" y1="{y}" x2="{left_x + panel_w}" y2="{y}" stroke="#e8e8e8"/>')

    # Right panel axes.
    max_count = 1
    counts_by_target: dict[str, Counter] = {}
    for target in targets:
        target_records = [r for r in records if r["target_model"] == target and r.get("jailbroken") is True]
        counter = Counter(first_iteration(int(r["queries_to_jailbreak"]), streams) for r in target_records)
        counts_by_target[target] = counter
        max_count = max(max_count, max(counter.values(), default=0))

    max_count = max(5, int(math.ceil(max_count / 5) * 5))

    def y_right(count: float) -> float:
        return top_y + panel_h * (1 - count / max_count)

    svg.extend(
        [
            f'<text x="{right_x + panel_w / 2}" y="{top_y - 18}" text-anchor="middle" font-family="Arial" font-size="14">(b) First-success iteration</text>',
            f'<line x1="{right_x}" y1="{top_y + panel_h}" x2="{right_x + panel_w}" y2="{top_y + panel_h}" stroke="#333"/>',
            f'<line x1="{right_x}" y1="{top_y}" x2="{right_x}" y2="{top_y + panel_h}" stroke="#333"/>',
            f'<text x="{right_x + panel_w / 2}" y="{top_y + panel_h + 42}" text-anchor="middle" font-family="Arial" font-size="12">Iteration of first success</text>',
            f'<text x="{right_x - 45}" y="{top_y + panel_h / 2}" transform="rotate(-90 {right_x - 45} {top_y + panel_h / 2})" text-anchor="middle" font-family="Arial" font-size="12"># successful behaviors</text>',
        ]
    )
    for tick in range(0, max_count + 1, max(1, max_count // 5)):
        y = y_right(tick)
        svg.append(f'<line x1="{right_x - 5}" y1="{y}" x2="{right_x}" y2="{y}" stroke="#333"/>')
        svg.append(f'<text x="{right_x - 9}" y="{y + 4}" text-anchor="end" font-family="Arial" font-size="11">{tick}</text>')
        if tick not in [0, max_count]:
            svg.append(f'<line x1="{right_x}" y1="{y}" x2="{right_x + panel_w}" y2="{y}" stroke="#e8e8e8"/>')

    # Draw target series.
    legend_x, legend_y = left_x + panel_w - 122, top_y + 8
    for idx, target in enumerate(targets):
        color = colors.get(target, default_colors[idx % len(default_colors)])
        target_records = [r for r in records if r["target_model"] == target]
        successes = sorted(
            int(r["queries_to_jailbreak"])
            for r in target_records
            if r.get("jailbroken") is True and r.get("queries_to_jailbreak") is not None
        )
        total = len(target_records)
        points = [(0, 0)]
        for q in range(1, query_budget + 1):
            count = sum(1 for success_q in successes if success_q <= q)
            percent = 100 * count / total if total else 0
            curve_rows.append(
                {
                    "target_model": target,
                    "target_label": target_label(target),
                    "query": q,
                    "cumulative_jailbreak_percent": round(percent, 2),
                    "cumulative_jailbreaks": count,
                    "total_behaviors": total,
                }
            )
            points.append((q, percent))
        point_str = " ".join(f"{x_left(q):.2f},{y_left(p):.2f}" for q, p in points)
        svg.append(f'<polyline points="{point_str}" fill="none" stroke="{color}" stroke-width="2"/>')
        svg.append(f'<rect x="{legend_x}" y="{legend_y + idx * 18}" width="12" height="3" fill="{color}"/>')
        svg.append(
            f'<text x="{legend_x + 18}" y="{legend_y + idx * 18 + 5}" font-family="Arial" font-size="11">{html.escape(target_label(target))}</text>'
        )

        # Bars, grouped by iteration 1..3.
        iterations = [1, 2, 3]
        group_w = panel_w / len(iterations)
        bar_w = min(24, group_w / (len(targets) + 1))
        for iteration in iterations:
            count = counts_by_target[target].get(iteration, 0)
            iteration_rows.append(
                {
                    "target_model": target,
                    "target_label": target_label(target),
                    "iteration": iteration,
                    "successful_behaviors": count,
                }
            )
            center = right_x + group_w * (iteration - 0.5)
            x = center - (len(targets) * bar_w) / 2 + idx * bar_w
            y = y_right(count)
            h = top_y + panel_h - y
            svg.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w - 2:.2f}" height="{h:.2f}" fill="{color}"/>')
            svg.append(
                f'<text x="{x + (bar_w - 2) / 2:.2f}" y="{y - 4:.2f}" text-anchor="middle" font-family="Arial" font-size="10">{count}</text>'
            )

    for iteration in [1, 2, 3]:
        group_w = panel_w / 3
        x = right_x + group_w * (iteration - 0.5)
        svg.append(f'<text x="{x}" y="{top_y + panel_h + 20}" text-anchor="middle" font-family="Arial" font-size="11">{iteration}</text>')

    # Right panel legend.
    rlegend_x, rlegend_y = right_x + panel_w - 122, top_y + 8
    for idx, target in enumerate(targets):
        color = colors.get(target, default_colors[idx % len(default_colors)])
        svg.append(f'<rect x="{rlegend_x}" y="{rlegend_y + idx * 18}" width="12" height="8" fill="{color}"/>')
        svg.append(
            f'<text x="{rlegend_x + 18}" y="{rlegend_y + idx * 18 + 8}" font-family="Arial" font-size="11">{html.escape(target_label(target))}</text>'
        )

    svg.append(
        f'<text x="{width / 2}" y="{height - 16}" text-anchor="middle" font-family="Arial" font-size="11" fill="#444">'
        'Source: PAIR runs, first 50 JailbreakBench harmful behaviors, N=30, K=3'
        '</text>'
    )
    svg.append("</svg>")
    return curve_rows, iteration_rows, "\n".join(svg)


def draw_efficiency_png(records: list[dict], query_budget: int, streams: int, output_path: Path) -> None:
    targets = sorted({record["target_model"] for record in records})
    colors = {
        "llama-3-8b-instruct-lite": (199, 83, 72),
        "gemma-3n-e4b-it": (79, 127, 163),
    }
    fallback_colors = [(199, 83, 72), (79, 127, 163), (95, 138, 88)]

    width, height = 1800, 760
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(26)
    label_font = font(22)
    tick_font = font(18)
    small_font = font(18)

    left_x, right_x = 130, 1050
    top_y, panel_w, panel_h = 95, 670, 500
    panel_gap = 140

    def x_left(query: float) -> float:
        return left_x + panel_w * query / query_budget

    def y_left(percent: float) -> float:
        return top_y + panel_h * (1 - percent / 100)

    text_center(draw, (left_x + panel_w / 2, 48), "(a) Cumulative jailbreak success", "#111111", title_font)
    text_center(draw, (right_x + panel_w / 2, 48), "(b) First-success iteration", "#111111", title_font)

    # Left axes.
    draw.line([(left_x, top_y + panel_h), (left_x + panel_w, top_y + panel_h)], fill="#333333", width=2)
    draw.line([(left_x, top_y), (left_x, top_y + panel_h)], fill="#333333", width=2)
    for tick in [0, 30, 60, 90]:
        x = x_left(tick)
        draw.line([(x, top_y + panel_h), (x, top_y + panel_h + 10)], fill="#333333", width=2)
        text_center(draw, (x, top_y + panel_h + 34), str(tick), "#111111", tick_font)
    for tick in [0, 25, 50, 75, 100]:
        y = y_left(tick)
        draw.line([(left_x - 10, y), (left_x, y)], fill="#333333", width=2)
        text_right(draw, (left_x - 18, y - 10), str(tick), "#111111", tick_font)
        if tick not in [0, 100]:
            draw.line([(left_x, y), (left_x + panel_w, y)], fill="#e7e7e7", width=1)
    text_center(draw, (left_x + panel_w / 2, top_y + panel_h + 70), "Queries to first jailbreak", "#111111", label_font)
    # PIL text rotation for y-label.
    y_label = Image.new("RGBA", (360, 34), (255, 255, 255, 0))
    y_draw = ImageDraw.Draw(y_label)
    y_draw.text((0, 0), "Cumulative JB% of behaviors", fill="#111111", font=label_font)
    rotated = y_label.rotate(90, expand=True)
    image.paste(rotated, (28, int(top_y + panel_h / 2 - rotated.height / 2)), rotated)

    # Curves and legend.
    for idx, target in enumerate(targets):
        color = colors.get(target, fallback_colors[idx % len(fallback_colors)])
        target_records = [r for r in records if r["target_model"] == target]
        successes = sorted(
            int(r["queries_to_jailbreak"])
            for r in target_records
            if r.get("jailbroken") is True and r.get("queries_to_jailbreak") is not None
        )
        total = len(target_records)
        points = [(x_left(0), y_left(0))]
        for q in range(1, query_budget + 1):
            count = sum(1 for success_q in successes if success_q <= q)
            points.append((x_left(q), y_left(100 * count / total if total else 0)))
        draw.line(points, fill=color, width=4)
        legend_y = top_y + 22 + idx * 32
        draw.line([(left_x + panel_w - 190, legend_y), (left_x + panel_w - 150, legend_y)], fill=color, width=5)
        draw.text((left_x + panel_w - 140, legend_y - 12), target_label(target), fill="#111111", font=small_font)

    # Right panel data.
    counts_by_target = {}
    max_count = 1
    for target in targets:
        target_records = [r for r in records if r["target_model"] == target and r.get("jailbroken") is True]
        counter = Counter(first_iteration(int(r["queries_to_jailbreak"]), streams) for r in target_records)
        counts_by_target[target] = counter
        max_count = max(max_count, max(counter.values(), default=0))
    max_count = max(5, int(math.ceil(max_count / 5) * 5))

    def y_right(count: float) -> float:
        return top_y + panel_h * (1 - count / max_count)

    draw.line([(right_x, top_y + panel_h), (right_x + panel_w, top_y + panel_h)], fill="#333333", width=2)
    draw.line([(right_x, top_y), (right_x, top_y + panel_h)], fill="#333333", width=2)
    tick_step = max(1, max_count // 5)
    for tick in range(0, max_count + 1, tick_step):
        y = y_right(tick)
        draw.line([(right_x - 10, y), (right_x, y)], fill="#333333", width=2)
        text_right(draw, (right_x - 18, y - 10), str(tick), "#111111", tick_font)
        if tick not in [0, max_count]:
            draw.line([(right_x, y), (right_x + panel_w, y)], fill="#e7e7e7", width=1)
    text_center(draw, (right_x + panel_w / 2, top_y + panel_h + 70), "Iteration of first success", "#111111", label_font)
    y_label = Image.new("RGBA", (280, 34), (255, 255, 255, 0))
    y_draw = ImageDraw.Draw(y_label)
    y_draw.text((0, 0), "# successful behaviors", fill="#111111", font=label_font)
    rotated = y_label.rotate(90, expand=True)
    image.paste(rotated, (right_x - 82, int(top_y + panel_h / 2 - rotated.height / 2)), rotated)

    iterations = [1, 2, 3]
    group_w = panel_w / len(iterations)
    bar_w = 42
    for iteration in iterations:
        center = right_x + group_w * (iteration - 0.5)
        text_center(draw, (center, top_y + panel_h + 34), str(iteration), "#111111", tick_font)
        for idx, target in enumerate(targets):
            color = colors.get(target, fallback_colors[idx % len(fallback_colors)])
            count = counts_by_target[target].get(iteration, 0)
            x = center - len(targets) * bar_w / 2 + idx * bar_w
            y = y_right(count)
            draw.rectangle([x, y, x + bar_w - 8, top_y + panel_h], fill=color)
            text_center(draw, (x + (bar_w - 8) / 2, y - 16), str(count), "#111111", tick_font)

    for idx, target in enumerate(targets):
        color = colors.get(target, fallback_colors[idx % len(fallback_colors)])
        legend_y = top_y + 22 + idx * 32
        draw.rectangle([right_x + panel_w - 190, legend_y - 10, right_x + panel_w - 166, legend_y + 6], fill=color)
        draw.text((right_x + panel_w - 154, legend_y - 14), target_label(target), fill="#111111", font=small_font)

    text_center(
        draw,
        (width / 2, height - 28),
        "Source: PAIR runs, first 50 JailbreakBench harmful behaviors, N=30, K=3",
        "#444444",
        small_font,
    )
    image.save(output_path)


def main() -> int:
    args = parse_args()
    records = load_records(args.input_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    heatmap_rows, heatmap_svg = build_heatmap(records)
    curve_rows, iteration_rows, efficiency_svg = build_efficiency(records, args.query_budget, args.streams)

    write_csv(
        args.output_dir / "category_heatmap_values.csv",
        heatmap_rows,
        ["target_model", "target_label", "category", "jailbroken", "total", "jailbreak_percent"],
    )
    write_csv(
        args.output_dir / "efficiency_curve_values.csv",
        curve_rows,
        ["target_model", "target_label", "query", "cumulative_jailbreak_percent", "cumulative_jailbreaks", "total_behaviors"],
    )
    write_csv(
        args.output_dir / "first_success_iteration_values.csv",
        iteration_rows,
        ["target_model", "target_label", "iteration", "successful_behaviors"],
    )
    (args.output_dir / "category_heatmap.svg").write_text(heatmap_svg, encoding="utf-8")
    (args.output_dir / "pair_efficiency.svg").write_text(efficiency_svg, encoding="utf-8")
    draw_heatmap_png(heatmap_rows, args.output_dir / "category_heatmap.png")
    draw_efficiency_png(records, args.query_budget, args.streams, args.output_dir / "pair_efficiency.png")
    print(f"Wrote analysis figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
