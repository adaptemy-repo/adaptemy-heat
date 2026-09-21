"""Analyze BRISQUE versus DMOS while holding source-video content fixed."""

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape


QP_NAME = re.compile(r"^(?P<source>.+)_qp(?P<qp>\d+)(?:_.*)?$")


def load_groups(path):
    groups = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            match = QP_NAME.match(row["video_id"])
            if not match:
                raise ValueError(f"Video ID has no _qp<number> suffix: {row['video_id']}")
            score = float(row["brisque_raw"])
            dmos = float(row["dmos"])
            if math.isfinite(score) and math.isfinite(dmos):
                groups[match["source"]].append({
                    "video_id": row["video_id"],
                    "qp": int(match["qp"]),
                    "brisque_raw": score,
                    "dmos": dmos,
                })
    return groups


def analyze(groups):
    centered = []
    summaries = []
    for source, entries in sorted(groups.items()):
        if len(entries) < 2:
            continue
        xs = [entry["brisque_raw"] for entry in entries]
        ys = [entry["dmos"] for entry in entries]
        correlation = (
            statistics.correlation(xs, ys)
            if len(set(xs)) > 1 and len(set(ys)) > 1 else math.nan
        )
        summaries.append({"source": source, "videos": len(entries), "plcc": correlation})
        for entry in entries:
            centered.append({
                "source": source,
                "video_id": entry["video_id"],
                "qp": entry["qp"],
                "brisque_raw": entry["brisque_raw"],
                "dmos": entry["dmos"],
                "brisque_delta": entry["brisque_raw"] - statistics.mean(xs),
                "dmos_delta": entry["dmos"] - statistics.mean(ys),
            })
    return centered, summaries


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_svg(path, rows, plcc):
    width, height = 880, 650
    left, right, top, bottom = 100, 35, 80, 90
    xs = [row["brisque_delta"] for row in rows]
    ys = [row["dmos_delta"] for row in rows]
    x_extent = max(abs(x) for x in xs) * 1.08 or 1.0
    y_extent = max(abs(y) for y in ys) * 1.08 or 1.0
    plot_w, plot_h = width - left - right, height - top - bottom

    def sx(value):
        return left + (value / x_extent + 1) * plot_w / 2

    def sy(value):
        return top + (1 - value / y_extent) * plot_h / 2

    palette = {27: "#087f8c", 37: "#e09f3e", 42: "#d1495b"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f9f8f4"/>',
        '<text x="100" y="42" font-family="sans-serif" font-size="22" fill="#202c32">Within-source BRISQUE vs DMOS</text>',
        f'<text x="100" y="66" font-family="sans-serif" font-size="14" fill="#50616a">Source-centered PLCC = {plcc:.4f}; each dot is one impaired video</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="white" stroke="#cad1d2"/>',
        f'<line x1="{sx(0):.1f}" y1="{top}" x2="{sx(0):.1f}" y2="{top + plot_h}" stroke="#aab5b7" stroke-dasharray="5 5"/>',
        f'<line x1="{left}" y1="{sy(0):.1f}" x2="{left + plot_w}" y2="{sy(0):.1f}" stroke="#aab5b7" stroke-dasharray="5 5"/>',
    ]
    for row in rows:
        color = palette.get(row["qp"], "#666666")
        parts.append(
            f'<circle cx="{sx(row["brisque_delta"]):.2f}" cy="{sy(row["dmos_delta"]):.2f}" '
            f'r="4.5" fill="{color}" fill-opacity="0.72">'
            f'<title>{escape(row["video_id"])}: BRISQUE delta {row["brisque_delta"]:.3f}, '
            f'DMOS delta {row["dmos_delta"]:.3f}</title></circle>'
        )
    parts.extend([
        f'<text x="{width / 2}" y="{height - 25}" text-anchor="middle" font-family="sans-serif" font-size="16">BRISQUE minus source mean</text>',
        f'<text transform="translate(25 {height / 2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="16">DMOS minus source mean</text>',
    ])
    for index, (qp, color) in enumerate(palette.items()):
        x = left + index * 100
        parts.append(f'<circle cx="{x}" cy="{height - 58}" r="5" fill="{color}"/>')
        parts.append(f'<text x="{x + 12}" y="{height - 53}" font-family="sans-serif" font-size="13">QP {qp}</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "data/clip_brisque_dmos_scores_all180.csv")
    parser.add_argument("--output", type=Path, default=root / "data/clip_brisque_within_source.svg")
    parser.add_argument("--scores-output", type=Path, default=root / "data/clip_brisque_within_source.csv")
    parser.add_argument("--groups-output", type=Path, default=root / "data/clip_brisque_source_correlations.csv")
    args = parser.parse_args()

    rows, summaries = analyze(load_groups(args.input))
    if len(rows) < 2:
        raise ValueError("Need at least two videos in multi-video source groups")
    plcc = statistics.correlation(
        [row["brisque_delta"] for row in rows], [row["dmos_delta"] for row in rows]
    )
    write_csv(args.scores_output, rows, (
        "source", "video_id", "qp", "brisque_raw", "dmos", "brisque_delta", "dmos_delta"
    ))
    write_csv(args.groups_output, summaries, ("source", "videos", "plcc"))
    write_svg(args.output, rows, plcc)
    valid = [row["plcc"] for row in summaries if math.isfinite(row["plcc"])]
    print(f"videos={len(rows)} sources={len(summaries)}")
    print(f"PLCC_WITHIN_SOURCE_CENTERED={plcc:.6f}")
    print(f"MEDIAN_PER_SOURCE_PLCC={statistics.median(valid):.6f}")
    print(f"plot={args.output}")
    print(f"scores={args.scores_output}")
    print(f"groups={args.groups_output}")


if __name__ == "__main__":
    main()
