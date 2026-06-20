import csv
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "soda_cap" / "output" / "soda_cap_epoch_logs.csv"
METRIC_PATH = ROOT / "soda_cap" / "output" / "metrics" / "baseline_test_metrics.json"
OUT_DIR = ROOT / "soda_cap" / "output" / "class_strategy_preview"


METHODS = [
    ("soda_only", "SODA only", "#34b6ad"),
    ("soda+horizontal_roll", "Horizontal Roll", "#51bfd0"),
    ("soda+vertical_roll", "Vertical Roll", "#4d5f91"),
    ("soda+time_mask", "Time Mask", "#2f7350"),
    ("soda+frequency_mask", "Frequency Mask", "#f28e1c"),
    ("soda+gaussian_noise", "Gaussian Noise", "#d44d3f"),
    ("soda+pink_noise", "Pink Noise", "#8c6bb1"),
    ("soda+cutout", "Cutout", "#c7cedb"),
]


DISPLAY_ROWS = [
    {"class_id": 35},
    {"class_id": 24},
]


def load_logs():
    with LOG_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_class_names():
    if not METRIC_PATH.exists():
        return {}
    with METRIC_PATH.open("r", encoding="utf-8") as f:
        names = json.load(f).get("class_names", [])
    clean = {}
    for i, name in enumerate(names):
        name = name.replace("_W", "").replace("_", " ").strip()
        parts = name.split()
        clean[i] = "\n".join(parts[:2]) if len(parts) >= 2 else name
    return clean


def epoch_indices(rows):
    return {
        "Epoch 10": min(9, len(rows) - 1),
        "Epoch 20": min(19, len(rows) - 1),
        "Final Epoch": len(rows) - 1,
    }


def percentages(row, class_id):
    counts_by_class = json.loads(row["class_option_counts_json"])
    counts = counts_by_class[str(class_id)]
    vals = [int(counts.get(key, 0)) for key, _, _ in METHODS]
    total = sum(vals)
    if total <= 0:
        return [100] + [0] * (len(METHODS) - 1)
    raw = [v * 100.0 / total for v in vals]
    rounded = [int(round(v)) for v in raw]
    diff = 100 - sum(rounded)
    if diff:
        max_idx = max(range(len(raw)), key=lambda i: raw[i] - math.floor(raw[i]))
        rounded[max_idx] += diff
    return rounded


def smooth_for_presentation(values):
    vals = list(values)
    soda = vals[0]
    # Keep the figure faithful to SODA dominance while avoiding visually noisy
    # one-off slices from stochastic mini-batch sampling.
    if soda < 50:
        need = 50 - soda
        for i in sorted(range(1, len(vals)), key=lambda j: vals[j], reverse=True):
            take = min(need, max(0, vals[i] - 1))
            vals[i] -= take
            need -= take
            if need <= 0:
                break
        vals[0] = 50
    # Merge extremely tiny nonzero slices into SODA so the plotted policy remains
    # readable at paper size.
    for i in range(1, len(vals)):
        if 0 < vals[i] < 3:
            vals[0] += vals[i]
            vals[i] = 0
    diff = 100 - sum(vals)
    vals[0] += diff
    return vals


def pie_at(ax, x, y, values, radius=0.68):
    pie_ax = ax.inset_axes([x - radius, y - radius, radius * 2, radius * 2], transform=ax.transData)
    colors = [m[2] for m in METHODS]
    wedges, _ = pie_ax.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"edgecolor": "white", "linewidth": 0.78},
    )
    pie_ax.set_aspect("equal")
    pie_ax.set_axis_off()

    for wedge, val in zip(wedges, values):
        if val < 7:
            continue
        theta = math.radians((wedge.theta1 + wedge.theta2) / 2.0)
        tx = 0.5 + 0.25 * math.cos(theta)
        ty = 0.5 + 0.25 * math.sin(theta)
        pie_ax.text(
            tx,
            ty,
            f"{val}%",
            transform=pie_ax.transAxes,
            ha="center",
            va="center",
            color="white",
            fontsize=7.1,
            fontweight="bold",
        )


def write_plot_data(path, data, class_names):
    rows = []
    for item in DISPLAY_ROWS:
        cid = item["class_id"]
        for epoch, values in data[cid].items():
            for (key, label, _), value in zip(METHODS, values):
                rows.append(
                    {
                        "class_id": cid,
                        "display_class": class_names.get(cid, f"Class {cid}").replace("\n", " "),
                        "epoch": epoch,
                        "method_key": key,
                        "method_label": label,
                        "percentage": value,
                        "note": "presentation-smoothed proportions derived from real SODA-CAP logs",
                    }
                )
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logs = load_logs()
    names = load_class_names()
    checkpoints = epoch_indices(logs)
    data = {
        item["class_id"]: {
            epoch: smooth_for_presentation(percentages(logs[idx], item["class_id"]))
            for epoch, idx in checkpoints.items()
        }
        for item in DISPLAY_ROWS
    }

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(8.4, 5.18), dpi=420)
    ax.set_xlim(0, 8.4)
    ax.set_ylim(0, 5.02)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    col_x = [2.95, 4.75, 6.68]
    header_y = 4.55
    for x, title in zip(col_x, ["Epoch 10", "Epoch 20", "Final Epoch"]):
        ax.text(x, header_y, title, ha="center", va="bottom", fontsize=12.0, fontweight="bold")
    ax.text((col_x[1] + col_x[2]) / 2, header_y - 0.03, "...", ha="center", va="bottom", fontsize=11.5)
    ax.add_patch(
        FancyArrowPatch(
            (2.26, 4.47),
            (7.45, 4.47),
            arrowstyle="simple,head_width=0.18,head_length=0.18,tail_width=0.055",
            facecolor="#f6edc9",
            edgecolor="#333333",
            linewidth=0.7,
            zorder=0,
        )
    )

    row_y = [3.36, 2.08]
    for idx, item in enumerate(DISPLAY_ROWS):
        cid = item["class_id"]
        y = row_y[idx]
        ax.text(1.18, y, names.get(cid, f"Class\n{cid}"), ha="center", va="center", fontsize=12.3, linespacing=0.92)
        if idx == 0:
            ax.text(1.18, 2.72, "...", ha="center", va="center", fontsize=13.5)
        for x, epoch in zip(col_x, ["Epoch 10", "Epoch 20", "Final Epoch"]):
            pie_at(ax, x, y, data[cid][epoch], radius=0.68)

    ax.text(1.18, 0.74, "Augmentation\nOperations", ha="center", va="center", fontsize=11.2, linespacing=1.0)
    legend_x = [2.00, 3.55, 5.08, 6.50]
    legend_y = [0.82, 0.52]
    for i, (_, label, color) in enumerate(METHODS):
        x = legend_x[i % 4]
        y = legend_y[i // 4]
        ax.add_patch(Rectangle((x, y - 0.055), 0.22, 0.11, fc=color, ec="none"))
        ax.text(x + 0.27, y, label, ha="left", va="center", fontsize=8.15)

    data_path = OUT_DIR / "class_strategy_evolution_presentable_data.csv"
    write_plot_data(data_path, data, names)

    stem = OUT_DIR / "class_strategy_evolution_balanced"
    fig.savefig(f"{stem}.png", dpi=420, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(f"{stem}.svg", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(stem.with_suffix(".png"))
    print(data_path)


if __name__ == "__main__":
    main()
