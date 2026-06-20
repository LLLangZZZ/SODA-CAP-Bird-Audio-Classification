import csv
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "soda_cap" / "output" / "soda_cap_epoch_logs.csv"
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
    {"class_id": 10, "name": "Buteo\nlagopus"},
    {"class_id": 30, "name": "Halcyon\nsmyrnensis"},
]


def load_logs():
    with LOG_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def epoch_indices(rows):
    return {
        "Epoch 10": min(9, len(rows) - 1),
        "Epoch 20": min(19, len(rows) - 1),
        "Final Epoch": len(rows) - 1,
    }


def percentages_for_class(row, class_id):
    counts_by_class = json.loads(row["class_option_counts_json"])
    counts = counts_by_class[str(class_id)]
    values = [int(counts.get(key, 0)) for key, _, _ in METHODS]
    total = sum(values)
    if total <= 0:
        return [100] + [0] * (len(METHODS) - 1)
    raw = [v * 100.0 / total for v in values]
    rounded = [int(round(v)) for v in raw]
    diff = 100 - sum(rounded)
    if diff:
        max_idx = max(range(len(raw)), key=lambda i: raw[i] - math.floor(raw[i]))
        rounded[max_idx] += diff
    return rounded


def pie_at(ax, x, y, values, radius=0.67):
    pie_ax = ax.inset_axes([x - radius, y - radius, radius * 2, radius * 2], transform=ax.transData)
    colors = [m[2] for m in METHODS]
    wedges, _ = pie_ax.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"edgecolor": "white", "linewidth": 0.75},
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
            fontsize=7.0,
            fontweight="bold",
        )


def write_plot_data(path, data):
    rows = []
    for item in DISPLAY_ROWS:
        class_id = item["class_id"]
        display = item["name"].replace("\n", " ")
        for epoch, values in data[class_id].items():
            for (key, label, _), value in zip(METHODS, values):
                rows.append(
                    {
                        "class_id": class_id,
                        "display_class": display,
                        "epoch": epoch,
                        "method_key": key,
                        "method_label": label,
                        "percentage": value,
                        "note": "real sampled proportions from soda_cap_epoch_logs.csv",
                    }
                )
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logs = load_logs()
    checkpoints = epoch_indices(logs)
    data = {
        item["class_id"]: {
            epoch: percentages_for_class(logs[idx], item["class_id"])
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

    fig, ax = plt.subplots(figsize=(8.4, 5.05), dpi=420)
    ax.set_xlim(0, 8.4)
    ax.set_ylim(0, 4.85)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    col_x = [2.95, 4.75, 6.68]
    header_y = 4.38
    for x, title in zip(col_x, ["Epoch 10", "Epoch 20", "Final Epoch"]):
        ax.text(x, header_y, title, ha="center", va="bottom", fontsize=12.0, fontweight="bold")
    ax.text((col_x[1] + col_x[2]) / 2, header_y - 0.03, "...", ha="center", va="bottom", fontsize=11.5)
    ax.add_patch(
        FancyArrowPatch(
            (2.26, 4.30),
            (7.45, 4.30),
            arrowstyle="simple,head_width=0.18,head_length=0.18,tail_width=0.055",
            facecolor="#f6edc9",
            edgecolor="#333333",
            linewidth=0.7,
            zorder=0,
        )
    )

    row_y = [3.20, 1.92]
    for idx, item in enumerate(DISPLAY_ROWS):
        y = row_y[idx]
        ax.text(1.18, y, item["name"], ha="center", va="center", fontsize=13.0, linespacing=0.92)
        if idx == 0:
            ax.text(1.18, 2.56, ":", ha="center", va="center", fontsize=17)
        for x, epoch in zip(col_x, ["Epoch 10", "Epoch 20", "Final Epoch"]):
            pie_at(ax, x, y, data[item["class_id"]][epoch], radius=0.67)

    ax.text(0.62, 0.77, "Augmentation\nOperations", ha="center", va="center", fontsize=11.3, linespacing=1.0)

    legend_x = [1.28, 2.95, 4.62, 6.24]
    legend_y = [0.78, 0.51]
    for i, (_, label, color) in enumerate(METHODS):
        x = legend_x[i % 4]
        y = legend_y[i // 4]
        ax.add_patch(Rectangle((x, y - 0.055), 0.22, 0.11, fc=color, ec="none"))
        ax.text(x + 0.27, y, label, ha="left", va="center", fontsize=8.35)

    data_path = OUT_DIR / "class_strategy_evolution_real_data.csv"
    write_plot_data(data_path, data)

    stem = OUT_DIR / "class_strategy_evolution_real"
    fig.savefig(f"{stem}.png", dpi=420, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(f"{stem}.svg", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(stem.with_suffix(".png"))
    print(data_path)


if __name__ == "__main__":
    main()
