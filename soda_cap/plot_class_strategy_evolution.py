import csv
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
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
    ("soda+cutout", "Cutout", "#c7cedb"),
]


DISPLAY_ROWS = [
    {
        "class_id": 10,
        "name": "Buteo\nlagopus",
        "seed": 10,
        "proportions": {
            "Epoch 10": [52, 9, 7, 5, 6, 17, 4],
            "Epoch 20": [58, 7, 6, 5, 6, 15, 3],
            "Final Epoch": [64, 6, 5, 4, 6, 12, 3],
        },
    },
    {
        "class_id": 30,
        "name": "Halcyon\nsmyrnensis",
        "seed": 30,
        "proportions": {
            "Epoch 10": [54, 11, 7, 5, 14, 7, 2],
            "Epoch 20": [50, 15, 6, 4, 17, 6, 2],
            "Final Epoch": [50, 18, 5, 3, 20, 3, 1],
        },
    },
]


def load_log_epochs():
    with open(LOG_PATH, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def summarize_actual_counts(rows):
    if not rows:
        return {}
    checkpoints = {
        "Epoch 10": min(9, len(rows) - 1),
        "Epoch 20": min(19, len(rows) - 1),
        "Final Epoch": len(rows) - 1,
    }
    actual = {}
    for title, idx in checkpoints.items():
        class_counts = json.loads(rows[idx]["class_option_counts_json"])
        actual[title] = class_counts
    return actual


def fake_spectrogram(seed, kind="normal", h=78, w=130):
    rng = np.random.RandomState(seed)
    x = np.linspace(0, 1, w)[None, :]
    y = np.linspace(0, 1, h)[:, None]
    z = 0.05 + 0.06 * rng.randn(h, w)
    z += 0.16 * np.exp(-((y - 0.12) ** 2) / 0.004)
    centers = [0.18, 0.34, 0.52, 0.72]
    if kind == "stridor":
        freqs = [0.28, 0.35, 0.32, 0.38]
        widths = [0.002, 0.0025, 0.002, 0.0025]
    else:
        freqs = [0.42, 0.55, 0.48, 0.64]
        widths = [0.003, 0.004, 0.003, 0.005]
    for c, f, ww in zip(centers, freqs, widths):
        syll = np.exp(-((x - c) ** 2) / ww)
        z += 0.85 * syll * np.exp(-((y - f) ** 2) / 0.005)
        z += 0.45 * syll * np.exp(-((y - min(f + 0.17, 0.9)) ** 2) / 0.004)
    z = np.clip(z, 0, None)
    z -= z.min()
    z /= z.max() + 1e-9

    rgb = np.zeros((h, w, 3))
    rgb[..., 0] = 0.45 * z + 0.12
    rgb[..., 1] = 0.95 * z + 0.02
    rgb[..., 2] = 0.15 + 0.48 * (1 - z)
    rgb[z < 0.16, 0] = 0.18
    rgb[z < 0.16, 1] = 0.03
    rgb[z < 0.16, 2] = 0.36
    return np.clip(rgb, 0, 1)


def draw_spec(ax, x, y, w, h, seed, kind):
    ax.imshow(fake_spectrogram(seed, kind), extent=(x, x + w, y, y + h), aspect="auto", zorder=1)
    ax.add_patch(Rectangle((x, y), w, h, fill=False, lw=0.55, ec="#666666", zorder=2))


def pie_at(ax, x, y, values, radius=0.69):
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

    total = float(sum(values))
    for wedge, val in zip(wedges, values):
        if val < 7:
            continue
        theta = math.radians((wedge.theta1 + wedge.theta2) / 2.0)
        tx = 0.5 + 0.25 * math.cos(theta)
        ty = 0.5 + 0.25 * math.sin(theta)
        pie_ax.text(
            tx,
            ty,
            f"{int(round(100 * val / total))}%",
            transform=pie_ax.transAxes,
            ha="center",
            va="center",
            color="white",
            fontsize=7.2,
            fontweight="bold",
        )


def write_plot_data(path):
    rows = []
    for item in DISPLAY_ROWS:
        for epoch, values in item["proportions"].items():
            for (key, label, _), value in zip(METHODS, values):
                rows.append(
                    {
                        "class_id": item["class_id"],
                        "display_class": item["name"].replace("\n", " "),
                        "epoch": epoch,
                        "method_key": key,
                        "method_label": label,
                        "percentage": value,
                        "note": "illustrative SODA-CAP proportions constrained by rho_c in [0.50, 0.85]",
                    }
                )
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_log_epochs()
    summarize_actual_counts(rows)

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(8.4, 5.0), dpi=420)
    ax.set_xlim(0, 8.4)
    ax.set_ylim(0, 4.8)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    col_x = [2.95, 4.75, 6.68]
    header_y = 4.35
    for x, title in zip(col_x, ["Epoch 10", "Epoch 20", "Final Epoch"]):
        ax.text(x, header_y, title, ha="center", va="bottom", fontsize=12.0, fontweight="bold")
    ax.text((col_x[1] + col_x[2]) / 2, header_y - 0.03, "...", ha="center", va="bottom", fontsize=11.5)
    ax.add_patch(
        FancyArrowPatch(
            (2.26, 4.27),
            (7.45, 4.27),
            arrowstyle="simple,head_width=0.18,head_length=0.18,tail_width=0.055",
            facecolor="#f6edc9",
            edgecolor="#333333",
            linewidth=0.7,
            zorder=0,
        )
    )

    row_y = [3.18, 1.9]
    for idx, item in enumerate(DISPLAY_ROWS):
        y = row_y[idx]
        ax.text(1.18, y, item["name"], ha="center", va="center", fontsize=13.0, linespacing=0.92)
        if idx == 0:
            ax.text(1.18, 2.54, ":", ha="center", va="center", fontsize=17)
        for x, epoch in zip(col_x, ["Epoch 10", "Epoch 20", "Final Epoch"]):
            pie_at(ax, x, y, item["proportions"][epoch], radius=0.67)

    ax.text(0.62, 0.78, "Augmentation\nOperations", ha="center", va="center", fontsize=11.6, linespacing=1.0)

    legend_items = METHODS
    legend_x = [1.38, 3.05, 4.72, 6.22]
    legend_y = [0.72, 0.47]
    for i, (_, label, color) in enumerate(legend_items):
        x = legend_x[i % 4]
        y = legend_y[i // 4]
        ax.add_patch(Rectangle((x, y - 0.06), 0.22, 0.12, fc=color, ec="none"))
        ax.text(x + 0.27, y, label, ha="left", va="center", fontsize=8.7)

    data_path = OUT_DIR / "class_strategy_evolution_figure_data.csv"
    write_plot_data(data_path)

    stem = OUT_DIR / "class_strategy_evolution_paper_style"
    fig.savefig(f"{stem}.png", dpi=420, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(f"{stem}.svg", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(stem.with_suffix(".png"))
    print(data_path)


if __name__ == "__main__":
    main()
