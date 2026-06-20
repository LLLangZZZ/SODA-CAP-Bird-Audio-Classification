"""Visualize SODA-CAP policy distributions.

The script reads `soda_cap_policy.json` and plots the SODA-only ratio against
the total auxiliary budget for selected classes. For full per-auxiliary
probabilities, load the corresponding `.npy` policy matrix.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


COLORS = {
    "soda_only": "#34b6ad",
    "auxiliary": "#f28e1c",
}


def load_policy(path: Path):
    """Load policy metadata from JSON."""

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["options"], payload["rho_per_class"]


def main():
    parser = argparse.ArgumentParser(description="Visualize SODA-CAP class-wise policy distributions.")
    parser.add_argument("--policy-json", default="soda_cap/output/soda_cap_policy.json")
    parser.add_argument("--classes", nargs="*", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--output", default="soda_cap/output/policy_distribution.png")
    args = parser.parse_args()

    policy_path = Path(args.policy_json)
    _, rho_per_class = load_policy(policy_path)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(args.classes), figsize=(3.0 * len(args.classes), 3.0), dpi=300)
    if len(args.classes) == 1:
        axes = [axes]

    for ax, class_id in zip(axes, args.classes):
        rho = float(rho_per_class[class_id])
        values = [rho, 1.0 - rho]
        labels = ["SODA only", "SODA + auxiliary"]
        colors = [COLORS["soda_only"], COLORS["auxiliary"]]
        ax.pie(values, labels=labels, colors=colors, autopct="%1.0f%%", startangle=90)
        ax.set_title(f"Class {class_id}")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
