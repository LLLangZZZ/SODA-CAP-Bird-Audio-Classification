"""Build the SODA-CAP class-adaptive policy from cached predictions.

Each cache is an `.npz` file containing at least `labels` and `pred`. The
required caches are `baseline.npz` and `soda.npz`; auxiliary caches are optional
but at least one must be available. The class-wise score follows the paper:

Score(c, a) = rescue(c, a) - lambda * damage(c, a) + beta * gain(c, a)
"""

import argparse
import csv
import os

import numpy as np

from config.config import CONFIG
from .utils import (
    BASELINE_NAME,
    DEFAULT_AUX_CANDIDATES,
    DEFAULT_OUTPUT_DIR,
    SODA_NAME,
    ensure_dir,
    option_names,
    save_json,
)


def load_cache(cache_dir: str, name: str):
    """Load cached predictions for one model or augmentation setting."""

    path = os.path.join(cache_dir, f"{name}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing prediction cache: {path}")
    return np.load(path)


def per_class_acc(labels, preds, num_classes):
    """Compute per-class accuracy and sample count."""

    acc = np.zeros(num_classes, dtype=np.float64)
    total = np.zeros(num_classes, dtype=np.float64)
    for class_id in range(num_classes):
        mask = labels == class_id
        total[class_id] = mask.sum()
        acc[class_id] = (preds[mask] == class_id).mean() if total[class_id] > 0 else 0.0
    return acc, total


def softmax_positive(scores, temperature):
    """Apply softmax only to positive scores."""

    scores = np.maximum(scores, 0.0)
    if scores.sum() <= 1e-12:
        return np.ones_like(scores) / len(scores)
    scaled = scores / max(temperature, 1e-6)
    scaled = scaled - scaled.max()
    exps = np.exp(scaled) * (scores > 0)
    return exps / exps.sum() if exps.sum() > 1e-12 else scores / scores.sum()


def compute_class_rho(
    mode: str,
    base_rho: float,
    rho_min: float,
    rho_max: float,
    soda_gain: float,
    best_score: float,
    strength: float,
    max_soda_gain: float = 1.0,
    max_best_score: float = 1.0,
) -> float:
    """Compute the SODA-only probability for one class."""

    if mode == "fixed":
        return float(np.clip(base_rho, rho_min, rho_max))
    if mode != "adaptive":
        raise ValueError(f"Unsupported rho mode: {mode}")

    norm_gain = soda_gain / max(max_soda_gain, 1e-8)
    norm_score = max(best_score, 0.0) / max(max_best_score, 1e-8)
    raw = base_rho + strength * norm_gain - strength * norm_score
    return float(np.clip(raw, rho_min, rho_max))


def main():
    parser = argparse.ArgumentParser(description="Build the SODA-CAP class-adaptive augmentation policy.")
    parser.add_argument("--cache-dir", default=os.path.join(DEFAULT_OUTPUT_DIR, "pred_cache", "val"))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--aux", nargs="*", default=DEFAULT_AUX_CANDIDATES)
    parser.add_argument("--rho", type=float, default=0.60, help="Base SODA-only probability.")
    parser.add_argument("--rho-mode", choices=["fixed", "adaptive"], default="adaptive")
    parser.add_argument("--rho-min", type=float, default=0.50)
    parser.add_argument("--rho-max", type=float, default=0.85)
    parser.add_argument("--rho-strength", type=float, default=0.50)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--lambda-risk", type=float, default=0.75)
    parser.add_argument("--gain-weight", type=float, default=0.50)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--min-score", type=float, default=0.0)
    args = parser.parse_args()

    out_dir = ensure_dir(args.output_dir)
    baseline = load_cache(args.cache_dir, BASELINE_NAME)
    soda = load_cache(args.cache_dir, SODA_NAME)
    labels = baseline["labels"]
    if not np.array_equal(labels, soda["labels"]):
        raise ValueError("Baseline and SODA caches have different sample orders.")

    num_classes = CONFIG["num_classes"]
    base_acc, class_total = per_class_acc(labels, baseline["pred"], num_classes)
    soda_acc, _ = per_class_acc(labels, soda["pred"], num_classes)

    aux_names = []
    aux_cache = {}
    for name in args.aux:
        path = os.path.join(args.cache_dir, f"{name}.npz")
        if not os.path.exists(path):
            print(f"[skip] missing auxiliary cache: {name}")
            continue
        data = load_cache(args.cache_dir, name)
        if not np.array_equal(labels, data["labels"]):
            raise ValueError(f"{name} cache has a different sample order.")
        aux_names.append(name)
        aux_cache[name] = data
    if not aux_names:
        raise ValueError("No auxiliary prediction caches are available.")

    options = option_names(aux_names)
    policy = np.zeros((num_classes, len(options)), dtype=np.float64)
    score_rows = []

    per_class_raw = []
    soda_gains = []
    best_scores = []
    for class_id in range(num_classes):
        mask = labels == class_id
        soda_correct = soda["pred"][mask] == class_id
        raw_scores = []
        for aux_name in aux_names:
            aux_correct = aux_cache[aux_name]["pred"][mask] == class_id
            aux_acc = aux_correct.mean() if mask.sum() else 0.0
            soda_wrong = ~soda_correct
            rescue = (aux_correct & soda_wrong).sum() / max(soda_wrong.sum(), 1)
            damage = ((~aux_correct) & soda_correct).sum() / max(soda_correct.sum(), 1)
            gain = aux_acc - base_acc[class_id]
            raw_scores.append(rescue - args.lambda_risk * damage + args.gain_weight * gain)

        raw_scores = np.asarray(raw_scores, dtype=np.float64)
        positive = np.where(raw_scores > args.min_score)[0]
        selected = (
            positive[np.argsort(raw_scores[positive])[::-1]][: max(1, args.top_k)]
            if len(positive)
            else np.array([], dtype=np.int64)
        )
        best_score = raw_scores[selected[0]] if len(selected) else 0.0
        soda_gain = soda_acc[class_id] - base_acc[class_id]
        per_class_raw.append((raw_scores, selected, soda_gain, best_score))
        soda_gains.append(soda_gain)
        best_scores.append(best_score)

    for class_id, (raw_scores, selected, soda_gain, best_score) in enumerate(per_class_raw):
        rho_c = compute_class_rho(
            args.rho_mode,
            args.rho,
            args.rho_min,
            args.rho_max,
            soda_gain,
            best_score,
            args.rho_strength,
            max_soda_gain=max(max(soda_gains), 1e-8),
            max_best_score=max(max(best_scores), 1e-8),
        )
        policy[class_id, 0] = rho_c
        if len(selected):
            aux_probs = softmax_positive(raw_scores[selected], args.temperature)
            for idx, prob in zip(selected, aux_probs):
                policy[class_id, idx + 1] = (1.0 - rho_c) * prob
        else:
            policy[class_id, 0] = 1.0
            rho_c = 1.0

        mask = labels == class_id
        soda_correct = soda["pred"][mask] == class_id
        for aux_idx, aux_name in enumerate(aux_names):
            aux_correct = aux_cache[aux_name]["pred"][mask] == class_id
            aux_acc = aux_correct.mean() if mask.sum() else 0.0
            soda_wrong = ~soda_correct
            rescue = (aux_correct & soda_wrong).sum() / max(soda_wrong.sum(), 1)
            damage = ((~aux_correct) & soda_correct).sum() / max(soda_correct.sum(), 1)
            gain = aux_acc - base_acc[class_id]
            score_rows.append(
                {
                    "class": class_id,
                    "class_total": int(class_total[class_id]),
                    "aux": aux_name,
                    "baseline_acc": base_acc[class_id],
                    "soda_acc": soda_acc[class_id],
                    "aux_acc": aux_acc,
                    "rescue": rescue,
                    "damage": damage,
                    "gain": gain,
                    "score": raw_scores[aux_idx],
                    "selected": bool(aux_idx in set(selected.tolist())),
                    "rho_c": rho_c,
                    "probability": policy[class_id, aux_idx + 1],
                }
            )

    np.save(os.path.join(out_dir, "soda_cap_policy.npy"), policy)
    save_json(
        os.path.join(out_dir, "soda_cap_policy.json"),
        {
            "options": options,
            "aux_names": aux_names,
            "policy_semantics": "SODA-centered policy with auxiliary probabilities assigned by rescue-damage-gain complementarity.",
            "rho": args.rho,
            "rho_mode": args.rho_mode,
            "rho_min": args.rho_min,
            "rho_max": args.rho_max,
            "top_k": args.top_k,
            "lambda_risk": args.lambda_risk,
            "gain_weight": args.gain_weight,
            "temperature": args.temperature,
            "rho_per_class": policy[:, 0].tolist(),
        },
    )

    csv_path = os.path.join(out_dir, "soda_cap_scores.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(score_rows[0].keys()))
        writer.writeheader()
        writer.writerows(score_rows)

    print(f"Policy matrix saved: {os.path.join(out_dir, 'soda_cap_policy.npy')}")
    print(f"Score table saved: {csv_path}")


if __name__ == "__main__":
    main()
