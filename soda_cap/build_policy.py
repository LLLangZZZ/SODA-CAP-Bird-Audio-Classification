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
    path = os.path.join(cache_dir, f"{name}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing prediction cache for {name}: {path}")
    return np.load(path)


def per_class_acc(labels, preds, num_classes):
    acc = np.zeros(num_classes, dtype=np.float64)
    total = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        mask = labels == c
        total[c] = mask.sum()
        acc[c] = (preds[mask] == c).mean() if total[c] > 0 else 0.0
    return acc, total


def softmax_positive(scores, temperature):
    scores = np.maximum(scores, 0.0)
    if scores.sum() <= 1e-12:
        return np.ones_like(scores) / len(scores)
    scaled = scores / max(temperature, 1e-6)
    scaled = scaled - scaled.max()
    exps = np.exp(scaled) * (scores > 0)
    if exps.sum() <= 1e-12:
        return scores / scores.sum()
    return exps / exps.sum()


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
    if mode == "fixed":
        return float(np.clip(base_rho, rho_min, rho_max))
    if mode != "adaptive":
        raise ValueError(f"Unsupported rho mode: {mode}")

    # Normalize soda_gain and best_score to [0, 1] so they are comparable and
    # the strength parameter has predictable effect.  Without normalization,
    # best_score (which can exceed 1.0) dominates and pushes rho to rho_min
    # for almost every class.
    norm_gain = soda_gain / max(max_soda_gain, 1e-8)
    norm_score = max(best_score, 0.0) / max(max_best_score, 1e-8)
    raw = base_rho + strength * norm_gain - strength * norm_score
    return float(np.clip(raw, rho_min, rho_max))


def compute_class_rho_raw(
    mode: str,
    base_rho: float,
    soda_gain: float,
    best_score: float,
    strength: float,
) -> float:
    if mode == "fixed":
        return float(base_rho)
    if mode != "adaptive":
        raise ValueError(f"Unsupported rho mode: {mode}")
    return float(base_rho + strength * soda_gain - strength * max(best_score, 0.0))


def main():
    parser = argparse.ArgumentParser(description="Build a SODA-centered class-wise complementary policy.")
    parser.add_argument("--cache-dir", default=os.path.join(DEFAULT_OUTPUT_DIR, "pred_cache", "val"))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--aux", nargs="*", default=DEFAULT_AUX_CANDIDATES)
    parser.add_argument("--rho", type=float, default=0.60, help="Base probability of SODA-only.")
    parser.add_argument("--rho-mode", choices=["fixed", "adaptive"], default="adaptive")
    parser.add_argument("--rho-min", type=float, default=0.50)
    parser.add_argument("--rho-max", type=float, default=0.85)
    parser.add_argument("--rho-strength", type=float, default=0.50)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--lambda-risk", type=float, default=0.75)
    parser.add_argument("--gain-weight", type=float, default=0.50)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument(
        "--relative-score-threshold",
        type=float,
        default=0.0,
        help="Keep an aux only if its score >= best_score * threshold. Example 0.5 keeps only reasonably strong complements.",
    )
    args = parser.parse_args()

    out_dir = ensure_dir(args.output_dir)
    baseline = load_cache(args.cache_dir, BASELINE_NAME)
    soda = load_cache(args.cache_dir, SODA_NAME)
    labels = baseline["labels"]
    if not np.array_equal(labels, soda["labels"]):
        raise ValueError("Baseline and SODA caches have different labels/order.")

    num_classes = CONFIG["num_classes"]
    base_acc, class_total = per_class_acc(labels, baseline["pred"], num_classes)
    soda_acc, _ = per_class_acc(labels, soda["pred"], num_classes)

    aux_names = []
    aux_cache = {}
    for name in args.aux:
        path = os.path.join(args.cache_dir, f"{name}.npz")
        if os.path.exists(path):
            data = load_cache(args.cache_dir, name)
            if not np.array_equal(labels, data["labels"]):
                raise ValueError(f"{name} cache has different labels/order.")
            aux_names.append(name)
            aux_cache[name] = data
        else:
            print(f"[skip] missing aux cache: {name}")

    if not aux_names:
        raise ValueError("No auxiliary caches found. Run infer_unmix.py for at least one aux model.")

    options = option_names(aux_names)
    policy = np.zeros((num_classes, len(options)), dtype=np.float64)
    score_table = []

    # First pass: compute raw scores and collect global normalization stats.
    all_soda_gains = []
    all_best_scores = []
    per_class_raw = []
    for c in range(num_classes):
        class_mask = labels == c
        soda_correct = soda["pred"][class_mask] == c
        raw_scores = []
        for aux_name in aux_names:
            aux = aux_cache[aux_name]
            aux_pred_c = aux["pred"][class_mask]
            aux_correct = aux_pred_c == c
            aux_acc = aux_correct.mean() if class_mask.sum() else 0.0

            soda_wrong = ~soda_correct
            rescue = (aux_correct & soda_wrong).sum() / max(soda_wrong.sum(), 1)
            damage = ((~aux_correct) & soda_correct).sum() / max(soda_correct.sum(), 1)
            gain = aux_acc - base_acc[c]
            complementarity = rescue - args.lambda_risk * damage
            score = complementarity + args.gain_weight * gain
            raw_scores.append(score)

        raw_scores = np.asarray(raw_scores, dtype=np.float64)
        keep = raw_scores > args.min_score
        if keep.any():
            keep_indices = np.where(keep)[0]
            order = keep_indices[np.argsort(raw_scores[keep_indices])[::-1]]
            selected = order[: max(1, args.top_k)]
        else:
            selected = np.array([], dtype=np.int64)

        best_score = raw_scores[selected[0]] if len(selected) > 0 else 0.0
        if len(selected) > 0 and args.relative_score_threshold > 0:
            cutoff = best_score * args.relative_score_threshold
            selected = np.asarray([idx for idx in selected if raw_scores[idx] >= cutoff], dtype=np.int64)
        soda_gain = soda_acc[c] - base_acc[c]
        all_soda_gains.append(soda_gain)
        all_best_scores.append(best_score)
        per_class_raw.append((raw_scores, selected, soda_gain, best_score))

    max_soda_gain = max(all_soda_gains) if all_soda_gains else 1.0
    max_best_score = max(all_best_scores) if all_best_scores else 1.0

    # Second pass: build policy with normalized rho computation.
    for c, (raw_scores, selected, soda_gain, best_score) in enumerate(per_class_raw):
        class_mask = labels == c
        soda_correct = soda["pred"][class_mask] == c
        rows = []
        for i, aux_name in enumerate(aux_names):
            aux = aux_cache[aux_name]
            aux_pred_c = aux["pred"][class_mask]
            aux_correct = aux_pred_c == c
            aux_acc = aux_correct.mean() if class_mask.sum() else 0.0

            soda_wrong = ~soda_correct
            rescue = (aux_correct & soda_wrong).sum() / max(soda_wrong.sum(), 1)
            damage = ((~aux_correct) & soda_correct).sum() / max(soda_correct.sum(), 1)
            gain = aux_acc - base_acc[c]
            complementarity = rescue - args.lambda_risk * damage
            score = complementarity + args.gain_weight * gain
            rows.append(
                {
                    "class": c,
                    "class_total": int(class_total[c]),
                    "aux": aux_name,
                    "baseline_acc": base_acc[c],
                    "soda_acc": soda_acc[c],
                    "aux_acc": aux_acc,
                    "rescue": rescue,
                    "damage": damage,
                    "gain": gain,
                    "complementarity": complementarity,
                    "score": score,
                }
            )

        rho_raw_c = compute_class_rho_raw(
            args.rho_mode,
            args.rho,
            soda_gain,
            best_score,
            args.rho_strength,
        )
        rho_c = compute_class_rho(
            args.rho_mode,
            args.rho,
            args.rho_min,
            args.rho_max,
            soda_gain,
            best_score,
            args.rho_strength,
            max_soda_gain=max(max_soda_gain, 1e-8),
            max_best_score=max(max_best_score, 1e-8),
        )

        policy[c, 0] = rho_c
        if len(selected) > 0:
            probs = softmax_positive(raw_scores[selected], args.temperature)
            for idx, p in zip(selected, probs):
                policy[c, idx + 1] = (1.0 - rho_c) * p
        else:
            policy[c, 0] = 1.0
            rho_c = 1.0

        for i, row in enumerate(rows):
            row["selected"] = bool(i in set(selected.tolist()))
            row["rho_raw_c"] = rho_raw_c
            row["rho_c"] = rho_c
            row["aux_budget_c"] = 1.0 - rho_c
            row["probability"] = policy[c, i + 1]
            score_table.append(row)

    np.save(os.path.join(out_dir, "soda_cap_policy.npy"), policy)
    save_json(
        os.path.join(out_dir, "soda_cap_policy.json"),
        {
            "options": options,
            "aux_names": aux_names,
            "policy_semantics": "model_level_proxy",
            "default_excludes": [
                "bird_mixup",
                "spectrogram_mixup",
                "mixed_frequency_masking",
            ],
            "offline_pitch_shift_training_semantics": (
                "source replacement from matching offline *_ps_* wav, then SODA; "
                "fallback to SODA-only if no offline wav exists"
            ),
            "rho": args.rho,
            "rho_mode": args.rho_mode,
            "rho_min": args.rho_min,
            "rho_max": args.rho_max,
            "rho_strength": args.rho_strength,
            "rho_per_class": policy[:, 0].tolist(),
            "top_k": args.top_k,
            "lambda_risk": args.lambda_risk,
            "gain_weight": args.gain_weight,
            "temperature": args.temperature,
            "policy_path": os.path.join(out_dir, "soda_cap_policy.npy"),
        },
    )

    csv_path = os.path.join(out_dir, "soda_cap_scores.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(score_table[0].keys()))
        writer.writeheader()
        writer.writerows(score_table)

    txt_path = os.path.join(out_dir, "soda_cap_policy.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"SODA-CAP policy: {num_classes} classes x {len(options)} options\n")
        f.write(
            f"rho_mode={args.rho_mode}, base_rho={args.rho:.3f}, "
            f"rho_min={args.rho_min:.3f}, rho_max={args.rho_max:.3f}\n"
        )
        f.write(f"options: {', '.join(options)}\n")
        for c in range(num_classes):
            active = [(options[i], policy[c, i]) for i in np.argsort(policy[c])[::-1] if policy[c, i] > 1e-4]
            f.write(f"Class {c:02d}: " + ", ".join(f"{name}={prob:.3f}" for name, prob in active) + "\n")

    print(f"Policy saved: {os.path.join(out_dir, 'soda_cap_policy.npy')}")
    print(f"Readable policy: {txt_path}")
    print(f"Score table: {csv_path}")


if __name__ == "__main__":
    main()
