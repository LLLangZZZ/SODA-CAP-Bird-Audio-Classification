# SODA-CAP Bird Audio Classification

This repository is a cleaned method-level release for the SODA-CAP bird audio classification paper. It is intended to document the core model and augmentation policy design, not to provide a complete reproduction package with data, trained weights, and cached predictions.

The method has two main components:

- **SODA**: a source-oriented spectrogram augmentation method. It estimates bird-dominant structure and noise-dominant residual content from log-Mel features, perturbs the residual noise style and strength, and preserves species-discriminative acoustic structure.
- **SODA-CAP**: a SODA-centered class-adaptive complementary augmentation policy. It uses cached validation predictions to score auxiliary augmentations with `rescue - lambda * damage + beta * gain`, then assigns class-wise probabilities to SODA-only and SODA-plus-auxiliary options.

## Included Files

- `models/ast_model.py`: the AST spectrogram classifier.
- `augmentations.py`: SODA and the seven auxiliary augmentation candidates used by the paper.
- `soda_cap/build_policy.py`: policy construction from cached validation predictions.
- `soda_cap/augment.py`: training-time class-wise policy sampling.
- `soda_cap/plot_policy_distribution.py`: policy distribution visualization.
- `config/config.py`: core hyperparameters matching the paper setup.

## Removed From This Release

- Trained models, pretrained weights, and large cache files.
- Dataset cleaning, quality filtering, split generation, and offline augmentation scripts.
- Historical exploratory code that is not part of the final SODA-CAP method.
- Local absolute paths and temporary output files.

## Repository Layout

```text
.
|-- augmentations.py
|-- config/
|   |-- __init__.py
|   `-- config.py
|-- models/
|   |-- __init__.py
|   `-- ast_model.py
`-- soda_cap/
    |-- __init__.py
    |-- augment.py
    |-- build_policy.py
    |-- plot_policy_distribution.py
    `-- utils.py
```

## Usage Notes

This release does not include the original dataset, model weights, or prediction caches. To reproduce experiments, prepare a 36-class bird audio dataset, train baseline/SODA/single-augmentation models, and generate cached prediction files such as `baseline.npz`, `soda.npz`, and one `.npz` file for each auxiliary augmentation.

Build a policy:

```bash
python -m soda_cap.build_policy --cache-dir soda_cap/output/pred_cache/val
```

Visualize selected class policies:

```bash
python -m soda_cap.plot_policy_distribution --classes 0 1 2 3
```
