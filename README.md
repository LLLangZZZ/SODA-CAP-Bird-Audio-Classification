# SODA-CAP Bird Species Classification

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
