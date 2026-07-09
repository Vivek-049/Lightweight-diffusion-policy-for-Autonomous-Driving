# Lightweight Diffusion Policy for Autonomous Driving

This repository contains a compact vision-language-action (VLA) diffusion policy for autonomous driving experiments. The policy conditions on a front-view image and a language/intent label, then predicts a short chunk of driving actions.

## Demo

The repository includes an inference demo video:

[inference_vla.mp4](./inference_vla.mp4)

## What is Included

- `train_vla_diffusion.py` - main PyTorch training script for the language-conditioned diffusion policy.
- `train_vla_diffusion.ipynb`, `train_vla_car.ipynb`, `train_colab.ipynb` - notebook versions and experiments.
- `training-data/` - JSON training samples with base64-encoded images, language IDs, timestamps, and driving actions.
- `training-data.zip` - zipped copy of the training data.
- `diffusion_policy.json`, `model.json`, `Trained model for forked road.json` - exported model/policy artifacts.
- `inference_vla.mp4` - inference rollout/demo video.

## Model Overview

The diffusion policy uses:

- 128 x 128 RGB image observations.
- Language/intent conditioning through an embedding layer.
- A lightweight CNN vision encoder.
- Transformer encoder/decoder blocks for multimodal conditioning.
- Diffusion over action chunks with DDIM sampling for faster inference.
- Action dimension of 4: forward, backward, left, and right.
- Action chunk size of 5 future steps.

## Training

Install the core Python dependencies:

```bash
pip install torch numpy pillow
```

Run training from the repository root:

```bash
python train_vla_diffusion.py
```

The script loads JSON files from `./training-data`, trains in two phases, and exports the learned policy.

## Dataset Format

Each training sample stores:

- an image observation,
- a `language_id` intent label,
- a timestamp,
- driving actions for `forward`, `backward`, `left`, and `right`.

The training script deduplicates samples by timestamp, groups samples by intent, and builds sliding action chunks for diffusion training.

## Notes

This project is intended as a lightweight research/prototype implementation for VLA-style autonomous driving policy learning.
