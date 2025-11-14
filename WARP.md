# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a PyTorch-based image classification project that uses the Oxford-IIIT Pet Dataset to classify pet breeds. The project implements a simple fully-connected neural network for training on 224x224 RGB images with 37 output classes.

## Commands

### Running the Project
```bash
python3 cats_dogs.py
```

This will:
- Download the Oxford-IIIT Pet dataset to `data/` (auto-downloaded on first run)
- Train the model for 5 epochs
- Print training loss and test accuracy after each epoch

### Dependencies
The project requires PyTorch and torchvision. These packages are inside the conda datascience environment:
```bash
conda activate datascience
```

## Architecture

### Model Structure
- **Input**: 224x224x3 RGB images (resized from original dataset)
- **Architecture**: Fully-connected neural network
  - Flatten layer
  - Linear(224×224×3 → 512) + ReLU
  - Linear(512 → 512) + ReLU  
  - Linear(512 → 37) output layer
- **Output**: 37 classes (Oxford-IIIT Pet breeds)

### Data Pipeline
- **Dataset**: `torchvision.datasets.OxfordIIITPet`
- **Splits**: 
  - Training: `split="trainval"` 
  - Testing: `split="test"`
- **Transforms**: Resize to (224, 224) + ToTensor()
- **Batch size**: 64
- **Storage**: Downloaded to `data/` directory (gitignored)

### Training Configuration
- **Loss function**: CrossEntropyLoss
- **Optimizer**: Adam (lr=1e-3)
- **Device**: Auto-detects CUDA if available, otherwise CPU
- **Epochs**: 5 (hardcoded in main execution)

## Known Issues

### Critical Bug
**Line 37**: The training dataloader is incorrectly initialized with `testing_data` instead of `training_data`:
```python
train_dataloader = DataLoader(testing_data, batch_size=batch_size)  # BUG
```
This means the model trains on the test set, making accuracy metrics invalid.

### Typo
**Line 7**: Comment has typo "crrect" should be "correct"

## Code Modification Guidelines

- The model uses a simple fully-connected architecture. For better performance on images, consider using CNNs (Conv2d layers).
- Device placement is handled automatically via the `device` variable.
- The dataset auto-downloads on first run, so `data/` directory will be created automatically.
- Loss is logged every 100 batches during training (line 90).
- Test evaluation uses `torch.no_grad()` for efficiency.
