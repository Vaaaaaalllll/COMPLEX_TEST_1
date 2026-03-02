# WARNING: template code, may need edits
# COMPLEX_TEST_1 - Cat Image Classifier

A simple, kid-friendly deep learning project for training and inferencing cat images using PyTorch.

## Features
- Train a custom cat image classifier from scratch
- Test model accuracy on validation data
- Run inference on new cat images
- Optimized for 4-8GB GPU RAM and 32GB system RAM
- No complex frameworks - pure PyTorch implementation

## Hardware Requirements
- GPU: 4-8GB VRAM (NVIDIA GPU recommended)
- RAM: 32GB system memory
- Storage: ~5GB for dataset and models

## Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Prepare Your Dataset
Organize your images:
```
data/
├── train/
│   ├── cat/
│   └── not_cat/
└── val/
    ├── cat/
    └── not_cat/
```

### 3. Train the Model
```bash
python src/train.py
```

### 4. Test the Model
```bash
python src/test.py
```

### 5. Run Inference
```bash
python src/inference.py --image path/to/your/cat.jpg
```

## Project Structure
```
COMPLEX_TEST_1/
├── src/              # Source code
├── data/             # Dataset directory
├── models/           # Saved models
├── docs/             # Documentation
├── examples/         # Example images
└── outputs/          # Inference results
```

## Documentation
See [docs/GUIDE.md](docs/GUIDE.md) for detailed instructions.

## Model Architecture
Using EfficientNet-B0 - a lightweight but powerful architecture perfect for limited GPU memory.

## License
MIT License