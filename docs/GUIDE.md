# WARNING: template code, may need edits
# Complete Guide: Cat Image Classifier

## Table of Contents
1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Dataset Preparation](#dataset-preparation)
5. [Training Your Model](#training-your-model)
6. [Testing Your Model](#testing-your-model)
7. [Running Inference](#running-inference)
8. [Troubleshooting](#troubleshooting)
9. [Tips for Better Results]

---

## Introduction

Welcome! This guide will help you train a deep learning model to recognize cat images. Don't worry if you're new to this - we've made it as simple as possible!

### What You'll Learn
- How to prepare your image dataset
- How to train a neural network
- How to test and use your trained model
- How to make predictions on new images

---

## System Requirements

### Minimum Requirements
- **GPU**: NVIDIA GPU with 4GB VRAM (e.g., GTX 1650)
- **RAM**: 32GB system memory
- **Storage**: 5GB free space
- **OS**: Windows 10/11, Linux, or macOS

### Software Requirements
- Python 3.8 or newer
- CUDA 11.0 or newer (for GPU support)
- pip (Python package manager)

### Checking Your GPU

**Windows:**
```bash
nvidia-smi
```

**Linux:**
```bash
lspci | grep -i nvidia
```

If you see your GPU information, you're good to go!

---

## Installation

### Step 1: Download the Project
```bash
git clone <repository-url>
cd COMPLEX_TEST_1
```

### Step 2: Create a Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

This will install:
- PyTorch (deep learning framework)
- torchvision (image processing)
- Other helper libraries

### Step 4: Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

You should see:
- PyTorch version
- `CUDA Available: True` (if you have a GPU)

---

## Dataset Preparation

### Understanding the Data Structure

Your images need to be organized like this:
```
data/
├── train/
│   ├── cat/           <- Put cat images here
│   └── not_cat/       <- Put non-cat images here
└── val/
    ├── cat/           <- Validation cat images
    └── not_cat/       <- Validation non-cat images
```

### Step 1: Create Folders
```bash
python src/utils.py
```

This creates all necessary folders.

### Step 2: Collect Images

**How many images do you need?**
- **Minimum**: 500 images per class (1000 total)
- **Recommended**: 2000+ images per class (4000+ total)
- **More is better!**

**Where to get images?**
- Your own photos
- Google Images (ensure you have rights to use them)
- Kaggle datasets
- Public image datasets

### Step 3: Organize Your Images

1. Put 80% of cat images in `data/train/cat/`
2. Put 20% of cat images in `data/val/cat/`
3. Put 80% of non-cat images in `data/train/not_cat/`
4. Put 20% of non-cat images in `data/val/not_cat/`

**Tip:** You can use our automatic split tool:
```bash
python -c "from src.utils import split_dataset; split_dataset('path/to/your/cat/images')"
```

### Step 4: Verify Your Dataset
```bash
python -c "from src.utils import check_dataset_balance; check_dataset_balance()"
```

This shows how many images you have in each category.

### Image Requirements
- **Format**: JPG, JPEG, or PNG
- **Size**: Any size (will be resized automatically)
- **Quality**: Clear, well-lit images work best
- **Variety**: Different angles, backgrounds, and lighting

---

## Training Your Model

### Understanding Training

Training is like teaching the model to recognize cats. The model:
1. Looks at thousands of images
2. Learns patterns that distinguish cats from non-cats
3. Gets better over time

### Step 1: Start Training
```bash
python src/train.py
```

### What You'll See

The training will show:
```
Epoch 1/30 [Train]: 100%|████████| 50/50 [00:45<00:00]
  loss: 0.523, acc: 75.3%
Epoch 1/30 [Val]: 100%|████████| 13/13 [00:08<00:00]
  loss: 0.412, acc: 82.1%
```

**What do these mean?**
- **Epoch**: One complete pass through all training images
- **Loss**: How wrong the model is (lower is better)
- **Acc**: Accuracy percentage (higher is better)
- **Train**: Training set performance
- **Val**: Validation set performance (unseen images)

### Step 2: Monitor Progress

The training will:
- Save the best model automatically
- Create checkpoints every 5 epochs
- Log progress to TensorBoard

**View training graphs:**
```bash
tensorboard --logdir=runs
```
Then open http://localhost:6006 in your browser.

### Step 3: Wait for Completion

**How long does it take?**
- With 4GB GPU: ~2-3 hours
- With 8GB GPU: ~1-2 hours
- Depends on dataset size

**When is it done?**
- After 30 epochs (default)
- Or when validation accuracy stops improving
- You'll see "TRAINING COMPLETED!"

### Saved Models

After training:
- `models/best_model.pth` - Best performing model
- `models/last_model.pth` - Final model

---

## Testing Your Model

### Why Test?

Testing shows how well your model works on images it has never seen before.

### Run Testing
```bash
python src/test.py
```

### Understanding Results

You'll see:
```
==========================================
TEST RESULTS
==========================================
Average Loss: 0.3245
Accuracy: 89.50%
Correct: 358/400
==========================================

Classification Report:
              precision    recall  f1-score   support

     not_cat     0.8923    0.8650    0.8785       200
         cat     0.8978    0.9250    0.9112       200

    accuracy                         0.8950       400
```

**What do these mean?**
- **Accuracy**: Overall correctness (aim for >85%)
- **Precision**: When it says "cat", how often is it right?
- **Recall**: Of all actual cats, how many did it find?
- **F1-score**: Balance between precision and recall

### Confusion Matrix

A visualization showing:
- True Positives: Correctly identified cats
- True Negatives: Correctly identified non-cats
- False Positives: Non-cats classified as cats
- False Negatives: Cats classified as non-cats

Saved to: `outputs/confusion_matrix.png`

---

## Running Inference

### Single Image Prediction

Predict if an image contains a cat:
```bash
python src/inference.py --image path/to/your/image.jpg
```

**Output:**
```
==================================================
PREDICTION RESULTS
==================================================
Image: my_cat.jpg
Prediction: cat
Confidence: 96.78%
==================================================

Class Probabilities:
  not_cat: 3.22%
  cat: 96.78%

Visualization saved to outputs/my_cat_prediction.png
```

### Batch Prediction

Predict multiple images at once:
```bash
python src/inference.py --folder path/to/image/folder
```

### Using a Different Model

Use the last checkpoint instead of best:
```bash
python src/inference.py --image my_image.jpg --model models/last_model.pth
```

### Disable Visualization

For faster processing:
```bash
python src/inference.py --image my_image.jpg --no-viz
```

---

## Troubleshooting

### Common Issues

#### 1. "CUDA out of memory"

**Solution:** Reduce batch size in `src/config.py`:
```python
BATCH_SIZE = 16  # Try 16 or 8 instead of 32
```

#### 2. "No images found in data/train"

**Solution:** Make sure images are in the correct folders:
```
data/train/cat/
data/train/not_cat/
```

#### 3. Training is very slow

**Possible causes:**
- GPU not being used (check with `nvidia-smi`)
- Too many images
- Slow hard drive

**Solutions:**
- Ensure CUDA is installed
- Reduce `NUM_WORKERS` in config
- Use SSD instead of HDD

#### 4. Low accuracy (<70%)

**Possible causes:**
- Not enough training data
- Poor quality images
- Imbalanced dataset

**Solutions:**
- Collect more images (aim for 2000+ per class)
- Remove blurry/unclear images
- Ensure equal numbers of cat and non-cat images
- Train for more epochs

#### 5. "ModuleNotFoundError"

**Solution:** Install missing package:
```bash
pip install <package-name>
```

---

## Tips for Better Results

### 1. Dataset Quality

✅ **Good:**
- Clear, well-lit images
- Various angles and positions
- Different cat breeds
- Various backgrounds
- Mix of indoor/outdoor

❌ **Avoid:**
- Blurry images
- Very dark/bright images
- Watermarked images
- Duplicate images

### 2. Data Augmentation

Enabled by default! This creates variations of your images:
- Random flips
- Rotations
- Color adjustments

Keeps your model from memorizing specific images.

### 3. Training Tips

- **Start small**: Test with a small dataset first
- **Monitor validation loss**: Should decrease over time
- **Early stopping**: Stop if validation loss increases for several epochs
- **Save checkpoints**: Don't lose progress if training crashes

### 4. Inference Tips

- Use the **best_model.pth** for production
- Test on diverse images
- If confidence is low (<60%), the model is uncertain
- Consider retraining if many predictions are wrong

### 5. Improving Accuracy

**If accuracy is 70-80%:**
- Add more training images
- Train for more epochs (50-100)
- Ensure dataset is balanced

**If accuracy is 80-90%:**
- Good! Minor improvements:
  - Fine-tune learning rate
  - Add more diverse images
  - Clean up mislabeled images

**If accuracy is >90%:**
- Excellent! You're done!
- Test on real-world images
- Deploy your model

---

## Advanced Configuration

Edit `src/config.py` to customize:

```python
# Training
BATCH_SIZE = 32          # Reduce if out of memory
NUM_EPOCHS = 30          # Increase for better accuracy
LEARNING_RATE = 0.001    # Lower for fine-tuning

# Model
INPUT_SIZE = 224         # Image size (don't change)
MODEL_NAME = 'efficientnet_b0'  # Lightweight model

# Data
USE_AUGMENTATION = True  # Enable data augmentation
```

---

## Getting Help

### Check Logs

All training logs are in the `runs/` folder.

### Common Commands

```bash
# Check dataset
python -c "from src.utils import check_dataset_balance; check_dataset_balance()"

# Visualize samples
python -c "from src.utils import visualize_samples; visualize_samples('data/train')"

# Test model
python src/test.py

# Quick inference
python src/inference.py --image test.jpg
```

---

## Next Steps

After successful training:

1. **Test thoroughly**: Try many different images
2. **Deploy**: Use the model in your application
3. **Collect feedback**: Find where it fails
4. **Retrain**: Add more data and retrain
5. **Iterate**: Keep improving!

---

## Summary

### Quick Reference

```bash
# 1. Install
pip install -r requirements.txt

# 2. Prepare data
python src/utils.py
# Then add images to data/train and data/val

# 3. Train
python src/train.py

# 4. Test
python src/test.py

# 5. Predict
python src/inference.py --image your_image.jpg
```

**That's it! You're now ready to train your cat classifier!** F431

Good luck, and have fun! F680
