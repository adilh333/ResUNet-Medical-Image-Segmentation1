# ResUNet-Medical-Image-Segmentation1
# Brain Tumour Detection and Segmentation using ResUNet

A deep learning pipeline for automated brain tumour detection and segmentation
from MRI scans, using a two-stage architecture: a ResNet-50 classification
network to detect the presence of a tumour, followed by a custom ResUNet
segmentation model to delineate its precise boundaries.

## Motivation

Standard U-Net architectures can lose fine-grained spatial detail during the
encoding phase, particularly for small or irregularly shaped tumours. This
project investigated whether replacing standard convolutional blocks with
residual blocks, adding shortcut connections that preserve feature maps across
depth, would improve boundary delineation on a constrained medical imaging
dataset.

## Dataset

The project uses the LGG MRI Segmentation dataset from Kaggle (TCGA Lower
Grade Glioma dataset), comprising 110 patients and 3,929 MRI scans with
corresponding expert-annotated binary tumour masks. Each scan is a 256x256
RGB image; the mask indicates the presence and location of tumour tissue.

- Training set: 3,339 images
- Validation set: approximately 590 images
- Positive cases (tumour present): subset with confirmed mask regions

## Architecture

The pipeline consists of two models trained independently:

### Stage 1: Classification (ResNet-50)
- Pre-trained ResNet-50 encoder with transfer learning from ImageNet weights
- Binary output: tumour present vs absent
- Trained for up to 50 epochs with early stopping (patience=5) and
  ReduceLROnPlateau
- Best validation accuracy: **95.6%** (epoch 17, val_loss: 0.160)

### Stage 2: Segmentation (ResUNet)
A custom encoder-decoder architecture combining residual blocks with U-Net
skip connections:

```
Input (256x256x3)
    │
    ├─ Stage 1: Conv2D(16) × 2 + BatchNorm → MaxPool
    ├─ Stage 2: ResBlock(32) → MaxPool
    ├─ Stage 3: ResBlock(64) → MaxPool
    ├─ Stage 4: ResBlock(128) → MaxPool
    │
    └─ Bottleneck: ResBlock(256)
         │
    ├─ Upsample + Skip(128) → ResBlock(128)
    ├─ Upsample + Skip(64)  → ResBlock(64)
    ├─ Upsample + Skip(32)  → ResBlock(32)
    ├─ Upsample + Skip(16)  → ResBlock(16)
    │
    └─ Conv2D(1, sigmoid) → Mask output
```

Total parameters: 1,210,513 (1,206,129 trainable)

Each residual block uses a shortcut path (1×1 convolution + BatchNorm)
added to the main path output, preserving gradient flow and enabling the
network to learn residual feature corrections rather than full
transformations.

### Loss function
Focal Tversky loss was used for segmentation training, which places higher
weight on false negatives (missed tumour pixels), more clinically critical
than false positives in this context. Standard binary cross-entropy was
used for classification.

## Results

| Stage | Metric | Score |
|-------|--------|-------|
| Classification (ResNet-50) | Validation Accuracy | 95.6% |
| Classification (ResNet-50) | Validation Loss | 0.160 |
| Segmentation (ResUNet) | Tversky Score | reported per epoch |

The combined pipeline correctly classifies whether a tumour is present
(Stage 1), and for positive cases, generates a pixel-level mask showing
tumour location and extent (Stage 2). Qualitative inspection of predictions
on 15 held-out positive cases confirmed accurate boundary delineation
in the majority of cases.

## Inference pipeline

At inference time:

1. The classification model receives an MRI scan and predicts tumour presence.
2. If confidence of tumour absence exceeds 99%, the image is labelled
   "no tumour" without segmentation (avoiding unnecessary computation).
3. Otherwise, the segmentation model generates a pixel mask for the scan.
4. The mask is overlaid on the original scan for visual review.

This cascaded design reduces false segmentation on clearly negative scans.

## Tech stack

- Python 3.7
- TensorFlow 2.x / Keras
- OpenCV, scikit-image
- NumPy, Pandas, Matplotlib, Seaborn
- Google Colab (training environment)

## Running the notebook

1. Open `brain_mri_detection_segmentation_resunet.ipynb` in Google Colab
2. Mount your Google Drive and place the LGG MRI Segmentation dataset at:
   `MyDrive/Brain MRI Tumor/archive/kaggle_3m/`
3. Run all cells in order. The notebook is structured in sections:
   - Section 1–3: Data loading and EDA
   - Section 4–7: Classification model (ResNet-50)
   - Section 8–9: Segmentation model (ResUNet) architecture and training
   - Section 10–11: Combined pipeline and evaluation
   - Section 12: Visualisation of predictions

The dataset is available at:
https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

## Key design decisions

**Why residual blocks in the encoder?** Standard convolutions can suppress
low-level spatial detail during downsampling. Residual connections allow
earlier feature maps to bypass transformation and contribute directly to the
decoder, which is particularly useful for small tumour boundaries where
precise pixel-level accuracy matters.

**Why a two-stage pipeline rather than segmentation alone?** Running
segmentation on every scan is computationally expensive and produces noisy
outputs on clearly negative cases. The classification stage acts as a
gate, reducing false activations and making the pipeline more suitable for
deployment in a resource-constrained clinical environment.

**Why Focal Tversky loss?** Standard binary cross-entropy does not
penalise false negatives (missed tumour pixels) more than false positives.
In a medical context, missing a tumour is significantly worse than a false
alarm, so a loss function that emphasises recall over precision is more
clinically appropriate.
