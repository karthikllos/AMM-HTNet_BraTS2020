# AMM-HTNet: Multimodal 3D Brain Tumor Segmentation

## Overview

AMM-HTNet is a hybrid CNN-Transformer network for segmenting brain tumors in 3D MRI, trained and evaluated on BraTS 2020. It has three main ideas:

- **AMMF (Adaptive Multi-Modal Fusion)** learns how much each MRI sequence should contribute at each voxel.
- A **hierarchical encoder** uses convolutions for fine local detail and Swin-style window attention for global context.
- A **composite loss** (Dice + weighted BCE + Boundary + deep supervision + consistency) handles class imbalance and fuzzy tumor borders.

## Problem and data

- **Task:** voxel-wise 3D segmentation of brain tumors.
- **Input:** 4 MRI sequences per patient (T1, T1ce, T2, FLAIR), stacked as a 4-channel volume.
- **Dataset:** BraTS 2020 training set, 368 usable cases, split 295 train / 73 validation.
- **Labels:** BraTS labels (1 = necrotic core, 2 = edema, 4 = enhancing tumor) are converted into three overlapping binary regions:
    - **WT** (Whole Tumor) = labels 1, 2, 4
    - **TC** (Tumor Core) = labels 1, 4
    - **ET** (Enhancing Tumor) = label 4
- **Output:** 3 sigmoid channels (multi-label), because the regions are nested and overlap.

## End-to-end pipeline

```
4 MRI sequences
  → per-modality z-score normalisation
  → foreground-biased 128³ patch sampling
  → 3D augmentation
  → AMMF (adaptive modality fusion)
  → Hierarchical encoder (conv, conv, Swin, Swin)
  → HCM bottleneck (axial attention)
  → Decoder with attention-gated skip connections
  → Segmentation head (WT / TC / ET)
        └→ Boundary Refinement Head (auxiliary, training only)
```

## 1. Data preprocessing

- **Loading:** NIfTI volumes are read with nibabel, and the 4 modalities are stacked into one array.
- **Normalisation:** z-score per modality, computed only over non-zero (brain) voxels. This handles intensity differences across scanners.
- **Patch sampling:** random 128³ crops. With 50% probability the crop is centred on a random tumor voxel, so patches aren't mostly background.
- **Padding:** volumes smaller than the patch are zero-padded.

## 2. Data augmentation

- Random flips along all three axes
- Random 90° rotations in a random plane
- Gaussian noise
- Per-modality intensity scaling (0.85-1.15)
- Gamma correction (0.7-1.5)
- **Two augmented views** of each sample are generated for the consistency loss.

## 3. Architecture

Base width is 32 channels, giving 32 → 64 → 128 → 256 across stages and about 6.6M parameters.

### 3.1 AMMF: Adaptive Multi-Modal Fusion (core contribution)

Different tumor regions depend on different sequences (ET on T1ce, WT on FLAIR, TC on T1/T1ce). Plain channel concatenation treats them equally, so AMMF learns the weighting.

1. Each modality goes through its **own 2-conv stem** (Conv3d → InstanceNorm → GELU, twice).
2. A **global gate** pools features from all modalities, passes them through an MLP, and applies a softmax over the 4 modalities. This gives one weight per modality per sample.
3. A **spatial gate** (1×1×1 conv + softmax over modalities) gives a weight per modality at every voxel.
4. The two weights are multiplied and renormalised, then used for a weighted sum of the modality features.
5. A **depthwise + pointwise conv** refines the fused features.
6. A **residual projection** from the raw 4-channel input is added for gradient flow.

### 3.2 Hierarchical encoder (hybrid conv + Swin-style)

| Stage | Block type | Resolution (128³ input) |
| --- | --- | --- |
| 0 | 3D conv blocks | 128³ |
| 1 | 3D conv blocks | 64³ |
| 2 | Swin-style window attention | 32³ |
| 3 | Swin-style window attention | 16³ |
- **Early stages use convolutions.** Attention at full resolution is expensive, and local texture and edges matter most there.
- **Deep stages use 3D window attention.** Features include 7×7×7 windows, shifted windows on alternate blocks, relative position bias, LayerNorm + MLP, and stochastic depth. Attention is affordable at low resolution and captures long-range context.
- **Downsampling** uses strided convolutions.
- Each stage outputs a **skip feature** for the decoder, giving global and local information at every scale.

### 3.3 HCM: Hybrid Context Module (bottleneck)

- **Axial attention** along H, W and D separately, which costs far less than full 3D attention.
- **Channel attention** (squeeze-and-excitation style).
- A **feed-forward block**, all with residual connections.

### 3.4 Decoder with AGSF (Attention-Guided Skip Fusion)

- Transposed-conv upsampling at each level.
- An **additive attention gate** on each encoder skip, with the decoder feature as the query. It suppresses irrelevant encoder features before concatenation.
- Two conv layers fuse the gated skip with the upsampled feature.

### 3.5 Deep supervision

1×1×1 heads at three decoder levels, upsampled to full size and supervised with weights 1.0 / 0.5 / 0.25.

### 3.6 BRH: Boundary Refinement Head

- A depthwise conv **initialised as a Laplacian edge filter** runs on the finest encoder skip.
- Its output is concatenated with the upsampled coarse prediction.
- Light depthwise-separable convs then produce a **boundary map** per region.
- This is an auxiliary branch for boundary supervision.

## 4. Loss function

`L = λ_dice·Dice + λ_ce·wBCE + λ_bdry·Boundary + Σ w_s·(Dice + wBCE)_ds + λ_con·Consistency`

with λ = 1.0, 1.0, 0.5, and 0.1 for the consistency term.

| Component | What it does |
| --- | --- |
| **Dice loss** | Soft Dice on sigmoid outputs, averaged over WT/TC/ET. Overlap-based, so robust to the tumor being a tiny fraction of the volume. |
| **Weighted BCE** | BCE multiplied by `(1 − p_t)²`, a focal-style weight that focuses on hard voxels. |
| **Boundary loss** | GT boundary = max-pool dilation minus mask. A Dice-style loss compares the BRH boundary map to it. |
| **Deep supervision** | Dice + weighted BCE on the 3 auxiliary decoder outputs. |
| **Consistency loss** | MSE between predictions on two augmented views, with the second view detached as a pseudo-target. |

## 5. Training setup

- **Optimiser:** AdamW (lr 1e-4, weight decay 1e-5, betas 0.9/0.999)
- **Schedule:** 10-epoch linear warmup, then cosine decay down to 1e-6
- **Duration and batching:** 300 epochs, batch size 1 (128³ patches on a T4 GPU)
- **Stability:** mixed precision (AMP), gradient clipping at 1.0
- **EMA:** weights tracked with decay 0.9999 and used for validation
- **Validation:** every 10 epochs, with the top 3 checkpoints plus the best model saved
- **Logging:** TensorBoard, a training log file, and a seeded setup (seed 42) for reproducibility

## 6. Inference pipeline

1. **Sliding-window inference** with 128³ windows, 50% overlap and Gaussian blending, so a full volume fits in GPU memory without seams.
2. **7-way flip TTA**: the input is flipped along different axis combinations, predicted, un-flipped and averaged.
3. **Threshold** at 0.5 on the sigmoid output.
4. **Connected-component cleanup:** removes isolated blobs under 50 voxels, mainly false positives in ET.

## 7. Evaluation metrics

Computed per region (WT, TC, ET) and aggregated as mean ± std:

- **Dice**: overlap
- **HD95**: 95th-percentile Hausdorff distance, a boundary-distance measure that is robust to outliers
- **Sensitivity, specificity, precision**
