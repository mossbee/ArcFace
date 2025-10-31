# Face Verification Evaluation Guide

This guide explains how to evaluate your trained ArcFace models on face verification tasks.

## Files Created

1. **`evaluate_verification.py`** - Evaluate a single model on test pairs
2. **`compare_models.py`** - Compare original vs fine-tuned models side-by-side

## Usage

### Option 1: Evaluate Single Model

```bash
python evaluate_verification.py \
    --pairs_file /kaggle/input/nd-twin-448-train/test_pairs.txt \
    --data_root /kaggle/input/nd-twin-448-train/ND_TWIN_448_TEST \
    --model_path /kaggle/input/nd-twin-448-train/ms1mv3_arcface_r100_fp16.pth \
    --network r100 \
    --batch_size 64 \
    --device cuda
```

### Option 2: Compare Both Models (Recommended)

```bash
python compare_models.py \
    --pairs_file /kaggle/input/nd-twin-448-train/test_pairs.txt \
    --data_root /kaggle/input/nd-twin-448-train/ND_TWIN_448_TEST \
    --original_model /kaggle/input/nd-twin-448-train/ms1mv3_arcface_r100_fp16.pth \
    --finetuned_model /kaggle/working/ArcFace/output/ndtwin_r100/model.pt \
    --network r100 \
    --batch_size 64
```

### Quick Kaggle Usage

```python
# In Kaggle notebook
import os
os.chdir('/kaggle/working/ArcFace')

# Compare both models
!python compare_models.py \
    --pairs_file /kaggle/input/nd-twin-448-train/test_pairs.txt \
    --data_root /kaggle/input/nd-twin-448-train/ND_TWIN_448_TEST \
    --original_model /kaggle/input/nd-twin-448-train/ms1mv3_arcface_r100_fp16.pth \
    --finetuned_model output/ndtwin_r100/model.pt \
    --batch_size 64
```

## Test Pairs File Format

The `test_pairs.txt` file should have the format:
```
relative/path/to/img1.jpg relative/path/to/img2.jpg label
```

Where:
- `label = 1` means same person (genuine pair)
- `label = 0` means different person (imposter pair)

Example:
```
90046/90046d15.jpg 90046/90046d56.jpg 1
90126/90126d74.jpg 90127/90127d19.jpg 0
```

## Metrics Explained

### Primary Metrics

- **AUC** (Area Under Curve): Overall model performance (0-1, higher is better)
- **EER** (Equal Error Rate): Point where FAR = FRR (lower is better)
- **Best Accuracy**: Maximum accuracy across all thresholds (higher is better)

### Additional Metrics

- **Precision**: TP / (TP + FP) - How many predicted matches are correct
- **Recall**: TP / (TP + FN) - How many actual matches were found
- **F1 Score**: Harmonic mean of precision and recall
- **TPR@FPR**: True Positive Rate at specific False Positive Rates

### Confusion Matrix

- **True Positives**: Correctly identified genuine pairs
- **False Positives**: Imposter pairs incorrectly identified as genuine
- **True Negatives**: Correctly identified imposter pairs
- **False Negatives**: Genuine pairs incorrectly identified as imposters

## Performance Tips

1. **Batch Size**: Increase for faster processing (default: 64)
   - GPU memory permitting, use 128 or 256
   
2. **Device**: Use 'cuda' for GPU acceleration

3. **Image Loading**: Images are automatically resized to 112x112

## Output Example

```
================================================================
  FINE-TUNED MODEL - Verification Results
================================================================
  AUC:                    0.9876
  EER:                    0.0234 (Threshold: 0.3456)
  Best Accuracy:          0.9654 (Threshold: 0.3789)
  Precision:              0.9701
  Recall:                 0.9608
  F1 Score:               0.9654

  Confusion Matrix:
    True Positives:       4821
    False Positives:      147
    True Negatives:       4853
    False Negatives:      179

  TPR at specific FPR:
    TPR@FPR=1e-04:        0.8234
    TPR@FPR=1e-03:        0.9012
    TPR@FPR=1e-02:        0.9456
    TPR@FPR=1e-01:        0.9789
================================================================
```

## Troubleshooting

### Images not found
- Check that `data_root` path is correct
- Verify that relative paths in `test_pairs.txt` are correct

### Out of memory
- Reduce `batch_size` (try 32, 16, or 8)
- Use CPU if necessary (`--device cpu`)

### Model loading errors
- Verify model path is correct
- Ensure network type matches model (r50, r100, etc.)
- Check that model file is not corrupted

## Save Results to File

```bash
python evaluate_verification.py \
    --pairs_file test_pairs.txt \
    --data_root /path/to/images \
    --model_path model.pth \
    --output results.txt
```

This will save all metrics to `results.txt`.

