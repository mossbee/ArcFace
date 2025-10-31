# Face Verification Evaluation - Quick Start

## 📁 New Files Created

1. **`evaluate_verification.py`** - Main evaluation script for single model
2. **`compare_models.py`** - Compare original vs fine-tuned models  
3. **`EVALUATION_GUIDE.md`** - Detailed documentation
4. **`kaggle_evaluation_example.py`** - Ready-to-use Kaggle notebook cells

## 🚀 Quick Start (Kaggle)

### Step 1: Navigate to ArcFace directory
```python
import os
os.chdir('/kaggle/working/ArcFace')
```

### Step 2: Compare Both Models
```bash
!python compare_models.py \
    --pairs_file /kaggle/input/nd-twin-448-train/test_pairs.txt \
    --data_root /kaggle/input/nd-twin-448-train/ND_TWIN_448_TEST \
    --original_model /kaggle/input/nd-twin-448-train/ms1mv3_arcface_r100_fp16.pth \
    --finetuned_model output/ndtwin_r100/model.pt
```

**That's it!** You'll get a complete comparison showing:
- ✅ Performance metrics for both models
- ✅ Side-by-side comparison table
- ✅ Accuracy improvement analysis

## 📊 What You'll See

```
============================================================================
  COMPARISON SUMMARY
============================================================================
Metric                    Original        Fine-tuned      Improvement    
----------------------------------------------------------------------------
AUC                       0.9234          0.9678          +0.0444 ✓
EER                       0.0456          0.0289          -0.0167 ✓
Best_Accuracy             0.9123          0.9567          +0.0444 ✓
Precision                 0.9056          0.9523          +0.0467 ✓
Recall                    0.9189          0.9611          +0.0422 ✓
F1_Score                  0.9122          0.9567          +0.0445 ✓
============================================================================

KEY FINDINGS:
  ✓ Fine-tuning improved accuracy by 4.44%
  ✓ Fine-tuning reduced EER by 1.67%
```

## 📈 Metrics Explained (Simple Version)

| Metric | What it means | Good value |
|--------|---------------|------------|
| **Accuracy** | % of correct predictions | Higher (>0.95) |
| **EER** | Error rate | Lower (<0.05) |
| **AUC** | Overall performance | Higher (>0.95) |
| **Precision** | Correct matches / Total predicted matches | Higher |
| **Recall** | Found matches / Total actual matches | Higher |

## 🔧 Common Options

```bash
# Use different batch size (for speed/memory)
--batch_size 128

# Use CPU instead of GPU
--device cpu

# Change model paths
--finetuned_model /path/to/your/model.pt
```

## 💡 Pro Tips

1. **Higher batch size = faster** (if you have GPU memory)
   - Default: 32
   - Try: 64, 128, or 256

2. **Typical evaluation time**: 1-5 minutes for ~10k pairs

3. **If you get memory errors**: Reduce batch size to 16 or 8

## 📝 Test Pairs Format

Your `test_pairs.txt` should look like:
```
person1/image1.jpg person1/image2.jpg 1    # Same person
person1/image1.jpg person2/image1.jpg 0    # Different people
```

## 🔍 Evaluate Single Model Only

If you only want to test one model:

```bash
!python evaluate_verification.py \
    --pairs_file /kaggle/input/nd-twin-448-train/test_pairs.txt \
    --data_root /kaggle/input/nd-twin-448-train/ND_TWIN_448_TEST \
    --model_path output/ndtwin_r100/model.pt \
    --output results.txt
```

## 🎯 Expected Performance

**Original MS1MV3 model** (pretrained on 93k identities):
- Accuracy: ~85-92% on new domain
- EER: ~0.04-0.08

**Fine-tuned model** (on your 377 identities):
- Accuracy: Should improve by 2-8%
- EER: Should decrease by 0.01-0.03
- Works much better on your specific twin dataset!

## ❓ Troubleshooting

**"Image not found" errors**
- Check paths in test_pairs.txt match actual file locations
- Verify data_root is correct

**"Out of memory" errors**
- Reduce batch_size: `--batch_size 16`
- Use CPU: `--device cpu`

**"Model loading failed"**
- Check model path is correct
- Ensure network type matches (r100, r50, etc.)

## 📚 More Details

See `EVALUATION_GUIDE.md` for comprehensive documentation.

See `kaggle_evaluation_example.py` for advanced usage examples.

---

**Ready to evaluate?** Just run the compare_models.py command above! 🚀

