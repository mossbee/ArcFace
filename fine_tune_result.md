================================================================================
FACE VERIFICATION MODEL COMPARISON
================================================================================

[1/2] Evaluating ORIGINAL model...
--------------------------------------------------------------------------------
Loading model from: /input/nd-twin-448-train/ms1mv3_arcface_r100_fp16.pth
Model loaded successfully on cuda
Loaded 1554 valid pairs from /input/nd-twin-448-train/test_pairs.txt
Processing 1554 pairs...
Computing embeddings:   0%|                              | 0/25 [00:00<?, ?it/s]/working/ArcFace/backbones/iresnet.py:149: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(self.fp16):
Computing embeddings: 100%|█████████████████████| 25/25 [00:20<00:00,  1.22it/s]

Successfully processed 1554 pairs

============================================================
  ORIGINAL MODEL - Verification Results
============================================================
  AUC:                    0.7134
  EER:                    0.3437 (Threshold: 0.6305)
  Best Accuracy:          0.6705 (Threshold: 0.6639)
  Precision:              0.6827
  Recall:                 0.5896
  F1 Score:               0.6327

  Confusion Matrix:
    True Positives:       441
    False Positives:      205
    True Negatives:       601
    False Negatives:      307

  TPR at specific FPR:
    TPR@FPR=1e-04: 0.0000
    TPR@FPR=1e-03: 0.0963
    TPR@FPR=1e-02: 0.2179
    TPR@FPR=1e-01: 0.3971
============================================================


[2/2] Evaluating FINE-TUNED model...
--------------------------------------------------------------------------------
Loading model from: /input/nd-twin-448-train/ms1mv3_arcface_r100_fp16_fine_tune_ndtwin.pt
Model loaded successfully on cuda
Processing 1554 pairs...
Computing embeddings:   0%|                              | 0/25 [00:00<?, ?it/s]/working/ArcFace/backbones/iresnet.py:149: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(self.fp16):
Computing embeddings: 100%|█████████████████████| 25/25 [00:20<00:00,  1.20it/s]

Successfully processed 1554 pairs

============================================================
  FINE-TUNED MODEL - Verification Results
============================================================
  AUC:                    0.9359
  EER:                    0.1328 (Threshold: 0.5850)
  Best Accuracy:          0.8732 (Threshold: 0.6148)
  Precision:              0.9057
  Recall:                 0.8222
  F1 Score:               0.8619

  Confusion Matrix:
    True Positives:       615
    False Positives:      64
    True Negatives:       742
    False Negatives:      133

  TPR at specific FPR:
    TPR@FPR=1e-04: 0.0000
    TPR@FPR=1e-03: 0.1444
    TPR@FPR=1e-02: 0.5535
    TPR@FPR=1e-01: 0.8316
============================================================


================================================================================
  COMPARISON SUMMARY
================================================================================
Metric                    Original        Fine-tuned      Improvement    
--------------------------------------------------------------------------------
AUC                       0.7134          0.9359          +0.2225 ✓      
EER                       0.3437          0.1328          -0.2109 ✓      
Best_Accuracy             0.6705          0.8732          +0.2027 ✓      
Precision                 0.6827          0.9057          +0.2231 ✓      
Recall                    0.5896          0.8222          +0.2326 ✓      
F1_Score                  0.6327          0.8619          +0.2292 ✓      
================================================================================

KEY FINDINGS:
  ✓ Fine-tuning improved accuracy by 20.27%
  ✓ Fine-tuning reduced EER by 21.09%

================================================================================