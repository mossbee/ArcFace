"""
Kaggle Notebook Example - Face Verification Evaluation
Copy and paste these cells into your Kaggle notebook
"""

# ============================================================================
# CELL 1: Setup
# ============================================================================
import os
os.chdir('/kaggle/working/ArcFace')

# Install requirements if needed
# !pip install scikit-learn scipy

# ============================================================================
# CELL 2: Quick Comparison (Most Common Use Case)
# ============================================================================
# Compare original model vs your fine-tuned model
!python compare_models.py \
    --pairs_file /kaggle/input/nd-twin-448-train/test_pairs.txt \
    --data_root /kaggle/input/nd-twin-448-train/ND_TWIN_448_TEST \
    --original_model /kaggle/input/nd-twin-448-train/ms1mv3_arcface_r100_fp16.pth \
    --finetuned_model output/ndtwin_r100/model.pt \
    --network r100 \
    --batch_size 64

# ============================================================================
# CELL 3: Evaluate Only Original Model
# ============================================================================
!python evaluate_verification.py \
    --pairs_file /kaggle/input/nd-twin-448-train/test_pairs.txt \
    --data_root /kaggle/input/nd-twin-448-train/ND_TWIN_448_TEST \
    --model_path /kaggle/input/nd-twin-448-train/ms1mv3_arcface_r100_fp16.pth \
    --network r100 \
    --batch_size 64 \
    --output original_model_results.txt

# ============================================================================
# CELL 4: Evaluate Only Fine-tuned Model
# ============================================================================
!python evaluate_verification.py \
    --pairs_file /kaggle/input/nd-twin-448-train/test_pairs.txt \
    --data_root /kaggle/input/nd-twin-448-train/ND_TWIN_448_TEST \
    --model_path output/ndtwin_r100/model.pt \
    --network r100 \
    --batch_size 64 \
    --output finetuned_model_results.txt

# ============================================================================
# CELL 5: Advanced - Use Python API Directly
# ============================================================================
from evaluate_verification import FaceVerificationEvaluator

# Initialize evaluator
evaluator = FaceVerificationEvaluator(
    model_path='output/ndtwin_r100/model.pt',
    network='r100',
    device='cuda',
    batch_size=64
)

# Load test pairs
pairs = evaluator.load_pairs(
    pairs_file='/kaggle/input/nd-twin-448-train/test_pairs.txt',
    data_root='/kaggle/input/nd-twin-448-train/ND_TWIN_448_TEST'
)

# Evaluate
metrics, similarities, labels = evaluator.evaluate_pairs(pairs)

# Print results
evaluator.print_metrics(metrics, model_name="Fine-tuned Model")

# Access metrics programmatically
print(f"Best Accuracy: {metrics['Best_Accuracy']:.4f}")
print(f"EER: {metrics['EER']:.4f}")
print(f"AUC: {metrics['AUC']:.4f}")

# ============================================================================
# CELL 6: Plot ROC Curve (Optional)
# ============================================================================
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

# Assuming you ran Cell 5 first
fpr, tpr, _ = roc_curve(labels, similarities)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
         label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC Curve - Face Verification', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.grid(alpha=0.3)
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# CELL 7: Compare Multiple Checkpoints (if you saved intermediate models)
# ============================================================================
import glob
from evaluate_verification import FaceVerificationEvaluator

# Find all model checkpoints
model_dir = 'output/ndtwin_r100'
checkpoint_pattern = f'{model_dir}/checkpoint_*.pt'
checkpoints = sorted(glob.glob(checkpoint_pattern))

if checkpoints:
    results = []
    
    for checkpoint_path in checkpoints:
        print(f"\nEvaluating: {checkpoint_path}")
        evaluator = FaceVerificationEvaluator(
            model_path=checkpoint_path,
            network='r100',
            batch_size=64
        )
        pairs = evaluator.load_pairs(
            '/kaggle/input/nd-twin-448-train/test_pairs.txt',
            '/kaggle/input/nd-twin-448-train/ND_TWIN_448_TEST'
        )
        metrics, _, _ = evaluator.evaluate_pairs(pairs)
        results.append({
            'checkpoint': checkpoint_path,
            'accuracy': metrics['Best_Accuracy'],
            'eer': metrics['EER'],
            'auc': metrics['AUC']
        })
    
    # Print comparison
    print("\n" + "="*80)
    print("CHECKPOINT COMPARISON")
    print("="*80)
    for result in results:
        print(f"{result['checkpoint']:50} | ACC: {result['accuracy']:.4f} | EER: {result['eer']:.4f} | AUC: {result['auc']:.4f}")
else:
    print("No checkpoints found")

# ============================================================================
# CELL 8: Speed Test
# ============================================================================
import time
from evaluate_verification import FaceVerificationEvaluator

batch_sizes = [16, 32, 64, 128]
evaluator = FaceVerificationEvaluator(
    model_path='output/ndtwin_r100/model.pt',
    network='r100'
)

pairs = evaluator.load_pairs(
    '/kaggle/input/nd-twin-448-train/test_pairs.txt',
    '/kaggle/input/nd-twin-448-train/ND_TWIN_448_TEST'
)

print("="*60)
print("BATCH SIZE SPEED TEST")
print("="*60)

for bs in batch_sizes:
    evaluator.batch_size = bs
    start_time = time.time()
    metrics, _, _ = evaluator.evaluate_pairs(pairs)
    elapsed = time.time() - start_time
    pairs_per_sec = len(pairs) / elapsed
    print(f"Batch Size {bs:3d}: {elapsed:.2f}s ({pairs_per_sec:.1f} pairs/sec)")

