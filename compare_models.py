"""
Compare original and fine-tuned models on face verification task.
This script evaluates both models and shows a comparison.
"""

import argparse
import os
from evaluate_verification import FaceVerificationEvaluator


def compare_models(pairs_file, data_root, original_model, finetuned_model, 
                   network='r100', batch_size=32, device='cuda'):
    """
    Compare original and fine-tuned models.
    
    Args:
        pairs_file: Path to test_pairs.txt
        data_root: Root directory for test images
        original_model: Path to original model weights
        finetuned_model: Path to fine-tuned model weights
        network: Backbone network type
        batch_size: Batch size for inference
        device: Device to use
    """
    
    print("="*80)
    print("FACE VERIFICATION MODEL COMPARISON")
    print("="*80)
    
    # Evaluate original model
    print("\n[1/2] Evaluating ORIGINAL model...")
    print("-"*80)
    evaluator_original = FaceVerificationEvaluator(
        model_path=original_model,
        network=network,
        device=device,
        batch_size=batch_size
    )
    pairs = evaluator_original.load_pairs(pairs_file, data_root)
    metrics_original, _, _ = evaluator_original.evaluate_pairs(pairs)
    evaluator_original.print_metrics(metrics_original, model_name="ORIGINAL MODEL")
    
    # Evaluate fine-tuned model
    print("\n[2/2] Evaluating FINE-TUNED model...")
    print("-"*80)
    evaluator_finetuned = FaceVerificationEvaluator(
        model_path=finetuned_model,
        network=network,
        device=device,
        batch_size=batch_size
    )
    metrics_finetuned, _, _ = evaluator_finetuned.evaluate_pairs(pairs)
    evaluator_finetuned.print_metrics(metrics_finetuned, model_name="FINE-TUNED MODEL")
    
    # Print comparison
    print("\n" + "="*80)
    print("  COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Metric':<25} {'Original':<15} {'Fine-tuned':<15} {'Improvement':<15}")
    print("-"*80)
    
    metrics_to_compare = [
        ('AUC', True),
        ('EER', False),
        ('Best_Accuracy', True),
        ('Precision', True),
        ('Recall', True),
        ('F1_Score', True),
    ]
    
    for metric_name, higher_is_better in metrics_to_compare:
        orig_val = metrics_original[metric_name]
        ft_val = metrics_finetuned[metric_name]
        
        if higher_is_better:
            improvement = ft_val - orig_val
            improvement_str = f"+{improvement:.4f}" if improvement >= 0 else f"{improvement:.4f}"
            if improvement > 0:
                improvement_str += " ✓"
        else:
            improvement = orig_val - ft_val
            improvement_str = f"-{abs(improvement):.4f}" if improvement >= 0 else f"+{abs(improvement):.4f}"
            if improvement > 0:
                improvement_str += " ✓"
        
        print(f"{metric_name:<25} {orig_val:<15.4f} {ft_val:<15.4f} {improvement_str:<15}")
    
    print("="*80)
    
    # Summary
    print("\nKEY FINDINGS:")
    acc_improvement = metrics_finetuned['Best_Accuracy'] - metrics_original['Best_Accuracy']
    eer_improvement = metrics_original['EER'] - metrics_finetuned['EER']
    
    if acc_improvement > 0:
        print(f"  ✓ Fine-tuning improved accuracy by {acc_improvement*100:.2f}%")
    else:
        print(f"  ✗ Fine-tuning decreased accuracy by {abs(acc_improvement)*100:.2f}%")
    
    if eer_improvement > 0:
        print(f"  ✓ Fine-tuning reduced EER by {eer_improvement*100:.2f}%")
    else:
        print(f"  ✗ Fine-tuning increased EER by {abs(eer_improvement)*100:.2f}%")
    
    print("\n" + "="*80 + "\n")
    
    return metrics_original, metrics_finetuned


def main():
    parser = argparse.ArgumentParser(description='Compare Original vs Fine-tuned Models')
    parser.add_argument('--pairs_file', type=str, 
                        default='/kaggle/input/nd-twin-448-train/test_pairs.txt',
                        help='Path to test_pairs.txt file')
    parser.add_argument('--data_root', type=str,
                        default='/kaggle/input/nd-twin-448-train/ND_TWIN_448_TEST',
                        help='Root directory for test images')
    parser.add_argument('--original_model', type=str,
                        default='/kaggle/input/nd-twin-448-train/ms1mv3_arcface_r100_fp16.pth',
                        help='Path to original model weights')
    parser.add_argument('--finetuned_model', type=str,
                        default='/kaggle/input/nd-twin-448-train/ms1mv3_arcface_r100_fp16_fine_tune_ndtwin.pt',
                        help='Path to fine-tuned model weights')
    parser.add_argument('--network', type=str, default='r100',
                        help='Backbone network (r50, r100, etc.)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (cuda or cpu)')
    
    args = parser.parse_args()
    
    compare_models(
        pairs_file=args.pairs_file,
        data_root=args.data_root,
        original_model=args.original_model,
        finetuned_model=args.finetuned_model,
        network=args.network,
        batch_size=args.batch_size,
        device=args.device
    )


if __name__ == "__main__":
    main()

