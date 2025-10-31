import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from backbones import get_model


class FaceVerificationEvaluator:
    def __init__(self, model_path, network='r100', device='cuda', batch_size=32):
        """
        Initialize the face verification evaluator.
        
        Args:
            model_path: Path to model weights
            network: Backbone network type (r50, r100, etc.)
            device: Device to run on ('cuda' or 'cpu')
            batch_size: Batch size for inference
        """
        self.device = device
        self.batch_size = batch_size
        
        # Load model
        print(f"Loading model from: {model_path}")
        self.net = get_model(network, fp16=False)
        self.net.load_state_dict(torch.load(model_path, map_location=device))
        self.net.to(device)
        self.net.eval()
        print(f"Model loaded successfully on {device}")
    
    def preprocess_image(self, img_path):
        """
        Preprocess a single image for inference.
        
        Args:
            img_path: Path to image file
            
        Returns:
            Preprocessed image tensor
        """
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Resize to 112x112
        img = cv2.resize(img, (112, 112))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Transpose to CHW format
        img = np.transpose(img, (2, 0, 1))
        
        # Convert to tensor and normalize
        img = torch.from_numpy(img).float()
        img.div_(255).sub_(0.5).div_(0.5)
        
        return img
    
    @torch.no_grad()
    def get_embedding(self, img_tensor):
        """
        Get embedding for a single image tensor.
        
        Args:
            img_tensor: Preprocessed image tensor (C, H, W)
            
        Returns:
            L2-normalized embedding
        """
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        embedding = self.net(img_tensor).cpu().numpy()
        # L2 normalize
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
        return embedding[0]
    
    @torch.no_grad()
    def get_embeddings_batch(self, img_tensors):
        """
        Get embeddings for a batch of image tensors.
        
        Args:
            img_tensors: List of preprocessed image tensors
            
        Returns:
            L2-normalized embeddings
        """
        batch = torch.stack(img_tensors).to(self.device)
        embeddings = self.net(batch).cpu().numpy()
        # L2 normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings
    
    def cosine_similarity(self, emb1, emb2):
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            emb1, emb2: L2-normalized embeddings
            
        Returns:
            Cosine similarity score
        """
        return np.dot(emb1, emb2)
    
    def load_pairs(self, pairs_file, data_root):
        """
        Load test pairs from file.
        
        Args:
            pairs_file: Path to pairs.txt file
            data_root: Root directory for images
            
        Returns:
            List of (img1_path, img2_path, label) tuples
        """
        pairs = []
        with open(pairs_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 3:
                    print(f"Warning: Skipping invalid line: {line}")
                    continue
                
                img1_rel, img2_rel, label = parts
                img1_path = os.path.join(data_root, img1_rel)
                img2_path = os.path.join(data_root, img2_rel)
                label = int(label)
                
                # Check if files exist
                if not os.path.exists(img1_path):
                    print(f"Warning: Image not found: {img1_path}")
                    continue
                if not os.path.exists(img2_path):
                    print(f"Warning: Image not found: {img2_path}")
                    continue
                
                pairs.append((img1_path, img2_path, label))
        
        print(f"Loaded {len(pairs)} valid pairs from {pairs_file}")
        return pairs
    
    def evaluate_pairs(self, pairs):
        """
        Evaluate all pairs and compute metrics.
        
        Args:
            pairs: List of (img1_path, img2_path, label) tuples
            
        Returns:
            Dictionary of evaluation metrics
        """
        similarities = []
        labels = []
        
        print(f"Processing {len(pairs)} pairs...")
        
        # Process in batches for speed
        for i in tqdm(range(0, len(pairs), self.batch_size), desc="Computing embeddings"):
            batch_pairs = pairs[i:i+self.batch_size]
            
            # Load and preprocess images
            img1_tensors = []
            img2_tensors = []
            batch_labels = []
            
            for img1_path, img2_path, label in batch_pairs:
                try:
                    img1_tensor = self.preprocess_image(img1_path)
                    img2_tensor = self.preprocess_image(img2_path)
                    img1_tensors.append(img1_tensor)
                    img2_tensors.append(img2_tensor)
                    batch_labels.append(label)
                except Exception as e:
                    print(f"Error processing pair: {img1_path}, {img2_path}: {e}")
                    continue
            
            if not img1_tensors:
                continue
            
            # Get embeddings in batch
            emb1_batch = self.get_embeddings_batch(img1_tensors)
            emb2_batch = self.get_embeddings_batch(img2_tensors)
            
            # Compute similarities
            for emb1, emb2, label in zip(emb1_batch, emb2_batch, batch_labels):
                sim = self.cosine_similarity(emb1, emb2)
                similarities.append(sim)
                labels.append(label)
        
        similarities = np.array(similarities)
        labels = np.array(labels)
        
        print(f"\nSuccessfully processed {len(similarities)} pairs")
        
        # Compute metrics
        metrics = self.compute_metrics(similarities, labels)
        
        return metrics, similarities, labels
    
    def compute_metrics(self, similarities, labels):
        """
        Compute verification metrics.
        
        Args:
            similarities: Array of similarity scores
            labels: Array of ground truth labels (0 or 1)
            
        Returns:
            Dictionary of metrics
        """
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(labels, similarities)
        roc_auc = auc(fpr, tpr)
        
        # Compute EER (Equal Error Rate)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        eer_threshold = interp1d(fpr, thresholds)(eer)
        
        # Find best threshold (maximize accuracy)
        accuracies = []
        for threshold in thresholds:
            predictions = (similarities >= threshold).astype(int)
            acc = np.mean(predictions == labels)
            accuracies.append(acc)
        
        best_acc_idx = np.argmax(accuracies)
        best_accuracy = accuracies[best_acc_idx]
        best_threshold = thresholds[best_acc_idx]
        
        # Compute metrics at best threshold
        predictions = (similarities >= best_threshold).astype(int)
        true_positives = np.sum((predictions == 1) & (labels == 1))
        false_positives = np.sum((predictions == 1) & (labels == 0))
        true_negatives = np.sum((predictions == 0) & (labels == 0))
        false_negatives = np.sum((predictions == 0) & (labels == 1))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # TPR at specific FPR values
        tpr_at_fpr = {}
        for target_fpr in [1e-4, 1e-3, 1e-2, 1e-1]:
            idx = np.argmin(np.abs(fpr - target_fpr))
            tpr_at_fpr[f'TPR@FPR={target_fpr:.0e}'] = tpr[idx]
        
        metrics = {
            'AUC': roc_auc,
            'EER': eer,
            'EER_Threshold': eer_threshold,
            'Best_Accuracy': best_accuracy,
            'Best_Threshold': best_threshold,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1,
            'True_Positives': int(true_positives),
            'False_Positives': int(false_positives),
            'True_Negatives': int(true_negatives),
            'False_Negatives': int(false_negatives),
            **tpr_at_fpr
        }
        
        return metrics
    
    def print_metrics(self, metrics, model_name="Model"):
        """
        Print evaluation metrics in a formatted way.
        """
        print(f"\n{'='*60}")
        print(f"  {model_name} - Verification Results")
        print(f"{'='*60}")
        print(f"  AUC:                    {metrics['AUC']:.4f}")
        print(f"  EER:                    {metrics['EER']:.4f} (Threshold: {metrics['EER_Threshold']:.4f})")
        print(f"  Best Accuracy:          {metrics['Best_Accuracy']:.4f} (Threshold: {metrics['Best_Threshold']:.4f})")
        print(f"  Precision:              {metrics['Precision']:.4f}")
        print(f"  Recall:                 {metrics['Recall']:.4f}")
        print(f"  F1 Score:               {metrics['F1_Score']:.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"    True Positives:       {metrics['True_Positives']}")
        print(f"    False Positives:      {metrics['False_Positives']}")
        print(f"    True Negatives:       {metrics['True_Negatives']}")
        print(f"    False Negatives:      {metrics['False_Negatives']}")
        print(f"\n  TPR at specific FPR:")
        for key in ['TPR@FPR=1e-04', 'TPR@FPR=1e-03', 'TPR@FPR=1e-02', 'TPR@FPR=1e-01']:
            if key in metrics:
                print(f"    {key}: {metrics[key]:.4f}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Face Verification Evaluation')
    parser.add_argument('--pairs_file', type=str, required=True,
                        help='Path to pairs.txt file')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory for test images')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model weights (.pth or .pt file)')
    parser.add_argument('--network', type=str, default='r100',
                        help='Backbone network (r50, r100, etc.)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (cuda or cpu)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file to save results (optional)')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = FaceVerificationEvaluator(
        model_path=args.model_path,
        network=args.network,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # Load pairs
    pairs = evaluator.load_pairs(args.pairs_file, args.data_root)
    
    if len(pairs) == 0:
        print("Error: No valid pairs found!")
        return
    
    # Evaluate
    metrics, similarities, labels = evaluator.evaluate_pairs(pairs)
    
    # Print results
    model_name = os.path.basename(args.model_path)
    evaluator.print_metrics(metrics, model_name=model_name)
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            f.write(f"Model: {args.model_path}\n")
            f.write(f"Pairs file: {args.pairs_file}\n")
            f.write(f"Total pairs: {len(pairs)}\n\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()

