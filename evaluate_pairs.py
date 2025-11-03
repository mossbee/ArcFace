#!/usr/bin/env python3
"""
Face Verification Script for ArcFace
Loads pretrained model and evaluates on image pairs
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

from backbones import get_model


class ImagePairDataset(Dataset):
    """Dataset for loading image pairs"""
    
    def __init__(self, pairs_file, dataset_root, transform=None):
        """
        Args:
            pairs_file: Path to text file with format "path1 path2 label"
            dataset_root: Root directory containing images
            transform: Optional transform to apply to images
        """
        self.dataset_root = Path(dataset_root)
        self.pairs = []
        
        # Read pairs file
        with open(pairs_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    img1_path, img2_path, label = parts
                    self.pairs.append((img1_path, img2_path, int(label)))
        
        print(f"Loaded {len(self.pairs)} image pairs")
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img1_rel, img2_rel, label = self.pairs[idx]
        
        # Construct absolute paths
        img1_path = self.dataset_root / img1_rel
        img2_path = self.dataset_root / img2_rel
        
        # Load and preprocess images
        img1 = self._load_image(img1_path)
        img2 = self._load_image(img2_path)
        
        return img1, img2, label
    
    def _load_image(self, img_path):
        """Load and preprocess a single image"""
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Resize to 112x112 (ArcFace input size)
        img = cv2.resize(img, (112, 112))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Transpose to CHW format
        img = np.transpose(img, (2, 0, 1))
        
        # Convert to torch tensor
        img = torch.from_numpy(img).float()
        
        # Normalize: (img / 255 - 0.5) / 0.5 = (img - 127.5) / 127.5
        img.div_(255).sub_(0.5).div_(0.5)
        
        return img


def collate_fn(batch):
    """Custom collate function to separate img1, img2, and labels"""
    img1_list, img2_list, label_list = zip(*batch)
    
    img1_batch = torch.stack(img1_list)
    img2_batch = torch.stack(img2_list)
    label_batch = torch.tensor(label_list)
    
    return img1_batch, img2_batch, label_batch


@torch.no_grad()
def extract_features(model, dataloader, device):
    """Extract features for all image pairs"""
    model.eval()
    
    all_feat1 = []
    all_feat2 = []
    all_labels = []
    
    print("Extracting features...")
    for img1_batch, img2_batch, label_batch in tqdm(dataloader):
        img1_batch = img1_batch.to(device)
        img2_batch = img2_batch.to(device)
        
        # Extract features
        feat1 = model(img1_batch)
        feat2 = model(img2_batch)
        
        # Normalize features (L2 normalization)
        feat1 = torch.nn.functional.normalize(feat1, p=2, dim=1)
        feat2 = torch.nn.functional.normalize(feat2, p=2, dim=1)
        
        all_feat1.append(feat1.cpu())
        all_feat2.append(feat2.cpu())
        all_labels.append(label_batch)
    
    # Concatenate all batches
    all_feat1 = torch.cat(all_feat1, dim=0)
    all_feat2 = torch.cat(all_feat2, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return all_feat1, all_feat2, all_labels


def compute_similarity(feat1, feat2):
    """Compute cosine similarity between feature pairs"""
    # Features are already L2-normalized, so cosine similarity is just dot product
    similarity = (feat1 * feat2).sum(dim=1)
    return similarity.numpy()


def calculate_metrics(scores, labels):
    """Calculate verification metrics"""
    labels = labels.numpy()
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    # Calculate AUC
    auc_score = auc(fpr, tpr)
    
    # Accuracy at optimal threshold (max Youden's J statistic)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    predictions = (scores >= optimal_threshold).astype(int)
    accuracy = np.mean(predictions == labels)
    
    # Calculate EER (Equal Error Rate)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[eer_idx]
    eer_threshold = thresholds[eer_idx]
    
    # TAR at specific FAR values
    def tar_at_far(fpr, tpr, target_far):
        idx = np.argmin(np.abs(fpr - target_far))
        return tpr[idx], fpr[idx]
    
    tar_1e_1, far_1e_1 = tar_at_far(fpr, tpr, 1e-1)
    tar_1e_2, far_1e_2 = tar_at_far(fpr, tpr, 1e-2)
    tar_1e_3, far_1e_3 = tar_at_far(fpr, tpr, 1e-3)
    tar_1e_4, far_1e_4 = tar_at_far(fpr, tpr, 1e-4)
    
    results = {
        'auc': auc_score,
        'accuracy': accuracy,
        'optimal_threshold': optimal_threshold,
        'eer': eer,
        'eer_threshold': eer_threshold,
        'tar@far=1e-1': tar_1e_1,
        'tar@far=1e-2': tar_1e_2,
        'tar@far=1e-3': tar_1e_3,
        'tar@far=1e-4': tar_1e_4,
        'num_same': np.sum(labels == 1),
        'num_diff': np.sum(labels == 0),
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Face Verification with ArcFace')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to pretrained model (e.g., ms1mv3_arcface_r100_fp16.pth)')
    parser.add_argument('--pairs_file', type=str, required=True,
                        help='Path to text file with image pairs (format: path1 path2 label)')
    parser.add_argument('--dataset_root', type=str, required=True,
                        help='Root directory containing images')
    parser.add_argument('--network', type=str, default='r100',
                        help='Backbone network (r50, r100, etc.)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--fp16', action='store_true',
                        help='Use fp16 for faster inference')
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    if not os.path.exists(args.pairs_file):
        raise FileNotFoundError(f"Pairs file not found: {args.pairs_file}")
    if not os.path.exists(args.dataset_root):
        raise FileNotFoundError(f"Dataset root not found: {args.dataset_root}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = get_model(args.network, fp16=args.fp16)
    
    # Load weights
    state_dict = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Model loaded successfully")
    
    # Create dataset and dataloader
    dataset = ImagePairDataset(args.pairs_file, args.dataset_root)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Extract features
    start_time = time.time()
    feat1, feat2, labels = extract_features(model, dataloader, device)
    extraction_time = time.time() - start_time
    
    print(f"\nFeature extraction completed in {extraction_time:.2f} seconds")
    print(f"Average time per pair: {extraction_time / len(dataset) * 1000:.2f} ms")
    
    # Compute similarity scores
    print("\nComputing similarity scores...")
    scores = compute_similarity(feat1, feat2)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    results = calculate_metrics(scores, labels)
    
    # Print results
    print("\n" + "="*60)
    print("VERIFICATION RESULTS")
    print("="*60)
    print(f"Dataset: {args.pairs_file}")
    print(f"Total pairs: {len(dataset)}")
    print(f"  - Same person (positive): {results['num_same']}")
    print(f"  - Different person (negative): {results['num_diff']}")
    print("-"*60)
    print(f"AUC: {results['auc']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f} (at threshold={results['optimal_threshold']:.4f})")
    print(f"EER: {results['eer']:.4f} (at threshold={results['eer_threshold']:.4f})")
    print("-"*60)
    print("TAR @ FAR:")
    print(f"  TAR @ FAR=1e-1: {results['tar@far=1e-1']:.4f}")
    print(f"  TAR @ FAR=1e-2: {results['tar@far=1e-2']:.4f}")
    print(f"  TAR @ FAR=1e-3: {results['tar@far=1e-3']:.4f}")
    print(f"  TAR @ FAR=1e-4: {results['tar@far=1e-4']:.4f}")
    print("="*60)
    
    # Save results to file
    output_file = args.pairs_file.replace('.txt', '_results.txt')
    with open(output_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("VERIFICATION RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Dataset: {args.pairs_file}\n")
        f.write(f"Total pairs: {len(dataset)}\n")
        f.write(f"  - Same person (positive): {results['num_same']}\n")
        f.write(f"  - Different person (negative): {results['num_diff']}\n")
        f.write("-"*60 + "\n")
        f.write(f"AUC: {results['auc']:.4f}\n")
        f.write(f"Accuracy: {results['accuracy']:.4f} (at threshold={results['optimal_threshold']:.4f})\n")
        f.write(f"EER: {results['eer']:.4f} (at threshold={results['eer_threshold']:.4f})\n")
        f.write("-"*60 + "\n")
        f.write("TAR @ FAR:\n")
        f.write(f"  TAR @ FAR=1e-1: {results['tar@far=1e-1']:.4f}\n")
        f.write(f"  TAR @ FAR=1e-2: {results['tar@far=1e-2']:.4f}\n")
        f.write(f"  TAR @ FAR=1e-3: {results['tar@far=1e-3']:.4f}\n")
        f.write(f"  TAR @ FAR=1e-4: {results['tar@far=1e-4']:.4f}\n")
        f.write("="*60 + "\n")
        f.write(f"\nExecution time: {extraction_time:.2f} seconds\n")
        f.write(f"Average time per pair: {extraction_time / len(dataset) * 1000:.2f} ms\n")
    
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()