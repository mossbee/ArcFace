"""
Script to evaluate face verification on custom image pairs using pretrained ArcFace model.
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import sklearn.preprocessing

from backbones import get_model
from eval.verification import evaluate


@torch.no_grad()
def load_model(weight_path, network='r100', device='cuda'):
    """
    Load pretrained ArcFace model.
    
    Args:
        weight_path: Path to the .pth file
        network: Backbone architecture (r18, r34, r50, r100, r200)
        device: 'cuda' or 'cpu'
    
    Returns:
        model: Loaded PyTorch model in eval mode
    """
    print(f"Loading model: {network} from {weight_path}")
    
    # Create model
    model = get_model(network, fp16=False)
    
    # Load weights
    checkpoint = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    
    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on {device}")
    return model


def load_image(img_path, target_size=(112, 112)):
    """
    Load and preprocess an image for ArcFace model.
    
    Args:
        img_path: Path to the image file
        target_size: Target size for resizing (width, height)
    
    Returns:
        img_tensor: Preprocessed image tensor (C, H, W)
    """
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")
    
    # Resize to target size
    img = cv2.resize(img, target_size)
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Transpose to (C, H, W)
    img = np.transpose(img, (2, 0, 1))
    
    # Convert to tensor and normalize
    img = torch.from_numpy(img).float()
    img = img / 255.0  # Scale to [0, 1]
    img = (img - 0.5) / 0.5  # Normalize to [-1, 1]
    
    return img


def read_pairs_file(pairs_path):
    """
    Read pairs file containing image paths and labels.
    
    Format: path1 path2 label (1 for same person, 0 for different person)
    
    Args:
        pairs_path: Path to the pairs text file
    
    Returns:
        pairs: List of tuples (path1, path2, label)
    """
    pairs = []
    with open(pairs_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                print(f"Warning: Skipping invalid line: {line}")
                continue
            path1, path2, label = parts[0], parts[1], int(parts[2])
            pairs.append((path1, path2, label))
    
    print(f"Loaded {len(pairs)} image pairs from {pairs_path}")
    return pairs


def extract_embeddings(model, pairs, dataset_root, batch_size=32, device='cuda'):
    """
    Extract embeddings for all images in the pairs.
    
    Args:
        model: Pretrained ArcFace model
        pairs: List of tuples (path1, path2, label)
        dataset_root: Root directory for the dataset
        batch_size: Batch size for inference
        device: 'cuda' or 'cpu'
    
    Returns:
        embeddings1: Embeddings for first images in pairs
        embeddings2: Embeddings for second images in pairs
        issame_list: List of labels (True for same person, False for different)
    """
    print("Extracting embeddings...")
    
    embeddings1 = []
    embeddings2 = []
    issame_list = []
    
    # Process in batches
    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i:i + batch_size]
        
        # Load and preprocess images for this batch
        batch_imgs1 = []
        batch_imgs2 = []
        batch_labels = []
        
        for path1, path2, label in batch_pairs:
            try:
                # Construct full paths
                full_path1 = os.path.join(dataset_root, path1)
                full_path2 = os.path.join(dataset_root, path2)
                
                # Load images
                img1 = load_image(full_path1)
                img2 = load_image(full_path2)
                
                batch_imgs1.append(img1)
                batch_imgs2.append(img2)
                batch_labels.append(label == 1)  # Convert to boolean
                
            except Exception as e:
                print(f"Error processing pair ({path1}, {path2}): {e}")
                continue
        
        if not batch_imgs1:
            continue
        
        # Stack into batch tensors
        batch_imgs1 = torch.stack(batch_imgs1).to(device)
        batch_imgs2 = torch.stack(batch_imgs2).to(device)
        
        # Extract features
        feats1 = model(batch_imgs1).cpu().numpy()
        feats2 = model(batch_imgs2).cpu().numpy()
        
        embeddings1.append(feats1)
        embeddings2.append(feats2)
        issame_list.extend(batch_labels)
        
        if (i // batch_size) % 10 == 0:
            print(f"Processed {i}/{len(pairs)} pairs...")
    
    # Concatenate all embeddings
    embeddings1 = np.vstack(embeddings1)
    embeddings2 = np.vstack(embeddings2)
    
    print(f"Extracted embeddings for {len(issame_list)} pairs")
    print(f"Embeddings shape: {embeddings1.shape}")
    
    return embeddings1, embeddings2, issame_list


def evaluate_verification(embeddings1, embeddings2, issame_list, nrof_folds=10):
    """
    Evaluate verification performance.
    
    Args:
        embeddings1: Embeddings for first images
        embeddings2: Embeddings for second images
        issame_list: List of labels
        nrof_folds: Number of folds for cross-validation
    
    Returns:
        Dictionary with accuracy, val, val_std, and far
    """
    print("\nEvaluating verification performance...")
    
    # Normalize embeddings
    embeddings1_norm = sklearn.preprocessing.normalize(embeddings1)
    embeddings2_norm = sklearn.preprocessing.normalize(embeddings2)
    
    # Concatenate for evaluation
    embeddings = np.concatenate([embeddings1_norm, embeddings2_norm], axis=0)
    
    # Run evaluation
    tpr, fpr, accuracy, val, val_std, far = evaluate(
        embeddings, issame_list, nrof_folds=nrof_folds
    )
    
    # Calculate metrics
    acc_mean = np.mean(accuracy)
    acc_std = np.std(accuracy)
    
    print(f"\n{'='*60}")
    print(f"Verification Results:")
    print(f"{'='*60}")
    print(f"Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"Validation Rate @ FAR=1e-3: {val:.4f} ± {val_std:.4f}")
    print(f"False Accept Rate: {far:.6f}")
    print(f"{'='*60}")
    
    return {
        'accuracy': acc_mean,
        'accuracy_std': acc_std,
        'val': val,
        'val_std': val_std,
        'far': far,
        'tpr': tpr,
        'fpr': fpr
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate face verification on custom pairs dataset')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to pretrained model (.pth file)')
    parser.add_argument('--network', type=str, default='r100',
                        help='Backbone network (r18, r34, r50, r100, r200)')
    parser.add_argument('--pairs', type=str, required=True,
                        help='Path to pairs file (format: path1 path2 label)')
    parser.add_argument('--dataset-root', type=str, required=True,
                        help='Root directory of the dataset')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--nfolds', type=int, default=10,
                        help='Number of folds for cross-validation')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Load model
    model = load_model(args.model, args.network, args.device)
    
    # Read pairs file
    pairs = read_pairs_file(args.pairs)
    
    # Extract embeddings
    embeddings1, embeddings2, issame_list = extract_embeddings(
        model, pairs, args.dataset_root, args.batch_size, args.device
    )
    
    # Evaluate
    results = evaluate_verification(embeddings1, embeddings2, issame_list, args.nfolds)
    
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()