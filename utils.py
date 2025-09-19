import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

def get_cifar10_dataloaders(config):
    """Create CIFAR-10 train and test dataloaders"""
    
    # CIFAR-10 normalization
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    
    # Training transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Test transforms
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Create datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=config.data_dir, train=True, download=True, transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=config.data_dir, train=False, download=True, transform=test_transform
    )
    
    # OPTIMIZED: Create dataloaders with performance enhancements
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, 
        num_workers=config.num_workers, 
        pin_memory=getattr(config, 'pin_memory', True),
        persistent_workers=getattr(config, 'persistent_workers', True) and config.num_workers > 0,
        prefetch_factor=getattr(config, 'prefetch_factor', 2) if config.num_workers > 0 else None
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers,
        pin_memory=getattr(config, 'pin_memory', True),
        persistent_workers=getattr(config, 'persistent_workers', True) and config.num_workers > 0,
        prefetch_factor=getattr(config, 'prefetch_factor', 2) if config.num_workers > 0 else None
    )
    
    return train_loader, test_loader

def pgd_attack(model, x, y, epsilon, step_size, num_steps, random_start=True, targeted=False):
    """
    FIXED: PGD attack implementation with proper gradient flow for quantized models
    
    Args:
        model: The model to attack
        x: Input tensor [batch_size, channels, height, width] (normalized)
        y: Target labels [batch_size]
        epsilon: Maximum perturbation magnitude in [0,1] space
        step_size: Step size for each iteration in [0,1] space
        num_steps: Number of attack iterations
        random_start: Whether to start from random perturbation
        targeted: Whether this is a targeted attack
        
    Returns:
        Adversarial examples [batch_size, channels, height, width] (normalized)
    """
    # Store original model training state
    was_training = model.training
    
    # FIXED: Always use training mode for quantized models to ensure gradient flow
    # The quantization layer requires training mode for proper gradient computation
    if hasattr(model, 'quantization') and model.quantization is not None:
        model.train()  # Essential for gradient flow through quantization
        # Ensure quantization layer is in training mode
        model.quantization.train()
    else:
        model.eval()  # Use eval mode for non-quantized models
    
    # CIFAR-10 normalization parameters
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(x.device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(x.device)
    
    # Denormalize input to [0,1] space for attack
    x_denorm = x * std + mean
    x_denorm = torch.clamp(x_denorm, 0, 1)  # Ensure valid [0,1] range
    
    x_denorm = x_denorm.clone().detach()
    y = y.clone().detach()
    
    # Initialize perturbation in [0,1] space
    if random_start:
        delta = torch.empty_like(x_denorm).uniform_(-epsilon, epsilon)
        delta = torch.clamp(delta, -epsilon, epsilon)
    else:
        delta = torch.zeros_like(x_denorm)
    
    # FIXED: Ensure proper gradient setup
    delta = delta.detach().requires_grad_(True)
    
    successful_steps = 0
    for step in range(num_steps):
        # FIXED: Recreate delta variable with gradients for each step
        delta = delta.detach().requires_grad_(True)
        
        # Create adversarial examples in [0,1] space
        adv_x_denorm = x_denorm + delta
        adv_x_denorm = torch.clamp(adv_x_denorm, 0, 1)  # Ensure valid [0,1] range
        
        # Renormalize for model input - this maintains gradient connection to delta
        adv_x_norm = (adv_x_denorm - mean) / std
        
        # Forward pass with labels for quantized models
        if hasattr(model, 'quantization') and model.quantization is not None:
            # For quantized models, pass labels to ensure proper quantization
            outputs = model(adv_x_norm, y)
        else:
            outputs = model(adv_x_norm)
        
        # Compute loss
        loss = F.cross_entropy(outputs, y)
        if targeted:
            loss = -loss
        
        # FIXED: Direct backward pass and gradient extraction
        try:
            loss.backward()
            
            if delta.grad is not None:
                successful_steps += 1
                # Update perturbation in [0,1] space using gradient sign
                grad_sign = delta.grad.sign()
                delta.data = delta.data + step_size * grad_sign
                delta.data = torch.clamp(delta.data, -epsilon, epsilon)
                delta.data = torch.clamp(x_denorm + delta.data, 0, 1) - x_denorm
            else:
                print(f"Warning: No gradients computed in PGD step {step}")
                
        except RuntimeError as e:
            print(f"Warning: Gradient computation failed in PGD step {step}: {e}")
            continue
    
    # Debug information
    if successful_steps < num_steps:
        print(f"Warning: Only {successful_steps}/{num_steps} PGD steps were successful")
    
    # Restore original model training state
    if was_training:
        model.train()
    else:
        model.eval()
    
    # Create final adversarial example in [0,1] space, then renormalize
    final_adv_denorm = (x_denorm + delta.detach()).clamp(0, 1)
    final_adv_norm = (final_adv_denorm - mean) / std
    
    return final_adv_norm

def generate_adversarial_examples(model, dataloader, config, num_examples=None):
    """
    Generate adversarial examples for a dataset
    
    Args:
        model: Model to attack
        dataloader: Data loader
        config: Configuration object
        num_examples: Maximum number of examples to generate (None for all)
        
    Returns:
        List of (clean_x, adv_x, y) tuples
    """
    model.eval()
    adversarial_examples = []
    
    with torch.no_grad():
        total_processed = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            if num_examples is not None and total_processed >= num_examples:
                break
                
            x, y = x.to(config.device), y.to(config.device)
            
            # Generate adversarial examples
            adv_x = pgd_attack(
                model, x, y,
                epsilon=config.epsilon,
                step_size=config.attack_step_size,
                num_steps=config.attack_steps,
                random_start=config.attack_random_start
            )
            
            # Store examples
            for i in range(x.size(0)):
                if num_examples is not None and total_processed >= num_examples:
                    break
                adversarial_examples.append((x[i].cpu(), adv_x[i].cpu(), y[i].cpu()))
                total_processed += 1
    
    return adversarial_examples

def evaluate_model(model, dataloader, config, attack_params=None):
    """
    Evaluate model on clean and adversarial examples
    
    Args:
        model: Model to evaluate
        dataloader: Test data loader
        config: Configuration object
        attack_params: Dict with attack parameters (None for clean evaluation)
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Evaluating", leave=False):
            x, y = x.to(config.device), y.to(config.device)
            
            if attack_params is not None:
                # Generate adversarial examples
                x = pgd_attack(
                    model, x, y,
                    epsilon=attack_params.get('epsilon', config.epsilon),
                    step_size=attack_params.get('step_size', config.attack_step_size),
                    num_steps=attack_params.get('num_steps', config.eval_attack_steps),
                    random_start=attack_params.get('random_start', True)
                )
            
            # Forward pass
            outputs = model(x)
            loss = criterion(outputs, y)
            
            # Compute accuracy
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(y).sum().item()
            total_samples += y.size(0)
            total_loss += loss.item()
    
    accuracy = 100.0 * total_correct / total_samples
    avg_loss = total_loss / len(dataloader)
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'correct': total_correct,
        'total': total_samples
    }

def compute_accuracy(outputs, targets):
    """Compute accuracy from outputs and targets"""
    _, predicted = outputs.max(1)
    correct = predicted.eq(targets).sum().item()
    total = targets.size(0)
    return 100.0 * correct / total

def compute_lipschitz_loss(phi_clean, phi_adv, x_clean, x_adv, lipschitz_constant):
    """
    Compute Lipschitz constraint loss
    
    Args:
        phi_clean: Latent representations of clean examples
        phi_adv: Latent representations of adversarial examples  
        x_clean: Clean input examples
        x_adv: Adversarial input examples
        lipschitz_constant: Lipschitz constant L
        
    Returns:
        Lipschitz constraint loss
    """
    # Compute distances in latent space (L2 norm)
    latent_dist = torch.norm(phi_clean - phi_adv, dim=1, p=2)
    
    # Compute distances in input space (L∞ norm)
    input_dist = torch.norm((x_clean - x_adv).view(x_clean.size(0), -1), dim=1, p=float('inf'))
    
    # Compute Lipschitz violation: max(0, ||φ(x) - φ(x')||_2 / ||x - x'||_∞ - L)
    lipschitz_ratio = latent_dist / (input_dist + 1e-8)  # Add small epsilon to avoid division by zero
    lipschitz_violation = F.relu(lipschitz_ratio - lipschitz_constant)
    
    return lipschitz_violation.mean()

def compute_kmeans_loss(latent_representations, labels, centers, gamma, num_classes):
    """
    OPTIMIZED: Vectorized k-means loss with between-class margin constraint
    
    Args:
        latent_representations: Latent features [batch_size, latent_dim]
        labels: Ground truth labels [batch_size]
        centers: Cluster centers [num_classes, clusters_per_class, latent_dim]
        gamma: Between-class margin
        num_classes: Number of classes
        
    Returns:
        K-means loss with margin constraint
    """
    batch_size = latent_representations.size(0)
    device = latent_representations.device
    
    # OPTIMIZED: Vectorized within-class clustering loss
    # Gather class centers for each sample: [batch_size, clusters_per_class, latent_dim]
    batch_centers = centers[labels]  # Shape: [batch_size, clusters_per_class, latent_dim]
    
    # Compute distances efficiently using broadcasting
    # latent_representations: [batch_size, 1, latent_dim], batch_centers: [batch_size, clusters_per_class, latent_dim]
    x_expanded = latent_representations.unsqueeze(1)  # [batch_size, 1, latent_dim]
    distances = torch.norm(x_expanded - batch_centers, dim=2, p=2)  # [batch_size, clusters_per_class]
    
    # Find minimum distances for each sample
    min_distances = torch.min(distances, dim=1)[0]  # [batch_size]
    clustering_loss = min_distances.mean()  # Average over batch
    
    # OPTIMIZED: Vectorized between-class margin constraint
    # Flatten all centers to compute pairwise distances efficiently
    all_centers = centers.view(-1, centers.size(-1))  # [num_classes*clusters_per_class, latent_dim]
    
    # Create class labels for each center
    center_classes = torch.arange(num_classes, device=device).repeat_interleave(centers.size(1))  # [num_classes*clusters_per_class]
    
    # Compute all pairwise distances between centers
    pairwise_distances = torch.cdist(all_centers, all_centers, p=2)  # [num_classes*clusters_per_class, num_classes*clusters_per_class]
    
    # Create mask for inter-class pairs (exclude intra-class pairs)
    inter_class_mask = center_classes.unsqueeze(0) != center_classes.unsqueeze(1)  # [num_centers, num_centers]
    
    # Apply margin constraint only to inter-class pairs
    margin_violations = F.relu(gamma - pairwise_distances)  # [num_centers, num_centers]
    inter_class_violations = margin_violations * inter_class_mask.float()
    
    # Average over valid inter-class pairs
    num_inter_class_pairs = inter_class_mask.sum().float().clamp(min=1)
    margin_loss = inter_class_violations.sum() / num_inter_class_pairs
    
    return clustering_loss + margin_loss

def plot_training_curves(train_metrics, val_metrics, config):
    """
    Plot training curves for losses and accuracies
    
    Args:
        train_metrics: Dictionary with training metrics per epoch
        val_metrics: Dictionary with validation metrics per epoch
        config: Configuration object
    """
    epochs = range(1, len(train_metrics['loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training Results - {config.get_component_status()}', fontsize=16)
    
    # Training and validation loss
    axes[0, 0].plot(epochs, train_metrics['loss'], 'b-', label='Training Loss', alpha=0.7)
    axes[0, 0].plot(epochs, val_metrics['clean_loss'], 'g-', label='Val Clean Loss', alpha=0.7)
    if 'robust_loss' in val_metrics:
        axes[0, 0].plot(epochs, val_metrics['robust_loss'], 'r-', label='Val Robust Loss', alpha=0.7)
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Training accuracy
    axes[0, 1].plot(epochs, train_metrics['clean_acc'], 'b-', label='Train Clean Acc', alpha=0.7)
    if 'robust_acc' in train_metrics:
        axes[0, 1].plot(epochs, train_metrics['robust_acc'], 'r-', label='Train Robust Acc', alpha=0.7)
    axes[0, 1].set_title('Training Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Validation accuracy
    axes[1, 0].plot(epochs, val_metrics['clean_acc'], 'g-', label='Val Clean Acc', alpha=0.7)
    if 'robust_acc' in val_metrics:
        axes[1, 0].plot(epochs, val_metrics['robust_acc'], 'r-', label='Val Robust Acc', alpha=0.7)
    axes[1, 0].set_title('Validation Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Component-specific losses (if available)
    if 'lipschitz_loss' in train_metrics:
        axes[1, 1].plot(epochs, train_metrics['lipschitz_loss'], 'purple', label='Lipschitz Loss', alpha=0.7)
    if 'kmeans_loss' in train_metrics:
        axes[1, 1].plot(epochs, train_metrics['kmeans_loss'], 'orange', label='K-means Loss', alpha=0.7)
    if 'center_cls_loss' in train_metrics:
        axes[1, 1].plot(epochs, train_metrics['center_cls_loss'], 'brown', label='Center Cls Loss', alpha=0.7)
    
    axes[1, 1].set_title('Component Losses')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(config.save_dir, exist_ok=True)
    plot_path = os.path.join(config.save_dir, f'training_curves_{config.get_component_status()}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to {plot_path}")

def save_checkpoint(model, optimizer, epoch, train_metrics, val_metrics, config, is_best=False):
    """Save model checkpoint"""
    os.makedirs(config.save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'config': config
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(config.save_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = os.path.join(config.save_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)
        print(f"Best model saved to {best_path}")
    
    return checkpoint_path

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint

class MetricsTracker:
    """Track training and validation metrics"""
    
    def __init__(self):
        self.train_metrics = {
            'loss': [], 'clean_acc': [], 'robust_acc': [],
            'lipschitz_loss': [], 'kmeans_loss': [], 'center_cls_loss': []
        }
        self.val_metrics = {
            'clean_loss': [], 'clean_acc': [], 'robust_loss': [], 'robust_acc': []
        }
    
    def update_train_metrics(self, **kwargs):
        """Update training metrics"""
        for key, value in kwargs.items():
            if key in self.train_metrics:
                self.train_metrics[key].append(value)
    
    def update_val_metrics(self, **kwargs):
        """Update validation metrics"""
        for key, value in kwargs.items():
            if key in self.val_metrics:
                self.val_metrics[key].append(value)
    
    def get_latest_metrics(self):
        """Get latest metrics as a formatted string"""
        train_latest = {k: v[-1] if v else 0.0 for k, v in self.train_metrics.items()}
        val_latest = {k: v[-1] if v else 0.0 for k, v in self.val_metrics.items()}
        
        return {**train_latest, **val_latest}
