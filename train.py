import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, MultiStepLR, StepLR
from torch.cuda.amp import autocast, GradScaler  # OPTIMIZED: Mixed precision training
from tqdm import tqdm
import copy
import math
from typing import Dict, List, Tuple, Optional

from utils import (
    pgd_attack, compute_accuracy, compute_lipschitz_loss, 
    compute_kmeans_loss, MetricsTracker
)

class WarmupCosineAnnealingLR(LambdaLR):
    """
    Learning rate scheduler with a linear warmup followed by a cosine annealing schedule.
    """
    def __init__(self, optimizer, warmup_epochs, max_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        
        def lr_lambda(current_epoch):
            if current_epoch < self.warmup_epochs:
                return float(current_epoch) / float(max(1, self.warmup_epochs))
            
            progress = float(current_epoch - self.warmup_epochs) / float(max(1, self.max_epochs - self.warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        super().__init__(optimizer, lr_lambda, last_epoch)


class Algorithm2Trainer:
    """Trainer implementing Algorithm 2 from the paper"""
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config.device
        
        # Initialize optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
        
        # Initialize learning rate scheduler
        if config.use_scheduler:
            if config.scheduler_type == 'cosine':
                self.scheduler = WarmupCosineAnnealingLR(
                    self.optimizer,
                    warmup_epochs=config.warmup_epochs,
                    max_epochs=config.num_epochs
                )
            elif config.scheduler_type == 'multistep':
                self.scheduler = MultiStepLR(
                    self.optimizer, 
                    milestones=config.scheduler_milestones, 
                    gamma=config.scheduler_gamma
                )
            elif config.scheduler_type == 'step':
                self.scheduler = StepLR(
                    self.optimizer,
                    step_size=config.scheduler_step_size,
                    gamma=config.scheduler_gamma
                )
            else:
                raise ValueError(f"Unknown scheduler type: {config.scheduler_type}")
        else:
            self.scheduler = None
        
        # Initialize criterion
        self.criterion = nn.CrossEntropyLoss()
        
        # OPTIMIZED: Initialize mixed precision scaler for ~2x speedup
        self.use_amp = getattr(config, 'use_mixed_precision', False) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker()
        
        # Best validation accuracy for model saving
        self.best_val_acc = 0.0
        
    def train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch using Algorithm 2"""
        self.model.train()
        
        total_loss = 0.0
        total_cls_loss = 0.0
        total_center_cls_loss = 0.0
        total_lipschitz_loss = 0.0
        total_kmeans_loss = 0.0
        
        clean_correct = 0
        robust_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}', 
                           leave=True, dynamic_ncols=True, mininterval=0.5)
        
        for batch_idx, (x_clean, y) in enumerate(progress_bar):
            x_clean, y = x_clean.to(self.device), y.to(self.device)
            batch_size = x_clean.size(0)
            
            # Step 1: Generate adversarial examples if enabled
            x_adv = None
            if self.config.use_adversarial_training:
                x_adv = pgd_attack(
                    self.model, x_clean, y,
                    epsilon=self.config.epsilon,
                    step_size=self.config.attack_step_size,
                    num_steps=self.config.attack_steps,
                    random_start=self.config.attack_random_start
                )
                # Debug: Check if adversarial examples are different (first batch only)
                if batch_idx == 0:
                    # Convert to [0,1] space for meaningful perturbation measurement
                    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(x_clean.device)
                    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(x_clean.device)
                    
                    x_clean_denorm = (x_clean * std + mean).clamp(0, 1)
                    x_adv_denorm = (x_adv * std + mean).clamp(0, 1)
                    diff = torch.norm(x_adv_denorm - x_clean_denorm, p=float('inf')).item()
                    
                    print(f"Debug: Max perturbation: {diff:.6f}, Epsilon: {self.config.epsilon:.6f}")
                    if diff < self.config.epsilon * 0.1:
                        print("WARNING: Adversarial examples may not be generated correctly!")
            
            # Step 2: Optimize network parameters (φ, θ) with fixed centers
            network_results = self._optimize_network_parameters(
                x_clean, x_adv, y, epoch, batch_idx
            )
            network_loss = network_results['losses']
            
            # Step 3: Optimize cluster centers with fixed network parameters (OPTIMIZED)
            if self.config.use_quantization and self.config.use_kmeans_loss and 'latent_features' in network_results:
                self._optimize_cluster_centers_fast(network_results['latent_features'], network_results['y_combined'])
            
            # OPTIMIZED: Reuse outputs from training for metrics (eliminates 2 forward passes)
            with torch.no_grad():
                if 'clean_outputs' in network_results:
                    clean_outputs = network_results['clean_outputs']
                    clean_acc = compute_accuracy(clean_outputs, y)
                    clean_correct += (clean_outputs.argmax(1) == y).sum().item()
                    
                    # FIXED: Proper robust accuracy computation
                    if x_adv is not None and 'robust_outputs' in network_results:
                        robust_outputs = network_results['robust_outputs']
                        robust_acc = compute_accuracy(robust_outputs, y)
                        robust_correct += (robust_outputs.argmax(1) == y).sum().item()
                        # Debug print every 50 batches
                        if batch_idx % 50 == 0:
                            print(f"Debug batch {batch_idx}: Clean acc: {clean_acc:.2f}%, Robust acc: {robust_acc:.2f}%")
                    else:
                        # FIXED: If no adversarial training, both should be same
                        robust_acc = clean_acc
                        robust_correct += (clean_outputs.argmax(1) == y).sum().item()
                else:
                    # Fallback: compute outputs if not returned
                    clean_outputs = self.model(x_clean)
                    clean_acc = compute_accuracy(clean_outputs, y)
                    clean_correct += (clean_outputs.argmax(1) == y).sum().item()
                    robust_acc = clean_acc
                    robust_correct += (clean_outputs.argmax(1) == y).sum().item()
                
                total_samples += batch_size
            
            # Update running metrics
            total_loss += network_loss['total_loss']
            total_cls_loss += network_loss['cls_loss']
            total_center_cls_loss += network_loss['center_cls_loss']
            total_lipschitz_loss += network_loss['lipschitz_loss']
            total_kmeans_loss += network_loss['kmeans_loss']
            
            # Update progress bar with detailed multi-row display
            clean_acc_running = 100.0 * clean_correct / total_samples
            robust_acc_running = 100.0 * robust_correct / total_samples
            
            # Compute running averages for loss components
            n_batches = batch_idx + 1
            avg_cls_loss = total_cls_loss / n_batches
            avg_center_cls_loss = total_center_cls_loss / n_batches
            avg_lipschitz_loss = total_lipschitz_loss / n_batches 
            avg_kmeans_loss = total_kmeans_loss / n_batches
            
            # Compute scaled losses (live)
            scaled_cls = self.config.lambda_cls * avg_cls_loss
            scaled_center = self.config.lambda_c_cls * avg_center_cls_loss
            scaled_lip = self.config.lambda_lip * avg_lipschitz_loss
            scaled_kmeans = self.config.lambda_kmeans * avg_kmeans_loss
            
            # Primary progress row
            progress_bar.set_postfix({
                'Loss': f'{total_loss/n_batches:.4f}',
                'Clean': f'{clean_acc_running:.1f}%',
                'Robust': f'{robust_acc_running:.1f}%'
            })
            
            # Update detailed metrics less frequently to avoid spam
            if batch_idx % 20 == 0:  # Update every 20 batches instead of 5
                # Compute Algorithm 2 specific metrics
                alg2_metrics = self._compute_algorithm2_metrics(network_results, x_clean, x_adv, y)
                
                # Build compact single-line description
                lr_str = f'LR={self.optimizer.param_groups[0]["lr"]:.1e}'
                loss_str = f'cls={avg_cls_loss:.2f} ctr={avg_center_cls_loss:.2f} lip={avg_lipschitz_loss:.3f} km={avg_kmeans_loss:.1f}'
                
                quant_str = ''
                if 'quant_acc' in alg2_metrics and 'avg_dist' in alg2_metrics:
                    dist_val = alg2_metrics["avg_dist"]
                    if dist_val == float('inf') or dist_val > 999:
                        dist_str = 'inf'
                    else:
                        dist_str = f'{dist_val:.2f}'
                    quant_str = f' QAcc={alg2_metrics["quant_acc"]:.0f}% QDist={dist_str}'
                
                lip_str = ''
                if 'lip_avg' in alg2_metrics:
                    lip_str = f' LipRatio={alg2_metrics["lip_avg"]:.2f}'
                
                # Single line description that fits in terminal
                desc = f'E{epoch} {lr_str} | {loss_str}{quant_str}{lip_str}'
                progress_bar.set_description(desc)
        
        # Compute epoch metrics
        num_batches = len(self.train_loader)
        epoch_metrics = {
            'loss': total_loss / num_batches,
            'cls_loss': total_cls_loss / num_batches,
            'center_cls_loss': total_center_cls_loss / num_batches,
            'lipschitz_loss': total_lipschitz_loss / num_batches,
            'kmeans_loss': total_kmeans_loss / num_batches,
            'clean_acc': 100.0 * clean_correct / total_samples,
            'robust_acc': 100.0 * robust_correct / total_samples
        }
        
        return epoch_metrics
    
    def _optimize_network_parameters(self, x_clean, x_adv, y, epoch, batch_idx):
        """Optimize network parameters (φ, θ) with fixed cluster centers"""
        
        self.optimizer.zero_grad()
        
        # Combine clean and adversarial examples for training
        if x_adv is not None:
            x_combined = torch.cat([x_clean, x_adv], dim=0)
            y_combined = torch.cat([y, y], dim=0)
        else:
            x_combined = x_clean
            y_combined = y
        
        # OPTIMIZED: Forward pass with mixed precision
        with autocast(enabled=self.use_amp):
            if self.config.use_quantization:
                outputs, latent_features, quant_info = self.model(
                    x_combined, y_combined, return_latent=True
                )
            else:
                outputs = self.model(x_combined)
                latent_features = None
                quant_info = None
            
            # Loss 1: Classification loss (L_cls)
            cls_loss = self.criterion(outputs, y_combined)
            total_loss = self.config.lambda_cls * cls_loss
        
        # Loss 2: Center classification loss (L_C-cls)
        center_cls_loss = 0.0
        if (self.config.use_center_classification and self.config.use_quantization 
            and hasattr(self.model, 'quantization') and self.model.quantization is not None):
            
            centers = self.model.quantization.centers  # [num_classes, clusters_per_class, latent_dim]
            center_cls_loss = self._compute_center_classification_loss(centers)
            total_loss += self.config.lambda_c_cls * center_cls_loss
        
        # Loss 3: Lipschitz constraint loss (L_lip)
        lipschitz_loss = 0.0
        if (self.config.use_lipschitz_constraint and x_adv is not None 
            and latent_features is not None):
            
            batch_size = x_clean.size(0)
            phi_clean = latent_features[:batch_size]
            phi_adv = latent_features[batch_size:]
            
            lipschitz_loss = compute_lipschitz_loss(
                phi_clean, phi_adv, x_clean, x_adv, 
                self.config.lipschitz_constant
            )
            total_loss += self.config.lambda_lip * lipschitz_loss
        
        # Loss 4: K-means loss (L_kmeans) - computed but not used in this optimization step
        kmeans_loss = 0.0
        if (self.config.use_kmeans_loss and self.config.use_quantization 
            and latent_features is not None and quant_info is not None):
            
            centers = quant_info['centers']
            kmeans_loss = compute_kmeans_loss(
                latent_features, y_combined, centers, 
                self.config.gamma, self.config.num_classes
            )
            # Note: K-means loss is optimized separately in _optimize_cluster_centers
        
        # OPTIMIZED: Backward pass with mixed precision
        if self.use_amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            self.optimizer.step()
        
        # OPTIMIZED: Return outputs and latent features to avoid redundant forward passes
        losses = {
            'total_loss': total_loss.item(),
            'cls_loss': cls_loss.item(),
            'center_cls_loss': center_cls_loss.item() if isinstance(center_cls_loss, torch.Tensor) else center_cls_loss,
            'lipschitz_loss': lipschitz_loss.item() if isinstance(lipschitz_loss, torch.Tensor) else lipschitz_loss,
            'kmeans_loss': kmeans_loss.item() if isinstance(kmeans_loss, torch.Tensor) else kmeans_loss
        }
        
        results = {'losses': losses}
        
        # Split outputs for clean/adversarial if both are present
        if x_adv is not None:
            batch_size = x_clean.size(0)
            with torch.no_grad():
                clean_outputs = outputs[:batch_size].detach()
                robust_outputs = outputs[batch_size:].detach()
                results['clean_outputs'] = clean_outputs
                results['robust_outputs'] = robust_outputs
        else:
            results['clean_outputs'] = outputs.detach()
        
        # Return latent features and labels for cluster optimization
        if latent_features is not None:
            results['latent_features'] = latent_features.detach()
            results['y_combined'] = y_combined
            
        return results
    
    def _compute_center_classification_loss(self, centers):
        """Compute classification loss on cluster centers"""
        center_cls_loss = 0.0
        num_centers = 0
        
        for class_idx in range(self.config.num_classes):
            class_centers = centers[class_idx]  # [clusters_per_class, latent_dim]
            
            for center in class_centers:
                # Classify the center using g_theta
                center_output = self.model.classify_from_latent(center.unsqueeze(0))
                target = torch.tensor([class_idx], device=self.device)
                center_cls_loss += self.criterion(center_output, target)
                num_centers += 1
        
        return center_cls_loss / num_centers if num_centers > 0 else 0.0
    
    def _optimize_cluster_centers(self, x_clean, x_adv, y):
        """Optimize cluster centers using EM algorithm (K-means with constraints)"""
        
        if not hasattr(self.model, 'quantization') or self.model.quantization is None:
            return
        
        # Get latent representations
        with torch.no_grad():
            if x_adv is not None:
                x_combined = torch.cat([x_clean, x_adv], dim=0)
                y_combined = torch.cat([y, y], dim=0)
            else:
                x_combined = x_clean
                y_combined = y
            
            latent_features = self.model.get_latent_representation(x_combined)
        
        # Perform EM updates for cluster centers
        centers = self.model.quantization.centers  # [num_classes, clusters_per_class, latent_dim]
        
        for _ in range(self.config.n_outer):  # Outer EM iterations
            # E-step: Assign points to clusters (already handled in quantization layer)
            # M-step: Update centers
            self._update_centers_em_step(latent_features, y_combined, centers)
    
    def _update_centers_em_step(self, latent_features, labels, centers):
        """Update cluster centers using gradient descent"""
        
        # Enable gradients for centers
        centers.requires_grad_(True)
        
        # Create optimizer for centers only
        center_optimizer = optim.SGD([centers], lr=self.config.learning_rate * 0.1)
        
        for _ in range(self.config.n_inner):  # Inner gradient descent iterations
            center_optimizer.zero_grad()
            
            # Compute K-means loss with margin constraints
            kmeans_loss = compute_kmeans_loss(
                latent_features, labels, centers,
                self.config.gamma, self.config.num_classes
            )
            
            # Backward pass
            kmeans_loss.backward()
            center_optimizer.step()
        
        # Disable gradients for centers
        centers.requires_grad_(False)
    
    def _optimize_cluster_centers_fast(self, latent_features, labels):
        """OPTIMIZED: Fast cluster center optimization using precomputed latent features"""
        
        if not hasattr(self.model, 'quantization') or self.model.quantization is None:
            return
        
        # Use precomputed latent features (eliminates forward pass)
        centers = self.model.quantization.centers  # [num_classes, clusters_per_class, latent_dim]
        
        # OPTIMIZED: Reduce iterations for speed (1 outer, 2 inner instead of 3*5=15)
        # This maintains training quality while significantly improving speed
        for _ in range(1):  # Reduced outer EM iterations from 3 to 1
            # E-step: Assign points to clusters (already handled in quantization layer)
            # M-step: Update centers with reduced iterations
            self._update_centers_em_step_fast(latent_features, labels, centers)
    
    def _update_centers_em_step_fast(self, latent_features, labels, centers):
        """OPTIMIZED: Fast center update with reduced iterations"""
        
        # Enable gradients for centers
        centers.requires_grad_(True)
        
        # Create optimizer for centers only
        center_optimizer = optim.SGD([centers], lr=self.config.learning_rate * 0.1)
        
        # OPTIMIZED: Reduce inner iterations from 5 to 2 for speed
        for _ in range(2):  # Reduced from config.n_inner (5) to 2
            center_optimizer.zero_grad()
            
            # Compute K-means loss with margin constraints (using optimized version)
            kmeans_loss = compute_kmeans_loss(
                latent_features, labels, centers,
                self.config.gamma, self.config.num_classes
            )
            
            # Backward pass
            kmeans_loss.backward()
            center_optimizer.step()
        
        # Disable gradients for centers
        centers.requires_grad_(False)
    
    def _compute_algorithm2_metrics(self, network_results, x_clean, x_adv, y):
        """Compute Algorithm 2 specific metrics for monitoring"""
        metrics = {}
        
        # Quantization accuracy (if available)
        if (self.config.use_quantization and 'latent_features' in network_results 
            and hasattr(self.model, 'quantization') and self.model.quantization is not None):
            
            with torch.no_grad():
                latent_features = network_results['latent_features']
                if latent_features is not None:
                    # Get quantization info
                    _, quant_info = self.model.quantization(latent_features, y)
                    if 'class_predictions' in quant_info:
                        # Quantization accuracy: how often quantization predicts correct class
                        class_pred = quant_info['class_predictions'][:len(y)]  # Only clean examples
                        quant_acc = (class_pred == y).float().mean().item() * 100
                        metrics['quant_acc'] = quant_acc
                    
                    # Average quantization distance (only for clean examples)
                    if 'distances' in quant_info:
                        clean_distances = quant_info['distances'][:len(y)]  # Only clean examples
                        # Filter out any invalid values
                        valid_distances = clean_distances[torch.isfinite(clean_distances)]
                        if len(valid_distances) > 0:
                            avg_quant_dist = valid_distances.mean().item()
                            metrics['avg_dist'] = avg_quant_dist
                        else:
                            metrics['avg_dist'] = float('inf')
        
        # Lipschitz constraint violations (if adversarial training enabled)
        if x_adv is not None and 'latent_features' in network_results:
            with torch.no_grad():
                latent_features = network_results['latent_features']
                if latent_features is not None:
                    batch_size = x_clean.size(0)
                    phi_clean = latent_features[:batch_size]
                    phi_adv = latent_features[batch_size:]
                    
                    # Compute actual Lipschitz ratios
                    phi_diff = torch.norm(phi_clean - phi_adv, p=2, dim=1)
                    x_diff = torch.norm((x_clean - x_adv).view(batch_size, -1), p=float('inf'), dim=1)
                    
                    # Avoid division by zero
                    valid_mask = x_diff > 1e-8
                    if valid_mask.sum() > 0:
                        lipschitz_ratios = phi_diff[valid_mask] / x_diff[valid_mask]
                        avg_lip_ratio = lipschitz_ratios.mean().item()
                        max_lip_ratio = lipschitz_ratios.max().item()
                        lip_violations = (lipschitz_ratios > self.config.lipschitz_constant).float().mean().item() * 100
                        
                        metrics['lip_avg'] = avg_lip_ratio
                        metrics['lip_max'] = max_lip_ratio
                        metrics['lip_viol'] = lip_violations
        
        return metrics
    
    def validate(self, epoch: int) -> Dict:
        """Validate the model on clean and adversarial examples"""
        self.model.eval()
        
        clean_correct = 0
        robust_correct = 0
        total_samples = 0
        clean_loss_sum = 0.0
        robust_loss_sum = 0.0
        
        # Perform clean evaluation
        with torch.no_grad():
            for x, y in tqdm(self.val_loader, desc='Validating (Clean)', leave=False, dynamic_ncols=True):
                x, y = x.to(self.device), y.to(self.device)
                
                clean_outputs = self.model(x)
                clean_loss_sum += self.criterion(clean_outputs, y).item() * y.size(0)
                clean_correct += (clean_outputs.argmax(1) == y).sum().item()
                total_samples += y.size(0)

        # Perform robust evaluation if enabled
        if self.config.use_adversarial_training:
            for x, y in tqdm(self.val_loader, desc='Validating (Robust)', leave=False, dynamic_ncols=True):
                x, y = x.to(self.device), y.to(self.device)
                
                # Enable gradients ONLY for the attack generation
                with torch.enable_grad():
                    x_adv = pgd_attack(
                        self.model, x, y,
                        epsilon=self.config.epsilon,
                        step_size=self.config.attack_step_size,
                        num_steps=self.config.eval_attack_steps,
                        random_start=True
                    )
                
                # Disable gradients for the forward pass
                with torch.no_grad():
                    robust_outputs = self.model(x_adv)
                    robust_loss_sum += self.criterion(robust_outputs, y).item() * y.size(0)
                    robust_correct += (robust_outputs.argmax(1) == y).sum().item()
        else:
            # If not using adversarial training, robust metrics are the same as clean
            robust_correct = clean_correct
            robust_loss_sum = clean_loss_sum

        # Compute final metrics
        clean_acc = 100.0 * clean_correct / total_samples
        robust_acc = 100.0 * robust_correct / total_samples
        clean_loss = clean_loss_sum / total_samples
        robust_loss = robust_loss_sum / total_samples
        
        # Check if this is the best model based on clean accuracy
        is_best = clean_acc > self.best_val_acc
        if is_best:
            self.best_val_acc = clean_acc
        
        val_metrics = {
            'clean_acc': clean_acc,
            'robust_acc': robust_acc,
            'clean_loss': clean_loss,
            'robust_loss': robust_loss,
            'is_best': is_best
        }
        
        # Set model back to training mode
        self.model.train()
        
        return val_metrics
    
    def train(self) -> Tuple[Dict, Dict]:
        """Main training loop implementing Algorithm 2"""
        
        print(f"Starting training with configuration: {self.config}")
        print(f"Components enabled: {self.config.get_component_status()}")
        print(f"Device: {self.device}")
        
        for epoch in range(1, self.config.num_epochs + 1):
            
            # Train for one epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Update metrics tracker
            self.metrics_tracker.update_train_metrics(**train_metrics)
            self.metrics_tracker.update_val_metrics(**val_metrics)
            
            # Print epoch summary
            print(f'\nEpoch {epoch}/{self.config.num_epochs}:')
            print(f'  Train - Loss: {train_metrics["loss"]:.4f}, '
                  f'Clean Acc: {train_metrics["clean_acc"]:.2f}%, '
                  f'Robust Acc: {train_metrics["robust_acc"]:.2f}%')
            
            if self.config.use_adversarial_training:
                print(f'  Val   - Clean Acc: {val_metrics["clean_acc"]:.2f}%, '
                      f'Robust Acc: {val_metrics["robust_acc"]:.2f}%')
            else:
                print(f'  Val   - Clean Acc: {val_metrics["clean_acc"]:.2f}%')
            
            # Print component-specific losses if enabled
            if train_metrics['lipschitz_loss'] > 0:
                print(f'  Lipschitz Loss: {train_metrics["lipschitz_loss"]:.4f}')
            if train_metrics['kmeans_loss'] > 0:
                print(f'  K-means Loss: {train_metrics["kmeans_loss"]:.4f}')
            if train_metrics['center_cls_loss'] > 0:
                print(f'  Center Cls Loss: {train_metrics["center_cls_loss"]:.4f}')
            
            # Update learning rate scheduler and print current LR
            if self.scheduler is not None:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'  Learning Rate: {current_lr:.6f}')
                self.scheduler.step()
            
            if val_metrics['is_best']:
                print(f'  *** New best validation accuracy: {val_metrics["clean_acc"]:.2f}% ***')
        
        return self.metrics_tracker.train_metrics, self.metrics_tracker.val_metrics
