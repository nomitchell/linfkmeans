import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Tuple, Optional

class QuantizationLayer(nn.Module):
    """Quantization layer that maps latent representations to cluster centers"""
    
    def __init__(self, num_classes: int, clusters_per_class: int, latent_dim: int):
        super().__init__()
        self.num_classes = num_classes
        self.clusters_per_class = clusters_per_class
        self.latent_dim = latent_dim
        
        # Initialize cluster centers for each class
        # Shape: [num_classes, clusters_per_class, latent_dim]
        self.centers = nn.Parameter(
            torch.randn(num_classes, clusters_per_class, latent_dim) * 0.1
        )
        
        self.training_mode = True
    
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        FIXED: Differentiable forward pass through quantization layer for adversarial training
        
        Args:
            x: Input tensor [batch_size, latent_dim]
            labels: Ground truth labels [batch_size] (only used for k-means loss computation, NOT for forward pass)
            
        Returns:
            quantized: Quantized representations [batch_size, latent_dim]
            info: Dictionary containing assignment information
        """
        batch_size = x.size(0)
        
        # FIXED: Always use all centers during forward pass (don't cheat with ground truth labels!)
        # The model must learn to select the correct class centers without knowing the answer
        all_centers = self.centers.view(-1, self.latent_dim)  # [num_classes*clusters_per_class, latent_dim]
        
        # Ensure both tensors have the same dtype for torch.cdist (fixes AMP dtype mismatch)
        all_centers = all_centers.to(dtype=x.dtype)
        
        # Compute all distances at once using torch.cdist
        distances = torch.cdist(x, all_centers, p=2)  # [batch_size, num_classes*clusters_per_class]
        
        # Find closest centers globally
        closest_indices = torch.argmin(distances, dim=1)  # [batch_size]
        
        # Convert to class and cluster indices
        class_predictions = closest_indices // self.clusters_per_class
        cluster_assignments = closest_indices % self.clusters_per_class
        
        # Get quantized representations
        quantized_hard = all_centers[closest_indices]  # [batch_size, latent_dim]
        
        # Get minimum distances
        batch_indices = torch.arange(batch_size, device=x.device)
        min_distances = distances[batch_indices, closest_indices]  # [batch_size]
        
        # STRAIGHT-THROUGH ESTIMATOR for training:
        # Forward pass: use hard quantization (maintains discrete assignment)
        # Backward pass: gradients flow through as if quantization was identity
        if self.training and x.requires_grad:
            # STE formula: output = input + (quantized - input).detach()
            # This preserves discrete quantization in forward pass while allowing gradients
            quantized = x + (quantized_hard - x).detach()
            # Ensure the output requires gradients for proper backpropagation
            quantized.requires_grad_(True)
        else:
            quantized = quantized_hard
        
        info = {
            'assignments': cluster_assignments,
            'distances': min_distances,
            'centers': self.centers.clone(),
            'class_predictions': class_predictions,
            'ground_truth_labels': labels  # Store for k-means loss computation
        }
        
        return quantized, info

class DecomposedResNet18(nn.Module):
    """ResNet18 decomposed into g_phi and g_theta with optional quantization"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.use_quantization = config.use_quantization
        self.quantization_split = config.quantization_split
        
        # Load ResNet18 and modify for CIFAR-10
        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()  # Remove maxpool for CIFAR-10
        resnet.fc = nn.Linear(resnet.fc.in_features, self.num_classes)
        
        # Split the network based on configuration
        self._split_network(resnet)
        
        # Add quantization layer if enabled
        if self.use_quantization:
            self.quantization = QuantizationLayer(
                num_classes=config.num_classes,
                clusters_per_class=config.clusters_per_class,
                latent_dim=self.latent_dim
            )
        else:
            self.quantization = None
    
    def _split_network(self, resnet: nn.Module):
        """Split ResNet18 into g_phi and g_theta at specified layer"""
        
        # Common layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.layer1 = resnet.layer1
        
        if self.quantization_split == 'layer1':
            # g_phi: conv1 + bn1 + relu + layer1
            self.g_phi = nn.Sequential(
                self.conv1, self.bn1, self.relu, self.layer1
            )
            # g_theta: layer2 + layer3 + layer4 + avgpool + fc
            self.g_theta = nn.Sequential(
                resnet.layer2, resnet.layer3, resnet.layer4,
                resnet.avgpool, nn.Flatten(), resnet.fc
            )
            self.latent_dim = 64  # Output channels of layer1
            
        elif self.quantization_split == 'layer2':
            self.layer2 = resnet.layer2
            self.g_phi = nn.Sequential(
                self.conv1, self.bn1, self.relu, self.layer1, self.layer2
            )
            self.g_theta = nn.Sequential(
                resnet.layer3, resnet.layer4,
                resnet.avgpool, nn.Flatten(), resnet.fc
            )
            self.latent_dim = 128  # Output channels of layer2
            
        elif self.quantization_split == 'layer3':
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
            self.g_phi = nn.Sequential(
                self.conv1, self.bn1, self.relu, self.layer1, self.layer2, self.layer3
            )
            self.g_theta = nn.Sequential(
                resnet.layer4, resnet.avgpool, nn.Flatten(), resnet.fc
            )
            self.latent_dim = 256  # Output channels of layer3
            
        elif self.quantization_split == 'layer4':
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
            self.layer4 = resnet.layer4
            self.g_phi = nn.Sequential(
                self.conv1, self.bn1, self.relu, self.layer1, 
                self.layer2, self.layer3, self.layer4
            )
            self.g_theta = nn.Sequential(
                resnet.avgpool, nn.Flatten(), resnet.fc
            )
            self.latent_dim = 512  # Output channels of layer4
        else:
            raise ValueError(f"Invalid quantization split: {self.quantization_split}")
    
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None, return_latent: bool = False):
        """
        Forward pass through the decomposed network with gradient debugging
        
        Args:
            x: Input tensor [batch_size, 3, 32, 32]
            labels: Ground truth labels (for training)
            return_latent: Whether to return latent representations
            
        Returns:
            If return_latent=False: logits [batch_size, num_classes]
            If return_latent=True: (logits, latent_features, quantization_info)
        """
        # DEBUG: Check if input requires gradients (indicates adversarial training)
        debug_adv_training = x.requires_grad
        
        # Forward through g_phi to get latent representation
        latent = self.g_phi(x)
        
        # DEBUG: Verify gradient flow through g_phi
        if debug_adv_training and not latent.requires_grad:
            print("Warning: Gradients not flowing through g_phi")
            print(f"  Input requires_grad: {x.requires_grad}")
            print(f"  Latent shape: {latent.shape}, device: {latent.device}")
        
        # Flatten spatial dimensions for quantization
        batch_size = latent.size(0)
        spatial_dims = latent.shape[2:]  # Save spatial dimensions
        latent_flat = latent.view(batch_size, self.latent_dim, -1).mean(dim=2)  # Global average pooling
        
        quantization_info = None
        if self.use_quantization and self.quantization is not None:
            # Apply quantization
            quantized_latent, quantization_info = self.quantization(latent_flat, labels)
            
            # DEBUG: Verify gradient flow through quantization
            if debug_adv_training and not quantized_latent.requires_grad:
                print("Warning: Gradients not flowing through quantization layer")
                print(f"  Quantization training mode: {self.quantization.training}")
                print(f"  Input to quantization requires_grad: {latent_flat.requires_grad}")
                print(f"  Quantized output requires_grad: {quantized_latent.requires_grad}")
                print(f"  Quantized output shape: {quantized_latent.shape}")
            
            # Reshape back to spatial dimensions if needed
            if len(spatial_dims) > 0:
                spatial_size = spatial_dims[0] * spatial_dims[1] if len(spatial_dims) == 2 else spatial_dims[0]
                latent_for_theta = quantized_latent.unsqueeze(-1).expand(-1, -1, spatial_size).view(
                    batch_size, self.latent_dim, *spatial_dims
                )
            else:
                latent_for_theta = quantized_latent
        else:
            latent_for_theta = latent
        
        # Forward through g_theta to get final predictions
        logits = self.g_theta(latent_for_theta)
        
        # DEBUG: Verify final gradient flow
        if debug_adv_training and not logits.requires_grad:
            print("Warning: Gradients not flowing through g_theta")
        
        if return_latent:
            return logits, latent_flat, quantization_info
        else:
            return logits
    
    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation from g_phi"""
        with torch.no_grad():
            latent = self.g_phi(x)
            batch_size = latent.size(0)
            latent_flat = latent.view(batch_size, self.latent_dim, -1).mean(dim=2)
            return latent_flat
    
    def classify_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Classify from latent representation using g_theta"""
        batch_size = latent.size(0)
        
        # Determine spatial dimensions based on quantization split
        if self.quantization_split == 'layer1':
            spatial_dims = (32, 32)  # After layer1
        elif self.quantization_split == 'layer2':
            spatial_dims = (16, 16)  # After layer2
        elif self.quantization_split == 'layer3':
            spatial_dims = (8, 8)    # After layer3
        elif self.quantization_split == 'layer4':
            spatial_dims = (4, 4)    # After layer4
        
        # Reshape latent to expected spatial dimensions
        spatial_size = spatial_dims[0] * spatial_dims[1]
        latent_reshaped = latent.unsqueeze(-1).expand(-1, -1, spatial_size).view(
            batch_size, self.latent_dim, *spatial_dims
        )
        
        return self.g_theta(latent_reshaped)

def create_model(config):
    """Factory function to create the model"""
    config.validate_config()
    model = DecomposedResNet18(config)
    return model.to(config.device)
