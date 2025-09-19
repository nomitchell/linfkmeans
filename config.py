import torch

class Config:
    """Configuration class for Algorithm 2 implementation"""
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data parameters
    data_dir = './data'
    batch_size = 256
    num_workers = 4
    num_classes = 10
    
    # Model parameters
    quantization_split = 'layer3'  # Where to split ResNet18: 'layer1', 'layer2', 'layer3', 'layer4'
    
    # Algorithm 2 hyperparameters
    epsilon = 4/255  # Perturbation budget
    lipschitz_constant = 1.0  # L in the paper
    clusters_per_class = 10  # κ in the paper
    gamma = None  # Between-class margin, computed as 2*(alpha + L*epsilon)
    alpha = 0.1  # Cluster radius
    
    # Loss coefficients (λ in the paper)
    lambda_cls = 1.0      # λ1: Classification loss weight
    lambda_c_cls = 1.0   # λ2: Center classification loss weight  
    lambda_lip = 0.0000      # λ3: Lipschitz constraint loss weight
    lambda_kmeans = 0.0001   # λ4: K-means loss weight
    
    # Training parameters
    learning_rate = 0.01  # Higher initial LR for scheduling
    momentum = 0.9
    weight_decay = 5e-4
    num_epochs = 100
    warmup_epochs = 1      # Number of epochs for learning rate warmup
    
    # Learning rate scheduling
    use_scheduler = True
    scheduler_type = 'cosine'  # 'cosine', 'multistep', 'step'
    scheduler_milestones = [60, 120, 160]  # For multistep scheduler
    scheduler_gamma = 0.2  # LR reduction factor
    scheduler_t_max = None  # For cosine (defaults to num_epochs)
    scheduler_step_size = 30  # For step scheduler
    
    # Algorithm 2 specific parameters
    n_total = num_epochs  # Total iterations in Algorithm 2
    n_inner = 5          # Inner iterations for center updates
    n_outer = 3          # Outer EM iterations
    n_adv_examples = 1   # N' adversarial examples per clean example
    
    # Attack parameters (OPTIMIZED for speed)
    attack_method = 'pgd'
    attack_steps = 5  # OPTIMIZED: Reduced from 10 to 5 for 2x speedup in adversarial generation
    attack_step_size = 2/255
    attack_random_start = True
    
    # Evaluation parameters
    eval_attack_steps = 20
    eval_attack_restarts = 5
    
    # PERFORMANCE OPTIMIZATIONS
    use_mixed_precision = False  # Enable AMP for ~2x speedup
    gradient_accumulation_steps = 1  # Increase effective batch size
    pin_memory = True  # Faster data loading
    persistent_workers = True  # Keep data workers alive between epochs
    prefetch_factor = 2  # Prefetch batches for faster loading
    
    # Component toggles for isolated testing
    use_adversarial_training = True   # Enable adversarial example generation
    use_quantization = True           # Enable quantization layer
    use_lipschitz_constraint = True   # Enable Lipschitz loss term
    use_center_classification = True  # Enable center classification loss
    use_kmeans_loss = True           # Enable k-means loss term
    
    # Logging and visualization
    save_dir = './artifacts'
    log_interval = 10  # Log every N batches during training
    save_interval = 10 # Save model every N epochs
    plot_results = True
    
    # Computed properties
    @property
    def computed_gamma(self):
        """Compute gamma as 2*(alpha + L*epsilon)"""
        return 2 * (self.alpha + self.lipschitz_constant * self.epsilon)
    
    def __post_init__(self):
        """Set computed values after initialization"""
        if self.gamma is None:
            self.gamma = self.computed_gamma
    
    def get_component_status(self):
        """Return string describing which components are enabled"""
        components = []
        if self.use_adversarial_training:
            components.append("AdversarialTraining")
        if self.use_quantization:
            components.append("Quantization")
        if self.use_lipschitz_constraint:
            components.append("LipschitzConstraint")
        if self.use_center_classification:
            components.append("CenterClassification")
        if self.use_kmeans_loss:
            components.append("KMeansLoss")
        
        if not components:
            return "BaselineResNet18"
        return "+".join(components)
    
    def validate_config(self):
        """Validate configuration parameters"""
        assert self.epsilon > 0, "Epsilon must be positive"
        assert self.lipschitz_constant > 0, "Lipschitz constant must be positive"
        assert self.clusters_per_class > 0, "Clusters per class must be positive"
        assert self.quantization_split in ['layer1', 'layer2', 'layer3', 'layer4'], \
            "Quantization split must be one of: layer1, layer2, layer3, layer4"
        
        # If quantization is disabled, related components should also be disabled
        if not self.use_quantization:
            if self.use_kmeans_loss:
                print("Warning: K-means loss disabled because quantization is disabled")
                self.use_kmeans_loss = False
            if self.use_center_classification:
                print("Warning: Center classification disabled because quantization is disabled")
                self.use_center_classification = False
    
    def __repr__(self):
        return f"Config(components={self.get_component_status()}, split={self.quantization_split}, eps={self.epsilon})"
