#!/usr/bin/env python3

import torch
import argparse
import os
import time
from datetime import datetime

from config import Config
from model import create_model
from train import Algorithm2Trainer
from evaluate import ComprehensiveEvaluator
from utils import (
    get_cifar10_dataloaders, plot_training_curves, 
    save_checkpoint, load_checkpoint
)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Algorithm 2 Implementation for CIFAR-10')
    
    # Mode selection
    parser.add_argument('--mode', choices=['train', 'eval', 'both'], default='both',
                        help='Mode: train, eval, or both')
    
    # Model configuration
    parser.add_argument('--quantization_split', choices=['layer1', 'layer2', 'layer3', 'layer4'], 
                        default='layer3', help='Where to split ResNet18 for quantization')
    parser.add_argument('--clusters_per_class', type=int, default=10,
                        help='Number of clusters per class')
    parser.add_argument('--epsilon', type=float, default=8/255,
                        help='Perturbation budget for adversarial training')
    
    # Component toggles
    parser.add_argument('--disable_adversarial', default=False,
                        help='Disable adversarial training')
    parser.add_argument('--disable_quantization', default=False,
                        help='Disable quantization (baseline ResNet18)')
    parser.add_argument('--disable_lipschitz', default=False,
                        help='Disable Lipschitz constraint')
    parser.add_argument('--disable_center_cls', default=False,
                        help='Disable center classification loss')
    parser.add_argument('--disable_kmeans', default=False,
                        help='Disable k-means loss')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate')
    
    # Learning rate scheduler parameters
    parser.add_argument('--disable_scheduler', default=False,
                        help='Disable learning rate scheduler')
    parser.add_argument('--scheduler_type', choices=['cosine', 'multistep', 'step'], 
                        default='cosine', help='Type of learning rate scheduler')
    parser.add_argument('--scheduler_gamma', type=float, default=0.2,
                        help='LR reduction factor for step/multistep schedulers')
    
    # Loss coefficients
    parser.add_argument('--lambda_cls', type=float, default=1.0,
                        help='Classification loss weight')
    parser.add_argument('--lambda_c_cls', type=float, default=1.0,
                        help='Center classification loss weight')
    parser.add_argument('--lambda_lip', type=float, default=0.0000,
                        help='Lipschitz constraint loss weight')
    parser.add_argument('--lambda_kmeans', type=float, default=0.0001,
                        help='K-means loss weight')
    
    # File paths
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint for evaluation or resume training')
    parser.add_argument('--save_dir', type=str, default='./artifacts',
                        help='Directory to save results')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no_plot', default=False,
                        help='Disable plotting')
    
    return parser.parse_args()

def setup_config(args):
    """Setup configuration from command line arguments"""
    config = Config()
    
    # Update config with command line arguments
    config.quantization_split = args.quantization_split
    config.clusters_per_class = args.clusters_per_class
    config.epsilon = args.epsilon
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.save_dir = args.save_dir
    config.plot_results = not args.no_plot
    
    # Loss coefficients
    config.lambda_cls = args.lambda_cls
    config.lambda_c_cls = args.lambda_c_cls
    config.lambda_lip = args.lambda_lip
    config.lambda_kmeans = args.lambda_kmeans
    
    # Scheduler configuration
    config.use_scheduler = not args.disable_scheduler
    config.scheduler_type = args.scheduler_type
    config.scheduler_gamma = args.scheduler_gamma
    
    # Component toggles
    config.use_adversarial_training = not args.disable_adversarial
    config.use_quantization = not args.disable_quantization
    config.use_lipschitz_constraint = not args.disable_lipschitz
    config.use_center_classification = not args.disable_center_cls
    config.use_kmeans_loss = not args.disable_kmeans
    
    # Validate and adjust configuration
    config.validate_config()
    
    # Update gamma based on final parameters
    config.__post_init__()
    
    return config

def set_random_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(config, checkpoint_path=None):
    """Train the model using Algorithm 2"""
    print("Setting up training...")
    
    # Create data loaders
    train_loader, val_loader = get_cifar10_dataloaders(config)
    
    # Create model
    model = create_model(config)
    print(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Load checkpoint if provided
    start_epoch = 1
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path, model)
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    
    # Create trainer
    trainer = Algorithm2Trainer(model, train_loader, val_loader, config)
    
    # Start training
    print("Starting training...")
    start_time = time.time()
    
    train_metrics, val_metrics = trainer.train()
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = os.path.join(config.save_dir, f'final_model_{timestamp}.pt')
    save_checkpoint(
        model, trainer.optimizer, config.num_epochs, 
        train_metrics, val_metrics, config
    )
    print(f"Final model saved to {final_model_path}")
    
    # Plot training curves
    if config.plot_results:
        plot_training_curves(train_metrics, val_metrics, config)
    
    return model, train_metrics, val_metrics

def evaluate_model(config, checkpoint_path):
    """Evaluate the model comprehensively"""
    print("Setting up evaluation...")
    
    # Create data loaders
    _, test_loader = get_cifar10_dataloaders(config)
    
    # Create model
    model = create_model(config)
    
    # Load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}")
        load_checkpoint(checkpoint_path, model)
    else:
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")
    
    # Create evaluator
    evaluator = ComprehensiveEvaluator(model, test_loader, config)
    
    # Run comprehensive evaluation
    print("Running comprehensive evaluation...")
    start_time = time.time()
    
    results = evaluator.comprehensive_evaluation()
    
    eval_time = time.time() - start_time
    print(f"Evaluation completed in {eval_time:.2f} seconds")
    
    # Print summary
    evaluator.print_evaluation_summary(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(config.save_dir, f'evaluation_results_{timestamp}.pt')
    torch.save(results, results_path)
    print(f"Evaluation results saved to {results_path}")
    
    return results

def main():
    """Main function"""
    args = parse_arguments()
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Setup configuration
    config = setup_config(args)
    
    # Create save directory
    os.makedirs(config.save_dir, exist_ok=True)
    
    print("="*60)
    print("ALGORITHM 2 IMPLEMENTATION FOR CIFAR-10")
    print("="*60)
    print(f"Configuration: {config}")
    print(f"Components: {config.get_component_status()}")
    print(f"Device: {config.device}")
    print("="*60)
    
    # Execute based on mode
    if args.mode in ['train', 'both']:
        print("TRAINING PHASE")
        print("-"*40)
        model, train_metrics, val_metrics = train_model(config, args.checkpoint)
        
        # Save the trained model path for evaluation
        if args.mode == 'both':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_for_eval = os.path.join(config.save_dir, f'final_model_{timestamp}.pt')
        
    if args.mode in ['eval', 'both']:
        print("\nEVALUATION PHASE")
        print("-"*40)
        
        # Determine checkpoint path
        if args.mode == 'eval':
            if not args.checkpoint:
                raise ValueError("Must provide --checkpoint for evaluation mode")
            checkpoint_for_eval = args.checkpoint
        
        # Run evaluation
        results = evaluate_model(config, checkpoint_for_eval)
    
    print("\nExecution completed successfully!")

if __name__ == '__main__':
    main()
