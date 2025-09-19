import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple, Optional

from utils import pgd_attack, compute_accuracy

class ComprehensiveEvaluator:
    """Comprehensive evaluation of trained models"""
    
    def __init__(self, model, test_loader, config):
        self.model = model
        self.test_loader = test_loader
        self.config = config
        self.device = config.device
        self.criterion = nn.CrossEntropyLoss()
    
    def evaluate_clean(self) -> Dict:
        """Evaluate on clean test data"""
        self.model.eval()
        
        total_correct = 0
        total_loss = 0.0
        total_samples = 0
        class_correct = torch.zeros(self.config.num_classes)
        class_total = torch.zeros(self.config.num_classes)
        
        with torch.no_grad():
            for x, y in tqdm(self.test_loader, desc='Clean Evaluation'):
                x, y = x.to(self.device), y.to(self.device)
                
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                
                _, predicted = outputs.max(1)
                correct = predicted.eq(y)
                
                total_correct += correct.sum().item()
                total_loss += loss.item()
                total_samples += y.size(0)
                
                # Per-class accuracy
                for i in range(self.config.num_classes):
                    class_mask = (y == i)
                    class_correct[i] += correct[class_mask].sum().item()
                    class_total[i] += class_mask.sum().item()
        
        # Compute metrics
        overall_acc = 100.0 * total_correct / total_samples
        avg_loss = total_loss / len(self.test_loader)
        
        # Per-class accuracies
        class_accuracies = {}
        for i in range(self.config.num_classes):
            if class_total[i] > 0:
                class_accuracies[f'class_{i}'] = 100.0 * class_correct[i] / class_total[i]
            else:
                class_accuracies[f'class_{i}'] = 0.0
        
        return {
            'accuracy': overall_acc,
            'loss': avg_loss,
            'correct': total_correct,
            'total': total_samples,
            **class_accuracies
        }
    
    def evaluate_robust(self, epsilon: float, num_steps: int, step_size: float, 
                       num_restarts: int = 1) -> Dict:
        """Evaluate robustness against PGD attacks"""
        self.model.eval()
        
        total_correct = 0
        total_loss = 0.0
        total_samples = 0
        class_correct = torch.zeros(self.config.num_classes)
        class_total = torch.zeros(self.config.num_classes)
        
        for x, y in tqdm(self.test_loader, desc=f'Robust Eval (ε={epsilon:.3f})'):
            x, y = x.to(self.device), y.to(self.device)
            
            # Generate adversarial examples with multiple restarts
            best_adv_x = None
            min_success_rate = float('inf')
            
            for restart in range(num_restarts):
                adv_x = pgd_attack(
                    self.model, x, y,
                    epsilon=epsilon,
                    step_size=step_size,
                    num_steps=num_steps,
                    random_start=True
                )
                
                # Evaluate this restart
                with torch.no_grad():
                    outputs = self.model(adv_x)
                    _, predicted = outputs.max(1)
                    success_rate = predicted.eq(y).float().mean().item()
                    
                    if success_rate < min_success_rate:
                        min_success_rate = success_rate
                        best_adv_x = adv_x
            
            # Evaluate on best adversarial examples
            with torch.no_grad():
                outputs = self.model(best_adv_x)
                loss = self.criterion(outputs, y)
                
                _, predicted = outputs.max(1)
                correct = predicted.eq(y)
                
                total_correct += correct.sum().item()
                total_loss += loss.item()
                total_samples += y.size(0)
                
                # Per-class accuracy
                for i in range(self.config.num_classes):
                    class_mask = (y == i)
                    class_correct[i] += correct[class_mask].sum().item()
                    class_total[i] += class_mask.sum().item()
        
        # Compute metrics
        overall_acc = 100.0 * total_correct / total_samples
        avg_loss = total_loss / len(self.test_loader)
        
        # Per-class accuracies
        class_accuracies = {}
        for i in range(self.config.num_classes):
            if class_total[i] > 0:
                class_accuracies[f'class_{i}'] = 100.0 * class_correct[i] / class_total[i]
            else:
                class_accuracies[f'class_{i}'] = 0.0
        
        return {
            'accuracy': overall_acc,
            'loss': avg_loss,
            'correct': total_correct,
            'total': total_samples,
            'epsilon': epsilon,
            'num_steps': num_steps,
            'num_restarts': num_restarts,
            **class_accuracies
        }
    
    def evaluate_multiple_epsilons(self, epsilons: List[float]) -> Dict:
        """Evaluate robustness across multiple epsilon values"""
        results = {}
        
        for eps in epsilons:
            step_size = eps / 4  # Standard choice: step_size = epsilon / 4
            robust_metrics = self.evaluate_robust(
                epsilon=eps,
                num_steps=self.config.eval_attack_steps,
                step_size=step_size,
                num_restarts=self.config.eval_attack_restarts
            )
            results[f'eps_{eps:.3f}'] = robust_metrics
        
        return results
    
    def evaluate_quantization_analysis(self) -> Dict:
        """Analyze quantization behavior if quantization is enabled"""
        if not self.config.use_quantization or not hasattr(self.model, 'quantization'):
            return {'message': 'Quantization not enabled'}
        
        self.model.eval()
        
        # Collect quantization statistics
        total_samples = 0
        total_quantization_distance = 0.0
        class_assignments = {i: torch.zeros(self.config.clusters_per_class) 
                           for i in range(self.config.num_classes)}
        center_utilization = torch.zeros(self.config.num_classes, self.config.clusters_per_class)
        
        with torch.no_grad():
            for x, y in tqdm(self.test_loader, desc='Quantization Analysis'):
                x, y = x.to(self.device), y.to(self.device)
                
                # Get latent representations and quantization info
                _, latent_features, quant_info = self.model(x, y, return_latent=True)
                
                if quant_info is not None:
                    assignments = quant_info['assignments']
                    distances = quant_info['distances']
                    
                    total_quantization_distance += distances.sum().item()
                    total_samples += x.size(0)
                    
                    # Track cluster assignments per class
                    for i in range(x.size(0)):
                        label = y[i].item()
                        cluster = assignments[i].item()
                        class_assignments[label][cluster] += 1
                        center_utilization[label, cluster] += 1
        
        # Compute statistics
        avg_quantization_distance = total_quantization_distance / total_samples if total_samples > 0 else 0.0
        
        # Cluster utilization analysis
        cluster_stats = {}
        for class_idx in range(self.config.num_classes):
            total_class_samples = class_assignments[class_idx].sum().item()
            if total_class_samples > 0:
                utilization = class_assignments[class_idx] / total_class_samples
                cluster_stats[f'class_{class_idx}_utilization'] = utilization.tolist()
                cluster_stats[f'class_{class_idx}_entropy'] = self._compute_entropy(utilization)
        
        return {
            'avg_quantization_distance': avg_quantization_distance,
            'total_samples_analyzed': total_samples,
            'cluster_utilization': cluster_stats,
            'center_utilization_matrix': center_utilization.tolist()
        }
    
    def _compute_entropy(self, probabilities: torch.Tensor) -> float:
        """Compute entropy of probability distribution"""
        # Add small epsilon to avoid log(0)
        probabilities = probabilities + 1e-8
        entropy = -(probabilities * torch.log(probabilities)).sum().item()
        return entropy
    
    def comprehensive_evaluation(self) -> Dict:
        """Run comprehensive evaluation with all metrics"""
        print("Running comprehensive evaluation...")
        
        results = {}
        
        # Clean evaluation
        print("Evaluating on clean data...")
        results['clean'] = self.evaluate_clean()
        
        # Robust evaluation at training epsilon
        print(f"Evaluating robustness at training epsilon ({self.config.epsilon:.3f})...")
        results['robust_training_eps'] = self.evaluate_robust(
            epsilon=self.config.epsilon,
            num_steps=self.config.eval_attack_steps,
            step_size=self.config.attack_step_size,
            num_restarts=self.config.eval_attack_restarts
        )
        
        # Multi-epsilon evaluation
        test_epsilons = [0.004, 0.008, 0.016, 0.032]  # 1/255, 2/255, 4/255, 8/255, 16/255, 32/255
        print(f"Evaluating robustness at multiple epsilons: {test_epsilons}")
        results['multi_epsilon'] = self.evaluate_multiple_epsilons(test_epsilons)
        
        # Quantization analysis
        if self.config.use_quantization:
            print("Analyzing quantization behavior...")
            results['quantization'] = self.evaluate_quantization_analysis()
        
        return results
    
    def print_evaluation_summary(self, results: Dict):
        """Print a formatted summary of evaluation results"""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        # Clean performance
        clean_acc = results['clean']['accuracy']
        print(f"Clean Accuracy: {clean_acc:.2f}%")
        
        # Robust performance at training epsilon
        if 'robust_training_eps' in results:
            robust_acc = results['robust_training_eps']['accuracy']
            training_eps = results['robust_training_eps']['epsilon']
            print(f"Robust Accuracy (ε={training_eps:.3f}): {robust_acc:.2f}%")
        
        # Multi-epsilon results
        if 'multi_epsilon' in results:
            print("\nRobustness across different epsilons:")
            for eps_key, metrics in results['multi_epsilon'].items():
                eps_val = metrics['epsilon']
                acc = metrics['accuracy']
                print(f"  ε={eps_val:.3f}: {acc:.2f}%")
        
        # Per-class accuracy (clean)
        print(f"\nPer-class Clean Accuracy:")
        for i in range(self.config.num_classes):
            class_acc = results['clean'].get(f'class_{i}', 0.0)
            print(f"  Class {i}: {class_acc:.2f}%")
        
        # Quantization analysis
        if 'quantization' in results and 'avg_quantization_distance' in results['quantization']:
            avg_dist = results['quantization']['avg_quantization_distance']
            print(f"\nQuantization Analysis:")
            print(f"  Average quantization distance: {avg_dist:.4f}")
            
            # Cluster utilization entropy
            cluster_stats = results['quantization']['cluster_utilization']
            print(f"  Cluster utilization entropy by class:")
            for class_idx in range(self.config.num_classes):
                entropy_key = f'class_{class_idx}_entropy'
                if entropy_key in cluster_stats:
                    entropy = cluster_stats[entropy_key]
                    print(f"    Class {class_idx}: {entropy:.3f}")
        
        print("="*60)
