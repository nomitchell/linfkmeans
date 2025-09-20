import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import sys
import os

# This script requires scikit-learn, matplotlib, and tqdm.
# You can install them with: pip install scikit-learn matplotlib tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# ==============================================================================
# 1. SELF-CONTAINED MODEL DEFINITION (ResNet-18)
# ==============================================================================

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class DecomposedResNet18(nn.Module):
    def __init__(self, num_blocks=(2, 2, 2, 2), num_classes=10, quantization_split='layer3'):
        super(DecomposedResNet18, self).__init__()
        self.in_planes = 64
        self.quantization_split = quantization_split
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlock, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes)
        self._build_decomposed_model()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _build_decomposed_model(self):
        feature_layers = [self.conv1, self.bn1, nn.ReLU(inplace=True), self.layer1]
        classifier_layers = [self.layer4, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), self.linear]
        if self.quantization_split == 'layer1': classifier_layers = [self.layer2, self.layer3] + classifier_layers
        elif self.quantization_split == 'layer2':
            feature_layers.append(self.layer2)
            classifier_layers = [self.layer3] + classifier_layers
        elif self.quantization_split == 'layer3': feature_layers.extend([self.layer2, self.layer3])
        elif self.quantization_split == 'layer4':
            feature_layers.extend([self.layer2, self.layer3, self.layer4])
            classifier_layers = [nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), self.linear]
        self.g_phi = nn.Sequential(*feature_layers)
        self.g_theta = nn.Sequential(*classifier_layers)

    def get_latent_representation(self, x): return self.g_phi(x)
    def classify_from_latent(self, z): return self.g_theta(z)
    def forward(self, x): return self.classify_from_latent(self.get_latent_representation(x))


# ==============================================================================
# 2. SELF-CONTAINED UTILITIES & LOSS FUNCTIONS
# ==============================================================================

# --- Data Loading ---
def get_cifar10_dataloaders(data_path, batch_size, num_workers):
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

# --- Adversarial Attack ---
def pgd_attack(model, images, labels, device, epsilon, alpha, num_steps):
    mu = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1).to(device)
    adv_images = images.clone().detach() + torch.empty_like(images).uniform_(-epsilon, epsilon)
    adv_images = torch.clamp(adv_images, min=0, max=1)
    for _ in range(num_steps):
        adv_images.requires_grad = True
        outputs = model((adv_images - mu) / std)
        loss = F.cross_entropy(outputs, labels)
        grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
        adv_images = (adv_images.detach() + alpha * grad.sign()).clamp(min=0, max=1)
        delta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()
    return adv_images

# --- Loss Functions ---
def compute_lipschitz_loss(phi_clean, phi_adv, x_clean, x_adv, L):
    phi_diff = torch.norm(phi_clean.view(phi_clean.size(0), -1) - phi_adv.view(phi_adv.size(0), -1), p=2, dim=1)
    x_diff = torch.norm(x_clean.view(x_clean.size(0), -1) - x_adv.view(x_adv.size(0), -1), p=float('inf'), dim=1)
    x_diff = torch.max(x_diff, torch.tensor(1e-8).to(x_diff.device))
    return F.relu(phi_diff / x_diff - L).mean()

def compute_margin_loss(phi_clean, labels, margin):
    phi_flat = phi_clean.view(phi_clean.size(0), -1)
    pairwise_dist = torch.cdist(phi_flat, phi_flat, p=2)
    label_matrix = labels.view(-1, 1) != labels.view(1, -1)
    # Select only distances between points of different classes
    different_class_dists = pairwise_dist[label_matrix]
    if different_class_dists.nelement() == 0: return torch.tensor(0.0).to(phi_clean.device)
    return F.relu(margin - different_class_dists).mean()

# ==============================================================================
# 3. TRAINING & ANALYSIS
# ==============================================================================

def train_phase1(model, loader, optimizer, device, args):
    model.train()
    progress_bar = tqdm(loader, desc=f"Phase 1 Epoch {args.epoch}/{args.epochs_phase1}")
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # --- Generate Adversarial Examples ---
        adv_list = [pgd_attack(model, inputs, targets, device, args.epsilon, args.alpha, args.attack_steps) for _ in range(args.s_prime)]
        adv_inputs = torch.cat(adv_list, dim=0)
        adv_targets = targets.repeat(args.s_prime)
        
        optimizer.zero_grad()
        
        # --- Normalize Inputs ---
        mu = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(device)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1).to(device)
        norm_clean = (inputs - mu) / std
        norm_adv = (adv_inputs - mu) / std
        
        # --- Loss Calculation ---
        # 1. Classification Loss on adversarial examples
        loss_cls = F.cross_entropy(model(norm_adv), adv_targets)
        
        # 2. Lipschitz and Margin losses (need latent features)
        phi_clean = model.get_latent_representation(norm_clean)
        phi_adv = model.get_latent_representation(norm_adv)
        
        loss_lip = compute_lipschitz_loss(phi_clean.repeat(args.s_prime, 1, 1, 1), phi_adv, inputs.repeat(args.s_prime, 1, 1, 1), adv_inputs, args.lipschitz_constant)
        
        # 3. Margin Loss on clean examples
        margin = 2 * args.lipschitz_constant * args.epsilon
        loss_margin = compute_margin_loss(phi_clean, targets, margin)
        
        total_loss = loss_cls + args.lambda_lip * loss_lip + args.lambda_margin * loss_margin
        total_loss.backward()
        optimizer.step()
        progress_bar.set_postfix({'L_cls': f'{loss_cls.item():.2f}', 'L_lip': f'{loss_lip.item():.2f}', 'L_margin': f'{loss_margin.item():.2f}'})

def validate(model, loader, device, epoch, phase):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            mu = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(device)
            std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1).to(device)
            outputs = model((inputs - mu) / std)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    print(f"Phase {phase} Epoch {epoch} Val Acc: {acc:.2f}%")
    return acc

def get_latent_features(model, loader, device):
    # ... (code from test_seperation.py, flattened) ...
    pass

def analyze_and_report_geometry(model, loader, device, num_clusters):
    # ... (code to run get_latent, analyze_clusters, calculate_gamma) ...
    pass

def plot_latent_space(model, loader, device, filename):
    # ... (code from test_seperation.py) ...
    pass

# ==============================================================================
# 4. MAIN EXECUTION SCRIPT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Implementation of Option 2: Two-Stage Robust Training")
    # Phase 1 Args
    parser.add_argument('--epochs_phase1', type=int, default=20, help='Epochs for Phase 1')
    parser.add_argument('--lambda_lip', type=float, default=0.5, help='Lambda for Lipschitz loss')
    parser.add_argument('--lambda_margin', type=float, default=0.5, help='Lambda for Margin loss')
    parser.add_argument('--s_prime', type=int, default=1, help='Number of adversarial examples per clean example')
    # Phase 2 Args
    parser.add_argument('--epochs_phase2', type=int, default=20, help='Epochs for Phase 2')
    parser.add_argument('--lambda_kmeans', type=float, default=0.1, help='Lambda for K-Means loss')
    parser.add_argument('--lambda_c_cls', type=float, default=0.1, help='Lambda for Center Classification loss')
    parser.add_argument('--num_clusters', type=int, default=5, help='Number of clusters per class')
    # Common Args
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--epsilon', type=float, default=8.0/255.0)
    parser.add_argument('--alpha', type=float, default=2.0/255.0)
    parser.add_argument('--attack_steps', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--quantization_split', type=str, default='layer3')
    parser.add_argument('--lipschitz_constant', type=float, default=10.0)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, test_loader = get_cifar10_dataloaders('./data', args.batch_size, 4)
    model = DecomposedResNet18(quantization_split=args.quantization_split).to(device)
    
    # --- PHASE 1 ---
    print("--- Starting Phase 1: Latent Space Structuring ---")
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_phase1)
    for epoch in range(1, args.epochs_phase1 + 1):
        args.epoch = epoch
        train_phase1(model, train_loader, optimizer, device, args)
        validate(model, test_loader, device, epoch, 1)
        scheduler.step()
        
    print("--- Phase 1 Complete. Analyzing latent space... ---")
    plot_latent_space(model, test_loader, device, "latent_space_phase1.png")
    
    # --- PHASE 2 ---
    print("\n--- Starting Phase 2: Quantization ---")
    # ... (Freeze g_phi, setup new optimizer for g_theta and centers, training loop) ...

if __name__ == '__main__':
    main()

# Note: Phase 2 is complex and requires careful implementation of center optimization.
# This skeleton provides the full setup for Phase 1 as requested.
# The `analyze_and_report_geometry` function would also need to be filled in and called
# during the Phase 2 training loop to track alpha and gamma evolution.
