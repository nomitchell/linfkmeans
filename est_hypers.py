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
import sys
import os

# This script provides heuristics to estimate good starting hyperparameters for test_two.py.
# It is self-contained and does not need to import from other project files.

# ==============================================================================
# 1. SELF-CONTAINED MODEL, DATA, AND LOSS DEFINITIONS
# (Copied from test_two.py for portability)
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
    def __init__(self, num_blocks=(2, 2, 2, 2), num_classes=10):
        super(DecomposedResNet18, self).__init__()
        self.in_planes = 64
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
        # We assume a 'layer3' split for this estimation script
        feature_layers = [self.conv1, self.bn1, nn.ReLU(inplace=True), self.layer1, self.layer2, self.layer3]
        classifier_layers = [self.layer4, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), self.linear]
        self.g_phi = nn.Sequential(*feature_layers)
        self.g_theta = nn.Sequential(*classifier_layers)

    def get_latent_representation(self, x): return self.g_phi(x)
    def forward(self, x): return self.g_theta(self.get_latent_representation(x))

def get_cifar10_dataloaders(data_path, batch_size, num_workers):
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

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

def compute_lipschitz_loss(phi_clean, phi_adv, x_clean, x_adv, L):
    phi_diff = torch.norm(phi_clean.view(phi_clean.size(0), -1) - phi_adv.view(phi_adv.size(0), -1), p=2, dim=1)
    x_diff = torch.norm(x_clean.view(x_clean.size(0), -1) - x_adv.view(x_adv.size(0), -1), p=float('inf'), dim=1)
    x_diff = torch.max(x_diff, torch.tensor(1e-8).to(x_diff.device))
    return F.relu(phi_diff / x_diff - L).mean()

def compute_margin_loss(phi_clean, labels, margin):
    phi_flat = phi_clean.view(phi_clean.size(0), -1)
    pairwise_dist = torch.cdist(phi_flat, phi_flat, p=2)
    label_matrix = labels.view(-1, 1) != labels.view(1, -1)
    different_class_dists = pairwise_dist[label_matrix]
    if different_class_dists.nelement() == 0: return torch.tensor(0.0).to(phi_clean.device)
    return F.relu(margin - different_class_dists).mean()

def compute_intra_margin_loss(phi_clean, labels):
    phi_flat = phi_clean.view(phi_clean.size(0), -1)
    pairwise_dist = torch.cdist(phi_flat, phi_flat, p=2)
    n = phi_flat.size(0)
    label_matrix = labels.view(-1, 1) == labels.view(1, -1)
    label_matrix.fill_diagonal_(False)
    same_class_dists = pairwise_dist[label_matrix]
    if same_class_dists.nelement() == 0: return torch.tensor(0.0).to(phi_clean.device)
    return same_class_dists.mean()

def get_latent_features(model, loader, device, max_batches=None):
    model.eval()
    all_features, all_labels = [], []
    mu = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1).to(device)
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            if max_batches and i >= max_batches: break
            inputs = inputs.to(device)
            phi = model.get_latent_representation((inputs - mu) / std)
            all_features.append(phi.view(phi.size(0), -1).cpu())
            all_labels.append(targets.cpu())
    return torch.cat(all_features), torch.cat(all_labels)

def estimate_optimal_clusters(class_features, k_min=2, k_max=10):
    if len(class_features) < k_max: return -1
    k_range = range(k_min, k_max + 1)
    inertias = []
    class_features_np = class_features.cpu().numpy()
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(class_features_np)
        inertias.append(kmeans.inertia_)
    try:
        if len(inertias) < 3: return k_min
        range_inertias = np.max(inertias) - np.min(inertias)
        if range_inertias < 1e-9: return k_min
        norm_inertias = (inertias - np.min(inertias)) / range_inertias
        diff1 = np.diff(norm_inertias, 1)
        diff2 = np.diff(diff1, 1)
        optimal_k = k_range[np.argmax(diff2) + 1]
    except (ValueError, IndexError):
        optimal_k = -1
    return optimal_k

# ==============================================================================
# 2. MAIN ESTIMATION SCRIPT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Heuristically estimate hyperparameters for two-stage robust training.")
    parser.add_argument('--pretrain_epochs', type=int, default=3, help='Epochs to pre-train model for analysis.')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--epsilon', type=float, default=8.0/255.0)
    parser.add_argument('--alpha', type=float, default=2.0/255.0)
    parser.add_argument('--attack_steps', type=int, default=10)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Running Hyperparameter Estimation on {device} ---")
    
    train_loader, test_loader = get_cifar10_dataloaders('./data', args.batch_size, 4)
    model = DecomposedResNet18().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # --- Part 1: Pre-train a basic AT model to get a reasonable latent space ---
    print(f"\n[Part 1/4] Pre-training a basic model for {args.pretrain_epochs} epochs...")
    mu = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1).to(device)
    for epoch in range(args.pretrain_epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Pre-train Epoch {epoch+1}")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            adv_inputs = pgd_attack(model, inputs, targets, device, args.epsilon, args.alpha, args.attack_steps)
            optimizer.zero_grad()
            outputs = model((adv_inputs - mu) / std)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix({'L_cls': f'{loss.item():.2f}'})
    print("Pre-training complete.")

    # --- Part 2: Estimate the Lipschitz constant (L) ---
    print("\n[Part 2/4] Estimating a reasonable Lipschitz constant (L)...")
    model.eval()
    all_lip_ratios = []
    # PGD attack requires gradients, so the main loop should not be in no_grad mode.
    for inputs, targets in tqdm(test_loader, desc="Calculating Lip Ratios"):
        inputs, targets = inputs.to(device), targets.to(device)
        # This part requires gradients to be enabled for the attack.
        adv_inputs = pgd_attack(model, inputs, targets, device, args.epsilon, args.alpha, args.attack_steps)
        
        # The subsequent analysis does not require gradients.
        with torch.no_grad():
            phi_clean = model.get_latent_representation((inputs - mu) / std)
            phi_adv = model.get_latent_representation((adv_inputs - mu) / std)

            phi_diff = torch.norm(phi_clean.view(phi_clean.size(0), -1) - phi_adv.view(phi_adv.size(0), -1), p=2, dim=1)
            x_diff = torch.norm(inputs.view(inputs.size(0), -1) - adv_inputs.view(adv_inputs.size(0), -1), p=float('inf'), dim=1)
            x_diff = torch.max(x_diff, torch.tensor(1e-8).to(x_diff.device))
            all_lip_ratios.append((phi_diff / x_diff).cpu())
            
    lip_ratios_tensor = torch.cat(all_lip_ratios)
    L_heuristic = np.percentile(lip_ratios_tensor.numpy(), 95)
    print(f"Heuristic L (95th percentile of observed ratios): {L_heuristic:.2f}")

    # --- Part 3: Estimate Lambdas by balancing loss magnitudes ---
    print("\n[Part 3/4] Estimating loss lambdas by balancing magnitudes...")
    model.eval()
    # Get one batch to calculate raw loss values
    inputs, targets = next(iter(train_loader))
    inputs, targets = inputs.to(device), targets.to(device)
    adv_inputs = pgd_attack(model, inputs, targets, device, args.epsilon, args.alpha, args.attack_steps)
    
    # Forward passes
    phi_clean = model.get_latent_representation((inputs - mu) / std)
    phi_adv = model.get_latent_representation((adv_inputs - mu) / std)
    outputs_adv = model.g_theta(phi_adv)
    
    # Calculate raw losses
    raw_l_cls = F.cross_entropy(outputs_adv, targets).item()
    raw_l_lip = compute_lipschitz_loss(phi_clean, phi_adv, inputs, adv_inputs, L_heuristic).item()
    margin = 2 * L_heuristic * args.epsilon
    raw_l_margin = compute_margin_loss(phi_clean, targets, margin).item()
    raw_l_intra = compute_intra_margin_loss(phi_clean, targets).item()

    # Heuristic: Balance other losses to be proportional to L_cls
    # We make margin losses an order of magnitude smaller to act as regularizers
    lambda_lip_h = raw_l_cls / (raw_l_lip + 1e-8)
    lambda_margin_h = (raw_l_cls / (raw_l_margin + 1e-8)) * 0.1
    lambda_intra_h = (raw_l_cls / (raw_l_intra + 1e-8)) * 0.1
    
    print(f"Raw losses on one batch: L_cls={raw_l_cls:.2f}, L_lip={raw_l_lip:.2f}, L_margin={raw_l_margin:.2f}, L_intra={raw_l_intra:.2f}")
    print(f"Heuristic lambdas: lambda_lip={lambda_lip_h:.2f}, lambda_margin={lambda_margin_h:.2f}, lambda_intra_margin={lambda_intra_h:.2f}")
    
    # --- Part 4: Estimate optimal number of clusters ---
    print("\n[Part 4/4] Estimating optimal number of clusters (k)...")
    val_features, val_labels = get_latent_features(model, test_loader, device, max_batches=20)
    features_by_class_for_est = {i: val_features[val_labels == i] for i in range(10)}
    estimated_k_values = []
    for i in range(10):
        class_features = features_by_class_for_est.get(i)
        if class_features is not None and len(class_features) > 15:
            optimal_k = estimate_optimal_clusters(class_features)
            if optimal_k > 0: estimated_k_values.append(optimal_k)
    
    k_heuristic = np.mean(estimated_k_values) if estimated_k_values else 5 # Default to 5 if estimation fails
    print(f"Heuristic k (average of per-class estimates): {k_heuristic:.2f}")
    
    # --- Final Summary ---
    print("\n" + "="*50)
    print("      SUGGESTED STARTING HYPERPARAMETERS")
    print("="*50)
    print("Based on a combination of heuristics and best practices:")

    # The previous dynamic estimation was unstable because it measured a chaotic,
    # poorly-trained model. These fixed defaults are based on our successful
    # rebalancing experiments and are designed for stable training.
    L_suggested = 10.0
    lambda_lip_suggested = 0.5
    lambda_margin_suggested = 0.1
    lambda_intra_suggested = 0.05
    
    # The k estimation is still useful, but we enforce a reasonable minimum.
    k_suggested = max(4, int(round(k_heuristic)))

    print("\n[INFO] L and Lambdas are set to robust defaults that prioritize stability.")
    print("[INFO] 'num_clusters' is estimated from the pre-trained model's latent space.")
    print("\n[IMPORTANT] Why L=10.0? A very small L (<1) can cause 'representation collapse',")
    print("            where the model maps all points of a class to a tiny cluster to satisfy")
    print("            the constraint. While this looks good on a t-SNE plot, it destroys useful")
    print("            information. L=10 provides strong regularization without forcing collapse.")


    run_command = (
        f"python test_two.py \\\n"
        f"    --epochs_phase1 40 \\\n"
        f"    --lipschitz_constant {L_suggested:.2f} \\\n"
        f"    --lambda_lip {lambda_lip_suggested:.2f} \\\n"
        f"    --lambda_margin {lambda_margin_suggested:.2f} \\\n"
        f"    --lambda_intra_margin {lambda_intra_suggested:.2f} \\\n"
        f"    --num_clusters {k_suggested}"
    )
    print("\nCopy and paste the following command to run test_two.py:")
    print(run_command)
    print("="*50)

if __name__ == '__main__':
    main()
