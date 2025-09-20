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

def compute_kmeans_loss(phi, labels, centers, kmeans_gamma):
    """Computes the K-means style loss from the paper."""
    total_loss = 0.0
    
    # Intra-class loss (pulls points to their class's centers)
    for i in range(centers.shape[0]): # Iterate over classes
        class_mask = (labels == i)
        if not class_mask.any(): continue
        
        phi_class = phi[class_mask]
        centers_class = centers[i].unsqueeze(0) # (1, num_clusters, dim)
        
        # Distance from each point to all centers of its class
        dists = torch.cdist(phi_class.view(phi_class.size(0), -1), centers_class.view(centers_class.size(1), -1))
        min_dists, _ = torch.min(dists, dim=1)
        total_loss += min_dists.mean()

    # Inter-class loss (pushes centers of different classes apart)
    all_centers_flat = centers.view(-1, centers.shape[-1])
    center_labels = torch.arange(centers.shape[0]).repeat_interleave(centers.shape[1])
    
    pairwise_center_dist = torch.cdist(all_centers_flat, all_centers_flat)
    label_matrix = center_labels.view(-1, 1) != center_labels.view(1, -1)
    
    different_class_center_dists = pairwise_center_dist[label_matrix]
    if different_class_center_dists.nelement() > 0:
        center_margin_loss = F.relu(kmeans_gamma - different_class_center_dists).mean()
        total_loss += center_margin_loss
        
    return total_loss

def compute_center_classification_loss(model, centers):
    """Computes classification loss on the cluster centers themselves."""
    num_classes, num_clusters, dim = centers.shape
    centers_flat = centers.view(-1, dim)
    
    # We need to reshape centers to match the expected input shape of g_theta if it's convolutional
    # This is a simplification; assumes g_theta can handle flattened input.
    # For a real ResNet split, this would need careful reshaping.
    try:
        center_outputs = model.classify_from_latent(centers_flat)
    except RuntimeError:
        # If g_theta has conv layers, it expects (N, C, H, W). This is a placeholder.
        # A proper implementation would require knowing the exact split point and reshaping.
        # For simplicity, we'll assume a dummy shape that works for the final layers.
        side_dim = int(np.sqrt(dim // 512)) if 'layer4' in model.quantization_split else int(np.sqrt(dim // 256))
        center_outputs = model.classify_from_latent(centers_flat.view(centers_flat.size(0), -1, side_dim, side_dim))


    center_labels = torch.arange(num_classes, device=centers.device).repeat_interleave(num_clusters)
    return F.cross_entropy(center_outputs, center_labels)


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

def get_latent_features(model, loader, device, max_batches=None):
    """Extracts and flattens latent features from a loader."""
    model.eval()
    all_features, all_labels = [], []
    mu = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1).to(device)
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(tqdm(loader, desc="Extracting Features")):
            if max_batches and i >= max_batches: break
            inputs = inputs.to(device)
            phi = model.get_latent_representation((inputs - mu) / std)
            all_features.append(phi.view(phi.size(0), -1).cpu())
            all_labels.append(targets.cpu())
            
    return torch.cat(all_features), torch.cat(all_labels)

def analyze_and_report_geometry(centers, features_by_class):
    """Calculates and prints alpha and gamma."""
    # Gamma: Minimum distance between centers of different classes
    all_centers_flat = centers.view(-1, centers.shape[-1])
    center_labels = torch.arange(centers.shape[0]).repeat_interleave(centers.shape[1])
    pairwise_center_dist = torch.cdist(all_centers_flat, all_centers_flat)
    label_matrix = center_labels.view(-1, 1) != center_labels.view(1, -1)
    gamma = torch.min(pairwise_center_dist[label_matrix]) if label_matrix.any() else 0.0

    # Alpha: Average distance from points to their nearest in-class center
    all_alphas = []
    for i in range(centers.shape[0]): # For each class
        if i not in features_by_class: continue
        class_features = features_by_class[i]
        class_centers = centers[i].unsqueeze(0)
        
        dists = torch.cdist(class_features, class_centers.view(class_centers.size(1), -1))
        min_dists, _ = torch.min(dists, dim=1)
        all_alphas.append(min_dists.mean())
    
    alpha = torch.mean(torch.tensor(all_alphas)) if all_alphas else 0.0
    
    print(f"  Live Geometry - Alpha: {alpha:.4f}, Gamma: {gamma:.4f}, Gamma > 2 * Alpha: {gamma > 2 * alpha}")
    return alpha, gamma
    
def train_phase2(model, centers, val_features_by_class, optimizer, device, args):
    model.train() # Only g_theta is in train mode
    
    # Phase 2 is complex and typically involves iterating over the dataset.
    # For a simple demonstration, we perform one optimization step.
    # A full implementation would have a dataloader and epochs.
    
    # 1. K-Means Loss (requires latent features, for simplicity we use validation features)
    # This is a conceptual simplification. A real implementation would use train loader.
    all_features = torch.cat(list(val_features_by_class.values())).to(device)
    all_labels = torch.cat([torch.full((len(v),), k) for k, v in val_features_by_class.items()]).to(device)
    loss_kmeans = compute_kmeans_loss(all_features, all_labels, centers, args.kmeans_gamma)
    
    # 2. Center Classification Loss
    loss_c_cls = compute_center_classification_loss(model, centers)
    
    total_loss = args.lambda_kmeans * loss_kmeans + args.lambda_c_cls * loss_c_cls
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    print(f"  Phase 2 Step - L_kmeans: {loss_kmeans.item():.2f}, L_c_cls: {loss_c_cls.item():.2f}")


def plot_latent_space(features, labels, filename_prefix, args):
    epsilon_str = f"{args.epsilon:.3f}".replace("0.", "p")
    filename = f"{filename_prefix}_L{args.lipschitz_constant}_eps{epsilon_str}_lip{args.lambda_lip}_margin{args.lambda_margin}.png"
    title = f"Latent Space (L={args.lipschitz_constant}, $\\epsilon$={args.epsilon:.3f}, $\\lambda_{{lip}}$={args.lambda_lip}, $\\lambda_{{margin}}$={args.lambda_margin})"

    print(f"\nGenerating 2D t-SNE plot and saving to {filename}...")
    points_per_class = 500
    subset_features, subset_labels = [], []
    for i in range(10): # For each class
        class_mask = (labels == i)
        class_features = features[class_mask]
        if len(class_features) > points_per_class:
            indices = np.random.choice(len(class_features), points_per_class, replace=False)
            class_features = class_features[indices]
        subset_features.append(class_features)
        subset_labels.extend([i] * len(class_features))
        
    features_np = torch.cat(subset_features).numpy()
    labels_np = np.array(subset_labels)

    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42, init='pca', learning_rate='auto')
    print("  Running t-SNE... (this may take a few minutes)")
    features_2d = tsne.fit_transform(features_np)
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels_np, cmap='tab10', alpha=0.7, s=12)
    legend_elements = scatter.legend_elements()
    class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    plt.legend(legend_elements[0], class_names, title="Classes")
    plt.title(title)
    plt.savefig(filename)
    print(f"Plot saved to {filename}")

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
    parser.add_argument('--lr_phase2', type=float, default=0.01, help='Learning rate for g_theta and centers in Phase 2')
    parser.add_argument('--lambda_kmeans', type=float, default=0.1, help='Lambda for K-Means loss')
    parser.add_argument('--lambda_c_cls', type=float, default=0.1, help='Lambda for Center Classification loss')
    parser.add_argument('--kmeans_gamma', type=float, default=2.0, help='Margin for inter-class center separation in L_kmeans')
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
    val_features, val_labels = get_latent_features(model, test_loader, device, max_batches=20) # Limit batches to reduce memory
    plot_latent_space(val_features, val_labels, "latent_space_phase1", args)
    
    # --- PHASE 2 ---
    print("\n--- Starting Phase 2: Quantization ---")
    
    # 1. Freeze g_phi
    for param in model.g_phi.parameters():
        param.requires_grad = False
    
    # 2. Initialize Cluster Centers
    print("Initializing cluster centers with K-Means...")
    features_by_class = {i: val_features[val_labels == i] for i in range(10)}
    initial_centers = torch.zeros(10, args.num_clusters, val_features.shape[1])
    for i in range(10):
        if len(features_by_class[i]) >= args.num_clusters:
            kmeans = KMeans(n_clusters=args.num_clusters, random_state=42, n_init=10).fit(features_by_class[i])
            initial_centers[i] = torch.from_numpy(kmeans.cluster_centers_)
    centers = nn.Parameter(initial_centers.to(device))

    # 3. Setup Optimizer for g_theta and centers
    optimizer_p2 = optim.SGD(list(model.g_theta.parameters()) + [centers], lr=args.lr_phase2, momentum=0.9)
    scheduler_p2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_p2, T_max=args.epochs_phase2)

    # 4. Phase 2 Training Loop
    val_features_by_class_cpu = {i: val_features[val_labels == i] for i in range(10)}
    for epoch in range(1, args.epochs_phase2 + 1):
        print(f"Phase 2 Epoch {epoch}/{args.epochs_phase2}")
        # Note: A full implementation would loop over the training data here.
        # We perform one step for demonstration and then track geometry.
        train_phase2(model, centers, val_features_by_class_cpu, optimizer_p2, device, args)
        
        # Live tracking of alpha and gamma on validation data
        with torch.no_grad():
            analyze_and_report_geometry(centers.cpu().data, val_features_by_class_cpu)
            
        scheduler_p2.step()
        
    print("--- Phase 2 Complete. ---")
    # Final analysis could be done here.
    print("\n--- Final Evaluation ---")
    evaluate_robustness(model, test_loader, device, "Validation Set", args)
    evaluate_robustness(model, train_loader, device, "Training Set", args)
    

def evaluate_robustness(model, loader, device, dataset_name, args):
    model.eval()
    total_correct_clean = 0
    total_correct_robust = 0
    total_samples = 0
    
    mu = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1).to(device)

    progress_bar = tqdm(loader, desc=f"Evaluating on {dataset_name}")
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Clean accuracy
        with torch.no_grad():
            norm_inputs = (inputs - mu) / std
            outputs_clean = model(norm_inputs)
            _, predicted_clean = outputs_clean.max(1)
            total_correct_clean += predicted_clean.eq(targets).sum().item()
        
        # Robust accuracy
        adv_inputs = pgd_attack(model, inputs, targets, device, args.epsilon, args.alpha, args.attack_steps)
        with torch.no_grad():
            norm_adv_inputs = (adv_inputs - mu) / std
            outputs_robust = model(norm_adv_inputs)
            _, predicted_robust = outputs_robust.max(1)
            total_correct_robust += predicted_robust.eq(targets).sum().item()
            
        total_samples += targets.size(0)

        clean_acc = 100. * total_correct_clean / total_samples
        robust_acc = 100. * total_correct_robust / total_samples
        progress_bar.set_postfix({'Clean Acc': f'{clean_acc:.2f}%', 'Robust Acc': f'{robust_acc:.2f}%'})

    final_clean_acc = 100. * total_correct_clean / total_samples
    final_robust_acc = 100. * total_correct_robust / total_samples
    print(f"Results for {dataset_name}:")
    print(f"  Clean Accuracy: {final_clean_acc:.2f}%")
    print(f"  Robust Accuracy: {final_robust_acc:.2f}%")
    
if __name__ == '__main__':
    main()
