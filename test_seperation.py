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
    """A self-contained ResNet-18 that can be split for feature extraction."""
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
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _build_decomposed_model(self):
        """Splits the model into a feature extractor (g_phi) and a classifier (g_theta)."""
        feature_layers = [self.conv1, self.bn1, nn.ReLU(inplace=True), self.layer1]
        classifier_layers = [self.layer4, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), self.linear]

        if self.quantization_split == 'layer1':
            classifier_layers = [self.layer2, self.layer3] + classifier_layers
        elif self.quantization_split == 'layer2':
            feature_layers.append(self.layer2)
            classifier_layers = [self.layer3] + classifier_layers
        elif self.quantization_split == 'layer3':
            feature_layers.extend([self.layer2, self.layer3])
        elif self.quantization_split == 'layer4':
            feature_layers.extend([self.layer2, self.layer3, self.layer4])
            classifier_layers = [nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), self.linear]
        else:
            raise ValueError(f"Invalid split point: {self.quantization_split}")

        self.g_phi = nn.Sequential(*feature_layers)
        self.g_theta = nn.Sequential(*classifier_layers)

    def get_latent_representation(self, x):
        return self.g_phi(x)

    def forward(self, x):
        latent = self.get_latent_representation(x)
        return self.g_theta(latent)


# ==============================================================================
# 2. SELF-CONTAINED ADVERSARIAL TRAINING UTILITIES
# ==============================================================================

def compute_lipschitz_loss(phi_clean, phi_adv, x_clean, x_adv, lipschitz_constant):
    """
    Computes the local Lipschitz loss based on the formula in paper.txt.
    Loss = relu(||g_phi(x) - g_phi(x')||_2 / ||x - x'||_inf - L)
    """
    # Flatten phi to calculate L2 norm on feature vectors
    phi_clean_flat = phi_clean.view(phi_clean.size(0), -1)
    phi_adv_flat = phi_adv.view(phi_adv.size(0), -1)
    phi_diff = torch.norm(phi_clean_flat - phi_adv_flat, p=2, dim=1)

    # Flatten x to calculate L-inf norm on input vectors ([0,1] image space)
    x_clean_flat = x_clean.view(x_clean.size(0), -1)
    x_adv_flat = x_adv.view(x_adv.size(0), -1)
    x_diff = torch.norm(x_clean_flat - x_adv_flat, p=float('inf'), dim=1)

    # Avoid division by zero for identical images
    x_diff = torch.max(x_diff, torch.tensor(1e-8).to(x_diff.device))

    lip_ratio = phi_diff / x_diff
    
    lip_loss = F.relu(lip_ratio - lipschitz_constant).mean()
    return lip_loss

def get_cifar10_dataloaders(data_path, batch_size, num_workers):
    """Returns self-contained CIFAR-10 dataloaders."""
    print("Loading CIFAR-10 data...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # We normalize inside the attack, not in the loader
    # This is crucial for PGD to work correctly on the [0, 1] image space.

    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print("Data loading complete.")
    return train_loader, test_loader

def pgd_attack(model, images, labels, device, epsilon, alpha, num_steps):
    """Generates adversarial examples using PGD."""
    # Standard CIFAR-10 normalization
    mu = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1).to(device)

    # The attack operates on the original [0,1] images
    adv_images = images.clone().detach()
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
    adv_images = torch.clamp(adv_images, min=0, max=1)

    for _ in range(num_steps):
        adv_images.requires_grad = True
        
        # Normalize before forward pass
        normalized_adv = (adv_images - mu) / std
        outputs = model(normalized_adv)

        loss = F.cross_entropy(outputs, labels)
        grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        
    return adv_images

def train_robust_model(model, train_loader, test_loader, device, args):
    """Trains a ResNet-18 model using PGD Adversarial Training."""
    print("\nStarting PGD adversarial training...")
    model.to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    
    # PGD attack parameters for training
    epsilon = args.epsilon / 255.0
    alpha = args.alpha / 255.0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Generate adversarial examples from the clean [0,1] images
            adv_inputs = pgd_attack(model, inputs, targets, device, epsilon, alpha, args.attack_steps)
            
            optimizer.zero_grad()
            
            # Normalize both clean and adversarial inputs for the model pass
            mu = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(device)
            std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1).to(device)
            normalized_adv = (adv_inputs - mu) / std
            
            # --- MODIFIED LOSS CALCULATION ---
            # Always calculate L_cls on adversarial examples
            outputs = model(normalized_adv)
            total_loss = criterion(outputs, targets)
            
            # Optionally, add the local Lipschitz loss
            if args.use_lipschitz_loss:
                normalized_inputs = (inputs - mu) / std
                phi_clean = model.get_latent_representation(normalized_inputs)
                phi_adv = model.get_latent_representation(normalized_adv)
                
                # Note: lip loss uses original [0,1] images for x_diff
                lip_loss = compute_lipschitz_loss(
                    phi_clean, phi_adv, inputs, adv_inputs, args.lipschitz_constant
                )
                total_loss += args.lambda_lip * lip_loss
            
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar.set_postfix({
                'Loss': f'{train_loss/(batch_idx+1):.3f}',
                'Acc': f'{100.*correct/total:.3f}%'
            })

        # Validate clean accuracy
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                mu = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(device)
                std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1).to(device)
                normalized_inputs = (inputs - mu) / std
                
                outputs = model(normalized_inputs)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f"Epoch {epoch+1} Summary: Train Acc: {100.*correct/total:.2f}% | Val (Clean) Acc: {val_acc:.2f}%")

        scheduler.step()
    
    print("Robust model training complete.")
    return model


# ==============================================================================
# 3. SELF-CONTAINED GEOMETRIC ANALYSIS FUNCTIONS
# ==============================================================================

def get_latent_features(model, loader, device):
    model.eval()
    print("\nExtracting latent features from the dataset for analysis...")
    features_by_class = {i: [] for i in range(10)}
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Feature Extraction"):
            images, labels = images.to(device), labels.to(device)
            mu = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(device)
            std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1).to(device)
            normalized_images = (images - mu) / std

            latent_features = model.get_latent_representation(normalized_images)
            
            # --- FIX: Flatten convolutional features for KMeans ---
            # The output of g_phi is a 4D tensor (N, C, H, W). KMeans requires 2D (n_samples, n_features).
            # We flatten the C, H, W dimensions into a single feature vector for each sample.
            batch_size = latent_features.shape[0]
            latent_features_flat = latent_features.view(batch_size, -1)
            
            latent_features_cpu = latent_features_flat.cpu().numpy()
            labels_cpu = labels.cpu().numpy()
            
            for i in range(len(labels_cpu)):
                features_by_class[labels_cpu[i]].append(latent_features_cpu[i])

    for i in range(10):
        features_by_class[i] = np.array(features_by_class[i])
    print("Feature extraction complete.")
    return features_by_class

def analyze_clusters(features_by_class, num_clusters):
    print(f"\nPerforming K-means clustering with k={num_clusters} for each class...")
    all_centers_by_class = {}
    all_alphas = []
    for class_idx in range(10):
        features = features_by_class[class_idx]
        if len(features) < num_clusters:
            print(f"  Warning: Class {class_idx} has only {len(features)} samples. Skipping clustering.")
            continue
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit(features)
        all_centers_by_class[class_idx] = kmeans.cluster_centers_
        distances_to_centers = kmeans.transform(features)
        min_distances = np.min(distances_to_centers, axis=1)
        class_alpha = np.mean(min_distances)
        all_alphas.append(class_alpha)
        print(f"  Class {class_idx}: Average alpha (QDist) = {class_alpha:.4f}")

    avg_alpha = np.mean(all_alphas)
    print(f"\nClustering complete. Average alpha across all classes: {avg_alpha:.4f}")
    return all_centers_by_class, avg_alpha

def calculate_gamma(centers_by_class):
    print("\nCalculating inter-class cluster separation (gamma)...")
    min_distances = []
    class_indices = list(centers_by_class.keys())
    for i in range(len(class_indices)):
        for j in range(i + 1, len(class_indices)):
            centers1 = centers_by_class[class_indices[i]]
            centers2 = centers_by_class[class_indices[j]]
            distance_matrix = cdist(centers1, centers2, 'euclidean')
            min_distances.append(np.min(distance_matrix))
    gamma = np.min(min_distances)
    print(f"Calculation complete. Gamma = {gamma:.4f}")
    return gamma

def plot_latent_space(features_by_class, filename="latent_space_tsne.png"):
    """
    Performs t-SNE dimensionality reduction on a subset of the latent features 
    and saves a 2D plot.
    """
    print(f"\nGenerating 2D t-SNE plot of the latent space...")

    # Prepare data for t-SNE: concatenate a subset of features and create a labels array
    all_features = []
    all_labels = []
    points_per_class = 500  # t-SNE is slow, so we use a subset

    for label, features in features_by_class.items():
        if len(features) > points_per_class:
            indices = np.random.choice(len(features), points_per_class, replace=False)
            features = features[indices]
            
        all_features.append(features)
        all_labels.extend([label] * len(features))

    if not all_features:
        print("  No features to plot. Skipping t-SNE.")
        return

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.array(all_labels)

    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, init='pca', learning_rate='auto')
    
    print(f"  Running t-SNE on {len(all_features)} points... (this may take a few minutes)")
    features_2d = tsne.fit_transform(all_features)
    print("  t-SNE complete.")

    # Create the plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=all_labels, cmap='tab10', alpha=0.7, s=12)
    
    # Create a legend
    legend_elements = scatter.legend_elements()
    class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    plt.legend(legend_elements[0], class_names, title="Classes")
    
    plt.title("2D t-SNE Visualization of the Latent Space")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save the plot
    plt.savefig(filename)
    print(f"Plot saved to {filename}")

# ==============================================================================
# 4. MAIN EXECUTION SCRIPT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Self-contained script to train a robust model and test its geometric separability.")
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs for adversarial training.')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate.')
    parser.add_argument('--epsilon', type=float, default=8.0, help='PGD attack epsilon (0-255).')
    parser.add_argument('--alpha', type=float, default=2.0, help='PGD attack step size (0-255).')
    parser.add_argument('--attack_steps', type=int, default=10, help='Number of PGD steps.')

    # New Lipschitz loss arguments
    parser.add_argument('--use_lipschitz_loss', action='store_true', help='Enable the local Lipschitz loss term during training.')
    parser.add_argument('--lambda_lip', type=float, default=0.5, help='Coefficient for the Lipschitz loss.')
    parser.add_argument('--lipschitz_constant', type=float, default=10.0, help='Target Lipschitz constant (L) for the loss.')

    # Analysis arguments
    parser.add_argument('--quantization_split', type=str, default='layer3', choices=['layer1', 'layer2', 'layer3', 'layer4'], help="Layer for latent space analysis.")
    parser.add_argument('--num_clusters', type=int, default=5, help='Number of clusters (k) per class for analysis.')
    parser.add_argument('--plot_latent_space', action='store_true', help='Generate a 2D t-SNE plot of the latent space after analysis.')

    # Common arguments
    parser.add_argument('--data_path', type=str, default='./data', help='Path to CIFAR-10 data.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training and analysis.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')

    args = parser.parse_args()

    # --- 1. Setup ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if args.use_lipschitz_loss:
        print("Lipschitz loss ENABLED with lambda = {:.2f} and L = {:.2f}".format(args.lambda_lip, args.lipschitz_constant))
    else:
        print("Lipschitz loss DISABLED. Training with standard PGD-AT.")

    train_loader, test_loader = get_cifar10_dataloaders(args.data_path, args.batch_size, args.num_workers)

    model = DecomposedResNet18(
        quantization_split=args.quantization_split,
        num_classes=10
    )

    # --- 2. Train Robust Model ---
    robust_model = train_robust_model(model, train_loader, test_loader, device, args)

    # --- 3. Run Geometric Analysis ---
    # Use the train_loader (without augmentations) for a cleaner analysis of the manifold
    clean_train_loader, _ = get_cifar10_dataloaders(args.data_path, args.batch_size, args.num_workers)

    features = get_latent_features(robust_model, clean_train_loader, device)
    centers, avg_alpha = analyze_clusters(features, args.num_clusters)
    
    if len(centers) < 2:
        print("Need at least two classes with valid clusters to calculate gamma. Exiting.")
        sys.exit(1)
        
    gamma = calculate_gamma(centers)

    # --- 4. Report Final Results ---
    print("\n" + "="*50)
    print("      DIAGNOSTIC EXPERIMENT RESULTS")
    print("="*50)
    print(f"  Average Intra-Cluster Radius (alpha): {avg_alpha:.4f}")
    print(f"  Minimum Inter-Class Separation (gamma): {gamma:.4f}")
    print("-"*50)

    is_assumption_plausible = gamma > 2 * avg_alpha
    print(f"Theoretical Robustness Condition Check: gamma > 2 * alpha")
    print(f"Empirical Check: {gamma:.4f} > 2 * {avg_alpha:.4f}  ==>  {gamma:.4f} > {2*avg_alpha:.4f}")
    
    print("\nConclusion:")
    if is_assumption_plausible:
        print("  [+] The core assumption appears PLAUSIBLE.")
        print("      The latent space of the trained robust model shows good geometric separation.")
        print("      This suggests that the failure mode of the main project may be in the")
        print("      joint optimization of Algorithm 2, not the underlying theory itself.")
    else:
        print("  [-] The core assumption appears UNLIKELY to hold in practice.")
        print("      The class clusters in the latent space are not well-separated.")
        print("      This suggests the theoretical condition may be too strong or geometrically")
        print("      infeasible for this dataset and architecture.")
    print("="*50)

    # --- 5. (Optional) Plot Latent Space ---
    if args.plot_latent_space:
        plot_latent_space(features)


if __name__ == '__main__':
    main()
