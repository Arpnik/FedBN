"""
Better noise visualization test with different std values
"""
import torch
import matplotlib.pyplot as plt
from noise_utils import NoiseInjector


def visualize_noise_levels():
    """Compare different noise standard deviations"""
    print("\n" + "=" * 70)
    print("GAUSSIAN NOISE STANDARD DEVIATION COMPARISON")
    print("=" * 70)

    # Create a clean test image with some structure
    clean_img = torch.ones(3, 256, 256) * 0.5
    clean_img[:, 64:192, 64:192] = 0.8  # Bright square
    clean_img[:, 100:156, 100:156] = 0.3  # Dark square inside

    # Test different std values
    std_values = [0.0, 0.1, 0.2, 0.3, 0.5]

    fig, axes = plt.subplots(1, len(std_values), figsize=(20, 4))
    fig.suptitle('Gaussian Noise with Different Standard Deviations (100% of pixels affected)',
                 fontsize=14, fontweight='bold')

    for idx, std in enumerate(std_values):
        if std == 0.0:
            noisy_img = clean_img
            title = 'Clean (std=0)'
        else:
            noise_injector = NoiseInjector(seed=42)
            # Apply noise to ALL pixels (ratio=1.0) to see the effect clearly
            noisy_img = noise_injector.add_gaussian_noise(
                clean_img.unsqueeze(0),
                noise_ratio=1.0,
                std=std
            ).squeeze(0)
            title = f'std={std}'

        # Convert to displayable format
        img_np = noisy_img.permute(1, 2, 0).numpy()

        axes[idx].imshow(img_np, vmin=0, vmax=1)
        axes[idx].set_title(title, fontsize=12)
        axes[idx].axis('off')

        # Calculate metrics
        if std > 0:
            diff = (noisy_img - clean_img).abs().mean().item()
            snr = clean_img.mean().item() / diff if diff > 0 else float('inf')
            axes[idx].text(0.5, -0.05, f'Diff: {diff:.4f}\nSNR: {snr:.2f}',
                          transform=axes[idx].transAxes, ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('noise_std_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: noise_std_comparison.png")
    print("=" * 70 + "\n")


def visualize_noise_ratios():
    """Compare different noise application ratios"""
    print("=" * 70)
    print("GAUSSIAN NOISE RATIO COMPARISON (std=0.2)")
    print("=" * 70)

    # Create a clean test image
    clean_img = torch.ones(3, 256, 256) * 0.5
    clean_img[:, 64:192, 64:192] = 0.8
    clean_img[:, 100:156, 100:156] = 0.3

    # Test different ratios with fixed std
    ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
    std = 0.2

    fig, axes = plt.subplots(1, len(ratios), figsize=(20, 4))
    fig.suptitle(f'Gaussian Noise with Different Application Ratios (std={std})',
                 fontsize=14, fontweight='bold')

    for idx, ratio in enumerate(ratios):
        noise_injector = NoiseInjector(seed=42)

        if ratio == 0.0:
            noisy_img = clean_img
            title = 'No Noise (0%)'
        else:
            noisy_img = noise_injector.add_gaussian_noise(
                clean_img.unsqueeze(0),
                noise_ratio=ratio,
                std=std
            ).squeeze(0)
            title = f'{int(ratio*100)}% affected'

        img_np = noisy_img.permute(1, 2, 0).numpy()

        axes[idx].imshow(img_np, vmin=0, vmax=1)
        axes[idx].set_title(title, fontsize=12)
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('noise_ratio_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: noise_ratio_comparison.png")
    print("=" * 70 + "\n")


def visualize_worst_cases():
    """Visualize worst-case scenarios"""
    print("=" * 70)
    print("WORST-CASE NOISE SCENARIOS")
    print("=" * 70)

    # Create a clean test image
    clean_img = torch.ones(3, 256, 256) * 0.5
    clean_img[:, 64:192, 64:192] = 0.8
    clean_img[:, 100:156, 100:156] = 0.3

    # Different worst-case scenarios
    scenarios = [
        ('Clean', 0.0, 0.0),
        ('Mild (0.1, 50%)', 0.1, 0.5),
        ('Moderate (0.15, 75%)', 0.15, 0.75),
        ('Strong (0.2, 100%)', 0.2, 1.0),
        ('Extreme (0.3, 100%)', 0.3, 1.0),
        ('Nuclear (0.5, 100%)', 0.5, 1.0),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Worst-Case Noise Scenarios for Robustness Testing',
                 fontsize=16, fontweight='bold')
    axes = axes.flatten()

    for idx, (name, std, ratio) in enumerate(scenarios):
        noise_injector = NoiseInjector(seed=42)

        if std == 0.0:
            noisy_img = clean_img
        else:
            noisy_img = noise_injector.add_gaussian_noise(
                clean_img.unsqueeze(0),
                noise_ratio=ratio,
                std=std
            ).squeeze(0)

        img_np = noisy_img.permute(1, 2, 0).numpy()

        axes[idx].imshow(img_np, vmin=0, vmax=1)
        axes[idx].set_title(name, fontsize=11, fontweight='bold')
        axes[idx].axis('off')

        # Add metrics
        if std > 0:
            diff = (noisy_img - clean_img).abs().mean().item()
            axes[idx].text(0.5, -0.08, f'Mean diff: {diff:.4f}',
                          transform=axes[idx].transAxes, ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('noise_worst_cases.png', dpi=150, bbox_inches='tight')
    print("Saved: noise_worst_cases.png")
    print("=" * 70 + "\n")


def print_recommendations():
    """Print noise parameter recommendations"""
    print("=" * 70)
    print("RECOMMENDATIONS FOR WORST-CASE TESTING")
    print("=" * 70)
    print()
    print("For Federated Learning Robustness Experiments:")
    print()
    print("1. CONSERVATIVE WORST CASE (Recommended)")
    print("   Command: --gaussian_std 0.2 --client_noise \"0:input=0.5;1:input=0.5\"")
    print("   Impact: ~15-25% accuracy drop")
    print("   Use: Standard robustness testing")
    print()
    print("2. EXTREME WORST CASE (Stress Test)")
    print("   Command: --gaussian_std 0.3 --client_noise \"0:input=1.0;1:input=1.0\"")
    print("   Impact: ~30-50% accuracy drop")
    print("   Use: Verify algorithm stability under severe corruption")
    print()
    print("3. HETEROGENEOUS NOISE (Realistic)")
    print("   Command: --gaussian_std 0.2 --client_noise \"0:input=0.8,label=0.2;1:input=0.5;2:label=0.3\"")
    print("   Impact: Variable per client")
    print("   Use: Simulate real-world heterogeneous noise conditions")
    print()
    print("4. LABEL NOISE WORST CASE")
    print("   Command: --client_noise \"0:label=0.4;1:label=0.4;2:label=0.4;3:label=0.4\"")
    print("   Impact: 40% wrong labels")
    print("   Use: Test robustness to annotation errors")
    print()
    print("=" * 70 + "\n")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("COMPREHENSIVE NOISE VISUALIZATION TEST")
    print("=" * 70)

    try:
        visualize_noise_levels()
        visualize_noise_ratios()
        visualize_worst_cases()
        print_recommendations()

        print("=" * 70)
        print("ALL VISUALIZATIONS GENERATED! ✓")
        print("Check: noise_std_comparison.png")
        print("Check: noise_ratio_comparison.png")
        print("Check: noise_worst_cases.png")
        print("=" * 70 + "\n")
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        print("Make sure matplotlib is installed: pip install matplotlib")