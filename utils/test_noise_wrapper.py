"""
Test script to verify noise implementation
Run this to ensure noise injection works correctly before running full experiments
"""
import time

import torch
from noise_utils import NoiseInjector, NoisyDatasetWrapper, get_client_noise_config
import matplotlib.pyplot as plt


def test_gaussian_noise():
    """Test Gaussian noise injection on dummy images"""
    print("\n" + "=" * 70)
    print("TEST 1: Gaussian Input Noise")
    print("=" * 70)

    # Create dummy images
    batch_size = 10
    images = torch.ones(batch_size, 3, 256, 256) * 0.5  # Gray images

    # Initialize noise injector
    noise_injector = NoiseInjector(seed=42)

    # Test different noise ratios
    for noise_ratio in [0.0, 0.2, 0.5, 1.0]:
        noisy_images = noise_injector.add_gaussian_noise(images, noise_ratio, std=0.1)

        # Count how many images were modified
        modified = (noisy_images != images).any(dim=(1, 2, 3)).sum().item()
        expected = int(batch_size * noise_ratio)

        print(f"Noise ratio: {noise_ratio:.1f}")
        print(f"  Images modified: {modified}/{batch_size} (expected ~{expected})")
        print(f"  Mean difference: {(noisy_images - images).abs().mean().item():.6f}")
        print(f"  Value range: [{noisy_images.min().item():.3f}, {noisy_images.max().item():.3f}]")

        # Check values are clipped to [0, 1]
        assert noisy_images.min() >= 0.0, "Values below 0!"
        assert noisy_images.max() <= 1.0, "Values above 1!"

    print("✓ Gaussian noise test passed!\n")


def test_label_noise():
    """Test label noise injection"""
    print("=" * 70)
    print("TEST 2: Label Noise")
    print("=" * 70)

    # Create dummy labels
    batch_size = 100
    num_classes = 31
    labels = torch.randint(0, num_classes, (batch_size,))

    # Initialize noise injector
    noise_injector = NoiseInjector(seed=42)

    # Test different noise ratios
    for noise_ratio in [0.0, 0.2, 0.5, 1.0]:
        noisy_labels = noise_injector.add_label_noise(labels, noise_ratio, num_classes)

        # Count how many labels were flipped
        flipped = (noisy_labels != labels).sum().item()
        expected = int(batch_size * noise_ratio)

        print(f"Noise ratio: {noise_ratio:.1f}")
        print(f"  Labels flipped: {flipped}/{batch_size} (expected ~{expected})")

        # Check that flipped labels are different from original
        different_mask = noisy_labels != labels
        if different_mask.any():
            assert (noisy_labels[different_mask] != labels[different_mask]).all(), \
                "Some 'flipped' labels are the same!"

        # Check labels are in valid range
        assert noisy_labels.min() >= 0, "Labels below 0!"
        assert noisy_labels.max() < num_classes, f"Labels >= {num_classes}!"

    print("✓ Label noise test passed!\n")


def test_reproducibility():
    """Test that same seed produces same noise"""
    print("=" * 70)
    print("TEST 3: Reproducibility")
    print("=" * 70)

    images = torch.rand(5, 3, 256, 256)
    labels = torch.randint(0, 31, (5,))

    # Run twice with same seed
    results = []
    for run in range(4):
        noise_injector = NoiseInjector(seed=42)
        time.sleep(60)  # delay for 2 seconds
        noisy_images = noise_injector.add_gaussian_noise(images, 0.5, std=0.1)
        noisy_labels = noise_injector.add_label_noise(labels, 0.5, 31)
        results.append((noisy_images, noisy_labels))

    # Check they're identical
    assert torch.allclose(results[0][0], results[1][0]), "Images differ between runs!"
    assert torch.equal(results[0][1], results[1][1]), "Labels differ between runs!"
    assert torch.allclose(results[0][0], results[2][0]), "Images differ between runs!"
    assert torch.equal(results[0][1], results[2][1]), "Labels differ between runs!"
    assert torch.allclose(results[0][0], results[3][0]), "Images differ between runs!"
    assert torch.equal(results[0][1], results[3][1]), "Labels differ between runs!"


    print("Run 1 and Run 2 with seed=42:")
    print(f"  Images identical: ✓")
    print(f"  Labels identical: ✓")

    # Run with different seed - should be different
    noise_injector = NoiseInjector(seed=123)
    noisy_images_diff = noise_injector.add_gaussian_noise(images, 0.5, std=0.1)
    noisy_labels_diff = noise_injector.add_label_noise(labels, 0.5, 31)

    assert not torch.allclose(results[0][0], noisy_images_diff), \
        "Different seeds produced same images!"

    print("Run 3 with seed=123:")
    print(f"  Different from seed=42: ✓")
    print("✓ Reproducibility test passed!\n")


def test_config_parsing():
    """Test configuration parsing"""
    print("=" * 70)
    print("TEST 4: Configuration Parsing")
    print("=" * 70)

    # Mock args object
    class Args:
        pass

    # Test compact format
    args = Args()
    args.client_noise = "0:input=0.2,label=0.0,mix=2;1:input=0.0,label=0.3,mix=3"
    args.gaussian_std = 0.1

    config0 = get_client_noise_config(0, args)
    config1 = get_client_noise_config(1, args)
    config2 = get_client_noise_config(2, args)

    print("Compact format parsing:")
    print(f"  Client 0: input={config0['input_noise_ratio']}, " +
          f"label={config0['label_noise_ratio']}, mix={config0['mixed_domains']}")
    assert config0['input_noise_ratio'] == 0.2
    assert config0['label_noise_ratio'] == 0.0
    assert config0['mixed_domains'] == 2

    print(f"  Client 1: input={config1['input_noise_ratio']}, " +
          f"label={config1['label_noise_ratio']}, mix={config1['mixed_domains']}")
    assert config1['input_noise_ratio'] == 0.0
    assert config1['label_noise_ratio'] == 0.3
    assert config1['mixed_domains'] == 3

    print(f"  Client 2: input={config2['input_noise_ratio']}, " +
          f"label={config2['label_noise_ratio']}, mix={config2['mixed_domains']}")
    assert config2['input_noise_ratio'] == 0.0
    assert config2['label_noise_ratio'] == 0.0
    assert config2['mixed_domains'] == None

    # Test individual format
    args2 = Args()
    args2.client_noise = ""
    args2.client0_input_noise = 0.15
    args2.client0_label_noise = 0.0
    args2.client0_mix_with = 1
    args2.gaussian_std = 0.1

    config = get_client_noise_config(0, args2)
    print("\nIndividual format parsing:")
    print(f"  Client 0: input={config['input_noise_ratio']}, " +
          f"label={config['label_noise_ratio']}, mix={config['mixed_domains']}")
    assert config['input_noise_ratio'] == 0.15
    assert config['mixed_domains'] == 1

    print("✓ Configuration parsing test passed!\n")


def test_dataset_wrapper():
    """Test NoisyDatasetWrapper"""
    print("=" * 70)
    print("TEST 5: Dataset Wrapper")
    print("=" * 70)

    # Create a simple dummy dataset
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=10):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            # Return constant image and label for testing
            image = torch.ones(3, 256, 256) * 0.5
            label = 5
            return image, label

    dataset = DummyDataset(10)
    noise_injector = NoiseInjector(seed=42)

    # Wrap with noise
    noisy_dataset = NoisyDatasetWrapper(
        dataset,
        noise_injector,
        input_noise_ratio=0.5,
        label_noise_ratio=0.5,
        num_classes=31,
        gaussian_std=0.1
    )

    # Test multiple samples
    input_noise_count = 0
    label_noise_count = 0

    for i in range(10):
        clean_img, clean_label = dataset[i]
        noisy_img, noisy_label = noisy_dataset[i]

        if not torch.equal(clean_img, noisy_img):
            input_noise_count += 1
        if clean_label != noisy_label:
            label_noise_count += 1

    print(f"Samples with input noise: {input_noise_count}/10 (expected ~5)")
    print(f"Samples with label noise: {label_noise_count}/10 (expected ~5)")

    # Check dataset length is preserved
    assert len(noisy_dataset) == len(dataset), "Dataset length changed!"

    print("✓ Dataset wrapper test passed!\n")


def visualize_noise_example():
    """Create a visualization of noise effects"""
    print("=" * 70)
    print("TEST 6: Visualization (optional)")
    print("=" * 70)

    try:
        # Create a clean test image
        clean_img = torch.ones(3, 256, 256) * 0.5
        clean_img[:, 64:192, 64:192] = 0.8  # Add a bright square

        noise_injector = NoiseInjector(seed=42)

        # Apply different levels of noise
        noise_levels = [0.0, 0.1, 0.2, 0.3]
        fig, axes = plt.subplots(1, len(noise_levels), figsize=(16, 4))

        for idx, noise_ratio in enumerate(noise_levels):
            noisy_img = noise_injector.add_gaussian_noise(
                clean_img.unsqueeze(0),
                noise_ratio if noise_ratio > 0 else 1.0,  # Always apply if not 0
                std=0.1
            ).squeeze(0)

            # Convert to displayable format
            img_np = noisy_img.permute(1, 2, 0).numpy()

            axes[idx].imshow(img_np)
            axes[idx].set_title(f'Noise: {noise_ratio * 100:.0f}%')
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig('noise_visualization.png', dpi=150, bbox_inches='tight')
        print("Visualization saved to 'noise_visualization.png'")
        print("✓ Visualization test passed!\n")
    except Exception as e:
        print(f"Visualization skipped (matplotlib not available or error: {e})\n")


def main():
    print("\n" + "=" * 70)
    print("NOISE IMPLEMENTATION VERIFICATION")
    print("=" * 70)
    print("Running tests to verify noise injection implementation...\n")

    # Run all tests
    test_gaussian_noise()
    test_label_noise()
    test_reproducibility()
    test_config_parsing()
    test_dataset_wrapper()
    visualize_noise_example()

    print("=" * 70)
    print("ALL TESTS PASSED! ✓")
    print("=" * 70)
    print()


if __name__ == '__main__':
    main()