"""
Quick diagnostic to verify noise is actually working in your setup
Run this with the EXACT same arguments as your training command
"""
import sys, os

from torch.utils.data import DataLoader
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sys
import torch
from utils.noise_utils import NoiseInjector, NoisyDatasetWrapper, get_client_noise_config
from utils.data_utils import OfficeDataset
import torchvision.transforms as transforms
import argparse


def quick_check():
    parser = argparse.ArgumentParser()
    parser.add_argument('--client_noise', type=str, default='', help='Same as training')
    parser.add_argument('--gaussian_std', type=float, default=0.1, help='Same as training')
    parser.add_argument('--noise_seed', type=int, default=42, help='Same as training')
    parser.add_argument('--client_datasets', type=str, default='0:1:2:3', help='Same as training')
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("QUICK NOISE REALITY CHECK")
    print("=" * 80)
    print(f"Arguments:")
    print(f"  --client_noise: '{args.client_noise}'")
    print(f"  --gaussian_std: {args.gaussian_std}")
    print(f"  --noise_seed: {args.noise_seed}")
    print(f"  --client_datasets: '{args.client_datasets}'")
    print("=" * 80)

    # Parse client datasets
    clients = args.client_datasets.split(':')
    client_dataset_assignment = [[int(d) for d in client.split(',')] for client in clients]

    print(f"\nNumber of clients: {len(client_dataset_assignment)}")
    for i, datasets in enumerate(client_dataset_assignment):
        print(f"  Client {i}: datasets {datasets}")

    # Check noise config for each client
    print(f"\nNoise Configuration:")
    for client_idx in range(len(client_dataset_assignment)):
        config = get_client_noise_config(client_idx, args)
        print(f"  Client {client_idx}:")
        print(f"    Input noise: {config['input_noise_ratio'] * 100:.0f}%")
        print(f"    Label noise: {config['label_noise_ratio'] * 100:.0f}%")
        print(f"    Gaussian std: {config['gaussian_std']}")

    # Load a tiny bit of real data
    print(f"\nLoading real data to test...")
    try:
        transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30, 30)),
            transforms.ToTensor(),
        ])

        dataset = OfficeDataset('../data', 'amazon', transform=transform)
        print(f"  Loaded {len(dataset)} samples from Amazon")

        # Create noise injector
        noise_injector = NoiseInjector(seed=args.noise_seed)

        # Get config for client 0
        config = get_client_noise_config(0, args)

        if config['input_noise_ratio'] == 0 and config['label_noise_ratio'] == 0:
            print(f"\n❌ NO NOISE CONFIGURED!")
            print(f"   You need to add --client_noise argument")
            print(f"   Example: --client_noise '0:input=1.0;1:input=1.0'")
            return

        # Wrap with noise
        noisy_dataset = NoisyDatasetWrapper(
            dataset, noise_injector,
            input_noise_ratio=config['input_noise_ratio'],
            label_noise_ratio=config['label_noise_ratio'],
            gaussian_std=config['gaussian_std']
        )

        print(f"\n  Created NoisyDatasetWrapper with:")
        print(f"    Input noise ratio: {config['input_noise_ratio']}")
        print(f"    Gaussian std: {config['gaussian_std']}")

        # Get same sample multiple times
        print(f"\nTesting noise randomness:")
        torch.manual_seed(4)  # Same seed as training
        img1, label1 = noisy_dataset[0]

        torch.manual_seed(4)  # Reset to same seed
        img2, label2 = noisy_dataset[0]

        diff = (img1 - img2).abs().mean().item()
        print(f"  Difference between two calls (same training seed): {diff:.6f}")

        if diff < 1e-6:
            print(f"  ❌ CRITICAL BUG: Noise is using global RNG!")
            print(f"     Noise will be deterministic and identical across runs!")
        elif diff > 0.001:
            print(f"  ✓ Noise IS random and independent of training seed")
        else:
            print(f"  ⚠️  Unclear - difference is very small")

        # Check if noise is actually strong
        clean_mean = 0.5  # Approximate mean of normalized images
        noisy_mean = img1.mean().item()
        noisy_std = img1.std().item()

        print(f"\nNoisy image statistics:")
        print(f"  Mean: {noisy_mean:.4f} (clean ~0.5)")
        print(f"  Std:  {noisy_std:.4f}")
        print(f"  Min:  {img1.min():.4f}")
        print(f"  Max:  {img1.max():.4f}")

        if config['gaussian_std'] >= 0.3:
            print(f"\n  ⚠️  WARNING: std={config['gaussian_std']} is EXTREMELY high!")
            print(f"     This will likely make training impossible.")
            print(f"     Try std=0.2 first to see gradual effect.")

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("To properly test noise impact, you MUST run TWO separate experiments:")
    print()
    print("1. Baseline (no noise):")
    print("   python fed_office.py --mode fedprox --wandb \\")
    print("       --wandb_run_name 'baseline-no-noise' \\")
    print("       --client_datasets '0,1:2,3' \\")
    print("       --wandb_project 'federated-final'")
    print()
    print("2. With noise:")
    print("   python fed_office.py --mode fedprox --wandb \\")
    print("       --wandb_run_name 'with-noise' \\")
    print("       --client_datasets '0,1:2,3' \\")
    print("       --wandb_project 'federated-final' \\")
    print("       --gaussian_std 0.2 \\")
    print("       --client_noise '0:input=0.5;1:input=0.5'")
    print()
    print("Then compare the two runs in Weights & Biases!")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    quick_check()