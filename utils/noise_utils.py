"""
Noise utilities for federated learning experiments
Supports Gaussian input noise and label noise injection
FIXED: Uses independent PyTorch Generator for reproducible, isolated noise
"""
import torch
import numpy as np
import random


class NoiseInjector:
    """
    Handles noise injection for federated learning clients
    Supports both input (Gaussian) noise and label noise
    FIXED: Uses separate PyTorch generator to avoid interference with training RNG
    """

    def __init__(self, seed=42):
        """
        Initialize noise injector with a seed for reproducibility

        Args:
            seed: Random seed for reproducible noise patterns
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # CRITICAL FIX: Create independent PyTorch generator
        self.torch_generator = torch.Generator()
        self.torch_generator.manual_seed(seed)

        # Also create CUDA generator if available
        if torch.cuda.is_available():
            self.torch_generator_cuda = torch.Generator(device='cuda')
            self.torch_generator_cuda.manual_seed(seed)
        else:
            self.torch_generator_cuda = None

    def add_gaussian_noise(self, images, noise_ratio, std=0.1):
        """
        Add Gaussian noise to input images

        Args:
            images: Tensor of images [B, C, H, W]
            noise_ratio: Probability of adding noise to each image (0-1)
            std: Standard deviation of Gaussian noise

        Returns:
            Noisy images tensor
        """
        if noise_ratio <= 0:
            return images

        noisy_images = images.clone()
        batch_size = images.size(0)

        # Determine which images get noise based on noise_ratio
        noise_mask = self.rng.rand(batch_size) < noise_ratio

        # Select appropriate generator based on device
        if images.is_cuda and self.torch_generator_cuda is not None:
            generator = self.torch_generator_cuda
        else:
            generator = self.torch_generator

        for i in range(batch_size):
            if noise_mask[i]:
                # CRITICAL FIX: Use independent generator instead of global RNG
                noise = torch.randn(
                    images[i].shape,
                    generator=generator,
                    device=images.device,
                    dtype=images.dtype
                ) * std
                noisy_images[i] = images[i] + noise
                # Clip to valid range [0, 1]
                noisy_images[i] = torch.clamp(noisy_images[i], 0, 1)

        return noisy_images

    def add_label_noise(self, labels, noise_ratio, num_classes):
        """
        Add label noise by randomly flipping labels

        Args:
            labels: Tensor of labels [B]
            noise_ratio: Probability of flipping each label (0-1)
            num_classes: Total number of classes

        Returns:
            Noisy labels tensor
        """
        if noise_ratio <= 0:
            return labels

        noisy_labels = labels.clone()
        batch_size = labels.size(0)

        # Determine which labels get flipped based on noise_ratio
        noise_mask = self.rng.rand(batch_size) < noise_ratio

        for i in range(batch_size):
            if noise_mask[i]:
                # Randomly select a different class
                original_label = labels[i].item()
                possible_labels = list(range(num_classes))
                possible_labels.remove(original_label)
                noisy_labels[i] = self.rng.choice(possible_labels)

        return noisy_labels


class NoisyDatasetWrapper(torch.utils.data.Dataset):
    """
    Wrapper for applying noise to datasets dynamically
    FIXED: Correctly determines number of classes
    """

    def __init__(self, dataset, noise_injector,
                 input_noise_ratio=0.0, label_noise_ratio=0.0,
                 num_classes=None, gaussian_std=0.1):
        """
        Args:
            dataset: Base dataset to wrap
            noise_injector: NoiseInjector instance
            input_noise_ratio: Probability of adding Gaussian noise to inputs
            label_noise_ratio: Probability of flipping labels
            num_classes: Number of classes for label noise (auto-detect if None)
            gaussian_std: Standard deviation for Gaussian noise
        """
        self.dataset = dataset
        self.noise_injector = noise_injector
        self.input_noise_ratio = input_noise_ratio
        self.label_noise_ratio = label_noise_ratio
        self.gaussian_std = gaussian_std

        if num_classes is None:
            # Try to infer from dataset
            if hasattr(dataset, 'classes'):
                self.num_classes = len(dataset.classes)
            else:
                # Scan dataset to find max label
                max_label = 0
                sample_size = min(100, len(dataset))
                for i in range(sample_size):
                    _, label = dataset[i]
                    if isinstance(label, torch.Tensor):
                        label = label.item()
                    max_label = max(max_label, label)
                self.num_classes = max_label + 1
                print(f"Auto-detected {self.num_classes} classes from dataset")
        else:
            self.num_classes = num_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, label = self.dataset[idx]

        # Add input noise if configured
        if self.input_noise_ratio > 0:
            # Add batch dimension for noise function
            data_batch = data.unsqueeze(0)
            noisy_data = self.noise_injector.add_gaussian_noise(
                data_batch, self.input_noise_ratio, self.gaussian_std
            )
            data = noisy_data.squeeze(0)

        # Add label noise if configured
        if self.label_noise_ratio > 0:
            # Convert to tensor if needed
            if not isinstance(label, torch.Tensor):
                label = torch.tensor(label)
            label_batch = label.unsqueeze(0)
            noisy_label = self.noise_injector.add_label_noise(
                label_batch, self.label_noise_ratio, self.num_classes
            )
            label = noisy_label.squeeze(0).item()

        return data, label


def get_client_noise_config(client_idx, args):
    """
    Get noise configuration for a specific client based on args

    Args:
        client_idx: Client index (0-3 for Amazon, Caltech, DSLR, Webcam)
        args: Arguments with noise configuration

    Returns:
        dict: Configuration with input_noise_ratio, label_noise_ratio, mixed_domains
    """
    config = {
        'input_noise_ratio': 0.0,
        'label_noise_ratio': 0.0,
        'mixed_domains': None,
        'gaussian_std': getattr(args, 'gaussian_std', 0.1)
    }

    # Parse client-specific noise configuration
    # Format: --client_noise "0:input=0.2,label=0.0,mix=3;1:input=0.0,label=0.3,mix=2"
    if hasattr(args, 'client_noise') and args.client_noise:
        noise_configs = args.client_noise.split(';')
        for noise_config in noise_configs:
            if not noise_config.strip():
                continue
            parts = noise_config.split(':')
            if len(parts) != 2:
                continue
            client_id = int(parts[0].strip())
            if client_id != client_idx:
                continue

            # Parse configuration for this client
            params = parts[1].split(',')
            for param in params:
                if '=' in param:
                    key, value = param.split('=')
                    key = key.strip()
                    value = value.strip()

                    if key == 'input':
                        config['input_noise_ratio'] = float(value)
                    elif key == 'label':
                        config['label_noise_ratio'] = float(value)
                    elif key == 'mix':
                        config['mixed_domains'] = int(value)

    # Alternative: individual client arguments
    if hasattr(args, f'client{client_idx}_input_noise'):
        config['input_noise_ratio'] = getattr(args, f'client{client_idx}_input_noise')
    if hasattr(args, f'client{client_idx}_label_noise'):
        config['label_noise_ratio'] = getattr(args, f'client{client_idx}_label_noise')
    if hasattr(args, f'client{client_idx}_mix_with'):
        config['mixed_domains'] = getattr(args, f'client{client_idx}_mix_with')

    return config


def print_noise_summary(datasets_names, noise_configs):
    """
    Print a summary of noise configuration for all clients

    Args:
        datasets_names: List of dataset names
        noise_configs: List of noise configurations
    """
    print("\n" + "=" * 70)
    print("NOISE CONFIGURATION SUMMARY")
    print("=" * 70)

    for idx, (name, config) in enumerate(zip(datasets_names, noise_configs)):
        print(f"\nClient {idx} ({name}):")
        print(f"  Input Noise:  {config['input_noise_ratio'] * 100:.1f}% (Gaussian, std={config['gaussian_std']})")
        print(f"  Label Noise:  {config['label_noise_ratio'] * 100:.1f}%")
        if config['mixed_domains'] is not None:
            mixed_with = datasets_names[config['mixed_domains']]
            print(f"  Mixed with:   Client {config['mixed_domains']} ({mixed_with})")
        else:
            print(f"  Mixed with:   None (single domain)")

    print("=" * 70 + "\n")