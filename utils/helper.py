import torch

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Use Apple GPU (Metal)
    elif torch.cuda.is_available():
        return torch.device("cuda")  # Use NVIDIA GPU (if somehow available)
    return torch.device("cpu")  # Fallback