import torch

def get_device() -> str:
    """
    Get the best available device (CUDA > MPS > CPU).
    
    Returns:
        'cuda' for NVIDIA GPUs
        'mps' for Apple Silicon GPUs
        'cpu' for CPU only
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

