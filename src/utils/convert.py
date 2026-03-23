import torch
import numpy as np

def to_torch_tensor(batch, device=None, dtype=torch.float32):
    """
    Convert a batch of data to a torch tensor.
    """
    if isinstance(batch, np.ndarray):
        tensor = torch.tensor(batch, dtype=dtype)
        if device is not None:
            tensor = tensor.to(device)
    elif isinstance(batch, torch.Tensor):
        tensor = batch.to(dtype=dtype)
        if device is not None:
            tensor = tensor.to(device)
    elif isinstance(batch, dict):
        tensor = {k: to_torch_tensor(v, device, dtype) for k, v in batch.items()}
    else:
        return batch  # Return as is if not a recognized type
    
    return tensor

def to_numpy_array(batch):
    """
    Convert a batch of data to a numpy array.
    """
    if isinstance(batch, torch.Tensor):
        return batch.cpu().numpy()
    elif isinstance(batch, np.ndarray):
        return batch
    elif isinstance(batch, dict):
        return {k: to_numpy_array(v) for k, v in batch.items()}
    else:
        return batch  # Return as is if not a recognized type