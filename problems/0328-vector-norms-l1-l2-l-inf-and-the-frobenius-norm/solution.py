import torch

def compute_norm(arr: torch.Tensor, norm_type: str) -> float:
    """
    Compute the specified norm of the input tensor.

    'l1', 'l2' and 'linf' are entrywise norms and accept a 1D or 2D tensor.
    'frobenius' is a matrix norm and must raise a ValueError if arr is not 2D.

    Args:
        arr: Input tensor (1D or 2D)
        norm_type: Type of norm ('l1', 'l2', 'linf', or 'frobenius')

    Returns:
        The computed norm as a float
    """
    
    if not arr.is_floating_point():
        arr = arr.float()
    f_arr = arr.flatten()
    if norm_type == "l1":
        return torch.linalg.norm(f_arr, ord=1).item()
        
    elif norm_type == "l2":
        return torch.linalg.norm(f_arr, ord=2).item()
        
    elif norm_type == "linf":
        return torch.linalg.norm(f_arr, ord=float('inf')).item()
        
    elif norm_type == "frobenius":
        
        if arr.ndim != 2:
            raise ValueError("Frobenius norm is strictly a matrix norm and requires a 2D tensor.")
        return torch.linalg.norm(arr, ord='fro').item()
        
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")
