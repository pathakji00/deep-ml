import torch

def dice_statistics(n: int) -> tuple[float, float]:
    """
    Compute the expected value and variance of a fair n-sided die roll using PyTorch.

    Args:
        n (int): Number of sides of the die

    Returns:
        tuple: (expected_value, variance)
    """
    # Your code here
    e = round(float((n+1)/2),4)
    var = round(float(((n**2) -1)/12),4)
    stats =  [e,var]
    return stats

