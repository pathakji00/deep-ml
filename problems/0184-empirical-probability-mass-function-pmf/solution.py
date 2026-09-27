import torch

def empirical_pmf(samples: torch.Tensor) -> list:
    """
    Given a 1D tensor of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    emp = []
    values , counts = torch.unique(samples,return_counts  = True)
    pmf = counts.float()/len(samples)
    for v,p in zip(values,pmf):
        emp.append((v.item(),p.item()))
    return emp
