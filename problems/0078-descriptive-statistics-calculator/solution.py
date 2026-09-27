import torch

def descriptive_statistics(data) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset using PyTorch.
    
    Args:
        data: List, torch.Tensor, or array-like of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    # Your code here
    data_t =  torch.tensor(data).flatten().float()
    mean_t = torch.mean(data_t).item()
    median_t = torch.quantile(data_t,0.5).item()
    mode_t = int((torch.mode(data_t,dim = 0).values.item()))
    var_t = torch.var(data_t,correction = 0).item()
    std_t = torch.std(data_t,correction = 0).item()
    q1= torch.quantile(data_t,.25).item()
    q2= torch.quantile(data_t,.50).item()
    q3= torch.quantile(data_t,.75).item()
    IQR = q3 -q1
    dataset = {
        'mean': mean_t,
        'median':median_t,
        'mode':mode_t,
        'variance':var_t,
        'standard_deviation': std_t,
        '25th_percentile':q1,
        '50th_percentile':q2,
        '75th_percentile':q3,
        'interquartile_range':IQR
        }

    return dataset

    