import torch

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> torch.Tensor:
    """
    Compute the derivative of f(x) = g(x)/h(x) at point x using the quotient rule.
    
    Args:
        g_coeffs: Coefficients of numerator polynomial in descending order
        h_coeffs: Coefficients of denominator polynomial in descending order
        x: Point at which to evaluate the derivative
        
    Returns:
        The derivative value f'(x) as a scalar torch.Tensor
    """
    # Your code here
    x_t = torch.tensor(x,dtype=torch.float32,requires_grad = True)
    def poly_func(coeff,x_val):
        coeff_t  = torch.tensor(coeff,dtype = torch.float32)
        n_terms = len(coeff)
        powers = torch.arange(n_terms-1, -1, -1,dtype = torch.float32)
        terms = coeff_t*(x_val**powers)
        return torch.sum(terms)
    g_x = poly_func(g_coeffs,x_t)
    h_x = poly_func(h_coeffs,x_t)
    f_x = g_x/h_x
    f_x.backward()
    return x_t.grad

    