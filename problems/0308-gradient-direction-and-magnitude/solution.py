import torch

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient (float)
		- direction: Unit vector (torch.Tensor) in direction of steepest ascent
		- descent_direction: Unit vector (torch.Tensor) in direction of steepest descent
	"""
	# Your code here
	grad_t = torch.tensor(gradient,dtype = torch.float32)
	mag = torch.linalg.norm(grad_t,ord = 2).item()
	if mag == 0.0:
		return {
			'magnitude': 0.0,
			'direction' : torch.zeros(2),
			'descent_direction' : torch.zeros(2)
		}
	dir_n = grad_t/mag
	descent = dir_n * (-1)
	dataset = {'magnitude':mag,
				'direction':dir_n,
				'descent_direction':descent}
	return dataset
	