import torch
class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # Learnable parameters
        self.gamma = torch.ones(dim, requires_grad=True)
        self.beta = torch.zeros(dim, requires_grad=True)
        # Buffers for running mean and variance (non-learnable)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            if x.ndim == 2:
                dim0 = 1
            elif x.ndim == 3: # For 3D input (e.g., (B, T, C))
                dim0 = (0, 1) # Mean over batch and time dimensions
            else:
                raise ValueError(f"Unsupported input dimensions: {x.ndim}")

            batch_mean = x.mean(dim=dim0, keepdim=True)
            batch_var = x.var(dim=dim0, keepdim=True)
            
            x_norm = (x - batch_mean) / (batch_var + self.eps).sqrt()
            
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
        else:
            # Use running statistics during inference
            x_norm = (x - self.running_mean) / (self.running_var + self.eps).sqrt()

        self.out = self.gamma * x_norm + self.beta
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]