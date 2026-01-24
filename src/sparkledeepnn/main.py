import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out), generator=g ) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None 
    
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

class BatchNorm1d:
    def __init__(self, dim, eps = 0.5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum 
        self.training = True 
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        #calculate the forward pass
        if self.training:
            xmean = x.mean(dim = 0, keepdim = True)
            xvar = x.var(dim = 0, keepdim = True)
        else:
            xmean = self.running_mean 
            xvar = self.running_var 
        
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta 

        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]
    
class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x) 
        return self.out
    def parameters(self):
        return []


# Create the dataset of the bigrams
def build_dataset(block: int, names: list[str]):
    X, Y = [], []
    for n in names:
        context = [0]*block
        for c in n + ".":
            X.append(context)
            Y.append(stoi[c])

            context = context[1:] + [stoi[c]]
    return torch.tensor(X), torch.tensor(Y)




### Main deep learning model training and evalution flow ###

# Load the dataset 
names = open('data/names.txt', 'r').read().splitlines()

# Create the vocabulary 
chars = sorted(list(set(''.join(names))))
stoi = {c: i + 1 for i, c in enumerate(chars)}
stoi['.'] = 0
itos = {i: c for c, i in stoi.items()}

n_embd = 10 # The dimensionality of the character embedding vector 
n_hidden = 100 # The number of hidden units in the feed forward network 

maxsteps = 200000
batch_size = 64
lossi = []
block_size = 3

g = torch.Generator().manual_seed(2147483647)

C = torch.randn((27, n_embd), generator=g)

Xtr, Ytr = build_dataset(block_size, names)

layers  = [
    Linear(n_embd * block_size, n_hidden, bias=False),
    BatchNorm1d(n_hidden),
    Tanh(),
    Linear(n_hidden, n_hidden, bias=False),
    BatchNorm1d(n_hidden),
    Tanh(),
    Linear(n_hidden, 27)
]

with torch.no_grad():
    layers[-1].weight *= 0.1 # make last layer less confident 
    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            layer.weight *= 5/3 # to unsquash the output. 
    
parameters = [C] + [p for layer in layers for p in layer.parameters()]

print(sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad = True


for i in range(maxsteps):
    #sample batch_size random sequences
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb = Xtr[ix]
    Yb = Ytr[ix]

    emb = C[Xb]
    x = emb.view(emb.shape[0], -1)
    for layer in layers:
        x = layer(x)
    
    loss = F.cross_entropy(x, Yb)

    #backward pass 
    for layer in layers:
        layer.out.retain_grad()
    for p in parameters:
        p.grad = None 
    loss.backward()

    #update
    lr = 0.1 if i < 100000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad.data

    # track stats
    if i % 10000 == 0:
        print(f"step {i}: loss {loss.log10().item()}")
    lossi.append(loss.log10().item())
    