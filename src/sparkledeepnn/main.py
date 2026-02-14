from Linear import Linear
from BatchNorm import BatchNorm1d
from Activations import Tanh

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


# Create the datasets of the bigrams
def build_datasets(block: int, names: list[str], ratio=(0.8, 0.1, 0.1)):
    X, Y = [], []
    for n in names:
        context = [0]*block
        for c in n + ".":
            X.append(context)
            Y.append(stoi[c])

            context = context[1:] + [stoi[c]]
    return torch.tensor(X[0:int(0.8*len(X))]), torch.tensor(Y[0:int(0.8*len(Y))]), torch.tensor(X[int(0.8*len(X)):int(0.9*len(X))]), torch.tensor(Y[int(0.8*len(Y)):int(0.9*len(Y))]), torch.tensor(X[int(0.9*len(X)):]), torch.tensor(Y[int(0.9*len(Y)):])


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

maxsteps = 130000
batch_size = 64
lossi = []
block_size = 5

g = torch.Generator().manual_seed(2147483647)

C = torch.randn((27, n_embd), generator=g)

# Create the train, validation, and test datasets based on our vocabulary and the names
Xtr, Ytr, Xval, Yval, Xte, Yte = build_datasets(block_size, names)


## Create the model layers and parameters
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

for p in parameters:
    p.requires_grad = True


## Train the model
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
    
## Sample from the trained model
for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
      emb = C[torch.tensor([context])] # (1, block_size, n_embd)
      x = emb.view(1, -1)
      for layer in layers:
        x = layer(x)
      logits = x
      probs = F.softmax(logits, dim=1)
      # sample from the distribution
      ix = torch.multinomial(probs, num_samples=1, generator=g).item()
      # shift the context window and track the samples
      context = context[1:] + [ix]
      out.append(ix)
      # if we sample the special '.' token, break
      if ix == 0:
        break
    
    print(''.join(itos[i] for i in out)) # decode and print the generated word
