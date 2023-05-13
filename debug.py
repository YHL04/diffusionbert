

import torch

logits = torch.tensor([1, 2, 3, 4])

x = torch.distributions.categorical.Categorical(logits=logits).sample()

print(x)

