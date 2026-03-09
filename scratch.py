import torch

mean = torch.tensor([0, 10])
std = torch.tensor([1, 2])
dist = torch.distributions.Normal(mean, std)

print(dist.log_prob(torch.tensor([4, 10])))
