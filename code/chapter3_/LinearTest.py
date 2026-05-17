import torch
import torch.nn as nn

a = nn.Linear(4, 4)
b = nn.Linear(4, 4)

print(a.weight)
print(b.weight)
print(torch.equal(a.weight, b.weight))  # 通常是 False