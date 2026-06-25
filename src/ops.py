import torch

import random_ext

x = torch.randn(3, device="cuda" if torch.cuda.is_available() else "cpu")
y = random_ext.random_op(x, 5)

print(y)
print(y.device, y.dtype)
