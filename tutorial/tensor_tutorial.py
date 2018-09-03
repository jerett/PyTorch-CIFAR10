

import torch


x = torch.empty(5, 3)
print(x)
x = torch.rand(5, 3)
print('x:', x)

y = torch.rand(5, 3)
print('y', y)
print(torch.add(x, y))

a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

if torch.cuda.is_available():
    device = torch.device('cuda')
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    print('x:', x)
    print('y:', y)
    z = x + y
    print(z)

