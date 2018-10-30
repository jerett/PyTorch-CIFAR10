

import torch


# x = torch.empty(5, 3)
# print(x)
# x = torch.rand(5, 3)
# print('x:', x)
#
# y = torch.rand(5, 3)
# print('y', y)
# print(torch.add(x, y))
#
# a = torch.ones(5)
# print(a)
#
# b = a.numpy()
# print(b)
#
# if torch.cuda.is_available():
#     device = torch.device('cuda')
#     y = torch.ones_like(x, device=device)
#     x = x.to(device)
#     print('x:', x)
#     print('y:', y)
#     z = x + y
#     print(z)



x = torch.FloatTensor([[1., 2.]])
w1 = torch.FloatTensor([[2.], [1.]])
w2 = torch.FloatTensor([3.])
w1.requires_grad = True
w2.requires_grad = True

d = torch.matmul(x, w1)  # 4
print(d)

# d_ = d.data
d_ = d.detach()

f = torch.matmul(d, w2)  # 12
print('f:', f)

d_[:] = 1

f.backward()
print('w2 grad:', w2.grad)

