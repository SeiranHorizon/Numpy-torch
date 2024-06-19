import torch

x = torch.tensor([[1, 2, 3]], dtype=torch.float)
w = torch.tensor([[2], [3], [4]], dtype=torch.float, requires_grad=True)
b = torch.tensor([[0]], dtype=torch.float, requires_grad=True)
y_ = torch.matmul(x, w) + b
print(y_)
y = torch.tensor([60], dtype=torch.float)
loss = (y_ - y) ** 2 / 2
print(loss)
y_.backward(loss)
print(w.grad)
print(b.grad)