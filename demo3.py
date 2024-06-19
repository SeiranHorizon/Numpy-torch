import torch

def sigmoid(x):
    return torch.exp(-x)

def sigmoid2(x):
    return 1 + torch.exp(-x)

# x = torch.tensor([1.], dtype=torch.float, requires_grad=True)
# y = sigmoid2(x)
# print(y)
# z = torch.tensor([10.], dtype=torch.float)
# y.backward(z)
# print(x.grad)

x = torch.tensor([1.1], dtype=torch.float, requires_grad=True)
# y = x ** 2
# print(y)
# y.backward()
# print(x.grad)

# loss = y - 10
# print(loss)
# loss.backward(loss)
# print(x.grad)

y = 1 / (1 + torch.exp(-x))
print(y)
y.backward()
print(x.grad)

# 0.33287108/1.3328711*1.3328711

x1 = torch.tensor([1.1], dtype=torch.float, requires_grad=True)
y1 = 1 / (1 + x1)
print(y1)
y1.backward()
print(x1.grad)

# -1 / 2.1 * 2.1

x2 = torch.tensor([1.1], dtype=torch.float, requires_grad=True)
y2 = 2 * x2
print(y2)
y2.backward()
print(x2.grad)
