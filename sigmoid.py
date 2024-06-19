from autograd import Tensor
import numpy as np

def sigmoid(x):
    return x.exp()

def sigmoid2(x):
    return Tensor(1).add(x.ng().exp())

x = Tensor([[1.1]], requires_grad=True)
# y = x.pow(Tensor(2))
# print(y.narray)

# y.backward()

# print(x.grad.narray)

# loss = y.sub(Tensor(10))

# print(loss.narray)

# loss.backward(loss)

# print(x.grad.narray)

# y = Tensor(1).div(Tensor(1).add(x.ng().exp()))
# print(y.narray)

y = Tensor(1).add(x.ng().exp())
print(y.narray)

y.backward()

print(x.grad.narray)

# x = Tensor([[1]], requires_grad=True)
# y = x.exp()
# print(y.narray)
# y.backward()
# print(x.grad.narray)