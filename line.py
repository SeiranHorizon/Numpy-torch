import numpy as np
from autograd import Tensor

t1 = Tensor(1.)
print(t1.narray.dtype)

xs = []
ys = []
for i in range(1, 10):
    xs.append(Tensor([[i]]))
    ys.append(Tensor([[i * 2]]))

w = Tensor([[.1]], requires_grad=True)
b = Tensor([.0], requires_grad=True)

for i in range(9):
    for j in range(9):
        x = xs[j]
        y = ys[j]
        print('x:')
        print(x.narray)
        print('y:')
        print(y.narray)
        y_ = x.matmul(w).add(b)
        print('y_:')
        print(y_.narray)
        loss = y_.sub(y).pow(Tensor(2.))
        print('loss:')
        print(loss.narray)
        loss.backward()

        # print(w.grad)
        # print(b.grad)
        
        w.narray = w.narray - (0.01 * w.grad.T)
        b.narray = b.narray - (0.01 * b.grad.T)

print(w.narray)
print(b.narray)