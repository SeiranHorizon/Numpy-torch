from autograd import Tensor

x = Tensor([[1., 2., 3.]])
w = Tensor([[2.], [3.], [4.]], requires_grad=True)
b = Tensor([.0], requires_grad=True)
y_ = x.matmul(w).add(b)
y = Tensor([60.])
loss = y_.sub(y).pow(Tensor(2.)).div(Tensor(2.))

print(loss.narray)

loss.backward()

print(w.grad)
print(b.grad)

w.narray = w.narray + (0.001 * w.grad.T)
b.narray = b.narray + (0.001 * b.grad.T)

print(w.narray)
print(b.narray)