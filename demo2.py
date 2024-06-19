import torch

xs = []
ys = []
for i in range(1, 10):
    xs.append(torch.tensor([i], dtype=torch.float))
    ys.append(torch.tensor([i * 2], dtype=torch.float))

print(xs)
print(ys)

w = torch.tensor([.1], dtype=torch.float, requires_grad=True)
b = torch.tensor([.0], dtype=torch.float, requires_grad=True)

for i in range(9):
    for j in range(9):
        x = xs[j]
        print("x:")
        print(x)
        y = ys[j]
        print("y:")
        print(y)
        y_ = x.matmul(w).add(b)
        print("y_:")
        print(y_)
        loss = (y_ - y) ** 2
        print("loss:")
        print(loss)

        loss.backward(loss)

        # print(w.grad)
        # print(b.grad)
        with torch.no_grad():
            w = w - 0.01 * w.grad
            b = b - 0.01 * b.grad

        w.requires_grad_(True)
        b.requires_grad_(True)

print(w)
print(b)
