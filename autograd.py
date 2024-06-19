import numpy as np

class Tensor:
    def __init__(self, narray, requires_grad=False):
        self.narray = np.array(narray, dtype=np.float32)
        self.op = ''
        self.requires_grad = requires_grad
        self.type = 'tensor'
        self.nodes = []
        self.grad = None

    def add(self, x):
        n = Tensor(self.narray + x.narray)
        n.nodes.append(self)
        n.nodes.append(x)
        n.type = 'op'
        n.op = 'add'
        return n
    
    def sub(self, x):
        n = Tensor(self.narray - x.narray)
        n.nodes.append(self)
        n.nodes.append(x)
        n.type = 'op'
        n.op = 'sub'
        return n

    def pow(self, x):
        n = Tensor(self.narray ** x.narray)
        n.nodes.append(self)
        n.nodes.append(x)
        n.type = 'op'
        n.op = 'pow'
        return n

    def div(self, x):
        n = Tensor(self.narray / x.narray)
        n.nodes.append(self)
        n.nodes.append(x)
        n.type = 'op'
        n.op = 'div'
        return n

    def ng(self):
        n = Tensor(-self.narray)
        n.nodes.append(self)
        n.type = 'op'
        n.op = 'ng'
        return n

    def exp(self):
        n = Tensor(np.exp(self.narray))
        n.nodes.append(self)
        n.type = 'op'
        n.op = 'exp'
        return n

    def matmul(self, x):
        n = Tensor(self.narray.dot(x.narray))
        n.nodes.append(self)
        n.nodes.append(x)
        n.type = 'op'
        n.op = 'matmul'
        return n

    def backward(self, grad=None):
        if self.type == "op":
            if self.op == "add":
                print('add')
                l = self.nodes[0]
                r = self.nodes[1]
                l.backward(grad)
                r.backward(grad)
            elif self.op == "sub":
                print('sub')
                l = self.nodes[0]
                r = self.nodes[1]
                l.backward(grad)
                r.backward(grad)
            elif self.op == "pow":
                print('pow')
                l = self.nodes[0]
                r = self.nodes[1]
                if self.nodes[0].requires_grad:
                    self.nodes[0].grad = Tensor(r.narray * np.power(l.narray, r.narray - 1))
                    if grad:
                        self.nodes[0].grad.narray = self.nodes[0].grad.narray * grad.narray
                return Tensor(r.narray * np.power(l.narray, r.narray - 1))
            elif self.op == "ng":
                print('ng')
                l = self.nodes[0]
                if self.nodes[0].requires_grad:
                    self.nodes[0].grad = Tensor(-1)
                    if grad:
                        self.nodes[0].grad.narray = self.nodes[0].grad.narray * grad.narray
            elif self.op == "exp":
                print('exp')
                l = self.nodes[0]
                if self.nodes[0].requires_grad:
                    self.nodes[0].grad = self
                l.backward(self)
            elif self.op == "div":
                print('div')
                l = self.nodes[0]
                r = self.nodes[1]
                def gradfun(dl, dr):
                    return dl * r - l * dr / r ** 2
        elif self.type == "tensor":
            return self