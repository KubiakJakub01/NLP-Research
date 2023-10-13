'''Implement the core engine of micrograd, including the Function and Tensor classes and the backward pass.

Inspired from: https://github.com/karpathy/micrograd/tree/master
'''

import numpy as np

class Tensor:
    '''A tensor is a multi-dimensional array of numbers.'''

    def __init__(self, data, requires_grad=False, _op=''):
        '''Store the data and gradient status.'''
        self.data = data
        if not isinstance(data, np.ndarray):
            self.data = np.array(data)
        self.requires_grad = requires_grad
        self._grad = None
        self._grad_fn = None

    @property
    def shape(self):
        '''Return the shape of the data.'''
        return self.data.shape

    @property
    def grad(self):
        '''Return the gradient of the tensor.'''
        return self._grad

    @property
    def grad_fn(self):
        '''Return the gradient function of the tensor.'''
        return self._grad_fn

    def __repr__(self):
        '''Return a string representation of the tensor.'''
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def backward(self, grad=None):
        '''Backward pass through the computational graph.'''
        if self.grad_fn is None:
            raise RuntimeError("Can't call backward on a tensor that has no grad_fn")
        if grad is None:
            grad = np.ones_like(self.data)
        self.grad_fn.backward(grad)

    def __add__(self, other):
        '''Add two tensors.'''
        out = Tensor(np.add(self.data, other.data), _op='+')

        def _backward(grad):
            self.grad = grad
            other.grad = grad
        out._grad_fn = _backward

        return out
    
    def __mul__(self, other):
        '''Multiply two tensors.'''
        out = Tensor(np.multiply(self.data, other.data), _op='*')

        def _backward(grad):
            self.grad = grad * other.data
            other.grad = grad * self.data
        out._grad_fn = _backward

        return out

    def __sub__(self, other):
        '''Subtract two tensors.'''
        out = Tensor(np.subtract(self.data, other.data), _op='-')

        def _backward(grad):
            self.grad = grad
            other.grad = -grad
        out._grad_fn = _backward

        return out
    
    def __truediv__(self, other):
        '''Divide two tensors.'''
        out = Tensor(np.divide(self.data, other.data), _op='/')

        def _backward(grad):
            self.grad = grad / other.data
            other.grad = -grad * self.data / (other.data ** 2)
        out._grad_fn = _backward

        return out
