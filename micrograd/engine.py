'''Implement the core engine of micrograd, including the Function and Tensor classes and the backward pass.

Inspired from: https://github.com/karpathy/micrograd/tree/master
'''

import numpy as np

class Tensor:
    '''A tensor is a multi-dimensional array of numbers.'''

    def __init__(self, data, requires_grad=False):
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
