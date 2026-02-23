
# ###############################################################
# CHAPTER 2: Quantum Computer Fundamentals
# ###############################################################

from __future__ import annotations 
import numpy as np
from absl import flags
import math

# ## 2.1 Tensors #################################################

class Tensor(np.ndarray):
    def __new__(cls, input_array, op_name=None) -> 'Tensor':
        obj = np.asarray(input_array, dtype=tensor_type()).view(cls)
        obj.name = op_name
        return obj

    def __array_finalize__(self, obj) -> None:
        if obj is None:
            return

class State(Tensor): pass 

flags.DEFINE_integer('tensor_width', 64, 'Width of complex (64,128)')

def tensor_width():
    return flags.FLAGS.tensor_width

def tensor_type():
    assert tensor_width() == 64 or tensor_width() == 128
    return np.complex64 if tensor_width() == 64 else np.complex128


def kron(self, arg: Tensor) -> Tensor:
    return self.__class__(np.kron(self, arg))

def __mul__(self, arg: Tensor) -> Tensor:
    return self.kron(arg)

def kpow(self, n: int) -> Tensor:
    if n == 0:
        return self.__class__(1,0)
    t = self
    for _ in range(n-1):
        t = np.kron(t, self)
    return self.__class__(t)

def is_close(self, arg) -> bool:
    return np.allclose(self, arg, atol=1e-6)

def is_hermitian(self) -> bool:
    if len(self.shape) != 2 or self.shape[0] != self.shape[1]:
        return False
    return self.is_close(np.conj(self.transpose()))

def is_unitary(self) -> bool:
    return Tensor(np.conj(self.transpose()) @ self).is_close(Tensor(np.eye(self.shape[0])))

def is_permutation(self) -> bool:
    x = self
    return (
        x.ndim == 2 and 
        x.shape[0] == x.shape[1] and
        (x.sum(axis=0) == 1).all() and
        (x.sum(axis=1) == 1).all() and
        ((x == 1) or (x == 0)).all()
        )

# ## 2.2 Qubits ##################################################

def qubit(alpha: complex = None, beta: complex = None) -> State:
    if alpha is None and beta is None:
        raise ValueError('alpha, beta, or both, need to be specified')
    if beta is None:
        beta = np.sqrt(1.0 - np.real(np.conj(alpha) * alpha))
    if alpha is None:
        alpha = np.sqrt(1.0 - np.real(np.conj(beta) * beta))
    return State(np.array([alpha, beta]))

    norm2 = np.real(np.conj(alpha)*alpha) + np.real(np.conj(beta)*beta)
    assert math.isclose(norm2, 1.0), 'Qubit probabilities not equal to 1'
    return State([alpha, beta])

# ## 2.3 Bloch Sphere #############################################

