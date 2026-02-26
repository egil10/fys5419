

import numpy as np

class One_qubit:
    def __init__(self):
        self.state = np.zeros(2, dtype=np.complex_)
        self.I = np.eye(2)
        self.Z = np.array([[1,0],[0,-1]])
        self.X = np.array([[0,1],[1,0]])
        self.Y = np.array([[0,-1j],[1j,0]])
        self.H = np.array([[1,1],[1,-1]]) / np.sqrt(2)
        self.S = np.array([[1,0],[0,1j]])


