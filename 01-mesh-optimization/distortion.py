import numpy as np
import matplotlib.pyplot as plt

A = np.array([[1, -np.sqrt(3.) / 3], [0, 2 * np.sqrt(3.) / 3]],dtype=float)

def distorsioTriangle(Xe):
    # Calcula la matriu diferencial Dϕ per al triangle Xe
    DPhi = np.zeros((2, 2))
    M1 = np.zeros((2, 2))
    M1[0, :] = Xe[1, :] - Xe[0, :]
    M1[1, :] = Xe[2, :] - Xe[0, :]
    DPhi = (M1.T)@A
    
    # Distorsió del triangle
    num = np.linalg.norm(DPhi, 'fro')**2
    return num / (2*abs(np.linalg.det(DPhi)))

def calculaDistorsioMalla(X, T):
    n = T.shape[0]
    suma = 0
    for e in range(n):
        suma += distorsioTriangle(X[T[e, :], :])**2
    return np.sqrt(suma)
