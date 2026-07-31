import numpy as np
import matplotlib.pyplot as plt

from mesh            import X,T,Nint,plotMesh,dofsToCoords,coordsToDofs
from differentiation import derivadaNumerica,hessianaNumerica
from distortion      import calculaDistorsioMalla

#Funció objectiu
def F(y):
    return calculaDistorsioMalla(dofsToCoords(y), T)

#Malla i distorsió inicials
plotMesh(X,'Initial mesh')
res = calculaDistorsioMalla(X, T)
print('Distorsió inicial: ',res)

y = coordsToDofs(X)
R = derivadaNumerica(F, y)
rk = 1
errors = []

# Mètode de Newton-Raphson
while rk >= 0.5e-7:
    J = hessianaNumerica(F, y)
    s = np.linalg.solve(J, -R)
    rk = np.linalg.norm(s) / np.linalg.norm(y + s)
    errors.append(rk)
    y = y + s
    R = derivadaNumerica(F, y)

X_final = dofsToCoords(y)
print('Posició primer node interior: ',X_final[0, :])
print('Distorsió final: ',calculaDistorsioMalla(X_final, T))
plotMesh(X_final, 'Final mesh')

num_iter = len(errors)
print("Nombre d'iteracions:", num_iter)

# Gràfica de convergència
plt.plot(range(1, num_iter+1), np.log10(errors), marker='o')
plt.xlabel('Iteració k')
plt.ylabel(r'$\log_{10}(r_k)$')
plt.grid(True)
plt.show()

