import numpy as np
import scipy
import matplotlib.pyplot as plt

def f(x):
    return np.sin(np.e**(2*x))

def trapezi(f, m, a, b):
    h = (b - a) / m
    I = f(a)
    for i in range(1, m):
        I += 2*f(a + h*i)
    return (I + f(b)) * h / 2

def simpson(f, m, a, b):
    h = (b - a) / (2*m)
    I = f(a) + f(b)
    for i in range(1, 2*m):
        coef = 4 if i % 2 == 1 else 2
        I += coef * f(a + i*h)
    return I * h / 3

def simpson_simple(f, a, b):
    return ((b - a) / 6) * (f(a) + 4*f((a + b)/2) + f(b))

def simpson_adaptativa(f, a, b, eps):
    S_ab = simpson_simple(f, a, b)
    c = (a + b)/2
    S_ac = simpson_simple(f, a, c)
    S_cb = simpson_simple(f, c, b)
    E_ab = abs(S_ab - (S_ac + S_cb))
    
    if E_ab < eps*(b-a):
        return S_ab, [a, b]
    else:
        # Crida recursiva a la meitat esquerra i dreta
        I_left, points_left = simpson_adaptativa(f, a, c, eps)
        I_right, points_right = simpson_adaptativa(f, c, b, eps)
        # Combinar punts evitant duplicats del punt central
        points = points_left[:-1] + points_right
        return I_left + I_right, points

a, b = 0, 2
I_exacta = scipy.integrate.quad(f, a, b)[0]
print("Integral exacta:", I_exacta)

errors_t = []
errors_s = []
n_trap = []
n_simp = []
ms = [4, 8, 16, 32]

for m in ms:
    I1 = trapezi(f, m, a, b)
    print("Trapezi m =", m, ":", I1)
    errors_t.append(np.abs(I_exacta - I1))
    n_trap.append(m + 1)  # m subintervals -> m+1 avaluacions

    I2 = simpson(f, m, a, b)
    print("Simpson m =", m, ":", I2)
    errors_s.append(np.abs(I_exacta - I2))
    n_simp.append(2*m + 1)  # 2m subintervals petits -> 2m+1 avaluacions

plt.plot(np.log10(n_trap), np.log10(errors_t), 'o-', label='ErrorTrapezi')
plt.plot(np.log10(n_simp), np.log10(errors_s), 'o-', label='ErrorSimpson')
plt.xlabel('log10(Número de evaluaciones de f(x))')
plt.ylabel('log10(Error absolut)')
plt.title("Error vs número d'avaluacions")
plt.legend()
plt.show()

# Calcular k_T y k_S
error_t_128 = np.abs(I_exacta - trapezi(f, 128, a, b))
error_s_128 = np.abs(I_exacta - simpson(f, 128, a, b))
k_T = error_t_128 * 128**2
k_S = error_s_128 * 128**4 
print("k_T ≈", k_T)
print("k_S ≈", k_S)
m_T = int((k_T / (0.5 * 10**-6)) ** 0.5)
print("m_T =", m_T)
m_S = int((k_S / (0.5 * 10**-6)) ** 0.25)
print("m_S =", m_S)
print("Error Trapezi =", np.abs(I_exacta - trapezi(f, m_T, a, b)))
print("Error Simpson =", np.abs(I_exacta - simpson(f, m_S, a, b)))

eps1 = 0.5e-3
eps2 = 0.5e-6
I_eps1, punts1 = simpson_adaptativa(f, a, b, eps1)
I_eps2, punts2 = simpson_adaptativa(f, a, b, eps2)
# Aproximación con eps = 1e-3
error1 = abs(I_exacta - I_eps1)
print("Simpson adaptatiu eps=1e-3:", I_eps1, "Error:", error1)

# Aproximación con eps = 1e-6
error2 = abs(I_exacta - I_eps2)
print("Simpson adaptatiu eps=1e-6:", I_eps2, "Error:", error2)

# Crear gràfica
x = np.linspace(a, b, 1000)
y = f(x)

# Gráfica para eps1
plt.plot(x, y, label='f(x)')
plt.plot(punts1, f(np.array(punts1)), 'bo')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.show()

# Gráfica para eps2
plt.plot(x, y, label='f(x)')
plt.plot(punts2, f(np.array(punts2)), 'bo')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.show()

print("Número de punts Simpson adaptatiu:", len(punts2))
print("Error trapezi amb m = 333:", np.abs(I_exacta - simpson(f, 333, a, b)))
