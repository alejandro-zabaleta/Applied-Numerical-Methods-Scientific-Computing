import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

def odef(t, y):
    R = 0.00132
    norma_v = np.sqrt(y[2]**2 + y[3]**2)
    return np.array([y[2], y[3], -R*norma_v*y[2], -R*norma_v*y[3] - 9.8])

def euler(f, a, b, m, alpha):
    h = (b - a) / m
    t = np.linspace(a, b, m + 1)
    y = np.zeros((m + 1, len(alpha)))
    y[0, :] = alpha

    for i in range(m):
        y[i + 1, :] = y[i, :] + h*f(t[i], y[i, :])

    return t, y

def RK4(f, a, b, m, alpha):
    h = (b - a) / m
    t = np.linspace(a, b, m + 1)
    y = np.zeros((m + 1, len(alpha)))
    y[0, :] = alpha

    for i in range(m):
        k1 = f(t[i], y[i, :])
        k2 = f(t[i] + h/2, y[i, :] + k1*h/2)
        k3 = f(t[i] + h/2, y[i, :] + k2*h/2)
        k4 = f(t[i + 1], y[i, :] + h*k3)
        y[i+1, :] = y[i, :] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    
    return t, y

def event(t, y):
    return y[1]  
event.terminal = True   
event.direction = -1    

def phi(theta):
    theta = float(theta)
    alpha = np.array([0, 0, 100*np.cos(theta), 100*np.sin(theta)])
    sol = solve_ivp(odef, [0, 20], alpha, method='RK45', events=event)
    x_final = sol.y[0, -1] 
    return x_final - 500

#Pregunta 1
a, b = 0, 10
m = 20
alpha = np.array([0, 0, 100*np.cos(np.pi / 4), 100*np.sin(np.pi / 4)])
y = euler(odef, a, b, m, alpha)[1]
print("Posició final Euler:", y[-1, 0], y[-1, 1])
plt.plot(y[:, 0], y[:, 1], 'o-', label = 'Euler')

y = RK4(odef, a, b, m, alpha)[1]
print("Posició final RK4:", y[-1, 0], y[-1, 1])
plt.plot(y[:, 0], y[:, 1], 'o-', label = 'RK4')


#Pregunta 2
m = 200
y = euler(odef, a, b, m, alpha)[1]
y2 = euler(odef, a, b, 2*m, alpha)[1]
xm = np.array([y[-1, 0], y[-1, 1]])
x2m = np.array([y2[-1, 0], y2[-1, 1]])
E = np.linalg.norm(xm - x2m)
r = E / np.linalg.norm(x2m)
print("Error absolut Euler:", E)
print("Error relatiu Euler:", r)

y = RK4(odef, a, b, m, alpha)[1]
y2 = RK4(odef, a, b, 2*m, alpha)[1]
xm = np.array([y[-1, 0], y[-1, 1]])
x2m = np.array([y2[-1, 0], y2[-1, 1]])
E = np.linalg.norm(xm - x2m)
r = E / np.linalg.norm(x2m)
print("Error absolut RK4:", E)
print("Error relatiu Rk4:", r)
plt.legend()
plt.show()


#Pregunta 3
ms = np.arange(10, 201)
errores_euler = []
errores_rk4 = []
evals_euler = []
evals_rk4 = []

for m in ms:
    # Euler
    y  = euler(odef, a, b, m, alpha)[1]
    y2 = euler(odef, a, b, 2*m, alpha)[1]
    xm  = np.array([y[-1, 0],  y[-1, 1]])
    x2m = np.array([y2[-1, 0], y2[-1, 1]])
    E = np.linalg.norm(xm - x2m)
    r = E / np.linalg.norm(x2m)
    errores_euler.append(r)
    evals_euler.append(m)      # 1 avaluació per pas

    # RK4
    y  = RK4(odef, a, b, m, alpha)[1]
    y2 = RK4(odef, a, b, 2*m, alpha)[1]
    xm  = np.array([y[-1, 0],  y[-1, 1]])
    x2m = np.array([y2[-1, 0], y2[-1, 1]])
    E = np.linalg.norm(xm - x2m)
    r = E / np.linalg.norm(x2m)
    errores_rk4.append(r)
    evals_rk4.append(4*m)      # 4 avaluacions per pas

# gràfica de convergència
log_evals_euler = np.log10(evals_euler)
log_errores_euler = np.log10(errores_euler)
log_evals_rk4 = np.log10(evals_rk4)
log_errores_rk4 = np.log10(errores_rk4)

plt.plot(log_evals_euler, log_errores_euler, 'o-', label='Euler')
plt.plot(log_evals_rk4, log_errores_rk4, 's-', label='RK4')
plt.title('Gràfica dels errors')
plt.legend()
plt.show()

# Pendents aproximadas  
pend_euler = (log_errores_euler[-1] - log_errores_euler[0]) / (log_evals_euler[-1] - log_evals_euler[0])
pend_rk4 = (log_errores_rk4[-1] - log_errores_rk4[0]) / (log_evals_rk4[-1] - log_evals_rk4[0])
print(f"Pendent aproximada Euler: {pend_euler:.2f}")
print(f"Pendent aproximada RK4: {pend_rk4:.2f}")

#Pregunta 4
#Solució de referència RK45
sol_ref = solve_ivp(odef, (a,b), alpha, method='RK45', rtol=1e-12, atol=1e-15)
x_ref = sol_ref.y[:,-1]
nfev_ref = sol_ref.nfev

m = 20
# Euler
t_e, y_e = euler(odef, a, b, m, alpha)
error_e = np.linalg.norm(y_e[-1,:] - x_ref) / np.linalg.norm(x_ref)
nfev_e = m

# RK4
t_rk4, y_rk4 = RK4(odef, a, b, m, alpha)
error_rk4 = np.linalg.norm(y_rk4[-1,:] - x_ref) / np.linalg.norm(x_ref)
nfev_rk4 = 4*m

# RK45 (pas variable)
sol_rk45 = solve_ivp(odef, (a,b), alpha, method='RK45', rtol=1e-6, atol=1e-9)
y_rk45 = sol_rk45.y
error_rk45 = np.linalg.norm(y_rk45[:,-1] - x_ref) / np.linalg.norm(x_ref)
nfev_rk45 = sol_rk45.nfev

# Resultats 
print("Métode     Error relatiu     Avaluacions")
print(f"Euler        {error_e:.2e}            {nfev_e}")
print(f"RK4          {error_rk4:.2e}            {nfev_rk4}")
print(f"RK45         {error_rk45:.2e}            {nfev_rk45}")

# Gràfica de trajectòries
plt.plot(y_e[:,0], y_e[:,1], 'o-', label='Euler')
plt.plot(y_rk4[:,0], y_rk4[:,1], 's-', label='RK4')
plt.plot(y_rk45[0,:], y_rk45[1,:], '-', label='RK45')
plt.legend()
plt.show()


#Pregunta 5
sol = solve_ivp(odef, [0, 20], alpha, 'RK45', events = event)
t_terra = sol.t_events[0][0]
x_terra = sol.y_events[0][0, 0]
print("Temps de vol:", t_terra)
print("Distància horitzontal:", x_terra)


#Pregunta 6
theta0 = np.pi / 4  # aproximació inicial
theta_sol = fsolve(phi, theta0)[0]

alpha = np.array([0, 0, 100*np.cos(theta_sol), 100*np.sin(theta_sol)])
sol = solve_ivp(odef, [0, 20], alpha, method='RK45', events=event)
x_terra = sol.y[0, -1]
t_terra = sol.t[-1]
print(f"Angle òptim (rad): {theta_sol}")