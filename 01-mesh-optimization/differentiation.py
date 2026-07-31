import numpy as np
import matplotlib.pyplot as plt

def derivadaNumerica(F,x):
    n = len(x)
    dF = np.zeros(n)
    eps = 4.64e-6
    A = np.array([x,0.5*np.ones(n)],dtype=float)
    ex = eps*np.max( A,axis = 0)
    
    temp = x.copy() +  ex ;
    ex   = temp.copy() - x.copy();
        
    for i in range(n):
        xi_ini = x[i].copy()
        
        x[i] = xi_ini + ex[i]
        F_plus = F(x)
        x[i] = xi_ini - ex[i]
        F_minus = F(x)
        dF[i] = (F_plus - F_minus)/(2*ex[i])
        
        x[i] = xi_ini
    return dF


def hessianaNumerica(F,x):
    n = len(x)
    H = np.zeros((n,n))
    eps = 1e-4
    A = np.array([x,0.5*np.ones(n)],dtype=float)
    ex = eps*np.max( A,axis = 0)
    
    temp = x.copy() +  ex ;
    ex   = temp.copy() - x.copy();
        
    Fx = F(x)
    
    for i in range(n):
        
        xi_ini = x[i].copy()
        
        x[i] = xi_ini + ex[i]
        F_plus = F(x)
        
        x[i] = xi_ini - ex[i]
        F_minus = F(x)
    
        H[i,i] = (F_plus - 2*Fx + F_minus)/(ex[i]**2)
        
        x[i] = xi_ini
        
        for j in range(i):
            xi_ini = x[i].copy()
            xj_ini = x[j].copy()
            
            x[i] = xi_ini - ex[i]
            x[j] = xj_ini - ex[j]
            Fx_mm = F(x)
            
            x[i] = xi_ini + ex[i]
            x[j] = xj_ini - ex[j]
            Fx_pm = F(x)
            
            x[i] = xi_ini - ex[i]
            x[j] = xj_ini + ex[j]
            Fx_mp = F(x)
            
            x[i] = xi_ini + ex[i]
            x[j] = xj_ini + ex[j]
            Fx_pp = F(x)
            
            H[i,j] = (Fx_pp - Fx_pm - Fx_mp + Fx_mm)/(4*ex[i]*ex[j])
            H[j,i] = H[i,j]
            
            x[i] = xi_ini
            x[j] = xj_ini
    return H

