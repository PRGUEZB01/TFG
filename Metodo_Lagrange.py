import math
import numpy as np
import random

from scipy.optimize import differential_evolution, NonlinearConstraint, minimize
import time
# Define the non-convex objective function
def objective_function(var, x0, y0, xt, yt, n):
    
    x=np.concatenate(([x0], var[:n],[xt]), axis=None)
    y=np.concatenate(([y0], var[n:2*n],[yt]), axis=None)
    assert len(x)==len(y)==n+2, f"The length of x, y must be equal to {n+2}. Obtained len(x)={len(x)} and len(x)={len(y)}." 
    
    dists = [(x[i]-x[i+1])**2 + (y[i]-y[i+1])**2 for i in range(n+1)] 
    return sum(dists)


def build_contrains(alphas, betas, radii, n):

    constrains = []
    for i in range(1,n+1):

        def cxi(var, index = i):
            x=np.concatenate(([alphas[0]], var[:n],[alphas[-1]]), axis=None)
            u=var[2*n:]
            return -2*(x[index-1]-x[index]) + 2*(x[index]-x[index+1]) - 2*u[index-1]*(x[index]-alphas[index]) 

        def cyi(var, index = i):
            y=np.concatenate(([betas[0]], var[n:2*n], [betas[-1]]), axis=None)
            u=var[2*n:]
            return -2*(y[index-1]-y[index]) + 2*(y[index]-y[index+1]) - 2*u[index-1]*(y[index]-betas[index]) 

        def cui(var, index = i):
            x=np.concatenate(([alphas[0]], var[:n],[alphas[-1]]), axis=None)
            y=np.concatenate(([betas[0]], var[n:2*n], [betas[-1]]), axis=None)
            return (x[index] - alphas[index])**2 + (y[index] - betas[index])**2 - radii[index-1]**2  

        # constrains.append(NonlinearConstraint(cxi, -np.inf, np.inf)) 
        # constrains.append(NonlinearConstraint(cyi, -np.inf, np.inf)) 
        # constrains.append(NonlinearConstraint(cui, -np.inf, np.inf)) 

        constrains.append({'type': 'eq', 'fun': cxi})
        constrains.append({'type': 'eq', 'fun': cyi})
        constrains.append({'type': 'eq', 'fun': cui})

    return constrains


## Returns different extrema, it may be wrong. Revise
def build_contrains_manually(alphas, betas, radii, n):

    x0 = lambda x: -2*(x[0]-alphas[0]) + 2*(x[0]-x[1]) - 2*x[6]*(x[0]-alphas[1])
    x1 = lambda x: -2*(x[1]-x[0]) + 2*(x[1]-x[2]) - 2*x[7]*(x[1]-alphas[2])
    x2 = lambda x: -2*(x[2]-x[1]) + 2*(x[2]-alphas[4]) - 2*x[8]*(x[2]-alphas[3])

    y0 = lambda x: -2*(x[3]-betas[0]) + 2*(x[3]-x[4]) - 2*x[6]*(x[3]-betas[1])
    y1 = lambda x: -2*(x[4]-x[3]) + 2*(x[4]-x[5]) - 2*x[7]*(x[4]-betas[2])
    y2 = lambda x: -2*(x[5]-x[4]) + 2*(x[5]-betas[4]) - 2*x[8]*(x[5]-betas[3])

    u0 = lambda x: (x[0] - alphas[1])**2 + (x[3] - betas[1])**2 - radii[0]**2
    u1 = lambda x: (x[1] - alphas[2])**2 + (x[4] - betas[2])**2 - radii[1]**2
    u2 = lambda x: (x[2] - alphas[3])**2 + (x[5] - betas[3])**2 - radii[2]**2

    return [
        {'type': 'eq', 'fun': x0}, 
        {'type': 'eq', 'fun': x1}, 
        {'type': 'eq', 'fun': x2},
        {'type': 'eq', 'fun': y0}, 
        {'type': 'eq', 'fun': y1}, 
        {'type': 'eq', 'fun': y2},
        {'type': 'eq', 'fun': u0}, 
        {'type': 'eq', 'fun': u1}, 
        {'type': 'eq', 'fun': u2} 
    ]


if __name__ == "__main__":
    Inicio= time.time()
    
    """ESCENARIO 1. Trayecto con 5 circunferencias"""
    alpha = [1, 3, 5,   6, 8]
    beta  = [1, 2, 1.5, 2, 5]
    # alpha = [5, 7, 9]
    # beta  = [2.5, 6, 2.5]
    radii = [1]
    n = len(radii)
    
    var = [2.1599064461010484, 4.595656404471263, 5.450977182001876, 1.457558463336896, 2.4146071597986167, 2.835807361368275,0,10,10]
    """Escenario 2.1 Variables de inicio. Punto crítico máximo"""
    # var = [5.61, 6,10]
    """Escenario 2.2 Variables de inicio. Punto crítico mínimo"""
    # var = [6.29,5.3,0.1]
    options = {
        'maxiter': 1000,  # Aumenta el número máximo de iteraciones
        # 'gtol': 1e-16,     # Ajusta la tolerancia de la función
        'disp': True      # Muestra el proceso de optimización
    }
    result= minimize(objective_function, var, args=(alpha[0], beta[0], alpha[-1], beta[-1], n), method='SLSQP', constraints=build_contrains(alpha, beta, radii, n), options=options)
    
    Final=time.time()    
    print("Tiempo de cálculos: ", Final-Inicio)
    print('Optimal solution:', result.x)
    print('Objective value at optimal solution:', result.fun)

    # print("Testing")
    # print("c1", (result.x[0]-alpha[1])**2+(result.x[3]-beta[1])**2)
    # print("c2", (result.x[1]-alpha[2])**2+(result.x[4]-beta[2])**2)
    # print("c3", (result.x[2]-alpha[3])**2+(result.x[5]-beta[3])**2)
    # print("fo", objective_function(result.x, alpha[0], beta[0], alpha[-1], beta[-1], n))

    P0 = (alpha[0], beta[0])
    P1 = (result.x[0], result.x[3])
    P2 = (result.x[1], result.x[4])
    P3 = (result.x[2], result.x[5])
    P4 = (alpha[-1], beta[-1])
    
    print("SOLUCIÓN\n",(result.x[0], result.x[3]),"\n", (result.x[1], result.x[4]),"\n", (result.x[2], result.x[5]) )
    
    #Descomentar y comentar lineas 80-117 para ver soluciones 2º escenario
# =============================================================================
#     """ESCENARIO. 2. Geometría analítica"""
#     alpha = [5, 7, 9]
#     beta  = [2.5, 6, 2.5]
#     radii = [1]
#     n = len(radii)
#     """Escenario 2.1 Variables de inicio. Punto crítico máximo"""
#     # var = [5.61, 6,10]
#     """Escenario 2.2 Variables de inicio. Punto crítico mínimo"""
#     # var = [6.29,5.3,0.1]
#     options = {
#         'maxiter': 1000,  # Aumenta el número máximo de iteraciones
#         # 'gtol': 1e-16,     # Ajusta la tolerancia de la función
#         'disp': True      # Muestra el proceso de optimización
#     }
#     result= minimize(objective_function, var, args=(alpha[0], beta[0], alpha[-1], beta[-1], n), method='SLSQP', constraints=build_contrains(alpha, beta, radii, n), options=options)
#     
#     Final=time.time()    
#     print("Tiempo de cálculos: ", Final-Inicio)
#     print('Optimal solution:', result.x)
#     print('Objective value at optimal solution:', result.fun)
# 
#     r = [
#         [alpha[0],beta[0]],
#         [result.x[0], result.x[1]],
#         [alpha[-1],beta[-1]],
#     ]
#     pathlength = -2
#     for i in range(len(r)-1):
#         pathlength += math.sqrt((r[i][0]-r[i+1][0])**2 + (r[i][1]-r[i+1][1])**2)
#     
#     print("waypoints", r)
#     print("pathlength", pathlength)
# =============================================================================
