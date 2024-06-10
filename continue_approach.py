import numpy as np
import random
from scipy.optimize import differential_evolution, NonlinearConstraint, minimize

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

if __name__ == "__main__":
    
    # alpha and beta are the centers of circles
    alpha = [1, 3, 5,   6, 8]
    beta  = [1, 2, 1.5, 2, 5]
    radii = [1, 1, 1]
    n = len(radii)

    var = np.random.rand(3*n)
    options = {
        'maxiter': 1000,  # Aumenta el número máximo de iteraciones
        # 'gtol': 1e-16,     # Ajusta la tolerancia de la función
        'disp': True      # Muestra el proceso de optimización
    }
    result= minimize(objective_function, var, args=(alpha[0], beta[0], alpha[-1], beta[-1], n), method='trust-constr', constraints=build_contrains(alpha, beta, radii, n), options=options)

    print('Optimal solution:', result.x)
    print('Objective value at optimal solution:', result.fun)

    print("Testing:")
    print("c1", (result.x[0]-alpha[1])**2+(result.x[3]-beta[1])**2)
    print("c2", (result.x[1]-alpha[2])**2+(result.x[4]-beta[2])**2)
    print("c3", (result.x[2]-alpha[3])**2+(result.x[5]-beta[3])**2)
    print("fo", objective_function(result.x,alpha[0], beta[0], alpha[-1], beta[-1], n))