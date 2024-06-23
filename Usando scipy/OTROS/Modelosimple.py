# -*- coding: utf-8 -*-
"""
Created on Mon May 20 17:54:54 2024

@author: Pablo Rodríguez Barra
"""
import numpy as np
from scipy.optimize import minimize

# =============================================================================
# #################------------Definición de funciones---------------#################
# def rectriccion(var,n, size, alpha, beta, r):
#     #Separo por variables el vector enviado. 
#     x=var[:size[0]]
#     y=var[size[0]:size[1]]
#     cons=var[size[1]:]
# 
#     print(x)
#     print(y)
#     print(cons)
# 
#   
#     res=np.zeros(n)
# 
#     res[0]= 2*(x[0]-alpha[0])-2*(x[1]-x[0])-2*cons[0]*(x[0]-alpha[1])
#     for i in range(1, len(x)-1):
#         res[i]= 2*(x[i]-x[i-1])-2*(x[i+1]-x[i])-2*cons[i]*(x[i]-alpha[i+1])
#     nx = len(x)
#     res[nx-1]= 2*(x[nx-1]-x[nx-2]) - 2*(x[nx-1]-alpha[nx+1]) - 2*cons[nx-1]*(x[nx-1]-alpha[nx])
#    
# 
#     res[nx]= 2*(y[0]-beta[0])-2*(y[1]-y[0])-2*cons[0]*(y[0]-beta[1])
#     for i in range(1, len(y)-1):
#         res[nx+i]= 2*(y[i]-y[i-1])-2*(y[i+1]-y[i])-2*cons[i]*(y[i]-beta[i+1])
#     ny = len(y)
#     res[nx+ny-1]= 2*(y[ny-1]-y[ny-2]) - 2*(y[ny-1]-beta[ny+1]) - 2*cons[ny-1]*(y[ny-1]-beta[ny])
# 
#     for i in range(len(cons)):
#        res[nx+ny+i] = -(x[i]-alpha[i+1])**2 - (y[i]-beta[i+1])**2 + r**2 
# 
#     return res
#     
# #Función objetivo
# def FO(var, size):
#     
#     x=var[:size[0]]
#     y=var[size[0]:size[1]]
#     x_m=x[:-1]
#     x_u=x[1:]
#     y_m=y[:-1]
#     y_u=y[1:]  
#     
#     obj= [0 for i in range(n-2)]
#     obj[:]=(x_u-x_m)**2+(y_u-y_m)**2
#     
#     resultado=0
#     for i in range(len(obj)): resultado= resultado+obj[i]
#   
#     resultado += (x[0]-1)**2 + (y[0]-1)**2
#     resultado += (8-x[-1])**2 + (5-y[-1])**2
# 
#     return resultado
# 
# #################------------Definición datos y variables del problema---------------#################
# 
# alpha=[1,3,5,6,8]
# beta=[1,2,1.5,2,5]
# r=1
# #Valores iniciales del bucle. 
# x=[3,5,6]
# y=[2,1.5,2]
# z=[1,2,3]
# 
# n=len(alpha)
# size= [len(x), len(x+y)]
# #Número de elementos que contendrá el vector solución resuelto
# num_elementos= 3*(n-2)
# #Vector con los valores inciales a resolver
# var= np.concatenate((x,y,z))
# print("var", var)
# 
# #Compruebo que todo funcione correctamente. 
# opciones = {
#     'maxiter': 7,  # Aumenta el número máximo de iteraciones
#     'ftol': 1e-9,     # Ajusta la tolerancia de la función
#     'disp': True      # Muestra el proceso de optimización
# }
# 
# #################------------Llamada a la funciones---------------#################
# 
# Restriccion={'type': 'eq', 'fun': rectriccion, 'args': (num_elementos, size, alpha, beta, r)}
# 
# for i in range (len(x)-1): Restriccion={'type': 'eq', 'fun': rectriccion, 'args': (num_elementos, size, alpha, beta, r)}
# print("VALOR DEL RESULTADO RESTRICCIÓN PARA PRIMERA ITERACIÓN", rectriccion(var, num_elementos, size, alpha, beta, r), "Debe coincidir con el valor del ejercicio resuelto justo debajo")
# 
# sol= minimize(FO, var, args=(size), method='CG', constraints=Restriccion, options=opciones)
# print("Variables optimizadas:", sol.x)
# # print("Valor de la función objetivo en el punto óptimo:", sol.fun)
# # print("¿La optimización tuvo éxito?:", sol.success)
# # print("Mensaje de la optimización:", sol.message)
# # print("Número de iteraciones:", sol.nit)
# 
# =============================================================================
def restriccion1  (xi, alpha, alpha_iu, x_iu, cons,p):
    
    res= 2*(xi-alpha)-2*(x_iu-xi)-2*cons*(xi-alpha_iu)
    return res

def restriccion2  (yi, beta, beta_iu ,y_iu, cons,p):
    res= 2*(yi-beta)-2*(y_iu-yi)-2*cons*(yi-beta_iu)
    return res

def restriccion3  (x_id,xi,x_iu,cons, alpha,p):
    res= 2*(xi-x_id)-2*(x_iu-xi)-2*cons*(xi-alpha)
    return res

def restriccion4  (y_id,  yi, y_iu, cons,betha,p):
    res= 2*(yi-y_id)-2*(y_iu-yi)-2*cons*(yi-betha)
    return res

def restriccion5  (xi, alpha, yi, beta, r,p):
    res = -(xi-alpha)**2 - (yi-beta)**2 + r**2
    return res

def restriccion6  (xi,x_iu, alpha,alpha_id, cons,p):
    res= 2*(xi-x_iu) - 2*(x_iu-alpha) - 2*cons*(xi-alpha_id)
    return res

def restriccion7 (yi, y_iu, beta,beta_id, cons,p):
    res= 2*(yi-y_iu) - 2*(y_iu-beta) - 2*cons*(yi-beta_id)
    return res

#Función objetivo
def FO(var, size):
    
    x=var[:size[0]]
    y=var[size[0]:size[1]]
    x_m=x[:-1]
    x_u=x[1:]
    y_m=y[:-1]
    y_u=y[1:]  
    
    obj= [0 for i in range(n-2)]
    obj[:]=(x_u-x_m)**2+(y_u-y_m)**2
    
    resultado=0
    for i in range(len(obj)): resultado= resultado+obj[i]
  
    resultado += (x[0]-1)**2 + (y[0]-1)**2
    resultado += (8-x[-1])**2 + (5-y[-1])**2

    return resultado



alpha=[1,3,5,6,8]
beta=[1,2,1.5,2,5]
r=1
#Valores iniciales del bucle. 
x=[3,5,6]
y=[2,1.5,2]
z=[1,2,3]

n=len(alpha)
size= [len(x), len(x+y)]
#Número de elementos que contendrá el vector solución resuelto
num_elementos= 3*(n-2)

var= np.concatenate((x,y,z))
print("var", var)

x=var[:size[0]]
y=var[size[0]:size[1]]
cons=var[size[1]:]

constraint=[]
Restriccion={'type': 'eq', 'fun': restriccion1, 'args': (x[0], alpha[0], alpha[1],x[1], cons[0])}
constraint.append(Restriccion)

Restriccion={'type': 'eq', 'fun': restriccion2, 'args': (y[0], beta[0],beta[1], y[1], cons[0])}
constraint.append(Restriccion)
for i in range(1,size[0]-1):
   Restriccion={'type': 'eq', 'fun': restriccion3, 'args': (x[i-1], x[i], x[i+1], cons[i], alpha[i])}
   constraint.append(Restriccion)
   Restriccion={'type': 'eq', 'fun': restriccion4, 'args': (y[i-1], y[i], y[i+1], cons[i], beta[i])}
   constraint.append(Restriccion)
for i in range (1,n-1):
    Restriccion={'type': 'eq', 'fun': restriccion5, 'args': (x[i-1], alpha[i], y[i-1], beta[i], r)}
    constraint.append(Restriccion)
   
Restriccion={'type': 'eq', 'fun': restriccion6, 'args': (x[-1], x[-2], alpha[-1],alpha[-2], cons[-1])}    
constraint.append(Restriccion)
Restriccion={'type': 'eq', 'fun': restriccion7, 'args': (y[-1], y[-2], beta[-1],beta[-2], cons[-1])}
constraint.append(Restriccion)
for i in range(len(constraint)):    
    print(constraint[i])


#Compruebo que todo funcione correctamente. 
opciones = {
    'maxiter': 1000,  # Aumenta el número máximo de iteraciones
    'ftol': 1e-9,     # Ajusta la tolerancia de la función
    'disp': True      # Muestra el proceso de optimización
}

print(var, size)
sol= minimize(FO, var, args=(size), method='SLSQP', constraints=constraint, options=opciones)
print("Variables optimizadas:", sol)


