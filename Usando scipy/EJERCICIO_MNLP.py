# -*- coding: utf-8 -*-
"""
Created on Mon May 20 17:54:54 2024

@author: Pablo Rodríguez Barra
"""
import numpy as np
from scipy.optimize import minimize

#################------------Definición de funciones---------------#################
def rectriccion(var,n, size, alpha, betha, r):
    #Separo por variables el vector enviado. 
    x=var[:size[0]]
    y=var[size[0]:size[1]]
    cons=var[size[1]:]
    
    #Declaro las variables Xi-1: x_d (down), Xi: x_m (medium), Xi+1: x_u (up)
    x_d=x[:-2]
    x_m=x[1:-1]
    x_u=x[2:]
    y_d=y[:-2]
    y_m=y[1:-1]
    y_u=y[2:]
    #Declaro las constantes lambda cm: No tiene en cuenta la primera y última circunferencia pues partimos de su centro
    cm=cons[1:-1]
    
    #Declaro alpha y betha para el bucle, no tiene en cuenta las dos primeras y dos últimas circunferencias. 
    a=alpha[2:-2]
    b=betha[2:-2]
    
    # m: Separa en el bucle loas rectricciones de las derivadas parciales de x, y, lambda. 
    m=size[0]
    res=np.zeros(n)

    #Restricciones
    res[0]= -2*(x[1]-x[0])-2*cons[0]*(x[0]-alpha[1])
    res[1]= -2*(y[1]-y[0])+2*cons[0]*(y[0]-betha[1])
    res[2]= -(x[0]-alpha[1])**2+(y[0]-betha[1])**2+r**2
    res[3:int((((m-2)*3)/3)+3)] =  2*(x_m-x_d)-2*(x_u-x_m)-2*cm*(x_m-a)
    res[int((((m-2)*3)/3)+3):int((2*(((m-2)*3)/3))+3)] = 2*(y_m-y_d)-2*(y_u-y_m)+2*cm*(y_m-b)
    res[int((2*(((m-2)*3)/3))+3):int((3*(((m-2)*3)/3))+3)] = -(x_m-a)**2+(y_m-b)**2+r**2
    res[-3]= 2*(x[-1]-x[-2])-2*(x[-1]-alpha[-2])
    res[-2]= 2*(y[-1]-y[-2])+2*(y[-1]-betha[-2])
    res[-1]= -(x[-1]-alpha[-2])**2+(y[-1]-betha[-2])**2+r**2
   
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
  
    return resultado

#################------------Definición datos y variables del problema---------------#################

alpha=[1,3,5,6,8]
betha=[1,2,1.5,2,5]
r=0.5
#Valores iniciales del bucle. 
x=[3,5,6]
y=[2.5,4.5,5.5]
z=[0.2,0.3,0.4]

n=len(alpha)
size= [len(x), len(x+y)]
#Número de elementos que contendrá el vector solución resuelto
num_elementos= 3*(n-2)
#Vector con los valores inciales a resolver
var= np.concatenate((x,y,z))

#Compruebo que todo funcione correctamente. 
opciones = {
    'maxiter': 1000,  # Aumenta el número máximo de iteraciones
    'ftol': 1e-9,     # Ajusta la tolerancia de la función
    'disp': True      # Muestra el proceso de optimización
}

#################------------Llamada a la funciones---------------#################

Restriccion={'type': 'eq', 'fun': rectriccion, 'args': (num_elementos, size, alpha, betha, r)}

print("VALOR DEL RESULTADO RESTRICCIÓN PARA PRIMERA ITERACIÓN", rectriccion(var, num_elementos, size, alpha, betha, r), "Debe coincidir con el valor del ejercicio resuelto justo debajo")

sol= minimize(FO, var, args=(size), method='Powell', constraints=Restriccion, options=opciones)
# print("Variables optimizadas:", sol.x)
# print("Valor de la función objetivo en el punto óptimo:", sol.fun)
# print("¿La optimización tuvo éxito?:", sol.success)
# print("Mensaje de la optimización:", sol.message)
# print("Número de iteraciones:", sol.nit)



#%%

#Resolución del modelo visto arriba para el punto inicial dado. 
import numpy as np

#################------------Declaro Variables y datos---------------#################

alpha=[1,3,5,6,8]
betha=[1,2,1.5,2,5]
r=0.5
x=[3,5,6]
y=[2.5,4.5,5.5]
z=[0.2,0.3,0.4]

n= [len(x), len(x+y)]
var= np.concatenate((x,y,z))

#################------------Restricciones---------------#################

x=var[:n[0]]
y=var[n[0]:n[1]]
cons=var[n[1]:]

x_d=x[:-2]
x_m=x[1:-1]
x_u=x[2:]
y_d=y[:-2]
y_m=y[1:-1]
y_u=y[2:]
cm=cons[1:-1]

a=alpha[2:-2]
b=betha[2:-2]
m=n[0]

res=np.zeros(len(var))


res[0]= -2*(x[1]-x[0])-2*cons[0]*(x[0]-alpha[1])
res[1]= -2*(y[1]-y[0])+2*cons[0]*(y[0]-betha[1])
res[2]= -(x[0]-alpha[1])**2+(y[0]-betha[1])**2+r**2
res[3:int((((m-2)*3)/3)+3)] =  2*(x_m-x_d)-2*(x_u-x_m)-2*cm*(x_m-a)
res[int((((m-2)*3)/3)+3):int((2*(((m-2)*3)/3))+3)] = 2*(y_m-y_d)-2*(y_u-y_m)+2*cm*(y_m-b)
res[int((2*(((m-2)*3)/3))+3):int((3*(((m-2)*3)/3))+3)] = -(x_m-a)**2+(y_m-b)**2+r**2
res[-3]= 2*(x[-1]-x[-2])-2*(x[-1]-alpha[-2])
res[-2]= 2*(y[-1]-y[-2])+2*(y[-1]-betha[-2])
res[-1]= -(x[-1]-alpha[-2])**2+(y[-1]-betha[-2])**2+r**2

print("El resultado de las restricciones", res)

#################------------FUNCIÓN OBJETIVO---------------#################
n=len(alpha)
x_m=x[:-1]
x_u=x[1:]
y_m=y[0:-1]
y_u=y[1:]  

#print(n-2)
obj= [0 for i in range(n-2)]
#print(obj, x_m)
obj[:]=(x_u-x_m)**2+(y_u-y_m)**2
#print(obj)
resultado=0
for i in range(len(obj)): resultado= resultado+obj[i]

print("El resultado de la función: ", resultado)




