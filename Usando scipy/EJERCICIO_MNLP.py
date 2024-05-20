# -*- coding: utf-8 -*-
"""
Created on Mon May 20 17:54:54 2024

@author: Pablo Rodríguez Barra
"""

import numpy as np
from scipy.optimize import minimize

def FO (x,y,n):
    xm=x[1:]
    xm_p1=x[2:]
    ym=y[1:]
    ym_p1=y[2:]  
    
    obj= [0 for i in range(n-2)]
    obj[:]=(xm_p1-xm)**2+(ym_p1-ym)**2
    
    resultado=0
    for i in range(len(obj)): resultado= resultado+obj[i]
    
    return resultado

def restricciones (x,y,cons,numero_elementos, alpha, betha,r):
    #REPASO. ¿ESTÁN BIEN INICIALIZADAS?
    xm=x[1:]
    xm_m1=x[:-1] #contiene todos los elementos menos el último, funciona como un x(i-1) en la restricción
    xm_p1=x[2:]
    ym=y[1:]
    ym_m1=y[:-1]
    ym_p1=y[2:]
    cm=cons[1:-1]
    a=alpha[1:]#contiene todos los elementos menos el primero (funciona como un alpha (i+1) en la restricción)
    b=betha[1:]
    
   
    m=len(x)
    
    res=np.zeros_like(numero_elementos)
    res[0]= -2*(x[1]-x[0])+2*cons[0]*(x[0]-alpha[1])
    res[1]= -2*(y[1]-y[0])-2*cons[0]*(y[0]-betha[1])
    res[2]= -(x[0]-alpha(1))**2+(y[0]-betha[1])+r**2
    res[3:-((m-6)/3)] =  2*(xm-xm_m1)-2*(xm_p1-xm)-2*cm*(xm-a)
    res[(m-6)/3:-((2*(m-6)/3)+3)] = 2*(ym-ym_m1)-2*(ym_p1-ym)+2*cm*(ym-b)
    res[2*((m-6)/3)+3:-((3*(m-6)/3)+3)] = -(xm-a)**2+(ym-b)+r**2
    res[-3]= 2*(x[-1]-x[-2])+2*(x[-1]-alpha[-2])
    res[-2]= 2*(y[-1]-y[-2])+2*(y[-1]*betha[-2])
    res[-1]= -(x[-1]-alpha(-1))**2+(y[-1]-betha[-1])+r**2
    
    return res

alpha=[1,3,5,6,8, 10]
betha=[1,2,1.5,2,5, 10]
r=0.4
n=len(alpha)
x0=[2 for i in range(n-2)]
y0=[2 for i in range(n-2)]
cons0=[2 for i in range (n-2)]
numero_elementos=3*n

Restriccion={'type': 'eq', 'fun': restricciones}

for i in x0:
    for j in y0:
        for k in cons0:
            sol= minimize(FO, (i,j,k), args=(alpha, betha, r), method='SLSQP', constraints=Restriccion)