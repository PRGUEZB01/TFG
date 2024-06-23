# -*- coding: utf-8 -*-
"""
Created on Mon May 13 17:43:26 2024

@author: Pablo Rodríguez Barra
"""

#Me va a proporcionar un punto que estoy obligado a hallar, puesto que tengo el mismo número
#de ecuaciones y variables. En el momento que no esté tan claro este valor, cuando pueda oscilar
#el valor de las variables, entonces tendremos que optimizarlo y buscar el óptimo. 
#Si se parte siempre del centro de la primera circunferencia al centro de la última entonces el 
#problema es más sencillo de resolver. 

#EN el modelo de abajo no estoy añadiendo la función objetivo
#LEARNING ROOT
from scipy.optimize import root
import numpy as np
import math

#Busca el óptimo para el caso de 3 circunferencias colineales con centro de la segunda en el origen de coordenadas
def opt_middle_point(point, alpha1, betha1, alpha3, betha3,r):
    x,y,l=point
    return [
        -2*(alpha1-x) + 2*(x-alpha3)-2*l*x,
        -2*(betha1-y) + 2*(y-betha3)-2*l*y,
        x**2+y**2-r**2
    ]


alpha1=-3;betha1=-2;alpha3=5;betha3=-1;r=1

initial_points=[0,-r,1] 
data=(alpha1, betha1, alpha3, betha3, r)
sol=root(opt_middle_point, initial_points, data, method='hybr')

if sol.success:
    print(sol.x)
else:          
    print(sol.message)
