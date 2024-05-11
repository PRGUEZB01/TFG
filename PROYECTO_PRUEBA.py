# -*- coding: utf-8 -*-
"""
Created on Sun May 12 00:39:59 2024

@author: Usuario
"""


import mip
import numpy as np
from mip import Model, xsum, MAXIMIZE, INF, CBC, INTEGER
rg=np.random.default_rng(2)

print("CÓDIGO REALIZADO EN CLASE DE FACTOR HUMANO DE LA UNIVERSIDAD DE SEVILLA EN EL CURSO 22/23")

#Parámetros
S= 52#Semanas totales
I=100#trabajadores
c_i=rg.integers(0,2,size=I)
J=4
a_j=np.array([8,9,12,5])#cuatro configuraciones
p_ij=np.zeros((I,J))
#numero max de horas totales por trabajador a t completo
ra_max=1920
#Coste de infradotación
c_u=rg.integers(300,4001)
#Coste unitario hora sobredotación
c_o=rg.integers(200,301)
#Demanda del periodo
d_s=rg.integers(10000,15001,size=S)
Modelo = mip.Model("PNEC.4",sense=mip.MINIMIZE,solver_name=mip.CBC)

#Variables
x=[[[Modelo.add_var(name="Trabajador asignado en la semana a configuración", var_type=mip.BINARY) for s in range(S)] for j in range(J)] for i in range(I)]
U=[Modelo.add_var(name="Horas no cubiertas", var_type=mip.INTEGER, lb=0, ub=mip.INF) for s in range (S)]
O=[Modelo.add_var(name="Horas en exceso", var_type=mip.INTEGER, lb=0, ub=mip.INF) for s in range (S)]
WL=[Modelo.add_var(name="Carga del trabajador anual", var_type=mip.INTEGER, lb=0, ub=mip.INF) for i in range (I)]
MWL=[Modelo.add_var(name="Carga del trabajador anual respecto de la media trabajadores", var_type=mip.INTEGER, lb=0, ub=mip.INF) for i in range (I)]

#FO
Modelo.objective= mip.xsum(c_u*U[s] + c_o*O[s] for s in range (S))+ mip.xsum(MWL[i] for i in range (I))

#Restricciones
#1.1 y 1.2
for i in range(I):
    if (c_i[i]==1):
        for s in range (S):
            Modelo += xsum(x[i][j][s] for j in range (J))==1
            
for i in range(I):
    if (c_i[i]==0):#ponerlo así por si hay un error humano o leemos otra cosa y no tomase valor 0 ni 1
        for s in range (S):
            Modelo += xsum(x[i][j][s] for j in range (J))<=1
#4
for s in range(S):
    Modelo+=xsum(a_j[j]*x[i][j][s] for j in range (J) for i in range (I)) + U[s] - O[s] == d_s[s] 
#5
for i in range(I):
    for j in range(J):
        Modelo += xsum(x[i][j][s] for s in range (S)) >= p_ij[i,j]*xsum(x[i][j_prima][s] for j_prima in range(J)for s in range (S) )#p_ij[i,j] así lo defino como matriz, aun que se puede llamar como [i][j], pero esta última es más ineficiente, pues haces dos llamadas

#6
for i in range (I):
    for s in range (s, S-1):
        if c_i[i]==0:
            Modelo+= xsum(x[i][j][s] for j in range (J))<= xsum(x[i][j][s+1] for j in range (J))

#7
for i in range (I):
    Modelo+= xsum(a_j[j]*x[i][j][s] for j in range (J) for s in range (S))<= (ra_max/S)*xsum(x[i][j][s] for j in range (J) for s in range (S))
                                                                  
#8 Cuidado es un sumatorio dentro de un sumatorio.
for i in range (I):
    Modelo+= WL[i]== xsum((a_j[j]/S) *xsum(x[i][j][s] for s in range (S))for j in range (J))
#9
for i in range (I):
    Modelo+= MWL[i]>= WL[i]- ((xsum(WL[i_prima] for i_prima in range (I) ))/I)
#10
for i in range (I):
    Modelo+= MWL[i]>=  ((xsum(WL[i_prima] for i_prima in range (I) ))/I)-WL[i]
    
#11 que es la 5

for i in range (I):
    if (c_i[i]==1):
       Modelo += WL[i] == xsum(a_j[j]*x[i][j][s] for j in range (J) for s in range (S))
    elif (c_i[i]==1):
        Modelo += WL[i] == xsum(8*(1-x[i][j][s] for j in range (J) for s in range (S))) + xsum(a_j[j]*x[i][j][s] for j in range(J) for s in range (S))

Modelo.optimize()
print(Modelo.status)