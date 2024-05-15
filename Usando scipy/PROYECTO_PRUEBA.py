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
def nuestrafuncion(start_points, alpha1,betha1, alpha3,betha3,r):
    [x,y,landa]=start_points
    res=start_points*0-1#la inicializa con esto
    res[0]=-2*(alpha1-x) +2*(x-alpha3)-2*landa*x
    res[1]=-2*(betha1-y) +2*(y-betha3)-2*landa*y
    res[2]=x**2+y**2-r**2
    return res

alpha1=-3;betha1=-2;alpha3=5;betha3=-1;r=0.3

initial_points=[1,1,1] 
data=(alpha1, betha1, alpha3, betha3, r)
sol=root(nuestrafuncion, initial_points, data, method='hybr')

if sol.success:print(sol.x)
else:          print(sol.message)

#NO ME DA LA SOLUCIÓN.Calculo puntos en borde de circunferencia 1 y 3

def calcular_bordes (puntos, alpha1, alpha3, betha1, betha3, x, y,r):
    [x1, y1, x3, y3]=puntos
    res=puntos*0-1
    res[0]=x1-alpha1**2+y1-betha1**2-r**2
    res[1]=-(y3-betha1)+((y-betha1)/(x-alpha1))*(x3-alpha1)
    res[2]=x3-alpha3**2+y3-betha3**2-r**2
    res[3]=-(y3-betha3)+((y-betha3)/(x-alpha3))*(x3-alpha3)
    return res

x=0.16641006;y=-0.24961509
puntos_iniciales=[0,0,0,0]
datos=(alpha1, alpha3,betha1, betha3,x,y ,r)
sol2=root(calcular_bordes, puntos_iniciales, datos, method='hybr')

if sol2.success: print(sol2.x)
else:print(sol2.message)

#%%

#MODELO MATEMÁTICO MILP PARA OBTENER LOS PUNTOS EN LAS CIRCUNFERENCIAS CON EL SOLVER CBC
#Y UTILIZANDO LOS OPERADORES DE LAGRANGE

#FALTAN PONER LAS RESTRICCIONES EN DERIVADAS PARCIALES. 
import mip
import numpy as np
from mip import Model, xsum, MINIMIZE, INF, CBC, INTEGER

#Variables
alpha=[1,3,5,6,8]
betha=[1,2,1.5,2,5]
r=0.4
n=len(alpha) #número de circunferencias

#Modelo
Modelo = Model("Prueba_lagrange_marsupial",sense=MINIMIZE,solver_name=CBC)
#Todas las circunferencias en el primer cuadrante. 
x=[Modelo.add_var(name="Coordenada x", var_type=INTEGER, lb=0, ub=INF) for i in range (n-2)]
y=[Modelo.add_var(name="Coordenada y", var_type=INTEGER, lb=0, ub=INF) for i in range (n-2)]
cons=[Modelo.add_var(name="Coordenada y", var_type=INTEGER, lb=-INF, ub=INF) for i in range (n-2)]

#Restricciones
# =============================================================================
# APLICAR DERIVADAS PARCIALES EN FUNCION DE X,Y Y CONS (LAMBDA) salen (3*(n-2)) restricciones
# =============================================================================
for i in range(num_trabajadores):
    Modelo += mip.xsum(x[i][j] for j in range(dias_año)) ==vacaciones[i]

#Objetivo
Modelo.objective= (alpha[0]-x[0])**2+(betha[0]-y[0])**2+(x[n-3]-alpha[n-1])**2+(y[n-3]-betha[n-1])**2+xsum((x[i+1]-x[i])**2 for i in range(n-2))
#Optimizamos el modelo y vemos si ha dado resultado. 
Modelo.optimize()
print(Modelo.status)


#%%

#EJEMPLO DE OPTIMIZACIÓN APLICANDO SCIPY EN UN RESTAURANTE DURANTE EL COVID 
import mip
from mip import Model, xsum, MAXIMIZE, INF, CBC, INTEGER

import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from funciones import generate_starting_points

# =============================================================================
#SISTEMA A OPTIMIZAR
#Min ⁡z=4x1^2-4x1^4+x1^(6/3)+x1x2-4x2^2+4x2^4
#Subject to:
#x1>=0.1
#x_2+=0.25
#x_1,x_2<=1
# =============================================================================

#DEFINIMOS NUESTRO SISTEMA CON FORMATO SCIPY
objective_function = lambda x: (4*x[0]**2)-(4*x[0]**4)+(x[0]**(6/3) + (x[0]*x[1]) - (4*x[1]**2) + (4*x[1]**4))

#se definen como ineq las restricciones de tipo >=
cons= [
       {'type': 'ineq', 'fun': lambda x: x[0]-0.1},
       {'type': 'ineq', 'fun': lambda x: x[1]-0.25}
]
#Las variables pueden tener cualquier valor entre el 0 y el 1
boundaries = [(0,1),(0,1)]

#Generar una lista de n puntos de inicio potenciales
starting_points= generate_starting_points(50)

#Vamos a aplicar el algoritmo
#Se llevan a cabo las iteraciones.
first_it =True
for point in starting_points:
    #Comienza el algoritmo
    res= minimize(objective_function,
                  [point[0], point[1]], 
                  method='SLSQP',
                  bounds= boundaries, 
                  constraints=cons)
    if first_it:
        better_solution_found= False
        best= res
    else: 
        if res.success and res.fun < best.fun:
            better_solution_found= True
            best=res
    #Si no ha encontrado una mejor sol. que la primera se queda como False
if best.success:
    print(f""" optimal solution found: 
          - proporción de asientos interiores disponibles: {round (best.x[0], 3)}
          -proporción de asientos exteriores disponibles: {round(best.x[1], 3)}
          -puntuación del índice de riesgo (que impuso el gobierno y se quería minimizar): {round(best.fun, 3)}""")
else: 
    print("No solution aviable")


#IMPRIMIR GRÁFICA
#Crea 2 vectores (arrays) equiespaciados entre 0 1 con 100 elementos
x1= np.linspace(0,1,100)
x2= np.linspace (0,1,100)
#Con meshgrid genera 2 matrices a partir de los 2 vectores, de forma que prueba todas las combinaciones posibles
x1, x2 = np.meshgrid(x1, x2)
#Z trata con las matrices, sustituye los valores de x1 y x2 (que representan todos los posibles casos) evaluando así todos los puntos en la fo. Es variable 2D. 
Z= objective_function ((x1, x2))

fig, ax = plt.subplots()
fig.set_size_inches(14.7, 8.27)

cs= ax.contour(x1,x2,Z,50,cmap='jet')
plt.clabel(cs, inline=1, fontsize=10) #pinta la FO
#Restricciones \geq es >= pintadas en la leyenda
plt.axvline(0.1, color='g', label=r'$x1 \geq 0.1$') #primera restricción
plt.axhline(0.25, color='r', label=r'$x2 \geq 0.25$')#segunda restricción
plt.legend (bbox_to_anchor=(1,1), loc=2, borderaxespad=0.1)

plt.xlabel(r'$x1$', fontsize=16)
plt.ylabel(r'$x2$', fontsize=16)
#Elimina los bordes izquierdo y derecho a los que se refiere como spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.show()

#%%
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from funciones import generate_starting_points

#DATOS
alpha1=-3;betha1=-2;alpha3=5;betha3=-1;r=0.3
#DEFINIMOS NUESTRO SISTEMA CON FORMATO SCIPY
objective_function = lambda x: (alpha1-x[0])**2 +(betha1-x[1])**2 + (x[0]-alpha3)**2+(x[1]-betha3)**2

#se definen como ineq las restricciones de tipo >=
cons= [
       {'type': 'eq', 'fun': lambda x: x[0]**2+x[1]**2-r**2},
       {'type': 'eq', 'fun': lambda x: -2*(alpha1-x[0]) +2*(x[0]-alpha3)-2*x[2]*x[0]},
       {'type': 'eq', 'fun': lambda x: -2*(betha1-x[1]) +2*(x[1]-betha3)-2*x[2]*x[1]}
]

#Las variables pueden tener cualquier valor entre el 0 y el 1
boundaries = [(0,100),(0,100), (0,100)]

res= minimize(objective_function,
               [point[0], point[1]], 
               method='SLSQP',
               bounds= boundaries, 
               constraints=cons)




