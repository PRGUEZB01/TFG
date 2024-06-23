# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 20:58:35 2024

@author: Pablo Rodríguez Barra

Input (blue_path, red_path)
output (frechet_path, aceleration_points, stops_points)
graphics outputs (free space diagram, frèchet distance between curves)

Se aplica una modificación del modelo de frèchet de T. Eiter and H Mannila
Computing Discrete Frèchet Distance, 1994. 
Donde:
    -Se descretizan los dos path por puntos
    -Se genera la matriz de distancias de frèchet que minimizan el máximo 
    -Se recorre la matriz inversamente para hallar el fréchet path, así como donde frena y acelera el dron terrestre 
    -Se obtienen las coordenadas de los puntos del frèchet path. 
    -Se grafica el diagrama de espacio libre: muestra (gris) puntos que cumplen distancia <= radio, (negro) frèchet path
    -Se grafican las curvas con la unión de los puntos seleccionados en el frèchet path

"""

import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np 
import pandas as pd

def euclidian_dist(P1, P2):
    return math.sqrt((P1[0]-P2[0])**2 + (P1[1]-P2[1])**2)


def free_space_diagram(curve1, curve2, epsilon, curve3):

    points = len(curve1)
    free_space = np.zeros((points, points))
    frechet_path=np.zeros((points, points))
    # Fill the free space diagram
    for i in range(points):
        for j in range(points):
            if euclidian_dist(curve1[i], curve2[j]) <= epsilon:
                free_space[i, j] = 1
   
    for i in range(len(curve3)):
         I=curve3[i][0]
         J=curve3[i][1]
         frechet_path[I][J]=1
    
    #Invierto el ordende las filas para la correcta representación gráfica
    free_space_ = np.flipud(free_space)
    frechet_path_=np.flipud(frechet_path)
    
    return free_space_, frechet_path_

def plot_graphic_free_space_diagram(free_space,  path):
    # Dibujar el diagrama de espacio libre
    fig, ax = plt.subplots()
    ax.imshow(free_space, cmap='Greys', interpolation='none')
    ax.imshow(path, cmap='Greys', interpolation='none', alpha=0.5)
    # Etiquetas y títulos
    ax.set_xlabel('terrestre')
    ax.set_ylabel('aéreo')
    ax.set_title('Free Space Diagram')
    
    
    # Configurar las etiquetas de los ejes
    ax.set_xticks(np.arange(len(human)))
    ax.set_xticklabels([f'{j}' for j in range(len(human))])
    ax.set_yticks(np.arange(len(dog)))
    ax.set_yticklabels([f'{(len(dog)-1)-i}' for i in range(len(dog))])
    if len(path)<10:
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.yaxis.set_major_locator(MultipleLocator(2))
        plt.xticks(rotation=270, ha='right')
    elif len(path)<25: 
        ax.xaxis.set_major_locator(MultipleLocator(3))
        ax.yaxis.set_major_locator(MultipleLocator(3))
        plt.xticks(rotation=270, ha='right')
    elif len(path)<51: 
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_major_locator(MultipleLocator(10))
        plt.xticks(rotation=270, ha='right')
    else: 
        ax.xaxis.set_major_locator(MultipleLocator(len(path) // 4))
        ax.yaxis.set_major_locator(MultipleLocator(len(path) // 4))


    plt.show()


def graphic_curves(air, land ,points_air,points_land):
    # Extraer las coordenadas x e y de P y Q
    air_x, air_y = zip(*air)
    land_x, land_y = zip(*land)

    additional_P_x, additional_P_y = zip(*points_air)
    points_land_x, points_land_y = zip(*points_land)

    # Configurar la figura y los ejes
    plt.figure(figsize=(8, 6))
    plt.title('Trayecto')
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    
    # Graficar P y Q
    plt.plot(air_x, air_y, marker='o', linestyle='-', color='blue', label='Dron aéreo')
    plt.plot(land_x, land_y, marker='o', linestyle='-', color='red', label='Dron terrestre')
    
    #Grafico los puntos
    plt.scatter(additional_P_x, additional_P_y, color='red')
    plt.scatter(points_land_x, points_land_y, color='blue')
    
    for i in range(len(air)):
        plt.annotate(f'A{i}', (air_x[i], air_y[i]), textcoords="offset points", xytext=(0,10), ha='center')
    for i in range(len(land)):
        plt.annotate(f'L{i}', (land_x[i], land_y[i]), textcoords="offset points", xytext=(0,10), ha='center')


    # Marcar los puntos individuales
    for i in range(len(air)):
        plt.annotate(f'A{i}', (air_x[i], air_y[i]), textcoords="offset points", xytext=(0,10), ha='center')
    for i in range(len(land)):
        plt.annotate(f'L{i}', (land_x[i], land_y[i]), textcoords="offset points", xytext=(0,10), ha='center')
    
    for (x1, y1), (x2, y2) in zip(points_air, points_land):
        plt.plot([x1, x2], [y1, y2], 'k--')

    plt.legend()
    
    # Mostrar la gráfica
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    

#INPUT DATA
r_a=[[1.0,1.0],[3.0,2.0],[5.0,1.5],[6.0,2.0],[8.0,5.0]]
r_t=[[1.923,1.357],[3.97, 2.16],[4.59, 2.405],[5.456, 2.827],[7.247, 4.357]]

n=6
human=[]
dog=[]
dog.append(r_a[0])
human.append(r_t[0])
aux11,aux22=r_t[0][0],r_t[0][1]
aux1,aux2=r_a[0][0],r_a[0][1]
for i in range(len(r_a)-1):
    #Sin valor absoluto, pues si importa la dirección que va a tomar. 
    c=[((r_a[i+1][k]-r_a[i][k])/n) for k in range(2)]
    cc=[(r_t[i+1][k]-r_t[i][k])/n for k in range(2)]
    for j in range(n):
        px,py=[],[]
        aux1+=c[0]
        aux2+=c[1]
        px.append(aux1)
        py.append(aux2)
        pxx,pyy=[],[]
        aux11+=cc[0]
        aux22+=cc[1]
        pxx.append(aux11)
        pyy.append(aux22)
        dog.append(px+py)
        human.append(pxx+pyy)   

# La matriz de distancias de Frechet
for k in range(len(r_a)):

    D=[[0 for i in range (len(dog))] for j in range (len(dog))]
    D[0][0]= euclidian_dist(dog[0],human[0])
    for i in range(1,len(dog)):
        D[i][0]= max(D[i-1][0],euclidian_dist(dog[i], human[0]))
        D[0][i]= max(D[0][i-1],euclidian_dist(dog[0], human[i]))
    for i in range(1,len(dog)):
        for j in range(1,len(dog)):
            
            D[i][j]=max(min(D[i-1][j], D[i][j-1], D[i-1][j-1]),euclidian_dist(dog[i],human[j]))
            
# m=np.array(D)

# pd.set_option('display.max_columns', None)  # Mostrar todas las columnas
# pd.set_option('display.width', 1000)        # Ajustar el ancho a 1000 caracteres
# pd.set_option('display.float_format', '{:.6f}'.format)  # Formato de impresión para floats


# df = pd.DataFrame(m)
# print("OBTENIDA: \n",df)   

"""Obtenemos el recorrido cumpliendo distancia de frechet"""      
ptos_frenado=[]
ptos_aceleracion=[]
Distancia=[D[i][j]]
path=[(i,j)]
i,j=len(D[0])-1, len(D[0])-1

while i+j>0:
   
    if i==0 :
        j-=1
        Distancia.append(D[i][j])
    elif j==0:
        i-=1
        Distancia.append(D[i][j])
    elif (D[i-1][j]<=D[i][j-1] and D[i][j-1]<= D[i-1][j-1]) or (D[i-1][j]<=D[i-1][j-1] and D[i-1][j-1]<=D[i][j-1]):
        ptos_frenado.append((i,j))
        i-=1
        Distancia.append(D[i][j])
        
    elif (D[i-1][j-1]<=D[i-1][j] and D[i-1][j]<=D[i][j-1]) or (D[i-1][j-1]<=D[i][j-1] and D[i][j-1]<=D[i-1][j]):
        j-=1; i-=1
        Distancia.append(D[i][j])
        
    else:
        ptos_aceleracion.append((i,j))
        j-=1
        Distancia.append(D[i][j])
        
    path.append((i,j))

Distancia.append(D[i-1][j-1])
path.reverse()
Distancia.reverse()

coordenada_dog,coordenada_human=[],[]
for i in range(len(path)):
    coordenada_dog.append(human[path[i][0]])
    coordenada_human.append(dog[path[i][1]])

#GRÁFICAS
graphic_curves(r_a, r_t, coordenada_dog, coordenada_human)
free_space, frechet_path=free_space_diagram(dog, human, 1, path)
plot_graphic_free_space_diagram(free_space, frechet_path)

#Resultados:
print(f"Las distancias entre puntos:\n {Distancia}\nLos puntos de frenado: ")
for i in range(len(ptos_frenado)):print(ptos_frenado[i])
print("\nLos puntos de aceleración: ")
for i in range(len(ptos_aceleracion)):print(ptos_aceleracion[i])
print(f"\nLa ruta seguida:\n{path}")