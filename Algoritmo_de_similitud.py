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
from copy import deepcopy

def Euclidian_dist(P1, P2):
    return math.sqrt((P1[0]-P2[0])**2 + (P1[1]-P2[1])**2)

def Calculate_speed_decisions(Route_1,Route_2):
    "compares the segment size of route 1 vs route 2"
    Short_length,Long_length,Same_lenght=[],[],[]
    for i in range (len(Route_1)-1):
        A=Euclidian_dist(Route_1[i], Route_1[i+1])
        B=Euclidian_dist(Route_2[i], Route_2[i+1])
        if A>B:
            Long_length.append(Route_1[i])
        elif A<B:
            Short_length.append(Route_1[i])
        else:
            Same_lenght.append(Route_1[i])
    return Long_length, Short_length, Same_lenght

def Calculate_free_space_diagram(Air, Land, Epsilon, Path):

    Points_number = len(Air)
    Free_space = np.zeros((Points_number, Points_number))
    Frechet_path=np.zeros((Points_number, Points_number))
    # Fill the free space diagram
    for i in range(Points_number):
        for j in range(Points_number):
            if Euclidian_dist(Air[i], Land[j]) <= Epsilon:
                Free_space[i, j] = 1
   
    for i in range(len(Path)):        
         Frechet_path[Path[i][0]][Path[i][1]]=1
    
    #Invierto el ordende las filas para la correcta representación gráfica
    Free_space_ = np.flipud(Free_space)
    Frechet_path_=np.flipud(Frechet_path)
    
    return Free_space_, Frechet_path_

def Plot_graphic_free_space_diagram(free_space,  path):
    # Dibujar el diagrama de espacio libre
    fig, ax = plt.subplots()
    ax.imshow(free_space, cmap='Greys', interpolation='none')
    ax.imshow(path, cmap='Greys', interpolation='none', alpha=0.5)
    # Etiquetas y títulos
    ax.set_xlabel('terrestre')
    ax.set_ylabel('aéreo')
    ax.set_title('Free Space Diagram')
    
    
    # Configurar las etiquetas de los ejes
    ax.set_xticks(np.arange(len(UGV)))
    ax.set_xticklabels([f'{j}' for j in range(len(UGV))])
    ax.set_yticks(np.arange(len(UAV)))
    ax.set_yticklabels([f'{(len(UAV)-1)-i}' for i in range(len(UAV))])
    if len(path)<10:
        ax.xaxis.set_major_locator(MultipleLocator(2)); ax.yaxis.set_major_locator(MultipleLocator(2))
        plt.xticks(rotation=270, ha='right')
    elif len(path)<25: 
        ax.xaxis.set_major_locator(MultipleLocator(3)); ax.yaxis.set_major_locator(MultipleLocator(3))
        plt.xticks(rotation=270, ha='right')
    elif len(path)<51: 
        ax.xaxis.set_major_locator(MultipleLocator(10)); ax.yaxis.set_major_locator(MultipleLocator(10))
        plt.xticks(rotation=270, ha='right')
    else: 
        ax.xaxis.set_major_locator(MultipleLocator(len(path) // 4)); ax.yaxis.set_major_locator(MultipleLocator(len(path) // 4))


    plt.show()


def Graphic_curves(air, land ,points_air,points_land):
    # Extraer las coordenadas x e y de P y Q
    air_x, air_y = zip(*air)
    land_x, land_y = zip(*land)

    additional_air_x, additional_air_y = zip(*points_air)
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
    plt.scatter(additional_air_x, additional_air_y, color='blue')
    plt.scatter(points_land_x, points_land_y, color='red')
    
    for i in range(len(air)):
        plt.annotate(f'BP{i}', (air_x[i], air_y[i]), textcoords="offset points", xytext=(0,10), ha='center')
    for i in range(len(land)):
        plt.annotate(f'RP{i}', (land_x[i], land_y[i]), textcoords="offset points", xytext=(0,10), ha='center')


    # Marcar los puntos individuales
    for i in range(len(air)):
        plt.annotate(f'BP{i}', (air_x[i], air_y[i]), textcoords="offset points", xytext=(0,10), ha='center')
    for i in range(len(land)):
        plt.annotate(f'RP{i}', (land_x[i], land_y[i]), textcoords="offset points", xytext=(0,10), ha='center')
    
    for (x1, y1), (x2, y2) in zip(points_air, points_land):
        plt.plot([x1, x2], [y1, y2], 'k--')

    plt.legend()
    
    # Mostrar la gráfica
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    
def Discret_route(r_a,r_t,N):
    UGV_points=[r_t[0]]
    UAV_points=[r_a[0]]
    aux1,aux2=r_a[0][0],r_a[0][1]
    aux11,aux22=r_t[0][0],r_t[0][1]
    for i in range(len(r_a)-1):
        #Sin valor absoluto, pues si importa la dirección que va a tomar. 
        c=[ (r_a[i+1][k]-r_a[i][k])/N for k in range(2)]
        cc=[(r_t[i+1][k]-r_t[i][k])/N for k in range(2)]
        for j in range(N):
            RPx,RPy=[],[]
            aux1+=c[0];       aux2+=c[1]
            RPx.append(aux1); RPy.append(aux2)
            BPx,BPy=[],[]
            aux11+=cc[0];     aux22+=cc[1]
            BPx.append(aux11);BPy.append(aux22)
            
            UAV_points.append(RPx+RPy)
            UGV_points.append(BPx+BPy)  
    return UAV_points, UGV_points

def Calculate_sequence(Points_number, DFM):
    Stop_points=[]
    i,j=Points_number,Points_number
    Distance=[DFM[i][j]]
    Path=[(i,j)]
    while i+j>0:
        if i==0 :
            j-=1

        elif j==0:
            i-=1
                     
        elif (DFM[i-1][j-1]<=DFM[i-1][j] and DFM[i-1][j-1]<=DFM[i][j-1]):
            j-=1; i-=1
           
        elif (DFM[i-1][j]<=DFM[i][j-1] and DFM[i-1][j]<= DFM[i-1][j-1]):
              Stop_points.append((i,j))
              i-=1         
        elif (DFM[i-1][j-1]<=DFM[i-1][j]):
            j-=1; i-=1
        else:
            Stop_points.append((i,j))
            i-=1  
            
           
        Distance.append(DFM[i][j])    
        Path.append((i,j))
    
    Distance.append(DFM[i-1][j-1])
    Path.reverse()
    Distance.reverse()
    
    return Path, Distance,Stop_points

def Distance_Frechet_Matriz(BP_points, RP_points, Circ_number):
   
    coord_UAV,coord_UGV=[],[]
    # for k in range(circ_number):
    DFM= np.zeros((len(BP_points), len(RP_points)))
    DFM[0][0]= Euclidian_dist(BP_points[0],RP_points[0])
    for i in range(1,len(BP_points)):
        DFM[i][0]= max(DFM[i-1][0],Euclidian_dist(BP_points[i], RP_points[0]))
    for j in range(1,len(RP_points)): 
        DFM[0][j]= max(DFM[0][j-1],Euclidian_dist(BP_points[0], RP_points[j]))
    for i in range(1,len(BP_points)):
        for j in range(1,len(RP_points)):
            DFM[i][j]=max(min(DFM[i-1][j], DFM[i][j-1], DFM[i-1][j-1]),Euclidian_dist(BP_points[i],RP_points[j]))
    m=np.array(DFM)

    pd.set_option('display.max_columns', None)  # Mostrar todas las columnas
    pd.set_option('display.width', 1000)        # Ajustar el ancho a 1000 caracteres
    pd.set_option('display.float_format', '{:.6f}'.format)  # Formato de impresión para floats


    df = pd.DataFrame(m)
    # print("OBTENIDA: \n",df)   
    path, Distance, Stop_points= Calculate_sequence(len(DFM[0])-1, DFM)     

    
    # print(UAV_points,"\n", UGV_points,"\n",path ,"\n")
    
    for i in range(len(path)):
        coord_UAV.append(BP_points[path[i][0]])
        coord_UGV.append(RP_points[path[i][1]])
    # for i in range (len(ptos_frenado)):
    #     frenado.append(UAV_points[path[i][0]])
    #     Frenado.append(UGV_points[path[i][1]])
    return path, Distance ,Stop_points, coord_UAV,coord_UGV

#INPUT DATA

""""Caso 2"""
# BP=[[1.0,1.0],[3.0,2.0],[5.0,1.5],[6.0,2.0],[8.0,5.0]]
# BP=[[3.7, 3.84], [4.17, 2.75], [3.71, 1.67], [2.67, 1.25], [1.64, 1.66], [1.17, 2.75], [1.63, 3.83], [2.67, 4.25], [3.8, 5]]
# RP=[[3.362854065925679, 3.005534530889891], 
#     [3.270548255682814, 2.781409547032251],
#     [3.084807466586902, 2.317405820304786],
#     [2.5915598315271073, 2.146575228282571],
#     [1.1367263868763278, 2.4061338152995377],
#     [1.2484401684728923, 3.646575228282571],
#     [2.2204531260914564, 4.5092386222004945],
#     [2.9029371405922686, 5.1193332436601615],
#     [2.9087587381325863, 5.125255790864059]]
# Circ_number= len(BP)
# SPN=3
# Extension=0.95

BP=[[2.1,3.1],[3.1,2.1],[3.13,4.51],[0.1,2.1],[4.1,0.1],
    [7.1,2.1],[8.25,3.11],[7.08,3.71],[3.11,5.77],[0.8,4.44]]
RP=[[2.995069705831446, 3.005924383059112], [3.005924383059112, 2.995069705831446], [2.822181871006898, 3.664276641292682], [0.9863269777109872, 2.2562833599002374], [4.0372191736302865, 0.9978076452338418], [7.256283359900237, 2.9863269777109873], [7.352192354766158, 3.047219173630287], [6.199667159339575, 3.5228794782640165], [3.015924383059112, 4.8749302941685535], [1.6803328406604252, 4.627120521735984]]
Circ_number= len(BP)
SPN=60
Extension=0.95
""""CASO DE PRUEBA FRECHET"""
# BP=[[1.0,1.0],[3.0,2.0],[5.0,1.5],[6.0,2.0],[8.0,5.0]]
# BP=[[1,2], [3,1.7], [5,2], [6,2]]
# RP=[[1,1], [2,1.3], [5,0.8], [6,0.8]]
# RP=[[3.362854065925679, 3.005534530889891], 
#     [3.270548255682814, 2.781409547032251],
#     [3.084807466586902, 2.317405820304786],
#     [2.5915598315271073, 2.146575228282571],
#     [1.1367263868763278, 2.4061338152995377],
#     [1.2484401684728923, 3.646575228282571],
#     [2.2204531260914564, 4.5092386222004945],
#     [2.9029371405922686, 5.1193332436601615],
#     [2.9087587381325863, 5.125255790864059]]
# Circ_number= len(BP)
# SPN=10
# Extension=1.4


"""Primero. Se discretiza la recta"""
UAV, UGV= Discret_route(BP,RP,SPN)
"""Segundo. Se aplica Frèchet obteniendo la ruta y la distancia, los puntos donde se frena y se acelera."""
Path, Distance, Stop_points, Coords_UAV, Coords_UGV= Distance_Frechet_Matriz(UAV, UGV, Circ_number)


""""Hay que calcular la matriz del diagrama de espacio libre en función de la matriz de distancias de Frèchet"""

Free_space, Frechet_path= Calculate_free_space_diagram(UAV, UGV, Extension, Path)

Aceleration_p, Deceleration_p, Constant_speed_p = Calculate_speed_decisions(Coords_UGV,Coords_UAV)
speed_decisions=[]
pi=deepcopy(Coords_UGV)

while len(pi) != 0:
    for i in range(len(Aceleration_p)):
        if pi[0]==Aceleration_p[i]:
            speed_decisions.append(str("A"))
    for i in range(len(Aceleration_p)):
        if pi[0]==Deceleration_p[i]:
            speed_decisions.append("D")
    else:
        speed_decisions.append("M")
    k=pi[0]
    pi.remove(k)
    
   
# print("El vector de decisión de velocidades: ", speed_decisions)
"""Calculo las variaciones de velocidad, para determinar el ritmo del vector"""
# Deceleration_points,Aceleration_points,Constant_speed_points=[],[],[]
# for i in range (len(coordenada_UGV)-1):
#     A=euclidian_dist(coordenada_UGV[i], coordenada_UGV[i+1])
#     B=euclidian_dist(coordenada_UAV[i], coordenada_UAV[i+1])
#     if A>B:
#         Aceleration_points.append(coordenada_UGV[i])
#     elif A<B:
#         Deceleration_points.append(coordenada_UGV[i])
#     else:
#         Constant_speed_points.append(coordenada_UGV[i])
print("El dron UGV recorre el segmento respecto del dron UAV:\nA una mayor velocidad en: ", Aceleration_p, 
      "\nA una menor velocidad en: ",Deceleration_p,"\n y a la misma velocidad en: ", Constant_speed_p)
#GRÁFICAS
Graphic_curves(BP, RP, Coords_UAV, Coords_UGV)
Plot_graphic_free_space_diagram(Free_space, Frechet_path)
# print("Distancia",Distance)

#Resultados:
# print(f"Las distancias entre puntos:\n {Distance}\nLos puntos de frenado: ")
for i in range(len(Stop_points)):print(Stop_points[i])
# print("\nLos puntos de aceleración: ")
# for i in range(len(ptos_aceleracion)):print(ptos_aceleracion[i])
# print(f"\nLa ruta seguida:\n{path}")
           
# m=np.array(D)

# pd.set_option('display.max_columns', None)  # Mostrar todas las columnas
# pd.set_option('display.width', 1000)        # Ajustar el ancho a 1000 caracteres
# pd.set_option('display.float_format', '{:.6f}'.format)  # Formato de impresión para floats


# df = pd.DataFrame(m)
# print("OBTENIDA: \n",df)   
   

