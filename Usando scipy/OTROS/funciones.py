# -*- coding: utf-8 -*-
"""
Created on Sun May 12 20:41:45 2024

@author: Usuario
"""
import random
#Función multistart

#Función que determina una lista de puntos de inicio aleatorios (tuplas), con el rango de x1 y x2

def  generate_starting_points (number_of_points):
    
#El número de puntos que voy a generar, esta función devuelve una lista de puntos de incio

    starting_points=[]
    for points in range (number_of_points):
        starting_points.append((random.random(), random.random()))
        
    return starting_points