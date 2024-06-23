# -*- coding: utf-8 -*-
"""
Created on Sun May 26 19:15:42 2024

@author: Usuario
"""
x=[2,6,2,6]
y=[3,1,2,2]
sol=((y[1]-y[0])*(x[2]-x[1])-(x[1]-x[0])*(y[2]*y[1]))*((y[1]-y[0])*(x[3]-x[1])-(x[1]-x[0])*(y[3]-y[1]))

l=2/0.5


x=[3,5,9,6,1]
y=[4,2,4,7,1]
#Estas bethas y alphas son las de los segmentos con placas. 
#Hay dos opciones:
    #Dar un vector con los segmentos que tienen placas,
    #Separar del vector con todos los segmentos aquellos que tienen placas. 
    
#Tambien existe, en la resolución del bucle está puesto que vaya de 2 en 2, pero claro si se da que no es una placa si y otra no
#entonces en el caso de que sea aleatorio, donde puedan existir 3 placas seguidas, o 3 segmentos azules sin placas, entonces 
#se le daría un nuevo vector que tenga cada cuando hay placas, y que vaya variando, que ahora salte de 2 en 2, que una lo salte en 4
#y que vaya leyendo los elementos de ese vector. 
    
#En este caso doy un vector con aquellos segmentos con placas. 
#Mientras el vector de X, Y se corresponde a todos los segmentos que componen el terrestre ya que pueden cruzarse todos. 
alpha=[3,5,9,7]
betha=[1,3,5,7]

#PARA N/2 SEGMENTOS
#Para cada segmento rojo, veo si se cruza con cualquier azul. 
print("Si se cumple que para un mismo segmento i da como resultado que sus dos ecuaciones devuelven un valor 0 o negativo entonces se cruzan, en caso de que una de ellas resulte >=0 entonces no se intersectan, 0 significa que se encuentran en el mismo punto x por ejemplo. ")
for i in range(len(x)-1):
    print(f"\nPara el segmento {i+1} del red path: \n")
    for j in range(0,len(alpha)-1,2):
        print(j)
        print(f"Con el segmento {j+1} del blue path:")
        print(((betha[j+1]-betha[j])*(x[i]-alpha[j+1])-(alpha[j+1]-alpha[j])*(y[i]-betha[j+1]))*((betha[j+1]-betha[j])*(x[i+1]-alpha[j+1])-(alpha[j+1]-alpha[j])*(y[i+1]-betha[j+1])))
        print(((y[i+1]-y[i])*(alpha[j]-x[i+1])-(x[i+1]-x[i])*(betha[j]-betha[j+1]))*((y[i+1]-y[i])*(alpha[j+1]-x[i+1])-(x[i+1]-x[i])*(betha[j+1]-y[i+1])))
        
        print(j)


#%%
def rectriccion(var,n, size, alpha_con_placa, betha_con_placa, n_cada_cuanto_hay_placa):
    
    x=var[:size[0]]
    y=var[size[0]:size[1]]
    
    x_m=x[:-1]
    x_u=x[1:]
    y_m=y[:-1]
    y_u=y[1:]
    a_m=alpha_con_placa[:-1]
    a_u=alpha_con_placa[1:]
    b_m=betha_con_placa[:-1]
    b_u=betha_con_placa[1:]
    
    res= [[0 for i in range(len(alpha_con_placa)-1)]for j in range (len(x)-1)]
    print("Si se cumple que para un mismo segmento i da como resultado que sus dos ecuaciones devuelven un valor 0 o negativo entonces se cruzan, en caso de que una de ellas resulte >=0 entonces no se intersectan, 0 significa que se encuentran en el mismo punto x por ejemplo. ")
    for i in range(len(x)-1):
        print(f"\nPara el segmento {i+1} del red path: \n")
        for j in range(0,len(alpha_con_placa)-1,2):
            print(j)
            print(f"Con el segmento {j+1} del blue path:")
            res[]=(((b_u-b_m)*(x_m-a_u)-(a_u-a_m)*(y_m-b_u))*((b_u-b_m)*(x_u-a_u)-(a_u-a_m)*(y_u-b_u)))
            res[]=(((y_u-y_m)*(a_m-x_u)-(x_u-x_m)*(b_m-b_u))*((y_u-y_m)*(a_u-x_u)-(x_u-x_m)*(b_u-y_u)))
    for i in range
            print(j)
    
    
    