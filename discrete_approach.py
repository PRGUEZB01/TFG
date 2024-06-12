
import math
import matplotlib.pyplot as plt
import numpy as np

def get_circle_points(x, y, r, n):
    
    posicion=(2*math.pi)/n
    
    tetha=[0]
    for i in range(n):
        tetha.append(posicion)
        posicion=posicion+(2*math.pi)/n
    coordenadas=[]
    for i in range(n):
        px, py= [],[]
        for j in range(1):
            aux1= x + r*math.cos(tetha[i])
            px.append(aux1)
            aux2= y + r*math.sin(tetha[i])
            py.append(aux2)
        
        coordenadas.append(px+py)
        
   # print("longitud", len(coordenadas))
    return coordenadas


def plot_points(point_list, h, k, r):
    x , y =[], []
    additional_x , additional_y= [], []
    for l in range (len(point_list)):
    # Generamos los puntos de la circunferencia
        theta = np.linspace(0, 2 * math.pi, 100)  # 100 puntos en el rango [0, 2π]
        x.append(h[l] + r * np.cos(theta))
        y.append(k[l] + r * np.sin(theta))
    
        # Puntos en el borde de la primera circunferencia 
        additional_x.append([point_list[l][i][0] for i in range(len(point_list[l]))])
        additional_y.append([point_list[l][i][1] for i in range(len(point_list[l]))])
   
    
# =============================================================================
#    puntos= np.linspace(0, 2*math.pi, 100)
#    adicionales=[[1,3],[3,3]]
#    x=[]
#    y=[]
# 
#    for i in range(len(puntos)):
#        x.append(2+math.cos(puntos[i]))
#        y.append(3+math.sin(puntos[i]))
# 
# 
#    plt.figure(figsize=(6,6))
# 
#    plt.plot(x,y, label='circunferencia')
# 
#    plt.scatter(adicionales[0],adicionales[1], color='blue', label="Colores centro")
#     
# =============================================================================
    
    # Creamos la gráfica en conjunto
    plt.figure(figsize=(6, 6))
    
    for i in range(len(point_list)):
        aux=1
       # plt.plot(x[i], y[i])
    
        # Añadimos los puntos adicionales
        plt.scatter(additional_x[i], additional_y[i], color='blue', zorder=5, )
        if i<len(point_list)-1:
            plt.plot([h[i], h[i+1]], [k[i], k[i+1]], linestyle='--', color='black')
        
        aux=aux+1
    # Configuramos la visualización. #Pinta todos los centros, ya que pinta el vector, hace plot de todo el vector, tanto scatter, como plot. 
    plt.scatter(h, k, color='red')  # Marcamos el centro
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Circunferencia con puntos adicionales')
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.grid(True)
    plt.legend()
    plt.axis('equal')  # Para que los ejes x e y tengan la misma escala

    
    
    # Mostramos la gráfica
    plt.show()



def compute_forward_weights(previous_points, weights, forward_points):
    
  
    #Matriz que se devolverá con los pesos calculados de todos los puntos
    forward_weights=[]
    
    #Voy calculando los pesos por parejas de circunferencias. 
    
    for k in range(len(forward_points)):
       
        W_results=[]
        #Obtengo los vectores de pesos de los fordward_points con cada previous point
        
        for i in range(len(forward_points[k])):
            
            valores=[0 for i in range(len(previous_points[k]))]
            
            if k==0:
                for j in range(len(previous_points[k])):
                    
                    valores[j]=math.sqrt((previous_points[k][j][0]-forward_points[k][i][0])**2 + (previous_points[k][j][1]-forward_points[k][i][1])**2)+weights[j]
            #Para el resto de circunferencias se usan los mínimos de los pesos que hemos calculado. 
            else:    
               
                for j in range(len(previous_points[k])):
                    
                   valores[j]=math.sqrt((previous_points[k][j][0]-forward_points[k][i][0])**2 + (previous_points[k][j][1]-forward_points[k][i][1])**2)+forward_weights[k-1][j]
              
            
            #Contiene todos los pesos calculados de los puntos de la circunferencia k 
            
            W_results.append(valores)
                    
        
        #Vector que recoge los pesos mínimos de cada circunferencia
        forward_weights_circ=[0 for i in range(len(forward_points[k]))]
        if k==0:
            x_ini=[0 for i in range(len(forward_points[k]))]
            y_ini=[0 for i in range(len(forward_points[k]))]
        #Para cada forward point voy a buscar el peso con mínimo valor y lo voy a asignar al vector que devuelve los pesos mínimos
        for i in range(len(forward_points[k])):
            
            I= min(W_results[i])
            forward_weights_circ[i]=I
            
            #Para la primera circunferencia, la posición donde se encuentra el peso mínimo es la misma donde está el punto que me lo proporcionó 
            if k==0:
                
                posicion=W_results[i].index(I)
                
                x_ini[i]=(previous_points[0][0][posicion])
                y_ini[i]=(previous_points[0][1][posicion])
                print("ESTASSS", x_ini, y_ini)
            
        forward_weights.append(forward_weights_circ)
        
    return x_ini, y_ini, forward_weights
    
"""
    Compute forward weigths for the list of forward_points given the previous_points and its weights:
    
    The return values must be:
     
    1- a list of weights for the forward points with the same dimension: len(result_weights) == len(forward_points)
    2- a list of predecesors (prev) with the same dimension: len(prev) == len(forward_points)

    Where:
     
    result_weights[i] = M and prev[i] = j if M is the minimum value so that:

    M = d(previous_points[j], forward_points[i]) + weights[j]

    (d is the euclidian distance between two points)

    ---------------------------

    *** Example 1: 
    previous_points = [(0,0), (0.5, 0.5)]
    weights = [1, 10]
    forward_points = [(1, 0)]

    forward_weights, prev = compute_forward_weights(previous_points, weights, forward_points)

    # forward_weights: [2]
    # prev: [0]
    # because d(previous_points[0],forward_points[0])+weights[0]=2 
    # Notice that the previous is (0,0) despite the point (0.5, 0.5) is nearer to (1,0), this is because the weight of (0.5, 0.5) is too large (10) 
    
    ---------------------------
    
    *** Example 2: 
    previous_points = [(0,0), (0.5, 0.5), (0,1)]
    weights = [1, 10, 1]
    forward_points = [(1,0), (1,1)]

    forward_weights, prev = compute_forward_weights(previous_points, weights, forward_points)

    # forward_weights: [2, 2]
    # prev: [0, 2]
    # because d(previous_points[0],forward_points[0])+weights[0]=2 and d(previous_points[2],forward_points[1])+weights[2]=2 

    '''
    pass
"""

if __name__ == "__main__":

    #El numero de puntos es: 
        
    #
    x, y = 0, 0
    radio = 1
    n = 5

    points = get_circle_points(x, y, radio, n)
    # Hasta aqui paso 1


    x, y = 3, 3
    radio = 1
    n = 5
    points2 = get_circle_points(x, y, radio, n)

    x, y = 5,6
    radio = 1
    n = 5
    points3 = get_circle_points(x, y, radio, n)

    x, y = 7,9
    radio = 1
    n = 5
    points4 = get_circle_points(x, y, radio, n)

    x, y = 8,10
    radio = 1
    n = 5
    points5 = get_circle_points(x, y, radio, n)

    point_list = [points, points2, points3,points4,points5]
    alpha=(0,3,5,7,8)
    betha=(0,3,6,9,10)
    # plot_points(point_list, alpha, betha, radio)
    # Hasta aqui paso 2
    
    W_ini=[0,0,0,0,0]
    #print("Lista de puntos", point_list, "\n", W_ini, "\n forward_points \n", point_list[1:])
    x_ini, y_ini, forward_weights=compute_forward_weights(point_list,W_ini, point_list[1:])
    print("\n \n", x_ini, y_ini)
    
    for i in range(len(forward_weights)):
        print(f"\n Pesos circunferencia {i}: ", forward_weights[i])
    
    
# =============================================================================
# 
# 
# # Creamos la gráfica
# plt.figure(figsize=(8, 8))
# 
# 
#     # Graficamos la primera circunferencia
#     plt.plot(x1, y1, label='Circunferencia 1')
#     plt.scatter(additional_x1, additional_y1, color='blue', zorder=5, label='Puntos adicionales 1')
#     plt.scatter(h1, k1, color='red', zorder=5, label='Centro (3, 2)')
# 
#     # Graficamos la segunda circunferencia
#     plt.plot(x2, y2, label='Circunferencia 2')
#     plt.scatter(additional_x2, additional_y2, color='green', zorder=5, label='Puntos adicionales 2')
#     plt.scatter(h2, k2, color='orange', zorder=5, label='Centro (-2, -3)')
#     
#     
# 
# # Configuramos la visualización
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Dos Circunferencias con puntos adicionales')
# plt.axhline(0, color='gray', linewidth=0.5)
# plt.axvline(0, color='gray', linewidth=0.5)
# plt.grid(True)
# plt.legend()
# plt.axis('equal')  # Para que los ejes x e y tengan la misma escala
# 
# # Mostramos la gráfica
# plt.show()
# 
# 
# =============================================================================


