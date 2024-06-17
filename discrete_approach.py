
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



def euclidian_dist(P1, P2):
    return math.sqrt((P1[0]-P2[0])**2 + (P1[1]-P2[1])**2)


def compute_forward_weights(previous_points, weights, forward_points):
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
 
    #Matriz que se devolverá con los pesos calculados de todos los puntos
    forward_weights=[]
    indices=[]
    
    #Voy calculando los pesos por parejas de circunferencias. 
    for i in range(len(forward_points)):

        weightsij = [euclidian_dist(forward_points[i], pj) + wj for pj, wj in zip(previous_points, weights)] 
        minij = min(weightsij)
        indexij = weightsij.index(minij)

        forward_weights.append(minij)
        indices.append(indexij)

    return forward_weights, indices 

    #     if k==0:
    #         for j in range(len(previous_points[k])):

    #             valores[j] = euclidian_dist(previous_points[k][j], forward_points[k][i]) + weights[j]
    #             # valores[j]=math.sqrt((previous_points[k][j][0]-forward_points[k][i][0])**2 + (previous_points[k][j][1]-forward_points[k][i][1])**2)+weights[j]
    #     #Para el resto de circunferencias se usan los mínimos de los pesos que hemos calculado. 
    #     else:    
            
    #         for j in range(len(previous_points[k])):
                
    #             valores[j]=euclidian_dist(previous_points[k][j], forward_points[k][i])+forward_weights[k-1][j]
            
        
    #     #Contiene todos los pesos calculados de los puntos de la circunferencia k 
        
    #     W_results.append(valores)
                    
        
    #     #Vector que recoge los pesos mínimos de cada circunferencia
    #     forward_weights_circ=[0 for i in range(len(forward_points[k]))]
    #     if k==0:
    #         x_ini=[0 for i in range(len(forward_points[k]))]
    #         y_ini=[0 for i in range(len(forward_points[k]))]
    #     #Para cada forward point voy a buscar el peso con mínimo valor y lo voy a asignar al vector que devuelve los pesos mínimos
    #     for i in range(len(forward_points[k])):
            
    #         I= min(W_results[i])
    #         forward_weights_circ[i]=I
            
    #         #Para la primera circunferencia, la posición donde se encuentra el peso mínimo es la misma donde está el punto que me lo proporcionó 
    #         if k==0:
                
    #             posicion=W_results[i].index(I)
                
    #             x_ini[i]=(previous_points[0][0][posicion])
    #             y_ini[i]=(previous_points[0][1][posicion])
    #             print("ESTASSS", x_ini, y_ini)
            
    #     forward_weights.append(forward_weights_circ)
        
    # return x_ini, y_ini, forward_weights



# def compute_discrete_opt_example_2points(circs):
#     """
#     circs es una lista de circunferencias -> lista de listas de puntos (pq cada circunferencia es una lista de puntos)
#     """

#     new_weights, indices = compute_forward_weights(circs[0], [0]*len(circs[0]), circs[1])
    
#     min_weight = min(new_weights)
#     P2_index = circs[1].index(min_weight)
#     P1_index = indices[P2_index]

#     return [circs[0][P1_index], circs[1][P2_index]]




def compute_discrete_opt(circs):
    """
    circs es una lista de circunferencias -> lista de listas de puntos (pq cada circunferencia es una lista de puntos)
    """
    return compute_discrete_opt_rec(circs, [0]*len(circs[0]))[0]


def compute_discrete_opt_rec(circs, prev_weights):
    """
    circs es una lista de circunferencias -> lista de listas de puntos (pq cada circunferencia es una lista de puntos)
    """
    if len(circs)==2:
        new_weights, indices = compute_forward_weights(circs[0], prev_weights, circs[1])
    
        min_weight = min(new_weights)
        P2_index = circs[1].index(min_weight)
        P1_index = indices[P2_index]

        return [circs[0][P1_index], circs[1][P2_index]], P1_index


    new_weights, indices = compute_forward_weights(circs[0], prev_weights, circs[1])
    path, index = compute_discrete_opt_rec(circs[1:], new_weights)
    p1_index = indices[index]

    return [circs[0][p1_index]] + path, p1_index


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
    
#%%

##########---------ALORITMO DISCRETO COMPELTO POR RECURSIVIDAD------------#############3
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
            aux2= y + r*math.sin(tetha[i])
            px.append(aux1); py.append(aux2)
            
        coordenadas.append(px+py)
    return coordenadas



def plot_points(point_list, h, k, r, sol_x, sol_y):
    
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
   
    
    # Creamos la gráfica en conjunto
    plt.figure(figsize=(6, 6))
    
    for i in range(len(point_list)):
        aux=1
       # plt.plot(x[i], y[i])
    
        # Añadimos los puntos adicionales
        plt.scatter(additional_x[i], additional_y[i], color='black',s=1, zorder=5, )
        #Pintamos las lineas
        if i<len(point_list)-1:
            plt.plot([h[i], h[i+1]], [k[i], k[i+1]], linestyle='--', color='blue')
            plt.plot([sol_x[i], sol_x[i+1]],[sol_y[i], sol_y[i+1]], linestyle='--', color='red' )
        aux=aux+1
        
    # Configuramos la visualización. #Pinta todos los centros, ya que pinta el vector, hace plot de todo el vector, tanto scatter, como plot. 
    plt.scatter(h, k, color='red')  
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Circunferencia con puntos adicionales')
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.grid(True)
    plt.legend()
    plt.axis('equal')  
    plt.show()
    
def euclidian_dist(P1, P2):
   
    return math.sqrt((P1[0]-P2[0])**2 + (P1[1]-P2[1])**2)


def compute_forward_weights(previous_points, weights, forward_points):
   
    forward_weights=[]
    indices=[]
    coord=[]
    
    #Voy calculando los pesos por parejas de circunferencias. 
    for i in range(len(forward_points)):
        
        weightsij = [euclidian_dist(forward_points[i], pj) + wj for pj, wj in zip(previous_points, weights)] 
        minij = min(weightsij)
        indexij = weightsij.index(minij)
        forward_weights.append(minij)
        #Posición donde está el peso mínimo que coincide con el punto que lo proporciona. 
        indices.append(indexij)
        coord.append(previous_points[indexij])

    return forward_weights, indices, coord

def compute_discrete_opt(weights,indices, coord_points, pos_interes, ruta):
    
    if i==0:
      
        k=min(weights)
        pos=weights.index(k)
        I=indices[pos]
        coordenadas=coord_points[I]
        ruta.append(pos_interes)
        
    else:
        I=0
        I=indices[I]
        coordenadas=coord_points[I]
        
    ruta.append(coordenadas)
    
    return I, ruta


def compute_discrete_opt_rec(circs, prev_weights, i):
    """
    circs es una lista de circunferencias -> lista de listas de puntos (pq cada circunferencia es una lista de puntos)
    """
    
    new_weights, indices, coordenadas = compute_forward_weights(circs[0], prev_weights, circs[1])
    P2_coord=None
    #coge el coste del punto que ha dado el mínimo a la segunda circunferencia. 
    if i==(len(alpha)-2):
        min_weight = min(new_weights) 
        pos=new_weights.index(min_weight)
        P2_coord = circs[1][pos]
        
    return coordenadas, indices, new_weights, P2_coord

if __name__ == "__main__":


    alpha = [1, 3, 5,   6, 8]
    beta  = [1, 2, 1.5, 2, 5]
    radio = 1
    n=800
    
    #######------ GENERO N PUNTOS POR CIRCUNFERENCIA -----######
    points=[0]*len(alpha)
    for i in range(len(alpha)):
        x,y=alpha[i], beta[i]
        points[i]=get_circle_points(x, y, radio, n)
   
    
    pesos, indices=[[0]*(n+1)]*len(alpha),[[0]*(n+1)]*len (alpha)
    pesos=[0]*(n)
    matriz_coordenadas=[]
    matriz_indices=[]
    
    #######------ GENERO DE FORMA DINÁMICA LOS PESOS POR PAREJAS DE CIRCUNFERENCIAS -----######
    for i in range(len(alpha)-1):
        # print(i)
        coordenadas, indices, list_weights, pos=compute_discrete_opt_rec([points[i]]+[points[i+1]], pesos,i)
        matriz_coordenadas.append(coordenadas)
        matriz_indices.append(indices)
        
        #Me hará falta para ir calculando el camino hacia adelante y luego para la primera iteración del ciclo de vuelta
        pesos = [list_weights[i] + pesos[i] for i in range(n)]
        # print("0 AAAA",pesos)
        # print("1AAAAA",matriz_coordenadas)
        # print("2AAAAA",matriz_indices)
    
    #######------ OBTENGO EL CAMÍNO DE MÍNIMA LONGITUD PARTIENDO DEL PUNTO CON MENOR PESO  -----######
    
    red_path=[]
   
    for i in range(len(alpha)-1):
        
        pos, red_path= compute_discrete_opt(pesos, matriz_indices[(len(alpha)-2)-i], matriz_coordenadas[(len(alpha)-2)-i], pos, red_path)
    

    red_path.reverse()
    print("EL PATH ÓPTIMO",red_path )
    print("SOLUCIÓN\n", red_path[1],"\n", red_path[2], "\n", red_path[3])
    coord_path= []
   
    path_x, path_y= [0]*len(red_path), [0]*len(red_path)
    for i in range(len(red_path)):
        path_x[i]=red_path[i][0]
        path_y[i]=red_path[i][1]
    #Gráficamos el resultado
    plot_points(points, alpha, beta, radio, path_x, path_y)
    
