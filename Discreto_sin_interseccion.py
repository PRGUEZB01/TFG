import math
import matplotlib.pyplot as plt
import numpy as np
import time 

def Get_circle_points(x, y, r, Points_number):
      
    Pi=[]
    for i in range(Points_number):
        Tetha=i*((2*math.pi)/Points_number)
        px= [x + r*math.cos(Tetha)]
        py= [y + r*math.sin(Tetha)]
        Pi.append(px+py)
      
    return Pi


def plot_solution(point_list, h, k, r, path1=None, path2=None):
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
        plt.scatter(additional_x[i], additional_y[i],s=1, color='blue', zorder=2)
        if i<len(point_list)-1:
            plt.plot([h[i], h[i+1]], [k[i], k[i+1]], linestyle='--', color='black')
        
        aux=aux+1
    
    
    if path1:
        Xs = [p[0] for p in path1]
        Ys = [p[1] for p in path1]
        plt.plot(Xs, Ys, "-r", label="Discrete approach")

    if path2:
        Xs = [p[0] for p in path2]
        Ys = [p[1] for p in path2]
        plt.plot(Xs, Ys, "-r", label="Continue approach")


    # Configuramos la visualización. #Pinta todos los centros, ya que pinta el vector, hace plot de todo el vector, tanto scatter, como plot. 
    plt.scatter(h, k, color='red')  # Marcamos el centro
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('RED PATH')
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.grid(True)
    plt.legend()
    plt.axis('equal')  # Para que los ejes x e y tengan la misma escala

    
    # Mostramos la gráfica
    plt.show()


def Euclidian_dist(P1, P2):
    return math.sqrt((P1[0]-P2[0])**2 + (P1[1]-P2[1])**2)


def Compute_forward_weights(Previous_points, Weights, Forward_points):
    
    #Matriz que se devolverá con los pesos calculados de todos los puntos
    Forward_weights=[]
    Positions=[]
    
    #Voy calculando los pesos por parejas de circunferencias. 
    for i in range(len(Forward_points)):

        Weightsij = [Euclidian_dist(Forward_points[i], pj) + wj for pj, wj in zip(Previous_points, Weights)] 
        Minij = min(Weightsij)
        Indexij = Weightsij.index(Minij)

        Forward_weights.append(Minij)
        Positions.append(Indexij)

    return Forward_weights, Positions 

   
#Coge la lista con todas las circunferencias y todos los puntos y crea el vector de pesos
def Compute_discrete_opt(Circs):
    """
    circs es una lista de circunferencias -> lista de listas de puntos (pq cada circunferencia es una lista de puntos)
    """
    #manda una lista de 0 como pesos iniciales la primera vez que se mete. 
    return Compute_discrete_opt_rec(Circs, [0]*len(Circs[0]))[0] #me quedo solo el primer elemento
                                                            #pues los índices me servían para iterar. 

#Programación dinámica. + recursividad.
def Compute_discrete_opt_rec(Circs, Prev_weights):
    """
    circs es una lista de circunferencias -> lista de listas de puntos (pq cada circunferencia es una lista de puntos)
    """
    if len(Circs)==2:
        New_weights, Positions = Compute_forward_weights(Circs[0], Prev_weights, Circs[1])
    
        MinWg = min(New_weights)
        P2_index = New_weights.index(MinWg)
        P1_index = Positions[P2_index]
        #Devuelve de la primera circunferencia el punto que proporciona ese mínimo a la segunda
        # y de la segunda el punto con mínimo peso. 
        return [Circs[0][P1_index], Circs[1][P2_index]], P1_index
    
    New_weights, Positions = Compute_forward_weights(Circs[0], Prev_weights, Circs[1])
    Path, Index = Compute_discrete_opt_rec(Circs[1:], New_weights)
    P1_index = Positions[Index]

    return [Circs[0][P1_index]] + Path, P1_index


if __name__ == "__main__":
    Inicio= time.time()
    # alpha = [1, 3, 5,   6, 8]
    # beta  = [1, 2, 1.5, 2, 5]  
    alpha = [5, 7, 9]
    beta  = [2.5, 6, 2.5]
    R = 1
    epsilon=0.1
    Extension=R-epsilon
    PN=130
    Blue_path=list(zip(alpha, beta))
    Pathlength = 0
    Points=[]
    for i in range(len(alpha)):
        Points.append(Get_circle_points(Blue_path[i][0], Blue_path[i][1], R, PN))
   
    Red_path= Compute_discrete_opt(Points)
    
    for i in range(len(Red_path)-1):
        Pathlength +=Euclidian_dist(Red_path[i],Red_path[i+1])
        
    print("waypoints RP", Red_path)
    print("pathlength RP", Pathlength)

    # r2 = [[1, 1], [2.334674650986388, 1.253446465442755], [4.2127650767489815, 2.116653205305837], [6.068108641928804, 2.997677910397245], [8, 5]]
    # r2= [[1, 1], [2.460381732528291, 1.2797138586569503], [4.3005095588776925, 2.066315391613762], [6.0877825716771214, 2.895708780863792], [8, 5]]
    r2=[5, 2.5], [7.0, 5.0], [9, 2.5]
    Pathlength2=-2
    for i in range(len(r2)-1):
        Pathlength2 +=Euclidian_dist(r2[i],r2[i+1])
    print("pathlength", Pathlength2)
    Final=time.time()    
    print("Tiempo de funcionamiento: ", Final-Inicio)
    plot_solution(Points, alpha, beta, R, None, r2)
