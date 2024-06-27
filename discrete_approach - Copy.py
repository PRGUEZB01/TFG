import math
import matplotlib.pyplot as plt
import numpy as np

def get_circle_points(x, y, r, n):
      
    pi=[]
    for i in range(n):
        tetha=i*((2*math.pi)/n)
        px= [x + r*math.cos(tetha)]
        py= [y + r*math.sin(tetha)]
        pi.append(px+py)
      
    return pi


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
        plt.plot(Xs, Ys, "-r", label="discrete approach")

    if path2:
        Xs = [p[0] for p in path2]
        Ys = [p[1] for p in path2]
        plt.plot(Xs, Ys, "-g", label="continue approach")


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
    
    #Matriz que se devolverá con los pesos calculados de todos los puntos
    forward_weights=[]
    positions=[]
    
    #Voy calculando los pesos por parejas de circunferencias. 
    for i in range(len(forward_points)):

        weightsij = [euclidian_dist(forward_points[i], pj) + wj for pj, wj in zip(previous_points, weights)] 
        minij = min(weightsij)
        indexij = weightsij.index(minij)

        forward_weights.append(minij)
        positions.append(indexij)

    return forward_weights, positions 

   
#Coge la lista con todas las circunferencias y todos los puntos y crea el vector de pesos
def compute_discrete_opt(circs):
    """
    circs es una lista de circunferencias -> lista de listas de puntos (pq cada circunferencia es una lista de puntos)
    """
    #manda una lista de 0 como pesos iniciales la primera vez que se mete. 
    return compute_discrete_opt_rec(circs, [0]*len(circs[0]))[0] #me quedo solo el primer elemento
                                                            #pues los índices me servían para iterar. 

#Programación dinámica. + recursividad.
def compute_discrete_opt_rec(circs, prev_weights):
    """
    circs es una lista de circunferencias -> lista de listas de puntos (pq cada circunferencia es una lista de puntos)
    """
    if len(circs)==2:
        new_weights, positions = compute_forward_weights(circs[0], prev_weights, circs[1])
    
        minWg = min(new_weights)
        P2_index = new_weights.index(minWg)
        P1_index = positions[P2_index]
        #Devuelve de la primera circunferencia el punto que proporciona ese mínimo a la segunda
        # y de la segunda el punto con mínimo peso. 
        return [circs[0][P1_index], circs[1][P2_index]], P1_index
    
    new_weights, positions = compute_forward_weights(circs[0], prev_weights, circs[1])
    path, index = compute_discrete_opt_rec(circs[1:], new_weights)
    p1_index = positions[index]

    return [circs[0][p1_index]] + path, p1_index


if __name__ == "__main__":

    alpha = [1, 3, 5,   6, 8]
    beta  = [1, 2, 1.5, 2, 5]
    radio = 1
    epsilon=0.1
    Extension=radio-epsilon
    division_number=20
    blue_path=list(zip(alpha, beta))
    points=[]
    for i in range(len(alpha)):
        points.append(get_circle_points(blue_path[i][0], blue_path[i][1], Extension, division_number))
   
    red_path= compute_discrete_opt(points)
    pathlength = 0
    for i in range(len(red_path)-1):
        pathlength +=euclidian_dist(red_path[i],red_path[i+1])
        
    print("waypoints", red_path)
    print("pathlength", pathlength)

    r2 = [[1, 1], [2.334674650986388, 1.253446465442755], [4.2127650767489815, 2.116653205305837], [6.068108641928804, 2.997677910397245], [8, 5]]

    plot_solution(points, alpha, beta, Extension, red_path, r2)
