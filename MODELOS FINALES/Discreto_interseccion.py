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

# def intersection (Sp, Pp, Fp):
#     print("uso",Sp)
#     print("con",Pp , Fp)
#     alpha= [Sp[0][0], Sp[1][0]]
#     beta=[Sp[0][1], Sp[1][1]]
#     x=[Pp[0], Fp[0]]
#     y=[Pp[1], Fp[1]]
    
#     Flag=True
    
#     ec1=    (((beta[1]-beta[0])*(x[0]-alpha[1])-(alpha[1]-alpha[0])*(y[0]-beta[1]))*((beta[1]-beta[0])*(x[1]-alpha[1])-(alpha[1]-alpha[0])*(y[1]-beta[1])))
#     ec2=    (((y[1]-y[0])*(alpha[0]-x[1])-(x[1]-x[0])*(beta[0]-beta[1]))*((y[1]-y[0])*(alpha[1]-x[1])-(x[1]-x[0])*(beta[1]-y[1])))
#     print(ec1, ec2)  
#     if ec1 <=0 and ec2 <=0:
#         Flag=False
#     print(Flag)
#     return  Flag
# #     

def point_orientation(p, q, r):
    """Return the orientation of the triplet (p, q, r).
    0 -> p, q and r are collinear
    1 -> Clockwise
    2 -> Counterclockwise
    """
    ecuation = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if ecuation == 0:
        return 0
    elif ecuation > 0:
        return 1
    else:
        return 2

def point_on_segment(p, q, r):
    """Check if point q lies on segment pr."""
    if min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1]):
        return True
    return False


def intersection(p1, q1, p2, q2):
    """Return True if line segments 'p1q1' and 'p2q2' intersect."""
    # Find the four orientations needed for the general and special cases
    
    # if p1 == q1 or p1 == q2 or p1 == q1 or p1 == q2
    
    o1 = point_orientation(p1, q1, p2)
    o2 = point_orientation(p1, q1, q2)
    o3 = point_orientation(p2, q2, p1)
    o4 = point_orientation(p2, q2, q1)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # Special cases
    # p1, q1 and p2 are collinear and p2 lies on segment p1q1
    if o1 == 0 and point_on_segment(p1, p2, q1):
        return True

    # p1, q1 and q2 are collinear and q2 lies on segment p1q1
    if o2 == 0 and point_on_segment(p1, q2, q1):
        return True

    # p2, q2 and p1 are collinear and p1 lies on segment p2q2
    if o3 == 0 and point_on_segment(p2, p1, q2):
        return True

    # p2, q2 and q1 are collinear and q1 lies on segment p2q2
    if o4 == 0 and point_on_segment(p2, q1, q2):
        return True

    return False


def euclidian_dist(P1, P2):
    return math.sqrt((P1[0]-P2[0])**2 + (P1[1]-P2[1])**2)


def compute_forward_weights(previous_points, weights, forward_points, Solar_panels):
    
    #Matriz que se devolverá con los pesos calculados de todos los puntos
    forward_weights=[]
    positions=[]
    M = 10**6
    print(Solar_panels)
    #Voy calculando los pesos por parejas de circunferencias. 
    for i in range(len(forward_points)):
            
            # comparar pj, forward_points[i], Solar_panels[k], Solar_panels[k+1]
            # weightsij = [M if any(do_intersect(pj,Solar_panels[k] ,forward_points[i],Solar_panels[k+1]) for k in range(len(Solar_panels)-1)) else euclidian_dist(forward_points[i], pj) + wj for pj, wj in zip(previous_points, weights)] 
            weightsij = [M if intersect_with_segments(pj, forward_points[i], Solar_panels) else euclidian_dist(forward_points[i], pj) + wj for pj, wj in zip(previous_points, weights)] 
            # weightsij = [euclidian_dist(forward_points[i], pj) + wj for pj, wj in zip(previous_points, weights)] 

            minij = min(weightsij)
            indexij = weightsij.index(minij)
    
            forward_weights.append(minij)
            positions.append(indexij)
          
    return forward_weights, positions 


def intersect_with_segments(P,Q, segments):
    
    for si in segments:
        if intersection(P,Q,si[0],si[1]):
            return True

    return False
        

#Coge la lista con todas las circunferencias y todos los puntos y crea el vector de pesos
def compute_discrete_opt(circs,Solar_panels):
    """
    circs es una lista de circunferencias -> lista de listas de puntos (pq cada circunferencia es una lista de puntos)
    """
    
    #manda una lista de 0 como pesos iniciales la primera vez que se mete. 
    return compute_discrete_opt_rec(circs, [0]*len(circs[0]),Solar_panels)[0] #me quedo solo el primer elemento
                                                                              #pues los índices me servían para iterar. 

#Programación dinámica. + recursividad.
def compute_discrete_opt_rec(circs, prev_weights, Solar_panels):
    """
    circs es una lista de circunferencias -> lista de listas de puntos (pq cada circunferencia es una lista de puntos)
    """
    
    
        
    if len(circs)==2:
        new_weights, positions = compute_forward_weights(circs[0], prev_weights, circs[1],Solar_panels)

        minWg = min(new_weights)
        P2_index = new_weights.index(minWg)
        P1_index = positions[P2_index]
        #Devuelve de la primera circunferencia el punto que proporciona ese mínimo a la segunda
        # y de la segunda el punto con mínimo peso. 
        return [circs[0][P1_index], circs[1][P2_index]], P1_index
   
    new_weights, positions = compute_forward_weights(circs[0], prev_weights, circs[1], Solar_panels)
   
    path, index = compute_discrete_opt_rec(circs[1:], new_weights, Solar_panels)
    p1_index = positions[index]

    return [circs[0][p1_index]] + path, p1_index


if __name__ == "__main__":

    # alpha = [1, 3, 5,   6, 8]
    # beta  = [1, 2, 1.5, 2, 5]
    alpha=[2.07, 7, 2.27, 8.99, 8.71]
    beta=[2.73, 3, 6.41, 5.51, 8.25]
    radio = 1
    epsilon=0.1
    Extension=radio-epsilon
    division_number=40
    blue_path=list(zip(alpha, beta))
    #Segmentos con placas. 
    # alpha=[2.07, 7, 2.27, 8.99, 8.71]
    # beta=[2.73, 3, 6.41, 5.51, 8.25]
    a_p =[2.27, 8.99]
    b_p =[6.41, 5.51]
    #Blue path con placas
    Solar_panels = [
        [[7, 3], [2.27, 6.41]],
        [[2.27, 6.41], [8.99, 5.51]],
        # [P, Q]
    ]
    # Solar_panels=list(zip(a_p, b_p))
    points=[]
    for i in range(len(alpha)):
        points.append(get_circle_points(blue_path[i][0], blue_path[i][1], Extension, division_number))
   
    red_path= compute_discrete_opt(points,Solar_panels)
    pathlength = 0
    for i in range(len(red_path)-1):
        pathlength +=euclidian_dist(red_path[i],red_path[i+1])
        
    # print("waypoints", red_path)
    # print("pathlength", pathlength)

    r2 = [[1, 1], [2.334674650986388, 1.253446465442755], [4.2127650767489815, 2.116653205305837], [6.068108641928804, 2.997677910397245], [8, 5]]

    plot_solution(points, alpha, beta, Extension, red_path, r2)
