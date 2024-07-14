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


def Plot_solution(point_list, h, k, r,obstacles, path1=None, path2=None):
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
    
    
    plt.figure(figsize=(6, 6))
    
    for i in range(len(point_list)):
        aux=1
      
    
        # Añadimos los puntos adicionales
        plt.scatter(additional_x[i], additional_y[i],s=1, color='blue', zorder=2)
        if i<len(point_list)-1:
            plt.plot([h[i], h[i+1]], [k[i], k[i+1]], linestyle='--', color='black')
        
        aux=aux+1
    for si in obstacles:
        plt.plot([si[0][0], si[1][0]], [si[0][1], si[1][1]],linewidth= 1, color='black')
    
    if path1:
        Xs = [p[0] for p in path1]
        Ys = [p[1] for p in path1]
        plt.plot(Xs, Ys, "-r", label="discrete approach")

    if path2:
        Xs = [p[0] for p in path2]
        Ys = [p[1] for p in path2]
        plt.plot(Xs, Ys, "-g", label="continue approach")
    

    #Pinta todos los centros, ya que pinta el vector, hace plot de todo el vector, tanto scatter, como plot. 
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


def Point_orientation(p, q, r):
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

def Point_on_segment(p, q, r):
    """Check if point q lies on segment pr."""
    if min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1]):
        return True
    return False


def Intersection(p1, q1, p2, q2):
    """Return True if line segments 'p1q1' and 'p2q2' intersect."""
    
    
    # if p1 == q1 or p1 == q2 or p1 == q1 or p1 == q2
    
    o1 = Point_orientation(p1, q1, p2)
    o2 = Point_orientation(p1, q1, q2)
    o3 = Point_orientation(p2, q2, p1)
    o4 = Point_orientation(p2, q2, q1)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # p1, q1 and p2 are collinear and p2 lies on segment p1q1
    if o1 == 0 and Point_on_segment(p1, p2, q1):
        return True

    # p1, q1 and q2 are collinear and q2 lies on segment p1q1
    if o2 == 0 and Point_on_segment(p1, q2, q1):
        return True

    # p2, q2 and p1 are collinear and p1 lies on segment p2q2
    if o3 == 0 and Point_on_segment(p2, p1, q2):
        return True

    # p2, q2 and q1 are collinear and q1 lies on segment p2q2
    if o4 == 0 and Point_on_segment(p2, q1, q2):
        return True

    return False


def Euclidian_dist(P1, P2):
    return math.sqrt((P1[0]-P2[0])**2 + (P1[1]-P2[1])**2)


def Compute_forward_weights(Previous_points, Weights, Forward_points, Solar_panels, Flag):
    
    Forward_weights=[]
    Positions=[]
    M = 10**6
  
    #Voy calculando los pesos por parejas de circunferencias. 
    for i in range(len(Forward_points)):
            
            
            Weightsij = [M if Intersect_with_segments(pj, Forward_points[i], Solar_panels,Flag) else Euclidian_dist(Forward_points[i], pj) + wj for pj, wj in zip(Previous_points, Weights)] 

            Minij = min(Weightsij)
            Indexij = Weightsij.index(Minij)
    
            Forward_weights.append(Minij)
            Positions.append(Indexij)
          
    return Forward_weights, Positions 


def Intersect_with_segments(P,Q, segments, Flag):
    # print(segments[-1][0])
    if segments[0][0][0]==None:
        return False
    for si in segments[:-1]:
        if Intersection(P,Q,si[0],si[1]):
            return True
        elif Flag==True and Intersection(Q,segments[-1][0],si[0],si[1])==True:
            return True
    return False
        

#Coge la lista con todas las circunferencias y todos los puntos y crea el vector de pesos
def Compute_discrete_opt(Circs,Solar_panels):
    """
    circs es una lista de circunferencias -> lista de listas de puntos (pq cada circunferencia es una lista de puntos)
    """
    #manda una lista de 0 como pesos iniciales la primera vez que se mete. 
    return Compute_discrete_opt_rec(Circs, [0]*len(Circs[0]),Solar_panels)[0] #me quedo solo el primer elemento
                                                                              #pues los índices me servían para iterar. 

#Programación dinámica. + recursividad.
def Compute_discrete_opt_rec(Circs, Prev_weights, Solar_panels):
    """
    circs es una lista de circunferencias -> lista de listas de puntos (pq cada circunferencia es una lista de puntos)
    """       
    if len(Circs)==2:
        New_weights, Positions = Compute_forward_weights(Circs[0], Prev_weights, Circs[1],Solar_panels, True)
        MinWg = min(New_weights)
        P2_index = New_weights.index(MinWg)
        P1_index = Positions[P2_index]
        #Devuelve de la primera circunferencia el punto que proporciona ese mínimo a la segunda
        # y de la segunda el punto con mínimo peso. 
        return [Circs[0][P1_index], Circs[1][P2_index]], P1_index
   
    New_weights, Positions = Compute_forward_weights(Circs[0], Prev_weights, Circs[1], Solar_panels, False)
   
    Path, Index = Compute_discrete_opt_rec(Circs[1:], New_weights, Solar_panels)
    P1_index = Positions[Index]

    return [Circs[0][P1_index]] + Path, P1_index


if __name__ == "__main__":

   """"CASO 1. OCTÓGONO"""
   
Prom_longitudes=[]
Prom_tiempo=[]
for i in range(1):
    # alpha=[3.7, 4.17, 3.71, 2.67, 1.64, 1.17, 1.63, 2.67, 3.8]
    # beta=[3.84, 2.75, 1.67, 1.25, 1.66, 2.75, 3.83, 4.25, 5]
    # R = 1
    # epsilon=0.1
    # Extension=R-epsilon
    # PN=130
    # Blue_path=list(zip(alpha, beta))
    # Solar_panels = [
    # [[3.7, 3.84], [2.67, 4.25]],
    # [[3.71, 1.67], [2.67, 1.25]],
    # [[2.67, 4.25], [1.63, 3.83]],
    # [[1.63, 3.83], [1.17, 2.75]],
    # [[3.8, 5]]
    # ]
    # Pathlength = 0
    
    """Caso 2. Comportamiento del espacio libre"""
    alpha=[2.1,3.1,3.13,0.1,4.1,7.1,8.25,7.08,3.11,0.8]
    beta=[3.1,2.1,4.51,2.1,0.1,2.1,3.11,3.71,5.77,4.44]
    R = 1
    epsilon=0.1
    Extension=R-epsilon
    PN=180
    Blue_path=list(zip(alpha, beta))
    Solar_panels = [
    [[2.1,3.1], [3.1,2.1]],
    [[3.13,4.51], [0.1,2.1]],
    [[7.08,3.71], [3.11,5.77]],
    [[0.8,4.44]]
    ]
    Pathlength = 0
    Inicio= time.time()
    Points=[]
    for i in range(len(alpha)):
        Points.append(Get_circle_points(Blue_path[i][0], Blue_path[i][1], Extension, PN))
           
    Red_path= Compute_discrete_opt(Points,Solar_panels)
    for i in range(len(Red_path)-1):
        Pathlength +=Euclidian_dist(Red_path[i],Red_path[i+1])
    
    print("waypoints", Red_path)
    print("pathlength", Pathlength)
    print("Distancias entre puntos seleccionados:\n")
    for i in range (len(Red_path)):
        print(Euclidian_dist(Blue_path[i], Red_path[i]))
    Final=time.time()  
    t_proceso=Final-Inicio
    print("Tiempo proceso: ", t_proceso)
    Plot_solution(Points, alpha, beta, Extension,Solar_panels[:-1], Red_path, None)
    Prom_longitudes.append(Pathlength)
    Prom_tiempo.append(t_proceso)

Promedio_longitud=sum(Prom_longitudes)/len(Prom_longitudes)
Promedio_tiempo=sum(Prom_tiempo)/len(Prom_tiempo)
print(f"SI SE APLICA PARA VARIAS ITERACIONES\nEl promedio de la longitud: {Promedio_longitud} \nEl promedio de tiempo: {Promedio_tiempo}\nLa longitud max: ",max(Prom_longitudes), "\nEl tiempo max: ", Prom_tiempo)
print(Prom_longitudes)

