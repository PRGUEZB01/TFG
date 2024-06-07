
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
    
    
    
    
    # Creamos la gráfica en conjunto
    plt.figure(figsize=(6, 6))
    
    for i in range(len(point_list)):
        aux=1
        plt.plot(x[i], y[i], label='Circunferencia'+str(i))
    
        # Añadimos los puntos adicionales
        plt.scatter(additional_x[i], additional_y[i], color='blue', zorder=5, label='Puntos adicionales')
        if i<len(point_list)-1:
            plt.plot([h[i], h[i+1]], [k[i], k[i+1]], linestyle='--', color='black', label='Línea entre centros')
        
        aux=aux+1
    # Configuramos la visualización
    plt.scatter(h, k, color='red', label='Centro (3, 2)')  # Marcamos el centro
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


if __name__ == "__main__":

    x, y = 0, 0
    radio = 1
    n = 10

    points = get_circle_points(x, y, radio, n)
    # Hasta aqui paso 1


    x, y = 3, 3
    radio = 1
    n = 140
    points2 = get_circle_points(x, y, radio, n)

    x, y = 5,6
    radio = 1
    n = 140
    points3 = get_circle_points(x, y, radio, n)



    point_list = [points, points2, points3]
    alpha=(0,3,5)
    betha=(0,3,6)
    plot_points(point_list, alpha, betha, radio)
    # Hasta aqui paso 2
    
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


