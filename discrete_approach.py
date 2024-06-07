

def get_circle_points(x, y, radio, n):
    pass 



def plot_points(point_list):
    pass


if __name__ == "__main__":

    x, y = 0, 0
    radio = 1
    n = 10

    points = get_circle_points(x, y, radio, n)
    print(points)
    # Hasta aqui paso 1

    x, y = 3, 3
    radio = 1
    n = 360
    points2 = get_circle_points(x, y, radio, n)

    point_list = [points, points2]
    plot_points(point_list)
    # Hasta aqui paso 2





