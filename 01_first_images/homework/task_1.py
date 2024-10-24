import cv2
import numpy as np


def find_way_from_maze(image: np.ndarray) -> tuple:
    """
    Найти путь через лабиринт.

    :param image: изображение лабиринта
    :return: координаты пути из лабиринта в виде (x, y), где x и y - это массивы координат
    """
    coords = None
    coords = [[],[]]

    x_pos = 0
    y_pos = 0

    #ищем стартовую точку
    while (image[y_pos, x_pos] == [0,0,0]).all():
        x_pos += 1

    path_direction = "DOWN"

    i_num = 0
    while(y_pos != image.shape[0] - 1):
        if path_direction == "DOWN":
            if (image[y_pos, x_pos - 1] == [0,0,0]).all():
                if (image[y_pos + 1, x_pos] != [0,0,0]).all():
                    # продолжить движение в том же направлении
                    y_pos += 1
                else:
                    # повернуть, потому что встретилось препятствие
                    path_direction = "RIGHT"
            else:
                # дойти до стенки
                x_pos -= 1
        if path_direction == "RIGHT":
            if (image[y_pos + 1, x_pos] == [0,0,0]).all():
                if (image[y_pos, x_pos + 1] != [0,0,0]).all():
                    x_pos += 1
                else:
                    path_direction = "UP"
            else:
                y_pos += 1
        if path_direction == "UP":
            if (image[y_pos, x_pos + 1] == [0,0,0]).all():
                if (image[y_pos - 1, x_pos] != [0,0,0]).all():
                    y_pos -= 1
                else:
                    path_direction = "LEFT"
            else:
                x_pos += 1
        if path_direction == "LEFT":
            if (image[y_pos - 1, x_pos] == [0,0,0]).all():
                if (image[y_pos, x_pos - 1] != [0,0,0]).all():
                    x_pos -= 1
                else:
                    path_direction = "DOWN"
            else:
                y_pos -= 1

        i_num += 1
        if(i_num > 100000):
            break
        
        if len(coords[0]) != 0:
            if x_pos > coords[0][-1]:
                # если мы начали куда-то двигаться, меняем направление
                path_direction = "RIGHT"
            if x_pos < coords[0][-1]:
                path_direction = "LEFT"
            if y_pos > coords[1][-1]:
                path_direction = "DOWN"
            if y_pos < coords[1][-1]:
                path_direction = "UP"

        coords[0].append(x_pos)
        coords[1].append(y_pos)
        print(x_pos, y_pos, path_direction)
        

    return coords[1], coords[0]
