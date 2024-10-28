import cv2
import numpy as np
import math


def rotate(image, point: tuple, angle: float) -> np.ndarray:
    """
    Повернуть изображение по часовой стрелке на угол от 0 до 360 градусов и преобразовать размер изображения.

    :param image: исходное изображение
    :param point: значение точки (x, y), вокруг которой повернуть изображение
    :param angle: угол поворота
    :return: повернутное изображение
    """
    angle = -angle

    def coord_shift(x, y):
        return x*math.cos(math.radians(angle)) - y*math.sin(math.radians(angle)), x*math.sin(math.radians(angle)) + y*math.cos(math.radians(angle))
    
    x_min = image.shape[1]
    x_max = 0
    y_min = image.shape[2]
    y_max = 0
    for y_pos in range(image.shape[0]):
        for x_pos in range(image.shape[1]):
            new_x, new_y = coord_shift(x_pos, y_pos)

            if new_x < x_min:
                x_min = int(new_x)
            if new_x > x_max:
                x_max = int(new_x)

            if new_y < y_min:
                y_min = int(new_y)
            if new_y > y_max:
                y_max = int(new_y)

    print(x_min,x_max)
    print(y_min,y_max)
    
    new_image = np.zeros((y_max - y_min, x_max - x_min, 3), dtype = int)

    for y_pos in range(image.shape[0]):
        for x_pos in range(image.shape[1]):
            new_x, new_y = coord_shift(x_pos, y_pos)
            try:
                new_image[round(new_y - y_min), round(new_x - x_min)] = np.array(image[y_pos, x_pos],dtype = int)
            except IndexError:
                pass


    return new_image

from sympy import Matrix, pprint

def apply_warpAffine(image, points1, points2) -> np.ndarray:
    """
    Применить афинное преобразование согласно переходу точек points1 -> points2 и
    преобразовать размер изображения.

    :param image:
    :param points1:
    :param points2:
    :return: преобразованное изображение
    """
    # Ваш код
    # prm = [
    #     [points2[0][0] / points1[0][0], points2[1][0] / points1[1][0], points2[2][0] / points1[2][0]],
    #     [points2[0][1] / points1[0][1], points2[1][1] / points1[1][1], points2[2][1] / points1[2][1]],
    #     [0,0,1]
    # ]
    # b = (points1[0][0] * points2[1][0] - points2[0][0] * points1[1][0]) / (points1[0][0] * points1[1][1] - points1[1][0] * points1[0][1])
    # a = (points2[0][0] - b * points1[0][1]) / points1[0][0]
    
    # c = (points1[0][0] * points2[1][1] - points2[0][1] * points1[1][0]) / (points1[0][0] * points1[1][1] - points1[1][0] * points1[0][1])
    # d = (points2[0][1] - b * points1[0][1]) / points1[0][0]

    augmented_matrix = Matrix([
        [points1[0][0], points1[0][1], 1, 0, 0, 0, points2[0][0]],
        [points1[1][0], points1[1][1], 1, 0, 0, 0, points2[1][0]],
        [points1[2][0], points1[2][1], 1, 0, 0, 0, points2[2][0]],
        [0, 0, 0, points1[0][0], points1[0][1], 1, points2[0][1]],
        [0, 0, 0, points1[1][0], points1[1][1], 1, points2[1][1]],
        [0, 0, 0, points1[2][0], points1[2][1], 1, points2[2][1]],
    ])

    row_reduced_matrix, _ = augmented_matrix.rref()
    pprint(row_reduced_matrix)

    a = float(row_reduced_matrix.row(0)[-1])
    b = float(row_reduced_matrix.row(1)[-1])
    c = float(row_reduced_matrix.row(2)[-1])
    d = float(row_reduced_matrix.row(3)[-1])
    e = float(row_reduced_matrix.row(4)[-1])
    f = float(row_reduced_matrix.row(5)[-1])

    prm = [
        [a,b,c],
        [d,e,f],
        [0,0,1]
    ]

    print(prm)

    def coord_shift(x, y):
        return x*prm[0][0] + y*prm[0][1] + prm[0][2], x*prm[1][0] + y*prm[1][1] + prm[1][2]

    x_min = image.shape[1]
    x_max = 0
    y_min = image.shape[2]
    y_max = 0
    for y_pos in range(image.shape[0]):
        for x_pos in range(image.shape[1]):
            new_x, new_y = coord_shift(x_pos, y_pos)

            if new_x < x_min:
                x_min = int(new_x)
            if new_x > x_max:
                x_max = int(new_x)

            if new_y < y_min:
                y_min = int(new_y)
            if new_y > y_max:
                y_max = int(new_y)

    print(x_min,x_max)
    print(y_min,y_max)
    
    new_image = np.zeros((y_max - y_min, x_max - x_min, 3), dtype = int)

    for y_pos in range(image.shape[0]):
        for x_pos in range(image.shape[1]):
            new_x, new_y = coord_shift(x_pos, y_pos)
            try:
                new_image[round(new_y - y_min), round(new_x - x_min)] = np.array(image[y_pos, x_pos],dtype = int)
            except IndexError:
                pass


    return new_image
