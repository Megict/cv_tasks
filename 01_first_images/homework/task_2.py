import cv2
import numpy as np


def find_road_number(image: np.ndarray) -> int:
    """
    Найти номер дороги, на которой нет препятсвия в конце пути.

    :param image: исходное изображение
    :return: номер дороги, на котором нет препятсвия на дороге
    """
    road_number = None
    # Ваш код тут
    car_lane = None
    cur_lane_number = 0
    lane_status = None
    for y_pos in range(image.shape[0]):
        for x_pos in range(image.shape[1]):
            if (image[y_pos, x_pos] < 10).all():
                cur_lane_number += 1
            if (image[y_pos, x_pos] == [ 49, 119, 253]).all():
                car_lane = cur_lane_number - 1
            if (image[y_pos, x_pos] == [254,  38,   1]).all():
                if lane_status != None:
                    lane_status[cur_lane_number - 1] = 0
        if y_pos == 0:
            lane_status = [1 for _ in range(cur_lane_number)]
        cur_lane_number = 0

    print(lane_status)
    print("car in lane", car_lane)

    if lane_status[car_lane] == 1:
        print("stay in your lane")
        road_number = None
    else:
        if max(lane_status) == 1:
            print("move to lane", np.argmax(lane_status))
            road_number = np.argmax(lane_status)
        else:
            print("no free lanes")
            road_number = None

    return road_number
