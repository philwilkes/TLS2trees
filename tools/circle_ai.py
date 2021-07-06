import numpy as np
import pandas as pd
from numba import jit
import time


def create_3d_circles_as_points_flat(x, y, z, r, circle_points=15):
    angle_between_points = np.linspace(0, 2 * np.pi, circle_points)
    points = np.zeros((0, 3))
    for i in angle_between_points:
        x2 = r * np.cos(i) + x
        y2 = r * np.sin(i) + y
        point = np.array([[x2, y2, z]])
        points = np.vstack((points, point))
    return points


def make_circles(circlearray):
    for i in circlearray:
        create_3d_circles_as_points_flat(i[0], i[1], i[2], i[3])


jit_create_3d_circles_as_points_flat = jit()(create_3d_circles_as_points_flat)


def make_circles2(circlearray):
    for i in circlearray:
        jit_create_3d_circles_as_points_flat(i[0], i[1], i[2], i[3])


jit_make_circles = jit()(make_circles2)

circle_array = np.hstack((np.random.uniform(0.05, 5, size=(10000, 1)), np.random.uniform(0.05, 5, size=(10000, 1)), np.random.uniform(0.05, 5, size=(10000, 1)), np.random.uniform(0.1, 2, size=(10000, 1))))

t1 = time.time()
make_circles(circle_array)
print('T1', time.time() - t1)

t1 = time.time()
jit_make_circles(circle_array)
print('T1', time.time() - t1)

t1 = time.time()
jit_make_circles(circle_array)
print('T1', time.time() - t1)

