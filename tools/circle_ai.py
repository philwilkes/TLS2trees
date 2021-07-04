import numpy as np
import pandas as pd


def create_3d_circles_as_points_flat(x, y, z, r, circle_points=15):
    angle_between_points = np.linspace(0, 2 * np.pi, circle_points)
    points = np.zeros((0, 3))
    for i in angle_between_points:
        x2 = r * np.cos(i) + x
        y2 = r * np.sin(i) + y
        point = np.array([[x2, y2, z]])
        points = np.vstack((points, point))
    return points


circle_array = np.vstack((np.random.uniform(0.05, 5, size=(10, 1)), np.random.uniform(0.05, 5, size=(10, 1)), np.random.uniform(0.05, 5, size=(10, 1))))



print(circle_array)


"""
Idea
Want AI to predict circle radius and centre from 2D points.
"""