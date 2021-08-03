import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.patches import Circle, PathPatch
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import LineModelND, CircleModel, ransac, EllipseModel
import mpl_toolkits.mplot3d.art3d as art3d
import math
import pandas as pd
from scipy import stats, spatial
import time
import warnings
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
from skimage.measure import LineModelND, CircleModel, ransac
import glob
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.spatial.distance import euclidean
from math import sin, cos, pi
import random
import os
from sklearn.neighbors import NearestNeighbors
from tools import load_file, save_file, subsample_point_cloud, get_heights_above_DTM, clustering
warnings.filterwarnings("ignore", category=RuntimeWarning)


class PostProcessing:
    def __init__(self, parameters):
        self.post_processing_time_start = time.time()
        self.parameters = parameters
        self.filename = self.parameters['input_point_cloud'].replace('\\', '/')
        self.output_dir = os.path.dirname(os.path.realpath(self.filename)).replace('\\', '/') + '/' + self.filename.split('/')[-1][:-4] + '_FSCT_output/'
        print(self.output_dir)
        self.filename = self.filename.split('/')[-1]
        if self.parameters['plot_radius'] != 0:
            self.filename = self.filename[:-4] + '_' + str(self.parameters['plot_radius'] + self.parameters['plot_radius_buffer']) + '_m_crop.las'

        self.noise_class_label = parameters['noise_class']
        self.terrain_class_label = parameters['terrain_class']
        self.vegetation_class_label = parameters['vegetation_class']
        self.cwd_class_label = parameters['cwd_class']
        self.stem_class_label = parameters['stem_class']
        print("Loading segmented point cloud...")
        self.point_cloud, self.headers_of_interest = load_file(self.output_dir+self.filename[:-4] + '_segmented.las', headers_of_interest=['x', 'y', 'z', 'label'])
        self.point_cloud = np.hstack((self.point_cloud, np.zeros((self.point_cloud.shape[0], 1))))  # Add height above DTM column
        self.headers_of_interest.append('height_above_DTM')  # Add height_above_DTM to the headers.
        self.label_index = self.headers_of_interest.index('label')
        self.point_cloud[:, self.label_index] = self.point_cloud[:, self.label_index] + 1  # index offset since noise_class was removed from inference.

    def make_DTM2(self, clustering_epsilon=0.3, min_cluster_points=250, smoothing_radius=1, crop_dtm=False):
        print("Making DTM...")
        """
        This function will generate a Digital Terrain Model (DTM) based on the terrain labelled points.
        """
        self.terrain_points = self.terrain_points[np.logical_and(self.terrain_points[:, 2] >= np.percentile(self.terrain_points[:, 2], 2.5),
                                                                 self.terrain_points[:, 2] <= np.percentile(self.terrain_points[:, 2], 80))]

        kdtree = spatial.cKDTree(self.terrain_points[:, :2], leafsize=10000)
        xmin = np.floor(np.min(self.terrain_points[:, 0])) - 3
        ymin = np.floor(np.min(self.terrain_points[:, 1])) - 3
        xmax = np.ceil(np.max(self.terrain_points[:, 0])) + 3
        ymax = np.ceil(np.max(self.terrain_points[:, 1])) + 3
        x_points = np.linspace(xmin, xmax, int(np.ceil((xmax - xmin) / self.parameters['fine_grid_resolution'])) + 1)
        y_points = np.linspace(ymin, ymax, int(np.ceil((ymax - ymin) / self.parameters['fine_grid_resolution'])) + 1)
        grid_points = np.zeros((0, 3))

        for x in x_points:
            for y in y_points:
                indices = []
                # radius_coarse = self.parameters['coarse_grid_radius']
                radius = self.parameters['fine_grid_resolution']
                while len(indices) < 100:

                    indices = kdtree.query_ball_point([x, y], r=radius)
                    radius += self.parameters['fine_grid_resolution']

                if len(indices) > 0:
                    z_points = self.terrain_points[indices, 2]
                    # z_points = z_points[np.logical_and(z_points <= np.percentile(z_points, 70), z_points >= np.percentile(z_points, 30))]
                    z = np.median(z_points)
                    grid_points = np.vstack((grid_points, np.array([[x, y, z]])))  # np.array([[np.median(self.terrain_points[indices,0]),np.median(self.terrain_points[indices,1]),z]]) ))

        if self.parameters['plot_radius'] > 0:
            plot_centre = np.loadtxt(self.output_dir + 'plot_centre_coords.csv')
            crop_radius = self.parameters['plot_radius'] + self.parameters['plot_radius_buffer']
            grid_points = grid_points[np.linalg.norm(grid_points[:, :2] - plot_centre, axis=1) <= crop_radius]

        elif crop_dtm:
            inds = [len(i) > 0 for i in
                    kdtree.query_ball_point(grid_points[:, :2], r=self.parameters['fine_grid_resolution'] * 5)]
            grid_points = grid_points[inds, :]

        if smoothing_radius > 0:
            kdtree2 = spatial.cKDTree(grid_points, leafsize=1000)
            results = kdtree2.query_ball_point(grid_points, r=smoothing_radius)
            smoothed_Z = np.zeros((0, 1))
            for i in results:
                smoothed_Z = np.vstack((smoothed_Z, np.nanmean(grid_points[i, 2])))
            grid_points[:, 2] = np.squeeze(smoothed_Z)

        return grid_points

    def process_point_cloud(self):
        self.terrain_points = self.point_cloud[self.point_cloud[:, self.label_index] == self.terrain_class_label]
        self.DTM = self.make_DTM2(smoothing_radius=3 * self.parameters['fine_grid_resolution'], crop_dtm=True)
        save_file(self.output_dir + 'DTM.las', self.DTM)
        save_file(self.output_dir + 'terrain_points.las', self.terrain_points)
