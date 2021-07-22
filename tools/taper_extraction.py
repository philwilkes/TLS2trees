import math
import sys
from copy import deepcopy
from multiprocessing import get_context
import networkx as nx
import numpy as np
import pandas as pd
import simplekml
import utm
import os
from scipy import spatial
from scipy.spatial import ConvexHull
from scipy.interpolate import CubicSpline
from scipy.interpolate import griddata
from skimage.measure import LineModelND, CircleModel, ransac
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from tools import load_file, save_file, low_resolution_hack_mode, subsample_point_cloud, clustering
import time
import hdbscan
from skspatial.objects import Plane
import warnings


def get_taper(single_tree_cyls, slice_heights, increment):
    """
    Accepts single tree of cylinders and start height, stop height and increment.
    From start height to stop height (relative to ground), extract the largest
    diameter at a specific increment for that tree.

    Returns:
    slice_heights
    List of slice heights.

    diameters
    List of diameters corresponding to slice heights.

    """
    cyl_dict = dict(x=0, y=1, z=2, nx=3, ny=4, nz=5, radius=6, CCI=7, branch_id=8, parent_branch_id=9,
                    tree_id=10, segment_volume=11, segment_angle_to_horiz=12, height_above_dtm=13)

    kdtree = spatial.cKDTree(np.atleast_2d(single_tree_cyls[:, cyl_dict['height_above_dtm']]).T)

    diameters = []

    for height in slice_heights:
        results = single_tree_cyls[kdtree.query_ball_point([height], r=increment/2)]
        if results.shape[0] > 0:
            diameters.append(np.max(results[:, cyl_dict['radius']]))
        else:
            diameters.append(0)

    return diameters


def extract_tapers_from_plot(plot_cyls, cyl_dict, slice_heights, increment):
    taper_array = np.zeros((0, 1 + slice_heights.shape[0]))

    for tree_id in np.unique(plot_cyls[:, cyl_dict['tree_id']])[:10]:
        individual_tree_cyls = plot_cyls[plot_cyls[:, cyl_dict['tree_id']] == tree_id]
        # if individual_tree_cyls.shape[0] > 0:
        diameters = get_taper(individual_tree_cyls, slice_heights, increment)
        taper_array = np.vstack((taper_array, np.hstack((np.array([tree_id]), diameters))))

    return taper_array


cyl_dict = dict(x=0, y=1, z=2, nx=3, ny=4, nz=5, radius=6, CCI=7, branch_id=8, parent_branch_id=9,
                tree_id=10, segment_volume=11, segment_angle_to_horiz=12, height_above_dtm=13)
all_cyls, _ = load_file('E:/PFOlsen/PFOlsenPlots/T1_class_FSCT_output/cleaned_cyls.las', headers_of_interest=list(cyl_dict))
start_height = 0
stop_height = 10
increment = 0.5
slice_heights = np.linspace(start_height, stop_height, int(np.ceil((stop_height - start_height) / increment) + 1))

save_directory = 'E:/PFOlsen/PFOlsenPlots/T1_class_FSCT_output/'

tapers = extract_tapers_from_plot(all_cyls, cyl_dict, slice_heights, increment)
pd.DataFrame(tapers, columns=['TreeId']+list(slice_heights)).to_csv(save_directory + 'tapers.csv', index=False)
