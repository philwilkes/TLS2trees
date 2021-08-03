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


class MeasureTree:
    def __init__(self, parameters):
        self.measure_time_start = time.time()
        self.parameters = parameters
        self.filename = self.parameters['input_point_cloud'].replace('\\', '/')
        self.output_dir = os.path.dirname(os.path.realpath(self.filename)).replace('\\', '/') + '/' + self.filename.split('/')[-1][:-4] + '_FSCT_output/'
        self.filename = self.filename.split('/')[-1]

        self.num_procs = parameters['num_procs']
        self.num_neighbours = parameters['num_neighbours']
        self.slice_thickness = parameters['slice_thickness']
        self.slice_increment = parameters['slice_increment']
        self.min_tree_volume = parameters['min_tree_volume']

        self.processing_report = pd.read_csv(self.output_dir + 'processing_report.csv', index_col=False)
        self.parameters['plot_radius'] = float(self.processing_report['Plot Radius'])
        self.parameters['plot_radius_buffer'] = float(self.processing_report['Plot Radius Buffer'])

        self.stem_points, headers_of_interest = load_file(self.output_dir + 'stem_points.las', headers_of_interest=['x', 'y', 'z', 'red', 'green', 'blue'])
        print('stempoints', headers_of_interest)
        if self.parameters['low_resolution_point_cloud_hack_mode']:
            self.stem_points = low_resolution_hack_mode(self.stem_points,
                                                        self.parameters['low_resolution_point_cloud_hack_mode'],
                                                        self.parameters['subsampling_min_spacing'],
                                                        self.parameters['num_procs'])
            save_file(self.output_dir + self.filename[:-4] + '_stem_points_hack_mode_cloud.las', self.stem_points)

        self.vegetation_points, headers_of_interest = load_file(self.output_dir + 'vegetation_points.las', headers_of_interest=['x', 'y', 'z', 'red', 'green', 'blue'])
        self.vegetation_points = np.hstack((self.vegetation_points, np.zeros((self.vegetation_points.shape[0], 2))))
        self.ground_veg = np.zeros((0, self.vegetation_points.shape[1]))
        self.terrain_points, headers_of_interest = load_file(self.output_dir + 'terrain_points.las', headers_of_interest=['x', 'y', 'z', 'red', 'green', 'blue'])
        self.cwd_points, headers_of_interest = load_file(self.output_dir + 'cwd_points.las', headers_of_interest=['x', 'y', 'z', 'red', 'green', 'blue'])

        self.DTM, headers_of_interest = load_file(self.output_dir + 'DTM.las')
        if self.parameters['filter_noise']:
            self.stem_points = self.noise_filtering(self.stem_points, min_neighbour_dist=0.03, min_neighbours=3)
        self.characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'dot', 'm', 'space', '_', '-', 'semiC',
                           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', '_M', 'N', 'O', 'P', 'Q', 'R',
                           'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        self.character_viz = []

        for i in self.characters:
            self.character_viz.append(np.genfromtxt('../tools/numbers/' + i + '.csv', delimiter=','))

        self.cyl_dict = dict(x=0, y=1, z=2, nx=3, ny=4, nz=5, radius=6, CCI=7, branch_id=8, parent_branch_id=9,
                             tree_id=10, segment_volume=11, segment_angle_to_horiz=12, height_above_dtm=13)

        self.veg_dict = dict(x=0, y=1, z=2, red=3, green=4, blue=5, tree_id=6, height_above_dtm=7)
        self.stem_dict = dict(x=0, y=1, z=2, red=3, green=4, blue=5, tree_id=6, height_above_dtm=7)

        self.text_point_cloud = np.zeros((0, 3))
        self.tree_measurements = np.zeros((0, 8))
        self.text_point_cloud = np.zeros((0, 3))

    def interpolate_cyl(self, cyl1, cyl2, resolution):
        """
        Convention to be used
        cyl_1 is child
        cyl_2 is parent
        """
        length = np.linalg.norm(np.array([cyl2[0], cyl2[1], cyl2[2]]) - np.array([cyl1[0], cyl1[1], cyl1[2]]))
        points_per_line = int(np.ceil(length / resolution))
        interpolated = np.zeros((0, 14))
        if cyl1.shape[0] > 0 and cyl2.shape[0] > 0:
            xyzinterp = np.linspace(cyl1[:3], cyl2[:3], points_per_line, axis=0)
            if xyzinterp.shape[0] > 0:
                interpolated = np.zeros((xyzinterp.shape[0], 14))
                interpolated[:, :3] = xyzinterp

                normal = (cyl2[:3] - cyl1[:3]) / np.linalg.norm(cyl2[:3] - cyl1[:3])

                if normal[2] < 0:
                    normal[:3] = normal[:3] * -1

                interpolated[:, 3:6] = normal

                interpolated[:, self.cyl_dict['tree_id']] = cyl1[self.cyl_dict['tree_id']]
                interpolated[:, self.cyl_dict['branch_id']] = cyl1[self.cyl_dict['branch_id']]
                interpolated[:, self.cyl_dict['parent_branch_id']] = cyl2[self.cyl_dict['branch_id']]

                interpolated[:, self.cyl_dict['segment_volume']] += (np.pi * np.mean(
                        interpolated[:, self.cyl_dict['radius']]) ** 2) * length  # volume

                interpolated[:, self.cyl_dict['radius']] = np.min(
                        [cyl1[self.cyl_dict['radius']], cyl2[self.cyl_dict['radius']]])

        return interpolated

    def fix_outliers(self, tree_cylinders):
        CCI_cond = tree_cylinders[:, self.cyl_dict['CCI']] > 0.3
        threshold = np.percentile(tree_cylinders[CCI_cond, self.cyl_dict['radius']], 70)
        dodgy_cylinders = tree_cylinders[tree_cylinders[:, self.cyl_dict['radius']] >= threshold]
        kdtree = spatial.cKDTree(tree_cylinders[:, :3])
        results = kdtree.query_ball_point(dodgy_cylinders[:, :3], r=dodgy_cylinders[:, self.cyl_dict['radius']] * 3)
        tree_cylinders[tree_cylinders[:, self.cyl_dict['radius']] >= threshold, self.cyl_dict['radius']] = [np.median(tree_cylinders[result, self.cyl_dict['radius']]) for result in results]
        return tree_cylinders

    @classmethod
    def compute_angle(cls, normal1, normal2):
        """
        Computes the angle in degrees between two 3D vectors.

        Args:
            normal1:
            normal2:

        Returns:
            theta: angle in degrees
        """
        normal1 = np.atleast_2d(normal1)
        normal2 = np.atleast_2d(normal2)

        norm1 = normal1 / np.atleast_2d(np.linalg.norm(normal1, axis=1)).T
        norm2 = normal2 / np.atleast_2d(np.linalg.norm(normal2, axis=1)).T
        dot = np.clip(np.einsum('ij,ij->i', norm1, norm2), -1, 1)
        theta = np.degrees(np.arccos(dot))
        return theta

    def cylinder_sorting(self, cylinder_array, angle_tolerance, search_angle, distance_tolerance):
        """
        Step 1 of sorting initial cylinders into individual trees.
        For a cylinder to be joined up with another cylinder in this step, it must meet the below conditions.

        All angles are specified in degrees.
            cylinder_array:
                The Numpy array of cylinders created during cylinder fitting.

            angle_tolerance:
                Angle tolerance refers to the angle between major axis vectors of the two cylinders being queried. If
                the angle is less than "angle_tolerance", this condition is satisfied.

            search_angle:
                Search angle refers to the angle between the major axis of cylinder 1, and the vector from cylinder 1's
                centre point to cylinder 2's centre point.

            distance_tolerance:
                Cylinder centre points must be within this distance to meet this condition. Think of a ball of radius
                "distance_tolerance".

        Returns: The sorted cylinder array.
        """

        def within_angle_tolerance(normal1, normal2, angle_tolerance):
            """Checks if normal1 and normal2 are within "angle_tolerance"
            of each other."""
            theta = self.compute_angle(normal1, normal2)
            return abs((theta > 90) * 180 - theta) <= angle_tolerance
            # return theta<=angle_tolerance

        def decision_tree(cyl1, cyl2, angle_tolerance, search_angle):
            """
            Decides if cyl2 should be joined to cyl1 and if they are the same tree.
            angle_tolerance is the maximum angle between normal vectors of cylinders to be considered the same branch.
            """
            vector_array = cyl2[:, :3] - np.atleast_2d(cyl1[:3])
            condition1 = within_angle_tolerance(cyl1[3:6], cyl2[:, 3:6], angle_tolerance)
            condition2 = within_angle_tolerance(cyl1[3:6], vector_array, search_angle)
            cyl2[np.logical_and(condition1, condition2), self.cyl_dict['tree_id']] = cyl1[self.cyl_dict['tree_id']]
            cyl2[np.logical_and(condition1, condition2), self.cyl_dict['parent_branch_id']] = cyl1[
                self.cyl_dict['branch_id']]
            return cyl2

        max_tree_label = 1

        cylinder_array = cylinder_array[
            cylinder_array[:, self.cyl_dict['radius']] != 0]  # ignore all points with radius of 0.
        unsorted_points = cylinder_array

        sorted_points = np.zeros((0, unsorted_points.shape[1]))
        total_points = len(unsorted_points)
        while unsorted_points.shape[0] > 1:
            if sorted_points.shape[0] % 200 == 0:
                print('\r', np.around(sorted_points.shape[0] / total_points, 3), end='')

            current_point_index = np.argmin(unsorted_points[:, 2])
            current_point = unsorted_points[current_point_index]
            if current_point[self.cyl_dict['tree_id']] == 0:
                current_point[self.cyl_dict['tree_id']] = max_tree_label
                max_tree_label += 1

            sorted_points = np.vstack((sorted_points, current_point))
            unsorted_points = np.vstack((unsorted_points[:current_point_index],
                                         unsorted_points[current_point_index + 1:]))
            kdtree = spatial.cKDTree(unsorted_points[:, :3], leafsize=1000)
            results = kdtree.query_ball_point(np.atleast_2d(current_point)[:, :3], r=distance_tolerance)[0]
            unsorted_points[results] = decision_tree(current_point,
                                                     unsorted_points[results],
                                                     angle_tolerance,
                                                     search_angle)
        print('1.000\n')
        return sorted_points

    @classmethod
    def make_cyl_visualisation(cls, cyl):
        """Creates a 3D point cloud representation of a circle."""
        p = MeasureTree.create_3d_circles_as_points_flat(cyl[0], cyl[1], cyl[2], cyl[6])
        points = MeasureTree.rodrigues_rot(p - cyl[:3], [0, 0, 1], cyl[3:6])
        points = np.hstack((points + cyl[:3], np.zeros((points.shape[0], 8))))
        points[:, -8:] = cyl[-8:]
        return points

    @classmethod
    def points_along_line(cls, x0, y0, z0, x1, y1, z1, resolution=0.05):
        """Creates a point cloud representation of a line."""
        points_per_line = int(np.linalg.norm(np.array([x1, y1, z1]) - np.array([x0, y0, z0])) / resolution)
        Xs = np.atleast_2d(np.linspace(x0, x1, points_per_line)).T
        Ys = np.atleast_2d(np.linspace(y0, y1, points_per_line)).T
        Zs = np.atleast_2d(np.linspace(z0, z1, points_per_line)).T
        return np.hstack((Xs, Ys, Zs))

    @classmethod
    def create_3d_circles_as_points_flat(cls, x, y, z, r, circle_points=15):
        """Creates a point cloud representation of a horizontal circle at coordinates x, y, z. and of radius r.
        Circle points is the number of points to use to represent each circle."""
        angle_between_points = np.linspace(0, 2 * np.pi, circle_points)
        points = np.zeros((0, 3))
        for i in angle_between_points:
            x2 = r * np.cos(i) + x
            y2 = r * np.sin(i) + y
            point = np.array([[x2, y2, z]])
            points = np.vstack((points, point))
        return points

    @classmethod
    def rodrigues_rot(cls, P, n0, n1):
        """RODRIGUES ROTATION
        - Rotate given points based on a starting and ending vector
        - Axis k and angle of rotation theta given by vectors n0,n1
        P_rot = P*cos(theta) + (k x P)*sin(theta) + k*<k,P>*(1-cos(theta))"""
        # If P is only 1d array (coords of single point), fix it to be matrix
        if P.ndim == 1:
            P = P[np.newaxis, :]

        # Get vector of rotation k and angle theta
        n0 = n0 / np.linalg.norm(n0)
        n1 = n1 / np.linalg.norm(n1)
        k = np.cross(n0, n1)
        if np.sum(k) != 0:
            k = k / np.linalg.norm(k)
        theta = np.arccos(np.dot(n0, n1))

        # Compute rotated points
        P_rot = np.zeros((len(P), 3))
        for i in range(len(P)):
            P_rot[i] = P[i] * np.cos(theta) + np.cross(k, P[i]) * np.sin(theta) + k * np.dot(k, P[i]) * (
                    1 - np.cos(theta))
        return P_rot

    @classmethod
    def fit_circle_3D(cls, points, V):
        CCI = 0
        r = 0
        P = points[:, :3]
        P_mean = np.mean(P, axis=0)
        P_centered = P - P_mean
        normal = V / np.linalg.norm(V)
        if normal[2] < 0:  # if normal vector is pointing down, flip it around the other way.
            normal = normal * -1

        # Project points to coords X-Y in 2D plane
        P_xy = MeasureTree.rodrigues_rot(P_centered, normal, [0, 0, 1])

        # Fit circle in new 2D coords with RANSAC
        if P_xy.shape[0] >= 20:

            model_robust, inliers = ransac(P_xy[:, :2], CircleModel, min_samples=int(P_xy.shape[0] * 0.3),
                                           residual_threshold=0.3, max_trials=2500)
            xc, yc = model_robust.params[0:2]
            r = model_robust.params[2]
            CCI = MeasureTree.circumferential_completeness_index([xc, yc], r, P_xy[:, :2])

        elif P_xy.shape[0] >= 10 and P_xy.shape[0] < 20:
            model_robust, inliers = ransac(P_xy[:, :2], CircleModel, min_samples=7,
                                           residual_threshold=0.4, max_trials=2500)
            xc, yc = model_robust.params[0:2]
            r = model_robust.params[2]
            CCI = MeasureTree.circumferential_completeness_index([xc, yc], r, P_xy[:, :2])

        if CCI < 0.2:
            r = 0
            xc, yc = np.mean(P_xy[:, :2], axis=0)
            CCI = 0

        # Transform circle center back to 3D coords
        cyl_centre = MeasureTree.rodrigues_rot(np.array([[xc, yc, 0]]), [0, 0, 1], normal) + P_mean
        cyl_output = np.array([[cyl_centre[0, 0], cyl_centre[0, 1], cyl_centre[0, 2], normal[0], normal[1], normal[2],
                                r, CCI, 0, 0, 0, 0, 0, 0]])
        return cyl_output

    def point_cloud_annotations(self, character_size, xpos, ypos, zpos, r, text):
        def convert_character_cells_to_points(character):
            character = np.rot90(character, axes=(1, 0))
            index_i = 0
            index_j = 0
            points = np.zeros((0, 3))
            for i in character:
                for j in i:
                    if j == 1:
                        points = np.vstack((points, np.array([[index_i, index_j, 0]])))
                    index_j += 1
                index_j = 0
                index_i += 1

            roll_mat = np.array([[1, 0, 0],
                                 [0, np.cos(-np.pi / 4), -np.sin(-np.pi / 4)],
                                 [0, np.sin(-np.pi / 4), np.cos(-np.pi / 4)]])
            points = np.dot(points, roll_mat)
            return points

        def get_character(char):
            if char == ':':
                return self.character_viz[self.characters.index('semiC')]
            elif char == '.':
                return self.character_viz[self.characters.index('dot')]
            elif char == ' ':
                return self.character_viz[self.characters.index('space')]
            elif char == 'M':
                return self.character_viz[self.characters.index('_M')]
            else:
                return self.character_viz[self.characters.index(char)]

        text_points = np.zeros((11, 0))
        for i in text:
            text_points = np.hstack((text_points, np.array(get_character(str(i)))))
        points = convert_character_cells_to_points(text_points)

        points = points * character_size + [xpos + 0.2 + 0.5 * r, ypos, zpos]
        return points

    @classmethod
    def fit_cylinder(cls, skeleton_points, point_cloud, num_neighbours, cyl_dict):
        point_cloud = point_cloud[:, :3]
        skeleton_points = skeleton_points[:, :3]
        cyl_array = np.zeros((0, 14))
        line_centre = np.mean(skeleton_points[:, :3], axis=0)
        _, _, vh = np.linalg.svd(line_centre - skeleton_points)
        line_v_hat = vh[0] / np.linalg.norm(vh[0])

        if skeleton_points.shape[0] <= num_neighbours:
            group = skeleton_points
            line_centre = np.mean(group[:, :3], axis=0)
            length = np.linalg.norm(np.max(group, axis=0) - np.min(group, axis=0))
            plane_slice = point_cloud[np.linalg.norm(abs(line_v_hat * (point_cloud - line_centre)), axis=1) < (
                        length / 2)]  # calculate distances to plane at centre of line.
            if plane_slice.shape[0] > 0:
                cylinder = MeasureTree.fit_circle_3D(plane_slice, line_v_hat)
                cyl_array = np.vstack((cyl_array, cylinder))
        else:
            while skeleton_points.shape[0] > num_neighbours:
                nn = NearestNeighbors()
                nn.fit(skeleton_points)
                starting_point = np.atleast_2d(skeleton_points[np.argmin(skeleton_points[:, 2])])
                group = skeleton_points[nn.kneighbors(starting_point,
                                                      n_neighbors=num_neighbours)[1][0]]
                line_centre = np.mean(group[:, :3], axis=0)
                length = np.linalg.norm(np.max(group, axis=0) - np.min(group, axis=0))
                plane_slice = point_cloud[np.linalg.norm(abs(line_v_hat * (point_cloud - line_centre)), axis=1) < (length / 2)]  # calculate distances to plane at centre of line.
                if plane_slice.shape[0] > 0:
                    cylinder = MeasureTree.fit_circle_3D(plane_slice, line_v_hat)
                    cyl_array = np.vstack((cyl_array, cylinder))
                skeleton_points = np.delete(skeleton_points, np.argmin(skeleton_points[:, 2]), axis=0)
        return cyl_array

    @classmethod
    def cylinder_cleaning_multithreaded(cls, args):
        sorted_cylinders, cleaned_measurement_radius, cyl_dict = args
        cleaned_cyls = np.zeros((0, np.shape(sorted_cylinders)[1]))

        # Cleaning step
        while sorted_cylinders.shape[0] > 2:
            start_point_idx = np.argmin(sorted_cylinders[:, 2])
            start_point = sorted_cylinders[start_point_idx, :]
            sorted_cylinders = np.delete(sorted_cylinders, start_point_idx, axis=0)

            kdtree = spatial.cKDTree(sorted_cylinders[:, :3])
            results = kdtree.query_ball_point(start_point[:3], cleaned_measurement_radius)
            neighbours = sorted_cylinders[results]
            best_cylinder = start_point
            if neighbours.shape[0] > 0:
                if np.max(neighbours[:, cyl_dict['CCI']]) > 0:
                    best_cylinder = neighbours[np.argsort(neighbours[:, cyl_dict['CCI']])][-1]
                if neighbours[neighbours[:, cyl_dict['CCI']] >= np.percentile(neighbours[:, cyl_dict['CCI']], 30),
                   :2].shape[0] > 0:
                    best_cylinder[:2] = np.median(neighbours[neighbours[:, cyl_dict['CCI']] >= np.percentile(
                            neighbours[:, cyl_dict['CCI']], 30), :2], axis=0)
                    best_cylinder[3:6] = np.median(neighbours[neighbours[:, cyl_dict['CCI']] >= np.percentile(
                            neighbours[:, cyl_dict['CCI']], 30), 3:6], axis=0)
                    best_cylinder[cyl_dict['radius']] = np.median(neighbours[
                                                                      neighbours[:, cyl_dict['CCI']] >= np.percentile(
                                                                              neighbours[:, cyl_dict['CCI']], 30),
                                                                      cyl_dict['radius']])
                    best_cylinder[2] = np.mean(neighbours[:, 2])

            cleaned_cyls = np.vstack((cleaned_cyls, best_cylinder))
            sorted_cylinders = np.delete(sorted_cylinders, results, axis=0)

        return cleaned_cyls

    def get_CCI_of_all_cyls(self, cyls):
        kdtree = spatial.cKDTree(self.stem_points[:, :3])
        i = 0
        num_cyls = cyls.shape[0]
        results = kdtree.query_ball_point(cyls[:, :3], r=cyls[:, self.cyl_dict['radius']] * 1.3)
        nearby_stem_points_list = [self.stem_points[result] for result in results]
        new_cyls = np.zeros((0, cyls.shape[1]))
        for cyl, nearby_stem_points in zip(cyls, nearby_stem_points_list):
            i += 1
            if i % 50 == 0:
                print(i, '/', num_cyls)
            # Translate cylinders to have current cylinder at origin. Rotate all cylinders about new origin so current cylinder vector is up.
            nearby_points_moved = self.rodrigues_rot(nearby_stem_points[:, :3] - cyl[:3], cyl[3:6], [0, 0, 1])
            # Find all cylinders within 2D radius of the current cylinder.
            # nearby_points_within_radius_mask = np.linalg.norm(nearby_points_moved[:,:2])<=cyl[self.cyl_dict['radius']]

            # Find all cylinders within "slice_increment" of the current cylinder in the longitudinal direction.
            nearby_points_within_thickness = np.abs(nearby_points_moved[:, 2]) < self.slice_increment

            nearby_points = nearby_points_moved[nearby_points_within_thickness]
            r = cyl[self.cyl_dict['radius']]
            CCI = MeasureTree.circumferential_completeness_index([0, 0], r, nearby_points[:, :2])
            cyl[self.cyl_dict['CCI']] = CCI
            new_cyls = np.vstack((new_cyls, cyl))
        return new_cyls

    @staticmethod
    def inside_conv_hull(point, hull, tolerance=1e-5):
        return all((np.dot(eq[:-1], point) + eq[-1] <= tolerance) for eq in hull.equations)

    # @staticmethod
    @classmethod
    def circumferential_completeness_index(cls, fitted_circle_centre, estimated_radius, slice_points):
        angular_region_degrees = 11
        minimum_radius_counted = estimated_radius * 0.7
        maximum_radius_counted = estimated_radius * 1.3
        num_sections = 360 / angular_region_degrees
        angles = np.linspace(-180, 180, num=int(num_sections), endpoint=False)
        theta = np.zeros((1, 1))
        completeness = 0
        for point in slice_points:
            if ((point[1] - fitted_circle_centre[1]) ** 2 + (
                    point[0] - fitted_circle_centre[0]) ** 2) ** 0.5 >= minimum_radius_counted and (
                    (point[1] - fitted_circle_centre[1]) ** 2 + (
                    point[0] - fitted_circle_centre[0]) ** 2) ** 0.5 <= maximum_radius_counted:
                theta = np.vstack((theta, (math.degrees(
                        math.atan2((point[1] - fitted_circle_centre[1]), (point[0] - fitted_circle_centre[0]))))))
        for angle in angles:
            if np.shape(np.where(theta[np.where(theta >= angle)] < (angle + angular_region_degrees)))[1] > 0:
                completeness += 1
        return completeness / num_sections

    @classmethod
    def threaded_cyl_fitting(cls, args):
        skel_cluster, point_cluster, cluster_id, num_neighbours, cyl_dict = args
        cyl_array = np.zeros((0, 14))
        if skel_cluster.shape[0] > num_neighbours:
            cyl_array = cls.fit_cylinder(skel_cluster, point_cluster, num_neighbours=num_neighbours, cyl_dict=cyl_dict)
            cyl_array[:, cyl_dict['branch_id']] = cluster_id
        return cyl_array

    @staticmethod
    def noise_filtering(points, min_neighbour_dist, min_neighbours):
        kdtree = spatial.cKDTree(points[:, :3], leafsize=1000)
        results = kdtree.query_ball_point(points[:, :3], r=min_neighbour_dist)
        if len(results) != 0:
            return points[[len(i) >= min_neighbours for i in results]]
        else:
            return points

    @staticmethod
    def clustering(points, eps=0.05, min_samples=2, n_jobs=1, mode='DBSCAN'):
        assert mode == 'DBSCAN' or mode == 'HDBSCAN'
        if mode == 'HDBSCAN':
            cluster_labels = hdbscan.HDBSCAN(min_cluster_size=100).fit_predict(points[:, :3])
            return np.hstack((points, np.atleast_2d(cluster_labels).T))

        elif mode == 'DBSCAN':
            db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', algorithm='kd_tree', n_jobs=n_jobs).fit(points[:, :3])
            return np.hstack((points, np.atleast_2d(db.labels_).T))

    @classmethod
    def slice_clustering(cls, input_data):
        cluster_array_internal = np.zeros((0, 6))
        medians = np.zeros((0, 3))
        new_slice, clustering_distance = input_data
        if new_slice.shape[0] > 1:
            new_slice = clustering(new_slice[:, :3], mode='HDBSCAN')
            for cluster_id in range(0, int(np.max(new_slice[:, -1])) + 1):
                cluster = new_slice[new_slice[:, -1] == cluster_id]
                median = np.median(cluster[:, :3], axis=0)
                medians = np.vstack((medians, median))
                cluster_array_internal = np.vstack((cluster_array_internal, np.hstack((cluster[:, :3], np.zeros((cluster.shape[0], 3)) + median))))
        return cluster_array_internal, medians

    #
    # @classmethod
    # def slice_clustering(cls, input_data):
    #     cluster_array_internal = np.zeros((0, 6))
    #     medians = np.zeros((0, 3))
    #     new_slice, clustering_distance = input_data
    #     if new_slice.shape[0] > 1:
    #         new_slice = clustering(new_slice[:, :3], eps=clustering_distance)
    #         for cluster_id in range(0, int(np.max(new_slice[:, -1])) + 1):
    #             cluster = new_slice[new_slice[:, -1] == cluster_id]
    #             median = np.median(cluster[:, :3], axis=0)
    #             medians = np.vstack((medians, median))
    #             cluster_array_internal = np.vstack(
    #                     (cluster_array_internal, np.hstack((cluster[:, :3], np.zeros((cluster.shape[0], 3)) + median))))
    #     return cluster_array_internal, medians

    @classmethod
    def within_angle_tolerances(cls, normal1, normal2, angle_tolerance):
        """Checks if normal1 and normal2 are within "angle_tolerance"
        of each other."""
        norm1 = normal1 / np.atleast_2d(np.linalg.norm(normal1, axis=1)).T
        norm2 = normal2 / np.atleast_2d(np.linalg.norm(normal2, axis=1)).T
        dot = np.clip(np.einsum('ij, ij->i', norm1, norm2), a_min=-1, a_max=1)
        theta = np.degrees(np.arccos(dot))
        return abs((theta > 90) * 180 - theta) <= angle_tolerance

    @classmethod
    def within_search_cone(cls, normal1, vector1_2, search_angle):
        norm1 = normal1 / np.linalg.norm(normal1)
        if not (vector1_2 == 0).all():
            norm2 = vector1_2 / np.linalg.norm(vector1_2)
            dot = np.dot(norm1, norm2)
            if dot > 1:  # floating point problems...
                dot = 1
            elif dot < -1:
                dot = -1

            theta = math.degrees(np.arccos(dot))
            # print('Cone Angle',abs((theta > 90)*180-theta) <= search_angle)
            return abs((theta > 90) * 180 - theta) <= search_angle
        else:
            return False

    def get_heights_above_DTM(self, points):
        grid = griddata((self.DTM[:, 0], self.DTM[:, 1]), self.DTM[:, 2], points[:, 0:2], method='linear',
                        fill_value=np.median(self.DTM[:, 2]))
        points[:, -1] = points[:, 2] - grid
        return points

    def run_measurement_extraction(self):
        if self.parameters['run_from_start']:

            skeleton_array = np.zeros((0, 3))
            cluster_array = np.zeros((0, 6))
            slice_heights = np.linspace(np.min(self.stem_points[:, 2]), np.max(self.stem_points[:, 2]), int(np.ceil(
                    (np.max(self.stem_points[:, 2]) - np.min(self.stem_points[:, 2])) / self.slice_increment)))
            # input_data = []
            print("Making and clustering slices...")
            i = 0
            max_i = slice_heights.shape[0]
            for slice_height in slice_heights:
                if i % 10 == 0:
                    # print ('{i}/{max_i}\r'.format(i,max_i),)
                    print('\r', i, '/', max_i, end='')
                i += 1
                # print('{:4.2f} m'.format(slice_height))
                new_slice = self.stem_points[np.logical_and(self.stem_points[:, 2] >= slice_height, self.stem_points[:, 2] < slice_height + self.slice_thickness)]
                if new_slice.shape[0] > 0:
                    cluster, skel = MeasureTree.slice_clustering([new_slice, self.parameters['slice_clustering_distance']])
                    skeleton_array = np.vstack((skeleton_array, skel))
                    cluster_array = np.vstack((cluster_array, cluster))
                    # input_data.append([new_slice, self.parameters['slice_clustering_distance']])
                    # np.savetxt(self.directory+'data/postprocessed_point_clouds/'+self.input_point_cloud+'/new_slice_'+str(i)+'.csv',new_slice)
            print('\r', max_i, '/', max_i, end='')
            print('\nDone\n')

            # input_data = []
            # print("Making slices...")
            # i = 0
            # max_i = slice_heights.shape[0]
            # for slice_height in slice_heights:
            #     if i % 10 == 0:
            #         # print ('{i}/{max_i}\r'.format(i,max_i),)
            #         print('\r', i, '/', max_i, end='')
            #     i += 1
            #     # print('{:4.2f} m'.format(slice_height))
            #     new_slice = self.stem_points[np.logical_and(self.stem_points[:, 2] >= slice_height, self.stem_points[:,
            #                                                                                         2] < slice_height + self.slice_thickness)]
            #     if new_slice.shape[0] > 0:
            #         input_data.append([new_slice, self.parameters['slice_clustering_distance']])
            #         # np.savetxt(self.directory+'data/postprocessed_point_clouds/'+self.input_point_cloud+'/new_slice_'+str(i)+'.csv',new_slice)
            # print('\r', max_i, '/', max_i, end='')
            # print('\nDone\n')
            #
            # print("Starting multithreaded slice clustering...")
            # j = 0
            # max_j = len(input_data)
            # clusteroutputlist = []
            # skeletonoutputlist = []
            # with get_context("spawn").Pool(processes=self.num_procs) as pool:
            #     for i in pool.imap_unordered(MeasureTree.slice_clustering, input_data):
            #         if j % 100 == 0:
            #             print('\r', j, '/', max_j, end='')
            #         j += 1
            #         cluster, skel = i
            #         clusteroutputlist.append(cluster)
            #         skeletonoutputlist.append(skel)
            # print('\r', max_j, '/', max_j, end='')
            # print('\nDone\n')
            # skeleton_array = np.vstack(skeletonoutputlist)
            # cluster_array = np.vstack(clusteroutputlist)
            # del clusteroutputlist, skeletonoutputlist

            print('Clustering skeleton...')
            skeleton_array = clustering(skeleton_array[:, :3], eps=self.slice_increment * 1.5)  # TODO changed from 2 recheck
            skeleton_cluster_visualisation = np.zeros((0, 5))
            for k in np.unique(skeleton_array[:, -1]):  # Just assigns random colours to the clusters to make it easier to see different neighbouring groups.
                skeleton_cluster_visualisation = np.vstack((skeleton_cluster_visualisation, np.hstack((skeleton_array[skeleton_array[:, -1] == k], np.zeros((skeleton_array[skeleton_array[:, -1] == k].shape[0], 1)) + np.random.randint(0, 10)))))

            print("Saving skeleton and cluster array...")
            save_file(self.output_dir + 'skeleton_cluster_visualisation.las', skeleton_cluster_visualisation, ['X', 'Y', 'Z', 'cluster'])

            print("Making kdtree...")
            # Assign unassigned skeleton points to the nearest group.
            unassigned_bool = skeleton_array[:, -1] == -1
            kdtree = spatial.cKDTree(skeleton_array[unassigned_bool][:, :3], leafsize=100000)
            distances, neighbours = kdtree.query(skeleton_array[unassigned_bool, :3], k=2)
            skeleton_array[unassigned_bool, -1][distances[:, 1] < self.slice_increment * 3] = \
                skeleton_array[unassigned_bool, -1][neighbours[:, 1]][distances[:, 1] < self.slice_increment * 3]

            input_data = []
            i = 0
            max_i = int(np.max(skeleton_array[:, -1]) + 1)
            cl_kdtree = spatial.cKDTree(cluster_array[:, 3:], leafsize=100000)
            cluster_ids = range(0, max_i)
            print('Making initial branch/stem section clusters...')

            # organised_clusters = np.zeros((0,5))
            for cluster_id in cluster_ids:
                if i % 100 == 0:
                    print('\r', i, '/', max_i, end='')
                i += 1
                skel_cluster = skeleton_array[skeleton_array[:, -1] == cluster_id, :3]
                sc_kdtree = spatial.cKDTree(skel_cluster, leafsize=100000)
                results = np.unique(np.hstack(sc_kdtree.query_ball_tree(cl_kdtree,
                                                                        r=0.000000001)))  # handling some floating point errors that were giving me problems... dodgy? #TODO
                cluster_array_clean = cluster_array[results, :3]
                input_data.append([skel_cluster[:, :3], cluster_array_clean[:, :3], cluster_id, self.num_neighbours,
                                   self.cyl_dict])
                # organised_clusters = np.vstack((organised_clusters,np.hstack((cluster_array_clean,
                #                                                               np.ones((cluster_array_clean.shape[0],1))*i,
                #                                                               np.ones((cluster_array_clean.shape[0],1))*np.random.randint(0,10)))  ))
            print('\r', max_i, '/', max_i, end='')
            print('\nDone\n')

            # np.savetxt(self.directory+'data/postprocessed_point_clouds/'+self.input_point_cloud+'/organised_clusters.csv',organised_clusters)

            print("Starting multithreaded cylinder fitting...")
            j = 0
            max_j = len(input_data)
            # full_cyl_array = np.zeros((0, 14))
            outputlist = []
            with get_context("spawn").Pool(processes=self.num_procs) as pool:
                for i in pool.imap_unordered(MeasureTree.threaded_cyl_fitting, input_data):
                    outputlist.append(i)
                    # full_cyl_array = np.vstack((full_cyl_array, i))
                    if j % 10 == 0:
                        print('\r', j, '/', max_j, end='')
                        # print(len(outputlist))
                    j += 1
            full_cyl_array = np.vstack(outputlist)
            print('\r', max_j, '/', max_j, end='')
            print('\nDone\n')

            print("Deleting cyls with CCI less than:", self.parameters['minimum_CCI'])
            full_cyl_array = full_cyl_array[full_cyl_array[:, self.cyl_dict['CCI']] >= self.parameters['minimum_CCI']]

            # cyl_array = [x,y,z,nx,ny,nz,r,CCI,branch_id,tree_id,segment_volume,parent_branch_id]
            print("Saving cylinder array...")
            save_file(self.output_dir + 'full_cyl_array.las', full_cyl_array, headers_of_interest=list(self.cyl_dict))
        full_cyl_array, _ = load_file(self.output_dir + 'full_cyl_array.las',
                                      headers_of_interest=list(self.cyl_dict))

        print("Sorting Cylinders...")
        full_cyl_array = self.cylinder_sorting(full_cyl_array,
                                               angle_tolerance=self.parameters['sorting_angle_tolerance'],
                                               search_angle=self.parameters['sorting_search_angle'],
                                               distance_tolerance=self.parameters['sorting_search_radius'])

        print('Correcting Cylinder assignments...')
        sorted_full_cyl_array = np.zeros((0, full_cyl_array.shape[1]))
        t_id = 1
        max_search_radius = self.parameters['max_search_radius']
        min_points = 5
        max_search_angle = self.parameters['max_search_angle']
        max_tree_id = np.unique(full_cyl_array[:, self.cyl_dict['tree_id']]).shape[0]
        for tree_id in np.unique(full_cyl_array[:, self.cyl_dict['tree_id']]):
            if int(tree_id) % 10 == 0:
                print("Tree ID", int(tree_id), '/', int(max_tree_id))
            tree = full_cyl_array[full_cyl_array[:, self.cyl_dict['tree_id']] == int(tree_id)]
            tree_kdtree = spatial.cKDTree(sorted_full_cyl_array[:, :3], leafsize=1000)
            if tree.shape[0] >= min_points:
                lowest_point = tree[np.argmin(tree[:, 2])]
                highest_point = tree[np.argmax(tree[:, 2])]
                lowneighbours = sorted_full_cyl_array[
                    tree_kdtree.query_ball_point(lowest_point[:3], r=max_search_radius)]
                highneighbours = sorted_full_cyl_array[
                    tree_kdtree.query_ball_point(highest_point[:3], r=max_search_radius)]

                lowest_point_z = lowest_point[2] - griddata((self.DTM[:, 0], self.DTM[:, 1]), self.DTM[:, 2],
                                                            lowest_point[0:2], method='linear',
                                                            fill_value=np.median(self.DTM[:, 2]))
                assigned = False
                if lowneighbours.shape[0] > 0:
                    angles = MeasureTree.compute_angle(lowest_point[3:6], lowest_point[:3] - lowneighbours[:, :3])
                    valid_angles = angles[angles <= max_search_angle]

                    if valid_angles.shape[0] > 0:
                        best_parent_point = lowneighbours[np.argmin(angles)]
                        tree = np.vstack((tree, self.interpolate_cyl(lowest_point, best_parent_point,
                                                                     resolution=self.slice_increment)))
                        tree[:, self.cyl_dict['tree_id']] = best_parent_point[self.cyl_dict['tree_id']]
                        sorted_full_cyl_array = np.vstack((sorted_full_cyl_array, tree))
                        assigned = True
                    else:
                        assigned = False

                elif highneighbours.shape[0] > 0:
                    angles = MeasureTree.compute_angle(highest_point[3:6], highneighbours[:, :3] - highest_point[:3])
                    valid_angles = angles[angles <= max_search_angle]

                    if valid_angles.shape[0] > 0:
                        best_parent_point = highneighbours[np.argmin(angles)]
                        tree = np.vstack((tree, self.interpolate_cyl(best_parent_point, highest_point,
                                                                     resolution=self.slice_increment)))
                        tree[:, self.cyl_dict['tree_id']] = best_parent_point[self.cyl_dict['tree_id']]
                        sorted_full_cyl_array = np.vstack((sorted_full_cyl_array, tree))
                        assigned = True
                    else:
                        assigned = False

                if assigned is False and lowest_point_z < 5:
                    tree[:, self.cyl_dict['tree_id']] = t_id
                    sorted_full_cyl_array = np.vstack((sorted_full_cyl_array, tree))
                    t_id += 1

        save_file(self.output_dir + 'sorted_full_cyl_array.las', sorted_full_cyl_array, headers_of_interest=list(self.cyl_dict))
        sorted_full_cyl_array, _ = load_file(self.output_dir + 'sorted_full_cyl_array.las',
                                             headers_of_interest=list(self.cyl_dict))
        print("Cylinder interpolation...")

        tree_list = []
        interpolated_full_cyl_array = np.zeros((0, 14))
        max_tree_id = np.unique(sorted_full_cyl_array[:, self.cyl_dict['tree_id']]).shape[0]
        for tree_id in np.unique(sorted_full_cyl_array[:, self.cyl_dict['tree_id']]):
            if int(tree_id) % 10 == 0:
                print("Tree ID", int(tree_id), '/', int(max_tree_id))
            current_tree = sorted_full_cyl_array[sorted_full_cyl_array[:, self.cyl_dict['tree_id']] == tree_id]
            interpolated_full_cyl_array = np.vstack((interpolated_full_cyl_array, current_tree))
            _, individual_branches_indices = np.unique(current_tree[:, self.cyl_dict['branch_id']], return_index=True)
            tree_list.append(nx.Graph())
            for branch in current_tree[individual_branches_indices]:
                branch_id = branch[self.cyl_dict['branch_id']]
                parent_branch_id = branch[self.cyl_dict['parent_branch_id']]
                tree_list[-1].add_edge(int(parent_branch_id), int(branch_id))
                current_branch = current_tree[current_tree[:, self.cyl_dict['branch_id']] == branch_id]
                parent_branch = current_tree[current_tree[:, self.cyl_dict['branch_id']] == parent_branch_id]

                current_branch_copy = deepcopy(current_branch[np.argsort(current_branch[:, 2])])
                while current_branch_copy.shape[0] > 1:
                    lowest_point = current_branch_copy[0]
                    current_branch_copy = current_branch_copy[1:]
                    # find nearest point. if nearest point > increment size, interpolate.
                    distances = np.abs(np.linalg.norm(current_branch_copy[:, :3] - lowest_point[:3], axis=1))
                    if distances[distances > 0].shape[0] > 0:
                        if np.min(distances[distances > 0]) > self.slice_increment:
                            interp_to_point = current_branch_copy[distances > 0]
                            if interp_to_point.shape[0] > 0:
                                interp_to_point = interp_to_point[np.argmin(distances[distances > 0])]

                            # Interpolates a single branch.
                            if interp_to_point.shape[0] > 0:
                                interpolated_cyls = self.interpolate_cyl(interp_to_point, lowest_point,
                                                                         resolution=self.slice_increment)
                                current_branch = np.vstack((current_branch, interpolated_cyls))
                                interpolated_full_cyl_array = np.vstack(
                                        (interpolated_full_cyl_array, interpolated_cyls))

                if parent_branch.shape[0] > 0:
                    parent_centre = np.mean(parent_branch[:, :3])
                    closest_point_index = np.argmin(np.linalg.norm(parent_centre - current_branch[:, :3]))
                    closest_point_of_current_branch = current_branch[closest_point_index]
                    kdtree = spatial.cKDTree(parent_branch[:, :3])
                    parent_points_in_range = parent_branch[
                        kdtree.query_ball_point(closest_point_of_current_branch[:3], r=max_search_radius)]
                    lowest_point_of_current_branch = current_branch[np.argmin(current_branch[:, 2])]
                    if parent_points_in_range.shape[0] > 0:
                        angles = MeasureTree.compute_angle(lowest_point_of_current_branch[3:6],
                                                           lowest_point_of_current_branch[:3] - parent_points_in_range[:, :3])
                        angles = angles[angles <= max_search_angle]

                        if angles.shape[0] > 0:
                            best_parent_point = parent_points_in_range[np.argmin(angles)]
                            # Interpolates from lowest point of current branch to smallest angle parent point.
                            interpolated_full_cyl_array = np.vstack((interpolated_full_cyl_array, self.interpolate_cyl(
                                    lowest_point_of_current_branch, best_parent_point,
                                    resolution=self.slice_increment)))
            current_tree = self.get_heights_above_DTM(current_tree)
            lowest_10_measured_tree_points = deepcopy(current_tree[np.argsort(current_tree[:, -1])][:10])
            lowest_measured_tree_point = np.median(lowest_10_measured_tree_points, axis=0)
            tree_base_point = deepcopy(current_tree[np.argmin(current_tree[:, self.cyl_dict['height_above_dtm']])])
            tree_base_point[2] = tree_base_point[2] - tree_base_point[self.cyl_dict['height_above_dtm']]

            interpolated_to_ground = self.interpolate_cyl(lowest_measured_tree_point, tree_base_point,
                                                          resolution=self.slice_increment)
            interpolated_full_cyl_array = np.vstack((interpolated_full_cyl_array, interpolated_to_ground))

        v1 = interpolated_full_cyl_array[:, 3:6]
        v2 = np.vstack((interpolated_full_cyl_array[:, 3],
                        interpolated_full_cyl_array[:, 4],
                        np.zeros((interpolated_full_cyl_array.shape[0])))).T
        interpolated_full_cyl_array[:, self.cyl_dict['segment_angle_to_horiz']] = self.compute_angle(v1, v2)
        interpolated_full_cyl_array = self.get_heights_above_DTM(interpolated_full_cyl_array)

        print("Cylinder Cleaning...")

        input_data = []
        i = 0
        tree_id_list = np.unique(interpolated_full_cyl_array[:, self.cyl_dict['tree_id']])
        max_tree_id = int(np.max(tree_id_list))
        for tree_id in tree_id_list:
            if tree_id % 10 == 0:
                print('\r', tree_id, '/', max_tree_id, end='')
            i += 1
            single_tree = interpolated_full_cyl_array[
                interpolated_full_cyl_array[:, self.cyl_dict['tree_id']] == tree_id]
            if single_tree.shape[0] > 0:
                single_tree = self.fix_outliers(single_tree)
                input_data.append([single_tree, self.parameters['cleaned_measurement_radius'], self.cyl_dict])

        print('\r', max_tree_id, '/', max_tree_id, end='')
        print('\nDone\n')

        print("Starting multithreaded cylinder cleaning...")
        j = 0
        max_j = len(input_data)

        cleaned_cyls_list = []
        with get_context("spawn").Pool(processes=self.num_procs) as pool:
            for i in pool.imap_unordered(MeasureTree.cylinder_cleaning_multithreaded, input_data):
                cleaned_cyls_list.append(i)
                if j % 11 == 0:
                    print('\r', j, '/', max_j, end='')
                j += 1
        cleaned_cyls = np.vstack(cleaned_cyls_list)

        del cleaned_cyls_list
        print('\r', max_j, '/', max_j, end='')
        print('\nDone\n')

        save_file(self.output_dir + 'cleaned_cyls.las', cleaned_cyls,
                  headers_of_interest=list(self.cyl_dict))

        max_angle = 30
        self.vegetation_points = self.get_heights_above_DTM(self.vegetation_points)
        self.ground_veg = self.vegetation_points[self.vegetation_points[:, self.veg_dict['height_above_dtm']] <= self.parameters['ground_veg_cutoff_height']]
        save_file(self.output_dir + 'ground_veg.las', self.ground_veg, headers_of_interest=list(self.veg_dict))

        self.vegetation_points = self.vegetation_points[self.vegetation_points[:, self.veg_dict['height_above_dtm']] > self.parameters['ground_veg_cutoff_height']]
        self.assigned_vegetation_points = np.zeros((0, self.vegetation_points.shape[1]))
        self.subsampled_sorted_veg = np.zeros((0, self.vegetation_points.shape[1]))
        self.vegetation_points_subsampled = subsample_point_cloud(self.vegetation_points, min_spacing=0.1, num_procs=self.parameters['num_procs'])
        stem_kdtree = spatial.cKDTree(cleaned_cyls[:, :2], leafsize=1000)
        results = stem_kdtree.query_ball_point(self.vegetation_points_subsampled[:, :2], r=self.parameters['veg_sorting_range'])

        i = 0
        print("Sorting vegetation...")
        for result in results:
            veg_point = self.vegetation_points_subsampled[i, :]
            nearby_cyls = cleaned_cyls[result]
            vector_array_1 = nearby_cyls[:, :3] - veg_point[:3]
            distances_2d = np.atleast_2d(np.linalg.norm(vector_array_1[:, :2], axis=1)).T
            vector_array_1 = vector_array_1/np.atleast_2d(np.linalg.norm(vector_array_1, axis=1)).T
            vector_array_2 = nearby_cyls[:, 3:6]

            vector_array_1[vector_array_1[:, 2] < 0, 2] = vector_array_1[vector_array_1[:, 2] < 0, 2] * -1
            vector_array_2[vector_array_2[:, 2] < 0, 2] = vector_array_2[vector_array_2[:, 2] < 0, 2] * -1
            angle_bool = self.compute_angle(vector_array_1, vector_array_2) <= max_angle

            if np.sum(angle_bool) != 0:
                best_tree_id = nearby_cyls[angle_bool][np.argmin(distances_2d[angle_bool]), self.cyl_dict['tree_id']]
                veg_point[self.veg_dict["tree_id"]] = best_tree_id
                self.subsampled_sorted_veg = np.vstack((self.subsampled_sorted_veg, np.atleast_2d(veg_point)))
            i += 1

        if self.subsampled_sorted_veg.shape[0] < 2:
            max_angle = 60
            i = 0
            print("Sorting vegetation...")
            for result in results:
                veg_point = self.vegetation_points_subsampled[i, :]
                nearby_cyls = cleaned_cyls[result]
                vector_array_1 = nearby_cyls[:, :3] - veg_point[:3]
                distances_2d = np.atleast_2d(np.linalg.norm(vector_array_1[:, :2], axis=1)).T
                vector_array_1 = vector_array_1 / np.atleast_2d(np.linalg.norm(vector_array_1, axis=1)).T
                vector_array_2 = nearby_cyls[:, 3:6]

                vector_array_1[vector_array_1[:, 2] < 0, 2] = vector_array_1[vector_array_1[:, 2] < 0, 2] * -1
                vector_array_2[vector_array_2[:, 2] < 0, 2] = vector_array_2[vector_array_2[:, 2] < 0, 2] * -1
                angle_bool = self.compute_angle(vector_array_1, vector_array_2) <= max_angle

                if np.sum(angle_bool) != 0:
                    best_tree_id = nearby_cyls[angle_bool][
                        np.argmin(distances_2d[angle_bool]), self.cyl_dict['tree_id']]
                    veg_point[self.veg_dict["tree_id"]] = best_tree_id
                    self.subsampled_sorted_veg = np.vstack((self.subsampled_sorted_veg, np.atleast_2d(veg_point)))
                i += 1

        if self.subsampled_sorted_veg.shape[0] >= 2:
            neighbours = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', metric='euclidean', radius=0.15).fit(self.subsampled_sorted_veg[:, :3])
            _, indices = neighbours.kneighbors(self.vegetation_points[:, :3])
            self.assigned_vegetation_points = self.vegetation_points
            print(self.assigned_vegetation_points.shape, self.subsampled_sorted_veg.shape)
            print(self.assigned_vegetation_points[:, self.veg_dict['tree_id']].shape, self.subsampled_sorted_veg[indices, self.veg_dict['tree_id']].shape)
            self.assigned_vegetation_points[:, self.veg_dict['tree_id']] = np.atleast_2d(self.subsampled_sorted_veg[indices, self.veg_dict['tree_id']]).T
            self.unassigned_vegetation_points = self.assigned_vegetation_points[self.assigned_vegetation_points[:, self.veg_dict['tree_id']] == 0]


        print("Measuring canopy gap fraction...")
        veg_kdtree = spatial.cKDTree(self.vegetation_points[:, :2], leafsize=10000)
        ground_veg_kdtree = spatial.cKDTree(self.ground_veg[:, :2], leafsize=10000)
        xmin = np.floor(np.min(self.terrain_points[:, 0]))
        ymin = np.floor(np.min(self.terrain_points[:, 1]))
        xmax = np.ceil(np.max(self.terrain_points[:, 0]))
        ymax = np.ceil(np.max(self.terrain_points[:, 1]))
        x_points = np.linspace(xmin, xmax, int(np.ceil((xmax - xmin) / self.parameters['Vegetation_coverage_resolution'])) + 1)
        y_points = np.linspace(ymin, ymax, int(np.ceil((ymax - ymin) / self.parameters['Vegetation_coverage_resolution'])) + 1)

        convexhull = spatial.ConvexHull(self.DTM[:, :2])
        self.canopy_density = np.zeros((0, 4))
        self.ground_area = 0
        self.canopy_area = 0
        self.ground_veg_area = 0
        for x in x_points:
            for y in y_points:
                if self.inside_conv_hull(np.array([x, y]), convexhull):
                    indices = veg_kdtree.query_ball_point([x, y], r=self.parameters['Vegetation_coverage_resolution'], p=10)
                    ground_veg_indices = ground_veg_kdtree.query_ball_point([x, y], r=self.parameters['Vegetation_coverage_resolution'], p=10)

                    self.ground_area += 1
                    if len(indices) > 5:
                        self.canopy_area += 1

                    if len(ground_veg_indices) > 5:
                        self.ground_veg_area += 1
                        self.canopy_density = np.vstack((self.canopy_density, np.array([[x, y, 0, len(indices)]])))

        print(self.canopy_area, self.ground_area, "Canopy Gap Fraction:", self.canopy_area / self.ground_area)
        np.savetxt(self.output_dir + 'canopy_density.csv', self.canopy_density)

        stem_points_sorted = np.zeros((0, len(list(self.stem_dict))))
        veg_points_sorted = np.zeros((0, len(list(self.veg_dict))))
        tree_kd_tree = spatial.cKDTree(self.stem_points[:, :3])
        canopy_density_kd_tree = spatial.cKDTree(self.canopy_density[:, :2])

        tree_data_dict = dict(Site=0, PlotID=1, treeNo=2, x_tree_base=3, y_tree_base=4, z_tree_base=5, DBH=6, Height=7,
                              Volume=8, Crown_mean_x=9, Crown_mean_y=10, Crown_top_x=11, Crown_top_y=12, Crown_top_z=13,
                              mean_understory_height_in_5m_radius=14)

        tree_data = np.zeros((0, 15))
        intelligent_plot_cropping = False
        plot_centre = np.loadtxt(self.output_dir + 'plot_centre_coords.csv')

        if self.parameters['plot_radius'] != 0 and self.parameters['plot_radius_buffer'] != 0:
            print("Using intelligent plot cropping mode...")
            intelligent_plot_cropping = True

        cleaned_cylinders = np.zeros((0, cleaned_cyls.shape[1]))
        for tree_id in np.unique(cleaned_cyls[:, self.cyl_dict['tree_id']]):
            tree = cleaned_cyls[cleaned_cyls[:, self.cyl_dict['tree_id']] == tree_id]
            tree_vegetation = self.assigned_vegetation_points[self.assigned_vegetation_points[:, self.veg_dict['tree_id']] == tree_id]
            combined = np.vstack((tree[:, :3], tree_vegetation[:, :3]))
            combined = np.hstack((combined, np.zeros((combined.shape[0], 1))))
            combined = self.get_heights_above_DTM(combined)

            # Get highest point of tree. Note, there is usually noise, so we use the 95th percentile.
            tree_max_point = combined[abs(combined[:, 2] - np.percentile(combined[:, 2], 95, interpolation='nearest')).argmin()]

            tree_base_point = deepcopy(combined[np.argmin(combined[:, -1])])
            z_tree_base = tree_base_point[2] - tree_base_point[-1]

            tree_mean_position = np.mean(combined[:, :2], axis=0)
            tree_height = tree_max_point[-1]
            del combined

            radii = 2 * tree[:, self.cyl_dict['radius']]
            results = tree_kd_tree.query_ball_point(tree[:, :3], r=radii)
            results = np.unique(np.hstack(results))
            tree_points = self.stem_points[np.asarray(results, dtype='int')]

            tree_points = np.hstack((tree_points, np.zeros((tree_points.shape[0], 2))))
            tree_points = self.get_heights_above_DTM(tree_points)
            tree_points[:, self.stem_dict['tree_id']] = tree_id
            base_northing = tree[np.argmin(tree[:, 2]), 0]
            base_easting = tree[np.argmin(tree[:, 2]), 1]
            DBH_slice = tree[np.logical_and(tree[:, self.cyl_dict['height_above_dtm']] >= 1.0,
                                            tree[:, self.cyl_dict['height_above_dtm']] <= 1.6)]
            DBH = 0
            DBH_X = 0
            DBH_Y = 0
            DBH_Z = 0
            if DBH_slice.shape[0] > 0:
                DBH = np.around(np.mean(DBH_slice[:, self.cyl_dict['radius']]) * 2, 3)
                DBH_X, DBH_Y, DBH_Z = np.mean(DBH_slice[:, :3], axis=0)
                mean_CCI_at_BH = np.mean(DBH_slice[:, self.cyl_dict['CCI']])
            volume = np.sum((np.pi * (tree[:, self.cyl_dict['radius']] ** 2)) * np.ceil(
                    self.parameters['cleaned_measurement_radius'] * 10) / 10)
            x_tree_base = base_northing
            y_tree_base = base_easting
            mean_vegetation_density_in_5m_radius = 0
            mean_understory_height_in_5m_radius = 0
            canopy_density_points = self.canopy_density[canopy_density_kd_tree.query_ball_point([DBH_X, DBH_Y], r=5)]
            nearby_understory_points = self.ground_veg[ground_veg_kdtree.query_ball_point([DBH_X, DBH_Y], r=5)]
            if canopy_density_points.shape[0] > 0:
                mean_vegetation_density_in_5m_radius = np.around(np.nanmean(canopy_density_points[:, 2]), 2)
            if nearby_understory_points.shape[0] > 0:
                mean_understory_height_in_5m_radius = np.around(np.nanmean(nearby_understory_points[:, self.veg_dict['height_above_dtm']]), 2)
            if tree.shape[0] > 0:
                description = 'Tree ' + str(int(tree_id))
                description = description + '\nDBH: ' + str(DBH) + ' m'
                description = description + '\nVolume: ' + str(np.around(volume, 3)) + ' m^3'
                description = description + '\nHeight: ' + str(np.around(tree_height, 3)) + ' m'
                description = description + '\nMean Veg Density (5 m radius): ' + str(
                        mean_vegetation_density_in_5m_radius) + ' units'
                description = description + '\nMean Understory Height (5 m radius): ' + str(
                        mean_understory_height_in_5m_radius) + ' m'

                print(description)
                this_trees_data = np.zeros((1, tree_data.shape[1]), dtype='object')
                this_trees_data[:, tree_data_dict['Site']] = self.parameters['Site']
                this_trees_data[:, tree_data_dict['PlotID']] = self.parameters['PlotID']
                this_trees_data[:, tree_data_dict['treeNo']] = int(tree_id)
                this_trees_data[:, tree_data_dict['x_tree_base']] = x_tree_base
                this_trees_data[:, tree_data_dict['y_tree_base']] = y_tree_base
                this_trees_data[:, tree_data_dict['z_tree_base']] = z_tree_base
                this_trees_data[:, tree_data_dict['DBH']] = DBH
                this_trees_data[:, tree_data_dict['Height']] = tree_height
                this_trees_data[:, tree_data_dict['Volume']] = volume
                this_trees_data[:, tree_data_dict['Crown_mean_x']] = tree_mean_position[0]
                this_trees_data[:, tree_data_dict['Crown_mean_y']] = tree_mean_position[1]
                this_trees_data[:, tree_data_dict['Crown_top_x']] = tree_max_point[0]
                this_trees_data[:, tree_data_dict['Crown_top_y']] = tree_max_point[1]
                this_trees_data[:, tree_data_dict['Crown_top_z']] = tree_max_point[2]
                this_trees_data[:, tree_data_dict['mean_understory_height_in_5m_radius']] = mean_understory_height_in_5m_radius

                text_size = 0.00256
                line_height = 0.025
                if DBH_X != 0 and DBH_Y != 0 and DBH_Z != 0 and x_tree_base != 0 and y_tree_base != 0:
                    line0 = self.point_cloud_annotations(text_size, DBH_X, DBH_Y + line_height, DBH_Z + line_height,
                                                         DBH * 0.5, '            DIAM: ' + str(np.around(DBH, 2)) + 'm')
                    line1 = self.point_cloud_annotations(text_size, DBH_X, DBH_Y, DBH_Z, DBH * 0.5,
                                                         '       CCI AT BH: ' + str(np.around(mean_CCI_at_BH, 2)))
                    line2 = self.point_cloud_annotations(text_size, DBH_X, DBH_Y - 2 * line_height,
                                                         DBH_Z - 2 * line_height, DBH * 0.5,
                                                         '          HEIGHT: ' + str(np.around(tree_height, 2)) + 'm')
                    line3 = self.point_cloud_annotations(text_size, DBH_X, DBH_Y - 3 * line_height,
                                                         DBH_Z - 3 * line_height, DBH * 0.5,
                                                         '          VOLUME: ' + str(np.around(volume, 2)) + 'm')
                    line4 = self.point_cloud_annotations(text_size, DBH_X, DBH_Y - 4 * line_height,
                                                         DBH_Z - 4 * line_height, DBH * 0.5, '    CHECK VOLUME: ' + str(
                            np.around((np.pi * (0.5 * DBH) ** 2) * tree_height, 2)) + 'm')  # TODO RECHECK and delete.

                    height_measurement_line = self.points_along_line(x_tree_base, y_tree_base, z_tree_base, x_tree_base,
                                                                     y_tree_base, z_tree_base + tree_height,
                                                                     resolution=0.025)

                    dbh_circle_points = self.create_3d_circles_as_points_flat(DBH_X, DBH_Y, DBH_Z, DBH / 2,
                                                                              circle_points=100)

                    if intelligent_plot_cropping:
                        if np.linalg.norm(np.array([x_tree_base, y_tree_base]) - np.array(plot_centre)) < self.parameters['plot_radius']:
                            tree_data = np.vstack((tree_data, this_trees_data))
                            stem_points_sorted = np.vstack((stem_points_sorted, tree_points))
                            veg_points_sorted = np.vstack((veg_points_sorted, tree_vegetation))
                            cleaned_cylinders = np.vstack((cleaned_cylinders, tree))
                            self.text_point_cloud = np.vstack((self.text_point_cloud, line0, line1, line2, line3, line4,
                                                               height_measurement_line, dbh_circle_points))

                    else:
                        tree_data = np.vstack((tree_data, this_trees_data))
                        stem_points_sorted = np.vstack((stem_points_sorted, tree_points))
                        veg_points_sorted = np.vstack((veg_points_sorted, tree_vegetation))
                        cleaned_cylinders = np.vstack((cleaned_cylinders, tree))
                        self.text_point_cloud = np.vstack((self.text_point_cloud, line0, line1, line2, line3, line4,
                                                           height_measurement_line, dbh_circle_points))

        save_file(self.output_dir + 'text_point_cloud.las', self.text_point_cloud)
        save_file(self.output_dir + 'stem_points_sorted.las', stem_points_sorted, headers_of_interest=['x', 'y', 'z', 'red', 'green', 'blue', 'tree_id', 'height_above_DTM'])
        save_file(self.output_dir + 'veg_points_sorted.las', veg_points_sorted, headers_of_interest=['x', 'y', 'z', 'red', 'green', 'blue', 'tree_id', 'height_above_DTM'])

        if 1:
            print("Making cleaned cylinder visualisation...")
            j = 0
            cleaned_cyl_vis = []
            max_j = np.shape(cleaned_cylinders)[0]
            # interpolated_cyl_vis = np.zeros((0,9))
            with get_context("spawn").Pool(processes=self.num_procs) as pool:
                for i in pool.imap_unordered(self.make_cyl_visualisation, cleaned_cylinders):
                    # interpolated_cyl_vis = np.vstack((interpolated_cyl_vis,i))
                    cleaned_cyl_vis.append(i)
                    if j % 100 == 0:
                        print('\r', j, '/', max_j, end='')
                    j += 1
            cleaned_cyl_vis = np.vstack(cleaned_cyl_vis)
            print('\r', max_j, '/', max_j, end='')
            print('\nDone\n')

            print("\nSaving cylinder visualisation...")
            save_file(self.output_dir + 'cleaned_cyl_vis.las', cleaned_cyl_vis,
                      headers_of_interest=list(self.cyl_dict))

        if intelligent_plot_cropping:
            self.terrain_points = self.terrain_points[np.linalg.norm(self.terrain_points[:, :2]-plot_centre, axis=1) < self.parameters['plot_radius']]
            self.cwd_points = self.cwd_points[np.linalg.norm(self.cwd_points[:, :2]-plot_centre, axis=1) < self.parameters['plot_radius']]
            self.ground_veg = self.ground_veg[np.linalg.norm(self.ground_veg[:, :2]-plot_centre, axis=1) < self.parameters['plot_radius']]

            self.DTM = self.DTM[np.linalg.norm(self.DTM[:, :2] - plot_centre, axis=1) < self.parameters['plot_radius']]
            save_file(self.output_dir + 'DTM.las', self.DTM)
            self.ground_veg = self.ground_veg[:, [self.veg_dict['x'],
                                                  self.veg_dict['y'],
                                                  self.veg_dict['z'],
                                                  self.veg_dict['red'],
                                                  self.veg_dict['green'],
                                                  self.veg_dict['blue']]]

            intelligently_cropped_point_cloud = np.vstack((self.terrain_points, self.cwd_points, self.ground_veg))  # , stem_points_sorted, veg_points_sorted))
            save_file(self.output_dir + 'intelligently_cropped_point_cloud.las', intelligently_cropped_point_cloud, headers_of_interest=['x', 'y', 'z', 'red', 'green', 'blue'])

        tree_data = pd.DataFrame(tree_data, columns=[i for i in tree_data_dict])
        tree_data.to_csv(self.output_dir + 'tree_data.csv', index=None, sep=',')

        hull = ConvexHull(self.DTM[:, :2])
        self.plot_area = np.around(hull.volume / 10000, 3)
        plane = Plane.best_fit(self.DTM)
        avg_gradient = self.compute_angle(plane.normal, [0, 0, 1])
        avg_gradient_north = self.compute_angle(plane.normal[[0, 2]], [0, 1])
        avg_gradient_east = self.compute_angle(plane.normal[[1, 2]], [0, 1])
        self.measure_time_end = time.time()
        self.measure_total_time = self.measure_time_end - self.measure_time_start

        try:
            processing_report = pd.read_csv(self.output_dir + 'processing_report.csv', index_col=None)
            processing_report['Measurement Time (s)'] = self.measure_total_time
            processing_report['Plot Area'] = self.plot_area
            processing_report['Num Trees in Plot'] = tree_data.shape[0]
            processing_report['Stems/ha'] = round(tree_data.shape[0] / self.plot_area)

            processing_report['Mean DBH'] = np.mean(tree_data['DBH'])
            processing_report['Median DBH'] = np.median(tree_data['DBH'])
            processing_report['Min DBH'] = np.min(tree_data['DBH'])
            processing_report['Max DBH'] = np.max(tree_data['DBH'])

            processing_report['Mean Height'] = np.mean(tree_data['Height'])
            processing_report['Median Height'] = np.median(tree_data['Height'])
            processing_report['Min Height'] = np.min(tree_data['Height'])
            processing_report['Max Height'] = np.max(tree_data['Height'])

            processing_report['Mean Volume'] = np.mean(tree_data['Volume'])
            processing_report['Median Volume'] = np.median(tree_data['Volume'])
            processing_report['Min Volume'] = np.min(tree_data['Volume'])
            processing_report['Max Volume'] = np.max(tree_data['Volume'])

            processing_report['Avg Gradient'] = avg_gradient
            processing_report['Avg Gradient North'] = avg_gradient_north
            processing_report['Avg Gradient East'] = avg_gradient_east

            processing_report['Canopy Gap Fraction'] = self.canopy_area / self.ground_area
            processing_report['Understory Veg Coverage Fraction'] = float(self.ground_veg_area) / float(self.ground_area)
            processing_report['CWD Coverage Fraction'] = self.measure_total_time

            processing_report.to_csv(self.output_dir + 'processing_report.csv', index=False)
            print("Measuring plot took", self.measure_total_time, 's')
        except FileNotFoundError:
            print("processing_report.csv file not found.")
        print("Measuring plot done.")
