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
from scipy import spatial  # TODO Test if sklearn kdtree is faster.
from scipy.spatial import ConvexHull
from scipy.interpolate import CubicSpline
from scipy.interpolate import griddata
from skimage.measure import LineModelND, CircleModel, ransac
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from tools import load_file, save_file, low_resolution_hack_mode, subsample_point_cloud, clustering, cluster_hdbscan, cluster_dbscan, get_heights_above_DTM
import time
import hdbscan
from skspatial.objects import Plane
import warnings


class MeasureTree:
    def __init__(self, parameters):
        self.measure_time_start = time.time()
        self.parameters = parameters
        self.filename = self.parameters['point_cloud_filename'].replace('\\', '/')
        self.output_dir = os.path.dirname(os.path.realpath(self.filename)).replace('\\', '/') + '/' + self.filename.split('/')[-1][:-4] + '_FSCT_output/'
        self.filename = self.filename.split('/')[-1]

        self.num_procs = parameters['num_procs']
        self.num_neighbours = parameters['num_neighbours']
        self.slice_thickness = parameters['slice_thickness']
        self.slice_increment = parameters['slice_increment']

        self.processing_report = pd.read_csv(self.output_dir + 'processing_report.csv', index_col=False)
        self.parameters['plot_radius'] = float(self.processing_report['Plot Radius'])
        self.parameters['plot_radius_buffer'] = float(self.processing_report['Plot Radius Buffer'])

        self.stem_points, headers_of_interest = load_file(self.output_dir + 'stem_points.las', headers_of_interest=['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'height_above_DTM'])
        self.stem_points = np.hstack((self.stem_points, np.zeros((self.stem_points.shape[0], 1))))

        print('stempoints', headers_of_interest)
        if self.parameters['low_resolution_point_cloud_hack_mode']:
            self.stem_points = low_resolution_hack_mode(self.stem_points,
                                                        self.parameters['low_resolution_point_cloud_hack_mode'],
                                                        self.parameters['subsampling_min_spacing'],
                                                        self.parameters['num_procs'])
            save_file(self.output_dir + self.filename[:-4] + '_stem_points_hack_mode_cloud.las', self.stem_points)

        self.DTM, headers_of_interest = load_file(self.output_dir + 'DTM.las')
        self.characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'dot', 'm', 'space', '_', '-', 'semiC',
                           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', '_M', 'N', 'O', 'P', 'Q', 'R',
                           'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        self.character_viz = []

        for i in self.characters:
            self.character_viz.append(np.genfromtxt('../tools/numbers/' + i + '.csv', delimiter=','))

        self.cyl_dict = dict(x=0, y=1, z=2, nx=3, ny=4, nz=5, radius=6, CCI=7, branch_id=8, parent_branch_id=9,
                             tree_id=10, segment_volume=11, segment_angle_to_horiz=12, height_above_dtm=13)

        self.veg_dict = dict(x=0, y=1, z=2, red=3, green=4, blue=5, label=6, height_above_dtm=7, tree_id=8)
        self.stem_dict = dict(x=0, y=1, z=2, red=3, green=4, blue=5, label=6, height_above_dtm=7, tree_id=8)
        self.tree_data_dict = dict(Site=0, PlotID=1, treeNo=2, x_tree_base=3, y_tree_base=4, z_tree_base=5, DBH=6,
                                   Height=7,
                                   Volume=8, Crown_mean_x=9, Crown_mean_y=10, Crown_top_x=11, Crown_top_y=12,
                                   Crown_top_z=13,
                                   mean_understory_height_in_5m_radius=14)
        self.terrain_points, headers_of_interest = load_file(self.output_dir + 'terrain_points.las',
                                                             headers_of_interest=['x', 'y', 'z', 'red', 'green', 'blue',
                                                                                  'label', 'height_above_DTM'])
        self.terrain_points = np.hstack((self.terrain_points, np.zeros((self.terrain_points.shape[0], 1))))

        self.vegetation_points, headers_of_interest = load_file(self.output_dir + 'vegetation_points.las',
                                                                headers_of_interest=['x', 'y', 'z', 'red', 'green',
                                                                                     'blue', 'label',
                                                                                     'height_above_DTM'])
        self.vegetation_points = np.hstack((self.vegetation_points, np.zeros((self.vegetation_points.shape[0], 1))))

        # Remove understorey vegetation and save it.
        ground_veg_mask = self.vegetation_points[:, self.veg_dict['height_above_dtm']] <= self.parameters[
            'ground_veg_cutoff_height']
        self.ground_veg = self.vegetation_points[ground_veg_mask]
        save_file(self.output_dir + 'ground_veg.las', self.ground_veg, headers_of_interest=list(self.veg_dict))
        self.vegetation_points = self.vegetation_points[np.logical_not(ground_veg_mask)]
        print("Measuring canopy gap fraction...")
        veg_kdtree = spatial.cKDTree(self.vegetation_points[:, :2], leafsize=10000)
        self.ground_veg_kdtree = spatial.cKDTree(self.ground_veg[:, :2], leafsize=10000)
        xmin = np.floor(np.min(self.terrain_points[:, 0]))
        ymin = np.floor(np.min(self.terrain_points[:, 1]))
        xmax = np.ceil(np.max(self.terrain_points[:, 0]))
        ymax = np.ceil(np.max(self.terrain_points[:, 1]))
        x_points = np.linspace(xmin, xmax,
                               int(np.ceil((xmax - xmin) / self.parameters['Vegetation_coverage_resolution'])) + 1)
        y_points = np.linspace(ymin, ymax,
                               int(np.ceil((ymax - ymin) / self.parameters['Vegetation_coverage_resolution'])) + 1)

        convexhull = spatial.ConvexHull(self.DTM[:, :2])
        self.ground_area = 0
        self.canopy_area = 0
        self.ground_veg_area = 0
        for x in x_points:
            for y in y_points:
                if self.inside_conv_hull(np.array([x, y]), convexhull):
                    indices = veg_kdtree.query_ball_point([x, y],
                                                          r=self.parameters['Vegetation_coverage_resolution'], p=10)
                    ground_veg_indices = self.ground_veg_kdtree.query_ball_point([x, y], r=self.parameters[
                        'Vegetation_coverage_resolution'], p=10)

                    self.ground_area += 1
                    if len(indices) > 5:
                        self.canopy_area += 1

                    if len(ground_veg_indices) > 5:
                        self.ground_veg_area += 1

        print(self.canopy_area, self.ground_area, "Canopy Gap Fraction:", self.canopy_area / self.ground_area)




        self.cwd_points, headers_of_interest = load_file(self.output_dir + 'cwd_points.las',
                                                         headers_of_interest=['x', 'y', 'z', 'red', 'green', 'blue',
                                                                              'label', 'height_above_DTM'])
        self.cwd_points = np.hstack((self.cwd_points, np.zeros((self.cwd_points.shape[0], 1))))

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
        radius_threshold = np.nanpercentile(tree_cylinders[:, self.cyl_dict['radius']], 70)
        doubtful_cyl_mask = np.logical_or(tree_cylinders[:, self.cyl_dict['radius']] >= radius_threshold,
                                          tree_cylinders[:, self.cyl_dict['radius']] >= self.parameters['maximum_stem_diameter'])
        if tree_cylinders[np.logical_not(doubtful_cyl_mask)].shape[0] > 1:
            neigh = NearestNeighbors(n_neighbors=2)
            neigh.fit(tree_cylinders[np.logical_not(doubtful_cyl_mask), :3])
            neighbour_list = neigh.kneighbors(tree_cylinders[doubtful_cyl_mask, :3], return_distance=False)
            tree_cylinders[doubtful_cyl_mask, self.cyl_dict['radius']] = [np.median(tree_cylinders[np.logical_not(doubtful_cyl_mask), self.cyl_dict['radius']][indices]) for indices in neighbour_list]
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

        def criteria_check(cyl1, cyl2, angle_tolerance, search_angle):
            """
            Decides if cyl2 should be joined to cyl1 and if they are the same tree.
            angle_tolerance is the maximum angle between normal vectors of cylinders to be considered the same branch.
            """
            vector_array = cyl2[:, :3] - np.atleast_2d(cyl1[:3])
            condition1 = within_angle_tolerance(cyl1[3:6], cyl2[:, 3:6], angle_tolerance)
            condition2 = within_angle_tolerance(cyl1[3:6], vector_array, search_angle)
            # condition3 = cyl2[:, self.cyl_dict['radius']] < cyl1[self.cyl_dict['radius']]*1.05
            # cyl2[np.logical_and(condition1, condition2, condition3), self.cyl_dict['tree_id']] = cyl1[self.cyl_dict['tree_id']]
            # cyl2[np.logical_and(condition1, condition2, condition3), self.cyl_dict['parent_branch_id']] = cyl1[self.cyl_dict['branch_id']]
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
            unsorted_points[results] = criteria_check(current_point,
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
    def rodrigues_rot(cls, points, vector1, vector2):
        """RODRIGUES ROTATION
        - Rotate given points based on a starting and ending vector
        - Axis k and angle of rotation theta given by vectors n0,n1
        P_rot = P*cos(theta) + (k x P)*sin(theta) + k*<k,P>*(1-cos(theta))"""
        if points.ndim == 1:
            points = points[np.newaxis, :]

        vector1 = vector1 / np.linalg.norm(vector1)
        vector2 = vector2 / np.linalg.norm(vector2)
        k = np.cross(vector1, vector2)
        if np.sum(k) != 0:
            k = k / np.linalg.norm(k)
        theta = np.arccos(np.dot(vector1, vector2))

        P_rot = np.zeros((len(points), 3))
        for i in range(len(points)):
            P_rot[i] = points[i] * np.cos(theta) + np.cross(k, points[i]) * np.sin(theta) + k * np.dot(k, points[i]) * (
                    1 - np.cos(theta))
        return P_rot

    @classmethod
    def fit_circle_3D(cls, points, V):
        """
        Fits a circle using Random Sample Consensus (RANSAC) to a set of points in a plane perpendicular to vector V.

        Args:
            points: Set of points to fit a circle to using RANSAC.
            V: Axial vector of the cylinder you're fitting.

        Returns:
            cyl_output: numpy array of the format [[x, y, z, x_norm, y_norm, z_norm, radius, CCI, 0, 0, 0, 0, 0, 0]]
        """

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
                                           residual_threshold=0.3, max_trials=2500)
            xc, yc = model_robust.params[0:2]
            r = model_robust.params[2]
            CCI = MeasureTree.circumferential_completeness_index([xc, yc], r, P_xy[:, :2])

        if CCI < 0.3:
            r = 0
            xc, yc = np.mean(P_xy[:, :2], axis=0)
            CCI = 0

        # Transform circle center back to 3D coords
        cyl_centre = MeasureTree.rodrigues_rot(np.array([[xc, yc, 0]]), [0, 0, 1], normal) + P_mean
        cyl_output = np.array([[cyl_centre[0, 0], cyl_centre[0, 1], cyl_centre[0, 2], normal[0], normal[1], normal[2],
                                r, CCI, 0, 0, 0, 0, 0, 0]])
        return cyl_output

    def point_cloud_annotations(self, character_size, xpos, ypos, zpos, offset, text):
        """
        Point based text visualisation. Makes text viewable as a point cloud.

        Args:
            character_size:
            xpos: x coord.
            ypos: y coord.
            zpos: z coord.
            offset: Offset for the x coord. Used to shift the text depending on tree radius.
            text: The text to be displayed.

        Returns:
            nx3 point cloud of the text.
        """
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

        points = points * character_size + [xpos + 0.2 + 0.5 * offset, ypos, zpos]
        return points

    @classmethod
    def fit_cylinder(cls, skeleton_points, point_cloud, num_neighbours):
        """
        Fits a 3D line to the skeleton points cluster provided.
        Uses this line as the major axis/axial vector of the cylinder to be fitted.
        Fits a series of circles perpendicular to this axis to the point cloud of this particular stem segment.

        Args:
            skeleton_points: A single cluster of skeleton points which should represent a segment of a tree/branch.
            point_cloud: The cluster of points belonging to the segment of the branch.
            num_neighbours: The number of skeleton points to use for fitting each circle in the segment. lower numbers
                            have fewer points to fit a circle to, but higher numbers are negatively affected by curved
                            branches. Recommend leaving this as it is.

        Returns:
            cyl_array: a numpy array based representation of the fitted circles/cylinders.
        """
        point_cloud = point_cloud[:, :3]
        skeleton_points = skeleton_points[:, :3]
        cyl_array = np.zeros((0, 14))
        line_centre = np.mean(skeleton_points[:, :3], axis=0)
        _, _, vh = np.linalg.svd(line_centre - skeleton_points)
        line_v_hat = vh[0] / np.linalg.norm(vh[0])

        if skeleton_points.shape[0] <= num_neighbours:
            group = skeleton_points
            line_centre = np.mean([np.min(group[:, :3], axis=0), np.min(group[:, :3], axis=0)], axis=0)
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
        """
        Cylinder Cleaning
        Works on a single tree worth of cylinders at a time.
        Starts at the lowest (z axis) cylinder.
        Finds neighbouring cylinders within "cleaned_measurement_radius".
        If no neighbours are found, cylinder is deleted.
        If neighbours are found, find the neighbour with the highest circumferential completeness index (CCI). This is
        probably the most trustworthy cylinder in the neighbourhood.

        If there are enough neighbours, use those with CCI >= the 30th percentile of CCIs in the neighbourhood.
        Use the medians of x, y, vx, vy, vz, radius as the cleaned cylinder values.
        Use the mean of the z coords of all neighbours for the cleaned cylinder z coord.
        """
        sorted_cylinders, cleaned_measurement_radius, cyl_dict = args
        cleaned_cyls = np.zeros((0, np.shape(sorted_cylinders)[1]))

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
                    best_cylinder = neighbours[np.argsort(neighbours[:, cyl_dict['CCI']])][-1]  # choose cyl with highest CCI.

                # compute 50th percentile of CCI in neighbourhood
                percentile_thresh = np.percentile(neighbours[:, cyl_dict['CCI']], 50)
                if neighbours[neighbours[:, cyl_dict['CCI']] >= percentile_thresh, :2].shape[0] > 0:
                    best_cylinder[:3] = np.median(neighbours[neighbours[:, cyl_dict['CCI']] >= percentile_thresh, :3], axis=0)
                    best_cylinder[3:6] = np.median(neighbours[neighbours[:, cyl_dict['CCI']] >= percentile_thresh, 3:6], axis=0)
                    best_cylinder[cyl_dict['radius']] = np.median(neighbours[neighbours[:, cyl_dict['CCI']] >= percentile_thresh, cyl_dict['radius']], axis=0)

            cleaned_cyls = np.vstack((cleaned_cyls, best_cylinder))
            sorted_cylinders = np.delete(sorted_cylinders, results, axis=0)

        return cleaned_cyls

    @staticmethod
    def inside_conv_hull(point, hull, tolerance=1e-5):
        """Checks if a point is inside a convex hull."""
        return all((np.dot(eq[:-1], point) + eq[-1] <= tolerance) for eq in hull.equations)

    @classmethod
    def circumferential_completeness_index(cls, fitted_circle_centre, estimated_radius, slice_points):
        """
        Computes the Circumferential Completeness Index (CCI) of a fitted circle.

        Args:
            fitted_circle_centre: x, y coords of the circle centre
            estimated_radius: circle radius
            slice_points: the points the circle was fitted to

        Returns:
            CCI
        """
        sector_angle = 9  # degrees
        num_sections = int(np.ceil(360 / sector_angle))
        sectors = np.linspace(-180, 180, num=num_sections, endpoint=False)

        centre_vectors = slice_points[:, :2] - fitted_circle_centre
        norms = np.linalg.norm(centre_vectors, axis=1)

        centre_vectors = centre_vectors / np.atleast_2d(norms).T
        centre_vectors = centre_vectors[np.logical_and(norms >= 0.7 * estimated_radius,
                                                       norms <= 1.3 * estimated_radius)]

        sector_vectors = np.vstack((np.cos(sectors), np.sin(sectors))).T
        CCI = np.sum([np.any(
            np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i', np.atleast_2d(sector_vector), centre_vectors), -1, 1)))
            < sector_angle / 2) for sector_vector in sector_vectors]) / num_sections

        return CCI

    @classmethod
    def threaded_cyl_fitting(cls, args):
        """Helper function for multithreaded cylinder fitting."""
        skel_cluster, point_cluster, cluster_id, num_neighbours, cyl_dict = args
        cyl_array = np.zeros((0, 14))
        if skel_cluster.shape[0] > num_neighbours:
            cyl_array = cls.fit_cylinder(skel_cluster, point_cluster, num_neighbours=num_neighbours)
            cyl_array[:, cyl_dict['branch_id']] = cluster_id
        return cyl_array

    @classmethod
    def slice_clustering(cls, new_slice):
        """Helper function for clustering stem slices and extracting the skeletons of these stems."""
        cluster_array_internal = np.zeros((0, 6))
        medians = np.zeros((0, 3))

        if new_slice.shape[0] > 1:
            new_slice = cluster_hdbscan(new_slice[:, :3])
            for cluster_id in range(0, int(np.max(new_slice[:, -1])) + 1):
                cluster = new_slice[new_slice[:, -1] == cluster_id]
                median = np.median(cluster[:, :3], axis=0)
                medians = np.vstack((medians, median))
                cluster_array_internal = np.vstack((cluster_array_internal, np.hstack((cluster[:, :3], np.zeros((cluster.shape[0], 3)) + median))))
        return cluster_array_internal, medians

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
        """Checks if the angle between vector1_2 and normal1 is less than search_angle."""
        norm1 = normal1 / np.linalg.norm(normal1)
        if not (vector1_2 == 0).all():
            norm2 = vector1_2 / np.linalg.norm(vector1_2)
            dot = np.clip(np.dot(norm1, norm2), -1, 1)
            theta = math.degrees(np.arccos(dot))
            return abs((theta > 90) * 180 - theta) <= search_angle
        else:
            return False

    def run_measurement_extraction(self):
        skeleton_array = np.zeros((0, 3))
        cluster_array = np.zeros((0, 6))
        slice_heights = np.linspace(np.min(self.stem_points[:, 2]), np.max(self.stem_points[:, 2]), int(np.ceil(
                (np.max(self.stem_points[:, 2]) - np.min(self.stem_points[:, 2])) / self.slice_increment)))

        print("Making and clustering slices...")
        i = 0
        max_i = slice_heights.shape[0]
        for slice_height in slice_heights:
            if i % 10 == 0:
                print('\r', i, '/', max_i, end='')
            i += 1
            new_slice = self.stem_points[np.logical_and(self.stem_points[:, 2] >= slice_height, self.stem_points[:, 2] < slice_height + self.slice_thickness)]
            if new_slice.shape[0] > 0:
                cluster, skel = MeasureTree.slice_clustering(new_slice)
                skeleton_array = np.vstack((skeleton_array, skel))
                cluster_array = np.vstack((cluster_array, cluster))
        print('\r', max_i, '/', max_i, end='')
        print('\nDone\n')

        print('Clustering skeleton...')
        skeleton_array = cluster_dbscan(skeleton_array[:, :3], eps=self.slice_increment * 1.5)
        skeleton_cluster_visualisation = np.zeros((0, 5))
        for k in np.unique(skeleton_array[:, -1]):  # Just assigns random colours to the clusters to make it easier to see different neighbouring groups.
            skeleton_cluster_visualisation = np.vstack((skeleton_cluster_visualisation, np.hstack((skeleton_array[skeleton_array[:, -1] == k], np.zeros((skeleton_array[skeleton_array[:, -1] == k].shape[0], 1)) + np.random.randint(0, 10)))))

        # print("Saving skeleton and cluster array...")
        # save_file(self.output_dir + 'skeleton_cluster_visualisation.las', skeleton_cluster_visualisation, ['X', 'Y', 'Z', 'cluster'])

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
                                                                    r=0.0001)))
            cluster_array_clean = cluster_array[results, :3]
            input_data.append([skel_cluster[:, :3], cluster_array_clean[:, :3], cluster_id, self.num_neighbours,
                               self.cyl_dict])

        print('\r', max_i, '/', max_i, end='')
        print('\nDone\n')

        print("Starting multithreaded cylinder fitting... This can take a while.")
        j = 0
        max_j = len(input_data)
        outputlist = []
        with get_context("spawn").Pool(processes=self.num_procs) as pool:
            for i in pool.imap_unordered(MeasureTree.threaded_cyl_fitting, input_data):
                outputlist.append(i)
                if j % 10 == 0:
                    print('\r', j, '/', max_j, end='')
                j += 1
        full_cyl_array = np.vstack(outputlist)
        print('\r', max_j, '/', max_j, end='')
        print('\nDone\n')

        print("Deleting cyls with CCI less than:", self.parameters['minimum_CCI'])
        full_cyl_array = full_cyl_array[full_cyl_array[:, self.cyl_dict['CCI']] >= self.parameters['minimum_CCI']]

        # cyl_array = [x,y,z,nx,ny,nz,r,CCI,branch_id,tree_id,segment_volume,parent_branch_id]
        print("Saving cylinder array...")
        # save_file(self.output_dir + 'full_cyl_array.las', full_cyl_array, headers_of_interest=list(self.cyl_dict))
        # full_cyl_array, _ = load_file(self.output_dir + 'full_cyl_array.las',
        #                               headers_of_interest=list(self.cyl_dict))

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

                if assigned is False and lowest_point_z < self.parameters['tree_base_cutoff_height']:
                    tree[:, self.cyl_dict['tree_id']] = t_id
                    sorted_full_cyl_array = np.vstack((sorted_full_cyl_array, tree))
                    t_id += 1

        # save_file(self.output_dir + 'sorted_full_cyl_array.las', sorted_full_cyl_array, headers_of_interest=list(self.cyl_dict))

        print("Cylinder interpolation...")

        tree_list = []
        interpolated_full_cyl_array = np.zeros((0, 14))
        max_tree_id = np.unique(sorted_full_cyl_array[:, self.cyl_dict['tree_id']]).shape[0]
        for tree_id in np.unique(sorted_full_cyl_array[:, self.cyl_dict['tree_id']]):
            if int(tree_id) % 10 == 0:
                print("Tree ID", int(tree_id), '/', int(max_tree_id))
            current_tree = sorted_full_cyl_array[sorted_full_cyl_array[:, self.cyl_dict['tree_id']] == tree_id]
            if current_tree.shape[0] >= self.parameters['min_tree_cyls']:
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
                current_tree = get_heights_above_DTM(current_tree, self.DTM)
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
        interpolated_full_cyl_array = get_heights_above_DTM(interpolated_full_cyl_array, self.DTM)

        # save_file(self.output_dir + 'interpolated_full_cyl_array.las', interpolated_full_cyl_array, headers_of_interest=list(self.cyl_dict))
        # interpolated_full_cyl_array, _ = load_file(self.output_dir + 'interpolated_full_cyl_array.las', headers_of_interest=list(self.cyl_dict))

        tree_data = np.zeros((0, 15))
        radial_tree_aware_plot_cropping = False
        square_tree_aware_plot_cropping = False
        plot_centre = np.loadtxt(self.output_dir + 'plot_centre_coords.csv')
        stem_points_sorted = np.zeros((0, len(list(self.stem_dict))))
        veg_points_sorted = np.zeros((0, len(list(self.veg_dict))))

        if self.parameters['plot_radius'] > 0 and self.parameters['plot_radius_buffer'] > 0:
            print("Using tree aware plot cropping mode...")
            radial_tree_aware_plot_cropping = True

        elif self.parameters['square_grid_slicing_size'] > 0 and self.parameters['grid_buffer_distance'] > 0:
            square_tree_aware_plot_cropping = True

        print("Cylinder Outlier Removal...")
        input_data = []
        i = 0
        tree_id_list = np.unique(interpolated_full_cyl_array[:, self.cyl_dict['tree_id']])

        if tree_id_list.shape[0] > 0:  # TODO Add Watershed canopy segmentation or similar for when no trees are detected in a plot.
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

            print("Starting multithreaded cylinder cleaning/smoothing...")
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
            cleaned_cylinders = np.zeros((0, cleaned_cyls.shape[1]))

            print("Sorting vegetation...")
            # Simple nearest neighbours vegetation sorting.
            kdtree = spatial.cKDTree(cleaned_cyls[:, :2], leafsize=1000)
            results = kdtree.query(self.vegetation_points[:, :2], k=1)
            mask = results[0] <= self.parameters['veg_sorting_range']
            self.vegetation_points = self.vegetation_points[mask]
            results = results[1][mask]
            self.vegetation_points[:, self.veg_dict['tree_id']] = cleaned_cyls[results, self.cyl_dict['tree_id']]

            if self.parameters['sort_stems'] or self.parameters['generate_output_point_cloud']:
                kdtree = spatial.cKDTree(cleaned_cyls[:, :2], leafsize=1000)
                results = kdtree.query(self.stem_points[:, :2], k=1)
                mask = results[0] <= self.parameters['veg_sorting_range']
                self.stem_points = self.stem_points[mask]
                results = results[1][mask]
                self.stem_points[:, self.stem_dict['tree_id']] = cleaned_cyls[results, self.cyl_dict['tree_id']]

            for tree_id in np.unique(cleaned_cyls[:, self.cyl_dict['tree_id']]):
                tree = cleaned_cyls[cleaned_cyls[:, self.cyl_dict['tree_id']] == tree_id]
                tree_vegetation = self.vegetation_points[self.vegetation_points[:, self.veg_dict['tree_id']] == tree_id]
                combined = np.vstack((tree[:, :3], tree_vegetation[:, :3]))
                combined = np.hstack((combined, np.zeros((combined.shape[0], 1))))
                combined = get_heights_above_DTM(combined, self.DTM)

                # Get highest point of tree. Note, there is usually noise, so we use the 98th percentile.
                tree_max_point = combined[abs(combined[:, 2] - np.percentile(combined[:, 2], 98, interpolation='nearest')).argmin()]

                tree_base_point = deepcopy(combined[np.argmin(combined[:, -1])])
                z_tree_base = tree_base_point[2] - tree_base_point[-1]

                tree_mean_position = np.mean(combined[:, :2], axis=0)
                tree_height = tree_max_point[-1]
                del combined

                if self.parameters['sort_stems'] or self.parameters['generate_output_point_cloud']:
                    tree_points = self.stem_points[self.stem_points[:, self.stem_dict['tree_id']] == tree_id]

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
                nearby_understory_points = self.ground_veg[self.ground_veg_kdtree.query_ball_point([DBH_X, DBH_Y], r=5)]

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
                    this_trees_data[:, self.tree_data_dict['Site']] = self.parameters['Site']
                    this_trees_data[:, self.tree_data_dict['PlotID']] = self.parameters['PlotID']
                    this_trees_data[:, self.tree_data_dict['treeNo']] = int(tree_id)
                    this_trees_data[:, self.tree_data_dict['x_tree_base']] = x_tree_base
                    this_trees_data[:, self.tree_data_dict['y_tree_base']] = y_tree_base
                    this_trees_data[:, self.tree_data_dict['z_tree_base']] = z_tree_base
                    this_trees_data[:, self.tree_data_dict['DBH']] = DBH
                    this_trees_data[:, self.tree_data_dict['Height']] = tree_height
                    this_trees_data[:, self.tree_data_dict['Volume']] = volume
                    this_trees_data[:, self.tree_data_dict['Crown_mean_x']] = tree_mean_position[0]
                    this_trees_data[:, self.tree_data_dict['Crown_mean_y']] = tree_mean_position[1]
                    this_trees_data[:, self.tree_data_dict['Crown_top_x']] = tree_max_point[0]
                    this_trees_data[:, self.tree_data_dict['Crown_top_y']] = tree_max_point[1]
                    this_trees_data[:, self.tree_data_dict['Crown_top_z']] = tree_max_point[2]
                    this_trees_data[:, self.tree_data_dict['mean_understory_height_in_5m_radius']] = mean_understory_height_in_5m_radius

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
                                                             '          VOLUME: ' + str(np.around(volume, 2)) + 'm3')

                        height_measurement_line = self.points_along_line(x_tree_base, y_tree_base, z_tree_base, x_tree_base,
                                                                         y_tree_base, z_tree_base + tree_height,
                                                                         resolution=0.025)

                        dbh_circle_points = self.create_3d_circles_as_points_flat(DBH_X, DBH_Y, DBH_Z, DBH / 2,
                                                                                  circle_points=100)

                        if radial_tree_aware_plot_cropping:
                            if np.linalg.norm(np.array([x_tree_base, y_tree_base]) - np.array(plot_centre)) < self.parameters['plot_radius']:
                                tree_data = np.vstack((tree_data, this_trees_data))
                                if self.parameters['sort_stems'] or self.parameters['generate_output_point_cloud']:
                                    stem_points_sorted = np.vstack((stem_points_sorted, tree_points))  # TODO make this separate loop as it will be faster.
                                veg_points_sorted = np.vstack((veg_points_sorted, tree_vegetation))
                                cleaned_cylinders = np.vstack((cleaned_cylinders, tree))
                                self.text_point_cloud = np.vstack((self.text_point_cloud, line0, line1, line2, line3,
                                                                   height_measurement_line, dbh_circle_points))
                        elif square_tree_aware_plot_cropping:
                            x_min_coord = plot_centre[0] - self.parameters['square_grid_slicing_size']/2
                            x_max_coord = plot_centre[0] + self.parameters['square_grid_slicing_size']/2
                            y_min_coord = plot_centre[1] - self.parameters['square_grid_slicing_size']/2
                            y_max_coord = plot_centre[1] + self.parameters['square_grid_slicing_size']/2

                            if x_tree_base >= x_min_coord and x_tree_base < x_max_coord and y_tree_base >= y_min_coord and y_tree_base < y_max_coord:
                                if self.parameters['sort_stems'] or self.parameters['generate_output_point_cloud']:
                                    stem_points_sorted = np.vstack((stem_points_sorted, tree_points))
                                tree_data = np.vstack((tree_data, this_trees_data))
                                veg_points_sorted = np.vstack((veg_points_sorted, tree_vegetation))
                                cleaned_cylinders = np.vstack((cleaned_cylinders, tree))
                                self.text_point_cloud = np.vstack((self.text_point_cloud, line0, line1, line2, line3,
                                                                   height_measurement_line, dbh_circle_points))
                        else:
                            tree_data = np.vstack((tree_data, this_trees_data))
                            if self.parameters['sort_stems'] or self.parameters['generate_output_point_cloud']:
                                stem_points_sorted = np.vstack((stem_points_sorted, tree_points))
                            veg_points_sorted = np.vstack((veg_points_sorted, tree_vegetation))
                            cleaned_cylinders = np.vstack((cleaned_cylinders, tree))
                            self.text_point_cloud = np.vstack((self.text_point_cloud, line0, line1, line2, line3,
                                                               height_measurement_line, dbh_circle_points))

            save_file(self.output_dir + 'text_point_cloud.las', self.text_point_cloud)
            if self.parameters['sort_stems'] or self.parameters['generate_output_point_cloud']:
                if not self.parameters['minimise_output_size_mode']:
                    save_file(self.output_dir + 'stem_points_sorted.las', stem_points_sorted, headers_of_interest=list(self.stem_dict))

            if not self.parameters['minimise_output_size_mode']:
                save_file(self.output_dir + 'veg_points_sorted.las', veg_points_sorted, headers_of_interest=list(self.veg_dict))

            if 1:
                print("Making cleaned cylinder visualisation...")
                j = 0
                cleaned_cyl_vis = []
                max_j = np.shape(cleaned_cylinders)[0]
                with get_context("spawn").Pool(processes=self.num_procs) as pool:
                    for i in pool.imap_unordered(self.make_cyl_visualisation, cleaned_cylinders):
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

        if radial_tree_aware_plot_cropping and self.parameters['generate_output_point_cloud']:
            self.terrain_points = self.terrain_points[np.linalg.norm(self.terrain_points[:, :2]-plot_centre, axis=1) < self.parameters['plot_radius']]
            self.cwd_points = self.cwd_points[np.linalg.norm(self.cwd_points[:, :2]-plot_centre, axis=1) < self.parameters['plot_radius']]
            self.ground_veg = self.ground_veg[np.linalg.norm(self.ground_veg[:, :2]-plot_centre, axis=1) < self.parameters['plot_radius']]

            self.DTM = self.DTM[np.linalg.norm(self.DTM[:, :2] - plot_centre, axis=1) < self.parameters['plot_radius']]
            save_file(self.output_dir + 'cropped_DTM.las', self.DTM)
            tree_aware_cropped_point_cloud = np.vstack((self.terrain_points, self.cwd_points, self.ground_veg, stem_points_sorted, veg_points_sorted))  # , stem_points_sorted, veg_points_sorted))

            save_file(self.output_dir + 'tree_aware_cropped_point_cloud.las', tree_aware_cropped_point_cloud, headers_of_interest=list(self.stem_dict))

        elif square_tree_aware_plot_cropping and self.parameters['generate_output_point_cloud']:
            x_min_coord = plot_centre[0] - self.parameters['square_grid_slicing_size'] / 2
            x_max_coord = plot_centre[0] + self.parameters['square_grid_slicing_size'] / 2
            y_min_coord = plot_centre[1] - self.parameters['square_grid_slicing_size'] / 2
            y_max_coord = plot_centre[1] + self.parameters['square_grid_slicing_size'] / 2

            self.terrain_points = self.terrain_points[np.logical_and(self.terrain_points[:, 0] >= x_min_coord, self.terrain_points[:, 0] < x_max_coord)]
            self.terrain_points = self.terrain_points[np.logical_and(self.terrain_points[:, 1] >= y_min_coord, self.terrain_points[:, 1] < y_max_coord)]

            self.cwd_points = self.cwd_points[np.logical_and(self.cwd_points[:, 0] >= x_min_coord, self.cwd_points[:, 0] < x_max_coord)]
            self.cwd_points = self.cwd_points[np.logical_and(self.cwd_points[:, 1] >= y_min_coord, self.cwd_points[:, 1] < y_max_coord)]

            self.ground_veg = self.ground_veg[np.logical_and(self.ground_veg[:, 0] >= x_min_coord, self.ground_veg[:, 0] < x_max_coord)]
            self.ground_veg = self.ground_veg[np.logical_and(self.ground_veg[:, 1] >= y_min_coord, self.ground_veg[:, 1] < y_max_coord)]

            self.DTM = self.DTM[np.logical_and(self.DTM[:, 0] >= x_min_coord, self.DTM[:, 0] < x_max_coord)]
            self.DTM = self.DTM[np.logical_and(self.DTM[:, 1] >= y_min_coord, self.DTM[:, 1] < y_max_coord)]

            save_file(self.output_dir + 'cropped_DTM.las', self.DTM)
            tree_aware_cropped_point_cloud = np.vstack((self.terrain_points, self.cwd_points, self.ground_veg, stem_points_sorted, veg_points_sorted))

            save_file(self.output_dir + 'tree_aware_cropped_point_cloud.las', tree_aware_cropped_point_cloud, headers_of_interest=list(self.stem_dict))

        elif self.parameters['generate_output_point_cloud']:
            tree_aware_cropped_point_cloud = np.vstack((self.terrain_points, self.cwd_points, self.ground_veg, stem_points_sorted, veg_points_sorted))
            save_file(self.output_dir + 'tree_aware_cropped_point_cloud.las', tree_aware_cropped_point_cloud, headers_of_interest=list(self.stem_dict))

        tree_data = pd.DataFrame(tree_data, columns=[i for i in self.tree_data_dict])
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
            if tree_data.shape[0] > 0:
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
                processing_report['Canopy Gap Fraction'] = self.canopy_area / self.ground_area

            else:
                processing_report['Mean DBH'] = 0
                processing_report['Median DBH'] = 0
                processing_report['Min DBH'] = 0
                processing_report['Max DBH'] = 0

                processing_report['Mean Height'] = 0
                processing_report['Median Height'] = 0
                processing_report['Min Height'] = 0
                processing_report['Max Height'] = 0

                processing_report['Mean Volume'] = 0
                processing_report['Median Volume'] = 0
                processing_report['Min Volume'] = 0
                processing_report['Max Volume'] = 0
                processing_report['Canopy Gap Fraction'] = 0

            processing_report['Avg Gradient'] = avg_gradient
            processing_report['Avg Gradient North'] = avg_gradient_north
            processing_report['Avg Gradient East'] = avg_gradient_east

            processing_report['Understory Veg Coverage Fraction'] = float(self.ground_veg_area) / float(self.ground_area)
            processing_report['CWD Coverage Fraction'] = self.measure_total_time

            processing_report.to_csv(self.output_dir + 'processing_report.csv', index=False)
            print("Measuring plot took", self.measure_total_time, 's')
        except FileNotFoundError:
            print("processing_report.csv file not found.")
        print("Measuring plot done.")
