import math
import sys
from copy import deepcopy
from multiprocessing import get_context
import networkx as nx
import numpy as np
import pandas as pd
import simplekml
import utm
from matplotlib import pyplot as plt
from scipy import spatial
from scipy.interpolate import CubicSpline
from scipy.interpolate import griddata
from skimage.measure import LineModelND, CircleModel, ransac
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

sys.setrecursionlimit(10 ** 6)


class MeasureTree:
    def __init__(self, parameters):
        self.parameters = parameters
        self.input_point_cloud = parameters['input_point_cloud'][:-4] + '_out'
        self.directory = parameters['directory']
        self.num_procs = parameters['num_procs']
        self.num_neighbours = parameters['num_neighbours']
        self.slice_thickness = parameters['slice_thickness']
        self.slice_increment = parameters['slice_increment']
        self.min_tree_volume = parameters['min_tree_volume']

        self.stem_points = np.array(pd.read_csv(self.directory + 'data/postprocessed_point_clouds/' + self.input_point_cloud + '/stem_points.csv', header=None, index_col=None, delim_whitespace=True))
        self.vegetation_points = np.array(pd.read_csv(self.directory + 'data/postprocessed_point_clouds/' + self.input_point_cloud + '/vegetation_points.csv', header=None, index_col=None, delim_whitespace=True))
        self.vegetation_points = self.vegetation_points[:, :3]
        self.vegetation_points = np.hstack((self.vegetation_points, np.zeros((self.vegetation_points.shape[0], 2))))
        self.ground_veg = np.zeros((0, self.vegetation_points.shape[1]))
        self.terrain_points = np.array(pd.read_csv(self.directory + 'data/postprocessed_point_clouds/' + self.input_point_cloud + '/terrain_points.csv', header=None, index_col=None, delim_whitespace=True))
        self.DTM = np.loadtxt("../data/postprocessed_point_clouds/" + self.input_point_cloud + "/DTM.csv")
        if self.parameters['filter_noise']:
            self.stem_points = self.noise_filtering(self.stem_points,min_neighbour_dist=0.03,min_neighbours=3)
        self.characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'dot', 'm', 'space', '_', '-', 'semiC',
                           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', '_M', 'N', 'O', 'P', 'Q', 'R',
                           'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        self.character_viz = []

        for i in self.characters:
            self.character_viz.append(np.genfromtxt(self.directory + '/tools/numbers/' + i + '.csv', delimiter=','))
        self.cyl_dict = {'x': 0,  # DO NOT CHANGE
                         'y': 1,  # DO NOT CHANGE
                         'z': 2,  # DO NOT CHANGE
                         'nx': 3,  # DO NOT CHANGE
                         'ny': 4,  # DO NOT CHANGE
                         'nz': 5,  # DO NOT CHANGE
                         'radius': 6,
                         'CCI': 7,
                         'branch_id': 8,
                         'parent_branch_id': 9,
                         'tree_id': 10,
                         'segment_volume': 11,
                         'segment_angle_to_horiz': 12,
                         'height_above_dtm': 13
                         }
        self.veg_dict = {'x': 0,
                         'y': 1,
                         'z': 2,
                         'tree_id': 3,
                         'height_above_dtm': 4
                         }

        self.kml = simplekml.Kml()
        self.text_point_cloud = np.zeros((0, 3))
        self.tree_measurements = np.zeros((0, 8))  # PlotID,tree_id,x,y,base_z,dbh,height,volume
        self.text_point_cloud = np.zeros((0, 3))

    def convert_coords_to_lat_long(self, easting, northing, point_name=None):
        lat, lon = utm.to_latlon(easting=easting,
                                 northing=northing,
                                 zone_number=self.parameters['UTM_zone_number'],
                                 zone_letter=self.parameters['UTM_zone_letter'],
                                 northern=self.parameters['UTM_is_north'],
                                 strict=None)
        return lat, lon, point_name

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
            # print(np.linspace(cyl1[:7],cyl2[:7],points_per_line,axis=0),np.linspace(cyl1[:7],cyl2[:7],points_per_line,axis=0).shape)
            xyzinterp = np.linspace(cyl1[:3], cyl2[:3], points_per_line, axis=0)
            if xyzinterp.shape[0] > 0:
                interpolated = np.zeros((xyzinterp.shape[0], 14))
                interpolated[:, :3] = xyzinterp
                interpolated[:, 3:6] = (cyl2[:3] - cyl1[:3]) / np.linalg.norm(cyl2[:3] - cyl1[:3])
                interpolated[:, self.cyl_dict['tree_id']] = cyl1[self.cyl_dict['tree_id']]
                # interpolated[:,self.cyl_dict['CCI']] = cyl1[self.cyl_dict['CCI']]
                interpolated[:, self.cyl_dict['branch_id']] = cyl1[self.cyl_dict['branch_id']]
                interpolated[:, self.cyl_dict['parent_branch_id']] = cyl2[self.cyl_dict['branch_id']]
                interpolated[:, self.cyl_dict['segment_volume']] += (np.pi * np.mean(
                    interpolated[:, self.cyl_dict['radius']]) ** 2) * length  # volume
                interpolated[:, self.cyl_dict['radius']] = np.min(
                    [cyl1[self.cyl_dict['radius']], cyl2[self.cyl_dict['radius']]])

        return interpolated

    @classmethod
    def compute_angle(cls, normal1, normal2):
        normal1 = np.atleast_2d(normal1)
        norm1 = normal1 / np.atleast_2d(np.linalg.norm(normal1, axis=1)).T
        norm2 = normal2 / np.atleast_2d(np.linalg.norm(normal2, axis=1)).T
        dot = np.clip(np.einsum('ij,ij->i', norm1, norm2), -1, 1)
        theta = np.degrees(np.arccos(dot))
        return theta

    def cylinder_sorting(self, cylinder_array, angle_tolerance, search_angle, distance_tolerance):
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

        cylinder_array = cylinder_array[cylinder_array[:, self.cyl_dict['radius']] != 0]  # ignore all points with radius of 0.
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
        p = MeasureTree.create_3d_circles_as_points_flat(cyl[0], cyl[1], cyl[2], cyl[6])
        points = MeasureTree.rodrigues_rot(p - cyl[:3], [0, 0, 1], cyl[3:6])
        points = np.hstack((points + cyl[:3], np.zeros((points.shape[0], 8))))
        points[:, -8:] = cyl[-8:]
        return points

    @classmethod
    def subsample_point_cloud(cls, X, min_spacing):
        print("Subsampling...")
        neighbours = NearestNeighbors(n_neighbors=2, algorithm='kd_tree', metric='euclidean').fit(X[:, :3])
        distances, indices = neighbours.kneighbors(X[:, :3])
        x_keep = X[distances[:, 1] >= min_spacing]
        i1 = [distances[:, 1] < min_spacing][0]
        i2 = [X[indices[:, 0], 2] < X[indices[:, 1], 2]][0]
        x_check = X[np.logical_and(i1, i2)]

        while np.shape(x_check)[0] > 1:
            neighbours = NearestNeighbors(n_neighbors=2, algorithm='kd_tree', metric='euclidean').fit(x_check[:, :3])
            distances, indices = neighbours.kneighbors(x_check[:, :3])
            x_keep = np.vstack((x_keep, x_check[distances[:, 1] >= min_spacing, :]))
            i1 = [distances[:, 1] < min_spacing][0]
            i2 = [x_check[indices[:, 0], 2] < x_check[indices[:, 1], 2]][0]
            x_check = x_check[np.logical_and(i1, i2)]
        x = x_keep
        return x

    @classmethod
    def points_along_line(cls, x0, y0, z0, x1, y1, z1, resolution=0.05):
        points_per_line = int(np.linalg.norm(np.array([x1, y1, z1]) - np.array([x0, y0, z0])) / resolution)
        Xs = np.atleast_2d(np.linspace(x0, x1, points_per_line)).T
        Ys = np.atleast_2d(np.linspace(y0, y1, points_per_line)).T
        Zs = np.atleast_2d(np.linspace(z0, z1, points_per_line)).T
        return np.hstack((Xs, Ys, Zs))

    @classmethod
    def create_3d_circles_as_points_flat(cls, x, y, z, r, circle_points=15):
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
        # fitted_circle_points = np.zeros((0,3))
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
        xc, yc = np.mean(P_xy[:, :2], axis=0)
        # Fit circle in new 2D coords with RANSAC
        if P_xy.shape[0] >= 20:

            model_robust, inliers = ransac(P_xy[:, :2], CircleModel, min_samples=int(P_xy.shape[0] * 0.3),
                                           residual_threshold=0.3, max_trials=10000)
            xc, yc = model_robust.params[0:2]
            r = model_robust.params[2]
            CCI = MeasureTree.circumferential_completeness_index([xc, yc], r, P_xy[:, :2])

        elif 10 <= P_xy.shape[0] < 20:
            model_robust, inliers = ransac(P_xy[:, :2], CircleModel, min_samples=7,
                                           residual_threshold=0.3, max_trials=10000)
            xc, yc = model_robust.params[0:2]
            r = model_robust.params[2]
            CCI = MeasureTree.circumferential_completeness_index([xc, yc], r, P_xy[:, :2])

        if CCI < 0.2:
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
        # print("Fitting cylinders...")
        point_cloud = point_cloud[:, :3]
        skeleton_points = skeleton_points[:, :3]
        cyl_array = np.zeros((0, 14))
        line_centre = np.mean(skeleton_points[:, :3], axis=0)
        _, _, vh = np.linalg.svd(line_centre - skeleton_points)
        line_v_hat = vh[0] / np.linalg.norm(vh[0])

        while skeleton_points.shape[0] > num_neighbours:
            nn = NearestNeighbors()
            nn.fit(skeleton_points)
            starting_point = np.atleast_2d(skeleton_points[np.argmin(skeleton_points[:, 2])])
            group = skeleton_points[nn.kneighbors(starting_point,
                                                  n_neighbors=num_neighbours)[1][0]]
            line_centre = np.mean(group[:, :3], axis=0)
            length = np.linalg.norm(np.max(group, axis=0) - np.min(group, axis=0))
            plane_slice = point_cloud[np.linalg.norm(abs(line_v_hat * (point_cloud - line_centre)), axis=1) < (
                        length / 2)]  # calculate distances to plane at centre of line.
            if plane_slice.shape[0] > 0:
                cylinder = MeasureTree.fit_circle_3D(plane_slice, line_v_hat)
                # cylinder = np.hstack((cylinder,np.array([[line_centre[0],line_centre[1],line_centre[2],length]])))
                cyl_array = np.vstack((cyl_array, cylinder))
            skeleton_points = np.delete(skeleton_points, np.argmin(skeleton_points[:, 2]), axis=0)
        min_samples = 7

        # Remove any cylinders greater than the 90th percentile to remove extreme outliers.
        if cyl_array.shape[0] > 0:
            cyl_array = cyl_array[
                np.logical_and(cyl_array[:, cyl_dict['radius']] > np.nanpercentile(cyl_array[:, cyl_dict['radius']], 5),
                               cyl_array[:, cyl_dict['radius']] < np.nanpercentile(cyl_array[:, cyl_dict['radius']],
                                                                                   95))]
        if cyl_array.shape[0] > min_samples:
            try:
                model_robust, inliers = ransac(cyl_array[:, [2, cyl_dict['radius']]], LineModelND,
                                               min_samples=min_samples, residual_threshold=0.05, max_trials=2000)
                cyl_array[:, cyl_dict['radius']] = model_robust.predict_y(cyl_array[:, 2])
                # segment_length = np.linalg.norm(np.max(cyl_array[:,:3],axis=0)-np.min(cyl_array[:,:3],axis=0))
                # mean_radius = np.mean(cyl_array[:,cyl_dict['radius']])
                # segment_volume = segment_length*(np.pi*mean_radius**2)
                # cyl_array[:,cyl_dict['segment_volume']] = segment_volume

                CS_x = CubicSpline(cyl_array[:, 2], cyl_array[:, 0])
                CS_y = CubicSpline(cyl_array[:, 2], cyl_array[:, 1])
                # CS_R = CubicSpline(cyl_array[:,2],cyl_array[:,cyl_dict['radius']])

                # CS_x.set_smoothing_factor(0.5)
                # CS_y.set_smoothing_factor(0.5)
                # CS_R.set_smoothing_factor(0.5)

                cyl_array[:, 0] = CS_x(cyl_array[:, 2])
                cyl_array[:, 1] = CS_y(cyl_array[:, 2])
                # cyl_array[:,cyl_dict['radius']] = CS_R(cyl_array[:,2])

                return cyl_array
            finally:
                return np.zeros((0, 14))
        else:
            return np.zeros((0, 14))

    def cylinder_cleaning(self, sorted_cylinders):
        def get_neighbours_in_cylinder(current_cylinder, nearby_cylinders):
            current_cylinder_vector = current_cylinder[3:6]

            # Translate cylinders to have current cylinder at origin. Rotate all cylinders about new origin so current cylinder vector is up.
            nearby_cylinders_moved = self.rodrigues_rot(nearby_cylinders[:, :3] - current_cylinder[:3],
                                                        current_cylinder_vector, [0, 0, 1])

            # Find all cylinders within 2D radius of the current cylinder.
            nearby_cylinders_within_radius_mask = np.linalg.norm(nearby_cylinders_moved[:, :2]) <= current_cylinder[
                self.cyl_dict['radius']]

            # Find all cylinders within "slice_increment" of the current cylinder in the longitudinal direction.
            nearby_cylinders_within_thickness = np.abs(nearby_cylinders_moved[:, 2]) < self.slice_increment

            return nearby_cylinders[
                np.logical_and(nearby_cylinders_within_radius_mask, nearby_cylinders_within_thickness)]

        cleaned_cyls = np.zeros((0, np.shape(sorted_cylinders)[1]))

        kdtree = spatial.cKDTree(sorted_cylinders[:, :3])
        for cylinder in sorted_cylinders:
            results = kdtree.query_ball_point(cylinder[:3], r=cylinder[self.cyl_dict['radius']])
            within_current_cylinder = get_neighbours_in_cylinder(cylinder, sorted_cylinders[results])
            # print(within_current_cylinder.shape)
            # choose the cylinder with the highest CCI. If all CCIs = 0, choose the cylinder with the largest radius.
            best_cylinder = np.zeros((0, np.shape(sorted_cylinders)[1]))
            if within_current_cylinder.shape[0] > 0:
                if np.max(within_current_cylinder[:, self.cyl_dict['CCI']]) > 0:
                    best_cylinder = within_current_cylinder[np.argsort(within_current_cylinder[:, self.cyl_dict['CCI']])][-1]
                else:
                    best_cylinder = within_current_cylinder[np.argsort(within_current_cylinder[:, self.cyl_dict['radius']])][-1]

            cleaned_cyls = np.vstack((cleaned_cyls, best_cylinder))
            cleaned_cyls = np.unique(cleaned_cyls, axis=0)

        # return cleaned_cyls
        return sorted_cylinders

    def cylinder_cleaning_v2(self, sorted_cylinders):
        cleaned_cyls = np.zeros((0, np.shape(sorted_cylinders)[1]))

        while sorted_cylinders.shape[0] > 2:
            start_point_idx = np.argmin(sorted_cylinders[:, 2])
            start_point = sorted_cylinders[start_point_idx, :]
            sorted_cylinders = np.delete(sorted_cylinders, start_point_idx, axis=0)

            kdtree = spatial.cKDTree(sorted_cylinders[:, :3])
            results = kdtree.query_ball_point(start_point[:3], self.parameters['cleaned_measurement_increment'])
            neighbours = sorted_cylinders[results]
            best_cylinder = start_point
            if neighbours.shape[0] > 0:
                if np.max(neighbours[:, self.cyl_dict['CCI']]) > 0:
                    best_cylinder = neighbours[np.argsort(neighbours[:, self.cyl_dict['CCI']])][-1]
                else:
                    best_cylinder = neighbours[np.argsort(neighbours[:, self.cyl_dict['radius']])][-1]
            cleaned_cyls = np.vstack((cleaned_cyls, best_cylinder))
            sorted_cylinders = np.delete(sorted_cylinders, results, axis=0)
        return cleaned_cyls

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
                    best_cylinder[:2] = np.median(
                        neighbours[neighbours[:, cyl_dict['CCI']] >= np.percentile(neighbours[:, cyl_dict['CCI']], 30), :2], axis=0)
                    best_cylinder[3:6] = np.median(
                        neighbours[neighbours[:, cyl_dict['CCI']] >= np.percentile(neighbours[:, cyl_dict['CCI']], 30), 3:6], axis=0)
                    best_cylinder[cyl_dict['radius']] = np.median(neighbours[neighbours[:, cyl_dict['CCI']] >= np.percentile(neighbours[:, cyl_dict['CCI']], 30), cyl_dict['radius']])
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

    @classmethod
    def clustering(cls, points, eps=0.05, n_jobs=1):
        # print("Clustering...")
        db = DBSCAN(eps=eps, min_samples=2, metric='euclidean', algorithm='kd_tree', n_jobs=n_jobs).fit(points)
        return np.hstack((points, np.atleast_2d(db.labels_).T))

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

    @classmethod
    def slice_clustering(cls, input_data):
        cluster_array_internal = np.zeros((0, 6))
        medians = np.zeros((0, 3))
        new_slice, clustering_distance = input_data
        if new_slice.shape[0] > 1:
            new_slice = MeasureTree.clustering(new_slice[:, :3], eps=clustering_distance)
            for cluster_id in range(0, int(np.max(new_slice[:, -1])) + 1):
                cluster = new_slice[new_slice[:, -1] == cluster_id]
                median = np.median(cluster[:, :3], axis=0)
                medians = np.vstack((medians, median))
                cluster_array_internal = np.vstack(
                    (cluster_array_internal, np.hstack((cluster[:, :3], np.zeros((cluster.shape[0], 3)) + median))))
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
            slice_heights = np.linspace(np.min(self.stem_points[:, 2]), np.max(self.stem_points[:, 2]), int(np.ceil(
                (np.max(self.stem_points[:, 2]) - np.min(self.stem_points[:, 2])) / self.slice_increment)))
            input_data = []
            print("Making slices...")
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
                    input_data.append([new_slice, self.parameters['slice_clustering_distance']])
                    # np.savetxt(self.directory+'data/postprocessed_point_clouds/'+self.input_point_cloud+'/new_slice_'+str(i)+'.csv',new_slice)
            print('\r', max_i, '/', max_i, end='')
            print('\nDone\n')

            print("Starting multithreaded slice clustering...")
            j = 0
            max_j = len(input_data)
            clusteroutputlist = []
            skeletonoutputlist = []
            with get_context("spawn").Pool(processes=self.num_procs) as pool:
                for i in pool.imap_unordered(MeasureTree.slice_clustering, input_data):
                    if j % 100 == 0:
                        print('\r', j, '/', max_j, end='')
                    j += 1
                    cluster, skel = i
                    clusteroutputlist.append(cluster)
                    skeletonoutputlist.append(skel)
            print('\r', max_j, '/', max_j, end='')
            print('\nDone\n')
            skeleton_array = np.vstack(skeletonoutputlist)
            cluster_array = np.vstack(clusteroutputlist)
            del clusteroutputlist, skeletonoutputlist

            print('Clustering skeleton...')
            skeleton_array = MeasureTree.clustering(skeleton_array[:, :3],
                                                    eps=self.slice_increment * 1.5)  # TODO changed from 2 recheck
            skeleton_cluster_visualisation = np.zeros((0, 5))
            for k in np.unique(skeleton_array[:, -1]):
                skeleton_cluster_visualisation = np.vstack((skeleton_cluster_visualisation, np.hstack((skeleton_array[skeleton_array[:, -1] == k], np.zeros((skeleton_array[skeleton_array[:, -1] == k].shape[0], 1)) + np.random.randint(0, 10)))))

            print("Saving skeleton and cluster array...")
            # pd.DataFrame(skeleton_cluster_visualisation).to_csv(self.directory+'data/postprocessed_point_clouds/'+self.input_point_cloud+'/skeleton_cluster_visualisation.csv',header=False,index=None,sep=' ')
            pd.DataFrame(skeleton_array).to_csv(
                self.directory + 'data/postprocessed_point_clouds/' + self.input_point_cloud + '/skeleton.csv',
                header=False, index=None, sep=' ')
            # pd.DataFrame(cluster_array).to_csv(self.directory+'data/postprocessed_point_clouds/'+self.input_point_cloud+'/cluster_array.csv',header=False,index=None,sep=' ')

            print("Loading skeleton and cluster array...")
            # skeleton_array = np.loadtxt(self.directory+'data/postprocessed_point_clouds/'+self.input_point_cloud+'/skeleton.csv')    
            # cluster_array = np.loadtxt(self.directory+'data/postprocessed_point_clouds/'+self.input_point_cloud+'/cluster_array.csv')    

            print("Making kdtree...")
            # Assign unassigned skeleton points to the nearest group.
            unassigned_bool = skeleton_array[:, -1] == -1
            kdtree = spatial.cKDTree(skeleton_array[unassigned_bool][:, :3], leafsize=100000)
            distances, neighbours = kdtree.query(skeleton_array[unassigned_bool, :3], k=2)
            skeleton_array[unassigned_bool, -1][distances[:, 1] < self.slice_increment * 3] = skeleton_array[unassigned_bool, -1][neighbours[:, 1]][distances[:, 1] < self.slice_increment * 3]

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
                                                                        r=0.000000001)))  # handling some floating point errors that were giving me problems... dodgy?
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

            outputlist = []
            with get_context("spawn").Pool(processes=self.num_procs) as pool:
                for i in pool.imap_unordered(MeasureTree.threaded_cyl_fitting, input_data):
                    outputlist.append(i)
                    # full_cyl_array = np.vstack((full_cyl_array,i))
                    if j % 11 == 0:
                        print('\r', j, '/', max_j, end='')
                    j += 1
            full_cyl_array = np.vstack(outputlist)

            del outputlist
            print('\r', max_i, '/', max_i, end='')
            print('\nDone\n')

            print("Deleting cyls with CCI less than:", self.parameters['minimum_CCI'])
            full_cyl_array = full_cyl_array[full_cyl_array[:, self.cyl_dict['CCI']] >= self.parameters['minimum_CCI']]

            # cyl_array = [x,y,z,nx,ny,nz,r,CCI,branch_id,tree_id,segment_volume,parent_branch_id]
            print("Saving cylinder array...")
            pd.DataFrame(full_cyl_array).to_csv(
                self.directory + 'data/postprocessed_point_clouds/' + self.input_point_cloud + '/full_cyl_array.csv',
                header=False, index=None, sep=' ')
            # full_cyl_array = np.array(pd.read_csv(self.directory+'data/postprocessed_point_clouds/'+self.input_point_cloud+'/full_cyl_array.csv',header=None,index_col=None,delim_whitespace=True))

        full_cyl_array = np.loadtxt(
            self.directory + 'data/postprocessed_point_clouds/' + self.input_point_cloud + '/full_cyl_array.csv')
        print("Sorting Cylinders...")

        full_cyl_array = self.cylinder_sorting(full_cyl_array,
                                               angle_tolerance=90,
                                               search_angle=20,
                                               distance_tolerance=3.0)

        print('Correcting Cylinder assignments...')
        sorted_full_cyl_array = np.zeros((0, full_cyl_array.shape[1]))
        t_id = 1
        max_search_radius = 1.
        min_points = 5
        max_search_angle = 30
        print(np.unique(full_cyl_array[:, self.cyl_dict['tree_id']]))
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
                if lowneighbours.shape[0] > 1:
                    # vector_angles = MeasureTree.compute_angle(lowest_point[3:6],lowneighbours[lowneighbours[:,2]<lowest_point[2],3:6])
                    # angles = MeasureTree.compute_angle(lowest_point[3:6],lowneighbours[lowneighbours[:,2]<lowest_point[2],3:6])
                    # lowneighbours = lowneighbours[lowneighbours[:,2]<lowest_point[2]]
                    # lowneighbours = lowneighbours[lowneighbours[:,self.cyl_dict['radius']]>=0.5*lowest_point[self.cyl_dict['radius']]]
                    if lowneighbours.shape[0] > 0:
                        angles = MeasureTree.compute_angle(lowest_point[3:6], lowest_point[:3] - lowneighbours[:, :3])
                        angles = angles[angles <= max_search_angle]

                        # vector_angles = vector_angles[vector_angles<=max_angle]
                        # search_angles = search_angles[search_angles<=max_search_angle]

                        if angles.shape[0] > 0:

                            best_parent_point = lowneighbours[np.argmin(angles)]
                            tree = np.vstack((tree, self.interpolate_cyl(lowest_point, best_parent_point,
                                                                         resolution=self.slice_increment)))
                            tree[:, self.cyl_dict['tree_id']] = best_parent_point[self.cyl_dict['tree_id']]
                            sorted_full_cyl_array = np.vstack((sorted_full_cyl_array, tree))

                        elif lowest_point_z < 3:
                            tree[:, self.cyl_dict['tree_id']] = t_id
                            sorted_full_cyl_array = np.vstack((sorted_full_cyl_array, tree))
                            t_id += 1

                elif lowest_point_z < 3:
                    tree[:, self.cyl_dict['tree_id']] = t_id
                    sorted_full_cyl_array = np.vstack((sorted_full_cyl_array, tree))
                    t_id += 1

                elif highneighbours.shape[0] > 1:
                    # highneighbours = highneighbours[highneighbours[:,2]>highest_point[2]]
                    # highneighbours = highneighbours[highneighbours[:,self.cyl_dict['radius']]<1.5*lowest_point[self.cyl_dict['radius']]]
                    if highneighbours.shape[0] > 0:

                        angles = MeasureTree.compute_angle(highest_point[3:6],
                                                           highneighbours[:, :3] - highest_point[:3])
                        angles = angles[angles <= max_search_angle]

                        if angles.shape[0] > 0:
                            best_parent_point = highneighbours[np.argmin(angles)]
                            tree = np.vstack((tree, self.interpolate_cyl(best_parent_point, highest_point,
                                                                         resolution=self.slice_increment)))
                            tree[:, self.cyl_dict['tree_id']] = best_parent_point[self.cyl_dict['tree_id']]
                            sorted_full_cyl_array = np.vstack((sorted_full_cyl_array, tree))

        pd.DataFrame(sorted_full_cyl_array).to_csv(
            self.directory + 'data/postprocessed_point_clouds/' + self.input_point_cloud + '/sorted_full_cyl_array.csv',
            header=False, index=None, sep=' ')

        sorted_full_cyl_array = np.loadtxt(
            self.directory + 'data/postprocessed_point_clouds/' + self.input_point_cloud + '/sorted_full_cyl_array.csv')
        # sorted_full_cyl_array = np.hstack((sorted_full_cyl_array,np.zeros((sorted_full_cyl_array.shape[0],1))))

        max_search_radius = 4.
        # max_search_angle = 15

        print("Cylinder interpolation...")

        tree_list = []
        interpolated_full_cyl_array = np.zeros((0, 14))
        max_tree_id = np.unique(sorted_full_cyl_array[:, self.cyl_dict['tree_id']]).shape[0]
        for tree_id in np.unique(sorted_full_cyl_array[:, self.cyl_dict['tree_id']]):
            if int(tree_id) % 10 == 0:
                print("Tree ID", int(tree_id), '/', int(max_tree_id))
            current_tree = sorted_full_cyl_array[sorted_full_cyl_array[:, self.cyl_dict['tree_id']] == tree_id]
            _, individual_branches_indices = np.unique(current_tree[:, self.cyl_dict['branch_id']], return_index=True)
            tree_list.append(nx.Graph())
            # print(individual_branches_indices.shape)
            for branch in current_tree[individual_branches_indices]:
                branch_id = branch[self.cyl_dict['branch_id']]
                parent_branch_id = branch[self.cyl_dict['parent_branch_id']]
                # print(branch_id,parent_branch_id)
                tree_list[-1].add_edge(int(parent_branch_id), int(branch_id))
                # print(tree_list[-1].edges)
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
                                # print(point-interp_to_point)
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
                        angles = MeasureTree.compute_angle(lowest_point_of_current_branch[3:6], lowest_point_of_current_branch[:3] - parent_points_in_range[:, :3])
                        angles = angles[angles <= max_search_angle]

                        if angles.shape[0] > 0:
                            best_parent_point = parent_points_in_range[np.argmin(angles)]
                            # Interpolates from lowest point of current branch to smallest angle parent point.
                            interpolated_full_cyl_array = np.vstack((interpolated_full_cyl_array, self.interpolate_cyl(
                                lowest_point_of_current_branch, best_parent_point, resolution=self.slice_increment)))

            interpolated_full_cyl_array = np.vstack((interpolated_full_cyl_array, sorted_full_cyl_array))
        pd.DataFrame(interpolated_full_cyl_array).to_csv(
            self.directory + 'data/postprocessed_point_clouds/' + self.input_point_cloud + '/interpolated_full_cyl_array.csv',
            header=False, index=None, sep=' ')

        v1 = interpolated_full_cyl_array[:, 3:6]
        v2 = np.vstack((interpolated_full_cyl_array[:, 3],
                        interpolated_full_cyl_array[:, 4],
                        np.zeros((interpolated_full_cyl_array.shape[0])))).T
        interpolated_full_cyl_array[:, self.cyl_dict['segment_angle_to_horiz']] = self.compute_angle(v1, v2)
        interpolated_full_cyl_array = self.get_heights_above_DTM(interpolated_full_cyl_array)

        pd.DataFrame(interpolated_full_cyl_array).to_csv(
            self.directory + 'data/postprocessed_point_clouds/' + self.input_point_cloud + '/interpolated_full_cyl_array.csv',
            header=False, index=None, sep=' ')

        interpolated_full_cyl_array = np.loadtxt(
            self.directory + 'data/postprocessed_point_clouds/' + self.input_point_cloud + '/interpolated_full_cyl_array.csv')

        # Need a step to remove overlapping interpolations and to remove cylinders inside other cylinders.
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
            lowest_measured_tree_point = deepcopy(single_tree[np.argmin(single_tree[:, -1])])
            tree_base_point = deepcopy(single_tree[np.argmin(single_tree[:, -1])])
            z_tree_base = tree_base_point[2] - tree_base_point[-1]
            tree_base_point[2] = z_tree_base
            single_tree = np.vstack((single_tree, self.interpolate_cyl(lowest_measured_tree_point, tree_base_point,
                                                                       resolution=self.slice_increment)))

            input_data.append([single_tree, self.parameters['cleaned_measurement_radius'], self.cyl_dict])

        print('\r', max_tree_id, '/', max_tree_id, end='')
        print('\nDone\n')

        # interpolated_full_cyl_array = self.get_CCI_of_all_cyls(interpolated_full_cyl_array) #multithread this!!! #TODO
        # multithreaded_CCI(args):
        # cyls,slice_increment,stem_points,stem_points_kdtree,cyl_dict = args

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
        print(cleaned_cyls.shape)
        print('\r', max_j, '/', max_j, end='')
        print('\nDone\n')

        pd.DataFrame(cleaned_cyls).to_csv(
            self.directory + 'data/postprocessed_point_clouds/' + self.input_point_cloud + '/cleaned_cyls.csv',
            header=False, index=None, sep=' ')
        if 1:
            print("Making cleaned cylinder visualisation...")
            j = 0
            cleaned_cyl_vis = []
            max_j = np.shape(cleaned_cyls)[0]
            # interpolated_cyl_vis = np.zeros((0,9))
            with get_context("spawn").Pool(processes=self.num_procs) as pool:
                for i in pool.imap_unordered(self.make_cyl_visualisation, cleaned_cyls):
                    # interpolated_cyl_vis = np.vstack((interpolated_cyl_vis,i))
                    cleaned_cyl_vis.append(i)
                    if j % 100 == 0:
                        print('\r', j, '/', max_j, end='')
                    j += 1
            cleaned_cyl_vis = np.vstack(cleaned_cyl_vis)
            print('\r', max_j, '/', max_j, end='')
            print('\nDone\n')

            print("\nSaving cylinder visualisation...")
            # pd.DataFrame(cleaned_cyl_vis).to_csv(self.directory+'data/postprocessed_point_clouds/'+self.input_point_cloud+'/cleaned_cyl_vis.csv',header=False,index=None,sep=' ')
            np.savetxt(
                self.directory + 'data/postprocessed_point_clouds/' + self.input_point_cloud + '/cleaned_cyl_vis.csv',
                cleaned_cyl_vis)

        if 1:
            veg_step_distance = 1
            num_veg_assignment_iterations = 10
            self.vegetation_points = self.get_heights_above_DTM(self.vegetation_points)

            self.ground_veg = self.vegetation_points[
                self.vegetation_points[:, self.veg_dict['height_above_dtm']] <= self.parameters['ground_veg_cutoff_height']]
            self.vegetation_points = self.vegetation_points[
                self.vegetation_points[:, self.veg_dict['height_above_dtm']] > self.parameters['ground_veg_cutoff_height']]
            stem_kdtree = spatial.cKDTree(cleaned_cyls[:, :3], leafsize=1000)
            results = stem_kdtree.query(self.vegetation_points[:, :3], k=1, distance_upper_bound=self.veg_step_distance)

            if self.parameters['canopy_mode'] == 'continuous':
                print('Sorting vegetation points using continous mode.')
                self.vegetation_points[results[0] < self.veg_step_distance, self.veg_dict['tree_id']] = cleaned_cyls[
                    results[1][results[0] < self.veg_step_distance], self.cyl_dict['tree_id']]
                assigned_vegetation_points = self.vegetation_points[
                    self.vegetation_points[:, self.veg_dict['tree_id']] != 0]
                unassigned_vegetation_points = self.vegetation_points[
                    self.vegetation_points[:, self.veg_dict['tree_id']] == 0]
                for iteration in range(0, num_veg_assignment_iterations):
                    assigned_vegetation_points_kdtree = spatial.cKDTree(assigned_vegetation_points[:, :3],
                                                                        leafsize=1000)
                    results = assigned_vegetation_points_kdtree.query(unassigned_vegetation_points[:, :3], k=1,
                                                                      distance_upper_bound=self.veg_step_distance)
                    # Assign the label of the nearest vegetation if less than 3 m away.
                    unassigned_vegetation_points[results[0] < self.veg_step_distance, self.veg_dict['tree_id']] = assigned_vegetation_points[results[1][results[0] < self.veg_step_distance], self.veg_dict['tree_id']]

                    assigned_vegetation_points = np.vstack((assigned_vegetation_points, unassigned_vegetation_points[unassigned_vegetation_points[:, self.veg_dict['tree_id']] != 0]))
                    unassigned_vegetation_points = unassigned_vegetation_points[unassigned_vegetation_points[:, self.veg_dict['tree_id']] == 0]

            elif self.parameters['canopy_mode'] == 'photogrammetry_mode':
                print('Sorting vegetation points using photogrammetry mode.')
                self.vegetation_points[results[0] < veg_step_distance, self.veg_dict['tree_id']] = cleaned_cyls[results[1][results[0] < veg_step_distance], self.cyl_dict['tree_id']]
                assigned_vegetation_points = self.vegetation_points[self.vegetation_points[:, self.veg_dict['tree_id']] != 0]
                unassigned_vegetation_points = self.vegetation_points[self.vegetation_points[:, self.veg_dict['tree_id']] == 0]
                assigned_vegetation_points_kdtree = spatial.cKDTree(assigned_vegetation_points[:, :2], leafsize=1000)
                results = assigned_vegetation_points_kdtree.query(unassigned_vegetation_points[:, :2], k=1,
                                                                  distance_upper_bound=veg_step_distance)
                unassigned_vegetation_points[results[0] < veg_step_distance, self.veg_dict['tree_id']] = assigned_vegetation_points[results[1][results[0] < veg_step_distance], self.veg_dict['tree_id']]
                assigned_vegetation_points = np.vstack((assigned_vegetation_points, unassigned_vegetation_points[unassigned_vegetation_points[:, self.veg_dict['tree_id']] != 0]))
                unassigned_vegetation_points = unassigned_vegetation_points[unassigned_vegetation_points[:, self.veg_dict['tree_id']] == 0]
                for iteration in range(0, 2):
                    assigned_vegetation_points_kdtree = spatial.cKDTree(assigned_vegetation_points[:, :3],
                                                                        leafsize=1000)
                    results = assigned_vegetation_points_kdtree.query(unassigned_vegetation_points[:, :3], k=1,
                                                                      distance_upper_bound=veg_step_distance)
                    # Assign the label of the nearest vegetation if less than 3 m away.
                    unassigned_vegetation_points[results[0] < veg_step_distance, self.veg_dict['tree_id']] = assigned_vegetation_points[results[1][results[0] < veg_step_distance], self.veg_dict['tree_id']]

                    assigned_vegetation_points = np.vstack((assigned_vegetation_points, unassigned_vegetation_points[
                        unassigned_vegetation_points[:, self.veg_dict['tree_id']] != 0]))
                    unassigned_vegetation_points = unassigned_vegetation_points[
                        unassigned_vegetation_points[:, self.veg_dict['tree_id']] == 0]

                for iteration in range(0, 5):
                    assigned_vegetation_points_kdtree = spatial.cKDTree(assigned_vegetation_points[:, :2],
                                                                        leafsize=1000)
                    results = assigned_vegetation_points_kdtree.query(unassigned_vegetation_points[:, :2], k=1,
                                                                      distance_upper_bound=self.veg_step_distance)
                    # Assign the label of the nearest vegetation if less than 3 m away.
                    unassigned_vegetation_points[results[0] < self.veg_step_distance, self.veg_dict['tree_id']] = assigned_vegetation_points[results[1][results[0] < self.veg_step_distance], self.veg_dict['tree_id']]

                    assigned_vegetation_points = np.vstack((assigned_vegetation_points, unassigned_vegetation_points[unassigned_vegetation_points[:, self.veg_dict['tree_id']] != 0]))
                    unassigned_vegetation_points = unassigned_vegetation_points[unassigned_vegetation_points[:, self.veg_dict['tree_id']] == 0]

            print("Saving vegetation points...")
            pd.DataFrame(assigned_vegetation_points).to_csv(self.directory + 'data/postprocessed_point_clouds/' + self.input_point_cloud + '/vegetation_points_assigned.csv', header=False, index=None, sep=' ')
            pd.DataFrame(unassigned_vegetation_points).to_csv(self.directory + 'data/postprocessed_point_clouds/' + self.input_point_cloud + '/unassigned_vegetation_points.csv', header=False, index=None, sep=' ')
            # pd.DataFrame(self.ground_veg).to_csv(self.directory+'data/postprocessed_point_clouds/'+self.input_point_cloud+'/ground_veg.csv',header=False,index=None,sep=' ')
            print("Done")

            # # # assigned_vegetation_points = np.loadtxt(self.directory+'data/postprocessed_point_clouds/'+self.input_point_cloud+'/vegetation_points_assigned.csv')#np.hstack((sorted_by_tree[:,:3],sorted_by_tree[:,-5:])))    

            print("Measuring canopy gap fraction...")
            veg_kdtree = spatial.cKDTree(assigned_vegetation_points[:, :2], leafsize=10000)

            xmin = np.floor(np.min(self.terrain_points[:, 0]))
            ymin = np.floor(np.min(self.terrain_points[:, 1]))
            xmax = np.ceil(np.max(self.terrain_points[:, 0]))
            ymax = np.ceil(np.max(self.terrain_points[:, 1]))
            x_points = np.linspace(xmin, xmax,
                                   int(np.ceil((xmax - xmin) / self.parameters['Canopy_coverage_resolution'])) + 1)
            y_points = np.linspace(ymin, ymax,
                                   int(np.ceil((ymax - ymin) / self.parameters['Canopy_coverage_resolution'])) + 1)

            convexhull = spatial.ConvexHull(self.terrain_points[:, :2])
            canopy_density = np.zeros((0, 4))
            ground_area = 0
            canopy_area = 0
            for x in x_points:
                for y in y_points:
                    if self.inside_conv_hull(np.array([x, y]), convexhull):
                        indices = veg_kdtree.query_ball_point([x, y], r=self.parameters['Canopy_coverage_resolution'],
                                                              p=10)
                        ground_area += 1
                        if len(indices) > 5:
                            canopy_area += 1
                            canopy_density = np.vstack((canopy_density, np.array([[x, y, 0, len(indices)]])))

            print(canopy_area, ground_area, "Canopy Gap Fraction:", canopy_area / ground_area)
            np.savetxt(
                self.directory + 'data/postprocessed_point_clouds/' + self.input_point_cloud + '/canopy_density.csv',
                canopy_density)

        if 1:
            dtmmin = np.min(self.DTM[:, :2], axis=0)
            dtmmax = np.max(self.DTM[:, :2], axis=0)
            plot_centre = (dtmmin + dtmmax) / 2

            fig5 = plt.figure(figsize=(7, 7))
            ax5 = fig5.add_subplot(1, 1, 1)
            plot_centre_lat, plot_centre_lon, _ = self.convert_coords_to_lat_long(plot_centre[0], plot_centre[1], ' ')

            dtm_boundaries = [[np.min(self.DTM[:, 0]), np.min(self.DTM[:, 1]), 'SouthWestCorner'],
                              [np.min(self.DTM[:, 0]), np.max(self.DTM[:, 1]), 'NorthWestCorner'],
                              [np.max(self.DTM[:, 0]), np.min(self.DTM[:, 1]), 'SouthEastCorner'],
                              [np.max(self.DTM[:, 0]), np.max(self.DTM[:, 1]), 'NorthEastCorner']]

            dtm_boundaries_lat = []
            dtm_boundaries_lon = []
            dtm_boundaries_names = []
            for i in dtm_boundaries:
                lat, lon, names = self.convert_coords_to_lat_long(i[0], i[1], i[2])
                dtm_boundaries_lat.append(lat)
                dtm_boundaries_lon.append(lon)
                dtm_boundaries_names.append(names)

            lat, lon, names = self.convert_coords_to_lat_long(plot_centre[0],
                                                              plot_centre[1],
                                                              'PlotCentre')
            dtm_boundaries_lat.append(lat)
            dtm_boundaries_lon.append(lon)
            dtm_boundaries_names.append(names)
            print(dtm_boundaries_lat)
            print(dtm_boundaries_lon)
            print(dtm_boundaries_names)

            dtm_boundaries = np.array([dtm_boundaries_lat, dtm_boundaries_lon, dtm_boundaries_names]).T
            pd.DataFrame(dtm_boundaries).to_csv(
                self.directory + 'data/postprocessed_point_clouds/' + self.input_point_cloud + "/Plot_Extents.csv",
                header=False, index=None, sep=',')
            for i in dtm_boundaries:
                self.kml.newpoint(name=i[2], coords=[(i[1], i[0])], description='Boundary point')

        if 1:
            ax5.set_title("Stem Map        Plot centre: " + str([plot_centre_lat, plot_centre_lon])[1:-1], fontsize=10)
            ax5.set_xlabel("X Position (m)")
            ax5.set_ylabel("Y Postition (m)")
            ax5.axis('equal')
            # ax5.scatter(self.DTM[:,0],self.DTM[:,1],c=self.DTM[:,2],s=200,marker='s')
            # ax5.plot(np.hstack((self.convex_hull_points[:,0],self.convex_hull_points[0,0])),np.hstack((self.convex_hull_points[:,1],self.convex_hull_points[0,1])),color='black')
            # ax5.scatter(self.circle_clusters[:,0],self.circle_clusters[:,1],color='red',s=4)levels=)
            zmin = np.floor(np.min(self.DTM[:, 2]))
            zmax = np.ceil(np.max(self.DTM[:, 2]))
            contour_resolution = 1  # metres
            sub_contour_resolution = contour_resolution / 5
            zrange = int(np.ceil((zmax - zmin) / contour_resolution)) + 1
            levels = np.linspace(zmin, zmax, zrange)

            sub_zrange = int(np.ceil((zmax - zmin) / sub_contour_resolution)) + 1
            sub_levels = np.linspace(zmin, zmax, sub_zrange)

            ax5.tricontour(self.DTM[:, 0] - plot_centre[0], self.DTM[:, 1] - plot_centre[1], self.DTM[:, 2],
                           levels=sub_levels, colors='brown', linestyles='dashed', linewidths=1)

            contours = ax5.tricontour(self.DTM[:, 0] - plot_centre[0], self.DTM[:, 1] - plot_centre[1], self.DTM[:, 2],
                                      levels=levels, colors='darkgreen')

            ax5.scatter([0], [0], marker='x', s=50, c='red')
            plt.clabel(contours, inline=True, fontsize=8)
            ax5.set_xlim([np.min(self.DTM[:, 0]) - plot_centre[0] - 5, np.max(self.DTM[:, 0]) - plot_centre[0] + 5])
            ax5.set_ylim([np.min(self.DTM[:, 1]) - plot_centre[1] - 5, np.max(self.DTM[:, 1]) - plot_centre[1] + 5])
            fig5.show(False)
            fig5.savefig(self.directory + 'data/postprocessed_point_clouds/' + self.input_point_cloud + "/Stem_Map.png",
                         dpi=600, bbox_inches='tight', pad_inches=0.0)
            ########################################################################################################################
            ###################################################################################################

            ########################################################################################################################

            # Canopy Density Plot 
            fig6 = plt.figure(figsize=(7, 7))
            ax6 = fig6.add_subplot(1, 1, 1)
            ax6.set_title("Stem Map", fontsize=10)
            ax6.set_xlabel("X Position (m)")
            ax6.set_ylabel("Y Postition (m)")
            ax6.axis('equal')
            ax6.scatter(self.DTM[:, 0], self.DTM[:, 1], c='white')
            ax6.scatter(canopy_density[:, 0], canopy_density[:, 1], c=canopy_density[:, 3], s=200, marker='s')
            fig6.show(False)
            fig6.savefig(
                self.directory + 'data/postprocessed_point_clouds/' + self.input_point_cloud + "/CanopyDensityPlot.png",
                dpi=600, bbox_inches='tight', pad_inches=0.0)
            ########################################################################################################################
            plt.close('all')

        sorted_tree_points = np.zeros((0, 5))
        tree_kd_tree = spatial.cKDTree(self.stem_points[:, :3])
        canopy_density_kd_tree = spatial.cKDTree(canopy_density[:, :2])
        ground_veg_kd_tree = spatial.cKDTree(self.ground_veg[:, :2])

        tree_data_dict = {'Site': 0,
                          'PlotID': 1,
                          'treeNo': 2,
                          'x_tree_base': 3,
                          'y_tree_base': 4,
                          'z_tree_base': 5,
                          'DBH': 6,
                          'Height': 7,
                          'Volume': 8,
                          'Crown_mean_x': 9,
                          'Crown_mean_y': 10,
                          'Crown_top_x': 11,
                          'Crown_top_y': 12,
                          'Crown_top_z': 13,
                          'mean_understory_height_in_10m_radius': 14,
                          }
        tree_data = np.zeros((0, 15))

        for tree_id in np.unique(cleaned_cyls[:, self.cyl_dict['tree_id']]):
            tree = cleaned_cyls[cleaned_cyls[:, self.cyl_dict['tree_id']] == tree_id]

            tree_vegetation = assigned_vegetation_points[
                assigned_vegetation_points[:, self.veg_dict['tree_id']] == tree_id]
            combined = np.vstack((tree[:, :3], tree_vegetation[:, :3]))
            combined = np.hstack((combined, np.zeros((combined.shape[0], 1))))
            combined = self.get_heights_above_DTM(combined)

            # Get highest point of tree. Note, there is usually noise, so we use the 95th percentile.
            tree_max_point = combined[
                abs(combined[:, 2] - np.percentile(combined[:, 2], 95, interpolation='nearest')).argmin()]

            tree_base_point = deepcopy(combined[np.argmin(combined[:, -1])])
            z_tree_base = tree_base_point[2] - tree_base_point[-1]

            tree_mean_position = np.mean(combined[:, :2], axis=0)
            tree_height = tree_max_point[-1]
            del combined

            # print("Tree no:"+str(tree_id),"   Tree height:",np.around(tree_height,3),'m')
            radii = 1.5 * tree[:, self.cyl_dict['radius']]
            results = tree_kd_tree.query_ball_point(tree[:, :3], r=radii)
            results = np.unique(np.hstack(results))
            tree_points = self.stem_points[np.asarray(results, dtype='int'), :3]

            tree_points = np.hstack((tree_points, np.zeros((tree_points.shape[0], 2))))
            tree_points = self.get_heights_above_DTM(tree_points)
            tree_points[:, 3] = tree_id
            sorted_tree_points = np.vstack((sorted_tree_points, tree_points))
            base_northing = tree[np.argmin(tree[:, 2]), 0]
            base_easting = tree[np.argmin(tree[:, 2]), 1]
            DBH_slice = tree[np.logical_and(tree[:, self.cyl_dict['height_above_dtm']] > 0.1,
                                            tree[:, self.cyl_dict['height_above_dtm']] < 1.6)]
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
            mean_vegetation_density_in_10m_radius = 'N/A'
            mean_understory_height_in_10m_radius = 'N/A'
            canopy_density_points = canopy_density[canopy_density_kd_tree.query_ball_point([DBH_X, DBH_Y], r=10)]
            nearby_understory_points = self.ground_veg[ground_veg_kd_tree.query_ball_point([DBH_X, DBH_Y], r=10)]
            if canopy_density_points.shape[0] > 0:
                mean_vegetation_density_in_10m_radius = np.around(np.nanmean(canopy_density_points[:, 2]), 2)
            if nearby_understory_points.shape[0] > 0:
                mean_understory_height_in_10m_radius = np.around(
                    np.nanmean(nearby_understory_points[:, self.veg_dict['height_above_dtm']]), 2)
            if tree.shape[0] > 0:
                # print("Tree "+str(int(tree_id))," Volume:",0,'m^3',", Height:",tree_height,'m')
                tree_lat, tree_lon, tree_name = self.convert_coords_to_lat_long(base_northing, base_easting,
                                                                                str(int(tree_id)))
                description = 'Tree ' + str(int(tree_id))
                description = description + '\nDBH: ' + str(DBH) + ' m'
                description = description + '\nVolume: ' + str(np.around(volume, 3)) + ' m^3'
                description = description + '\nHeight: ' + str(np.around(tree_height, 3)) + ' m'
                description = description + '\nMean Veg Density (10 m radius): ' + str(
                    mean_vegetation_density_in_10m_radius) + ' units'
                description = description + '\nMean Understory Height (10 m radius): ' + str(
                    mean_understory_height_in_10m_radius) + ' m'

                print(description)
                self.kml.newpoint(name=tree_name, coords=[(tree_lon, tree_lat)], description=description)
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
                this_trees_data[:, tree_data_dict['mean_understory_height_in_10m_radius']] = mean_understory_height_in_10m_radius
                tree_data = np.vstack((tree_data, this_trees_data))

            text_size = 0.00256
            line_height = 0.025
            if DBH_Z != 0:
                line0 = self.point_cloud_annotations(text_size, DBH_X, DBH_Y + line_height, DBH_Z + line_height, DBH * 0.5, '            DIAM: ' + str(np.around(DBH, 2)) + 'm')
                line1 = self.point_cloud_annotations(text_size, DBH_X, DBH_Y, DBH_Z, DBH * 0.5,                                     '       CCI AT BH: ' + str(np.around(mean_CCI_at_BH, 2)))
                line2 = self.point_cloud_annotations(text_size, DBH_X, DBH_Y - 2 * line_height, DBH_Z - 2 * line_height, DBH * 0.5, '          HEIGHT: ' + str(np.around(tree_height, 2)) + 'm')
                line3 = self.point_cloud_annotations(text_size, DBH_X, DBH_Y - 3 * line_height, DBH_Z - 3 * line_height, DBH * 0.5, '          VOLUME: ' + str(np.around(volume, 2)) + 'm')
                line4 = self.point_cloud_annotations(text_size, DBH_X, DBH_Y - 3 * line_height, DBH_Z - 4 * line_height, DBH * 0.5, '    CHECK VOLUME: ' + str(np.around((np.pi * (0.5 * DBH) ** 2) * tree_height, 2)) + 'm')
                # print(X,Y,Z,X,Y,Z+height)
                height_measurement_line = self.points_along_line(x_tree_base, y_tree_base, z_tree_base, x_tree_base,
                                                                 y_tree_base, z_tree_base + tree_height,
                                                                 resolution=0.025)
                dbh_circle_points = self.create_3d_circles_as_points_flat(DBH_X, DBH_Y, DBH_Z, DBH / 2,
                                                                          circle_points=100)
                self.text_point_cloud = np.vstack((self.text_point_cloud, line0, line1, line2, line3, line4,
                                                   height_measurement_line, dbh_circle_points))

        # # tree_information = tree_information[tree_information[:,5]>self.min_tree_volume] #remove trees with tiny volumes.
        # # np.savetxt(self.directory+'data/postprocessed_point_clouds/'+self.input_point_cloud+'/tree_information.csv',tree_information)#np.hstack((sorted_by_tree[:,:3],sorted_by_tree[:,-5:])))    
        # # print("Tree Information saved.")
        # # # print("Saving sorted cylinder array...")
        # # # print(sorted_by_tree.shape)
        # # # np.savetxt(self.directory+'data/postprocessed_point_clouds/'+self.input_point_cloud+'/sorted_by_tree.csv',sorted_by_tree)
        # # np.savetxt(self.directory+'data/postprocessed_point_clouds/'+self.input_point_cloud+'/processed_trees.csv',processed_trees) 
        # # np.savetxt(self.directory+'data/postprocessed_point_clouds/'+self.input_point_cloud+'/tree_measurements.csv',tree_measurements) 

        pd.DataFrame(self.text_point_cloud).to_csv(
            self.directory + 'data/postprocessed_point_clouds/' + self.input_point_cloud + '/text_point_cloud.csv',
            header=False, index=None, sep=' ')
        pd.DataFrame(sorted_tree_points).to_csv(
            self.directory + 'data/postprocessed_point_clouds/' + self.input_point_cloud + '/sorted_tree_points.csv',
            header=False, index=None, sep=' ')
        pd.DataFrame(tree_data).to_csv(
            self.directory + 'data/postprocessed_point_clouds/' + self.input_point_cloud + '/tree_data.csv',
            header=[i for i in tree_data_dict], index=None, sep=',')
        self.kml.save(
            self.directory + 'data/postprocessed_point_clouds/' + self.input_point_cloud + "/Plot_Extents.kml")


if __name__ == '__main__':
    plt.ioff()
    filenames = [
        # 'NDT_PROJ_Leach_P111_TLS.csv',
        # 'NDT_PROJ_Leach_P61_TLS.csv',
        # 'NDT_PROJ_Denham_P264_TLS.csv',
        # 'NDT_PROJ_Denham_P257_TLS.csv',
        'taper29_graphic_version.csv',
        # 'measure_combined_test.csv',
        # 's2p1NSW.csv',
        # 'Fleas P1_med.csv',
        # 'Fleas_P1.csv',
        # 'Fleas_P2.csv',
        # 'Fleas_P3.csv',
        # 'Leach_P61.csv',
        # 'Leach_P111.csv',
        # 'Denham_P264.csv',
        # 'Denham_P257.csv',
        # "TAPER49_class_1.0_cmDS.csv",
        # "TAPER48_class_1.0_cmDS.csv",
        # "TAPER46_class_1.0_cmDS.csv",
        # "TAPER47_class_1.0_cmDS.csv",
        # "TAPER45_class_1.0_cmDS.csv",
        # "TAPER44_class_1.0_cmDS.csv",
        # "TAPER43_class_1.0_cmDS.csv",
        # "TAPER42_class_1.0_cmDS.csv",
        # "TAPER41_class_1.0_cmDS.csv",
        # "TAPER40_class_1.0_cmDS.csv",
        # "TAPER39_class_1.0_cmDS.csv",
        # "TAPER38_class_1.0_cmDS.csv",
        # "TAPER37_class_1.0_cmDS.csv",
        # "TAPER36_class_1.0_cmDS.csv",
        # "TAPER35_class_1.0_cmDS.csv",
        # "TAPER34_class_1.0_cmDS.csv",
        # "TAPER33_class_1.0_cmDS.csv",
        # "TAPER32_class_1.0_cmDS.csv",
        # "TAPER31_class_1.0_cmDS.csv",
        # "TAPER30_class_1.0_cmDS.csv",
        # "TAPER29_class_1.0_cmDS.csv",
        # "TAPER50_class_1.0_cmDS.csv",
        # "T27_class_1.0_cmDS.csv",
        # "T26_class_1.0_cmDS.csv",
        # "T25_class_1.0_cmDS.csv",
        # "T23_class_1.0_cmDS.csv",
        # "T22_class_1.0_cmDS.csv",
        # "T21_class_1.0_cmDS.csv",
        # "T20_class_1.0_cmDS.csv",
        # "T19_class_1.0_cmDS.csv",
        # "T18_class_1.0_cmDS.csv",
        # "T017_class_1.0_cmDS.csv",
        # "T16_class_1.0_cmDS.csv",
        # "T15_class_1.0_cmDS.csv",
        # "T14_class_1.0_cmDS.csv",
        # "T13_class_1.0_cmDS.csv",
        # "T12_class_1.0_cmDS.csv",
        # "T11_class_1.0_cmDS.csv",
        # "T10_class_1.0_cmDS.csv",
        # "T9_class_1.0_cmDS.csv",
        # "T8_class_1.0_cmDS.csv",
        # "T7_class_1.0_cmDS.csv",
        # "T6_class_1.0_cmDS.csv",
        # "T05_class_1.0_cmDS.csv",
        # "T4_class_1.0_cmDS.csv",
        # "T3_class_1.0_cmDS.csv",
        # "T02_class_1.0_cmDS.csv",
        # "T1_class_1.0_cmDS.csv",

    ]

    for file in filenames:
        print(file)
        # try:
        parameters = dict(directory='../', fileset='test', input_point_cloud=None, model_filename='../model/model.pth',
                          batch_size=20, box_dimensions=[6, 6, 6], box_overlap=[0.5, 0.5, 0.5], min_points_per_box=1000,
                          max_points_per_box=20000, subsample=False, subsampling_min_spacing=0.01, num_procs=20,
                          noise_class=0, terrain_class=1, vegetation_class=2, cwd_class=3, stem_class=4,
                          coarse_grid_resolution=6, fine_grid_resolution=0.5, max_diameter=5, num_neighbours=3,
                          slice_thickness=0.15, slice_increment=0.05, slice_clustering_distance=0.1,
                          cleaned_measurement_radius=0.18, minimum_CCI=0.3, min_tree_volume=0.005,
                          Canopy_coverage_resolution=0.5, ground_veg_cutoff_height=3, canopy_mode='continuous',
                          Site='not_specified', PlotID='not_specified', UTM_zone_number=50, UTM_zone_letter=None,
                          UTM_is_north=False, run_from_start=1)

        parameters['input_point_cloud'] = file
        measure1 = MeasureTree(parameters)
        measure1.run_measurement_extraction()
        # except:
        #     print('exception',file)
