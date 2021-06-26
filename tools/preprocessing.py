import numpy as np
import time
import glob
import random
import warnings
import pandas as pd
from copy import deepcopy
import matplotlib
from scipy import spatial
from sklearn.neighbors import NearestNeighbors
import threading
from tools import load_file, save_file, make_folder_structure
import os


class Preprocessing:
    def __init__(self, parameters):
        self.preprocessing_time_start = time.time()
        self.parameters = parameters
        self.filename = self.parameters['input_point_cloud'].replace('\\', '/')
        self.directory = os.path.dirname(os.path.realpath(self.filename)).replace('\\', '/') + '/'
        self.filename = self.filename.split('/')[-1]
        self.box_dimensions = np.array(self.parameters['box_dimensions'])
        self.box_overlap = np.array(self.parameters['box_overlap'])
        self.min_points_per_box = self.parameters['min_points_per_box']
        self.max_points_per_box = self.parameters['max_points_per_box']
        self.subsample = self.parameters['subsample']
        self.subsampling_min_spacing = self.parameters['subsampling_min_spacing']
        self.num_procs = parameters['num_procs']

        self.output_dir, self.working_dir = make_folder_structure(self.directory + self.filename)

        self.point_cloud, headers = load_file(self.directory + self.filename,
                                              self.parameters['plot_centre'],
                                              self.parameters['plot_radius'])

        if self.parameters['plot_radius'] != 0:
            save_file(self.output_dir + self.filename[:-4] + '_' + str(self.parameters['plot_radius']) + '_m_crop.las',
                      self.point_cloud)

        self.point_cloud = self.point_cloud[:, :3]  # Trims off unneeded dimensions if present.

        if self.parameters['low_resolution_point_cloud_hack_mode']:
            print('Using low resolution point cloud hack mode...')
            print('Original point cloud shape:', self.point_cloud.shape)
            for i in range(self.parameters['low_resolution_point_cloud_hack_mode']):
                duplicated = deepcopy(self.point_cloud)
                duplicated[:, :3] = duplicated[:, :3] + np.random.normal(-0.01, 0.01, size=(duplicated.shape[0], 3))
                self.point_cloud = np.vstack((self.point_cloud, duplicated))

            print('Hacked point cloud shape:', self.point_cloud.shape)

        self.global_shift = [np.mean(self.point_cloud[:, 0]), np.mean(self.point_cloud[:, 1]),
                             np.mean(self.point_cloud[:, 2])]

        self.point_cloud[:, :3] = self.point_cloud[:, :3] - self.global_shift

        print('Global Shift:', self.global_shift, 'm')
        np.savetxt(self.working_dir + 'global_shift.csv', self.global_shift)

    @staticmethod
    def threaded_boxes(point_cloud, box_size, min_points_per_box, max_points_per_box, path, id_offset, point_divisions):

        box_centre_mins = point_divisions - 0.5 * box_size
        box_centre_maxes = point_divisions + 0.5 * box_size
        i = 0
        pds = len(point_divisions)
        while i < pds:
            box = point_cloud
            box = box[np.logical_and(np.logical_and(np.logical_and(box[:, 0] >= box_centre_mins[i, 0],
                                                                   box[:, 0] < box_centre_maxes[i, 0]),
                                                    np.logical_and(box[:, 1] >= box_centre_mins[i, 1],
                                                                   box[:, 1] < box_centre_maxes[i, 1])),
                                     np.logical_and(box[:, 2] >= box_centre_mins[i, 2],
                                                    box[:, 2] < box_centre_maxes[i, 2]))]

            if box.shape[0] > min_points_per_box:
                if box.shape[0] > max_points_per_box:
                    indices = list(range(0, box.shape[0]))
                    random.shuffle(indices)
                    random.shuffle(indices)
                    box = box[indices[:max_points_per_box], :]
                    box = np.asarray(box, dtype='float64')

                box[:, :3] = box[:, :3]
                np.save(path + str(id_offset + i).zfill(7) + '.npy', box)
            i += 1
        return 1

    def preprocess_point_cloud(self):
        print("Pre-processing point cloud...")
        point_cloud = self.point_cloud  # [self.point_cloud[:,4]!=5]
        Xmax = np.max(point_cloud[:, 0])
        Xmin = np.min(point_cloud[:, 0])
        Ymax = np.max(point_cloud[:, 1])
        Ymin = np.min(point_cloud[:, 1])
        Zmax = np.max(point_cloud[:, 2])
        Zmin = np.min(point_cloud[:, 2])

        X_range = Xmax - Xmin
        Y_range = Ymax - Ymin
        Z_range = Zmax - Zmin

        num_boxes_x = int(np.ceil(X_range / self.box_dimensions[0]))
        num_boxes_y = int(np.ceil(Y_range / self.box_dimensions[1]))
        num_boxes_z = int(np.ceil(Z_range / self.box_dimensions[2]))
        print(self.box_overlap)
        x_vals = np.linspace(Xmin, Xmin + (num_boxes_x * self.box_dimensions[0]),
                             int(num_boxes_x / (1 - self.box_overlap[0])) + 1)
        y_vals = np.linspace(Ymin, Ymin + (num_boxes_y * self.box_dimensions[1]),
                             int(num_boxes_y / (1 - self.box_overlap[1])) + 1)
        z_vals = np.linspace(Zmin, Zmin + (num_boxes_z * self.box_dimensions[2]),
                             int(num_boxes_z / (1 - self.box_overlap[2])) + 1)

        box_centres = np.vstack(np.meshgrid(x_vals, y_vals, z_vals)).reshape(3, -1).T
        point_divisions = []

        for thread in range(self.num_procs):
            point_divisions.append([])

        points_to_assign = box_centres

        while points_to_assign.shape[0] > 0:
            for i in range(self.num_procs):
                point_divisions[i].append(points_to_assign[0, :])
                points_to_assign = points_to_assign[1:]
                if points_to_assign.shape[0] == 0:
                    break
        threads = []
        prev_id_offset = 0
        for thread in range(self.num_procs):
            id_offset = 0
            for t in range(thread):
                id_offset = id_offset + len(point_divisions[t])
            print('Thread:', thread, prev_id_offset, id_offset)
            prev_id_offset = id_offset
            t = threading.Thread(target=Preprocessing.threaded_boxes, args=(self.point_cloud,
                                                                            self.box_dimensions,
                                                                            self.min_points_per_box,
                                                                            self.max_points_per_box,
                                                                            self.working_dir,
                                                                            id_offset,
                                                                            point_divisions[thread],))
            threads.append(t)

        for x in threads:
            x.start()

        for x in threads:
            x.join()

        self.preprocessing_time_end = time.time()
        self.preprocessing_time_total = self.preprocessing_time_end - self.preprocessing_time_start
        print("Preprocessing took", self.preprocessing_time_total, 's')
        times = pd.DataFrame(np.array([[self.preprocessing_time_total, 0, 0, 0]]), columns=['Preprocessing_Time (s)',
                                                                                            'Semantic_Segmentation_Time (s)',
                                                                                            'Post_processing_time (s)',
                                                                                            'Measurement Time (s)'])
        times.to_csv(self.output_dir + 'run_times.csv', index=False)
        print("Preprocessing done\n")
