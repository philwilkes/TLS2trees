from abc import ABC
import torch
import torch_geometric
from torch_geometric.data import Dataset, DataLoader, Data
import numpy as np
import glob
import pandas as pd
from preprocessing import Preprocessing
from model import Net
from sklearn.neighbors import NearestNeighbors
from scipy import spatial
import os
import time
from file_handling import load_file, save_file


class TestingDataset(Dataset, ABC):
    def __init__(self, dataset_name, root_dir, points_per_box, device):
        super().__init__()
        self.filenames = glob.glob(root_dir + dataset_name + '*.npy')
        self.device = device
        self.points_per_box = points_per_box
        self.sem_seg_end_time = None
        self.output = None

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        point_cloud = np.load(self.filenames[index])
        pos = point_cloud[:, :3]
        extra_info = point_cloud[:, 3:6]
        pos = torch.from_numpy(pos.copy()).type(torch.float).to(self.device).requires_grad_(False)

        # Place sample at origin
        local_shift = torch.round(torch.mean(pos[:, :3], axis=0)).requires_grad_(False)
        pos = pos - local_shift
        extra_info = torch.from_numpy(extra_info.copy()).type(torch.float).to(self.device).requires_grad_(False)
        data = Data(pos=pos, x=None, extra_info=extra_info, local_shift=local_shift)
        return data


def assign_labels_to_original_point_cloud(original, labeled, label_index):
    print("Assigning segmentation labels to original point cloud...")
    kdtree = spatial.cKDTree(labeled[:, :3])
    labels = np.atleast_2d(labeled[kdtree.query(original[:, :3], k=2)[1][:, 1], label_index]).T
    original = np.hstack((original, labels))
    return original





def choose_most_confident_label(point_cloud, original_point_cloud):
    print("Choosing most confident labels...")
    neighbours = NearestNeighbors(n_neighbors=16, algorithm='kd_tree', metric='euclidean', radius=0.05).fit(
            point_cloud[:, :3])
    _, indices = neighbours.kneighbors(original_point_cloud[:, :3])
    labels = np.zeros((original_point_cloud.shape[0], 5))
    labels[:, 4] = np.argmax(np.median(point_cloud[indices][:, :, -4:], axis=1), axis=1)
    original_point_cloud = np.hstack((original_point_cloud, np.atleast_2d(labels[:, -1]).T))
    return original_point_cloud


class SemanticSegmentation:
    def __init__(self, parameters):
        self.parameters = parameters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.directory = self.parameters['directory']
        self.point_cloud_filename = self.parameters['input_point_cloud'][:-4] + '_out'
        self.make_folder_structure()
        self.output_dir = None
        self.sem_seg_start_time = None
        self.output_point_cloud = None
        self.sem_seg_end_time = None
        self.sem_seg_total_time = None
        self.output = None

    def make_folder_structure(self):
        if not os.path.isdir(self.directory + "data/original_point_clouds"):
            os.makedirs(self.directory + "data/original_point_clouds")
            print('Created "data/original_point_clouds" directory')

        if not os.path.isdir(self.directory + "data/working_directory/" + self.parameters['input_point_cloud'][:-4]):
            os.makedirs(self.directory + "data/working_directory/" + self.parameters['input_point_cloud'][:-4])
            print('Created "data/working_directory/' + self.parameters['input_point_cloud'][:-4])

        if not os.path.isdir(self.directory + "data/segmented_point_clouds"):
            os.makedirs(self.directory + "data/segmented_point_clouds")
            print('Created "data/segmented_point_clouds" directory')

        if not os.path.isdir(self.directory + "data/postprocessed_point_clouds"):
            os.makedirs(self.directory + "data/postprocessed_point_clouds")
            print('Created "data/postprocessed_point_clouds" directory')

        if not os.path.isdir(self.directory + "data/postprocessed_point_clouds/" + self.point_cloud_filename):
            os.makedirs(self.directory + "data/postprocessed_point_clouds/" + self.point_cloud_filename)
            print('Created "data/postprocessed_point_clouds" directory')
        self.output_dir = self.directory + "data/postprocessed_point_clouds/" + self.point_cloud_filename + '/'

    def run_preprocessing(self):
        preprocessing = Preprocessing(self.parameters)
        point_cloud = self.parameters['directory'] + "data/original_point_clouds/" + self.parameters[
            'input_point_cloud']
        print(point_cloud)
        preprocessing.load_point_cloud(filename=point_cloud)
        preprocessing.preprocess_point_cloud()
        return

    def inference(self):
        self.sem_seg_start_time = time.time()
        test_dataset = TestingDataset(dataset_name=self.parameters['fileset'],
                                      root_dir=self.parameters['directory'] + "data/working_directory/" + self.parameters['input_point_cloud'][:-4] + '/',
                                      points_per_box=self.parameters['max_points_per_box'],
                                      device=self.device)

        test_loader = DataLoader(test_dataset, batch_size=self.parameters['batch_size'], shuffle=False,
                                 num_workers=0, pin_memory=True)

        global_shift = np.loadtxt(
                self.parameters['directory'] + "data/working_directory/" + self.parameters['input_point_cloud'][
                                                                           :-4] + '/' + 'global_shift.csv',
                dtype='float64')
        print("Global shift:", global_shift)

        model = Net(num_classes=4).to(self.device)
        model.load_state_dict(torch.load(self.parameters['directory'] + 'model/' + self.parameters['model_filename']),
                              strict=False)
        model.eval()
        num_boxes = test_dataset.__len__()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                outshape = np.shape(data.extra_info.cpu().detach())[1]
                break
            self.output_point_cloud = np.zeros((0, 3 + outshape + 4))

            for i, data in enumerate(test_loader):
                print('\r' + str(i * self.parameters['batch_size']) + '/' + str(num_boxes))
                data = data.to(self.device)
                out = model(data)
                out = out.permute(2, 1, 0).squeeze()
                batches = np.unique(data.batch.cpu())
                out = torch.softmax(out.cpu().detach(), axis=1)
                pos = data.pos.cpu()
                extra_info_out = data.extra_info.cpu()
                output = np.hstack((pos, extra_info_out, out))
                for batch in batches:
                    outputb = np.asarray(output[data.batch.cpu() == batch])
                    outputb[:, :3] = outputb[:, :3] + np.asarray(data.local_shift.cpu())[3 * batch:3 + (3 * batch)]
                    self.output_point_cloud = np.vstack((self.output_point_cloud, outputb))
            print('\r' + str(num_boxes)+'/'+str(num_boxes))
        del outputb, out, batches, pos, extra_info_out, output  # clean up anything no longer needed to free RAM.

        print("Loading original point cloud...")
        original_point_cloud = np.array(pd.read_csv(
                self.directory + "data/original_point_clouds/" + self.parameters['input_point_cloud'][:-4] + '.csv',
                header=None, index_col=None, delim_whitespace=True))
        original_point_cloud = original_point_cloud[:, :3]
        original_point_cloud[:, :3] = original_point_cloud[:, :3] - global_shift

        if self.parameters['subsample']:
            original_point_cloud = subsample_point_cloud(original_point_cloud,
                                                         self.parameters['subsampling_min_spacing'])

        self.output = np.asarray(choose_most_confident_label(self.output_point_cloud, original_point_cloud), dtype='float64')
        self.output[:, :3] = self.output[:, :3] + global_shift
        print("Saving...")
        save_file(self.parameters['directory'] + "data/segmented_point_clouds/" + self.parameters['input_point_cloud'][:-4] + '_out' + '.las',
                  self.output)

        print("Saved")
        self.sem_seg_end_time = time.time()
        self.sem_seg_total_time = self.sem_seg_end_time - self.sem_seg_start_time
        np.savetxt(self.parameters['directory'] + "data/postprocessed_point_clouds/" + self.parameters['input_point_cloud'][:-4] + '_out/' + "semantic_segmentation_time.csv",
                   np.array([self.sem_seg_total_time]))
        print("Semantic segmentation took", self.sem_seg_total_time, 's')
        print("Semantic segmentation done")
