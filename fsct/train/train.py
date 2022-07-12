import os
import glob
import random
import threading
import argparse
import tempfile

import itertools
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import NearestNeighbors

from fsct.tools import *
from fsct.model import Net
from fsct.train.other_parameters import other_parameters

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Dataset, DataLoader, Data
from torch.multiprocessing import Pool, Process, set_start_method

try: # required on JASMIN
     set_start_method('spawn')
except RuntimeError:
    pass

class TrainingDataset:
    def __init__(self, params):
        super().__init__()
        self.filenames = glob.glob(os.path.join(params.dtrain, "*.npy"))
        self.params = params

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        
        point_cloud = np.load(self.filenames[index])
        
        if point_cloud.shape[0] > self.params.max_sample_points:
            shuffle_index = list(range(point_cloud.shape[0]))
            random.shuffle(shuffle_index)
            point_cloud = point_cloud[shuffle_index[: self.params.max_sample_points]]

        x, y = point_cloud[:, :3], point_cloud[:, 3]
        x, y = augmentations(x, y, self.params.min_sample_points)
        if np.all(y != 0): y[y == 2] = 3  # if no ground is present, CWD is relabelled as stem.
        x = torch.from_numpy(x.copy()).type(torch.float).to(self.params.device)
        y = torch.from_numpy(y.copy()).type(torch.long).to(self.params.device)

        # Place sample at origin
        x -= torch.mean(x[:, :3], axis=0)

        return Data(pos=x, x=None, y=y)

class ValidationDataset:
    def __init__(self, params):
        super().__init__()
        self.filenames = glob.glob(os.path.join(params.dvalidate, "*.npy"))
        self.params = params

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        with torch.no_grad():
            point_cloud = np.load(self.filenames[index])
            x, y = point_cloud[:, :3], point_cloud[:, 3]
            x = torch.from_numpy(x.copy()).type(torch.float).to(self.params.device)
            y = torch.from_numpy(y.copy()).type(torch.long).to(self.params.device)

            # Place sample at origin
            x -= torch.mean(x[:, :3], axis=0)

            return Data(pos=x, x=None, y=y)
        
def augmentations(x, y, min_sample_points):
    
    def rotate_3d(points, rotations):
        rotations[0] = np.radians(rotations[0])
        rotations[1] = np.radians(rotations[1])
        rotations[2] = np.radians(rotations[2])

        roll_mat = np.array(
            [
                [1, 0, 0],
                [0, np.cos(rotations[0]), -np.sin(rotations[0])],
                [0, np.sin(rotations[0]), np.cos(rotations[0])],
            ]
        )

        pitch_mat = np.array(
            [
                [np.cos(rotations[1]), 0, np.sin(rotations[1])],
                [0, 1, 0],
                [-np.sin(rotations[1]), 0, np.cos(rotations[1])],
            ]
        )

        yaw_mat = np.array(
            [
                [np.cos(rotations[2]), -np.sin(rotations[2]), 0],
                [np.sin(rotations[2]), np.cos(rotations[2]), 0],
                [0, 0, 1],
            ]
        )

        points[:, :3] = np.matmul(np.matmul(np.matmul(points[:, :3], roll_mat), pitch_mat), yaw_mat)
        return points

    def random_scale_change(points, min_multiplier, max_multiplier):
        points = points * np.random.uniform(min_multiplier, max_multiplier)
        return points

    def random_point_removal(x, y, min_sample_points):
        indices = np.arange(np.shape(x)[0])
        np.random.shuffle(indices)
        num_points_to_keep = min_sample_points + int(np.random.uniform(0, 0.95) * (np.shape(x)[0] - min_sample_points))
        indices = indices[:num_points_to_keep]
        return x[indices], y[indices]

    def random_noise_addition(points):
        # 50% chance per sample of adding noise.
        random_noise_std_dev = np.random.uniform(0.01, 0.025)
        if np.random.uniform(0, 1) >= 0.5:
            points = points + np.random.normal(0, random_noise_std_dev, size=(np.shape(points)[0], 3))
        return points

    if np.all(y != 0) and np.all(
        y != 2
    ):  # if no terrain or CWD are present, it's ok to rotate extremely. Terrain shouldn't be above stems or CWD.
        rotations = [np.random.uniform(-90, 90), np.random.uniform(-90, 90), np.random.uniform(-180, 180)]
    else:
        rotations = [np.random.uniform(-25, 25), np.random.uniform(-25, 25), np.random.uniform(-180, 180)]
    x = rotate_3d(x, rotations)
    x = random_scale_change(x, 0.8, 1.2)
    if np.random.uniform(0, 1) >= 0.5 and x.shape[0] > min_sample_points:
        x, y = subsample_point_cloud(x, y, np.random.uniform(0.01, 0.025), min_sample_points)

    if np.random.uniform(0, 1) >= 0.8 and x.shape[0] > min_sample_points:
        x, y = random_point_removal(x, y, min_sample_points)

    x = random_noise_addition(x)
    
    return x, y

def subsample_point_cloud(x, y, min_spacing, min_sample_points):
    x = np.hstack((x, np.atleast_2d(y).T))
    neighbours = NearestNeighbors(n_neighbors=2, algorithm="kd_tree", metric="euclidean").fit(x[:, :3])
    distances, indices = neighbours.kneighbors(x[:, :3])
    x_keep = x[distances[:, 1] >= min_spacing]
    i1 = [distances[:, 1] < min_spacing][0]
    i2 = [x[indices[:, 0], 2] < x[indices[:, 1], 2]][0]
    x_check = x[np.logical_and(i1, i2)]

    while np.shape(x_check)[0] > 1:
        neighbours = NearestNeighbors(n_neighbors=2, algorithm="kd_tree", metric="euclidean").fit(x_check[:, :3])
        distances, indices = neighbours.kneighbors(x_check[:, :3])
        x_keep = np.vstack((x_keep, x_check[distances[:, 1] >= min_spacing, :]))
        i1 = [distances[:, 1] < min_spacing][0]
        i2 = [x_check[indices[:, 0], 2] < x_check[indices[:, 1], 2]][0]
        x_check = x_check[np.logical_and(i1, i2)]
    if x_keep.shape[0] >= min_sample_points:
        return x_keep[:, :3], x_keep[:, 3]
    else:
        return x[:, :3], x[:, 3]
    

def run_training(params):

    if params.dl_cpu_cores == 0:
        print("Using default number of CPU cores (all of them).")
        params.dl_cpu_cores = os.cpu_count()
    if params.verbose: print(f'Running deep learning using {params.dl_cpu_cores}/{os.cpu_count()} CPU cores')

    train_dataset = TrainingDataset(params)
    if len(train_dataset) == 0: raise Exception("No training samples found.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=params.train_batch_size,
        shuffle=True,
        num_workers=params.dl_cpu_cores,
        drop_last=True,
    )

    if params.validate:
        validation_dataset = ValidationDataset(params)

        validation_loader = DataLoader(
            validation_dataset,
            batch_size=params.train_batch_size,
            shuffle=True,
            num_workers=params.dl_cpu_cores,
            drop_last=True,
        )

    # create model instance, reload existing model and other stuff
    model = Net(num_classes=4).to(params.device)
    if os.path.isfile(params.model):
        if params.verbose: print("Loading existing model...")
        model.load_state_dict(torch.load(params.model), strict=False) # load exiting model
        # also load training history csv
        if os.path.isfile(params.out + '.training_history.csv'):
            params.train_history = pd.read_csv(f'{params.out}.training_history.csv')
    else:
        params.train_history = pd.DataFrame(columns=['epoch', 'epoch_loss', 'epoch_acc', 'val_epoch_loss', 'val_epoch_acc'])
    model = model.to(params.device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params.learning_rate)
    criterion = nn.CrossEntropyLoss()
    val_epoch_acc, val_epoch_loss = 0, 0

    # train model 
    for epoch in tqdm(range(params.iterations), total=params.iterations):
        print("=====================================================================")
        print("EPOCH ", epoch)
        # TRAINING
        model.train()
        running_loss = 0.0
        running_acc = 0
        running_point_cloud_vis = np.zeros((0, 5))
        for i, data in enumerate(train_loader):
            data.pos = data.pos.to(params.device)

            data.y = torch.unsqueeze(data.y, 0).to(params.device)
            outputs = model(data)
            loss = criterion(outputs, data.y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.detach().item()
            running_acc += torch.sum(preds == data.y.data).item() / data.y.shape[1]
            running_point_cloud_vis = np.vstack(
                (
                    running_point_cloud_vis,
                    np.hstack((data.pos.cpu() + np.array([i * 7, 0, 0]), data.y.cpu().T, preds.cpu().T)),
                )
            )
            if i % 5 == 0:
                print(
                    "Train sample accuracy: ",
                    np.around(running_acc / (i + 1), 4),
                    ", Loss: ",
                    np.around(running_loss / (i + 1), 4),
                )

                if params.generate_point_cloud_vis:
                    running_point_cloud_vis = pd.DataFrame(running_point_cloud_vis, 
                                                           columns=['x', 'y', 'z', 'label', 'pred'])
                    ply_io.write_ply(f'{params.out}.{epoch}.ply', running_point_cloud_vis)

        # write training history to file
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_acc / len(train_loader)
        params.train_history.loc[len(params.train_history)] = [epoch, epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc]
        params.train_history.to_csv(f'{params.out}.training_history.csv', header=True, index=False)
        if params.verbose: print(f'Train epoch accuracy: {np.around(epoch_acc, 4)} Loss: {np.around(epoch_loss, 4)}\n')

        # VALIDATION
        if params.validate:
            if params.verbose: print("Validation")
            model.eval()
            running_loss = 0.0
            running_acc = 0
            i = 0
            for data in validation_loader:
                data.pos = data.pos.to(params.device)
                data.y = torch.unsqueeze(data.y, 0).to(params.device)

                outputs = model(data)
                loss = criterion(outputs, data.y)

                _, preds = torch.max(outputs, 1)
                running_loss += loss.detach().item()
                running_acc += torch.sum(preds == data.y.data).item() / data.y.shape[1]
                if i % 5 == 0:
                    print(f'Validation sample accuracy: {np.around(running_acc / (i + 1), 4)} Loss: {np.around(running_loss / (i + 1), 4)}')

                i += 1

            # write validation history to file
            val_epoch_loss = running_loss / len(validation_loader)
            val_epoch_acc = running_acc / len(validation_loader)
            params.train_history.loc[len(params.train_history)] = [epoch, epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc]
            params.train_history.to_csv(f'{params.out}.training_history.csv', header=True, index=False)
            print(f'Validation epoch accuracy: {np.around(val_epoch_acc, 4)} Loss: {np.around(val_epoch_loss, 4)}')
            print("=====================================================================")
        
        # save model
        torch.save(model.state_dict(), os.path.join(params.model))
            
            
def preprocessing_setup(dataset, out_dir, params):

    if params.verbose: print(f'preprocessing {dataset}') 
    if isinstance(dataset, str):
        if os.path.isfile(dataset): dataset = [dataset]   
        elif os.path.isdir(dataset): dataset = glob.glob(os.path.join(dataset, '*.ply'))
        else: raise Exception(f'{dataset} is not a .ply or a directory')        
    elif isinstance(dataset, list):
        pass
    else:
        raise Exception(f'{dataset} is not a .ply, a directory or list')
                                                         
    if len(dataset) == 0: raise Exception(f'no .ply files in {dataset}')

    for pc in dataset:
        pc = ply_io.read_ply(pc)
        if params.label not in pc.columns: raise Exception(f'{params.label} not in pc fields, available columns are {", ".join([c for c in pc.columns[3:]])}')
        chunk_pc(pc.rename(columns={params.label:'label'}), out_dir, params)

if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', '-t', nargs='*', default='', type=str, help='path to training point cloud')
    parser.add_argument('--validate', '-v', nargs='*', default=False, type=str, help='path to validation point clouds')
    parser.add_argument('--chunks-dir', default='', type=str, help='path to chunks directory')
    
    parser.add_argument('--label', default='label', type=str, help='name of "label" field in input')
    parser.add_argument('--iterations', default=200, type=int, help='number of training iterations to run')
    
    # model arguments
    parser.add_argument('--model', '-m', type=str, required=True, help='output directory')
    
    # preprocessing and reading chunks
    parser.add_argument('--save-chunks-to', default=None, help='directory to save chunks to, otherwise saved to a temporaray directory')

    # if applying to tiled data
    parser.add_argument('--tile-index', default='', type=str, help='path to tile index in space delimited format "TILE X Y"')
    parser.add_argument('--buffer', default=0, type=float, help='included data from neighbouring tiles')
                              
    parser.add_argument('--verbose', action='store_true', help="print stuff")

    params = parser.parse_args()
    
    for k, v in other_parameters.items(): setattr(params, k, v) 
    
    if not torch.cuda.is_available() and params.device == 'cuda':
        print('NVIDIA graphics card is not available - setting device to CPU')
        params.device = 'cpu'
    
    if params.cpu_cores == 0:
        params.cpu_cores = os.cpu_count()
        if params.verbose: print(f'Processing using {params.cpu_cores}/{os.cpu_count()} CPU cores')
    
    if params.chunks_dir != '':
        if not os.path.isdir(params.chunks_dir):
            raise Exception(f'no such directory: {params.chunks_dir}')
        if not os.path.isdir(os.path.join(params.chunks_dir, 'train')):
            raise Exception(f'no such directory: {os.path.join(params.chunks_dir, "train")}')
        if not os.path.isdir(os.path.join(params.chunks_dir, 'validate')):
            raise Exception(f'no such directory: {os.path.join(params.chunks_dir, "validate")}')  
        params.dtrain = os.path.join(params.chunks_dir, 'train')
        params.dvalidate = os.path.join(params.chunks_dir, 'validate')
        if len(glob.glob(os.path.join(params.dtrain, '*.npy'))) == 0: raise Exception(f'no .npy files in {params.train}')
        if len(glob.glob(os.path.join(params.dvalidate, '*.npy'))) == 0: raise Exception(f'no .npy files in {params.validate}')
    else: # requires preprocessing
        if params.save_chunks_to == None:
            params.chunks_dir = tempfile.TemporaryDirectory().name
        else: 
            params.chunks_dir = params.save_chunks_to
        params.dtrain = os.path.join(params.chunks_dir, 'train')
        preprocessing_setup(params.train, params.dtrain, params)
        
        if params.validate:
            params.dvalidate = os.path.join(params.chunks_dir, 'validate')
            preprocessing_setup(params.validate, params.dvalidate, params)
                          
    params.out = os.path.splitext(params.model)[0]

    run_training(params)
