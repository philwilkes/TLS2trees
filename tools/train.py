from abc import ABC
import torch
import torch.nn.functional as F
from torch_geometric.nn import knn_interpolate
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.data import Dataset, DataLoader, Data
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
import numpy as np
import glob
import pandas as pd
import random
import math
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import os
from sklearn.neighbors import NearestNeighbors
from tools import subsample_point_cloud, load_file, save_file


def augmentations(X, no_terrain, no_cwd):
    def rotate_3d(points, rotations):
        rotations[0] = math.radians(rotations[0])
        rotations[1] = math.radians(rotations[1])
        rotations[2] = math.radians(rotations[2])

        roll_mat = np.array([[1, 0, 0],
                             [0, math.cos(rotations[0]), -math.sin(rotations[0])],
                             [0, math.sin(rotations[0]), math.cos(rotations[0])]])

        pitch_mat = np.array([[math.cos(rotations[1]), 0, math.sin(rotations[1])],
                              [0, 1, 0],
                              [-math.sin(rotations[1]), 0, math.cos(rotations[1])]])

        yaw_mat = np.array([[math.cos(rotations[2]), -math.sin(rotations[2]), 0],
                            [math.sin(rotations[2]), math.cos(rotations[2]), 0],
                            [0, 0, 1]])

        points[:, :3] = np.matmul(np.matmul(np.matmul(points[:, :3], roll_mat), pitch_mat), yaw_mat)
        return points

    def random_scale_change(points, min_multiplier, max_multiplier):
        points = points * np.random.uniform(min_multiplier, max_multiplier)
        # points[:,:3] = points[:,:3] + np.random.uniform(-5,5,size=(1,3))
        return points

    def random_point_removal(self,points):
        idx = list(range(np.shape(points)[0]))
        random.shuffle(idx)
        random.shuffle(idx)
        return points[:int(np.random.uniform(0.5,1)*np.shape(points)[0]), :]

    def random_noise_addition(points):
        # 50% chance per sample of adding noise.
        random_noise_std_dev = np.random.uniform(0.01, 0.025)
        if np.random.uniform(0, 1) >= 0.5:
            points = points + np.random.normal(0, random_noise_std_dev, size=(np.shape(points)[0], 3))
        return points

    if no_terrain and no_cwd:
        rotations = [np.random.uniform(-90, 90), np.random.uniform(-90, 90),
                     np.random.uniform(-180, 180)]  # np.random.randint(0,3)*90]
    else:
        rotations = [np.random.uniform(-5, 5), np.random.uniform(-5, 5),
                     np.random.uniform(-180, 180)]  # np.random.randint(0,3)*90]
    X = rotate_3d(X, rotations)
    X = random_scale_change(X, 0.8, 1.2)
    # if np.random.uniform(0,1) >= 0.5:
    # X = subsample_point_cloud(X,y,np.random.uniform(0.01,0.025))
    X = random_noise_addition(X)
    return X


class TrainingDataset(Dataset, ABC):
    def __init__(self, dataset_name, root_dir, points_per_box, device):
        super().__init__()
        self.filenames = glob.glob(root_dir + dataset_name + '*.npy')
        print(root_dir + 'data/' + dataset_name + '*.csv')
        self.points_per_box = points_per_box
        self.label_index = 6
        self.device = device

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        point_cloud = np.load(self.filenames[index])
        if point_cloud.shape[0] > self.points_per_box:
            l = list(range(point_cloud.shape[0]))
            random.shuffle(l)
            point_cloud = point_cloud[l[:self.points_per_box]]

        X = point_cloud[:, :3]
        y = point_cloud[:, self.label_index]
        X = augmentations(X, np.all(y != 0), np.all(y != 2))  # Check if any terrain or CWD is present
        if np.all(y != 0):
            y[y == 2] = 3  # if no ground is present, CWD is relabelled as stem.
        X = torch.from_numpy(X.copy()).type(torch.float).to(self.device)
        y = torch.from_numpy(y.copy()).type(torch.long).to(self.device)

        # Place sample at origin
        global_shift = torch.mean(X[:, :3], axis=0)
        X = X - global_shift

        data = Data(pos=X, x=None, y=y)
        return data


class TestingDataset(Dataset, ABC):
    def __init__(self, dataset_name, root_dir, points_per_box, device):
        super().__init__()
        self.filenames = glob.glob(root_dir + dataset_name + '*.npy')
        self.points_per_box = points_per_box
        self.label_index = 6
        self.device = device

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        point_cloud = np.load(self.filenames[index])
        X = point_cloud[:, :3]
        y = point_cloud[:, self.label_index]
        X = torch.from_numpy(X.copy()).type(torch.float).to(self.device)
        y = torch.from_numpy(y.copy()).type(torch.long).to(self.device)

        # Place sample at origin
        global_shift = torch.mean(X[:, :3], axis=0)
        X = X - global_shift

        data = Data(pos=X, x=None, y=y)
        return data


class SAModule(torch.nn.Module, ABC):
    def __init__(self, ratio, r, NN):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(NN)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module, ABC):
    def __init__(self, NN):
        super(GlobalSAModule, self).__init__()
        self.NN = NN

    def forward(self, x, pos, batch):
        x = self.NN(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


def MLP(channels, batch_norm=True):
    return Seq(*[
            Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
            for i in range(1, len(channels))
    ])


class FPModule(torch.nn.Module, ABC):
    def __init__(self, k, NN):
        super(FPModule, self).__init__()
        self.k = k
        self.NN = NN

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.NN(x)
        return x, pos_skip, batch_skip


class Net(torch.nn.Module, ABC):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.sa1_module = SAModule(0.1, 0.2, MLP([3, 128, 256, 512]))
        self.sa2_module = SAModule(0.05, 0.4, MLP([512 + 3, 512, 1024, 1024]))
        self.sa3_module = GlobalSAModule(MLP([1024 + 3, 1024, 2048, 2048]))

        self.fp3_module = FPModule(1, MLP([3072, 1024, 1024]))
        self.fp2_module = FPModule(3, MLP([1536, 1024, 1024]))
        self.fp1_module = FPModule(3, MLP([1024, 1024, 1024]))

        self.conv1 = torch.nn.Conv1d(1024, 1024, 1)
        self.conv2 = torch.nn.Conv1d(1024, num_classes, 1)
        self.drop1 = torch.nn.Dropout(0.2)
        self.bn1 = torch.nn.BatchNorm1d(1024)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)

        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)
        x = x.unsqueeze(dim=0)
        x = x.permute(0, 2, 1)
        x = self.drop1(F.relu(self.bn1(self.conv1(x))))
        x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        return x


if __name__ == '__main__':
    from preprocessing import Preprocessing

    parameters = dict(directory='../', load_existing_model=1, num_epochs=2000, learning_rate=0.000025, fileset=None,
                      input_point_cloud=None, model_filename='model7_no_noise_smaller_batches.pth',
                      box_dimensions=[6, 6, 6], box_overlap=[0.75, 0.75, 0.75], min_points_per_box=1000,
                      max_points_per_box=20000, subsample=False, subsampling_min_spacing=0.025, num_procs=20,
                      batch_size=8)

    # if 0:
    #     parameters['fileset'] = 'train_aug'
    #     parameters['input_point_cloud'] = 'final_train_aug.csv'
    #     folder_structure(parameters)
    #     preprocessing = Preprocessing(parameters)
    #     preprocessing.load_point_cloud(
    #         filename=parameters['directory'] + 'data/original_point_clouds/' + parameters['input_point_cloud'])
    #     preprocessing.preprocess_point_cloud()
    # if 0:
    #     parameters['fileset'] = 'test_aug'
    #     parameters['input_point_cloud'] = 'final_validation_aug.csv'
    #     folder_structure(parameters)
    #     preprocessing = Preprocessing(parameters)
    #     preprocessing.load_point_cloud(
    #         filename=parameters['directory'] + 'data/original_point_clouds/' + parameters['input_point_cloud'])
    #     preprocessing.preprocess_point_cloud()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parameters['fileset'] = 'train_aug'
    parameters['input_point_cloud'] = 'final_train_aug.csv'
    train_dataset = TrainingDataset(dataset_name='train_aug',
                                    root_dir=parameters['directory'] + "data/working_directory/" + parameters[
                                                                                                       'input_point_cloud'][
                                                                                                   :-4] + '/',
                                    points_per_box=parameters['max_points_per_box'],
                                    device=device)

    train_loader = DataLoader(train_dataset, batch_size=parameters['batch_size'], shuffle=True,
                              num_workers=0, pin_memory=True)

    parameters['fileset'] = 'test_aug'
    parameters['input_point_cloud'] = 'final_validation_aug.csv'
    test_dataset = TestingDataset(dataset_name='test_aug',
                                  root_dir=parameters['directory'] + "data/working_directory/" + parameters[
                                                                                                     'input_point_cloud'][
                                                                                                 :-4] + '/',
                                  points_per_box=parameters['max_points_per_box'],
                                  device=device)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True,
                             num_workers=0, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(num_classes=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters['learning_rate'])


    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']


    if parameters['load_existing_model']:
        print('Loading existing model...')
        model.load_state_dict(torch.load(parameters['directory'] + 'model/' + parameters['model_filename']),
                              strict=False)
    optimiser = optim.Adam(model.parameters(), lr=parameters['learning_rate'])

    if parameters['load_existing_model']:
        try:
            history_array = np.array(pd.read_csv(
                parameters['directory'] + 'model/' + 'history_array' + parameters['model_filename'][:-4] + '.csv',
                header=None, index_col=None, delim_whitespace=True))
            print("Loaded past history array...")
            past_i = history_array[-1, 0]

        except:
            history_array = np.zeros((0, 6))
            pd.DataFrame(history_array).to_csv(
                parameters['directory'] + 'model/' + 'history_array' + parameters['model_filename'][:-4] + '.csv',
                header=None, index=None, sep=' ')
            print("Created new history array...")
            try:
                past_i = history_array[-1, 0]
            except:
                past_i = 0
    else:
        history_array = np.zeros((0, 6))
        pd.DataFrame(history_array).to_csv(
            parameters['directory'] + 'model/' + 'history_array' + parameters['model_filename'][:-4] + '.csv',
            header=None, index=None, sep=' ')

        print("Created new history array...")
        past_i = 0

    num_epochs_seen = 200  # from 22, changed weights from [0.9,0.9,0.9,0.9,1] to [0.85,0.85,0.85,1,0.85] to boost up CWD class more.,
    samples_seen = past_i
    model.train()
    train_acc = -1
    test_acc = -1
    total_loss = correct_nodes = total_nodes = total_testloss = correct_testnodes = total_testnodes = 0
    for epoch in range(0, parameters['num_epochs']):
        for i, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            out = out.permute(2, 1, 0).squeeze()

            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(out, data.y)
            loss.backward()

            optimizer.step()
            total_loss += loss.item()
            out = np.argmax(out.cpu().detach(), axis=1)
            correct_nodes += (np.array(out) == np.array(data.y.cpu())).sum()
            total_nodes += data.num_nodes
            train_acc = correct_nodes / total_nodes

            # if (i + 1) % 50 == 0:
            #     print(i+1)
            # pd.DataFrame(np.hstack((data.pos.cpu(),np.atleast_2d(out).T,np.atleast_2d(data.y.cpu()).T))).to_csv(parameters['directory']+'most_recent__train_out.csv',header=None,index=None,sep=' ')

            if (i + 1) % 1000 == 0:
                print("Saving model...")
                torch.save(model.state_dict(), parameters['directory'] + 'model/' + parameters['model_filename'])
                print("Saved model...")

            if (i + 1) % 100 == 0:
                history_array = np.vstack((history_array, np.array([[samples_seen + i,  # 0
                                                                     total_loss / 100,
                                                                     train_acc,
                                                                     -1,
                                                                     -1,
                                                                     get_lr(optimizer)]])))
                pd.DataFrame(history_array).to_csv(
                    parameters['directory'] + 'model/' + 'history_array' + parameters['model_filename'][:-4] + '.csv',
                    header=None, index=None, sep=' ')

                print(f'[{i + 1}/{len(train_loader)}] Loss: {total_loss / 20:.4f} 'f'Train Acc: {train_acc:.4f}')
                total_loss = correct_nodes = total_nodes = 0

            if (i + 1) % 500 == 0:
                model.eval()
                visualiser = np.zeros((0, 5))
                for j, data in enumerate(test_loader):
                    data = data.to(device)
                    out = model(data)
                    out = out.permute(2, 1, 0).squeeze()
                    testloss = criterion(out, data.y)
                    # testloss = F.nll_loss(out, data.y)
                    out = np.argmax(out.cpu().detach(), axis=1)
                    total_testloss += testloss.item()
                    correct_testnodes += (np.array(out) == np.array(data.y.cpu())).sum()

                    total_testnodes += data.num_nodes
                    test_acc = correct_testnodes / total_testnodes
                    if (j + 1) % 10 == 0:
                        visualiser = np.vstack((visualiser, np.hstack((data.pos.cpu() + np.array([j, 0, 0]),
                                                                       np.atleast_2d(out).T,
                                                                       np.atleast_2d(data.y.cpu()).T))))

                    if (j + 1) % 100 == 0:
                        pd.DataFrame(visualiser).to_csv(parameters['directory'] + 'most_recent_out.csv', header=None,
                                                        index=None, sep=' ')
                        break

                print(f'[{i + 1}/{len(train_loader)}] Test Loss: {total_testloss / 20:.4f} 'f'Test Acc: {test_acc:.4f}')

                history_array = np.vstack((history_array, np.array([[samples_seen + i,  # 0
                                                                     -1,
                                                                     -1,
                                                                     total_testloss / 100,
                                                                     test_acc,
                                                                     get_lr(optimizer)]])))
                total_testloss = correct_testnodes = total_testnodes = 0
                pd.DataFrame(history_array).to_csv(
                    parameters['directory'] + 'model/' + 'history_array' + parameters['model_filename'] + '.csv',
                    header=None, index=None, sep=' ')
                model.train()
                samples_seen = i + samples_seen
        num_epochs_seen += 1
        torch.save(model.state_dict(),
                   parameters['directory'] + 'model/' + parameters['model_filename'][:-4] + '_' + str(
                       num_epochs_seen) + '.pth')
        # pd.DataFrame(history_array).to_csv(parameters['directory']+'model/'+'history_array' +parameters['model_filename'][:-4]+'.csv',header=None, index=None,sep=' ')
        # scheduler.step()
