import torch
import torch.nn.functional as F
# import torch_geometric.transforms as T
from torch_geometric.nn import knn_interpolate
from torch_geometric.utils import intersection_and_union as i_and_u
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.data import Dataset, DataLoader, Data
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
import numpy as np
import glob
import pandas as pd
import random
import math
from preprocessing import Preprocessing
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
from torch.optim.lr_scheduler import ExponentialLR

class TrainingDataset(Dataset):
    def __init__(self, training_set_name, root_dir, points_per_box):
        self.filenames = glob.glob(root_dir+'data/'+training_set_name+'*.csv')
        print(root_dir+'data/'+training_set_name+'*.csv')
        self.points_per_box = points_per_box
        self.label_index = 6
    def __len__(self):
        return len(self.filenames)
        
    def __getitem__(self,index):
        point_cloud = np.array(pd.read_csv(self.filenames[index],header=None,index_col=None,delim_whitespace=True))
        
        if point_cloud.shape[0] > self.points_per_box:
            l = list(range(point_cloud.shape[0]))
            random.shuffle(l)
            point_cloud = point_cloud[l[:self.points_per_box]]
            
        X = point_cloud[:,:3]
        y = point_cloud[:,self.label_index]
        X = self.augmentations(X)
        X = torch.from_numpy(X.copy()).type(torch.float).to(device)
        y = torch.from_numpy(y.copy()).type(torch.long).to(device)
        
        #Place sample at origin
        global_shift = torch.mean(X[:,:3],axis=0)
        X = X - global_shift
        
        data = Data(pos=X, x=None, y=y)
        return data
    
    def augmentations(self,X):
        def rotate_3d(points,rotations):
            rotations[0] = math.radians(rotations[0])
            rotations[1] = math.radians(rotations[1])
            rotations[2] = math.radians(rotations[2])
            
            roll_mat = np.array([[1,0,0],
                                 [0,math.cos(rotations[0]), -math.sin(rotations[0])],
                                 [0,math.sin(rotations[0]),math.cos(rotations[0])]])
            
            pitch_mat = np.array([[math.cos(rotations[1]),0,math.sin(rotations[1])],
                                  [0,1,0],
                                  [-math.sin(rotations[1]),0,math.cos(rotations[1])]])
            
            yaw_mat = np.array([[math.cos(rotations[2]),-math.sin(rotations[2]),0],
                                [math.sin(rotations[2]),math.cos(rotations[2]),0],
                                [0,0,1]])
        
            points[:,:3] = np.matmul(np.matmul(np.matmul(points[:,:3],roll_mat),pitch_mat),yaw_mat)
            return points
        
        def random_scale_change(points,min_multiplier,max_multiplier):
            points = points*np.random.uniform(min_multiplier,max_multiplier)
            # points[:,:3] = points[:,:3] + np.random.uniform(-5,5,size=(1,3))
            return points
        
        rotations = [np.random.uniform(-10,10),np.random.uniform(-10,10),np.random.randint(0,3)*90]
        X = rotate_3d(X,rotations)
        X = random_scale_change(X,0.8,1.2)
        return X

    def random_point_removal(self,points):
            idx = list(range(np.shape(points)[0]))
            random.shuffle(idx)
            random.shuffle(idx)
            return points[:int(np.random.uniform(0.5,1)*np.shape(points)[0]),:]
   
class TestingDataset(Dataset):
    def __init__(self, testing_set_name, root_dir, points_per_box):
        self.filenames = glob.glob(root_dir+'data/'+testing_set_name+'*.csv')
        print(root_dir+'data/'+testing_set_name+'*.csv')
        self.points_per_box = points_per_box
        self.label_index = 6
    def __len__(self):
        return len(self.filenames)
        
    def __getitem__(self,index):
        point_cloud = np.array(pd.read_csv(self.filenames[index],header=None,index_col=None,delim_whitespace=True))
        
        if point_cloud.shape[0] > self.points_per_box:
            l = list(range(point_cloud.shape[0]))
            random.shuffle(l)
            point_cloud = point_cloud[l[:self.points_per_box]]
            
        X = point_cloud[:,:3]
        y = point_cloud[:,self.label_index]
        X = torch.from_numpy(X.copy()).type(torch.float).to(device)
        y = torch.from_numpy(y.copy()).type(torch.long).to(device)
        
        #Place sample at origin
        global_shift = torch.mean(X[:,:3],axis=0)
        X = X - global_shift
        
        data = Data(pos=X, x=None, y=y)
        return data

    def random_point_removal(self,points):
            idx = list(range(np.shape(points)[0]))
            random.shuffle(idx)
            random.shuffle(idx)
            return points[:int(np.random.uniform(0.5,1)*np.shape(points)[0]),:]


class SAModule(torch.nn.Module):
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

class GlobalSAModule(torch.nn.Module):
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

class FPModule(torch.nn.Module):
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

# class Net(torch.nn.Module):
#     def __init__(self, num_classes):
#         super(Net, self).__init__()
#         self.sa1_module = SAModule(0.2, 0.2, MLP([3 + 3, 64, 64, 128]))
#         self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
#         self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

#         self.fp3_module = FPModule(1, MLP([1024 + 256, 256, 256]))
#         self.fp2_module = FPModule(3, MLP([256 + 128, 256, 128]))
#         self.fp1_module = FPModule(3, MLP([128 + 3, 128, 128, 128]))

#         self.lin1 = torch.nn.Linear(128, 128)
#         self.lin2 = torch.nn.Linear(128, 128)
#         self.lin3 = torch.nn.Linear(128, num_classes)

#     def forward(self, data):
#         sa0_out = (data.x, data.pos, data.batch)
#         sa1_out = self.sa1_module(*sa0_out)
#         sa2_out = self.sa2_module(*sa1_out)
#         sa3_out = self.sa3_module(*sa2_out)

#         fp3_out = self.fp3_module(*sa3_out, *sa2_out)
#         fp2_out = self.fp2_module(*fp3_out, *sa1_out)
#         x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

#         x = F.relu(self.lin1(x))
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.lin2(x)
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.lin3(x)
#         return F.log_softmax(x, dim=-1)

class Net(torch.nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        #idea why this isn't working... too many points?
        self.sa1_module = SAModule(0.1, 0.2, MLP([3, 128, 256, 512]))
        self.sa2_module = SAModule(0.05, 0.4, MLP([512 + 3, 512, 1024, 1024]))
        self.sa3_module = GlobalSAModule(MLP([1024 + 3, 1024, 2048, 2048]))

        self.fp3_module = FPModule(1, MLP([3072, 1024, 1024]))
        self.fp2_module = FPModule(3, MLP([1536, 1024, 1024]))
        self.fp1_module = FPModule(3, MLP([1024, 1024, 1024]))

        # self.lin1 = torch.nn.Linear(512,256)
        # self.lin2 = torch.nn.Linear(256, 128)
        # self.lin3 = torch.nn.Linear(128, num_classes)
        self.conv1 = torch.nn.Conv1d(1024, 1024, 1)
        self.conv2 = torch.nn.Conv1d(1024, num_classes, 1)
        self.drop1 = torch.nn.Dropout(0.5)
        self.bn1 = torch.nn.BatchNorm1d(1024)
        
    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)

        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)
        # fp1_out = self.fp1_module(*fp2_out, *sa0_out)
        # x = F.relu(self.lin1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.lin2(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.lin3(x)
        # return F.log_softmax(x, dim=1)
        x = x.unsqueeze(dim=0)
        x = x.permute(0,2,1)
        x = self.drop1(F.relu(self.bn1(self.conv1(x))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        return x

    # def __init__(self, num_classes):
    #     super(PointNet2SemSeg, self).__init__()
    #     self.sa1 = PointNetSetAbstraction(1024, 0.2, 128, 3, [128, 256, 512], False)
    #     self.sa2 = PointNetSetAbstraction(512, 0.4, 128, 512 + 3, [512, 1024, 2048], False)
    #     self.sa3 = PointNetSetAbstraction(None, None, None, 2048 + 3, [2048, 2048, 4096], True)
    #     self.fp3 = PointNetFeaturePropagation(6144, [2048, 2048])
    #     self.fp2 = PointNetFeaturePropagation(2560, [2048, 2048])
    #     self.fp1 = PointNetFeaturePropagation(2048, [2048, 2048])
    #     self.conv1 = nn.Conv1d(2048, 2048, 1)
    #     self.bn1 = nn.BatchNorm1d(2048)
    #     self.drop1 = nn.Dropout(0.5)
    #     self.conv2 = nn.Conv1d(2048, num_classes, 1)

    # def forward(self, xyz):
    #     l1_xyz, l1_points = self.sa1(xyz, None)
    #     l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
    #     l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

    #     l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
    #     l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
    #     l0_points = self.fp1(xyz, l1_xyz, None, l1_points)

    #     x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
    #     x = self.conv2(x)
    #     x = F.log_softmax(x, dim=1)
    #     return x

if __name__ == '__main__':
    parameters = {'directory':'G:/pointnet3-master/', #C:\Users\seank\Documents\GitHub\Forest-Structural-Complexity-Tool\data
              'fileset':'train_manual_paper/train',
              'fileset_test':'test_manual_paper/train',
              'is_labeled':True,
              'GPS_coords':False,
              # 'model_filename':'pointnet2modded5.pth',nonoriginalpointnet2
              'model_filename':'manual_modified_pointnet_paper_geo2.pth',
              'load_existing_model':1,
              'num_epochs':2000,
              'learning_rate':0.0001,
              'batch_size':10, #could be interesting to see if batchsize of 1 is better
              'box_dimensions':[6,6,6],
              'box_overlap':[0.5,0.5,0.5],
              'min_points_per_box':5000,
              'max_points_per_box':15000,
              'subsample':False,
              'subsampling_min_spacing':0.025,
              'num_threads':20, #for preprocessing only.
              }
    
    if 0:
        preprocessing = Preprocessing(parameters)
        training_point_cloud_list = glob.glob(parameters['directory']+'train_all_combined.csv') #
        for point_cloud in training_point_cloud_list:
              print(point_cloud)
              preprocessing.load_point_cloud(filename=point_cloud)
              preprocessing.preprocess_point_cloud()
    
    train_dataset = TrainingDataset(training_set_name = parameters['fileset'],
                                        root_dir = parameters['directory'],
                                        points_per_box=parameters['max_points_per_box'])
    
    train_loader = DataLoader(train_dataset, batch_size=parameters['batch_size'], shuffle=True,
                                        num_workers=0,pin_memory=True)
    
    test_dataset = TestingDataset(testing_set_name = parameters['fileset_test'],
                                        root_dir = parameters['directory'],
                                        points_per_box=parameters['max_points_per_box'])
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True,
                                        num_workers=0,pin_memory=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(num_classes=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    
    if parameters['load_existing_model']:
        print('Loading existing model...')
        model.load_state_dict(torch.load(parameters['directory']+parameters['model_filename']),strict=False)
    optimiser = optim.Adam(model.parameters(), lr = parameters['learning_rate'])
    # optimiser = optim.SGD(net.parameters(), lr = parameters['learning_rate'])
    scheduler = ExponentialLR(optimiser, gamma=0.9)
    
    if parameters['load_existing_model']:
        try:
            history_array = np.array(pd.read_csv(parameters['directory']+'history_array' +parameters['model_filename'][:-4]+'.csv',header=None,index_col=None,delim_whitespace=True))
            print("Loaded past history array...")
            past_i = history_array[-1,0]
                
        except:
            history_array = np.zeros((0,5))
            pd.DataFrame(history_array).to_csv(parameters['directory']+'history_array' +parameters['model_filename'][:-4]+'.csv',header=None, index=None,sep=' ')
            print("Created new history array...")
            try:
                past_i = history_array[-1,0]
            except:
                past_i = 0
    else:
        history_array = np.zeros((0,5))
        pd.DataFrame(history_array).to_csv(parameters['directory']+'history_array' +parameters['model_filename'][:-4]+'.csv',header=None, index=None,sep=' ')
    
        print("Created new history array...")
        past_i = 0
    
    num_epochs_seen = 0
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
            out = out.permute(2,1,0).squeeze()
            criterion = torch.nn.CrossEntropyLoss(torch.Tensor([0.9,0.9,0.9,0.9,1]).to(device))
            loss = criterion(out, data.y)
            # loss = F.nll_loss(out, data.y)
            loss.backward()
            
            optimizer.step()
            total_loss += loss.item()
            out = np.argmax(out.cpu().detach(),axis=1)
            # print((np.array(out)==np.array(data.y.cpu())).sum())
            correct_nodes += (np.array(out)==np.array(data.y.cpu())).sum()
            total_nodes += data.num_nodes
            train_acc = correct_nodes / total_nodes
            
            if (i + 1) % 10 == 0:
                print(i+1)
                # pd.DataFrame(np.hstack((data.pos.cpu(),np.atleast_2d(out).T,np.atleast_2d(data.y.cpu()).T))).to_csv(parameters['directory']+'most_recent__train_out.csv',header=None,index=None,sep=' ')

            
            if (i + 1) % 20 == 0:
                history_array = np.vstack((history_array,np.array([[samples_seen+i,#0
                                                                    total_loss / 20,
                                                                    train_acc,
                                                                    -1,
                                                                    -1]])))
                print(f'[{i+1}/{len(train_loader)}] Loss: {total_loss / 20:.4f} 'f'Train Acc: {train_acc:.4f}')
                total_loss = correct_nodes = total_nodes = 0
                print("Saving model...")
                torch.save(model.state_dict(),parameters['directory']+parameters['model_filename'])
                pd.DataFrame(history_array).to_csv(parameters['directory']+'history_array' +parameters['model_filename'][:-4]+'.csv',header=None, index=None,sep=' ')
                print("Saved model...")
            
            if (i+1) % 50 == 0:
                model.eval()
                
                for j, data in enumerate(test_loader):
                    data = data.to(device)
                    out = model(data)
                    out = out.permute(2,1,0).squeeze()
                    testloss = criterion(out, data.y)
                    # testloss = F.nll_loss(out, data.y)
                    out = np.argmax(out.cpu().detach(),axis=1)
                    total_testloss += testloss.item()
                    correct_testnodes += (np.array(out)==np.array(data.y.cpu())).sum()
    
                    total_testnodes += data.num_nodes
                    test_acc = correct_testnodes / total_testnodes
                    
                    if j > 19:
                        # print(np.shape(data.pos.cpu()),np.shape(out))
                        
                        pd.DataFrame(np.hstack((data.pos.cpu(),np.atleast_2d(out).T,np.atleast_2d(data.y.cpu()).T))).to_csv(parameters['directory']+'most_recent_out.csv',header=None,index=None,sep=' ')
                        break
                    
                print(f'[{i+1}/{len(train_loader)}] Test Loss: {total_testloss / 20:.4f} 'f'Test Acc: {test_acc:.4f}')
    
                history_array = np.vstack((history_array,np.array([[samples_seen+i,#0
                                                                -1,
                                                                -1,
                                                                total_testloss/20,
                                                                test_acc]])))
                total_testloss = correct_testnodes = total_testnodes = 0
                pd.DataFrame(history_array).to_csv(parameters['directory']+'history_array' +parameters['model_filename']+'.csv',header=None, index=None,sep=' ')
                model.train()
                samples_seen = i + samples_seen
        num_epochs_seen += 1
        torch.save(model.state_dict(),parameters['directory']+parameters['model_filename'][:-4]+'_'+str(num_epochs_seen)+'.pth')
        pd.DataFrame(history_array).to_csv(parameters['directory']+'history_array' +parameters['model_filename'][:-4]+'.csv',header=None, index=None,sep=' ')
        scheduler.step()
