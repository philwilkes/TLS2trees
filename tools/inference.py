import torch
from torch_geometric.data import Dataset, DataLoader, Data
import numpy as np
import glob
import pandas as pd
from preprocessing import Preprocessing
from train import Net
from sklearn.neighbors import NearestNeighbors
from scipy import spatial
import os
import time

class TestingDataset(Dataset):
    def __init__(self, dataset_name, root_dir, points_per_box, device):
        self.filenames = glob.glob(root_dir+dataset_name+'*.npy')
        self.device = device
        self.points_per_box = points_per_box
    def __len__(self):
        return len(self.filenames)
        
    def __getitem__(self,index):
        point_cloud = np.load(self.filenames[index])
        pos = point_cloud[:,:3]
        extra_info = point_cloud[:,3:6]
        pos = torch.from_numpy(pos.copy()).type(torch.float).to(self.device).requires_grad_(False)
        
        #Place sample at origin
        local_shift = torch.round(torch.mean(pos[:,:3],axis=0)).requires_grad_(False)
        pos = pos - local_shift
        extra_info = torch.from_numpy(extra_info.copy()).type(torch.float).to(self.device).requires_grad_(False)
        data = Data(pos=pos, x=None,extra_info=extra_info,local_shift=local_shift)
        return data

class semantic_segmentation():
    def __init__(self, parameters):
        self.parameters = parameters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.directory = self.parameters['directory']
        self.point_cloud_filename = self.parameters['input_point_cloud'][:-4]+'_out'
        self.make_folder_structure()
    def make_folder_structure(self):
        if not os.path.isdir(self.directory+"data/original_point_clouds"):
            os.makedirs(self.directory+"data/original_point_clouds")
            print('Created "data/original_point_clouds" directory')
            
        if not os.path.isdir(self.directory+"data/working_directory/"+self.parameters['input_point_cloud'][:-4]):
            os.makedirs(self.directory+"data/working_directory/"+self.parameters['input_point_cloud'][:-4])
            print('Created "data/working_directory/'+self.parameters['input_point_cloud'][:-4])
            
        if not os.path.isdir(self.directory+"data/segmented_point_clouds"):
            os.makedirs(self.directory+"data/segmented_point_clouds")
            print('Created "data/segmented_point_clouds" directory')
            
        if not os.path.isdir(self.directory+"data/postprocessed_point_clouds"):
            os.makedirs(self.directory+"data/postprocessed_point_clouds")
            print('Created "data/postprocessed_point_clouds" directory')
        
        if not os.path.isdir(self.directory+"data/postprocessed_point_clouds/"+self.point_cloud_filename):
            os.makedirs(self.directory+"data/postprocessed_point_clouds/"+self.point_cloud_filename)
            print('Created "data/postprocessed_point_clouds" directory')
        self.output_dir = self.directory+"data/postprocessed_point_clouds/"+self.point_cloud_filename+'/'
    
    
    def run_preprocessing(self):
    
        preprocessing = Preprocessing(self.parameters)
        point_cloud = self.parameters['directory']+"data/original_point_clouds/"+self.parameters['input_point_cloud']
        print(point_cloud)
        preprocessing.load_point_cloud(filename=point_cloud)
        preprocessing.preprocess_point_cloud()
        return
    
    def assign_labels_to_original_point_cloud(self,original,labeled,label_index):
        print("Assigning segmentation labels to original point cloud...")
        kdtree = spatial.cKDTree(labeled[:,:3])
        labels = np.atleast_2d(labeled[kdtree.query(original[:,:3], k=2)[1][:,1],label_index]).T
        original = np.hstack((original,labels))
        return original
    
    def subsample_point_cloud(self,X,min_spacing):
        print("Subsampling...")
        neighbours = NearestNeighbors(n_neighbors=2, algorithm='kd_tree',metric='euclidean').fit(X[:,:3])
        distances, indices = neighbours.kneighbors(X[:,:3])
        X_keep = X[distances[:,1]>=min_spacing]
        i1 = [distances[:,1]<min_spacing][0]
        i2 = [X[indices[:,0],2]<X[indices[:,1],2]][0]
        X_check = X[np.logical_and(i1,i2)]
        
        while np.shape(X_check)[0] > 1:
            neighbours = NearestNeighbors(n_neighbors=2, algorithm='kd_tree',metric='euclidean').fit(X_check[:,:3])
            distances, indices = neighbours.kneighbors(X_check[:,:3])
            X_keep = np.vstack((X_keep,X_check[distances[:,1]>=min_spacing,:]))
            i1 = [distances[:,1]<min_spacing][0]
            i2 = [X_check[indices[:,0],2]<X_check[indices[:,1],2]][0]
            X_check = X_check[np.logical_and(i1,i2)]
        # X = np.delete(X,np.unique(indices[distances[:,1]<min_spacing]),axis=0)
        X = X_keep
        return X
    
    def choose_most_confident_label(self, point_cloud, original_point_cloud):
        #TODO work on checking this. There may be considerable improvements possible.
        print("Choosing most confident labels...")
        neighbours = NearestNeighbors(n_neighbors=16, algorithm='kd_tree',metric='euclidean',radius=0.05).fit(point_cloud[:,:3])
        _, indices = neighbours.kneighbors(original_point_cloud[:,:3])
        labels = np.zeros((original_point_cloud.shape[0],5))
        # labels[:,0] = np.argmax(np.max(point_cloud[indices][:,:,-4:],axis=1),axis=1)
        labels[:,4] = np.argmax(np.median(point_cloud[indices][:,:,-4:],axis=1),axis=1)
        # labels[:,:4] = np.median(point_cloud[indices][:,:,-4:],axis=1)
        # labels[:,1] = np.argmax(np.max(point_cloud[indices][:,:,-4:],axis=1),axis=1)
        original_point_cloud = np.hstack((original_point_cloud,np.atleast_2d(labels[:,-1]).T))
        
        # neighbours = NearestNeighbors(n_neighbors=20, algorithm='kd_tree',metric='euclidean').fit(point_cloud[:,:3])
        # _, indices = neighbours.kneighbors(point_cloud[:,:3])
        # labels = np.zeros((point_cloud.shape[0],2))
        # labels[:,0] = np.argmax(np.max(point_cloud[indices][:,:,-4:],axis=1),axis=1)
        # labels[:,0] = np.argmax(np.median(point_cloud[indices][:,:,-6:-2],axis=1),axis=1)
        # labels[:,1] = np.argmax(np.max(point_cloud[indices][:,:,-6:-2],axis=1),axis=1)
        # point_cloud = np.hstack((point_cloud,labels))
        
        
        
        # _,indices = np.unique(point_cloud[:,:3],axis=0,return_index=True)
        # return point_cloud[indices]
        return original_point_cloud
    
    def inference(self):
        self.sem_seg_start_time = time.time()
        test_dataset = TestingDataset(dataset_name = self.parameters['fileset'],
                                    root_dir = self.parameters['directory']+"data/working_directory/"+self.parameters['input_point_cloud'][:-4]+'/',
                                    points_per_box=self.parameters['max_points_per_box'],
                                    device= self.device)

        test_loader = DataLoader(test_dataset, batch_size=self.parameters['batch_size'], shuffle=False,
                                            num_workers=0,pin_memory=True)
        
        global_shift = np.loadtxt(self.parameters['directory']+"data/working_directory/"+self.parameters['input_point_cloud'][:-4]+'/'+'global_shift.csv',dtype='float64')
        print("Global shift:",global_shift)
        
        # model = Net(num_classes=5).to(self.device)
        model = Net(num_classes=4).to(self.device)
        model.load_state_dict(torch.load(self.parameters['directory']+'model/'+self.parameters['model_filename']),strict=False)
        model.eval()
        num_boxes = test_dataset.__len__()
        with torch.no_grad():
            for i,data in enumerate(test_loader):
                outshape = np.shape(data.extra_info.cpu().detach())[1]
                break
            self.output_point_cloud = np.zeros((0,3+outshape+4))
            # output_point_cloud = np.zeros((0,3+outshape+5))
            
            for i,data in enumerate(test_loader):#, 0): testing without this
                print(i*self.parameters['batch_size'],'/',num_boxes)
                data = data.to(self.device)
                out = model(data)
                out = out.permute(2,1,0).squeeze()
                batches = np.unique(data.batch.cpu())
                out = torch.softmax(out.cpu().detach(),axis=1)
                pos = data.pos.cpu()
                extra_info_out = data.extra_info.cpu()
                output = np.hstack((pos,extra_info_out,out))
                for batch in batches:
                    outputb = np.asarray(output[data.batch.cpu()==batch])
                    outputb[:,:3] = outputb[:,:3] + np.asarray(data.local_shift.cpu())[3*batch:3+(3*batch)]
                    self.output_point_cloud = np.vstack((self.output_point_cloud,outputb))
        
        del outputb, out, batches, pos, extra_info_out, output #clean up anything no longer needed to free RAM.
        
        print("Loading original point cloud...")
        original_point_cloud = np.array(pd.read_csv(self.directory+"data/original_point_clouds/"+self.parameters['input_point_cloud'][:-4]+'.csv',header=None,index_col=None,delim_whitespace=True))
        original_point_cloud = original_point_cloud[:,:3]
        original_point_cloud[:,:3] = original_point_cloud[:,:3] - global_shift
        if self.parameters['subsample']==True:
            original_point_cloud = self.subsample_point_cloud(original_point_cloud,self.parameters['subsampling_min_spacing'])
        self.output = np.asarray(self.choose_most_confident_label(self.output_point_cloud,original_point_cloud),dtype='float64')
        self.output[:,:3] = self.output[:,:3] + global_shift
        print("Saving...")
        pd.DataFrame(self.output).to_csv(self.parameters['directory']+"data/segmented_point_clouds/"+self.parameters['input_point_cloud'][:-4]+'_out'+'.csv',header=None,index=None,sep=' ')
        print("Saved")
        self.sem_seg_end_time = time.time()
        self.sem_seg_total_time = self.sem_seg_end_time - self.sem_seg_start_time
        np.savetxt(self.parameters['directory']+"data/postprocessed_point_clouds/"+self.parameters['input_point_cloud'][:-4]+'_out/'+"semantic_segmentation_time.csv",np.array([self.sem_seg_total_time]))
        print("Semantic segmentation took", self.sem_seg_total_time, 's')
        print("Semantic segmentation done")

# parameters = {'directory':'../',
#             # 'fileset':'test_aug',
#             'fileset':'test',
#             # 'input_point_cloud':'Denham_P290_TLS_cropped_1cmDS.csv',
#             'input_point_cloud':'20190917_Tumba001_1-14_merged_1cm_SA_.csv',
#            # 'input_point_cloud':'Samford_merged_2.csv',
#             # 'input_point_cloud':'HovermapDogParkBackpack.csv',
#             # 'input_point_cloud':'tls_benchmark_P5.csv',
#             # 'input_point_cloud':'tls_benchmark_P6.csv',
#             # 'input_point_cloud':'tls_b_test1.csv',
#             # 'input_point_cloud':'B3t1a.csv',
#             # 'input_point_cloud':'final_test_aug.csv',
#             # 'input_point_cloud':'final_test.csv',
            
#           # 'model_filename':'../model/model5_no_noise.pth',
#           'model_filename':'../model/model6_no_noise.pth',
#           'batch_size':20,
#           'box_dimensions':[6,6,6],
#             'box_overlap':[0.5,0.5,0.5],
#             # 'box_overlap':[0.,0.,0.],
#             'min_points_per_box':1000,
#           'max_points_per_box':20000,
#           'subsample':True,
#           'subsampling_min_spacing':0.01,
#           'num_procs':20,
#           }

# sem_seg = semantic_segmentation(parameters)
# # sem_seg.run_preprocessing()
# sem_seg.inference()



