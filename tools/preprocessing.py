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

class Preprocessing:
    def __init__(self,parameters):
        self.preprocessing_time_start = time.time()
        self.parameters = parameters
        self.directory = parameters['directory']
        self.box_dimensions = np.array(self.parameters['box_dimensions'])
        self.box_overlap = np.array(self.parameters['box_overlap'])
        self.fileset = parameters['fileset'] 
        self.min_points_per_box = parameters['min_points_per_box']
        self.max_points_per_box = parameters['max_points_per_box']
        self.subsample = parameters['subsample']
        self.subsampling_min_spacing = parameters['subsampling_min_spacing']
        self.num_procs = parameters['num_procs']
        
    def load_point_cloud(self,filename):
        print("Loading point cloud from file...")
        # start = time.time()
        self.point_cloud = np.array(pd.read_csv(filename,header=None,delim_whitespace=True))
        print("Original point cloud shape:",self.point_cloud.shape)
        
        # if self.subsample:            
        #     self.point_cloud = self.subsample_point_cloud(self.point_cloud,min_spacing=self.subsampling_min_spacing)
        #     print("Subsampled point cloud shape:",self.point_cloud.shape)
        #     print("Loading point cloud took ",'{0:.2f}'.format(time.time() - start),'s')
        #     print("Point cloud shape is",self.point_cloud.shape)
    
        self.global_shift = [np.mean(self.point_cloud[:,0]),np.mean(self.point_cloud[:,1]),np.mean(self.point_cloud[:,2])]
        self.point_cloud[:,:3] = self.point_cloud[:,:3] - self.global_shift
        print(self.directory+"data/working_directory/"+self.parameters['input_point_cloud'][:-4]+'/global_shift.csv')
        np.savetxt(self.directory+"data/working_directory/"+self.parameters['input_point_cloud'][:-4]+'/global_shift.csv',self.global_shift)
    
    # def subsample_point_cloud(self,X,min_spacing):
    #         print("Subsampling...")
    #         neighbours = NearestNeighbors(n_neighbors=2, algorithm='kd_tree',metric='euclidean').fit(X[:,:3])
    #         distances, indices = neighbours.kneighbors(X[:,:3])
    #         X_keep = X[distances[:,1]>=min_spacing]
    #         i1 = [distances[:,1]<min_spacing][0]
    #         i2 = [X[indices[:,0],2]<X[indices[:,1],2]][0]
    #         X_check = X[np.logical_and(i1,i2)]
            
    #         while np.shape(X_check)[0] > 1:
    #             neighbours = NearestNeighbors(n_neighbors=2, algorithm='kd_tree',metric='euclidean').fit(X_check[:,:3])
    #             distances, indices = neighbours.kneighbors(X_check[:,:3])
    #             X_keep = np.vstack((X_keep,X_check[distances[:,1]>=min_spacing,:]))
    #             i1 = [distances[:,1]<min_spacing][0]
    #             i2 = [X_check[indices[:,0],2]<X_check[indices[:,1],2]][0]
    #             X_check = X_check[np.logical_and(i1,i2)]
    #         X = X_keep
    #         return X
    
    def preprocess_point_cloud(self):
        print("Pre-processing point cloud...")
        def make_boxes():
            print("Making boxes...")
            point_cloud = self.point_cloud#[self.point_cloud[:,4]!=5]
            self.box_size = self.box_dimensions
            Xmax = np.max(point_cloud[:,0])
            Xmin = np.min(point_cloud[:,0])
            Ymax = np.max(point_cloud[:,1])
            Ymin = np.min(point_cloud[:,1])
            Zmax = np.max(point_cloud[:,2])
            Zmin = np.min(point_cloud[:,2])
            
            X_range = Xmax - Xmin
            Y_range = Ymax - Ymin
            Z_range = Zmax - Zmin
            
            num_boxes_x = int(np.ceil(X_range/self.box_size[0]))
            num_boxes_y = int(np.ceil(Y_range/self.box_size[1]))
            num_boxes_z = int(np.ceil(Z_range/self.box_size[2]))
            print(self.box_overlap)
            x_vals = np.linspace(Xmin,  Xmin+(num_boxes_x*self.box_size[0]),  int(num_boxes_x/(1-self.box_overlap[0]))+1)
            y_vals = np.linspace(Ymin,  Ymin+(num_boxes_y*self.box_size[1]),  int(num_boxes_y/(1-self.box_overlap[1]))+1)
            z_vals = np.linspace(Zmin,  Zmin+(num_boxes_z*self.box_size[2]),  int(num_boxes_z/(1-self.box_overlap[2]))+1)
            
            box_centres = np.vstack(np.meshgrid(x_vals,y_vals,z_vals)).reshape(3,-1).T
            
            #This checks if there are voxelbox files already in the data directory.
            #If there are, it gets the largest file ID number and adds one for the new start point.
            files = glob.glob(self.parameters['directory']+"data/working_directory/"+self.parameters['input_point_cloud'][:-4]+self.fileset+'*.csv')
            max_file_id = 0
            for file in files:
                file_id = int(file[-9:-4])
                if file_id>max_file_id:
                    max_file_id = file_id
            if max_file_id != 0:
                max_file_id += 1
            
            # indices = list(range(l))
            def threaded_boxes(point_cloud,box_size,min_points_per_box,max_points_per_box,path, max_file_id, id_offset, point_divisions):

                box_centre_mins = point_divisions-0.5*box_size
                box_centre_maxes = point_divisions+0.5*box_size
                i = 0
                pds = len(point_divisions)
                while i < pds:
                    box = point_cloud
                    box = box[np.logical_and(np.logical_and(np.logical_and(box[:,0]>=box_centre_mins[i,0],
                                                                           box[:,0]<box_centre_maxes[i,0]),
                                                            np.logical_and(box[:,1]>=box_centre_mins[i,1],
                                                                           box[:,1]<box_centre_maxes[i,1])),
                                             np.logical_and(box[:,2]>=box_centre_mins[i,2],
                                                            box[:,2]<box_centre_maxes[i,2]))]
                    
                    if box.shape[0]>min_points_per_box:
                        if box.shape[0]>max_points_per_box:
                            indices = list(range(0,box.shape[0]))
                            random.shuffle(indices)
                            random.shuffle(indices)
                            box = box[indices[:max_points_per_box],:]
                            box = np.asarray(box,dtype='float64')
                        
                        box[:,:3] = box[:,:3]
                        np.save(path+str(max_file_id+id_offset+i).zfill(7)+'.npy',box)
                    i += 1
                return 1
            path = self.parameters['directory']+"data/working_directory/"+self.parameters['input_point_cloud'][:-4]+'/'+self.fileset
            num_procs = self.num_procs
            point_divisions = []
            
            for thread in range(num_procs):
                point_divisions.append([])
            
            points_to_assign = box_centres
            
            while points_to_assign.shape[0] > 0:
                for i in range(num_procs):
                    point_divisions[i].append(points_to_assign[0,:])
                    points_to_assign = points_to_assign[1:]
                    if points_to_assign.shape[0] == 0:
                        break
            threads = []
            prev_id_offset = 0
            for thread in range(num_procs):
                id_offset = 0
                for t in range(thread):
                    id_offset = id_offset + len(point_divisions[t])
                print('Thread:',thread,prev_id_offset, id_offset,path)
                prev_id_offset = id_offset
                t = threading.Thread(target=threaded_boxes, args=(self.point_cloud,
                                                              self.box_size,
                                                              self.min_points_per_box,
                                                              self.max_points_per_box,
                                                              # self.global_shift,
                                                              path, 
                                                              max_file_id, 
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
            np.savetxt(self.parameters['directory']+"data/postprocessed_point_clouds/"+self.parameters['input_point_cloud'][:-4]+"_out/preprocessing_time.csv",np.array([self.preprocessing_time_total]))
            print("Preprocessing done")
        make_boxes()
