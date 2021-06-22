import numpy as np
from sklearn import linear_model
from skimage.measure import CircleModel, ransac, LineModelND
import math
import pandas as pd
from scipy import spatial, interpolate
from scipy.interpolate import griddata
from sklearn.cluster import DBSCAN
from multiprocessing import Pool, get_context
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import sys

class MeasureTree:
    def __init__(self,parameters):
        self.parameters = parameters
        self.input_point_cloud = parameters['input_point_cloud'][:-4]+'_out'
        self.directory = parameters['directory']
        self.num_procs = parameters['num_procs']
        self.num_neighbours = parameters['num_neighbours']
        self.slice_thickness = parameters['slice_thickness']
        self.slice_increment = parameters['slice_increment']
        self.min_tree_volume = parameters['min_tree_volume']
        self.diameter_measurement_increment = parameters['diameter_measurement_increment']
        self.diameter_measurement_height_range = parameters['diameter_measurement_height_range']
        self.point_cloud = np.array(pd.read_csv(self.directory+'data/postprocessed_point_clouds/'+self.input_point_cloud+'/cleaned_PC.csv',header=None,index_col=None,delim_whitespace=True))
        self.stem_points = self.point_cloud[self.point_cloud[:,-2]==4]
        self.vegetation_points = self.point_cloud[self.point_cloud[:,-2]==2]
        self.stem_points = self.noise_filtering(self.stem_points,min_neighbour_dist=0.03,min_neighbours=3)
        self.stem_points = self.subsample_point_cloud(X=self.stem_points,min_spacing=0.01)
        self.characters = ['0','1','2','3','4','5','6','7','8','9','dot','m','space','_','-','semiC','A','B','C','D','E','F','G','H','I','J','K','L','_M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
        self.character_viz = []
        for i in self.characters:
            self.character_viz.append(np.genfromtxt(self.directory+'/tools/numbers/' +i+'.csv',delimiter=','))
          
    # @staticmethod
    @classmethod
    def get_2_opposing_neighbours(cls,point,other_points,max_search_distance=0.5, search_cone_angle = 60, max_radius_ratio=1.5,min_radius_ratio=0.5):
        closest_point_pos = np.zeros((0,point.shape[0]))
        closest_point_neg = np.zeros((0,point.shape[0]))
        kdtree = spatial.cKDTree(other_points[:,:3],leafsize=1000)
        results = kdtree.query_ball_point(point[:3],r=max_search_distance)
        cyl_vector = point[3:6]
        current_radius = point[6]
        if current_radius == 0:
            return closest_point_pos,closest_point_neg
        """
        Need to find closest point in similar direction and similar radius.
        """
        neighbours = other_points[results]
        neighbours = neighbours[np.logical_and(neighbours[:,6]/current_radius>=min_radius_ratio,neighbours[:,6]/current_radius<=max_radius_ratio)]
        neighbours = np.hstack((neighbours,np.atleast_2d(np.linalg.norm(neighbours[:,:3] - point[:3],axis=1)).T))#add distances column from all neighbours to point
        neighbours = neighbours[neighbours[:,-1]>0]
        
        if neighbours.shape[0]>0:
            dp = np.dot((neighbours[:,:3]-point[:3])/np.atleast_2d(np.linalg.norm(neighbours[:,:3]-point[:3],axis=1)).T,cyl_vector)
            dp[dp>1]=1 #Correct floating point errors
            dp[dp<-1]=-1 #Correct floating point errors
            dp = np.arccos(dp)
            pos_neighbours = neighbours[np.logical_and(math.radians(search_cone_angle)>dp,dp>0)]
            neg_neighbours = neighbours[np.logical_and(-math.radians(search_cone_angle)<dp,dp<0)]
            if pos_neighbours.shape[0]>0:
                closest_point_pos = pos_neighbours[np.argmin(pos_neighbours[:,-1],axis=0)]
                
            if neg_neighbours.shape[0]>0:
                closest_point_neg = neg_neighbours[np.argmin(neg_neighbours[:,-1],axis=0)]
        
        return closest_point_pos,closest_point_neg
    
    # @staticmethod
    @classmethod
    def interpolate_cyl(cls,cyl1,cyl2,resolution):
        #[x,y,z,xn,yn,zn,r,cci,original_cyl_line Cx,Cy,Cz, original_cyl_length]
        points_per_line = int(np.linalg.norm(np.array([cyl2[0],cyl2[1],cyl2[2]])-np.array([cyl1[0],cyl1[1],cyl1[2]]))/resolution)
        interpolated = np.linspace(cyl1,cyl2,points_per_line)
        interpolated[:,6] = np.min(interpolated[:,6])
        return interpolated
    
    # @staticmethod
    @classmethod
    def cylinder_sorting(cls,cylinder_array,angle_tolerance,search_angle,distance_tolerance,min_radius_ratio,max_radius_ratio):
        def decision_tree(cyl1,cyl2,angle_tolerance,search_angle,min_radius_ratio,max_radius_ratio):
            """
            Decides if cyl2 should be joined to cyl1 and if they are the same branch.
            angle_tolerance is the maximum angle between normal vectors of cylinders to be considered the same branch.
            """
            def within_angle_tolerances(normal1,normal2,angle_tolerance):
                """Checks if normal1 and normal2 are within "angle_tolerance"
                of each other."""
                norm2 = np.zeros(normal2.shape)
                mask = np.logical_not(np.all(normal2==0,axis=1))
                
                norm1 = normal1/np.atleast_2d(np.linalg.norm(normal1)).T
                norm2[mask] = normal2[mask]/np.atleast_2d(np.linalg.norm(normal2[mask],axis=1)).T
                dot = np.clip(np.einsum('ij, ij->i', norm1, norm2),a_min=-1,a_max=1)
                theta = np.degrees(np.arccos(dot))
                return abs((theta > 90)*180-theta) <= angle_tolerance

            def similar_radius(radius1,radius2,min_ratio,max_ratio):
                return np.logical_and(radius2/radius1 >= min_ratio,radius2/radius1 <= max_ratio)

            vector_array = cyl2[:,:3]-np.atleast_2d(cyl1[:3])
            
            condition1 = within_angle_tolerances(cyl1[3:6],cyl2[:,3:6],angle_tolerance)
            condition2 = within_angle_tolerances(cyl1[3:6],vector_array,search_angle)
            condition3 = similar_radius(cyl1[6],cyl2[:,6],min_radius_ratio,max_radius_ratio)
            # print(np.sum(np.logical_and(condition1,condition2)),np.shape(mask)[0])
            # cyl2[mask,-1][np.logical_and(condition1,condition2)] = cyl1[-1]
            cyl2[np.logical_and(condition3,np.logical_and(condition1,condition2)),-1] = cyl1[-1]
            return cyl2
        
        max_tree_label = 1
        cylinder_array = cylinder_array[cylinder_array[:,6]!=0]
        unsorted_points = np.hstack((cylinder_array,np.zeros((cylinder_array.shape[0],1))))
        # unsorted_points = np.unique(unsorted_points,axis=0)
        unsorted_points = np.vstack(([tuple(row) for row in unsorted_points]))

        sorted_points = np.zeros((0,unsorted_points.shape[1]))
        total_points = len(unsorted_points)
        while unsorted_points.shape[0]>1:
            if sorted_points.shape[0] % 200 == 0:
                print('\r',np.around(sorted_points.shape[0]/total_points,3),end='')
            
            current_point_index = np.argmin(unsorted_points[:,2])
            current_point = unsorted_points[current_point_index]
            if current_point[-1] == 0:
                current_point[-1] = max_tree_label
                max_tree_label += 1

            sorted_points = np.vstack((sorted_points,current_point))
            unsorted_points = np.vstack((unsorted_points[:current_point_index],
                                         unsorted_points[current_point_index+1:]))
            results = spatial.cKDTree(unsorted_points[:,:3],leafsize=1000).query_ball_point(np.atleast_2d(current_point)[:,:3],r=distance_tolerance)[0]
            unsorted_points[results] = decision_tree(current_point,
                                                     unsorted_points[results],
                                                     angle_tolerance,
                                                     search_angle,
                                                     min_radius_ratio,
                                                     max_radius_ratio)
        print('1.000\n')
        return sorted_points, unsorted_points
    # @staticmethod
    @classmethod
    def make_cyl_visualisation(cls,cyl):
        p = MeasureTree.create_3d_circles_as_points_flat(cyl[0],cyl[1],cyl[2],cyl[6])
        points = MeasureTree.rodrigues_rot(p-cyl[:3],[0,0,1],cyl[3:6])
        points = np.hstack((points+cyl[:3],np.zeros((points.shape[0],1))))
        points[:,3] = cyl[-2]
        return points
    
    # @staticmethod
    @classmethod
    def subsample_point_cloud(cls,X,min_spacing):
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
        X = X_keep
        return X
    
    # @staticmethod
    @classmethod
    def points_along_line(cls,x0,y0,z0,x1,y1,z1,resolution=0.05):
        points_per_line = int(np.linalg.norm(np.array([x1,y1,z1])-np.array([x0,y0,z0]))/resolution)
        Xs = np.atleast_2d(np.linspace(x0,x1,points_per_line)).T
        Ys = np.atleast_2d(np.linspace(y0,y1,points_per_line)).T
        Zs = np.atleast_2d(np.linspace(z0,z1,points_per_line)).T
        return np.hstack((Xs,Ys,Zs))
    
    # @staticmethod
    @classmethod
    def create_3d_circles_as_points_flat(cls,x,y,z,r,circle_points=15):
        angle_between_points = np.linspace(0,2*np.pi,circle_points)
        points = np.zeros((0,3))
        for i in angle_between_points:
            x2 = r*np.cos(i)+x
            y2 = r*np.sin(i)+y
            point = np.array([[x2,y2,z]])
            points = np.vstack((points,point))
        return points
    
    # @staticmethod
    @classmethod
    def rodrigues_rot(cls,P, n0, n1):
        """RODRIGUES ROTATION
        - Rotate given points based on a starting and ending vector
        - Axis k and angle of rotation theta given by vectors n0,n1
        P_rot = P*cos(theta) + (k x P)*sin(theta) + k*<k,P>*(1-cos(theta))"""
        # If P is only 1d array (coords of single point), fix it to be matrix
        if P.ndim == 1:
            P = P[np.newaxis,:]
        
        # Get vector of rotation k and angle theta
        n0 = n0/np.linalg.norm(n0)
        n1 = n1/np.linalg.norm(n1)
        k = np.cross(n0,n1)
        if np.sum(k) != 0:
            k = k/np.linalg.norm(k)
        theta = np.arccos(np.dot(n0,n1))
        
        # Compute rotated points
        P_rot = np.zeros((len(P),3))
        for i in range(len(P)):
            P_rot[i] = P[i]*np.cos(theta) + np.cross(k,P[i])*np.sin(theta) + k*np.dot(k,P[i])*(1-np.cos(theta))
        return P_rot
    
    # @staticmethod
    @classmethod
    def fit_circle_3D(cls,points,V,return_cyl_vis=False):
        # fitted_circle_points = np.zeros((0,3))
        CCI = 0
        P = points[:,:3]
        P_mean = np.mean(P,axis=0)
        P_centered = P - P_mean
        normal = V/np.linalg.norm(V)
        # Project points to coords X-Y in 2D plane
        P_xy = MeasureTree.rodrigues_rot(P_centered, normal, [0,0,1])
        # Fit circle in new 2D coords with RANSAC
        if P_xy.shape[0]>=35:
            
            model_robust, inliers = ransac(P_xy[:,:2], CircleModel, min_samples=int(P_xy.shape[0]*0.1),
                                           residual_threshold=0.025, max_trials=2000)
            xc,yc = model_robust.params[0:2]
            r = model_robust.params[2]
            CCI = MeasureTree.circumferential_completeness_index([xc,yc],r,P_xy[:,:2])
        
        # elif P_xy.shape[0]>=10:
        #     model_robust, inliers = ransac(P_xy[:,:2], CircleModel, min_samples=7,
        #                                    residual_threshold=0.1, max_trials=500)
        #     xc,yc = model_robust.params[0:2]
        #     r = model_robust.params[2]
        #     CCI = MeasureTree.circumferential_completeness_index([xc,yc],r,P_xy[:,:2])

        if CCI < 0.2:
            r = np.std(P_xy[:,:2].flatten())*0.5
            xc,yc = np.mean(P_xy[:,:2],axis=0)
            # print(xc,yc,r)
            CCI = 0
        # Transform circle center back to 3D coords
        cyl_centre = MeasureTree.rodrigues_rot(np.array([[xc,yc,0]]), [0,0,1], normal) + P_mean
        if return_cyl_vis:
            fitted_circle_points = MeasureTree.create_3d_circles_as_points_flat(xc,yc,0,r)
            fitted_circle_points = MeasureTree.rodrigues_rot(fitted_circle_points, [0,0,1], normal) + P_mean
            return np.array([[cyl_centre[0,0],cyl_centre[0,1],cyl_centre[0,2],normal[0],normal[1],normal[2],r,CCI]]),np.hstack((fitted_circle_points,np.zeros((fitted_circle_points.shape[0],1))+CCI))
        else:
            return np.array([[cyl_centre[0,0],cyl_centre[0,1],cyl_centre[0,2],normal[0],normal[1],normal[2],r,CCI]])

    def point_cloud_annotations(self,character_size,xpos,ypos,zpos,r,text):
        def convert_character_cells_to_points(character):
            character = np.rot90(character,axes=(1,0))
            index_i = 0
            index_j = 0
            points = np.zeros((0,3))
            for i in character:
                for j in i:
                    if j == 1:
                        points = np.vstack((points,np.array([[index_i,index_j,0]])))
                    index_j += 1
                index_j = 0
                index_i += 1
            
            roll_mat = np.array([[1,0,0],
                                 [0, np.cos(-np.pi/4), -np.sin(-np.pi/4)],
                                 [0, np.sin(-np.pi/4),np.cos(-np.pi/4)]])
            points = np.dot(points,roll_mat)
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
        text_points = np.zeros((11,0))
        for i in text:
            text_points = np.hstack((text_points,np.array(get_character(str(i)))))
        points = convert_character_cells_to_points(text_points)
        
        points = points*character_size + [xpos+0.2+0.5*r,ypos,zpos]
        return points

    # @staticmethod
    @classmethod
    def fit_cylinder(cls,skeleton_points,point_cloud,num_neighbours,slice_increment):
        # print("Fitting cylinders...")
        point_cloud = point_cloud[:,:3]
        skeleton_points = skeleton_points[:,:3]
        cyl_array = np.zeros((0,8))
        line_centre = np.mean(skeleton_points[:,:3],axis=0)
        _, _, vh = np.linalg.svd(line_centre-skeleton_points)
        line_v_hat = vh[0]/np.linalg.norm(vh[0])
        
        while skeleton_points.shape[0]>num_neighbours:
            nn = NearestNeighbors()
            nn.fit(skeleton_points)
            minpoint = np.min(skeleton_points[:,2])
            starting_point = np.atleast_2d(skeleton_points[skeleton_points[:,2]==minpoint])
            group = skeleton_points[nn.kneighbors(starting_point,
                                                  n_neighbors=num_neighbours)[1][0]]
            line_centre = np.mean(group[:,:3],axis=0)
            length = np.linalg.norm(np.max(group,axis=0)-np.min(group,axis=0))
            plane_slice = point_cloud[np.linalg.norm(abs(line_v_hat*(point_cloud-line_centre)),axis=1)<(length/2)] #calculate distances to plane at centre of line.
            if plane_slice.shape[0]>0:
                cylinder = MeasureTree.fit_circle_3D(plane_slice,line_v_hat,return_cyl_vis=False)
                # cylinder = np.hstack((cylinder,np.array([[line_centre[0],line_centre[1],line_centre[2],length]])))
                cyl_array = np.vstack((cyl_array, cylinder))
            skeleton_points = skeleton_points[skeleton_points[:,2]!=minpoint]
        
        return cyl_array
    
    # @staticmethod
    @classmethod
    def clustering(cls,points,eps=0.05):
        # print("Clustering...")
        db = DBSCAN(eps=eps, min_samples=2,metric='euclidean', algorithm='kd_tree').fit(points)
        return np.hstack((points,np.atleast_2d(db.labels_).T))
    
    # @staticmethod
    @classmethod
    def circumferential_completeness_index(cls,fitted_circle_centre, estimated_radius, slice_points):
        angular_region_degrees = 10
        minimum_radius_counted = estimated_radius*0.7
        maximum_radius_counted = estimated_radius*1.3
        num_sections = 360/angular_region_degrees
        angles = np.linspace(-180,180,num=int(num_sections),endpoint=False)
        theta = np.zeros((1,1))
        completeness=0
        for point in slice_points:
            if ((point[1]-fitted_circle_centre[1])**2+(point[0]-fitted_circle_centre[0])**2)**0.5 >= minimum_radius_counted and ((point[1]-fitted_circle_centre[1])**2+(point[0]-fitted_circle_centre[0])**2)**0.5 <= maximum_radius_counted:
                theta = np.vstack((theta,(math.degrees(math.atan2((point[1]-fitted_circle_centre[1]),(point[0]-fitted_circle_centre[0]))))))
        for angle in angles:
            if np.shape(np.where(theta[np.where(theta>=angle)]<(angle+angular_region_degrees)))[1] > 0:
                completeness += 1
        return completeness/num_sections
    
    # @staticmethod
    @classmethod
    def threaded_cyl_fitting(cls,args):
        skel_cluster,point_cluster,num_neighbours,slice_increment = args
        cyl_array = np.zeros((0,8))
        if skel_cluster.shape[0] > num_neighbours:
            cyl_array = MeasureTree.fit_cylinder(skel_cluster,point_cluster,num_neighbours=num_neighbours,slice_increment=slice_increment)
        return cyl_array
    
    def noise_filtering(self,points,min_neighbour_dist,min_neighbours):
        kdtree = spatial.cKDTree(points[:,:3],leafsize=1000)
        results = kdtree.query_ball_point(points[:,:3],r=min_neighbour_dist)
        if len(results)!=0:
            return points[[len(i)>=min_neighbours for i in results]]
        else:
            return points
    
    # @staticmethod
    @classmethod
    def slice_clustering(cls,input_data):
        cluster_array_internal = np.zeros((0,6))
        means = np.zeros((0,3))
        new_slice,slice_increment = input_data
        if new_slice.shape[0]>0:
            new_slice = MeasureTree.clustering(new_slice[:,:3],eps=slice_increment*3)
            for cluster_id in range(0,int(np.max(new_slice[:,-1]))+1):
                cluster = new_slice[new_slice[:,-1]==cluster_id]
                mean = np.median(cluster[:,:3],axis=0)
                means = np.vstack((means,mean))
                cluster_array_internal = np.vstack((cluster_array_internal,np.hstack((cluster[:,:3],np.zeros((cluster.shape[0],3))+mean))))
        return cluster_array_internal,means
    
    # @staticmethod
    @classmethod
    def within_angle_tolerances(cls,normal1,normal2,angle_tolerance):
        """Checks if normal1 and normal2 are within "angle_tolerance"
        of each other."""
        norm1 = normal1/np.atleast_2d(np.linalg.norm(normal1,axis=1)).T
        norm2 = normal2/np.atleast_2d(np.linalg.norm(normal2,axis=1)).T
        dot = np.clip(np.einsum('ij, ij->i', norm1, norm2),a_min=-1,a_max=1)
        theta = np.degrees(np.arccos(dot))
        return abs((theta > 90)*180-theta) <= angle_tolerance
    
    # @staticmethod
    @classmethod
    def within_search_cone(cls,normal1,vector1_2,search_angle):
        norm1 = normal1/np.linalg.norm(normal1)
        if not (vector1_2==0).all():
            norm2 = vector1_2/np.linalg.norm(vector1_2)
            dot = np.dot(norm1,norm2)
            if dot > 1: #floating point problems...
                dot = 1
            elif dot < -1:
                dot = -1
            
            theta = math.degrees(np.arccos(dot))
            # print('Cone Angle',abs((theta > 90)*180-theta) <= search_angle)
            return abs((theta > 90)*180-theta) <= search_angle
        else:
            return False
    
    def run_measurement_extraction(self):
        slice_heights = np.linspace(np.min(self.stem_points[:,2]),np.max(self.stem_points[:,2]),int(np.ceil((np.max(self.stem_points[:,2])-np.min(self.stem_points[:,2]))/self.slice_increment)))
        means_array = np.zeros((0,3))
        self.input_data = []
        print("Making slices...")
        i = 0
        max_i = slice_heights.shape[0]
        for slice_height in slice_heights:
            if i % 10 == 0:
                # print ('{i}/{max_i}\r'.format(i,max_i),)
                print('\r',i,'/',max_i,end='')
            i += 1
            # print('{:4.2f} m'.format(slice_height))
            new_slice = self.stem_points[np.logical_and(self.stem_points[:,2]>=slice_height,self.stem_points[:,2]<slice_height+self.slice_thickness)]
            if new_slice.shape[0]>0:
                self.input_data.append([new_slice,self.slice_increment])
        print('\r',max_i,'/',max_i,end='')
        print('\nDone\n')
        cluster_array = np.zeros((0,6))
        print("Starting multithreaded slice clustering...")
        j = 0
        max_j = len(self.input_data)
        with get_context("spawn").Pool(processes=self.num_procs) as pool:
            for i in pool.imap_unordered(MeasureTree.slice_clustering,self.input_data):
                if j % 100 == 0:
                    print('\r',j,'/',max_j,end='')
                j += 1
                cluster, ms = i
                cluster_array = np.vstack((cluster_array,cluster))
                means_array = np.vstack((means_array,ms))
        print('\r',max_j,'/',max_j,end='')
        print('\nDone\n')
        
        print('Clustering skeleton...')
        means_array = MeasureTree.clustering(means_array[:,:3],eps=self.slice_increment*2)
        # print("Saving Means...")
        # np.savetxt(self.directory+'data/postprocessed_point_clouds/'+self.input_point_cloud+'/means.csv',means_array)    
        # np.savetxt(self.directory+'data/postprocessed_point_clouds/'+self.input_point_cloud+'/cluster_array.csv',cluster_array)    
        # print("Loading")
        # means_array = np.array(pd.read_csv(self.directory+'data/postprocessed_point_clouds/'+self.input_point_cloud+'/means.csv',header=None,index_col=None,delim_whitespace=True))
        # cluster_array = np.array(pd.read_csv(self.directory+'data/postprocessed_point_clouds/'+self.input_point_cloud+'/cluster_array.csv',header=None,index_col=None,delim_whitespace=True))
        np.save(self.directory+'data/postprocessed_point_clouds/'+self.input_point_cloud+'/cluster_array.npy',cluster_array) # np.savetxt(self.directory+'data/postprocessed_point_clouds/'+self.input_point_cloud+'/means.csv',means_array)    
        np.save(self.directory+'data/postprocessed_point_clouds/'+self.input_point_cloud+'/means_array.npy',means_array)    
        print("Loading")
        means_array = np.load(self.directory+'data/postprocessed_point_clouds/'+self.input_point_cloud+'/means_array.npy')
        cluster_array = np.load(self.directory+'data/postprocessed_point_clouds/'+self.input_point_cloud+'/cluster_array.npy')
        print('Done\n')
        
        #Assign unassigned mean points to the nearest group.
        _,neighbours = spatial.cKDTree(means_array[means_array[:,-1]!=-1,:3],leafsize=1000).query(means_array[means_array[:,-1]==-1,:3], k=2)
        means_array[means_array[:,-1]==-1,-1] = means_array[means_array[:,-1]!=-1][:,-1][neighbours[:,1]]
        
        input_data = []
        i = 0
        max_i = int(np.max(means_array[:,-1])+1)
        cl_kdtree = spatial.cKDTree(cluster_array[:,3:],leafsize=1000)
        cluster_ids = range(0,max_i)
        print('Making initial branch/stem section clusters...')
        
        for cluster_id in cluster_ids:
            if i % 100 == 0:
                print('\r',i,'/',max_i,end='')
            i += 1
            skel_cluster = means_array[means_array[:,-1]==cluster_id,:3]
            sc_kdtree = spatial.cKDTree(skel_cluster,leafsize=1000)
            results = np.unique(np.hstack(sc_kdtree.query_ball_tree(cl_kdtree,r=0.000000001)))
            ca = cluster_array[results,:3]
            input_data.append([skel_cluster[:,:3],ca[:,:3],self.num_neighbours,self.slice_increment])
        print('\r',max_i,'/',max_i,end='')
        print('\nDone\n')
        
        print("Starting multithreaded cylinder fitting...")
        j = 0
        max_j = len(input_data)
        # cyl_vis_array = np.zeros((0,4))
        full_cyl_array = np.zeros((0,8))
        with get_context("spawn").Pool(processes=self.num_procs) as pool:
            for i in pool.imap_unordered(MeasureTree.threaded_cyl_fitting,input_data):
                full_cyl_array = np.vstack((full_cyl_array,i))
                if j % 10 == 0:
                    print('\r',j,'/',max_j,end='')
                j += 1
        print('\r',max_i,'/',max_i,end='')
        print('\nDone\n')
        #cyl_array = [x,y,z,nx,ny,nz,r,CCI,line_centre[0],line_centre[1],line_centre[2],length]
        print("Saving cylinder array...")
        np.savetxt(self.directory+'data/postprocessed_point_clouds/'+self.input_point_cloud+'/cyls.csv',full_cyl_array)    
        # global sorted_points, sorted_by_tree,sorted_by_tree2, full_cyl_array,sorted_points_other,sorted_by_tree_other,sorted_by_tree2_other

        # full_cyl_array = np.loadtxt(self.directory+'data/postprocessed_point_clouds/'+self.input_point_cloud+'/cyls.csv')    
        DTM = np.loadtxt("../data/postprocessed_point_clouds/"+self.input_point_cloud+"/DTM.csv")
        print("Sorting Cylinders...")
        sorted_points, unsorted_points = MeasureTree.cylinder_sorting(full_cyl_array,
                                                          angle_tolerance=30,
                                                          search_angle=50,
                                                          distance_tolerance=2.0,
                                                          min_radius_ratio=0.0,
                                                          max_radius_ratio=2000.)
        
        np.savetxt(self.directory+'data/postprocessed_point_clouds/'+self.input_point_cloud+'/sorted_points.csv',sorted_points)    
        # sorted_points = np.loadtxt(self.directory+'data/postprocessed_point_clouds/'+self.input_point_cloud+'/sorted_points.csv')    

        print('Correcting Cylinder assignments...')
        sorted_by_tree = np.zeros((0,sorted_points.shape[1]))
        t_id = 1
        max_search_radius = 2.0
        min_points = 5
        search_cone_angle = 20
        to_be_assigned = sorted_points[:]

        for tree_id in np.unique(to_be_assigned[:,-1]):
            tree = to_be_assigned[to_be_assigned[:,-1]==int(tree_id)]
            tree_kdtree = spatial.cKDTree(sorted_by_tree[:,:3],leafsize=1000)
            if tree.shape[0]>=min_points:
                lowest_point = tree[np.argmin(tree[:,2]),:3]
                neighbours = sorted_by_tree[tree_kdtree.query_ball_point(lowest_point,r=max_search_radius)]
                search_direction = np.array([0,0,-1])#median_tree_cyl_vector * np.array([1,1,-1])
                dp = np.dot((neighbours[:,:3]-lowest_point)/np.atleast_2d(np.linalg.norm(neighbours[:,:3]-lowest_point,axis=1)).T,search_direction)
                dp = np.arccos(np.clip(dp,-1,1))
                neighbours = neighbours[np.logical_and(math.radians(np.abs(search_cone_angle))>dp,dp>0)]
                neighbours_kdtree = spatial.cKDTree(neighbours[:,:3],leafsize=1000)
                lowest_point_z = lowest_point[2] - griddata((DTM[:,0],DTM[:,1]),DTM[:,2],lowest_point[0:2],method='linear',fill_value=np.median(DTM[:,2]))
                if neighbours.shape[0]>1:
                    closest_valid_neighbour = neighbours[neighbours_kdtree.query(lowest_point, k=2)[1][1]]
                    closest_valid_tree_id = closest_valid_neighbour[-1]
                    tree[:,-1] = closest_valid_tree_id
                    sorted_by_tree = np.vstack((sorted_by_tree,tree))
                
                elif lowest_point_z < 1:
                    tree[:,-1] = t_id
                    sorted_by_tree = np.vstack((sorted_by_tree,tree))
                    t_id += 1
        
        
        # sorted_by_tree = np.loadtxt(self.directory+'data/postprocessed_point_clouds/'+self.input_point_cloud+'/sorted_by_tree.csv')    
        
        sorted_by_tree = sorted_by_tree[np.argsort(sorted_by_tree[:,6])]
        sorted_by_tree2 = np.zeros((0,sorted_by_tree.shape[1]))
        while sorted_by_tree.shape[0] > 0:
            kdtree = spatial.cKDTree(sorted_by_tree[:,:3],leafsize=1000)
            results = kdtree.query_ball_point(sorted_by_tree[0,:3],r=sorted_by_tree[0,6]*2)
            keep_point = sorted_by_tree[np.argmax(sorted_by_tree[results,6])]
            sorted_by_tree2 = np.vstack((sorted_by_tree2,keep_point))
            sorted_by_tree = sorted_by_tree[1:,:]
            # print(sorted_by_tree.shape,sorted_by_tree2.shape)
        # sorted_by_tree2 = np.vstack({tuple(row) for row in sorted_by_tree2})
        sorted_by_tree2 = np.vstack(([tuple(row) for row in sorted_by_tree2]))


        
        
        np.savetxt(self.directory+'data/postprocessed_point_clouds/'+self.input_point_cloud+'/sorted_by_tree2.csv',sorted_by_tree2)    
        sorted_by_tree = sorted_by_tree2
        tree_information = np.zeros((0,7))
        print("Sorting vegetation points...")
        self.ground_veg_cutoff = 2
        self.num_veg_assignment_iterations = 20
        self.veg_step_distance = 1
        self.ground_veg = self.vegetation_points[self.vegetation_points[:,-1]<=self.ground_veg_cutoff]
        self.vegetation_points = self.vegetation_points[self.vegetation_points[:,-1]>self.ground_veg_cutoff]
        # assign vegetation to trees
        # and low lying vegetation to separate category.
        self.vegetation_points = np.hstack((self.vegetation_points,np.zeros((self.vegetation_points.shape[0],1))))
        self.ground_veg = np.hstack((self.ground_veg,np.zeros((self.ground_veg.shape[0],1))))
        
        stem_kdtree = spatial.cKDTree(sorted_by_tree[:,:3],leafsize=1000)
        results = stem_kdtree.query(self.vegetation_points[:,:3], k=1,distance_upper_bound=self.veg_step_distance)
        # print(sorted_by_tree[result[1][1],-3])
        self.vegetation_points[results[0]<self.veg_step_distance,-1] = sorted_by_tree[results[1][results[0]<self.veg_step_distance],-1] 
        assigned_vegetation_points = self.vegetation_points[self.vegetation_points[:,-1]!=0]
        unassigned_vegetation_points = self.vegetation_points[self.vegetation_points[:,-1]==0]
        
        for iteration in range(0,self.num_veg_assignment_iterations):
            assigned_vegetation_points_kdtree = spatial.cKDTree(assigned_vegetation_points[:,:3],leafsize=1000)
            results = assigned_vegetation_points_kdtree.query(unassigned_vegetation_points[:,:3], k=1,distance_upper_bound=self.veg_step_distance)
            #Assign the label of the nearest vegetation if less than 3 m away.
            unassigned_vegetation_points[results[0]<self.veg_step_distance,-1] = assigned_vegetation_points[results[1][results[0]<self.veg_step_distance],-1] 
    
            assigned_vegetation_points = np.vstack((assigned_vegetation_points,unassigned_vegetation_points[unassigned_vegetation_points[:,-1]!=0]))
            unassigned_vegetation_points = unassigned_vegetation_points[unassigned_vegetation_points[:,-1]==0]
        
        
        print("Saving vegetation points...")
        np.savetxt(self.directory+'data/postprocessed_point_clouds/'+self.input_point_cloud+'/vegetation_points_assigned.csv',assigned_vegetation_points)#np.hstack((sorted_by_tree[:,:3],sorted_by_tree[:,-5:])))    
        np.savetxt(self.directory+'data/postprocessed_point_clouds/'+self.input_point_cloud+'/ground_veg.csv',self.ground_veg)#np.hstack((sorted_by_tree[:,:3],sorted_by_tree[:,-5:])))    
        print("Done")
        # assigned_vegetation_points = np.loadtxt(self.directory+'data/postprocessed_point_clouds/'+self.input_point_cloud+'/vegetation_points_assigned.csv')#np.hstack((sorted_by_tree[:,:3],sorted_by_tree[:,-5:])))    
        
        """
        Idea is you get the neighbours on either side of a cyl point.
        If both neighbours are within the resolution range, ignore them.
        If one neighbour is within the resolution range and the other neighbour isn't, interpolate the one that is further away.
        Loop this over the entire point set (individual tree cyl points) until no more neighbours are within the resolution range.
        
        After that.
        Get volume from lowest point to nearest point. Add to volume.
        Remove start point.
        
        get volume from previous nearest point to next nearest point.
        add to volume.
        remove start point
         
        loop until volume measured!
        
        run through all cyls.
        check if they have 0, 1 or 2 neighbours in search range.
        if 0 in search range, delete cylinder.
        if 1 in search range, interpolate
        """
        search_cone_angle = 30
        max_search_distance = 1.
        min_resolution = 0.025
        max_radius_ratio = 1000
        min_radius_ratio = 0.0001
        processed_trees = np.zeros((0,sorted_by_tree.shape[1]+1))
        text_point_cloud = np.zeros((0,3))
        tree_measurements = np.zeros((0,7)) #X,Y,Z, TREE HEIGHT, HEIGHT BIN, DIAMETER, TREE_ID
        print("Interpolating cylinders and extracting tree information...")
        for tree_id in range(1,int(np.max(sorted_by_tree[:,-1])+1)):
            tree = sorted_by_tree[sorted_by_tree[:,-1]==tree_id]
            tree_vegetation = assigned_vegetation_points[assigned_vegetation_points[:,-1]==tree_id]
            
            current_tree = np.zeros((0,tree.shape[1]))
            if tree.shape[0] > min_points:
                
                for cyl in tree:
                    closest_neighbour,opposing_neighbour = MeasureTree.get_2_opposing_neighbours(cyl,tree,
                                                                                      max_search_distance=max_search_distance,
                                                                                      search_cone_angle=search_cone_angle,
                                                                                      max_radius_ratio=max_radius_ratio,
                                                                                      min_radius_ratio=min_radius_ratio)
                    if np.sum(closest_neighbour)!=0 and np.sum(opposing_neighbour)!=0:
                        current_tree = np.vstack((current_tree,cyl))
                        if closest_neighbour[-1] > min_resolution:
                            current_tree = np.vstack((current_tree,MeasureTree.interpolate_cyl(cyl,closest_neighbour[:-1],resolution=min_resolution)[1:-1,:]))
                        
                        if opposing_neighbour[-1] > min_resolution:
                            current_tree = np.vstack((current_tree,MeasureTree.interpolate_cyl(cyl,opposing_neighbour[:-1],resolution=min_resolution)[1:-1,:]))
                    
                    elif np.sum(closest_neighbour)!=0 and np.sum(opposing_neighbour)==0:
                        current_tree = np.vstack((current_tree,cyl))
                        if closest_neighbour[-1] > min_resolution:
                            current_tree = np.vstack((current_tree,MeasureTree.interpolate_cyl(cyl,closest_neighbour[:-1],resolution=min_resolution)[1:-1,:]))
                    
                    elif np.sum(closest_neighbour)==0 and np.sum(opposing_neighbour)!=0:
                        current_tree = np.vstack((current_tree,cyl))
                        if opposing_neighbour[-1] > min_resolution:
                            current_tree = np.vstack((current_tree,MeasureTree.interpolate_cyl(cyl,opposing_neighbour[:-1],resolution=min_resolution)[1:-1,:]))
                wood_volume = 0
                """
                Need to remove smaller cylinders from inside bigger ones.
                """
                for cyl in current_tree:
                    
                    kdtree = spatial.cKDTree(current_tree[:,:3],leafsize=1000)
                    results = kdtree.query_ball_point(cyl[:3],r=cyl[6]*1.2) #Find other cylinders within 1.2 x radius of the cylinder point.
                    neighbours = current_tree[results]
                    current_tree = np.delete(current_tree,results,axis=0)
                    current_tree = np.vstack((current_tree,neighbours[neighbours[:,6]>=0.8*cyl[6]])) #Remove cylinders less than 0.8*radius of current cyl point.
                processed_tree = current_tree[:]
                
                while current_tree.shape[0]>1:
                    lowest_point = current_tree[np.argmin(current_tree[:,2])]
                    kdtree = spatial.cKDTree(current_tree[:,:3],leafsize=1000)
                    closest_point = current_tree[kdtree.query(lowest_point[:3], k=2)[1][1]]
                    section_length = np.linalg.norm(closest_point[:3] - lowest_point[:3])
                    section_volume = (1/3)*math.pi*section_length*((lowest_point[6]**2)+lowest_point[6]*(closest_point[6]+closest_point[6]**2))
                    wood_volume = wood_volume + section_volume
                    current_tree = np.vstack((current_tree[:np.argmin(current_tree[:,2])],current_tree[np.argmin(current_tree[:,2])+1:]))
                
                grid = griddata((DTM[:,0],DTM[:,1]),DTM[:,2],processed_tree[:,0:2],method='linear',fill_value=np.median(DTM[:,2]))
                processed_tree = np.hstack((processed_tree,np.atleast_2d(processed_tree[:,2] - grid).T))
                processed_trees = np.vstack((processed_trees,processed_tree))

                XY_at_BH = processed_tree[np.logical_and(processed_tree[:,-1]>1.0,processed_tree[:,-1]<1.6),:2]
                DBH = np.mean(processed_tree[np.logical_and(processed_tree[:,-1]>1.0,processed_tree[:,-1]<1.6),6])*2
                median_CCI_at_BH = np.mean(processed_tree[np.logical_and(processed_tree[:,-1]>1.0,processed_tree[:,-1]<1.6),7])
                if np.isnan(DBH):
                    DBH = 0
                    median_CCI_at_BH = 0
                crown_volume = 0
                #NEED TO ADD DBH IN
                if np.any(np.isnan(np.mean(XY_at_BH,axis=0))):
                    X,Y = processed_tree[np.argmin(processed_tree[:,-1]),:2]
                else:
                    X,Y = np.mean(XY_at_BH,axis=0)
                    
                Z = griddata((DTM[:,0],DTM[:,1]),DTM[:,2],np.array([[X,Y]]),method='linear',fill_value=np.median(DTM[:,2]))[0]
                # lowest_measurement_to_DTM = processed_tree[np.argmin(processed_tree[:,-1]),-1]
                if tree_vegetation[:,-2].shape[0]==0:
                    tree_vegetation_height = 0
                else:
                    tree_vegetation_height = np.max(tree_vegetation[:,-2])
                    
                height = np.max(np.array([np.max(processed_tree[:,-1]),tree_vegetation_height])) #Get either the highest vegetation point or highest stem point (whichever is greater)
                if np.isnan(height) or height < 0:
                    height = 0
                if np.isnan(wood_volume) or wood_volume < 0:
                    wood_volume = 0    
                
                #TODO
                #Extract diameters at heights and match them up to corresponding manual measurements
                
                diameter_measurement_heights = np.linspace(0,np.ceil(height/self.diameter_measurement_increment)*self.diameter_measurement_increment,int(np.ceil(height/self.diameter_measurement_increment))+1)
                for diameter_measurement_height in diameter_measurement_heights:
                    cyls_near_measurement_height = processed_tree[np.logical_and(processed_tree[:,-1]>=diameter_measurement_height-0.5*self.diameter_measurement_height_range,
                                                                                  processed_tree[:,-1]<diameter_measurement_height+0.5*self.diameter_measurement_height_range)]
                    try:
                        tree_measurement = np.array([[X,Y,Z+diameter_measurement_height,height,diameter_measurement_height,np.max(cyls_near_measurement_height[:,6])*2,tree_id]])#X,Y,Z, TREE HEIGHT, HEIGHT BIN, DIAMETER, TREE_ID
                        tree_measurements = np.vstack((tree_measurements,tree_measurement))
                    except:
                        None
                tree_info = np.array([[X,Y,Z,DBH,height,wood_volume,crown_volume]])




                text_size = 0.0025
                line_height = 0.025
                if height > 1.3:
                    height_offset = 1.3
                else:
                    height_offset = height
                line0 = self.point_cloud_annotations(text_size,X,Y+line_height,Z+line_height+height_offset,DBH*0.5,    '     DIAM: ' + str(np.around(DBH,2))+'m')
                line1 = self.point_cloud_annotations(text_size,X,Y,Z+height_offset,DBH*0.5,                            'CCI AT BH: ' + str(np.around(median_CCI_at_BH,2)))
                line2 = self.point_cloud_annotations(text_size,X,Y-2*line_height,Z-2*line_height+height_offset,DBH*0.5,'   HEIGHT: ' + str(np.around(height,2))+'m')
                line3 = self.point_cloud_annotations(text_size,X,Y-3*line_height,Z-3*line_height+height_offset,DBH*0.5,'   VOLUME: ' + str(np.around(wood_volume,2))+'m')
                # print(X,Y,Z,X,Y,Z+height)
                height_measurement_line = self.points_along_line(X,Y,Z,X,Y,Z+height,resolution=0.025)
                dbh_circle_points = self.create_3d_circles_as_points_flat(X,Y,Z+height_offset,DBH/2,circle_points=100)
                text_point_cloud = np.vstack((text_point_cloud,line0,line1,line2,line3,height_measurement_line,dbh_circle_points))
                tree_information = np.vstack((tree_information,tree_info))
                
        tree_information = tree_information[tree_information[:,5]>self.min_tree_volume] #remove trees with tiny volumes.
        np.savetxt(self.directory+'data/postprocessed_point_clouds/'+self.input_point_cloud+'/tree_information.csv',tree_information)#np.hstack((sorted_by_tree[:,:3],sorted_by_tree[:,-5:])))    
        np.savetxt(self.directory+'data/postprocessed_point_clouds/'+self.input_point_cloud+'/text_point_cloud.csv',text_point_cloud)#np.hstack((sorted_by_tree[:,:3],sorted_by_tree[:,-5:])))    
        print("Tree Information saved.")
        # print("Saving sorted cylinder array...")
        # print(sorted_by_tree.shape)
        # np.savetxt(self.directory+'data/postprocessed_point_clouds/'+self.input_point_cloud+'/sorted_by_tree.csv',sorted_by_tree)
        np.savetxt(self.directory+'data/postprocessed_point_clouds/'+self.input_point_cloud+'/processed_trees.csv',processed_trees) 
        np.savetxt(self.directory+'data/postprocessed_point_clouds/'+self.input_point_cloud+'/tree_measurements.csv',tree_measurements) 
    
        print("Making sorted cylinder visualisation...")
        j = 0
        max_j = np.shape(processed_trees)[0]
        cyl_vis_final = np.zeros((0,4))
        with get_context("spawn").Pool(processes=self.num_procs) as pool:
            for i in pool.imap_unordered(MeasureTree.make_cyl_visualisation,processed_trees):
                cyl_vis_final = np.vstack((cyl_vis_final,i))
                if j % 100 == 0:
                    print('\r',j,'/',max_j,end='')
                j += 1
    
        
        print("Saving cylinder visualisation...")
        np.savetxt(self.directory+'data/postprocessed_point_clouds/'+self.input_point_cloud+'/cyl_vis2.csv',cyl_vis_final)    
        print("Done")

if __name__ == '__main__':
    from inference import semantic_segmentation
    from post_segmentation_script import PostProcessing
    # from measure import MeasureTree
    point_clouds_to_process = ["tls_b_test1.csv",
                                "tls_b_test2.csv",
                                "tls_b_test3.csv",]
    
    # point_clouds_to_process = ["tls_benchmark_P1.csv",
    #                            "tls_benchmark_P2.csv",
    #                            "tls_benchmark_P3.csv",
    #                            "tls_benchmark_P4.csv",
    #                            "tls_benchmark_P5.csv",
    #                            "tls_benchmark_P6.csv"]
    
    parameters = {'directory':'../',
              'fileset':'test',
              'input_point_cloud':None,
              'model_filename':'../model/model.pth',
              'batch_size':20,
              'box_dimensions':[6,6,6],
               'box_overlap':[0.5,0.5,0.5],
               # 'box_overlap':[0.,0.,0.],
               'min_points_per_box':1000,
              'max_points_per_box':20000,
              'subsample':False,
              'subsampling_min_spacing':0.025,
              'num_procs':20,
              'noise_class':0,
              'terrain_class':1,
              'vegetation_class':2,
              'cwd_class':3,
              'stem_class':4,
    
              #DTM Settings
              'coarse_grid_resolution':6,
              'fine_grid_resolution':0.5,
              'max_diameter':5,
              # Measurement settings6
              'num_neighbours':5,
              'slice_thickness':0.15,
              'slice_increment':0.02,
              'diameter_measurement_increment':0.1,
              'diameter_measurement_height_range':0.1,
              'min_tree_volume':0.0001}
    
    for point_cloud in point_clouds_to_process:
        parameters['input_point_cloud'] = point_cloud
        
        # sem_seg = semantic_segmentation(parameters)
        # sem_seg.run_preprocessing()
        # sem_seg.inference()
        
        # object_1 = PostProcessing(parameters)
        # object_1.process_point_cloud()
    
        measurements = MeasureTree(parameters)
        measurements.run_measurement_extraction()