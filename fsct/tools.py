from sklearn.neighbors import NearestNeighbors
import numpy as np
import glob
from sklearn.neighbors import NearestNeighbors
from multiprocessing import Pool, get_context
import pandas as pd
import os
import shutil
from sklearn.cluster import DBSCAN
from scipy.interpolate import griddata
from copy import deepcopy
from multiprocessing import get_context
from scipy import spatial
import string
import struct
from scipy import ndimage

import laspy
from fsct.io import ply_io, pcd_io

class dict2class:

    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)

def make_folder_structure(params):
    
    if params.odir == None:
        params.odir = os.path.join(params.directory, params.filename + '_FSCT_output')
    
    params.working_dir = os.path.join(params.odir, params.basename + '.tmp')
    
    if not os.path.isdir(params.odir):
        os.makedirs(odir)

    if not os.path.isdir(params.working_dir):
        os.makedirs(params.working_dir)
    else:
        shutil.rmtree(params.working_dir, ignore_errors=True)
        os.makedirs(params.working_dir)
        
    if params.verbose:
        print('output directory:', params.odir)
        print('scratch directory:', params.working_dir)
    
    return params
    
def voxelise(tmp, length, method='random', z=True):

    tmp.loc[:, 'xx'] = tmp.x // length * length
    tmp.loc[:, 'yy'] = tmp.y // length * length
    if z: tmp.loc[:, 'zz'] = tmp.z // length * length

    if method == 'random':
            
        code = lambda: ''.join(np.random.choice([x for x in string.ascii_letters], size=8))
            
        xD = {x:code() for x in tmp.xx.unique()}
        yD = {y:code() for y in tmp.yy.unique()}
        if z: zD = {z:code() for z in tmp.zz.unique()}
            
        tmp.loc[:, 'VX'] = tmp.xx.map(xD) + tmp.yy.map(yD) 
        if z: tmp.VX += tmp.zz.map(zD)
   
    elif method == 'bytes':
        
        code = lambda row: np.array([row.xx, row.yy] + [row.zz] if z else []).tobytes()
        tmp.loc[:, 'VX'] = self.pc.apply(code, axis=1)
        
    else:
        raise Exception('method {} not recognised: choose "random" or "bytes"')
 
    return tmp 


def downsample(pc, vlength, 
               accurate=False,
               keep_columns=[], 
               keep_points=False,
               voxel_method='random', 
               return_VX=False,
               verbose=False):

    """
    Downsamples a point cloud so that there is one point per voxel.
    Points are selected as the point closest to the median xyz value
    
    Parameters
    ----------
    
    pc: pd.DataFrame with x, y, z columns
    vlength: float
    
    
    Returns
    -------
    
    pd.DataFrame with boolean downsample column
    
    """

    pc = pc.drop(columns=[c for c in ['downsample', 'VX'] if c in pc.columns])   
    columns = pc.columns.to_list() + keep_columns # required for tidy up later
    if return_VX: columns += ['VX']
    pc = voxelise(pc, vlength, method=voxel_method)

    if accurate:
        # groubpy to find central (closest to median) point
        groupby = pc.groupby('VX')
        pc.loc[:, 'mx'] = groupby.x.transform(np.median)
        pc.loc[:, 'my'] = groupby.y.transform(np.median)
        pc.loc[:, 'mz'] = groupby.z.transform(np.median)
        pc.loc[:, 'dist'] = np.linalg.norm(pc[['x', 'y', 'z']].to_numpy(dtype=np.float32) - 
                                           pc[['mx', 'my', 'mz']].to_numpy(dtype=np.float32), axis=1)
        pc.loc[:, 'downsample'] = False
        pc.loc[~pc.sort_values(['VX', 'dist']).duplicated('VX'), 'downsample'] = True

    else:
        pc.loc[:, 'downsample'] = False
        pc.loc[~pc.VX.duplicated(), 'downsample'] = True
        
    if keep_points:
        return pc[columns + ['downsample']]
    else:
        return pc.loc[pc.downsample][columns]
    
def compute_plot_centre(pc):
    """calculate plot centre"""
    plot_min, plot_max = pc[['x', 'y']].min(), pc[['x', 'y']].max()
    return (plot_min + ((plot_max - plot_min) / 2)).values

def compute_bbox(pc):
    bbox_min = pc.min().to_dict()
    bbox_min = {k + 'min':v for k, v in bbox_min.items()}
    bbox_max = pc.max().to_dict()
    bbox_max = {k + 'max':v for k, v in bbox_max.items()}
    return dict2class({**bbox_min, **bbox_max})
    
def load_file(filename, 
              plot_centre=None, plot_radius=0, plot_radius_buffer=0, 
              additional_fields=[], verbose=False):
    
    file_extension = os.path.splitext(filename)[1]
    headers = ['x', 'y', 'z'] + additional_fields

    if file_extension == '.las' or file_extension == '.laz':
        inFile = laspy.read(filename)
        pc = np.vstack((inFile.x, inFile.y, inFile.z))
        for header in additional_fields:
            if header in list(inFile.point_format.dimension_names):
                pc = np.vstack((pc, getattr(inFile, header)))
            else:
                headers.drop(header)
        pc = pd.DataFrame(data=pc, columns=headers)

    elif file_extension == '.ply':
        pc = ply_io.read_ply(filename)
        pc = pc[headers]
        
    elif file_extension == '.pcd':
        pc = pcd_io.read_pcd(filename)
        pc = pc[headers]
        
    else:
        raise Exception('point cloud format not recognised' + filename)

    original_num_points = len(pc)
    
    if len(pc) > 0: # clip round plot
        if plot_radius > 0:
            if plot_centre is None:
                plot_centre = compute_plot_centre(pc)

            distances = np.linalg.norm(pc[['x', 'y']].values - plot_centre, axis=1)
            keep_points = distances < plot_radius + plot_radius_buffer
            pc = pc.loc[keep_points]
    
    if verbose: print(f'read in {filename} with {len(pc)} points')
    
    return pc


def save_file(filename, pointcloud, additional_fields=[], verbose=False):

#     if pointcloud.shape[0] == 0: 
#         print(filename, 'is empty...')
#     else:
    if verbose:
        print('Saving file:', filename)
        
    cols = ['x', 'y', 'z'] + additional_fields

    if filename.endswith('.las'):
        las = laspy.create(file_version="1.4", point_format=7)
        las.header.offsets = np.min(pointcloud[:, :3], axis=0)
        las.header.scales = [0.001, 0.001, 0.001]

        las.x = pointcloud[:, 0]
        las.y = pointcloud[:, 1]
        las.z = pointcloud[:, 2]

        if len(additional_fields) != 0:
            additional_fields = additional_fields[3:]

            #  The reverse step below just puts the headings in the preferred order. They are backwards without it.
            col_idxs = list(range(3, pointcloud.shape[1]))
            additional_fields.reverse()

            col_idxs.reverse()
            for header, i in zip(additional_fields, col_idxs):
                column = pointcloud[:, i]
                if header in ['red', 'green', 'blue']:
                    setattr(las, header, column)
                else:
                    las.add_extra_dim(laspy.ExtraBytesParams(name=header, type="f8"))
                    setattr(las, header, column)
        las.write(filename)
        if not verbose:
            print("Saved.")

    elif filename.endswith('.csv'):
        pd.DataFrame(pointcloud).to_csv(filename, header=None, index=None, sep=' ')
        print("Saved to:", filename)

    elif filename.endswith('.ply'):

        if not isinstance(pointcloud, pd.DataFrame):
            cols = list(set(cols))
            pointcloud = pd.DataFrame(pointcloud, columns=cols)
        
        ply_io.write_ply(filename, pointcloud[cols])
        print("Saved to:", filename)

def low_resolution_hack_mode(point_cloud, num_iterations, min_spacing, num_procs):
    print('Using low resolution point cloud hack mode...')
    print('Original point cloud shape:', point_cloud.shape)
    point_cloud_original = deepcopy(point_cloud)
    for i in range(num_iterations):
        duplicated = deepcopy(point_cloud_original)

        duplicated[:, :3] = duplicated[:, :3] + np.hstack(
                (np.random.normal(-0.025, 0.025, size=(duplicated.shape[0], 1)),
                 np.random.normal(-0.025, 0.025, size=(duplicated.shape[0], 1)),
                 np.random.normal(-0.025, 0.025, size=(duplicated.shape[0], 1))))
        point_cloud = np.vstack((point_cloud, duplicated))
        point_cloud = subsample_point_cloud(point_cloud, min_spacing, num_procs)
    print('Hacked point cloud shape:', point_cloud.shape)
    return point_cloud

def make_dtm(params):
    
    """ 
    This function will generate a Digital Terrain Model (dtm) based on the terrain labelled points.
    """

    if params.verbose: print("Making dtm...")

    params.grid_resolution = .5

    ### voxelise, identify lowest points and create DTM
    params.pc = voxelise(params.pc, params.grid_resolution, z=False)
    VX_map = params.pc.loc[~params.pc.VX.duplicated()][['xx', 'yy', 'VX']]
    ground = params.pc.loc[params.pc.label == params.terrain_class] 
    ground.loc[:, 'zmin'] = ground.groupby('VX').z.transform(np.median)
    ground = ground.loc[ground.z == ground.zmin]
    ground = ground.loc[~ground.VX.duplicated()]

    X, Y = np.meshgrid(np.arange(params.pc.xx.min(), params.pc.xx.max() + params.grid_resolution, params.grid_resolution),
                       np.arange(params.pc.yy.min(), params.pc.yy.max() + params.grid_resolution, params.grid_resolution))

    ground_arr = pd.DataFrame(data=np.vstack([X.flatten(), Y.flatten()]).T, columns=['xx', 'yy']) 
    ground_arr = pd.merge(ground_arr, VX_map, on=['xx', 'yy'], how='outer') # map VX to ground_arr
    ground_arr = pd.merge(ground[['z', 'VX']], ground_arr, how='right', on=['VX']) # map z to ground_arr
    ground_arr.sort_values(['xx', 'yy'], inplace=True)
    
    # loop over incresing size of window until no cell are nan
    ground_arr.loc[:, 'ZZ'] = np.nan
    size = 3 
    while np.any(np.isnan(ground_arr.ZZ)):
        ground_arr.loc[:, 'ZZ'] = ndimage.generic_filter(ground_arr.z.values.reshape(*X.shape), # create raster, 
                                                         lambda z: np.nanmedian(z), size=size).flatten()
        size += 2

    ground_arr.to_csv(os.path.join(params.odir, 'dem.csv'), index=False)

    # apply to all points   
    MAP = ground_arr.set_index('VX').ZZ.to_dict()
    params.pc.loc[:, 'nz'] = params.pc.z - params.pc.VX.map(MAP)  
    
    return params

