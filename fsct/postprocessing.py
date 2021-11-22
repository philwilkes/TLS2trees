import os
import time

import numpy as np
import pandas as pd
from scipy import ndimage

from fsct.tools import *

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def PostProcessing(params):

    post_processing_time_start = time.time()

    # calcualte height abouve ground and add ground normalised field
    params = make_dtm(params)
    params.pc.loc[params.pc.nz <= params.ground_height_threshold, 'label'] = params.terrain_class
    
#     # export point classes
#     for C in ['terrain_class', 'vegetation_class', 'cwd_class', 'stem_class']:
    
#         class_id = getattr(params, C)
#         op = os.path.join(params.odir, f'{params.basename}.{C.split("_")[0]}.{params.output_fmt}')
#         save_file(op, 
#                   params.pc.loc[(~params.pc.buffer) & (params.pc.label == class_id)], 
#                   additional_fields=['label', 'nz'])

    return params

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