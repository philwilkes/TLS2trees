import itertools
import os
import threading
import time

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from fsct.tools import *


def save_pts(params, I, bx, by, bz):

    pc = params.pc.loc[(params.pc.x.between(bx, bx + params.box_dims[0])) &
                       (params.pc.y.between(by, by + params.box_dims[0])) &
                       (params.pc.z.between(bz, bz + params.box_dims[0]))]

    if len(pc) > params.min_points_per_box:

        if len(pc) > params.max_points_per_box:
            pc = pc.sample(n=params.max_points_per_box)

        np.save(os.path.join(params.working_dir, f'{I:07}'), pc[['x', 'y', 'z']].values)

def Preprocessing(params):
    
    if params.verbose: print('\n----- preprocessing started -----')
    start_time = time.time()
    params.point_cloud = os.path.abspath(params.point_cloud)
    params.directory, params.filename = os.path.split(params.point_cloud)
    params.basename = os.path.splitext(params.filename)[0]
    params.tile = params.filename.split('.')[0]
    params.input_format = os.path.splitext(params.point_cloud)[1]
    if not isinstance(params.tile, int) and params.tile.isdigit(): params.tile = int(params.tile)

    # create directory structure
    params = make_folder_structure(params)

    # read in pc
    params.pc, params.additional_headers = load_file(filename=params.point_cloud,
                                                     additional_headers=True,
                                                     verbose=params.verbose)

    # compute plot centre, global shift and bounding box
    params.plot_centre = compute_plot_centre(params.pc)
    params.global_shift = params.pc[['x', 'y', 'z']].mean()
    params.bbox = compute_bbox(params.pc[['x', 'y', 'z']])
    
    # buffer 
    params.pc.loc[:, 'buffer'] = False # might not be needed
    if params.buffer > 0:
        
        tile_index = pd.read_csv(params.tile_index, sep=' ', names=['fname', 'x', 'y'])

        # locate 8 nearest tiles
        nn = NearestNeighbors(n_neighbors=9).fit(tile_index[['x', 'y']])
        distance, neighbours = nn.kneighbors(tile_index.loc[tile_index.fname == params.tile][['x', 'y']], 
                                   return_distance=True)
        neighbours = neighbours[np.where(distance <= params.max_distance_between_tiles)]

        # read in tiles
        buffer = pd.DataFrame()
        for tile in tqdm(tile_index.loc[neighbours[1:]].itertuples(), 
                         total=len(neighbours)-1,
                         desc='buffering tile with neighbouring points',
                         disable=False if params.verbose else True):
            fname = glob.glob(os.path.join(params.directory, f'{tile.fname}*{params.input_format}'))
            if len(fname) > 0: buffer = buffer.append(load_file(os.path.join(params.directory, fname[0])))

        # select desired points
        buffer = buffer.loc[(buffer.x.between(params.pc.x.min() - params.buffer, 
                                              params.pc.x.max() + params.buffer)) &
                            (buffer.y.between(params.pc.y.min() - params.buffer, 
                                              params.pc.y.max() + params.buffer))]

        buffer.loc[:, 'buffer'] = True
        if params.verbose: print(f'buffer adds an additional {len(buffer)} points')
        params.pc = params.pc.append(buffer)

    if params.subsample: # subsample if specified
        if params.verbose: print('downsampling to: %s m' % params.subsampling_min_spacing)
        params.pc = downsample(params.pc, params.subsampling_min_spacing, 
                             accurate=False, keep_points=False)

    # apply global shift
    if params.verbose: print('global shift:', params.global_shift.values)
    params.pc[['x', 'y', 'z']] = params.pc[['x', 'y', 'z']] - params.global_shift
	
    params.pc.reset_index(inplace=True)
    params.pc.loc[:, 'pid'] = params.pc.index

    # generate bounding boxes
    xmin, xmax = np.floor(params.pc.x.min()), np.ceil(params.pc.x.max())
    ymin, ymax = np.floor(params.pc.y.min()), np.ceil(params.pc.y.max())
    zmin, zmax = np.floor(params.pc.z.min()), np.ceil(params.pc.z.max())

    box_overlap = params.box_dims[0] * params.box_overlap[0]

    x_cnr = np.arange(xmin - box_overlap, xmax + box_overlap, box_overlap)
    y_cnr = np.arange(ymin - box_overlap, ymax + box_overlap, box_overlap)
    z_cnr = np.arange(zmin - box_overlap, zmax + box_overlap, box_overlap)

    # multithread segmenting points into boxes and save
    threads = []
    for i, (bx, by, bz) in enumerate(itertools.product(x_cnr, y_cnr, z_cnr)):
        threads.append(threading.Thread(target=save_pts, args=(params, i, bx, by, bz)))

    for x in tqdm(threads, 
                  desc='generating data blocks',
                  disable=False if params.verbose else True):
        x.start()

    for x in threads:
        x.join()

    if params.verbose: print("Preprocessing done in {} seconds\n".format(time.time() - start_time))
    
    return params
