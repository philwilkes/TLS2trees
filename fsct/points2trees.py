import os
import time
import threading
import itertools
import multiprocessing
import argparse

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.neighbors import NearestNeighbors
import networkx as nx

from fsct.tools import *

pd.options.mode.chained_assignment = None


def generate_path(bx, by, params):
    
    samples = params.all_samples.loc[(params.all_samples.x.between(bx - params.overlap, 
                                                               bx + params.box_length + params.overlap)) &
                                   (params.all_samples.y.between(by - params.overlap, 
                                                               by + params.box_length + params.overlap))]

    stem_skeleton = params.all_skeleton.loc[(params.all_skeleton.x.between(bx - params.overlap, 
                                                                       bx + params.box_length + params.overlap)) &
                                          (params.all_skeleton.y.between(by - params.overlap, 
                                                                       by + params.box_length + params.overlap))]

    if not isinstance(stem_skeleton.dbh_node.dtype, bool):
        stem_skeleton.dbh_node = stem_skeleton.dbh_node.astype(bool)

    if len(stem_skeleton.loc[stem_skeleton.dbh_node == 1]) > 0:
        ### build graph ###
        # compute nearest neighbours for each vertex in cluster convex hull
        num_neighbours = 200
        nn = NearestNeighbors(n_neighbors=num_neighbours).fit(samples[['x', 'y', 'z']])
        distances, indices = nn.kneighbors()    
        from_to_all = pd.DataFrame(np.vstack([np.repeat(samples.t_clstr.values, num_neighbours), 
                                          samples.iloc[indices.ravel()].t_clstr.values, 
                                          distances.ravel()]).T, 
                                   columns=['source', 'target', 'length'])

        # remove X-X connections
        from_to_all = from_to_all.loc[from_to_all.target != from_to_all.source]
        # and build edge database where edges with min distance between clusters persist
        edges = from_to_all.groupby(['source', 'target']).length.min().reset_index()
        # remove edges that are likely leaps between trees
#         edges = edges.loc[edges.length <= .2]
        edges = edges.loc[edges.length <= .5]

        # compute graph
        G = nx.from_pandas_edgelist(edges, edge_attr=['length'])
        distance, shortest_path = nx.multi_source_dijkstra(G, 
                                                           sources=list(stem_skeleton.loc[stem_skeleton.dbh_node].t_clstr),
                                                           weight='length')

        paths = pd.DataFrame(index=distance.keys(), data=distance.values(), columns=['distance'])
        paths.loc[:, 'base'] = np.nan
        for p in paths.index: paths.loc[p, 'base'] = shortest_path[p][0]
        paths.reset_index(inplace=True)
        
        return paths
        
    else: 
        return pd.DataFrame(columns=['index', 'distance', 'base'])
    

def add_leaves(fn, params):

    ply = ply_io.read_ply(fn)
    lvs = ply.loc[(ply.label == 1) & (ply.nz >= 2)]
    lvs = lvs.drop(columns=['distance'] if 'distance' in lvs.columns else[])

    # voxelise
    lvs = voxelise(lvs, length=.2)
    lvs_median = lvs.groupby('VX')[['x', 'y', 'z']].median().reset_index()
    lvs_median.loc[:, 'label'] = 1

    # subsample stem pc
    sub_samples = params.all_samples.loc[(params.all_samples.x.between(lvs_median.x.min() - 5, 
                                                                       lvs_median.x.max() + 5)) &
                                         (params.all_samples.y.between(lvs_median.y.min() - 5, 
                                                                       lvs_median.y.max() + 5))]
    sub_samples = sub_samples.loc[~np.isnan(sub_samples.stem)]
    sub_samples.loc[:, 'label'] = 2

    # and combine leaves and wood
    branch_and_leaves = lvs_median.append(sub_samples[['x', 'y', 'z', 'label', 'stem']])
    branch_and_leaves.reset_index(inplace=True, drop=True)

    # find neighbouring branch and leaf points - used as entry points
    nn = NearestNeighbors(n_neighbors=2).fit(branch_and_leaves[['x', 'y', 'z']])
    distances, indices = nn.kneighbors()    
    closest_point_to_leaf = indices[:len(lvs_median), :].flatten() # only leaf points
    idx = np.isin(closest_point_to_leaf, branch_and_leaves.loc[branch_and_leaves.label == 2].index)
    close_branch_points = closest_point_to_leaf[idx] # points where the branch is closest

    # remove all branch points that are not close to leaves
    idx = np.hstack([branch_and_leaves.iloc[:len(lvs_median)].index.values, close_branch_points])
    bal = branch_and_leaves.loc[branch_and_leaves.index.isin(np.unique(idx))]

    # compute nearest neighbours for each vertex in cluster convex hull
    num_neighbours = 200
    nn = NearestNeighbors(n_neighbors=num_neighbours).fit(bal[['x', 'y', 'z']])
    distances, indices = nn.kneighbors()    


    from_to_all = pd.DataFrame(np.vstack([np.repeat(bal.index.values, num_neighbours), 
                                          bal.iloc[indices.ravel()].index.values, 
                                          distances.ravel()]).T, 
                               columns=['source', 'target', 'length'])

    # remove X-X connections
    from_to_all = from_to_all.loc[from_to_all.target != from_to_all.source]
    # and build edge database where edges with min distance between clusters persist
    edges = from_to_all.groupby(['source', 'target']).length.min().reset_index()
    # remove edges that are likely leaps between trees
    #         edges = edges.loc[edges.length <= .2]
    # edges = edges.loc[edges.length <= 1]
    edges = edges.loc[edges.length <= .5]

    print(edges)

    # compute graph
    G = nx.from_pandas_edgelist(edges, edge_attr=['length'])
    cbp = np.unique(close_branch_points[np.isin(close_branch_points, edges.source.values)])

    distance, shortest_path = nx.multi_source_dijkstra(G, 
                                                       sources=list(cbp),
                                                       weight='length')

    # graph to df
    paths = pd.DataFrame(index=distance.keys(), data=distance.values(), columns=['distance'])
    for p in paths.index: paths.loc[p, 't_index'] = shortest_path[p][0]
    paths.reset_index(inplace=True)
    paths = paths.loc[paths.distance > 0]

    # linking indexs to stem number
    top2stem = branch_and_leaves.loc[branch_and_leaves.label == 2]['stem'].to_dict()
    paths.loc[:, 'stem_'] = paths.t_index.map(top2stem)
    #     paths.loc[:, 'stem'] = paths.stem_.map(base2i)

    # linking index to VX number
    index2VX = branch_and_leaves.loc[branch_and_leaves.label == 1]['VX'].to_dict()
    paths.loc[:, 'VX'] = paths['index'].map(index2VX)

    # linking VX to stem
    lvs = pd.merge(lvs, paths[['VX', 'stem_', 'distance']], on='VX')

    # colour the same as stem
    lvs = pd.merge(lvs, params.RGB, left_on='stem_', right_on='base', )
    lvs[['red', 'green', 'blue']] = (lvs[['red', 'green', 'blue']] * 1.2).astype(int)
    lvs.loc[lvs.red > 255, 'red'] = 255
    lvs.loc[lvs.blue > 255, 'blue'] = 255
    lvs.loc[lvs.green > 255, 'green'] = 255

    # and save
    for lv in lvs.base.unique():

        if os.path.isfile(os.path.join(params.odir, f'T{int(lv)}.leafon.ply')):
            with params.Lock: stem = ply_io.read_ply(os.path.join(params.odir, f'T{int(lv)}.leafon.ply'))
        else:
            with params.Lock: stem = ply_io.read_ply(os.path.join(params.odir, f'T{int(lv)}.leafoff.ply'))
            stem.loc[:, 'wood'] = 1

        l2a = lvs.loc[lvs.base == lv]
        l2a.loc[:, 'wood'] = 0
        stem = stem.append(l2a[['x', 'y', 'z', 'red', 'green', 'blue', 'base', 'wood', 'distance']])

        stem = stem.loc[~stem.duplicated()]
        with params.Lock: ply_io.write_ply(os.path.join(params.odir, f'T{int(lv)}.leafon.ply'), stem)

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--idir', '-i', type=str, default='', required=True, help='fsct directory')
    parser.add_argument('--odir', '-o', type=str, required=True, help='output directory')
    parser.add_argument('--box-length', default=10, type=float, help='processing grid size')
    parser.add_argument('--overlap', default=5, type=float, help='grid overlap')
    parser.add_argument('--n-prcs', default=5, type=int, help='number of cores')
    parser.add_argument('--add-leaves', action='store_true', help='add leaf points')
    parser.add_argument('--verbose', action='store_true', help='print something')
    params = parser.parse_args()
    
    if params.idir == '':
        raise Exception('specify either --idir')    
    
    params.all_samples = pd.DataFrame()
    params.all_skeleton = pd.DataFrame()
    sample = 0

    # read in convex hull nodes and stem skeletons
    # add an additioal field with a unique cluster id
    L = glob.glob('*/*.samples.ply')
    for i, t in tqdm(enumerate(L),
                     total=len(L),
                     desc='read in skeleton and convex hull', 
                     disable=False if params.verbose else True):
        try:
            tmp = ply_io.read_ply(t)
            tmp.loc[:, 'fn'] = int(t.split('.')[0])
            tmp.loc[:, 't_clstr'] = tmp.clstr + sample
            sample += len(np.unique(tmp.clstr))
            params.all_samples = params.all_samples.append(tmp, ignore_index=True)

            D = tmp[['clstr', 't_clstr']].loc[~tmp.clstr.duplicated()].set_index('clstr').to_dict()
            if os.path.isfile(t.replace('samples', 'stem_skeleton')):
                skel = ply_io.read_ply(t.replace('samples', 'stem_skeleton'))
                skel.loc[:, 'fn'] = int(t.split('.')[0])
                skel.dbh_node = skel.dbh_node.astype(bool)
                skel.loc[:, 't_clstr'] = skel.clstr.map(D['t_clstr'])
                params.all_skeleton = params.all_skeleton.append(skel, ignore_index=True)
        except:
            raise Exception(t)


    # reset indexes
    params.all_samples.reset_index(inplace=True, drop=True)
    params.all_skeleton.reset_index(inplace=True, drop=True)
    
    # create grid to iterate over
    params.bbox = compute_bbox(params.all_samples)
    x_cnr = np.arange(params.bbox.xmin - params.overlap, params.bbox.xmax + params.overlap, params.box_length)
    y_cnr = np.arange(params.bbox.ymin - params.overlap, params.bbox.ymax + params.overlap, params.box_length)

    # multiprocessing
    Pool = multiprocessing.Pool(params.n_prcs)
    m = multiprocessing.Manager()
    params.Lock = m.Lock()

    all_paths = pd.DataFrame()
    all_paths = all_paths.append(Pool.starmap_async(generate_path, tqdm([(bx, by, params) for bx, by in 
                                                                         itertools.product(x_cnr, y_cnr)], 
                                                                         desc='generating plot wide graph', 
                                                                         disable=False if params.verbose else True)).get())
    Pool.close()
    Pool.join()

    # removes paths that are longer for same clstr
    all_paths = all_paths.sort_values(['index', 'distance'])
    all_paths = all_paths.loc[~all_paths['index'].duplicated()] 
    
    if params.verbose: print('merging skeleton points with graph')
    stems = pd.merge(params.all_skeleton, all_paths, left_on='t_clstr', right_on='index', how='left')
    
    # give a unique colour to each tree (helps with visualising)
    stems.drop(columns=[c for c in stems.columns if c.startswith('red') or 
                                                    c.startswith('green') or 
                                                    c.startswith('blue')], inplace=True)
    unique_stems = stems.base.unique()
    params.RGB = pd.DataFrame(data=np.vstack([unique_stems, 
                                              np.random.randint(0, 255, size=(3, len(unique_stems)))]).T, 
                              columns=['base', 'red', 'green', 'blue'])
    params.RGB.loc[np.isnan(params.RGB.base), :] = [np.nan, 211, 211, 211] # color unassigned points grey
    stems = pd.merge(stems, params.RGB, on='base', how='right')
    
    # read in all "stems" tiles and assign all stem points to a tree
    trees = pd.DataFrame(columns=['x', 'y', 'z', 'clstr', 'base'])

    L = glob.glob('*/???.downsample.stems.ply')
    for fn in tqdm(L, 
                   total=len(L), 
                   desc='assigning points to trees', 
                   disable=False if params.verbose else True):
        ply = ply_io.read_ply(fn)
        N = int(os.path.split(fn)[1].split('.')[0])
        PLY = pd.merge(ply, 
                       stems.loc[stems.fn == N][['clstr', 'base', 'red', 'green', 'blue']], on='clstr')
        trees = trees.append(PLY, ignore_index=True)
    
    # write out all trees
    for i, b in tqdm(enumerate(trees.base.unique()), 
                     total=len(trees.base.unique()), 
                     desc='writing stems to file', 
                     disable=False if params.verbose else True):
        if np.isnan(b): continue
        ply_io.write_ply(os.path.join(params.odir, f'T{int(b)}.leafoff.ply'), trees.loc[trees.base == b])
    
    # link stem number to t_clstr
    stem2tlsctr = stems[['t_clstr', 'base']].loc[~np.isnan(stems.base)].set_index('t_clstr').to_dict()['base']
    params.all_samples.loc[:, 'stem'] = params.all_samples.t_clstr.map(stem2tlsctr)
        
    if params.add_leaves:
        
        Pool = multiprocessing.Pool(params.n_prcs)
        m = multiprocessing.Manager()
        params.Lock = m.Lock()

        #add_leaves('129.downsample.segmented.ply', params)

        Pool.starmap_async(add_leaves, 
                           tqdm([(fn, params) for fn in glob.glob(os.path.join(params.idir, '*.segmented.ply'))], 
                                desc='adding leaves',
                                disable=False if params.verbose else True))
        Pool.close()
        Pool.join()
