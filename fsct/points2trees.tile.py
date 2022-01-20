import os
import time
import threading
import itertools
import multiprocessing
import argparse
import string

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import networkx as nx

from fsct.tools import *
from fsct.fit_cylinders import RANSAC_helper

import warnings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

def generate_path(samples, skeleton):
    
    if not isinstance(skeleton.dbh_node.dtype, bool):
        skeleton.dbh_node = skeleton.dbh_node.astype(bool)

    if len(skeleton.loc[skeleton.dbh_node == 1]) > 0:
        ### build graph ###
        # compute nearest neighbours for each vertex in cluster convex hull
        num_neighbours = 50
        nn = NearestNeighbors(n_neighbors=num_neighbours).fit(samples[['x', 'y', 'z']])
        distances, indices = nn.kneighbors()    
        from_to_all = pd.DataFrame(np.vstack([np.repeat(samples.clstr.values, num_neighbours), 
                                          samples.iloc[indices.ravel()].clstr.values, 
                                          distances.ravel()]).T, 
                                   columns=['source', 'target', 'length'])

        # remove X-X connections
        from_to_all = from_to_all.loc[from_to_all.target != from_to_all.source]
        # and build edge database where edges with min distance between clusters persist
        edges = from_to_all.groupby(['source', 'target']).length.min().reset_index()
        # remove edges that are likely leaps between trees
        edges = edges.loc[edges.length <= .2]
        #edges = edges.loc[edges.length <= .5]
        
        stems_in_tile = list(skeleton.loc[(skeleton.dbh_node)].clstr)
        # removes isolated dbh points i.e. > edge.length
        stems_in_tile = [s for s in stems_in_tile if s in edges.source.values] 

        # compute graph
        G = nx.from_pandas_edgelist(edges, edge_attr=['length'])
        distance, shortest_path = nx.multi_source_dijkstra(G, 
                                                           sources=stems_in_tile,
                                                           weight='length')

        paths = pd.DataFrame(index=distance.keys(), data=distance.values(), columns=['distance'])
        paths.loc[:, 'base'] = params.not_base
        for p in paths.index: paths.loc[p, 'base'] = shortest_path[p][0]
        paths.reset_index(inplace=True)
        
        return paths
        
    else: 
        return pd.DataFrame(columns=['index', 'distance', 'base'])

    
def unique_cluster_id(clstr):
    
    gen_tclstr = lambda: ''.join(np.random.choice(list(string.ascii_letters), size=8))
    return {c:gen_tclstr() for c in clstr}


def cube(pc):
    
    if len(pc) > 5:
        vertices = ConvexHull(pc[['x', 'y', 'z']]).vertices
        idx = np.random.choice(vertices, size=len(vertices), replace=False)
        return pc.loc[pc.index[idx]]
    else:
        return pc 
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--tile', '-t', type=str, default='', required=True, help='fsct directory')
    parser.add_argument('--odir', '-o', type=str, required=True, help='output directory')
    parser.add_argument('--tindex', type=str, required=True, help='path to tile index')
    parser.add_argument('--n-tiles', default=3, type=int, help='enlarges the number of tiles i.e. 3x3 or tiles or 5 x 5 tiles')
    parser.add_argument('--n-prcs', default=5, type=int, help='number of cores')
    parser.add_argument('--slice-height', default=1.4, type=float, help='slice height for identifying stems')
    parser.add_argument('--slice-thickness', default=.2, type=float, help='slice thickness for identifying stems')
    parser.add_argument('--add-leaves', action='store_true', help='add leaf points')
    parser.add_argument('--pandarallel', action='store_true', help='use pandarallel')
    parser.add_argument('--verbose', action='store_true', help='print something')
    params = parser.parse_args()

    if params.tile == '':
        raise Exception('specify --tile')  
        
    if params.pandarallel:
        try:
            from pandarallel import pandarallel
            pandarallel.initialize(progress_bar=True if params.verbose else False)
        except:
            print('--- pandarallel not installed ---')
            params.pandarallel = False

    params.not_base = -1  
    xyz = ['x', 'y', 'z'] # shorthand

    params.dir, params.fn = os.path.split(params.tile)
    params.n = int(params.fn.split('.')[0])
#     params.tmp = glob.glob(os.path.join(params.dir, f'{params.n:03}.*.tmp'))[0]

    params.pc = ply_io.read_ply(params.tile)
    params.pc.loc[:, 'buffer'] = False
    params.pc.loc[:, 'fn'] = params.n

    bbox = {}
    bbox['xmin'], bbox['xmax'] = params.pc.x.min(), params.pc.x.max()
    bbox['ymin'], bbox['ymax'] = params.pc.y.min(), params.pc.y.max()
    bbox = dict2class(bbox)

    # neighbouring tiles to process
    params.ti = pd.read_csv(params.tindex, 
                            sep=' ', 
                            names=['tile', 'x', 'y'])
    n_tiles = NearestNeighbors(n_neighbors=len(params.ti)).fit(params.ti[['x', 'y']])
    distance, indices = n_tiles.kneighbors(params.ti.loc[params.ti.tile == params.n][['x', 'y']])
    # todo: this could be made smarter e.g. using distance
    buffer_tiles = params.ti.loc[indices[0][1:params.n_tiles**2]]['tile'].values

    for i, t in tqdm(enumerate(buffer_tiles),
                     total=len(buffer_tiles),
                     desc='read in neighbouring tiles', 
                     disable=False if params.verbose else True):

        try:
            b_tile = glob.glob(os.path.join(params.dir, f'{t:03}*.ply'))[0]
            tmp = ply_io.read_ply(b_tile)
            tmp = tmp.loc[(tmp.x.between(bbox.xmin - 10, bbox.xmax + 10)) & 
                          (tmp.y.between(bbox.ymin - 10, bbox.ymax + 10))]
            if len(tmp) == 0: continue
            tmp.loc[:, 'buffer'] = True
            tmp.loc[:, 'fn'] = t

            params.pc = params.pc.append(tmp, ignore_index=True)
        except:
            path = os.path.join(params.dir, f'{t:03}*.ply')
            raise Exception(f'tile {path} not available')
        
    if 'nz' in params.pc.columns: params.pc.rename(columns={'nz':'n_z'}, inplace=True)

    ### generate skeleton points
    if params.verbose: print('\n----- skeletonisation started -----')

    # extract stems points and slice slice
    stem_pc = params.pc.loc[params.pc.label == 3]

    # slice stem_pc
    stem_pc.loc[:, 'slice'] = (stem_pc.z // params.slice_thickness).astype(int) * params.slice_thickness
    stem_pc.loc[:, 'n_slice'] = (stem_pc.n_z // params.slice_thickness).astype(int)
    params.slice_height = int(params.slice_height / params.slice_thickness)

    # cluster within height slices
    stem_pc.loc[:, 'clstr'] = -1
    label_offset = 0

    for slice_height in tqdm(np.sort(stem_pc.n_slice.unique()), 
                             disable=False if params.verbose else True,
                             desc='slice data vertically and clustering'):

        new_slice = stem_pc.loc[stem_pc.n_slice == slice_height]

        if len(new_slice) > 200:
            dbscan = DBSCAN(eps=.1, min_samples=20).fit(new_slice[xyz])
            new_slice.loc[:, 'clstr'] = dbscan.labels_
            new_slice.loc[new_slice.clstr > -1, 'clstr'] += label_offset
            stem_pc.loc[new_slice.index, 'clstr'] = new_slice.clstr
            label_offset = stem_pc.clstr.max() + 1
    
    # group skeleton points
    grouped = stem_pc.loc[stem_pc.clstr != -1].groupby('clstr')
    if params.verbose: print('fitting convex hulls to clusters')
    if params.pandarallel:
        chull = grouped.parallel_apply(cube) # parallel_apply only works witn pd < 1.3
    else:
        chull = grouped.apply(cube) # don't think works with Jasmin or parallel_apply only works witn pd < 1.3
    chull = chull.reset_index(drop=True) 
    
    ### identify possible stems ###
    if params.verbose: print('identifying stems...')
    skeleton = grouped[xyz + ['n_z', 'n_slice', 'slice']].median().reset_index()
    skeleton.loc[:, 'dbh_node'] = False

    dbh_nodes = skeleton.loc[skeleton.n_slice == params.slice_height].clstr
    dbh_slice = stem_pc.loc[stem_pc.clstr.isin(dbh_nodes)]

    if len(dbh_slice) > 0:

        # remove noise from dbh slice
        nn = NearestNeighbors(n_neighbors=10).fit(dbh_slice[xyz])
        distances, indices = nn.kneighbors()
        dbh_slice.loc[:, 'nn'] = distances[:, 1:].mean(axis=1)
        dbh_slice = dbh_slice.loc[dbh_slice.nn < dbh_slice.nn.quantile(q=.9)]

        # run dbscan over dbh_slice
        dbscan = DBSCAN(eps=.1, min_samples=75).fit(dbh_slice[['x', 'y']])
        dbh_slice.loc[:, 'clstr_db'] = dbscan.labels_
        dbh_slice = dbh_slice.loc[dbh_slice.clstr_db > -1]

        if len(dbh_slice) > 10: 

            # ransac cylinder fitting
            if params.verbose: print('fitting cylinders to possible stems...')
            if params.pandarallel:
                dbh_cylinder = dbh_slice.groupby('clstr').parallel_apply(RANSAC_helper, 1000, 20).to_dict()
            else:
                dbh_cylinder = dbh_slice.groupby('clstr').apply(RANSAC_helper, 1000, 10).to_dict()
            dbh_cylinder = pd.DataFrame(dbh_cylinder).T
            dbh_cylinder.columns = ['radius', 'centre', 'CV', 'cnt']
            dbh_cylinder.loc[:, ['x', 'y', 'z']] = [[*row.centre] for row in dbh_cylinder.itertuples()]
            dbh_cylinder = dbh_cylinder.drop(columns=['centre']).astype(float)

            # identify clusters where cylinder CV <= .75 and label as nodes
            skeleton.loc[skeleton.clstr.isin(dbh_cylinder.loc[(dbh_cylinder.CV <= .4) &
                                                              (dbh_cylinder.radius > .05) &
                                                              (dbh_cylinder.cnt > 200)].index.values), 'dbh_node'] = True

    in_tile_stem_nodes = skeleton.loc[(skeleton.dbh_node) & 
                                      (skeleton.x.between(bbox.xmin, bbox.xmax)) &
                                      (skeleton.y.between(bbox.ymin, bbox.ymax))].clstr
    
    # generates paths through all stem points
    if params.verbose: print('generating graph, this may take a while...')
    all_paths = generate_path(chull, skeleton)

    # removes paths that are longer for same clstr
    all_paths = all_paths.sort_values(['index', 'distance'])
    all_paths = all_paths.loc[~all_paths['index'].duplicated()] 

    if params.verbose: print('merging skeleton points with graph')
    stems = pd.merge(skeleton, all_paths, left_on='clstr', right_on='index', how='left')

    # give a unique colour to each tree (helps with visualising)
    stems.drop(columns=[c for c in stems.columns if c.startswith('red') or 
                                                    c.startswith('green') or 
                                                    c.startswith('blue')], inplace=True)
    unique_stems = stems.base.unique()
    RGB = pd.DataFrame(data=np.vstack([unique_stems, 
                                       np.random.randint(0, 255, size=(3, len(unique_stems)))]).T, 
                       columns=['base', 'red', 'green', 'blue'])
    RGB.loc[RGB.base == params.not_base, :] = [np.nan, 211, 211, 211] # color unassigned points grey
    stems = pd.merge(stems, RGB, on='base', how='right')

    # read in all "stems" tiles and assign all stem points to a tree
    trees = pd.merge(stem_pc, stems[['clstr', 'base', 'red', 'green', 'blue']], on='clstr')
    trees.loc[:, 'cnt'] = trees.groupby('base').base.transform('count')
    trees = trees.loc[trees.cnt > 10000]
    in_tile_stem_nodes = trees.loc[trees.base.isin(in_tile_stem_nodes)].base.unique()

    # write out all trees
    params.base_I, I = {}, 0
    for i, b in tqdm(enumerate(in_tile_stem_nodes), 
                     total=len(in_tile_stem_nodes), 
                     desc='writing stems to file', 
                     disable=False if params.verbose else True):
        if b == params.not_base: continue
        ply_io.write_ply(os.path.join(params.odir, f'{params.n:03}_T{I}.leafoff.ply'), trees.loc[trees.base == b])
        params.base_I[b] = I
        I += 1    

    if params.add_leaves:
        
        if params.verbose: print('adding leaves to stems, this may take a while...')
        
        # link stem number to clstr
        stem2tlsctr = stems[['clstr', 'base']].loc[stems.base != params.not_base].set_index('clstr').to_dict()['base']
        chull.loc[:, 'stem'] = chull.clstr.map(stem2tlsctr)
        chull = chull.loc[[False if np.isnan(s) else True for s in chull.stem]]
        chull.loc[:, 'label'] = 2

        # process leaf points
        lvs = params.pc.loc[(params.pc.label == 1) & (params.pc.n_z >= 2)].copy()
        lvs = voxelise(lvs, length=.2)
        lvs_median = lvs.groupby('VX')[xyz].median().reset_index()
        lvs_median.loc[:, 'label'] = 1

        # and combine leaves and wood
        branch_and_leaves = lvs_median.append(chull[['x', 'y', 'z', 'label', 'stem']])
        branch_and_leaves.reset_index(inplace=True, drop=True)

        # find neighbouring branch and leaf points - used as entry points
        nn = NearestNeighbors(n_neighbors=2).fit(branch_and_leaves[xyz])
        distances, indices = nn.kneighbors()   
        closest_point_to_leaf = indices[:len(lvs_median), :].flatten() # only leaf points
        idx = np.isin(closest_point_to_leaf, branch_and_leaves.loc[branch_and_leaves.label == 2].index)
        close_branch_points = closest_point_to_leaf[idx] # points where the branch is closest

        # remove all branch points that are not close to leaves
        idx = np.hstack([branch_and_leaves.iloc[:len(lvs_median)].index.values, close_branch_points])
        bal = branch_and_leaves.loc[branch_and_leaves.index.isin(np.unique(idx))]

        # compute nearest neighbours for each vertex
        num_neighbours = 50
        nn = NearestNeighbors(n_neighbors=num_neighbours).fit(bal[xyz])
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

        # compute graph
        G = nx.from_pandas_edgelist(edges, edge_attr=['length'])
        cbp = np.unique(close_branch_points[np.isin(close_branch_points, edges.source.values)])

        distance, shortest_path = nx.multi_source_dijkstra(G, 
                                                           sources=list(cbp),
                                                           weight='length')

        # graph to df
        paths = pd.DataFrame(index=distance.keys(), data=distance.values(), columns=['distance'])
        paths = paths.reset_index().rename(columns={'index':'s_index'})
        paths.loc[:, 't_index'] = paths.s_index.apply(lambda ix: shortest_path[ix][0]) 
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
        lvs = pd.merge(lvs, RGB, left_on='stem_', right_on='base', )
        lvs[['red', 'green', 'blue']] = (lvs[['red', 'green', 'blue']] * 1.2).astype(int)
        lvs.loc[lvs.red > 255, 'red'] = 255
        lvs.loc[lvs.blue > 255, 'blue'] = 255
        lvs.loc[lvs.green > 255, 'green'] = 255

        # and save
        for lv in lvs.loc[lvs.base.isin(in_tile_stem_nodes)].base.unique():

            I = params.base_I[lv]

            stem = ply_io.read_ply(os.path.join(params.odir, f'{params.n:03}_T{I}.leafoff.ply'))
            stem.loc[:, 'wood'] = 1

            l2a = lvs.loc[lvs.base == lv]
            l2a.loc[:, 'wood'] = 0
            stem = stem.append(l2a[['x', 'y', 'z', 'red', 'green', 'blue', 'base', 'wood', 'distance']])

            stem = stem.loc[~stem.duplicated()]
            ply_io.write_ply(os.path.join(params.odir, f'{params.n:03}_T{I}.leafon.ply'), stem)
