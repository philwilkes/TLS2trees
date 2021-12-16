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
import networkx as nx

from fsct.tools import *

pd.options.mode.chained_assignment = None

def generate_path(samples, stem_skeleton):
    
    if not isinstance(stem_skeleton.dbh_node.dtype, bool):
        stem_skeleton.dbh_node = stem_skeleton.dbh_node.astype(bool)

    if len(stem_skeleton.loc[stem_skeleton.dbh_node == 1]) > 0:
        ### build graph ###
        # compute nearest neighbours for each vertex in cluster convex hull
        num_neighbours = 50
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
        edges = edges.loc[edges.length <= .2]
        #edges = edges.loc[edges.length <= .5]
        
        stems_in_tile = list(stem_skeleton.loc[(stem_skeleton.dbh_node)].t_clstr)

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
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--tile', '-t', type=str, default='', required=True, help='fsct directory')
    parser.add_argument('--odir', '-o', type=str, required=True, help='output directory')
    parser.add_argument('--tindex', type=str, required=True, help='path to tile index')
    parser.add_argument('--n-tiles', default=3, type=int, help='enlarges the number of tiles i.e. 3x3 or tiles or 5 x 5 tiles')
    parser.add_argument('--n-prcs', default=5, type=int, help='number of cores')
    parser.add_argument('--add-leaves', action='store_true', help='add leaf points')
    parser.add_argument('--verbose', action='store_true', help='print something')
    params = parser.parse_args()

    if params.tile == '':
        raise Exception('specify --tile')    

    params.not_base = 'XXXXXXXX'  

    params.dir, params.fn = os.path.split(params.tile)
    params.n = int(params.fn.split('.')[0])
    params.tmp = glob.glob(os.path.join(params.dir, f'{params.n:03}.*.tmp'))[0]

    # all_samples are the convex hull points
    params.all_samples = ply_io.read_ply(glob.glob(os.path.join(params.tmp, f'{params.n:03}*samples*'))[0])
    params.all_samples.loc[:, 'buffer'] = False
    params.all_samples.loc[:, 'fn'] = params.n
    t_clstr = unique_cluster_id(params.all_samples.clstr)
    params.all_samples.loc[:, 't_clstr'] = params.all_samples.clstr.map(t_clstr)

    # stem skeleton are the stem points (prob coild be read from XXX*.segmented.ply)
    params.all_skeleton = ply_io.read_ply(glob.glob(os.path.join(params.tmp, f'{params.n:03}*stem_skeleton*'))[0])
    params.all_skeleton.loc[:, 'buffer'] = False
    params.all_skeleton.loc[:, 'fn'] = params.n
    params.all_skeleton.loc[:, 't_clstr'] = params.all_skeleton.clstr.map(t_clstr)
    # variable contains the ids of all stems in the focus tile, 
    # this is used later when saving just the focus trees
    params.in_tile_stem_nodes = params.all_skeleton.loc[params.all_skeleton.dbh_node == 1].t_clstr

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
                     desc='read in skeleton and convex hull', 
                     disable=False if params.verbose else True):
        try:
            b_tile = glob.glob(os.path.join(params.dir, f'{t:03}*', f'{t:03}*samples*'))[0]
            tmp = ply_io.read_ply(b_tile)
            tmp.loc[:, 'buffer'] = True
            tmp.loc[:, 'fn'] = t

            # generate unique cluster name
            t_clstr = unique_cluster_id(tmp.clstr)
            tmp.loc[:, 't_clstr'] = tmp.clstr.map(t_clstr) # and map

            # append to all_samples
            params.all_samples = params.all_samples.append(tmp, ignore_index=True)

            # read in stem points
            if os.path.isfile(b_tile.replace('samples', 'stem_skeleton')):
                skel = ply_io.read_ply(b_tile.replace('samples', 'stem_skeleton'))
                skel.loc[:, 'buffer'] = True
                skel.loc[:, 'fn'] = t
                skel.dbh_node = skel.dbh_node.astype(bool)
                skel.loc[:, 't_clstr'] = skel.clstr.map(t_clstr)
                params.all_skeleton = params.all_skeleton.append(skel, ignore_index=True)
        except:
            t = os.path.join(params.dir, f'{t:03}*', f'{t:03}*samples*')
            raise Exception(f'tile does not exist: {t}')

    # reset indexes
    params.all_samples.reset_index(inplace=True, drop=True)
    params.all_skeleton.reset_index(inplace=True, drop=True)

    # generates paths through all stem points
    if params.verbose: print('generating graph, this may take a while...')
    all_paths = generate_path(params.all_samples, params.all_skeleton)

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
    params.RGB.loc[params.RGB.base == params.not_base, :] = [np.nan, 211, 211, 211] # color unassigned points grey
    stems = pd.merge(stems, params.RGB, on='base', how='right')

    # read in all "stems" tiles and assign all stem points to a tree
    trees = pd.DataFrame(columns=['x', 'y', 'z', 'clstr', 'base'])

    for i, t in tqdm(enumerate(list(buffer_tiles) + [params.n]), 
                     total=len(list(buffer_tiles) + [params.n]),
                     desc='read in skeleton and convex hull', 
                     disable=False if params.verbose else True):

        fn = glob.glob(os.path.join(params.dir, f'{t:03}*', f'{t:03}.downsample.stems.ply'))[0]
        ply = ply_io.read_ply(fn)
        N = int(os.path.split(fn)[1].split('.')[0])
        # knowing the tile here is important as the global clstr is
        # not stored in each tile
        PLY = pd.merge(ply, stems.loc[stems.fn == N][['clstr', 'base', 'red', 'green', 'blue']], on='clstr')
        trees = trees.append(PLY, ignore_index=True)

    # write out all trees
    params.base_I, I = {}, 0
    for i, b in tqdm(enumerate(params.in_tile_stem_nodes), 
                     total=len(params.in_tile_stem_nodes), 
                     desc='writing stems to file', 
                     disable=False if params.verbose else True):
        if b == params.not_base: continue
        ply_io.write_ply(os.path.join(params.odir, f'{params.n:03}_T{I}.leafoff.ply'), trees.loc[trees.base == b])
        params.base_I[b] = I
        I += 1    

    if params.add_leaves:
        
        stems.base = [params.not_base if isinstance(b, float) else b for b in stems.base]

        # link stem number to t_clstr
        stem2tlsctr = stems[['t_clstr', 'base']].loc[stems.base != params.not_base].set_index('t_clstr').to_dict()['base']
        params.all_samples.loc[:, 'stem'] = params.all_samples.t_clstr.map(stem2tlsctr)
        
        # read in leaf point clouds and attribute to trees
        lvs = pd.DataFrame()
        for i, t in tqdm(enumerate(list(buffer_tiles) + [params.n]),
                         total=len(list(buffer_tiles) + [params.n]),
                         desc='add leaves', 
                         disable=False if params.verbose else True):

            tmp = ply_io.read_ply(glob.glob(os.path.join(params.dir, f'{t:03}*.segmented.ply'))[0])
            if 'nz' in tmp.columns: tmp.rename(columns={'nz':'n_z'}, inplace=True)
            tmp = tmp.loc[(tmp.label == 1) & (tmp.n_z >= 2)]
            lvs = lvs.append(tmp)

        lvs = lvs.drop(columns=['distance'] if 'distance' in lvs.columns else[])

        # voxelise
        lvs = voxelise(lvs, length=.2)
        lvs_median = lvs.groupby('VX')[['x', 'y', 'z']].median().reset_index()
        lvs_median.loc[:, 'label'] = 1

        # subsample stem pc
        params.all_samples = params.all_samples.loc[[False if isinstance(x, float) else 
                                                     True for x in params.all_samples.stem]]
        params.all_samples.loc[:, 'label'] = 2

        # and combine leaves and wood
        branch_and_leaves = lvs_median.append(params.all_samples[['x', 'y', 'z', 'label', 'stem']])
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
        num_neighbours = 50
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
        for lv in lvs.loc[lvs.base.isin(params.in_tile_stem_nodes)].base.unique():

            I = params.base_I[lv]

            if os.path.isfile(os.path.join(params.odir, f'{params.n:03}_T{I}.leafon.ply')):
                stem = ply_io.read_ply(os.path.join(params.odir, f'{params.n:03}_T{I}.leafon.ply'))
            else:
                stem = ply_io.read_ply(os.path.join(params.odir, f'{params.n:03}_T{I}.leafoff.ply'))
                stem.loc[:, 'wood'] = 1

            l2a = lvs.loc[lvs.base == lv]
            l2a.loc[:, 'wood'] = 0
            stem = stem.append(l2a[['x', 'y', 'z', 'red', 'green', 'blue', 'base', 'wood', 'distance']])

            stem = stem.loc[~stem.duplicated()]
            ply_io.write_ply(os.path.join(params.odir, f'{params.n:03}_T{I}.leafon.ply'), stem)
