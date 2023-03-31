import os
import multiprocessing
import argparse

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

def generate_path(samples, origins, n_neighbours=200, max_length=0):

    # compute nearest neighbours for each vertex in cluster convex hull
    nn = NearestNeighbors(n_neighbors=n_neighbours).fit(samples[['x', 'y', 'z']])
    distances, indices = nn.kneighbors()    
    from_to_all = pd.DataFrame(np.vstack([np.repeat(samples.clstr.values, n_neighbours), 
                                          samples.iloc[indices.ravel()].clstr.values, 
                                          distances.ravel()]).T, 
                               columns=['source', 'target', 'length'])

    # remove X-X connections
    from_to_all = from_to_all.loc[from_to_all.target != from_to_all.source]

    # and build edge database where edges with min distance between clusters persist
    edges = from_to_all.groupby(['source', 'target']).length.min().reset_index()
    # remove edges that are likely leaps between trees
    edges = edges.loc[edges.length <= max_length]

    # removes isolated origin points i.e. > edge.length
    origins = [s for s in origins if s in edges.source.values] 

    # compute graph
    G = nx.from_pandas_edgelist(edges, edge_attr=['length'])
    distance, shortest_path = nx.multi_source_dijkstra(G, 
                                                       sources=origins,
                                                       weight='length')

    paths = pd.DataFrame(index=distance.keys(), data=distance.values(), columns=['distance'])
    paths.loc[:, 'base'] = params.not_base
    for p in paths.index: paths.loc[p, 'base'] = shortest_path[p][0]
    paths.reset_index(inplace=True)
    paths.columns = ['clstr', 'distance', 't_clstr']
    
    # identify nodes that are branch tips
    node_occurance = {}
    for v in shortest_path.values():
        for n in v:
            if n in node_occurance.keys(): node_occurance[n] += 1
            else: node_occurance[n] = 1

    tips = [k for k, v in node_occurance.items() if v == 1]

    paths.loc[:, 'is_tip'] = False
    paths.loc[paths.clstr.isin(tips), 'is_tip'] = True

    return paths


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
    parser.add_argument('--overlap', default=False, type=float, help='buffer to crop adjacent tiles')
    parser.add_argument('--slice-thickness', default=.2, type=float, help='slice thickness for constructing graph')
    parser.add_argument('--find-stems-height', default=1.5, type=float, help='height for identifying stems')    
    parser.add_argument('--find-stems-thickness', default=.5, type=float, help='thickness of slice used for identifying stems')
    parser.add_argument('--find-stems-min-radius', default=.025, type=float, help='minimum radius of found stems')
    parser.add_argument('--find-stems-min-points', default=200, type=int, help='minimum number of points for found stems')
    parser.add_argument('--graph-edge-length', default=1, type=float, help='maximum distance used to connect points in graph')
    parser.add_argument('--graph-maximum-cumulative-gap', default=np.inf, type=float, 
                        help='maximum cumulative distance between a base and a cluster')
    parser.add_argument('--min-points-per-tree', default=0, type=int, help='minimum number of points for a identified tree')
    parser.add_argument('--add-leaves', action='store_true', help='add leaf points')
    parser.add_argument('--add-leaves-voxel-length', default=.5, type=float, help='voxel size when add leaves')
    parser.add_argument('--add-leaves-edge-length', default=1, type=float, 
                        help='maximum distance used to connect points in leaf graph')
    parser.add_argument('--save-diameter-class', action='store_true', help='save into diameter class directories')
    parser.add_argument('--ignore-missing-tiles', action='store_true', help='ignore missing neighbouring tiles')
    parser.add_argument('--pandarallel', action='store_true', help='use pandarallel')
    parser.add_argument('--verbose', action='store_true', help='print something')
    params = parser.parse_args()
        
    if params.pandarallel:
        try:
            from pandarallel import pandarallel
            pandarallel.initialize(progress_bar=True if params.verbose else False, use_memory_fs=False)
        except:
            print('--- pandarallel not installed ---')
            params.pandarallel = False
    
    if params.verbose:
        print('---- parameters ----')
        for k, v in params.__dict__.items():
            print(f'{k:<35}{v}')

    params.not_base = -1
    xyz = ['x', 'y', 'z'] # shorthand

    params.dir, params.fn = os.path.split(params.tile)
    params.n = params.fn.split('.')[0]
    params.n = int(params.n) if params.n.isdigit() else params.n

    params.pc = ply_io.read_ply(params.tile)
    params.pc.loc[:, 'buffer'] = False
    params.pc.loc[:, 'fn'] = params.n
    
    # Shift to local coordinates from UTM
    params.global_shift = params.pc[['x', 'y']].mean()
    if params.verbose: print('global shift:', params.global_shift.values)
    params.pc[['x', 'y']] -= params.global_shift

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
            b_tile = glob.glob(os.path.join(params.dir, f'{t}*.ply'))[0]
            tmp = ply_io.read_ply(b_tile)
            
            # Apply global shift to tile
            tmp[['x', 'y']] -= params.global_shift
            
            if params.overlap:
                tmp = tmp.loc[(tmp.x.between(bbox.xmin - params.overlap, bbox.xmax + params.overlap)) & 
                              (tmp.y.between(bbox.ymin - params.overlap, bbox.ymax + params.overlap))]
            if len(tmp) == 0: continue
            tmp.loc[:, 'buffer'] = True
            tmp.loc[:, 'fn'] = t
            params.pc = params.pc.append(tmp, ignore_index=True)
        except:
            path = os.path.join(params.dir, f'{t}*.ply')
            if params.ignore_missing_tiles:
                print(f'tile {path} not available')
            else:
                raise Exception(f'tile {path} not available')
    
    # --- this can be dropeed soon --- 
    if 'nz' in params.pc.columns: params.pc.rename(columns={'nz':'n_z'}, inplace=True)
        
    # save space
    params.pc = params.pc[[c for c in ['x', 'y', 'z', 'n_z', 'label', 'buffer', 'fn']]]
    params.pc[['x', 'y', 'z', 'n_z']] = params.pc[['x', 'y', 'z', 'n_z']].astype(np.float32)
    params.pc[['label', 'fn']] = params.pc[['label', 'fn']].astype(np.int16)

    ### generate skeleton points
    if params.verbose: print('\n----- skeletonisation started -----')

    # extract stems points and slice slice
    stem_pc = params.pc.loc[params.pc.label == 3]

    # slice stem_pc
    stem_pc.loc[:, 'slice'] = (stem_pc.z // params.slice_thickness).astype(int) * params.slice_thickness
    stem_pc.loc[:, 'n_slice'] = (stem_pc.n_z // params.slice_thickness).astype(int)

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

    # dbh_nodes = skeleton.loc[skeleton.n_slice == params.slice_height].clstr
    find_stems_min = int(params.find_stems_height // params.slice_thickness) 
    find_stems_max = int((params.find_stems_height + params.find_stems_thickness) // params.slice_thickness)  + 1
    dbh_nodes_plus = skeleton.loc[skeleton.n_slice.between(find_stems_min, find_stems_max)].clstr
    dbh_slice = stem_pc.loc[stem_pc.clstr.isin(dbh_nodes_plus)]

    if len(dbh_slice) > 0:

        # remove noise from dbh slice
        nn = NearestNeighbors(n_neighbors=10).fit(dbh_slice[xyz])
        distances, indices = nn.kneighbors()
        dbh_slice.loc[:, 'nn'] = distances[:, 1:].mean(axis=1)
        dbh_slice = dbh_slice.loc[dbh_slice.nn < dbh_slice.nn.quantile(q=.9)]

        # run dbscan over dbh_slice to find potential stems
        dbscan = DBSCAN(eps=.2, min_samples=50).fit(dbh_slice[['x', 'y']])
        dbh_slice.loc[:, 'clstr_db'] = dbscan.labels_
        dbh_slice = dbh_slice.loc[dbh_slice.clstr_db > -1]
        dbh_slice.loc[:, 'cclstr'] = dbh_slice.groupby('clstr_db').clstr.transform('min')

        if len(dbh_slice) > 10: 

            # ransac cylinder fitting
            if params.verbose: print('fitting cylinders to possible stems...')
            if params.pandarallel:
                dbh_cylinder = dbh_slice.groupby('cclstr').parallel_apply(RANSAC_helper, 100, ).to_dict()
            else:
                dbh_cylinder = dbh_slice.groupby('cclstr').apply(RANSAC_helper, 100, ).to_dict()
            dbh_cylinder = pd.DataFrame(dbh_cylinder).T
            dbh_cylinder.columns = ['radius', 'centre', 'CV', 'cnt']
            dbh_cylinder.loc[:, ['x', 'y', 'z']] = [[*row.centre] for row in dbh_cylinder.itertuples()]
            dbh_cylinder = dbh_cylinder.drop(columns=['centre']).astype(float)

            # identify clusters where cylinder CV <= .75 and label as nodes
            skeleton.loc[skeleton.clstr.isin(dbh_cylinder.loc[(dbh_cylinder.radius > params.find_stems_min_radius) &
                                                              (dbh_cylinder.cnt > params.find_stems_min_points) &
                                                              (dbh_cylinder.CV <= .15)].index.values), 'dbh_node'] = True

    in_tile_stem_nodes = skeleton.loc[(skeleton.dbh_node) & 
                                      (skeleton.x.between(bbox.xmin, bbox.xmax)) &
                                      (skeleton.y.between(bbox.ymin, bbox.ymax))].clstr
    
    # generates paths through all stem points
    if params.verbose: print('generating graph, this may take a while...')
    wood_paths = generate_path(chull, 
                               skeleton.loc[skeleton.dbh_node].clstr, 
                               n_neighbours=200, 
                               max_length=params.graph_edge_length)

    # removes paths that are longer for same clstr
    wood_paths = wood_paths.sort_values(['clstr', 'distance'])
    wood_paths = wood_paths.loc[~wood_paths['clstr'].duplicated()] 
    
    # remove clusters that are linked to a base by a cumulative
    # distance greater than X 
    wood_paths = wood_paths.loc[wood_paths.distance <= params.graph_maximum_cumulative_gap]

    if params.verbose: print('merging skeleton points with graph')
    stems = pd.merge(skeleton, wood_paths, on='clstr', how='left')

    # give a unique colour to each tree (helps with visualising)
    stems.drop(columns=[c for c in stems.columns if c.startswith('red') or 
                                                    c.startswith('green') or 
                                                    c.startswith('blue')], inplace=True)

    # generate unique RGB for each stem
    unique_stems = stems.t_clstr.unique()
    RGB = pd.DataFrame(data=np.vstack([unique_stems, 
                                       np.random.randint(0, 255, size=(3, len(unique_stems)))]).T, 
                       columns=['t_clstr', 'red', 'green', 'blue'])
    RGB.loc[RGB.t_clstr == params.not_base, :] = [np.nan, 211, 211, 211] # color unassigned points grey
    stems = pd.merge(stems, RGB, on='t_clstr', how='right')

    # read in all "stems" tiles and assign all stem points to a tree
    trees = pd.merge(stem_pc, 
                     stems[['clstr', 't_clstr', 'distance', 'red', 'green', 'blue']], 
                     on='clstr')
    trees[['x', 'y']] += params.global_shift  # Shift back from local to UTM
    trees.loc[:, 'cnt'] = trees.groupby('t_clstr').t_clstr.transform('count')
    trees = trees.loc[trees.cnt > params.min_points_per_tree]
    in_tile_stem_nodes = trees.loc[trees.t_clstr.isin(in_tile_stem_nodes)].t_clstr.unique()

    # write out all trees
    params.base_I, I = {}, 0
    for i, b in tqdm(enumerate(dbh_cylinder.loc[in_tile_stem_nodes].sort_values('radius', ascending=False).index), 
                     total=len(in_tile_stem_nodes), 
                     desc='writing stems to file', 
                     disable=False if params.verbose else True):

        if b == params.not_base: 
            continue
    
        if params.save_diameter_class:
            d_dir = f'{(dbh_cylinder.loc[b].radius * 2 // .1) / 10:.1f}'
            if not os.path.isdir(os.path.join(params.odir, d_dir)):
                os.makedirs(os.path.join(params.odir, d_dir))
            ply_io.write_ply(os.path.join(params.odir, d_dir, f'{params.n}_T{I}.leafoff.ply'), 
                             trees.loc[trees.t_clstr == b])  
        else:
            ply_io.write_ply(os.path.join(params.odir, f'{params.n}_T{I}.leafoff.ply'), 
                             trees.loc[trees.t_clstr == b])
        params.base_I[b] = I
        I += 1  

    if params.add_leaves:
        
        if params.verbose: print('adding leaves to stems, this may take a while...')

        # link stem number to clstr
        stem2tlsctr = stems[['clstr', 't_clstr']].loc[stems.t_clstr != params.not_base].set_index('clstr').to_dict()['t_clstr']
        chull.loc[:, 'stem'] = chull.clstr.map(stem2tlsctr)

        # identify unlabelled woody points to add back to leaves
        unlabelled_wood = chull.loc[[True if np.isnan(s) else False for s in chull.stem]]
        unlabelled_wood = stem_pc.loc[stem_pc.clstr.isin(unlabelled_wood.clstr.to_list() + [-1])]

        # extract wood points that are attributed to a base and that are the 
        # the last clstr of the graph i.e. a tip
        is_tip = wood_paths.set_index('clstr')['is_tip'].to_dict()
        chull = chull.loc[[False if np.isnan(s) else True for s in chull.stem]]
        chull.loc[:, 'is_tip'] = chull.clstr.map(is_tip)
        chull = chull.loc[(chull.is_tip) & (chull.n_z > params.find_stems_height)]
        chull.loc[:, 'xlabel'] = 2

        # process leaf points
        lvs = params.pc.loc[(params.pc.label == 1) & (params.pc.n_z >= 2)].copy()
        lvs = lvs.append(unlabelled_wood, ignore_index=True)
        lvs.reset_index(inplace=True)

        # voxelise
        lvs = voxelise(lvs, length=params.add_leaves_voxel_length)
        lvs_gb = lvs.groupby('VX')[xyz]
        lvs_min = lvs_gb.min()
        lvs_max = lvs_gb.max()
        lvs_med = lvs_gb.median()

        # find faces of leaf voxels and create database 
        cnrs = np.vstack([lvs_min.x, lvs_med.y, lvs_med.z]).T
        clstr = np.tile(np.arange(len(lvs_min.index)) + 1 + chull.clstr.max(), 6)
        VX = np.tile(lvs_min.index, 6)
        cnrs = np.vstack([cnrs, np.vstack([lvs_max.x, lvs_med.y, lvs_med.z]).T])
        cnrs = np.vstack([cnrs, np.vstack([lvs_med.x, lvs_min.y, lvs_med.z]).T])
        cnrs = np.vstack([cnrs, np.vstack([lvs_med.x, lvs_max.y, lvs_med.z]).T])
        cnrs = np.vstack([cnrs, np.vstack([lvs_med.x, lvs_med.y, lvs_min.z]).T])
        cnrs = np.vstack([cnrs, np.vstack([lvs_med.x, lvs_med.y, lvs_max.z]).T])
        cnrs = pd.DataFrame(cnrs, columns=['x', 'y', 'z'])
        cnrs.loc[:, 'xlabel'] = 1
        cnrs.loc[:, 'clstr'] = clstr
        cnrs.loc[:, 'VX'] = VX

        # and combine leaves and wood
        branch_and_leaves = cnrs.append(chull[['x', 'y', 'z', 'label', 'stem', 'xlabel', 'clstr']])
        branch_and_leaves.reset_index(inplace=True, drop=True)

        # find neighbouring branch and leaf points - used as entry points
        nn = NearestNeighbors(n_neighbors=2).fit(branch_and_leaves[xyz])
        distances, indices = nn.kneighbors()   
        closest_point_to_leaf = indices[:len(cnrs), :].flatten() # only leaf points
        idx = np.isin(closest_point_to_leaf, branch_and_leaves.loc[branch_and_leaves.xlabel == 2].index)
        close_branch_points = closest_point_to_leaf[idx] # points where the branch is closest

        # remove all branch points that are not close to leaves
        idx = np.hstack([branch_and_leaves.iloc[:len(cnrs)].index.values, close_branch_points])
        bal = branch_and_leaves.loc[branch_and_leaves.index.isin(np.unique(idx))]

        # generate a leaf paths graph
        leaf_paths = generate_path(bal, 
                                   bal.loc[bal.xlabel == 2].clstr.unique(), 
                                   max_length=1, # i.e. any leaves which are separated by greater are ignored
                                   n_neighbours=20)
             
        leaf_paths = leaf_paths.sort_values(['clstr', 'distance'])
        leaf_paths = leaf_paths.loc[~leaf_paths['clstr'].duplicated()] # removes duplicate paths
        leaf_paths = leaf_paths.loc[leaf_paths.distance > 0] # removes within cluseter paths 

        # linking indexs to stem number
        top2stem = branch_and_leaves.loc[branch_and_leaves.xlabel == 2].set_index('clstr')['stem'].to_dict()
        leaf_paths.loc[:, 't_clstr'] = leaf_paths.t_clstr.map(top2stem)
        #     paths.loc[:, 'stem'] = paths.stem_.map(base2i)

        # linking index to VX number
        index2VX = branch_and_leaves.loc[branch_and_leaves.xlabel == 1].set_index('clstr')['VX'].to_dict()
        leaf_paths.loc[:, 'VX'] = leaf_paths['clstr'].map(index2VX)

        # colour the same as stem
        lvs = pd.merge(lvs, leaf_paths[['VX', 't_clstr', 'distance']], on='VX', how='left')

        # and save
        
        for lv in tqdm(in_tile_stem_nodes):

            I = params.base_I[lv]

            wood_fn = glob.glob(os.path.join(params.odir, '*', f'{params.n}_T{I}.leafoff.ply'))[0]

            stem = ply_io.read_ply(os.path.join(wood_fn))
            stem.loc[:, 'wood'] = 1

            stem[['x', 'y']] -= params.global_shift  # Shift from UTM to local
            
            l2a = lvs.loc[lvs.t_clstr == lv]
            if len(l2a) > 0:
                l2a.loc[:, 'wood'] = 0
                
                # colour the same as stem
                rgb = RGB.loc[RGB.t_clstr == lv][['red', 'green', 'blue']].values[0] * 1.2
                l2a.loc[:, ['red', 'green', 'blue']] = [c if c <= 255 else 255 for c in rgb]

                stem = stem.append(l2a[['x', 'y', 'z', 'label', 'red', 'green', 'blue', 't_clstr', 'wood', 'distance']])

            stem = stem.loc[~stem.duplicated()]
            stem[['x', 'y']] += params.global_shift  # Reset from local coords to UTM
            ply_io.write_ply(wood_fn.replace('leafoff', 'leafon'), 
                             stem[['x', 'y', 'z', 'red', 'green', 'blue', 'label', 't_clstr', 'wood', 'distance']])
            if params.verbose: print(f"leaf on saved to: {wood_fn.replace('leafoff', 'leafon')}") 
