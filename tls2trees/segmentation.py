import os
import time
import threading
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import networkx as nx

from fsct.tools import *
from fsct.fit_cylinders import RANSAC_helper

pd.options.mode.chained_assignment = None

def cube(pc):
    
    if len(pc) > 5:
        vertices = ConvexHull(pc[['x', 'y', 'z']]).vertices
        idx = np.random.choice(vertices, size=len(vertices), replace=False)
        return pc.loc[pc.index[idx]]
    else:
        return pc 

def Segmentation(params):
    
    xyz = ['x', 'y', 'z'] # simplifies specifiying xyz columns
    
    if params.verbose: print('\n----- skeletonisation started -----')
    start_time = time.time()
    
    # extract stems points and slice slice
    stems = params.pc.loc[(~params.pc.buffer) & (params.pc.label == params.stem_class)]

    if 'nz' in stems.columns: stems.rename(columns={'nz':'n_z'}, inplace=True)
 
    # slice stems
    slice_thickness = .1
    stems.loc[:, 'slice'] = (stems.z // slice_thickness).astype(int) * slice_thickness
    stems.loc[:, 'n_slice'] = (stems.n_z // slice_thickness).astype(int)
    
    # cluster within height slices
    stems.loc[:, 'clstr'] = -1
    label_offset = 0


    for slice_height in tqdm(np.sort(stems.slice.unique()), 
                             disable=False if params.verbose else True,
                             desc='slice data vertically and clustering'):

        new_slice = stems.loc[np.isclose(stems['slice'], slice_height)]

        if len(new_slice) > 100:
            dbscan = DBSCAN(eps=.1, min_samples=20).fit(new_slice[xyz])
            new_slice.loc[:, 'clstr'] = dbscan.labels_
            new_slice.loc[new_slice.clstr > -1, 'clstr'] += label_offset
            stems.loc[new_slice.index, 'clstr'] = new_slice.clstr
            label_offset = stems.clstr.max() + 1

    # stem point clouds
    save_file(os.path.join(params.working_dir, f'{params.basename}.stems.ply'),
              stems,                                     
              additional_fields=['clstr'])
    
    # group clusters and compute convex hulls
    #pandarallel.initialize(progress_bar=params.verbose)
    grouped = stems.loc[stems.clstr != -1].groupby('clstr')
    #samples = grouped.parallel_apply(cube) # parallel_apply only works witn pd < 1.3
    samples = grouped.apply(cube) # don't think works with Jasmin or parallel_apply only works witn pd < 1.3
    samples = samples.reset_index(drop=True)

    # edges from convex hull 
    save_file(os.path.join(params.working_dir, f'{params.basename}.samples.ply'), 
              samples, 
              additional_fields=['clstr'])

    ### identify possible stems ###
    stem_skeleton = grouped[xyz + ['n_slice', 'slice']].median().reset_index()
    stem_skeleton.loc[:, 'dbh_node'] = False
    # find the highest stem to take a slice, this is to reduce imapct of slope
    dbh_slice_N = stem_skeleton.loc[stem_skeleton.n_slice == 13].slice.max()
    dbh_nodes = stem_skeleton.loc[stem_skeleton.slice == dbh_slice_N].clstr
    dbh_slice = stems.loc[stems.clstr.isin(dbh_nodes)]
   
    if len(dbh_slice) > 0:

        # remove noise from dbh slice
        nn = NearestNeighbors(n_neighbors=10).fit(dbh_slice[xyz])
        distances, indices = nn.kneighbors()
        dbh_slice.loc[:, 'nn'] = distances[:, 1:].mean()
        dbh_slice = dbh_slice.loc[dbh_slice.nn < .05]
        
        # run dbscan over dbh_slice
        dbscan = DBSCAN(eps=.2, min_samples=100).fit(dbh_slice[['x', 'y']])
        dbh_slice.loc[:, 'clstr_db'] = dbscan.labels_
        dbh_slice = dbh_slice.loc[dbh_slice.clstr_db > -1]
       
        if len(dbh_slice) > 10: 
 
            # ransac cylinder fitting
            #dbh_cylinder = dbh_slice.groupby('clstr').parallel_apply(RANSAC_helper, 10, 50).to_dict()
            dbh_cylinder = dbh_slice.groupby('clstr').apply(RANSAC_helper, 10, 50).to_dict()
            dbh_cylinder = pd.DataFrame(dbh_cylinder).T
            dbh_cylinder.columns = ['radius', 'centre', 'error', 'err_std']
            dbh_cylinder.loc[:, 'CV'] = dbh_cylinder.err_std / dbh_cylinder.error # CV used to determine cylinrical-ness
            
            dbh_cylinder.to_csv(os.path.join(params.working_dir, f'{params.basename}.stem_attributes.csv'), index=False)
            
            # identify clusters where cylinder CV <= .75 and label as nodes
            stem_skeleton.loc[stem_skeleton.clstr.isin(dbh_cylinder.loc[dbh_cylinder.CV <= .75].index.values), 'dbh_node'] = True
    
            if params.verbose: print('saving stem locations to:', os.path.join(params.working_dir,
                                                                               f'{params.basename}.stem_location.ply'))
            save_file(os.path.join(params.working_dir, f'{params.basename}.stem_location.ply'),
                      stem_skeleton.loc[stem_skeleton.dbh_node], 
                      additional_fields=['clstr'])

    # stem_skeleton
    save_file(os.path.join(params.working_dir, f'{params.basename}.stem_skeleton.ply'), 
              stem_skeleton, 
              additional_fields=['clstr', 'dbh_node'])
 
    return params

### commented out but might want to reinstate for a single tile ###    
#    ### build graph ###
#    # compute nearest neighbours for each vertex in cluster convex hull
#    num_neighbours = 200
#    nn = NearestNeighbors(n_neighbors=num_neighbours).fit(samples[['x', 'y', 'z']])
#    distances, indices = nn.kneighbors()    
#    from_to_all = pd.DataFrame(np.vstack([np.repeat(samples.iloc[list(indices[:, 0])].clstr.values, num_neighbours), 
#                                      samples.iloc[indices.ravel()].clstr.values, 
#                                      distances.ravel()]).T, 
#                               columns=['source', 'target', 'length'])
#    
#    # remove X-X connections
#    from_to_all = from_to_all.loc[from_to_all.target != from_to_all.source]
#    # and build edge database where edges with min distance between clusters persist
#    edges = from_to_all.groupby(['source', 'target']).length.min().reset_index()
#    # remove edges that are likely leaps between trees
#    edges = edges.loc[edges.length <= .2]
#    #edges = edges.loc[edges.length <= .5]
#    
#    # compute graph
#    G = nx.from_pandas_edgelist(edges, edge_attr=['length'])
#    distance, shortest_path = nx.multi_source_dijkstra(G, 
#                                                       sources=list(stem_skeleton.loc[stem_skeleton.dbh_node].clstr),
#                                                       weight='length')
#    
#    # assign nodes to trees
#    trees = {}
#    for n, clstr in shortest_path.items():
#        base = clstr[0]
#        if base in trees.keys():
#            trees[base] = trees[base].union(set(clstr + [n]))
#        else: trees[base] = set(clstr + [n])
#            
#    # assign tree_id to stem points 
#    stems.loc[:, 'tree'], stem_skeleton.loc[:, 'tree'] = np.nan, np.nan
#    for tree, nodes in trees.items():
#        stems.loc[stems.clstr.isin(nodes), 'tree'] = tree
#        stem_skeleton.loc[stem_skeleton.clstr.isin(nodes), 'tree'] = tree
#        
#    # give a unique colour to each tree (helps with visualising)
#    stems.drop(columns=[c for c in stems.columns if c.startswith('red') or 
#                                                    c.startswith('green') or 
#                                                    c.startswith('blue')], inplace=True)
#    unique_stems = stems.tree.unique()
#    RGB = pd.DataFrame(data=np.vstack([unique_stems, np.random.randint(0, 255, size=(3, len(unique_stems)))]).T, 
#                       columns=['tree', 'red', 'green', 'blue'])
#    RGB.loc[np.isnan(RGB.tree), :] = [np.nan, 211, 211, 211] # color unassigned points grey
#    stems = pd.merge(stems, RGB, on='tree', how='right')
#
#    # ...and export
#    if params.verbose: 
#        print('saving trees to:', os.path.abspath(os.path.join(params.working_dir, f'{params.basename}.trees.ply')))
#    save_file(os.path.join(params.working_dir, f'{params.basename}.trees.ply'), 
#              stems, 
#              additional_fields=['clstr', 'tree', 'red', 'green', 'blue'])
#                             
#    
#    return params
