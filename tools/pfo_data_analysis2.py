import math
import sys
from copy import deepcopy
from multiprocessing import get_context
import networkx as nx
import numpy as np
import pandas as pd
import simplekml
import utm
import os
from scipy import spatial
from scipy.spatial import ConvexHull
from scipy.interpolate import CubicSpline
from scipy.interpolate import griddata
from skimage.measure import LineModelND, CircleModel, ransac
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from matplotlib import pyplot as plt
from tools import load_file, save_file, low_resolution_hack_mode, subsample_point_cloud, clustering
import time
import hdbscan
from skspatial.objects import Plane
import warnings
from matplotlib import cm
from matplotlib.colors import Normalize


def create_3d_circles_as_points_flat(x, y, z, r, circle_points=15, label=0):
    angle_between_points = np.linspace(0, 2 * np.pi, circle_points)
    points = np.zeros((0, 5))
    for i in angle_between_points:
        x2 = r * np.cos(i) + x
        y2 = r * np.sin(i) + y
        point = np.array([[x2, y2, z, r, label]])
        points = np.vstack((points, point))
    return points


def get_taper(single_tree_cyls, slice_heights):
    """
    Accepts single tree of cylinders and start height, stop height and increment.
    From start height to stop height (relative to ground), extract the largest
    diameter at a specific increment for that tree.

    Returns:
    slice_heights
    List of slice heights.

    diameters
    List of diameters corresponding to slice heights.

    """
    cyl_dict = dict(x=0, y=1, z=2, nx=3, ny=4, nz=5, radius=6, CCI=7, branch_id=8, parent_branch_id=9,
                    tree_id=10, segment_volume=11, segment_angle_to_horiz=12, height_above_dtm=13)

    diameters = []
    CCI = []
    single_tree_cyls
    for height in slice_heights:
        results = single_tree_cyls[np.logical_and(single_tree_cyls[:, cyl_dict['height_above_dtm']] >= height-0.1,
                                                  single_tree_cyls[:, cyl_dict['height_above_dtm']] <= height+0.1)]
        if results.shape[0] > 0:
            index = np.argmax(results[:, cyl_dict['radius']])
            # index = list(results[:, cyl_dict['radius']]).index(np.percentile(results[:, cyl_dict['radius']],
            #                                                                  50, interpolation='nearest'))
            diameters.append(results[index, cyl_dict['radius']] * 2)
            CCI.append(results[index, cyl_dict['CCI']])
        else:
            CCI.append(0)
            diameters.append(0)

    return diameters, CCI


def extract_tapers_from_plot(plot_cyls, cyl_dict, slice_heights):
    taper_array = np.zeros((0, 1 + slice_heights.shape[0]))
    CCI_array = np.zeros((0, 1 + slice_heights.shape[0]))

    for tree_id in np.unique(plot_cyls[:, cyl_dict['tree_id']]):
        individual_tree_cyls = plot_cyls[plot_cyls[:, cyl_dict['tree_id']] == tree_id]
        # if individual_tree_cyls.shape[0] > 0:
        diameters, CCI = get_taper(individual_tree_cyls, slice_heights)
        taper_array = np.vstack((taper_array, np.hstack((np.array([tree_id]), diameters))))
        CCI_array = np.vstack((CCI_array, np.hstack((np.array([tree_id]), CCI))))

    return taper_array, CCI_array


def get_nearest_tree(reference_dataset, automatic_dataset, max_search_radius, ref_dict, auto_dict, sorted_trees_dict):
    tree_id_ref = reference_dataset[:, ref_dict['TreeNumber']]
    x_ref = reference_dataset[:, ref_dict['x_tree']]
    y_ref = reference_dataset[:, ref_dict['y_tree']]
    height_ref = reference_dataset[:, ref_dict['Height(m)']]
    dbh_ref = reference_dataset[:, ref_dict['DBH(mm)']] / 1000
    vol_ref = reference_dataset[:, ref_dict['DBH(mm)']] * 0
    tree_id_auto = automatic_dataset[:, auto_dict['treeNo']]
    x_auto = automatic_dataset[:, auto_dict['x_tree_base']]
    y_auto = automatic_dataset[:, auto_dict['y_tree_base']]
    z_base = automatic_dataset[:, auto_dict['z_tree_base']]
    height_auto = automatic_dataset[:, auto_dict['Height']]
    dbh_auto = automatic_dataset[:, auto_dict['DBH']]
    vol_auto = automatic_dataset[:, auto_dict['Volume']]

    reference_data_array = np.vstack((x_ref, y_ref, tree_id_ref, height_ref, dbh_ref, vol_ref)).T
    auto_data_array = np.vstack((x_auto, y_auto, z_base, tree_id_auto, height_auto, dbh_auto, vol_auto)).T
    sorted_trees_array = np.zeros((0, 13))

    if auto_data_array.shape[0] != 0:
        # print(auto_data_array.shape,reference_data_array.shape)
        # print(np.mean(auto_data_array[:,:2],axis=0),np.mean(reference_data_array[:,:2],axis=0))
        auto_data_array_unsorted = deepcopy(auto_data_array)
        for tree in reference_data_array:
            # print(tree)
            if ~np.isnan(tree[4]):
                best_tree_id = 0
                best_match = ''
                ref_kdtree = spatial.cKDTree(auto_data_array_unsorted[:, :2])
                results = ref_kdtree.query_ball_point(tree[:2], r=max_search_radius)
                candidate_tree_matches = auto_data_array_unsorted[results]
                candidate_tree_matches = candidate_tree_matches[
                    candidate_tree_matches[:, 4] > 0]  # trees with a valid DBH
                # print('Ref tree id:',tree[2],'Matching Trees:',candidate_tree_matches)
                if candidate_tree_matches.shape[0] > 0:
                    dbh_diff = candidate_tree_matches[:, 4] - tree[4]
                    best_match = candidate_tree_matches[np.argmin(np.abs(dbh_diff))]
                    # print(tree[4],best_match[4],tree[4]-best_match[4])
                    best_tree_id = best_match[2]
                    auto_data_array_unsorted = auto_data_array_unsorted[auto_data_array_unsorted[:, 2] != best_tree_id]
                sorted_tree = np.zeros((1, 13))
                sorted_tree[:, sorted_trees_dict['tree_id_ref']] = tree[2]
                sorted_tree[:, sorted_trees_dict['x_ref']] = tree[0]
                sorted_tree[:, sorted_trees_dict['y_ref']] = tree[1]
                sorted_tree[:, sorted_trees_dict['height_ref']] = tree[3]
                sorted_tree[:, sorted_trees_dict['dbh_ref']] = tree[4]
                sorted_tree[:, sorted_trees_dict['vol_ref']] = tree[5]

                if len(best_match) != 0:
                    sorted_tree[:, sorted_trees_dict['tree_id_auto']] = best_match[2]
                    sorted_tree[:, sorted_trees_dict['x_auto']] = best_match[0]
                    sorted_tree[:, sorted_trees_dict['y_auto']] = best_match[1]
                    sorted_tree[:, sorted_trees_dict['height_auto']] = best_match[3]
                    sorted_tree[:, sorted_trees_dict['dbh_auto']] = best_match[4]
                    sorted_tree[:, sorted_trees_dict['vol_auto']] = best_match[5]
                sorted_trees_array = np.vstack((sorted_trees_array, sorted_tree))
    return sorted_trees_array


point_clouds = ['T1_class', 'T02_class', 'T3_class', 'T4_class', 'T05_class',
                'T6_class', 'T7_class', 'T8_class', 'T9_class', 'T10_class',
                'T11_class', 'T12_class', 'T13_class', 'T14_class', 'T15_class',
                'T16_class', 'T017_class', 'T18_class', 'T19_class', 'T20_class',
                'T21_class', 'T22_class', 'T23_class', 'T25_class',
                'T26_class', 'T27_class', 'T28_class', 'TAPER29_class', 'TAPER30_class',
                'TAPER31_class', 'TAPER32_class', 'TAPER33_class', 'TAPER34_class', 'TAPER35_class',
                'TAPER36_class', 'TAPER37_class', 'TAPER38_class', 'TAPER39_class', 'TAPER40_class',
                'TAPER41_class', 'TAPER42_class', 'TAPER43_class', 'TAPER44_class', 'TAPER45_class',
                'TAPER46_class', 'TAPER47_class', 'TAPER48_class', 'TAPER49_class', 'TAPER50_class']

names = ['TAPER01', 'TAPER02', 'TAPER03', 'TAPER04', 'TAPER05',
         'TAPER06', 'TAPER07', 'TAPER08', 'TAPER09', 'TAPER10',
         'TAPER11', 'TAPER12', 'TAPER13', 'TAPER14', 'TAPER15',
         'TAPER16', 'TAPER17', 'TAPER18', 'TAPER19', 'TAPER20',
         'TAPER21', 'TAPER22', 'TAPER23', 'TAPER25',
         'TAPER26', 'TAPER27', 'TAPER28', 'TAPER29', 'TAPER30',
         'TAPER31', 'TAPER32', 'TAPER33', 'TAPER34', 'TAPER35',
         'TAPER36', 'TAPER37', 'TAPER38', 'TAPER39', 'TAPER40',
         'TAPER41', 'TAPER42', 'TAPER43', 'TAPER44', 'TAPER45',
         'TAPER46', 'TAPER47', 'TAPER48', 'TAPER49', 'TAPER50']

cyl_dict = dict(x=0, y=1, z=2, nx=3, ny=4, nz=5, radius=6, CCI=7, branch_id=8, parent_branch_id=9,
                tree_id=10, segment_volume=11, segment_angle_to_horiz=12, height_above_dtm=13)

reference_data_GT = pd.read_csv('E:/PFOlsen/green_triangle.csv')
reference_data_WA = pd.read_csv('E:/PFOlsen/western_australia.csv')

GT_tree_locations = pd.read_csv(
    'E:/PFOlsen/PFOlsenPlots/greentriangle/04_Spatial/tree_locations_from_PFO/taper_tree_Location.csv')
WA_tree_locations = pd.read_csv(
    'E:/PFOlsen/PFOlsenPlots/pfowesternaustralia/04_Spatial/tree_locations_from_PFO/TAPER_tree_locations.csv')

GT_tree_locations = np.array(GT_tree_locations)
WA_tree_locations = np.array(WA_tree_locations)

reference_data_GT.insert(4, 'x_tree', 0)
reference_data_GT.insert(5, 'y_tree', 0)
reference_data_WA.insert(4, 'x_tree', 0)
reference_data_WA.insert(5, 'y_tree', 0)

reference_headings = list(reference_data_WA.columns.values)
ref_dict = {i: reference_headings.index(i) for i in reference_headings}
reference_data = np.asarray(pd.concat([reference_data_GT, reference_data_WA]))

for PlotId in np.unique(reference_data[:, ref_dict['PlotId']]):
    # print(PlotId)
    current_plot_data = reference_data[:, reference_data[:, ref_dict['PlotId']] == PlotId]
    for TreeNumber in np.unique(current_plot_data[:, ref_dict['TreeNumber']]):
        # print(PlotId,TreeNumber)
        current_tree =
        row = WA_tree_locations[
            np.logical_and(WA_tree_locations[:, 0] == PlotId, WA_tree_locations[:, 1] == TreeNumber)]
        if row.shape[0] > 0:
            current_plot_data[current_plot_data[:, ref_dict['TreeNumber']] == TreeNumber), ref_dict['x_tree']] = row[0, 2]
            current_plot_data[np.logical_and(current_plot_data[:, ref_dict['PlotId']] == PlotId,
                                          current_plot_data[:, ref_dict['TreeNumber']] == TreeNumber), ref_dict[
                               'y_tree']] = row[0, 3]

        row2 = GT_tree_locations[
            np.logical_and(GT_tree_locations[:, 0] == PlotId, GT_tree_locations[:, 1] == TreeNumber)]
        if row2.shape[0] > 0:
            reference_data[np.logical_and(reference_data[:, ref_dict['PlotId']] == PlotId,
                                          reference_data[:, ref_dict['TreeNumber']] == TreeNumber), ref_dict[
                               'x_tree']] = row2[0, 2]
            reference_data[np.logical_and(reference_data[:, ref_dict['PlotId']] == PlotId,
                                          reference_data[:, ref_dict['TreeNumber']] == TreeNumber), ref_dict[
                               'y_tree']] = row2[0, 3]

cyl_dict = dict(x=0, y=1, z=2, nx=3, ny=4, nz=5, radius=6, CCI=7, branch_id=8, parent_branch_id=9,
                tree_id=10, segment_volume=11, segment_angle_to_horiz=12, height_above_dtm=13)

make_plots = 0

start_height = 0
stop_height = 38
increment = 0.1
slice_heights = np.around(np.linspace(start_height, stop_height, int(np.ceil((stop_height - start_height) / increment) + 1)), 1)
#
cylinders_per_plot_data = []
fsct_trees_per_plot_data = []
save_directory = 'E:/PFOlsen/FSCT_OUTPUTS/'
for PlotId in np.unique(reference_data[:, ref_dict['PlotId']]):
    circle_visualisation = np.zeros((0, 5))

    directory = 'E:/PFOlsen/PFOlsenPlots/' + point_clouds[names.index(PlotId)] + '_FSCT_output/'
    print(PlotId, point_clouds[names.index(PlotId)])
    cyls, headers = load_file(directory + 'cleaned_cyls.las', headers_of_interest=list(cyl_dict), silent=True)
    # print(cyls[0])
    # print(headers)
    tapers, CCI = extract_tapers_from_plot(cyls, cyl_dict, slice_heights)
    if make_plots:
        fig1 = plt.figure(figsize=(15, 2))
        fig1.show(False)
        ax1 = fig1.add_subplot(1, 1, 1)
        ax1.set_title("FSCT Stem Taper" + PlotId, fontsize=10)
        ax1.set_xlabel("Height above ground (m)")
        ax1.set_ylabel("Diameter (m)")
        # ax1.set_xlim([0, stop_height])
        ax1.set_ylim([0, np.max(tapers[1:, 1:])])
        for tree_id in np.unique(tapers[:, 0]):
            tree = tapers[tapers[:, 0] == tree_id].T[1:]
            x = np.atleast_2d(slice_heights).T[tree != 0]
            y = tree[tree != 0]
            ax1.plot(x, y, linewidth=0.5)

        fig1.savefig(directory + 'taper_plot.png', dpi=600, bbox_inches='tight', pad_inches=0.0)
        fig1.savefig(save_directory + 'TAPER_PLOTS/' + PlotId + '_taper_plot.png', dpi=600, bbox_inches='tight',
                     pad_inches=0.0)
        plt.close()

    pd.DataFrame(tapers, columns=['TreeId'] + list(slice_heights)).to_csv(directory + 'tapers.csv', index=False)
    pd.DataFrame(tapers, columns=['TreeId'] + list(slice_heights)).to_csv(
        save_directory + 'AUTOMATED_TAPER_OUTPUT/' + PlotId + '_tapers.csv', index=False)
    pd.DataFrame(CCI, columns=['TreeId'] + list(slice_heights)).to_csv(
        save_directory + 'AUTOMATED_TAPER_OUTPUT/' + PlotId + '_CCI.csv', index=False)
    df = pd.read_csv(directory + 'tree_data.csv')
    # print(df.shape)
    df['PlotID'] = PlotId
    fsct_trees_per_plot_data.append(df)

fsct_data_combined = pd.concat(fsct_trees_per_plot_data)
auto_headings = list(fsct_data_combined.columns.values)
auto_dict = {i: auto_headings.index(i) for i in auto_headings}
fsct_data_combined = np.array(fsct_data_combined)

sorted_trees_dict = {'PlotId'      : 0,
                     'tree_id_ref' : 1,
                     'x_ref'       : 2,
                     'y_ref'       : 3,
                     'height_ref'  : 4,
                     'dbh_ref'     : 5,
                     'vol_ref'     : 6,
                     'tree_id_auto': 7,
                     'x_auto'      : 8,
                     'y_auto'      : 9,
                     'height_auto' : 10,
                     'dbh_auto'    : 11,
                     'vol_auto'    : 12}

matched_data_all = np.zeros((0, 13))
matched_dataframe_combined = pd.DataFrame(columns=list(sorted_trees_dict))
for plot in np.unique(reference_data[:, ref_dict['PlotId']]):

    reference_plot = reference_data[reference_data[:, ref_dict['PlotId']] == plot]
    automatic_plot = fsct_data_combined[fsct_data_combined[:, auto_dict['PlotID']] == plot]

    # pd.DataFrame(reference_plot, columns=list(ref_dict)).to_csv(save_directory + 'reference_plot' + plot + '.csv')
    # pd.DataFrame(automatic_plot, columns=list(auto_dict)).to_csv(save_directory + 'automatic_plot' + plot + '.csv')
    # np.savetxt(save_directory + 'automatic_plot' + plot + '.csv', automatic_plot)
    if automatic_plot.shape[0] != 0:
        matched_data = get_nearest_tree(reference_plot, automatic_plot, max_search_radius=1, ref_dict=ref_dict,
                                        auto_dict=auto_dict, sorted_trees_dict=sorted_trees_dict)
        print(matched_data.shape, plot)
        valid_dbh = matched_data[:, sorted_trees_dict['dbh_auto']] != 0
        matched_dataframe = pd.DataFrame(matched_data, columns=list(sorted_trees_dict))
        matched_dataframe['PlotId'] = plot

        matched_dataframe.to_csv(save_directory + 'MATCHED_DATASETS/' + plot + 'matched_data.csv', index=False)

        if np.sum(valid_dbh) > 0:
            if make_plots:
                fig2 = plt.figure(figsize=(12, 12))
                fig2.show(False)
                fig2.suptitle("Plot " + plot, size=16)
                ax1 = fig2.add_subplot(2, 2, 1)
                ax1.set_title("Reference vs Automated DBH", fontsize=10)
                ax1.set_xlabel("Reference DBH (m)")
                ax1.set_ylabel("Automated DBH (m)")
                ax1.axis('equal')
                lim = np.max([np.max(matched_data[valid_dbh, sorted_trees_dict['dbh_ref']]),
                              np.max(matched_data[valid_dbh, sorted_trees_dict['dbh_auto']])]) + 0.1
                ax1.set_xlim([0, lim])
                ax1.set_ylim([0, lim])
                ax1.plot([0, lim], [0, lim], color='lightgrey', linewidth=0.5, )
                ax1.scatter(matched_data[valid_dbh, sorted_trees_dict['dbh_ref']],
                            matched_data[valid_dbh, sorted_trees_dict['dbh_auto']], s=30, marker='.')

                ax2 = fig2.add_subplot(2, 2, 2)
                ax2.set_title("DBH Error Histogram", fontsize=10)
                ax2.set_xlabel("DBH Error (m)")
                ax2.set_ylabel("Frequency")

                bin_width = 0.05
                bins = np.arange(-0.4 - 0.5 * bin_width, 0.4 + 0.5 * bin_width, bin_width)

                ax2.hist(matched_data[valid_dbh, sorted_trees_dict['dbh_ref']] - matched_data[
                    valid_dbh, sorted_trees_dict['dbh_auto']],
                         bins=bins,
                         range=(-0.5, 0.5),
                         linewidth=0.5,
                         edgecolor='black',
                         facecolor='green',
                         align='mid')

                ax3 = fig2.add_subplot(2, 2, 3)
                ax3.set_title("Reference vs Automated Height", fontsize=10)
                ax3.set_xlabel("Reference Height (m)")
                ax3.set_ylabel("Automated Height (m)")
                ax3.axis('equal')
                lim = np.max([np.max(matched_data[:, sorted_trees_dict['height_ref']]),
                              np.max(matched_data[:, sorted_trees_dict['height_auto']])]) + 0.1
                ax3.set_xlim([0, lim])
                ax3.set_ylim([0, lim])
                ax3.plot([0, lim], [0, lim], color='lightgrey', linewidth=0.5, )
                ax3.scatter(matched_data[:, sorted_trees_dict['height_ref']],
                            matched_data[:, sorted_trees_dict['height_auto']], s=30, marker='.')

                ax4 = fig2.add_subplot(2, 2, 4)
                ax4.set_title("Height Error Histogram", fontsize=10)
                ax4.set_xlabel("Height Error (m)")
                ax4.set_ylabel("Frequency")
                poslim = np.max(
                        matched_data[:, sorted_trees_dict['height_ref']] - matched_data[:,
                                                                           sorted_trees_dict['height_auto']])
                neglim = abs(np.min(
                        matched_data[:, sorted_trees_dict['height_ref']] - matched_data[:,
                                                                           sorted_trees_dict['height_auto']]))
                lim = np.round(np.max([neglim, poslim]) / 2) * 2
                bins = np.linspace(-lim - 2, lim + 2, int(np.ceil(2 * lim / 2)) + 4)

                ax4.hist(
                        matched_data[:, sorted_trees_dict['height_ref']] - matched_data[:,
                                                                           sorted_trees_dict['height_auto']],
                        bins=bins,
                        range=(-lim, lim),
                        linewidth=0.5,
                        edgecolor='black',
                        facecolor='green')
                fig2.savefig(save_directory + 'DBH_AND_HEIGHT_PLOTS/' + plot + '_DBH_and_height_plot.png', dpi=600,
                             bbox_inches='tight',
                             pad_inches=0.0)
                plt.close()
            matched_data_all = np.vstack((matched_data_all, matched_data))
        matched_dataframe_combined = matched_dataframe_combined.append(matched_dataframe)

matched_data = matched_data_all
valid_dbh = matched_data[:, sorted_trees_dict['dbh_auto']] != 0
valid_heights = matched_data[:, sorted_trees_dict['height_auto']] != 0

if make_plots:
    fig1 = plt.figure(figsize=(12, 12))
    fig1.suptitle("All Plots Combined", size=16)
    ax1 = fig1.add_subplot(2, 2, 1)
    ax1.set_title("Reference vs Automated DBH", fontsize=10)
    ax1.set_xlabel("Reference DBH (m)")
    ax1.set_ylabel("Automated DBH (m)")
    ax1.axis('equal')
    lim = np.max([np.max(matched_data[valid_dbh, sorted_trees_dict['dbh_ref']]),
                  np.max(matched_data[valid_dbh, sorted_trees_dict['dbh_auto']])]) + 0.1
    ax1.set_xlim([0, lim])
    ax1.set_ylim([0, lim])
    ax1.plot([0, lim], [0, lim], color='lightgrey', linewidth=0.5, )
    ax1.scatter(matched_data[valid_dbh, sorted_trees_dict['dbh_ref']],
                matched_data[valid_dbh, sorted_trees_dict['dbh_auto']], s=30, marker='.')

    ax2 = fig1.add_subplot(2, 2, 2)
    ax2.set_title("DBH Error Histogram", fontsize=10)
    ax2.set_xlabel("DBH Error (m)")
    ax2.set_ylabel("Frequency")
    bin_width = 0.05
    bins = np.arange(-0.4 - 0.5 * bin_width, 0.4 + 0.5 * bin_width, bin_width)

    # bins = np.linspace(-lim-0.05,lim+0.05,int(np.ceil(2*lim/0.01))+4)
    # print(bins)

    ax2.hist(
            matched_data[valid_dbh, sorted_trees_dict['dbh_auto']] - matched_data[
                valid_dbh, sorted_trees_dict['dbh_ref']],
            bins=bins,
            range=(-0.5, 0.5),
            linewidth=0.5,
            edgecolor='black',
            facecolor='green',
            align='mid')

    ax3 = fig1.add_subplot(2, 2, 3)
    ax3.set_title("Reference vs Automated Height", fontsize=10)
    ax3.set_xlabel("Reference Height (m)")
    ax3.set_ylabel("Automated Height (m)")
    ax3.axis('equal')
    lim = np.max([np.max(matched_data[valid_heights, sorted_trees_dict['height_ref']]),
                  np.max(matched_data[valid_heights, sorted_trees_dict['height_auto']])]) + 0.1
    ax3.set_xlim([0, lim])
    ax3.set_ylim([0, lim])
    ax3.plot([0, lim], [0, lim], color='lightgrey', linewidth=0.5, )
    ax3.scatter(matched_data[valid_heights, sorted_trees_dict['height_ref']],
                matched_data[valid_heights, sorted_trees_dict['height_auto']], s=30, marker='.')

    ax4 = fig1.add_subplot(2, 2, 4)
    ax4.set_title("Height Error Histogram", fontsize=10)
    ax4.set_xlabel("Height Error (m)")
    ax4.set_ylabel("Frequency")
    poslim = np.max(matched_data[valid_heights, sorted_trees_dict['height_ref']] - matched_data[
        valid_heights, sorted_trees_dict['height_auto']])
    neglim = abs(np.min(matched_data[valid_heights, sorted_trees_dict['height_ref']] - matched_data[
        valid_heights, sorted_trees_dict['height_auto']]))
    lim = np.round(np.max([neglim, poslim]) / 2) * 2
    bins = np.linspace(-lim - 2, lim + 2, int(np.ceil(2 * lim / 2)) + 4)

    ax4.hist(matched_data[valid_heights, sorted_trees_dict['height_auto']] - matched_data[
        valid_heights, sorted_trees_dict['height_ref']],
             bins=bins,
             range=(-lim, lim),
             linewidth=0.5,
             edgecolor='black',
             facecolor='green')

    fig1.savefig(save_directory + 'DBH_AND_HEIGHT_PLOTS/' + 'COMBINED_DBH_and_height_plot.png', dpi=600,
                 bbox_inches='tight', pad_inches=0.0)
    plt.close()

matched_dataframe_combined.to_csv(save_directory + 'MATCHED_DATASETS/' + 'matched_data_combined.csv', index=False)

# matched_dataframe_combined = pd.read_csv(save_directory + 'MATCHED_DATASETS/' + 'matched_data_combined.csv', index=False)
# for PlotId in np.unique(matched_dataframe_combined[:, ref_dict['PlotId']]):

# [plot_id, tree_id, measurement_height, reference_diameter, auto_diameter, autoCCI]
taper_comparison_array = np.zeros((0, 8))
print(reference_headings)
print(reference_data)
ref_dataframe = pd.DataFrame(reference_data, columns=reference_headings)

ref_measurement_heights = ['0.1', '0.3', '0.8', '1.3', '2.0', '3.5', '5.0', '6.5', '8.0', '9.5', '11.0', '12.5', '14.0', '15.5', '17.0', '18.5', '20.0', '21.5', '23.0', '24.5',
                           '26.0', '27.5', '29.0', '30.5', '32.0', '33.5', '35.0', '36.5', '38.0']

for PlotId in np.unique(matched_dataframe_combined['PlotId']):
    # print(PlotId)
    auto_taper_df = pd.read_csv(save_directory + 'AUTOMATED_TAPER_OUTPUT/' + PlotId + '_tapers.csv')
    auto_CCI_df = pd.read_csv(save_directory + 'AUTOMATED_TAPER_OUTPUT/' + PlotId + '_CCI.csv')
    ref_taper_df = ref_dataframe[ref_dataframe['PlotId'] == PlotId]
    matched_circle_visualisation = np.zeros((0, 5))

    for tree_id_ref, tree_id_auto in zip(
            np.unique(matched_dataframe_combined[matched_dataframe_combined['PlotId'] == PlotId]['tree_id_ref']),
            np.unique(matched_dataframe_combined[matched_dataframe_combined['PlotId'] == PlotId]['tree_id_auto'])):
        # print(tree_id_ref, tree_id_auto)
        current_tree_auto_tapers = auto_taper_df[auto_taper_df['TreeId'] == tree_id_auto]
        current_tree_auto_CCI = auto_CCI_df[auto_CCI_df['TreeId'] == tree_id_auto]
        current_tree_ref_tapers = ref_taper_df[ref_taper_df['TreeNumber'] == tree_id_ref]
        x = float(current_tree_ref_tapers['x_tree'])
        y = float(current_tree_ref_tapers['y_tree'])
        for measurement_height in ref_measurement_heights:
            try:
                ref = float(current_tree_ref_tapers[measurement_height])
            except KeyError:
                ref = 0

            if np.isnan(ref):
                ref = 0
            auto = current_tree_auto_tapers[measurement_height]
            autoCCI = current_tree_auto_CCI[measurement_height]
            # if np.shape(auto)[0] == 0:
            #     auto = 0
            #     autoCCI = 0
            # else:
            #     print("WTF>>??????", auto, autoCCI)
            try:
                taper_comparison_array = np.vstack((taper_comparison_array, np.array(
                        [[x, y, float(measurement_height), float(PlotId[5:]), float(tree_id_ref), ref / 1000., float(auto), float(autoCCI)]])))
            except TypeError:
                # print(auto)
                # print(autoCCI)
                # print(measurement_height)
                continue
            # except KeyError:
            #     print(measurement_height)
            #     # print(current_tree_auto_tapers.columns.values)
            #     auto = np.array(current_tree_auto_tapers[measurement_height])
            #     autoCCI = np.array(current_tree_auto_CCI[measurement_height])
            z = float(measurement_height)
            r = 0.5 * ref / 1000.
            label = 0
            if r != 0:
                matched_circle_visualisation = np.vstack((matched_circle_visualisation,
                                                          create_3d_circles_as_points_flat(x=x, y=y, z=z, r=r, label=label)))
                r = 0.5 * float(auto)
                label = 1
                matched_circle_visualisation = np.vstack((matched_circle_visualisation,
                                                          create_3d_circles_as_points_flat(x=x, y=y, z=z, r=r, label=label)))

    np.savetxt(save_directory + 'CIRCLE_VISUALISATIONS/' + PlotId + '_matched_circle_visualisation.csv', matched_circle_visualisation)


