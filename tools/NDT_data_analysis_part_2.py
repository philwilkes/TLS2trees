import numpy as np
import pandas as pd
import simplekml
import utm
import glob
from scipy import spatial
from copy import deepcopy
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.patches import Circle, PathPatch
from matplotlib import cm

plt.ioff()

"""
Compares Photogrammetric vs TLS results.
"""


def convert_coords_to_lat_long(easting, northing, point_name=None):
    lat, lon = utm.to_latlon(easting=easting,
                             northing=northing,
                             zone_number=50,
                             zone_letter=None,
                             northern=False,
                             strict=None)
    return lat, lon, point_name


def get_nearest_tree(automatic_dataset_1, automatic_dataset_2, max_search_radius):
    tree_id_auto1 = automatic_dataset_1['treeNo']
    x_auto1 = automatic_dataset_1['x_tree_base']
    y_auto1 = automatic_dataset_1['y_tree_base']
    height_auto1 = automatic_dataset_1['Height']
    dbh_auto1 = automatic_dataset_1['DBH']
    vol_auto1 = automatic_dataset_1['Volume']

    tree_id_auto2 = automatic_dataset_2['treeNo']
    x_auto2 = automatic_dataset_2['x_tree_base']
    y_auto2 = automatic_dataset_2['y_tree_base']
    height_auto2 = automatic_dataset_2['Height']
    dbh_auto2 = automatic_dataset_2['DBH']
    vol_auto2 = automatic_dataset_2['Volume']

    auto_data_array1 = np.vstack((x_auto1, y_auto1, tree_id_auto1, height_auto1, dbh_auto1, vol_auto1)).T
    auto_data_array2 = np.vstack((x_auto2, y_auto2, tree_id_auto2, height_auto2, dbh_auto2, vol_auto2)).T

    auto_data_array_unsorted = deepcopy(auto_data_array2)
    sorted_trees_dict = {'tree_id_auto1' : 0,
                         'x_auto1'       : 1,
                         'y_auto1'       : 2,
                         'height_auto1'  : 3,
                         'dbh_auto1'     : 4,
                          'vol_auto1'    : 5,
                          'tree_id_auto2': 6,
                          'x_auto2'      : 7,
                          'y_auto2'      : 8,
                          'height_auto2' : 9,
                         'dbh_auto2'    : 10,
                         'vol_auto2'    : 11}

    sorted_trees_array = np.zeros((0, 12))
    species_column = []
    for tree in auto_data_array1:
        # print(tree)
        if ~np.isnan(tree[4]):
            best_tree_id = 0
            kdtree = spatial.cKDTree(auto_data_array_unsorted[:, :2])
            results = kdtree.query_ball_point(tree[:2], r=max_search_radius)
            candidate_tree_matches = auto_data_array_unsorted[results]
            candidate_tree_matches = candidate_tree_matches[
                candidate_tree_matches[:, 3] > 5]  # trees greater than 5 m tall
            candidate_tree_matches = candidate_tree_matches[candidate_tree_matches[:, 4] > 0]  # trees with a valid DBH
            best_match = np.zeros((1,))
            if candidate_tree_matches.shape[0] > 0:
                dbh_diff = candidate_tree_matches[:, 4] - tree[4]
                best_match = candidate_tree_matches[np.argmin(np.abs(dbh_diff))]
                # print(tree[4],best_match[4],tree[4]-best_match[4])
                best_tree_id = best_match[2]
                auto_data_array_unsorted = auto_data_array_unsorted[auto_data_array_unsorted[:, 2] != best_tree_id]
            sorted_tree = np.zeros((1, 12))
            if best_match.shape[0] > 1:
                sorted_tree[:, sorted_trees_dict['tree_id_auto1']] = tree[2]
                sorted_tree[:, sorted_trees_dict['x_auto1']] = tree[0]
                sorted_tree[:, sorted_trees_dict['y_auto1']] = tree[1]
                sorted_tree[:, sorted_trees_dict['height_auto1']] = tree[3]
                sorted_tree[:, sorted_trees_dict['dbh_auto1']] = tree[4]
                sorted_tree[:, sorted_trees_dict['vol_auto1']] = tree[5]
                sorted_tree[:, sorted_trees_dict['tree_id_auto2']] = best_match[2]
                sorted_tree[:, sorted_trees_dict['x_auto2']] = best_match[0]
                sorted_tree[:, sorted_trees_dict['y_auto2']] = best_match[1]
                sorted_tree[:, sorted_trees_dict['height_auto2']] = best_match[3]
                sorted_tree[:, sorted_trees_dict['dbh_auto2']] = best_match[4]
                sorted_tree[:, sorted_trees_dict['vol_auto2']] = best_match[5]
                sorted_trees_array = np.vstack((sorted_trees_array, sorted_tree))
    return sorted_trees_array, species_column


auto_plots1 = ['Leach P61',
               'Leach P111',
               'Denham P257',
               'Denham P264'
               ]

auto_plots2 = ['Leach_P61_TLS',
               'Leach_P111_TLS',
               'Denham_P257_TLS',
               'Denham_P264_TLS'
               ]

offsets1 = [[-1.257, -0.426],
            [-1.47, -1.75],
            [-0.17, -3.015],
            [-1.16, -1.32]
            ]

offsets2 = [[0, 0],
            [0, 0],
            [0, 0],
            [0, 0]
            ]


for auto_plot1, auto_plot2, offset1, offset2 in zip(auto_plots1, auto_plots2, offsets1, offsets2):
    # try:
    directory1 = glob.glob('C:/Users/seank/Documents/NDT Project/Western Australia/*' + auto_plot1 + '*/')[0]
    automatic_plot_data1 = pd.read_csv(directory1 + 'tree_data.csv')
    automatic_plot_data1['x_tree_base'] = automatic_plot_data1['x_tree_base'] + offset1[0]
    automatic_plot_data1['y_tree_base'] = automatic_plot_data1['y_tree_base'] + offset1[1]

    directory2 = glob.glob('C:/Users/seank/Documents/NDT Project/Western Australia/*' + auto_plot2 + '*/')[0]
    automatic_plot_data2 = pd.read_csv(directory2 + 'tree_data.csv')
    automatic_plot_data2['x_tree_base'] = automatic_plot_data2['x_tree_base'] + offset2[0]
    automatic_plot_data2['y_tree_base'] = automatic_plot_data2['y_tree_base'] + offset2[1]

    print('\n', auto_plot1, auto_plot2)

    matched_data, species_column = get_nearest_tree(automatic_plot_data1, automatic_plot_data2, max_search_radius=2)

    sorted_trees_dict = {'tree_id_auto1': 0,
                         'x_auto1'      : 1,
                         'y_auto1'      : 2,
                         'height_auto1' : 3,
                         'dbh_auto1'    : 4,
                         'vol_auto1'    : 5,
                         'tree_id_auto2': 6,
                         'x_auto2'      : 7,
                         'y_auto2'      : 8,
                         'height_auto2' : 9,
                         'dbh_auto2'    : 10,
                         'vol_auto2'    : 11}

    fig1 = plt.figure(figsize=(12, 12))
    fig1.suptitle("Plot " + auto_plot1, size=16)
    ax1 = fig1.add_subplot(2, 2, 1)
    ax1.set_title("Reference vs Automated DBH", fontsize=10)
    ax1.set_xlabel("Reference DBH (m)")
    ax1.set_ylabel("Automated DBH (m)")
    ax1.axis('equal')
    lim = np.max([np.max(matched_data[:, sorted_trees_dict['dbh_auto1']]),
                  np.max(matched_data[:, sorted_trees_dict['dbh_auto2']])]) + 0.1
    ax1.set_xlim([0, lim])
    ax1.set_ylim([0, lim])
    ax1.plot([0, lim], [0, lim], color='lightgrey', linewidth=0.5, )
    ax1.scatter(matched_data[:, sorted_trees_dict['dbh_auto1']], matched_data[:, sorted_trees_dict['dbh_auto2']], s=30,
                marker='.')

    ax2 = fig1.add_subplot(2, 2, 2)
    ax2.set_title("DBH Error Histogram", fontsize=10)
    ax2.set_xlabel("DBH Error (m)")
    ax2.set_ylabel("Frequency")
    poslim = np.max(matched_data[:, sorted_trees_dict['dbh_auto1']] - matched_data[:, sorted_trees_dict['dbh_auto2']])
    neglim = abs(np.min(matched_data[:, sorted_trees_dict['dbh_auto1']] - matched_data[:, sorted_trees_dict['dbh_auto2']]))
    lim = round(np.max([neglim, poslim]) * 10) / 10  # round to nearest 0.1

    bins = np.linspace(-lim - 0.05, lim + 0.05, int(np.ceil(2 * lim / 0.1)) + 2)

    ax2.hist(matched_data[:, sorted_trees_dict['dbh_auto1']] - matched_data[:, sorted_trees_dict['dbh_auto2']],
             bins=bins,
             range=(-lim, lim),
             linewidth=0.5,
             edgecolor='black',
             facecolor='green',
             align='mid')

    ax3 = fig1.add_subplot(2, 2, 3)
    ax3.set_title("Reference vs Automated Height", fontsize=10)
    ax3.set_xlabel("Reference Height (m)")
    ax3.set_ylabel("Automated Height (m)")
    ax3.axis('equal')
    lim = np.max([np.max(matched_data[:, sorted_trees_dict['height_auto1']]),
                  np.max(matched_data[:, sorted_trees_dict['height_auto2']])]) + 0.1
    ax3.set_xlim([0, lim])
    ax3.set_ylim([0, lim])
    ax3.plot([0, lim], [0, lim], color='lightgrey', linewidth=0.5, )
    ax3.scatter(matched_data[:, sorted_trees_dict['height_auto1']], matched_data[:, sorted_trees_dict['height_auto2']],
                s=30, marker='.')

    ax4 = fig1.add_subplot(2, 2, 4)
    ax4.set_title("Height Error Histogram", fontsize=10)
    ax4.set_xlabel("Height Error (m)")
    ax4.set_ylabel("Frequency")
    poslim = np.max(
        matched_data[:, sorted_trees_dict['height_auto1']] - matched_data[:, sorted_trees_dict['height_auto2']])
    neglim = abs(
        np.min(matched_data[:, sorted_trees_dict['height_auto1']] - matched_data[:, sorted_trees_dict['height_auto2']]))
    lim = np.round(np.max([neglim, poslim]) / 2) * 2
    bins = np.linspace(-lim - 2, lim + 2, int(np.ceil(2 * lim / 2)) + 4)

    ax4.hist(matched_data[:, sorted_trees_dict['height_auto1']] - matched_data[:, sorted_trees_dict['height_auto2']],
             bins=bins,
             range=(-lim, lim),
             linewidth=0.5,
             edgecolor='black',
             facecolor='green')

    fig1.show(False)
    # fig1.savefig(directory + plot + '_DBH_and_height_plot.png', dpi=600, bbox_inches='tight', pad_inches=0.0)
    fig1.savefig('C:/Users/seank/Documents/NDT Project/Western Australia/NDT_DATA/' + auto_plot1 + '_DBH_and_height_plot_auto_vs_auto.png', dpi=600, bbox_inches='tight', pad_inches=0.0)

    # pd.DataFrame(matched_data).to_csv(directory+'matched_tree_data.csv',header=[i for i in sorted_trees_dict],index=None,sep=',')
    pd.DataFrame(matched_data).to_csv(
        'C:/Users/seank/Documents/NDT Project/Western Australia/NDT_DATA/' + auto_plot1 + '_matched_tree_data_auto_vs_auto.csv',
        header=[i for i in sorted_trees_dict], index=None, sep=',')

    # kml = simplekml.Kml()
    # for i in matched_data:
    #     tree_lat, tree_lon, tree_name = convert_coords_to_lat_long(i[2], i[3], i[1])
    #
    #     # nearest_tree_in_auto_data =
    #
    #     description = 'DBH: ' + str(i[4]) + '\nHeight: ' + str(i[5]) + '\nEst_Vol_m3: ' + str(i[6])
    #     kml.newpoint(name=str(tree_name), coords=[(tree_lon, tree_lat)], description=description)
    # kml.save('C:/Users/seank/Documents/NDT Project/Western Australia/NDT_DATA/' + str(auto_plot) + '.kml')
    # except FileNotFoundError:
    #     None
