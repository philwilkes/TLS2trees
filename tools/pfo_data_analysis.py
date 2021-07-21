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
from tools import load_file
plt.ioff()


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
    height_auto = automatic_dataset[:, auto_dict['Height']]
    dbh_auto = automatic_dataset[:, auto_dict['DBH']]
    vol_auto = automatic_dataset[:, auto_dict['Volume']]

    reference_data_array = np.vstack((x_ref, y_ref, tree_id_ref, height_ref, dbh_ref, vol_ref)).T
    auto_data_array = np.vstack((x_auto, y_auto, tree_id_auto, height_auto, dbh_auto, vol_auto)).T
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


def convert_coords_to_lat_long(easting, northing, point_name=None):
    lat, lon = utm.to_latlon(easting=easting,
                             northing=northing,
                             zone_number=50,
                             zone_letter=None,
                             northern=False,
                             strict=None)
    return lat, lon, point_name


point_clouds = ['T1_class',
                'T02_class',
                'T3_class',
                'T4_class',
                'T05_class',
                'T6_class',
                'T7_class',
                'T8_class',
                'T9_class',
                'T10_class',

                'T11_class',
                'T12_class',
                'T13_class',
                'T14_class',
                'T15_class',
                'T16_class',
                'T017_class',
                'T18_class',
                'T19_class',
                'T20_class',

                'T21_class',
                'T22_class',
                'T23_class',
                'T25_class',
                'T26_class',
                'T27_class',
                'T28_class',
                'TAPER29_class',
                'TAPER30_class',

                'TAPER31_class',
                'TAPER32_class',
                'TAPER33_class',
                'TAPER34_class',
                'TAPER35_class',
                'TAPER36_class',
                'TAPER37_class',
                'TAPER38_class',
                'TAPER39_class',
                'TAPER40_class',

                'TAPER41_class',
                'TAPER42_class',
                'TAPER43_class',
                'TAPER44_class',
                'TAPER45_class',
                'TAPER46_class',
                'TAPER47_class',
                'TAPER48_class',
                'TAPER49_class',
                'TAPER50_class',
                ]

names = ['TAPER01',
         'TAPER02',
         'TAPER03',
         'TAPER04',
         'TAPER05',
         'TAPER06',
         'TAPER07',
         'TAPER08',
         'TAPER09',
         'TAPER10',

         'TAPER11',
         'TAPER12',
         'TAPER13',
         'TAPER14',
         'TAPER15',
         'TAPER16',
         'TAPER17',
         'TAPER18',
         'TAPER19',
         'TAPER20',

         'TAPER21',
         'TAPER22',
         'TAPER23',
         'TAPER25',
         'TAPER26',
         'TAPER27',
         'TAPER28',
         'TAPER29',
         'TAPER30',

         'TAPER31',
         'TAPER32',
         'TAPER33',
         'TAPER34',
         'TAPER35',
         'TAPER36',
         'TAPER37',
         'TAPER38',
         'TAPER39',
         'TAPER40',

         'TAPER41',
         'TAPER42',
         'TAPER43',
         'TAPER44',
         'TAPER45',
         'TAPER46',
         'TAPER47',
         'TAPER48',
         'TAPER49',
         'TAPER50']

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
    for TreeNumber in np.unique(reference_data[:, ref_dict['TreeNumber']]):
        # print(PlotId,TreeNumber)
        row = WA_tree_locations[
            np.logical_and(WA_tree_locations[:, 0] == PlotId, WA_tree_locations[:, 1] == TreeNumber)]
        if row.shape[0] > 0:
            reference_data[np.logical_and(reference_data[:, ref_dict['PlotId']] == PlotId,
                                          reference_data[:, ref_dict['TreeNumber']] == TreeNumber), ref_dict[
                               'x_tree']] = row[0, 2]
            reference_data[np.logical_and(reference_data[:, ref_dict['PlotId']] == PlotId,
                                          reference_data[:, ref_dict['TreeNumber']] == TreeNumber), ref_dict[
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

cylinders_per_plot_data = []
fsct_trees_per_plot_data = []
for PlotId in np.unique(reference_data[:, ref_dict['PlotId']]):
    directory = 'E:/PFOlsen/PFOlsenPlots/' + point_clouds[names.index(PlotId)] + '_FSCT_output/'
    # print(PlotId, names.index(PlotId), point_clouds[names.index(PlotId)])

    cyls, headers = load_file(directory + 'cleaned_cyls.las', headers_of_interest=list(cyl_dict))
    cylinders_per_plot_data.append(cyls)

    df = pd.read_csv(directory + 'tree_data.csv')
    print(df.shape)
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

missing = [30, 35, 36, 39, 50]
# 39 has one tree... the rest have plenty

for plot in np.unique(reference_data[:, ref_dict['PlotId']]):
    save_directory = 'E:/PFOlsen/FSCT_OUTPUTS/'
    reference_plot = reference_data[reference_data[:, ref_dict['PlotId']] == plot]
    automatic_plot = fsct_data_combined[fsct_data_combined[:, auto_dict['PlotID']] == plot]
    if automatic_plot.shape[0] != 0:
        # print(reference_plot.shape,automatic_plot.shape)
        matched_data = get_nearest_tree(reference_plot, automatic_plot, max_search_radius=2, ref_dict=ref_dict,
                                        auto_dict=auto_dict, sorted_trees_dict=sorted_trees_dict)
        valid_dbh = matched_data[:, sorted_trees_dict['dbh_auto']] != 0
        if np.sum(valid_dbh) > 0:
            fig1 = plt.figure(figsize=(12, 12))
            fig1.show(False)
            fig1.suptitle("Plot " + plot, size=16)
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
            poslim = np.max(matched_data[valid_dbh, sorted_trees_dict['dbh_ref']] - matched_data[
                valid_dbh, sorted_trees_dict['dbh_auto']])
            neglim = abs(np.min(matched_data[valid_dbh, sorted_trees_dict['dbh_ref']] - matched_data[
                valid_dbh, sorted_trees_dict['dbh_auto']]))
            lim = np.around(np.max([neglim, poslim]), 3)

            # bins = np.linspace(-lim-0.05,lim+0.05,int(np.ceil(2*lim/0.01))+4)
            # print(bins)

            ax2.hist(matched_data[valid_dbh, sorted_trees_dict['dbh_ref']] - matched_data[
                valid_dbh, sorted_trees_dict['dbh_auto']],
                     # bins=bins,
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
            lim = np.max([np.max(matched_data[:, sorted_trees_dict['height_ref']]),
                          np.max(matched_data[:, sorted_trees_dict['height_auto']])]) + 0.1
            ax3.set_xlim([0, lim])
            ax3.set_ylim([0, lim])
            ax3.plot([0, lim], [0, lim], color='lightgrey', linewidth=0.5, )
            ax3.scatter(matched_data[:, sorted_trees_dict['height_ref']],
                        matched_data[:, sorted_trees_dict['height_auto']], s=30, marker='.')

            ax4 = fig1.add_subplot(2, 2, 4)
            ax4.set_title("Height Error Histogram", fontsize=10)
            ax4.set_xlabel("Height Error (m)")
            ax4.set_ylabel("Frequency")
            poslim = np.max(
                matched_data[:, sorted_trees_dict['height_ref']] - matched_data[:, sorted_trees_dict['height_auto']])
            neglim = abs(np.min(
                matched_data[:, sorted_trees_dict['height_ref']] - matched_data[:, sorted_trees_dict['height_auto']]))
            lim = np.round(np.max([neglim, poslim]) / 2) * 2
            bins = np.linspace(-lim - 2, lim + 2, int(np.ceil(2 * lim / 2)) + 4)

            ax4.hist(
                matched_data[:, sorted_trees_dict['height_ref']] - matched_data[:, sorted_trees_dict['height_auto']],
                bins=bins,
                range=(-lim, lim),
                linewidth=0.5,
                edgecolor='black',
                facecolor='green')

            fig1.savefig(save_directory + plot + '_DBH_and_height_plot.png', dpi=600, bbox_inches='tight',
                         pad_inches=0.0)
            plt.close()

            matched_data_all = np.vstack((matched_data_all, matched_data))

matched_data = matched_data_all
valid_dbh = matched_data[:, sorted_trees_dict['dbh_auto']] != 0
valid_heights = matched_data[:, sorted_trees_dict['height_auto']] != 0

if 1:
    fig1 = plt.figure(figsize=(12, 12))
    fig1.suptitle("Plot " + plot, size=16)
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
    poslim = np.max(
        matched_data[valid_dbh, sorted_trees_dict['dbh_ref']] - matched_data[valid_dbh, sorted_trees_dict['dbh_auto']])
    neglim = abs(np.min(
        matched_data[valid_dbh, sorted_trees_dict['dbh_ref']] - matched_data[valid_dbh, sorted_trees_dict['dbh_auto']]))
    lim = np.around(np.max([neglim, poslim]), 3)

    # bins = np.linspace(-lim-0.05,lim+0.05,int(np.ceil(2*lim/0.01))+4)
    # print(bins)

    ax2.hist(
        matched_data[valid_dbh, sorted_trees_dict['dbh_ref']] - matched_data[valid_dbh, sorted_trees_dict['dbh_auto']],
        # bins=bins,
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

    ax4.hist(matched_data[valid_heights, sorted_trees_dict['height_ref']] - matched_data[
        valid_heights, sorted_trees_dict['height_auto']],
             bins=bins,
             range=(-lim, lim),
             linewidth=0.5,
             edgecolor='black',
             facecolor='green')

# for tree in matched_data:
#     None

