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


def convert_coords_to_lat_long(easting, northing, point_name=None):
    lat, lon = utm.to_latlon(easting=easting,
                             northing=northing,
                             zone_number=50,
                             zone_letter=None,
                             northern=False,
                             strict=None)
    return lat, lon, point_name


reference_data = pd.read_csv('C:/Users/seank/Documents/NDT Project/Western Australia/NDT_DATA/all trees.csv')

PlotID = np.array(reference_data['PlotID_'])
tree_id = np.array(reference_data['treeNo'])
base_x = np.array(reference_data['x_tree'])
base_y = np.array(reference_data['y_tree'])
base_z = np.zeros((PlotID.shape[0]))

dbh = np.array(reference_data['DBHOB__cm_']) / 100
height = np.array(reference_data['Tree_Height__m_'])
bole_vol_1 = np.array(reference_data['Est__Gross_Bole_Vol_m3'])
bole_vol_2 = np.array(reference_data['Est__Gross_Bole__v2__Vol_m3'])

array = np.vstack((PlotID, tree_id, base_x, base_y, base_z, dbh, height, bole_vol_1, bole_vol_2)).T

plots_to_compare = ['Leach P61',
                    'Leach P111',
                    'Denham P257',
                    'Denham P264',
                    'Leach_P61_TLS',
                    'Leach_P111_TLS',
                    'Denham_P257_TLS',
                    'Denham_P264_TLS']

ref_names_plots_to_compare = ['P61',
                              'P111',
                              'P257',
                              'P264',
                              'P61',
                              'P111',
                              'P257',
                              'P264']

offsets = [[1.82, 1.384],
           [1.02, 0.24],
           [1.065, 1.256],
           [0.554, 3.561],
           [0, 0],
           [0, 0],
           [0, 0],
           [0, 0]]





def get_nearest_tree(reference_dataset, automatic_dataset, max_search_radius):
    tree_id_ref = reference_dataset['treeNo']
    x_ref = reference_dataset['x_tree']
    y_ref = reference_dataset['y_tree']
    height_ref = reference_dataset['Tree_Height__m_']
    dbh_ref = reference_dataset['DBHOB__cm_'] / 100
    vol1_ref = reference_dataset['Est__Gross_Bole_Vol_m3']
    vol2_ref = reference_dataset['Est__Gross_Bole__v2__Vol_m3']
    species = reference_dataset['Spcs_Label1']
    tree_id_auto = automatic_dataset['treeNo']
    x_auto = automatic_dataset['x_tree_base']
    y_auto = automatic_dataset['y_tree_base']
    height_auto = automatic_dataset['Height']
    dbh_auto = automatic_dataset['DBH']
    vol_auto = automatic_dataset['Volume']

    reference_data_array = np.vstack((x_ref, y_ref, tree_id_ref, height_ref, dbh_ref, vol1_ref, vol2_ref, species)).T
    auto_data_array = np.vstack((x_auto, y_auto, tree_id_auto, height_auto, dbh_auto, vol_auto)).T

    auto_data_array_unsorted = deepcopy(auto_data_array)
    sorted_trees_dict = {'tree_id_ref' : 0,
                         'x_ref'       : 1,
                         'y_ref'       : 2,
                         'height_ref'  : 3,
                         'dbh_ref'     : 4,
                         'vol1_ref'    : 5,
                         'vol2_ref'    : 6,
                         'tree_id_auto': 7,
                         'x_auto'      : 8,
                         'y_auto'      : 9,
                         'height_auto' : 10,
                         'dbh_auto'    : 11,
                         'vol_auto'    : 12,
                         'species'     : 13
                         }
    sorted_trees_array = np.zeros((0, 13))
    species_column = []
    for tree in reference_data_array:
        # print(tree)
        if ~np.isnan(tree[4]):
            best_tree_id = 0
            ref_kdtree = spatial.cKDTree(auto_data_array_unsorted[:, :2])
            results = ref_kdtree.query_ball_point(tree[:2], r=max_search_radius)
            candidate_tree_matches = auto_data_array_unsorted[results]
            candidate_tree_matches = candidate_tree_matches[
                candidate_tree_matches[:, 3] > 5]  # trees greater than 5 m tall
            candidate_tree_matches = candidate_tree_matches[candidate_tree_matches[:, 4] > 0]  # trees with a valid DBH
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
            sorted_tree[:, sorted_trees_dict['vol1_ref']] = tree[5]
            sorted_tree[:, sorted_trees_dict['vol2_ref']] = tree[6]
            species_column.append(tree[7])

            sorted_tree[:, sorted_trees_dict['tree_id_auto']] = best_match[2]
            sorted_tree[:, sorted_trees_dict['x_auto']] = best_match[0]
            sorted_tree[:, sorted_trees_dict['y_auto']] = best_match[1]
            sorted_tree[:, sorted_trees_dict['height_auto']] = best_match[3]
            sorted_tree[:, sorted_trees_dict['dbh_auto']] = best_match[4]
            sorted_tree[:, sorted_trees_dict['vol_auto']] = best_match[5]
            sorted_trees_array = np.vstack((sorted_trees_array, sorted_tree))
    return sorted_trees_array, species_column


for auto_plot, plot, offset in zip(plots_to_compare, ref_names_plots_to_compare, offsets):
    try:
        directory = glob.glob('C:/Users/seank/Documents/NDT Project/Western Australia/*' + auto_plot + '*/')[0]
        print(directory, offset)
        file = directory + 'tree_data.csv'
        automatic_plot_data = pd.read_csv(file)
        print(automatic_plot_data['x_tree_base'][0])
        automatic_plot_data['x_tree_base'] = automatic_plot_data['x_tree_base'] + offset[0]
        print(automatic_plot_data['x_tree_base'][0])
        automatic_plot_data['y_tree_base'] = automatic_plot_data['y_tree_base'] + offset[1]
        print('\n', plot, auto_plot)
        plot_array = array[array[:, 0] == plot]

        matched_data, species_column = get_nearest_tree(reference_data[reference_data['PlotID_'] == plot],
                                                        automatic_plot_data, max_search_radius=2)

        sorted_trees_dict = {'tree_id_ref' : 0,
                             'x_ref'       : 1,
                             'y_ref'       : 2,
                             'height_ref'  : 3,
                             'dbh_ref'     : 4,
                             'vol1_ref'    : 5,
                             'vol2_ref'    : 6,
                             'tree_id_auto': 7,
                             'x_auto'      : 8,
                             'y_auto'      : 9,
                             'height_auto' : 10,
                             'dbh_auto'    : 11,
                             'vol_auto'    : 12,
                             'species'     : 13
                             }

        fig1 = plt.figure(figsize=(12, 12))
        fig1.suptitle("Plot " + plot, size=16)
        ax1 = fig1.add_subplot(2, 2, 1)
        ax1.set_title("Reference vs Automated DBH", fontsize=10)
        ax1.set_xlabel("Reference DBH (m)")
        ax1.set_ylabel("Automated DBH (m)")
        ax1.axis('equal')
        lim = np.max([np.max(matched_data[:, sorted_trees_dict['dbh_ref']]),
                      np.max(matched_data[:, sorted_trees_dict['dbh_auto']])]) + 0.1
        ax1.set_xlim([0, lim])
        ax1.set_ylim([0, lim])
        ax1.plot([0, lim], [0, lim], color='lightgrey', linewidth=0.5, )
        ax1.scatter(matched_data[:, sorted_trees_dict['dbh_ref']], matched_data[:, sorted_trees_dict['dbh_auto']], s=30,
                    marker='.')

        ax2 = fig1.add_subplot(2, 2, 2)
        ax2.set_title("DBH Error Histogram", fontsize=10)
        ax2.set_xlabel("DBH Error (m)")
        ax2.set_ylabel("Frequency")
        poslim = np.max(matched_data[:, sorted_trees_dict['dbh_ref']] - matched_data[:, sorted_trees_dict['dbh_auto']])
        neglim = abs(np.min(matched_data[:, sorted_trees_dict['dbh_ref']] - matched_data[:, sorted_trees_dict['dbh_auto']]))
        lim = round(np.max([neglim, poslim]) * 10) / 10  # round to nearest 0.1

        bins = np.linspace(-lim - 0.05, lim + 0.05, int(np.ceil(2 * lim / 0.1)) + 2)

        ax2.hist(matched_data[:, sorted_trees_dict['dbh_ref']] - matched_data[:, sorted_trees_dict['dbh_auto']],
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
        lim = np.max([np.max(matched_data[:, sorted_trees_dict['height_ref']]),
                      np.max(matched_data[:, sorted_trees_dict['height_auto']])]) + 0.1
        ax3.set_xlim([0, lim])
        ax3.set_ylim([0, lim])
        ax3.plot([0, lim], [0, lim], color='lightgrey', linewidth=0.5, )
        ax3.scatter(matched_data[:, sorted_trees_dict['height_ref']], matched_data[:, sorted_trees_dict['height_auto']],
                    s=30, marker='.')

        ax4 = fig1.add_subplot(2, 2, 4)
        ax4.set_title("Height Error Histogram", fontsize=10)
        ax4.set_xlabel("Height Error (m)")
        ax4.set_ylabel("Frequency")
        poslim = np.max(
            matched_data[:, sorted_trees_dict['height_ref']] - matched_data[:, sorted_trees_dict['height_auto']])
        neglim = abs(
            np.min(matched_data[:, sorted_trees_dict['height_ref']] - matched_data[:, sorted_trees_dict['height_auto']]))
        lim = np.round(np.max([neglim, poslim]) / 2) * 2
        bins = np.linspace(-lim - 2, lim + 2, int(np.ceil(2 * lim / 2)) + 4)

        ax4.hist(matched_data[:, sorted_trees_dict['height_ref']] - matched_data[:, sorted_trees_dict['height_auto']],
                 bins=bins,
                 range=(-lim, lim),
                 linewidth=0.5,
                 edgecolor='black',
                 facecolor='green')

        fig1.show(False)
        fig1.savefig(directory + plot + '_DBH_and_height_plot.png', dpi=600, bbox_inches='tight', pad_inches=0.0)
        fig1.savefig('C:/Users/seank/Documents/NDT Project/Western Australia/NDT_DATA/' + plot + '_DBH_and_height_plot.png', dpi=600, bbox_inches='tight', pad_inches=0.0)

        matched_data = np.hstack((matched_data, np.array([species_column]).T))
        # pd.DataFrame(matched_data).to_csv(directory+'matched_tree_data.csv',header=[i for i in sorted_trees_dict],index=None,sep=',')
        pd.DataFrame(matched_data).to_csv(
            'C:/Users/seank/Documents/NDT Project/Western Australia/NDT_DATA/' + plot + '_matched_tree_data.csv',
            header=[i for i in sorted_trees_dict], index=None, sep=',')

        kml = simplekml.Kml()
        for i in plot_array:
            tree_lat, tree_lon, tree_name = convert_coords_to_lat_long(i[2], i[3], i[1])

            # nearest_tree_in_auto_data =

            description = 'DBH: ' + str(i[4]) + '\nHeight: ' + str(i[5]) + '\nEst__Gross_Bole_Vol_m3: ' + str(
                    i[6]) + '\nEst__Gross_Bole_V2_Vol_m3: ' + str(i[7])
            kml.newpoint(name=str(tree_name), coords=[(tree_lon, tree_lat)], description=description)
        kml.save('C:/Users/seank/Documents/NDT Project/Western Australia/NDT_DATA/' + str(plot) + '.kml')
    except FileNotFoundError:
        None
