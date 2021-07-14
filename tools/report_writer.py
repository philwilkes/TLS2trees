from mdutils.mdutils import MdUtils
import markdown
from mdutils import Html
import pandas as pd
import numpy as np
from tools import load_file, subsample_point_cloud
from matplotlib import pyplot as plt
import utm
import simplekml
import os
from scipy.spatial import ConvexHull
from matplotlib import cm
import warnings


class ReportWriter:
    def __init__(self, parameters):
        self.parameters = parameters
        self.filename = self.parameters['input_point_cloud'].replace('\\', '/')
        self.output_dir = os.path.dirname(os.path.realpath(self.filename)).replace('\\', '/') + '/' + self.filename.split('/')[-1][:-4] + '_FSCT_output/'
        self.filename = self.filename.split('/')[-1]
        self.DTM, _ = load_file(self.output_dir + 'DTM.las')
        self.cwd_points, _ = load_file(self.output_dir + 'cwd_points.las')
        self.veg_dict = dict(x=0, y=1, z=2, red=3, green=4, blue=5, tree_id=6, height_above_dtm=7)
        self.ground_veg, _ = load_file(self.output_dir + 'ground_veg.las', headers_of_interest=list(self.veg_dict))
        self.kml = simplekml.Kml()
        self.tree_data = pd.read_csv(self.output_dir + 'tree_data.csv')
        self.plot_area = 0
        self.treeNo = np.array(self.tree_data['treeNo'])
        self.x_tree_base = np.array(self.tree_data['x_tree_base'])
        self.y_tree_base = np.array(self.tree_data['y_tree_base'])
        self.DBH = np.array(self.tree_data['DBH'])
        self.height = np.array(self.tree_data['Height'])
        self.Volume = np.array(self.tree_data['Volume'])
        self.processing_report = pd.read_csv(self.output_dir + 'processing_report.csv', index_col=False)
        self.plot_area = float(self.processing_report['Plot Area'])
        self.stems_per_ha = int(self.processing_report['Stems/ha'])
        self.parameters['plot_centre'] = np.loadtxt(self.output_dir + 'plot_centre_coords.csv')
        self.plot_outputs()
        self.create_report()

    def create_report(self):
        filename = self.output_dir + 'Plot_Report'
        mdFile = MdUtils(file_name=filename, title='Forest Structural Complexity Tool - Plot Report')
        mdFile.new_header(level=1, title='')  # style is set 'atx' format by default.
        if self.parameters['UTM_is_north']:
            hemisphere = 'North'
        else:
            hemisphere = 'South'
        level = 2

        mdFile.new_header(level=level, title='Plot ID: ' + str(self.parameters['PlotID']) + ' Site: ' + str(self.parameters['Site']))
        mdFile.new_header(level=level, title='Point Cloud Filename: ' + self.filename)
        mdFile.new_header(level=level, title='Plot Centre: ' + str(self.parameters['plot_centre'][0]) + ' N, ' + str(self.parameters['plot_centre'][1]) + ' E, UTM Zone: ' + ' ' + str(self.parameters['UTM_zone_number']) + ' ' + str(self.parameters['UTM_zone_letter']) + ', Hemisphere: ' + hemisphere)
        mdFile.new_header(level=level, title='Plot Radius: ' + str(self.parameters['plot_radius']) + ' m, ' + ' Plot Radius Buffer: ' + str(self.parameters['plot_radius_buffer']) + ' m, Plot Area: '+ str(self.plot_area) + ' ha')
        mdFile.new_header(level=level, title='Stems/ha:  ' + str(self.stems_per_ha))
        mdFile.new_header(level=level, title='Mean DBH: ' + str(np.around(np.mean(self.DBH), 3)) + ' m')
        mdFile.new_header(level=level, title='Median DBH: ' + str(np.around(np.median(self.DBH), 3)) + ' m')
        mdFile.new_header(level=level, title='Min DBH: ' + str(np.around(np.min(self.DBH), 3)) + ' m')
        mdFile.new_header(level=level, title='Max DBH: ' + str(np.around(np.max(self.DBH), 3)) + ' m')

        mdFile.new_paragraph()

        """
        
        
        Stems/ha:  
               
        Mean DBH:
        Median DBH:
        Min DBH:
        Max DBH:
        
        Mean Height:
        Median Height:
        Min Height:
        Max Height:
        
        Mean Volume:
        Median Volume:
        Min Volume:
        Max Volume:
        Total Volume:   m^3
        Total Volume (DBH > 0.1 m):   m^3
        
        Average Terrain Gradient:    degrees
        Mean Understory height:
        Understory coverage fraction:     
        Canopy Coverage fraction:      
        CWD coverage fraction:
                
        Run times:
        Preprocessing_Time (s)
        Semantic_Segmentation_Time (s),
        Post_processing_time (s),
        Measurement Time (s)
        
        Parameters used:
        max_diameter=5,
        slice_thickness=0.2, 
        slice_increment=0.05, 
        slice_clustering_distance=0.2, 
        cleaned_measurement_radius=0.18,
        minimum_CCI=0.3, 
        min_tree_volume=0.005, 
        ground_veg_cutoff_height=3, 
        veg_sorting_range=5,
        canopy_mode='continuous',
        filter_noise=0,
        low_resolution_point_cloud_hack_mode=0
        """
        mdFile.new_paragraph()
        path = self.output_dir + "Stem_Map.png"
        mdFile.new_paragraph(Html.image(path=path, size='1000'))

        mdFile.create_md_file()

        markdown.markdownFromFile(input=filename + '.md',
                                  output=filename + '.html',
                                  encoding='utf8')

    def plot_outputs(self):
        self.ground_veg_map = self.ground_veg[:, [0, 1, self.veg_dict['height_above_dtm']]]
        self.ground_veg_map[self.ground_veg[:, self.veg_dict['height_above_dtm']] >= 0.5, 2] = 1
        self.ground_veg_map[self.ground_veg[:, self.veg_dict['height_above_dtm']] < 0.5, 2] = 0.5

        dtmmin = np.min(self.DTM[:, :2], axis=0)
        dtmmax = np.max(self.DTM[:, :2], axis=0)
        plot_max_distance = np.max(dtmmax - dtmmin)
        plot_centre = (dtmmin + dtmmax) / 2
        if self.parameters['plot_centre'] is None:
            self.parameters['plot_centre'] = plot_centre
        fig5 = plt.figure(figsize=(7, 7))
        ax5 = fig5.add_subplot(1, 1, 1)
        plot_centre_lat, plot_centre_lon, = utm.to_latlon(easting=plot_centre[0],
                                                          northing=plot_centre[1],
                                                          zone_number=self.parameters['UTM_zone_number'],
                                                          zone_letter=self.parameters['UTM_zone_letter'],
                                                          northern=self.parameters['UTM_is_north'],
                                                          strict=None)

        dtm_boundaries = [[np.min(self.DTM[:, 0]), np.min(self.DTM[:, 1]), 'SouthWestCorner'],
                          [np.min(self.DTM[:, 0]), np.max(self.DTM[:, 1]), 'NorthWestCorner'],
                          [np.max(self.DTM[:, 0]), np.min(self.DTM[:, 1]), 'SouthEastCorner'],
                          [np.max(self.DTM[:, 0]), np.max(self.DTM[:, 1]), 'NorthEastCorner']]

        dtm_boundaries_lat = []
        dtm_boundaries_lon = []
        dtm_boundaries_names = []
        for i in dtm_boundaries:
            lat, lon = utm.to_latlon(easting=i[0],
                                     northing=i[1],
                                     zone_number=self.parameters['UTM_zone_number'],
                                     zone_letter=self.parameters['UTM_zone_letter'],
                                     northern=self.parameters['UTM_is_north'],
                                     strict=None)

            dtm_boundaries_lat.append(lat)
            dtm_boundaries_lon.append(lon)
            dtm_boundaries_names.append(i[2])

        plot_centre_lat, plot_centre_lon = utm.to_latlon(easting=plot_centre[0],
                                                         northing=plot_centre[1],
                                                         zone_number=self.parameters['UTM_zone_number'],
                                                         zone_letter=self.parameters['UTM_zone_letter'],
                                                         northern=self.parameters['UTM_is_north'],
                                                         strict=None)
        dtm_boundaries_lat.append(plot_centre_lat)
        dtm_boundaries_lon.append(plot_centre_lon)
        dtm_boundaries_names.append('PlotCentre')

        dtm_boundaries = np.array([dtm_boundaries_lat, dtm_boundaries_lon, dtm_boundaries_names]).T
        pd.DataFrame(dtm_boundaries).to_csv(self.output_dir + 'Plot_Extents.csv', header=False, index=None, sep=',')
        for i in dtm_boundaries:
            self.kml.newpoint(name=i[2], coords=[(i[1], i[0])], description='Boundary point')

        ax5.set_title("Plot Map")
        ax5.set_xlabel("Easting + " + str(self.parameters['plot_centre'][0]) + ' (m)')
        ax5.set_ylabel("Northing + " + str(self.parameters['plot_centre'][1]) + ' (m)')
        # ax5.text("Plot centre: " + str([plot_centre_lat, plot_centre_lon])[1:-1], fontsize=10)
        ax5.axis('equal')
        ax5.set_facecolor('whitesmoke')
        zmin = np.floor(np.min(self.DTM[:, 2]))
        zmax = np.ceil(np.max(self.DTM[:, 2]))
        contour_resolution = 1  # metres
        sub_contour_resolution = contour_resolution / 5
        zrange = int(np.ceil((zmax - zmin) / contour_resolution)) + 1
        levels = np.linspace(zmin, zmax, zrange)

        sub_zrange = int(np.ceil((zmax - zmin) / sub_contour_resolution)) + 1
        sub_levels = np.linspace(zmin, zmax, sub_zrange)
        sub_levels = sub_levels[sub_levels % contour_resolution != 0]  # remove sub contours where there are full size contours.

        hull = ConvexHull(self.DTM[:, :2])
        shape_points = self.DTM[hull.vertices]
        shape_points = np.vstack((shape_points, shape_points[0]))
        ax5.fill(shape_points[:, 0] - plot_centre[0], shape_points[:,1] - plot_centre[1], c='white', alpha=1, zorder=0)
        ax5.plot(shape_points[:, 0] - plot_centre[0], shape_points[:, 1] - plot_centre[1], c='k', alpha=1, linewidth=0.5, zorder=1)

        ax5.scatter(self.ground_veg_map[self.ground_veg_map[:, 2] == 0.5, 0] - plot_centre[0], self.ground_veg_map[self.ground_veg_map[:, 2] == 0.5, 1] - plot_centre[1], marker='.', s=4, c='#B2F2BB', zorder=3)
        ax5.scatter(self.ground_veg_map[self.ground_veg_map[:, 2] == 1, 0] - plot_centre[0], self.ground_veg_map[self.ground_veg_map[:, 2] == 1, 1] - plot_centre[1], marker='.', s=4, c='#8CE99A', zorder=3)

        ax5.scatter(self.cwd_points[:, 0] - plot_centre[0], self.cwd_points[:, 1] - plot_centre[1], marker='.', s=1, c='yellow', zorder=3)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            subcontours = ax5.tricontour(self.DTM[:, 0] - plot_centre[0], self.DTM[:, 1] - plot_centre[1], self.DTM[:, 2], levels=sub_levels, colors='burlywood', linestyles='dashed', linewidths=2, zorder=3)
            contours = ax5.tricontour(self.DTM[:, 0] - plot_centre[0], self.DTM[:, 1] - plot_centre[1], self.DTM[:, 2], levels=levels, colors='darkgreen', linewidths=2, zorder=5)

        plt.clabel(subcontours, inline=True, fmt='%1.1f', fontsize=6, zorder=4)
        plt.clabel(contours, inline=True, fmt='%1.0f', fontsize=10, zorder=6)

        ax5.scatter(self.x_tree_base - plot_centre[0], self.y_tree_base - plot_centre[1], marker='.', s=70, c='black', zorder=7)
        ax5.scatter(self.x_tree_base - plot_centre[0], self.y_tree_base - plot_centre[1], marker='.', s=30, c='red', zorder=8)

        tree_label_offset = np.array([-0.01, 0.01]) * plot_max_distance
        for i in range(0, self.x_tree_base.shape[0]):
            ax5.text((self.x_tree_base[i] - plot_centre[0]) + tree_label_offset[0], (self.y_tree_base[i] - plot_centre[1]) + tree_label_offset[1], self.treeNo[i], fontsize=6, zorder=9)

        ax5.scatter([0], [0], marker='x', s=60, c='black', zorder=10)

        ax5.set_xlim([np.min(self.DTM[:, 0]) - plot_centre[0], np.max(self.DTM[:, 0]) - plot_centre[0]])
        ax5.set_ylim([np.min(self.DTM[:, 1]) - plot_centre[1], np.max(self.DTM[:, 1]) - plot_centre[1]])
        fig5.show(False)

        fig5.savefig(self.output_dir + 'Stem_Map.png', dpi=600, bbox_inches='tight', pad_inches=0.0)

        # print(list(self.tree_data.columns))

        # plt.figure()
        # ax = plt.gca()
        # col_labels = list(self.tree_data.columns)
        # # row_labels = ['row1', 'row2', 'row3']
        # # table_vals = [[11, 12, 13], [21, 22, 23], [31, 32, 33]]
        # # the rectangle is where I want to place the table
        # ax.table(cellText=np.array(self.tree_data.T),
        #           rowLabels=col_labels,
        #           loc='center')
        # ax.axis('tight')
        # ax.axis('off')
        # # plt.text(12, 3.4, 'Table Title', size=8)
        # plt.show()
        plt.close('all')

