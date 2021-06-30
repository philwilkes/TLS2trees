from mdutils.mdutils import MdUtils
import markdown
from mdutils import Html
import pandas as pd
import numpy as np


class ReportWriter:
    def __init__(self, parameters, data_array, outpur_dir):
        self.parameters = parameters
        self.data_array = data_array
        self.output_dir = outpur_dir
        self.create_report()

    def create_report(self):
        filename = self.output_dir + 'Plot_Report'
        mdFile = MdUtils(file_name=filename, title='Plot Report'+self.parameters['Site']+' '+self.parameters['PlotID'])
        mdFile.new_header(level=1, title='Overview')  # style is set 'atx' format by default.
        mdFile.new_paragraph(
                """
                Site='not_specified',
                PlotID='not_specified',
                plot_centre=None,
                plot_radius=0,
                plot_radius_buffer=0,
                UTM_zone_number=50,
                UTM_zone_letter=None,
                UTM_is_north=False,
                
                Plot Area: ha
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
                """)

        mdFile.new_header(level=2, title='Segmented Point Cloud - Oblique View')  # style is set 'atx' format by default.
        path = self.output_dir + "Segmented Point Cloud - Oblique View.png"
        # mdFile.new_paragraph(Html.image(path=path, size='1000'))

        mdFile.new_header(level=2, title='Original Point Cloud - Oblique View')  # style is set 'atx' format by default.
        path = self.output_dir + "CanopyDensityPlot.png"
        # mdFile.new_paragraph(Html.image(path=path, size='1000'))

        mdFile.new_header(level=2, title='Cylinders and DTM - Oblique View')  # style is set 'atx' format by default.
        path = self.output_dir + "CanopyDensityPlot.png"
        # mdFile.new_paragraph(Html.image(path=path, size='1000'))

        mdFile.new_header(level=2, title='Plot Map - Stem locations dotted, plot boundaries drawn, terrain topo')  # style is set 'atx' format by default.
        path = self.output_dir + "Stem_Map.png"
        mdFile.new_paragraph(Html.image(path=path, size='1000'))

        mdFile.new_header(level=2, title='Canopy Density Plot')  # style is set 'atx' format by default.
        path = self.output_dir + "CanopyDensityPlot.png"
        # mdFile.new_paragraph(Html.image(path=path, size='1000'))

        mdFile.new_header(level=2, title='Canopy Density Plot')  # style is set 'atx' format by default.
        path = self.output_dir + "CanopyDensityPlot.png"
        # mdFile.new_paragraph(Html.image(path=path, size='1000'))

        mdFile.new_header(level=2, title='Canopy Density Plot')  # style is set 'atx' format by default.
        path = self.output_dir + "CanopyDensityPlot.png"
        mdFile.new_paragraph(Html.image(path=path, size='1000'))

        mdFile.create_md_file()

        markdown.markdownFromFile(
                input=filename + '.md',
                output=filename + '.html',
                encoding='utf8',
        )

    def plot_outputs(self):
        dtmmin = np.min(self.DTM[:, :2], axis=0)
        dtmmax = np.max(self.DTM[:, :2], axis=0)
        plot_centre = (dtmmin + dtmmax) / 2

        fig5 = plt.figure(figsize=(7, 7))
        ax5 = fig5.add_subplot(1, 1, 1)
        plot_centre_lat, plot_centre_lon, _ = self.convert_coords_to_lat_long(plot_centre[0], plot_centre[1], ' ')

        dtm_boundaries = [[np.min(self.DTM[:, 0]), np.min(self.DTM[:, 1]), 'SouthWestCorner'],
                          [np.min(self.DTM[:, 0]), np.max(self.DTM[:, 1]), 'NorthWestCorner'],
                          [np.max(self.DTM[:, 0]), np.min(self.DTM[:, 1]), 'SouthEastCorner'],
                          [np.max(self.DTM[:, 0]), np.max(self.DTM[:, 1]), 'NorthEastCorner']]

        dtm_boundaries_lat = []
        dtm_boundaries_lon = []
        dtm_boundaries_names = []
        for i in dtm_boundaries:
            lat, lon, names = self.convert_coords_to_lat_long(i[0], i[1], i[2])
            dtm_boundaries_lat.append(lat)
            dtm_boundaries_lon.append(lon)
            dtm_boundaries_names.append(names)

        lat, lon, names = self.convert_coords_to_lat_long(plot_centre[0],
                                                          plot_centre[1],
                                                          'PlotCentre')
        dtm_boundaries_lat.append(lat)
        dtm_boundaries_lon.append(lon)
        dtm_boundaries_names.append(names)

        dtm_boundaries = np.array([dtm_boundaries_lat, dtm_boundaries_lon, dtm_boundaries_names]).T
        pd.DataFrame(dtm_boundaries).to_csv(self.output_dir + 'Plot_Extents.csv', header=False, index=None, sep=',')
        for i in dtm_boundaries:
            self.kml.newpoint(name=i[2], coords=[(i[1], i[0])], description='Boundary point')

        ax5.set_title("Plot Map")
        ax5.set_xlabel("X Position (m)")
        ax5.set_ylabel("Y Postition (m)")
        ax5.text("Plot centre: " + str([plot_centre_lat, plot_centre_lon])[1:-1], fontsize=10)
        ax5.axis('equal')
        zmin = np.floor(np.min(self.DTM[:, 2]))
        zmax = np.ceil(np.max(self.DTM[:, 2]))
        contour_resolution = 1  # metres
        sub_contour_resolution = contour_resolution / 5
        zrange = int(np.ceil((zmax - zmin) / contour_resolution)) + 1
        levels = np.linspace(zmin, zmax, zrange)

        sub_zrange = int(np.ceil((zmax - zmin) / sub_contour_resolution)) + 1
        sub_levels = np.linspace(zmin, zmax, sub_zrange)

        ax5.tricontour(self.DTM[:, 0] - plot_centre[0], self.DTM[:, 1] - plot_centre[1], self.DTM[:, 2],
                       levels=sub_levels, colors='brown', linestyles='dashed', linewidths=1)

        contours = ax5.tricontour(self.DTM[:, 0] - plot_centre[0], self.DTM[:, 1] - plot_centre[1], self.DTM[:, 2],
                                  levels=levels, colors='darkgreen')

        ax5.scatter([0], [0], marker='x', s=50, c='red')
        plt.clabel(contours, inline=True, fontsize=8)
        ax5.set_xlim([np.min(self.DTM[:, 0]) - plot_centre[0] - 5, np.max(self.DTM[:, 0]) - plot_centre[0] + 5])
        ax5.set_ylim([np.min(self.DTM[:, 1]) - plot_centre[1] - 5, np.max(self.DTM[:, 1]) - plot_centre[1] + 5])
        fig5.show(False)
        fig5.savefig(self.output_dir + 'Stem_Map.png', dpi=600, bbox_inches='tight', pad_inches=0.0)
        ########################################################################################################################

        # Canopy Density Plot
        fig6 = plt.figure(figsize=(7, 7))
        ax6 = fig6.add_subplot(1, 1, 1)
        ax6.set_title("Stem Map", fontsize=10)
        ax6.set_xlabel("X Position (m)")
        ax6.set_ylabel("Y Postition (m)")
        ax6.axis('equal')
        ax6.scatter(self.DTM[:, 0], self.DTM[:, 1], c='white')
        ax6.scatter(canopy_density[:, 0], canopy_density[:, 1], c=canopy_density[:, 3], s=200, marker='s')
        fig6.show(False)
        fig6.savefig(self.output_dir + 'CanopyDensityPlot.png', dpi=600, bbox_inches='tight', pad_inches=0.0)
        ########################################################################################################################
        plt.close('all')


if __name__ == '__main__':
    parameters = dict(input_point_cloud=None,
                      batch_size=18,
                      num_procs=20,
                      max_diameter=5,
                      slice_thickness=0.2,  # default = 0.2
                      slice_increment=0.05,  # default = 0.05
                      slice_clustering_distance=0.2,  # default = 0.1
                      cleaned_measurement_radius=0.18,
                      minimum_CCI=0.3,
                      min_tree_volume=0.005,
                      ground_veg_cutoff_height=3,
                      veg_sorting_range=5,
                      canopy_mode='continuous',
                      Site='',  # Insert name
                      PlotID='',  # Insert name
                      plot_centre=None,
                      plot_radius=0,
                      plot_radius_buffer=0,
                      UTM_zone_number=50,
                      UTM_zone_letter=None,
                      UTM_is_north=False,
                      filter_noise=0,
                      low_resolution_point_cloud_hack_mode=0)

    ReportWriter(parameters=parameters, data_array=None, outpur_dir="C:/Users/seank/OneDrive - University of Tasmania/2. NDT Project 2020/NSW/Shared/high s2 p1 NSW_FSCT_output/")
