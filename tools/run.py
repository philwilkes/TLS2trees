from preprocessing import Preprocessing
from inference import SemanticSegmentation
from post_segmentation_script import PostProcessing
from report_writer import ReportWriter
import glob
import numpy as np
from measure import MeasureTree
import tkinter as tk
import tkinter.filedialog as fd
from other_parameters import other_parameters
import glob
import os
import sys


if __name__ == '__main__':
    """
    If you want to select individual files, leave directory_mode set to 0 or False.
    If you want to process ALL '.las' files within a directory and its sub-directories, set directory_mode to 1 or True.
    "Directory mode" will ignore FSCT_output '.las' files. """
    directory_mode = 0

    root = tk.Tk()
    if directory_mode:
        point_clouds_to_process = []
        directory = fd.askdirectory(parent=root, title='Choose directory')
        unfiltered_point_clouds_to_process = glob.glob(directory + '/**/*.las', recursive=True)
        for i in unfiltered_point_clouds_to_process:
            if 'FSCT_output' not in i:
                point_clouds_to_process.append(i)
    else:
        point_clouds_to_process = fd.askopenfilenames(parent=root, title='Choose files', filetypes=[("LAS", "*.las"), ("LAZ", "*.laz"), ("CSV", "*.csv")])
    root.destroy()

    # point_clouds_to_process = ["C:/Users/seank/Downloads/CULS/CULS/plot_1_annotated.las"]
    # point_clouds_to_process = [
                               # 'C:/Users/seank/Documents/NDT Project/Western Australia/Leach P111.las',
                               # 'C:/Users/seank/Documents/NDT Project/Western Australia/Leach P61.las',
                               # 'C:/Users/seank/Documents/NDT Project/Western Australia/Denham P264.las',
                               # 'C:/Users/seank/Documents/NDT Project/Western Australia/Denham P257.las',
                               # 'C:/Users/seank/Documents/NDT Project/Western Australia/Leach_P111_TLS.las',
                               #
                               # 'C:/Users/seank/Documents/NDT Project/Western Australia/Leach_P61_TLS.las',
                               # 'C:/Users/seank/Documents/NDT Project/Western Australia/Denham_P264_TLS.las',
                               # 'C:/Users/seank/Documents/NDT Project/Western Australia/Denham_P257_TLS.las',
                               # 'C:/Users/seank/Documents/NDT Project/Western Australia/Fleas P1.las',
                               # 'C:/Users/seank/Documents/NDT Project/Western Australia/Fleas P2.las',
                               #
                               # 'C:/Users/seank/Documents/NDT Project/Western Australia/Fleas P3.las',

                               # NSW
                               # 'C:/Users/seank/Documents/NDT Project/New South Wales/Site1Plot1.las',
                               # 'C:/Users/seank/Documents/NDT Project/New South Wales/Site1Plot2.las',
                               # 'C:/Users/seank/Documents/NDT Project/New South Wales/Site2Plot1.las',
                               # 'C:/Users/seank/Documents/NDT Project/New South Wales/Site3SingleTree.las',

                               # ]
    # plot_centres = [
                    # [445915.24, 6314467.17],
                    # [445949.37, 6314325.78],
                    # [417213.82, 6335818.69],
                    # [417596.50, 6335782.84],
                    # [445913.42, 6314465.786],
                    #
                    # [445948.35, 6314325.54],
                    # [417212.755, 6335817.434],
                    # [417595.946, 6335779.279],
                    # None,
                    # None,
                    #
                    # None,

                    # NSW
                    # [478984.407, 6661926.300],
                    # [478965.77, 6661983.050],
                    # [519795.32, 6690123.967],
                    # None,
    # ]

    # plot_radii = [
                  # 20,
                  # 20,
                  # 20,
                  # 20,
                  # 20,
                  #
                  # 20,
                  # 20,
                  # 20,
                  # 20,
                  # 20,
                  #
                  # 20,
                  #
                  # 20,
                  # 20,
                  # 9,
                  # 3
    # ]

    # for point_cloud, plot_centre, radius in zip(point_clouds_to_process, plot_centres, plot_radii):
    for point_cloud in point_clouds_to_process:
        print(point_cloud)

        parameters = dict(input_point_cloud=point_cloud,
                          batch_size=18,  # If you get CUDA errors, lower this. This is suitable for 24 GB of vRAM.
                          num_procs=10,  # Number of CPU cores you want to use. If you run out of RAM, lower this.
                          max_diameter=5,  # Maximum diameter setting. Any measurements greater than this are considered erroneous and are ignored.
                          slice_thickness=0.2,  # default = 0.2
                          slice_increment=0.05,  # default = 0.05
                          slice_clustering_distance=0.1,  # default = 0.1
                          cleaned_measurement_radius=0.18,
                          subsample=True,
                          subsampling_min_spacing=0.01,
                          minimum_CCI=0.3,  # Minimum valid Circuferential Completeness Index (CCI) for non-interpolated circle/cylinder fitting. Any measurements with CCI below this are deleted.
                          min_tree_volume=0.005,  # Measurements from trees with volume (m3) less than this are ignored in the outputs.
                          ground_veg_cutoff_height=3,  # Any vegetation points below this height are considered to be understory and are not assigned to individual trees.
                          veg_sorting_range=10,
                          Site='',  # Enter the site name if you wish. Only used for report generation.
                          PlotID='',  # Enter the plot name/ID if you wish. Only used for report generation.
                          plot_centre=None,  # [X, Y] Coordinates of the plot centre (metres). If "None", plot_centre is the median XY coords of the point cloud.
                          plot_radius=10,  # If 0 m, the plot is not cropped. Otherwise, the plot is cylindrically cropped from the plot centre with plot_radius + plot_radius_buffer.
                          plot_radius_buffer=3,  # See README.md  This is used for "Intelligent Plot Cropping Mode".
                          UTM_zone_number=50,  # Self explanatory.
                          UTM_zone_letter='',  # Self explanatory.
                          UTM_is_north=False,   # If in the northern hemisphere, set this to True.
                          filter_noise=0,
                          low_resolution_point_cloud_hack_mode=0)  # See README.md for details. Dodgy hack that can be useful on low resolution point clouds. Approximately multiplies the number of points in the point cloud by this number.

        parameters.update(other_parameters)
        preprocessing = Preprocessing(parameters)
        preprocessing.preprocess_point_cloud()
        del preprocessing

        sem_seg = SemanticSegmentation(parameters)
        sem_seg.inference()
        del sem_seg

        object_1 = PostProcessing(parameters)
        object_1.process_point_cloud()
        del object_1

        measure1 = MeasureTree(parameters)
        measure1.run_measurement_extraction()
        del measure1

        ReportWriter(parameters)

