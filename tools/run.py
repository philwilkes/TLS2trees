from preprocessing import Preprocessing
from inference import SemanticSegmentation
from post_segmentation_script import PostProcessing
from report_writer import ReportWriter
import glob
import numpy as np
from measure import MeasureTree
# from test import MeasureTree
import tkinter as tk
import tkinter.filedialog as fd
from other_parameters import other_parameters
import glob
import os


if __name__ == '__main__':
    """
    If you want to select individual files, leave directory_mode set to 0 or False.
    If you want to process ALL '.las' files within a directory and its sub-directories, set directory_mode to 1 or True.
    "Directory mode" will ignore FSCT_output '.las' files. """
    directory_mode = 1

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

    # point_clouds_to_process = ["C:/Users/seank/OneDrive - University of Tasmania/2. NDT Project 2020/NSW/Shared/1high s3 single tree.las"]

    for point_cloud in point_clouds_to_process:
        print(point_cloud)

        parameters = dict(input_point_cloud=point_cloud,
                          batch_size=18,  # If you get CUDA errors, lower this. This is suitable for 24 GB of vRAM.
                          num_procs=20,  # Number of CPU cores you want to use.
                          max_diameter=5,  # Maximum diameter setting. Any measurements greater than this are considered erroneous and are ignored.
                          slice_thickness=0.2,  # default = 0.2
                          slice_increment=0.05,  # default = 0.05
                          slice_clustering_distance=0.1,  # default = 0.1
                          cleaned_measurement_radius=0.18,
                          minimum_CCI=0.3,  # Minimum valid Circuferential Completeness Index (CCI) for non-interpolated circle/cylinder fitting. Any measurements with CCI below this are deleted.
                          min_tree_volume=0.005,  # Measurements from trees with volume (m3) less than this are ignored in the outputs.
                          ground_veg_cutoff_height=3,  # Any vegetation points below this height are considered to be understory and are not assigned to individual trees.
                          veg_sorting_range=10,
                          Site='not_specified',  # Enter the site name if you wish. Only used for report generation.
                          PlotID='not_specified',  # Enter the plot name/ID if you wish. Only used for report generation.
                          plot_centre=None,  # [X, Y] Coordinates of the plot centre (metres). If "None", plot_centre is the median XY coords of the point cloud.
                          plot_radius=0,  # If 0 m, the plot is not cropped. Otherwise, the plot is cylindrically cropped from the plot centre with plot_radius + plot_radius_buffer.
                          plot_radius_buffer=0,  # See README.md  This is used for "Intelligent Plot Cropping Mode".
                          UTM_zone_number=50,  # Self explanatory.
                          UTM_zone_letter=None,  # Self explanatory.
                          UTM_is_north=False,   # If in the northern hemisphere, set this to True.
                          filter_noise=0,
                          low_resolution_point_cloud_hack_mode=0)  # See README.md for details. Dodgy hack that can be useful on low resolution point clouds. Approximately multiplies the number of points in the point cloud by this number.

        parameters.update(other_parameters)
        try:
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

        except:
            None

        #
        #     # creating/opening a file
        #     f = open(str(logname)+"_error_log.txt", "a")
        #
        #     # writing in the file
        #     f.write(str(point_cloud))
        #     f.write(str(Argument))
        #
        #     # closing the file
        #     f.close()


