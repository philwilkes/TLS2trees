from preprocessing import Preprocessing
from inference import SemanticSegmentation
from post_segmentation_script import PostProcessing
import glob
import numpy as np
from measure import MeasureTree
import tkinter as tk
import tkinter.filedialog as fd
from other_parameters import other_parameters
import glob
import os


if __name__ == '__main__':
    """
    If you want to select individual files, leave directory_mode set to 0 or False.
    If you want to process ALL '.las' files within a directory and its sub-directories, set directory_mode to 1 or True.
    "Directory mode" will ignore FSCT_output '.las' files.
    """

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
        point_clouds_to_process = fd.askopenfilenames(parent=root, title='Choose files', filetypes=[("LAS", "*.las"), ("CSV", "*.csv")])
    root.destroy()

    for point_cloud in point_clouds_to_process:
        print(point_cloud)

        parameters = dict(input_point_cloud=point_cloud,
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
                          Site='not_specified',
                          PlotID='not_specified',
                          plot_centre=None,
                          plot_radius=4,
                          plot_radius_buffer=3,
                          UTM_zone_number=50,
                          UTM_zone_letter=None,
                          UTM_is_north=False,
                          filter_noise=0,
                          low_resolution_point_cloud_hack_mode=0)  # TODO could add this mode to measure.

        parameters.update(other_parameters)

        # preprocessing = Preprocessing(parameters)
        # preprocessing.preprocess_point_cloud()
        # del preprocessing
        #
        # sem_seg = SemanticSegmentation(parameters)
        # sem_seg.inference()
        # del sem_seg
        #
        # object_1 = PostProcessing(parameters)
        # object_1.process_point_cloud()
        # del object_1

        # measure1 = MeasureTree(parameters)
        # measure1.run_measurement_extraction()
        # del measure1
