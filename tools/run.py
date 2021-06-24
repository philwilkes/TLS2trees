from inference import SemanticSegmentation
from post_segmentation_script import PostProcessing
import glob
import numpy as np
from measure import MeasureTree
import tkinter as tk
import tkinter.filedialog as fd

if __name__ == '__main__':
    root = tk.Tk()
    point_clouds_to_process = fd.askopenfilenames(parent=root, title='Choose files', filetypes=[("LAS", "*.las"),
                                                                                                ("CSV", "*.csv")])
    point_clouds_to_process = [i.split('/')[-1] for i in point_clouds_to_process]
    root.destroy()

    print(point_clouds_to_process)
    for point_cloud in point_clouds_to_process:
        print(point_cloud)

        parameters = dict(directory='../',
                          fileset='test',
                          input_point_cloud=point_cloud,
                          model_filename='../model/model.pth',
                          batch_size=20,
                          box_dimensions=[6, 6, 6],
                          box_overlap=[0.5, 0.5, 0.5],
                          min_points_per_box=1000,
                          max_points_per_box=30000,
                          subsample=False,
                          subsampling_min_spacing=0.01,
                          num_procs=20,
                          noise_class=0,
                          terrain_class=1,
                          vegetation_class=2,
                          cwd_class=3,
                          stem_class=4,
                          coarse_grid_resolution=6,
                          fine_grid_resolution=0.5,


                          max_diameter=5,
                          num_neighbours=3,
                          slice_thickness=0.2,#default = 0.2
                          slice_increment=0.05,#default = 0.05
                          slice_clustering_distance=0.2, #default = 0.1
                          cleaned_measurement_radius=0.18,
                          minimum_CCI=0.3,
                          min_tree_volume=0.005,
                          Canopy_coverage_resolution=0.5,
                          ground_veg_cutoff_height=3,


                          canopy_mode='photogrammetry_mode',
                          Site='not_specified',
                          PlotID='not_specified',
                          plot_centre=None,
                          plot_radius=0,
                          UTM_zone_number=50,
                          UTM_zone_letter=None,
                          UTM_is_north=False,
                          run_from_start=1,
                          filter_noise=0,
                          low_resolution_point_cloud_hack_mode=0)

        sem_seg = SemanticSegmentation(parameters)
        sem_seg.run_preprocessing()
        # sem_seg.inference()
        #
        # object_1 = PostProcessing(parameters)
        # object_1.process_point_cloud(point_cloud=sem_seg.output)
        #
        # del sem_seg

        measure1 = MeasureTree(parameters)
        measure1.run_measurement_extraction()
        del measure1
