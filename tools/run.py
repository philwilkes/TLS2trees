from run_tools import FSCT, directory_mode, file_mode
from other_parameters import other_parameters


if __name__ == '__main__':
    """Choose one of the following or modify as needed.
    Directory mode will find all .las files within a directory and sub directories but will ignore any .las files in
    folders with "FSCT_output" in their names.
    
    File mode will allow you to select multiple .las files within a directory.
    
    Alternatively, you can just list the point cloud file paths.
    
    If you have multiple point clouds and plot coords for each, you'll need to 
    
    """
    # point_clouds_to_process = directory_mode()
    # point_clouds_to_process = ['list your point cloud filepaths here']
    point_clouds_to_process = file_mode()

    for point_cloud_filename in point_clouds_to_process:
        parameters = dict(point_cloud_filename=point_cloud_filename,
                          batch_size=18,  # If you get CUDA errors, lower this. This is suitable for 24 GB of vRAM.
                          num_procs=10,  # Number of CPU cores you want to use. If you run out of RAM, lower this.
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
                          plot_radius=5,  # If 0 m, the plot is not cropped. Otherwise, the plot is cylindrically cropped from the plot centre with plot_radius + plot_radius_buffer.
                          plot_radius_buffer=0,  # See README.md  This is used for "Intelligent Plot Cropping Mode".
                          UTM_zone_number=50,
                          UTM_zone_letter='',
                          UTM_is_north=False,  # If in the northern hemisphere, set this to True.
                          filter_noise=0,
                          low_resolution_point_cloud_hack_mode=0,  # See README.md for details. Very ugly hack that can sometimes be useful on low resolution point clouds.
                          delete_working_directory=True  # Generally leave this on. Deletes the files used for segmentation after segmentation is finished.
                                                         # You may wish to turn it off if you want to modify the segmentation code so you don't need to run pre-processing every time.
                          )
        parameters.update(other_parameters)
        FSCT(parameters=parameters,
             preprocess=0,
             segmentation=0,
             postprocessing=0,
             measure_plot=True,
             make_report=True)
