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
    # point_clouds_to_process = ['E:/PFOlsen2/PFOlsenPlots/T1_class.las']
    point_clouds_to_process = file_mode()

    for point_cloud_filename in point_clouds_to_process:
        parameters = dict(point_cloud_filename=point_cloud_filename,
                          # Adjust if needed
                          plot_centre=None,  # [X, Y] Coordinates of the plot centre (metres). If "None", plot_centre is the median XY coords of the point cloud.

                          # Circular Plot options - Leave at 0 if not using.
                          plot_radius=0,  # If 0 m, the plot is not cropped. Otherwise, the plot is cylindrically cropped from the plot centre with plot_radius + plot_radius_buffer.
                          plot_radius_buffer=3,  # See README. If non-zero, this is used for "Tree Aware Plot Cropping Mode".

                          # Rectangular/Tiled Plot options - Leave at 0 if not using.
                          x_length=0,  # NOT YET IMPLEMENTED
                          y_length=0,  # NOT YET IMPLEMENTED
                          edge_buffer=0,  # NOT YET IMPLEMENTED

                          Site='',  # Enter the site name if you wish. Only used for report generation.
                          PlotID='',  # Enter the plot name/ID if you wish. Only used for report generation.
                          UTM_zone_number=50,  # Optional: Set this or the Lat Lon outputs will be incorrect.
                          UTM_zone_letter='',  # Optional: Used for the plot report.
                          UTM_is_north=False,  # If in the northern hemisphere, set this to True.

                          # Set these appropriately for your hardware.
                          batch_size=18,  # If you get CUDA errors, try lowering this. This is suitable for 24 GB of vRAM.
                          num_procs=10,  # Number of CPU cores you want to use. If you run out of RAM, lower this.

                          # Optional settings - Generally leave as they are.
                          sort_stems=1,  # If you don't need the sorted stem points, turning this off speeds things up.
                                         # Veg sorting is required for tree height measurement, but stem sorting isn't necessary for general use.

                          generate_output_point_cloud=1,  # Turn on if you would like a semantic and instance segmented point cloud. This mode will override the "sort_stems" setting if on.
                                                          # If you activate "tree aware plot cropping mode", this function will use it.
                          ground_veg_cutoff_height=3,  # Any vegetation points below this height are considered to be understory and are not assigned to individual trees.
                          veg_sorting_range=5,  # Vegetation points can be, at most, this far away from a cylinder horizontally to be matched to a particular tree.
                          stem_sorting_range=2,  # Stem points can be, at most, this far away from a cylinder in 3D to be matched to a particular tree.
                          low_resolution_point_cloud_hack_mode=0,  # See README.md for details. Very ugly hack that can sometimes be useful on low resolution point clouds.

                          delete_working_directory=True,  # Generally leave this on. Deletes the files used for segmentation after segmentation is finished.
                                                          # You may wish to turn it off if you want to re-run/modify the segmentation code so you don't need to run pre-processing every time.
                          minimise_output_size_mode=0  # Will delete a number of non-essential outputs to reduce storage use.
                          )

        parameters.update(other_parameters)
        FSCT(parameters=parameters,
             # Set below to 0 or 1 (or True/False). Each step requires the previous step to have been run already.
             # For standard use, just leave a   ll set to 1.
             preprocess=0,
             segmentation=0,
             postprocessing=1,
             measure_plot=0,
             make_report=0)
