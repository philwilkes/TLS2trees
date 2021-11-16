from run_tools import FSCT, directory_mode, file_mode
from other_parameters import other_parameters


if __name__ == '__main__':
    """
    This script is an example of how to provide multiple different plot centres with your input point clouds.
    """
    point_clouds_to_process = ['E:/example_dir/T1.las',
                               'E:/example_dir/T2.las']

    plot_centres = [[10, 5],  # X1, Y1
                    [40, 30]]  # X2, Y2...

    for point_cloud_filename, plot_centre in zip(point_clouds_to_process, plot_centres):
        parameters = dict(point_cloud_filename=point_cloud_filename,
                          plot_centre=plot_centre,
                          plot_radius=0,
                          plot_radius_buffer=0,
                          batch_size=18,
                          num_procs=18,
                          slice_thickness=0.15,
                          slice_increment=0.05,
                          sort_stems=1,
                          height_percentile=100,
                          tree_base_cutoff_height=10,
                          generate_output_point_cloud=1,
                          ground_veg_cutoff_height=3,
                          veg_sorting_range=1.5,
                          stem_sorting_range=1,
                          low_resolution_point_cloud_hack_mode=0,
                          maximum_stem_diameter=3,
                          delete_working_directory=True,
                          minimise_output_size_mode=0
                          )

        parameters.update(other_parameters)
        FSCT(parameters=parameters,
             preprocess=1,
             segmentation=1,
             postprocessing=1,
             measure_plot=1,
             make_report=1,
             clean_up_files=0)
