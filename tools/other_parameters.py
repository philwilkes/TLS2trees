# Don't change these unless you really understand what you are doing with them/are learning the code base.
# These have been tuned to work on most high resolution forest point clouds without changing them.
other_parameters = dict(model_filename='../model/model.pth',
                        box_dimensions=[6, 6, 6],
                        box_overlap=[0.5, 0.5, 0.5],
                        min_points_per_box=1000,
                        max_points_per_box=20000,
                        noise_class=0,
                        terrain_class=1,
                        vegetation_class=2,
                        cwd_class=3,
                        stem_class=4,
                        grid_resolution=0.5,
                        supplementary_map_resolution=0.1,
                        num_neighbours=5,
                        Vegetation_coverage_resolution=1.0,
                        sorting_search_angle=20,
                        sorting_search_radius=1,
                        sorting_angle_tolerance=90,
                        max_search_radius=3,
                        max_search_angle=30,
                        slice_thickness=0.2,  # default = 0.2
                        slice_increment=0.05,  # default = 0.05
                        cleaned_measurement_radius=0.18,  # During cleaning, this w
                        subsample=True,
                        subsampling_min_spacing=0.01,
                        minimum_CCI=0.3,  # Minimum valid Circuferential Completeness Index (CCI) for non-interpolated circle/cylinder fitting. Any measurements with CCI below this are deleted.
                        min_tree_cyls=10)
