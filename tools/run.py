from inference import SemanticSegmentation
from post_segmentation_script import PostProcessing
import glob
import numpy as np
from measure import MeasureTree

# point_clouds_to_process = ['20190917_Tumba001_1-14_merged_1cm_SA_.csv',
#                            'Samford_merged_2.csv'
#                            ]


# 
# point_clouds_to_process.append(glob.glob(("samford*.csv")))
if __name__ == '__main__':

    point_clouds_to_process = [
            # 'TLS_1.csv',
            # 'TLS_2.csv',
            # 'TLS_3.csv',
            # 'TLS_4.csv',
            # 'HOVERMAP_1.csv',
            # 'HOVERMAP_2.csv',
            # 'HOVERMAP_3.csv',
            # 'UAS_AP_1.csv',
            # 'TLS_BENCHMARK_1.csv',
            # 'TLS_BENCHMARK_2.csv',
            # 'TLS_BENCHMARK_3.csv',
            # 'TLS_BENCHMARK_4.csv',
            # 'TLS_BENCHMARK_5.csv',
            # 'TLS_BENCHMARK_6.csv',
            # 'VUX_1LR_1.csv',
            # 'VUX_1LR_2.csv',
            # 'UAS_UC_AP_1.csv',
            # 'UAS_AP_2.csv',
            # 'HQPLR008_class_elev_d_orient_re_z56.csv'
            # 'final_test.csv',
            # "NDT_PROJ_Leach_P111_TLS.csv",
            # "NDT_PROJ_Denham_P257_TLS.csv",
            # "NDT_PROJ_Denham_P264_TLS.csv",
            # "NDT_PROJ_Leach_P61_TLS.csv",
            # "mal1_000040.csv",
            'taper29_graphic_version.csv',
            # 'TAPER40_class_1.0_cmDS.csv',
            # 'Fleas_P1.csv',
            # 's2p1NSW.csv'
            # "TAPER49_class_1.0_cmDS.csv",
            # "TAPER48_class_1.0_cmDS.csv",
            # "TAPER47_class_1.0_cmDS.csv",
            # "TAPER46_class_1.0_cmDS.csv",
            # "TAPER45_class_1.0_cmDS.csv",
            # "TAPER44_class_1.0_cmDS.csv",
            # "TAPER43_class_1.0_cmDS.csv",
            # "TAPER42_class_1.0_cmDS.csv",
            # "TAPER41_class_1.0_cmDS.csv",
            # "TAPER40_class_1.0_cmDS.csv",
            # "TAPER39_class_1.0_cmDS.csv",
            # "TAPER38_class_1.0_cmDS.csv",
            # "TAPER37_class_1.0_cmDS.csv",
            # "TAPER36_class_1.0_cmDS.csv",
            # "TAPER35_class_1.0_cmDS.csv",
            # "TAPER34_class_1.0_cmDS.csv",
            # "TAPER33_class_1.0_cmDS.csv",
            # "TAPER32_class_1.0_cmDS.csv",
            # "TAPER31_class_1.0_cmDS.csv",
            # "TAPER30_class_1.0_cmDS.csv",
            # "TAPER29_class_1.0_cmDS.csv",
            # "TAPER50_class_1.0_cmDS.csv",
            # "T27_class_1.0_cmDS.csv",
            # "T26_class_1.0_cmDS.csv",
            # "T25_class_1.0_cmDS.csv",
            # "T23_class_1.0_cmDS.csv",
            # "T22_class_1.0_cmDS.csv",
            # "T21_class_1.0_cmDS.csv",
            # "T20_class_1.0_cmDS.csv",
            # "T19_class_1.0_cmDS.csv",
            # "T18_class_1.0_cmDS.csv",
            # "T017_class_1.0_cmDS.csv",
            # "T16_class_1.0_cmDS.csv",
            # "T15_class_1.0_cmDS.csv",
            # "T14_class_1.0_cmDS.csv",
            # "T13_class_1.0_cmDS.csv",
            # "T12_class_1.0_cmDS.csv",
            # "T11_class_1.0_cmDS.csv",
            # "T10_class_1.0_cmDS.csv",
            # "T9_class_1.0_cmDS.csv",
            # "T8_class_1.0_cmDS.csv",
            # "T7_class_1.0_cmDS.csv",
            # "T6_class_1.0_cmDS.csv",
            # "T05_class_1.0_cmDS.csv",
            # "T4_class_1.0_cmDS.csv",
            # "T3_class_1.0_cmDS.csv",
            # "T02_class_1.0_cmDS.csv",
            # "T1_class_1.0_cmDS.csv",

            # 'Fleas_P2.csv',
            # 'Fleas_P3.csv',
            # 'Leach_P61.csv',
            # 'Leach_P111.csv',
            # 'Denham_P264.csv',
            # 'Denham_P257.csv',
    ]

    for point_cloud in point_clouds_to_process:
        print(point_cloud)
        # try:

        parameters = {'directory'                        : '../',
                      'fileset'                          : 'test',
                      'input_point_cloud'                : point_cloud,
                      'model_filename'                   : '../model/model6_no_noise.pth',
                      'batch_size'                       : 20,
                      # could be interesting to see if batchsize of 1 is better
                      'box_dimensions'                   : [6, 6, 6],
                      'box_overlap'                      : [0.5, 0.5, 0.5],
                      'min_points_per_box'               : 1000,
                      'max_points_per_box'               : 20000,
                      'subsample'                        : False,
                      'subsampling_min_spacing'          : 0.01,
                      'num_procs'                        : 20,
                      'noise_class'                      : 0,
                      'terrain_class'                    : 1,
                      'vegetation_class'                 : 2,
                      'cwd_class'                        : 3,
                      'stem_class'                       : 4,

                      # DTM Settings
                      'coarse_grid_resolution'           : 6,
                      'fine_grid_resolution'             : 0.5,
                      'max_diameter'                     : 5,

                      # Measurement settings
                      'num_neighbours'                   : 4,
                      'slice_thickness'                  : 0.15,
                      'slice_increment'                  : 0.05,  # 05,
                      'diameter_measurement_increment'   : 0.1,
                      'diameter_measurement_height_range': 0.1,
                      'min_tree_volume'                  : 0.005,
                      'Canopy_coverage_resolution'       : 0.5,

                      # Filenaming settings
                      'Site'                             : 'not_specified',
                      'PlotID'                           : 'not_specified',
                      'UTM_zone_number'                  : 50,
                      'UTM_zone_letter'                  : None,
                      'UTM_is_north'                     : False,
                      }

        sem_seg = SemanticSegmentation(parameters)
        sem_seg.run_preprocessing()
        sem_seg.inference()

        object_1 = PostProcessing(parameters)
        object_1.process_point_cloud(point_cloud=sem_seg.output)
        # object_1.process_point_cloud(point_cloud=np.zeros((0,1)))
        del sem_seg

        # measure1 = MeasureTree(parameters)
        # measure1.run_measurement_extraction()
        del object_1
