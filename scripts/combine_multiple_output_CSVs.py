import pandas as pd

from run_tools import FSCT, directory_mode, file_mode


def combine_multiple_output_CSVs(point_clouds_to_process):
    combined_dataframe = pd.read_csv(point_clouds_to_process[0][:-4]+'_FSCT_output/processing_report.csv', index_col=[0])
    if len(point_clouds_to_process) > 1:
        for point_cloud_filename in point_clouds_to_process[1:]:
            report_filename = point_cloud_filename[:-4]+'_FSCT_output/processing_report.csv'
            # print(report_filename)
            combined_dataframe = pd.concat([combined_dataframe, pd.read_csv(report_filename, index_col=[0])])
    return combined_dataframe


def get_lowest_common_directory(point_clouds_to_process):
    path_list = []
    for filepath in point_clouds_to_process:
        path_list.append(filepath.split('/'))

    path_set = set(path_list[0])
    while len(path_list) > 1:
        path_list = path_list[1:]
        path_set = path_set.intersection(path_list[0])
    output_directory = []
    for path in path_list:
        for directory in path:
            if directory in path_set:
                output_directory.append(directory)

    output_directory = '/'.join(output_directory)
    return output_directory


if __name__ == '__main__':
    """Choose one of the following or modify as needed.
    Directory mode will find all .las files within a directory and sub directories but will ignore any .las files in
    folders with "FSCT_output" in their names.

    File mode will allow you to select multiple .las files within a directory.

    Alternatively, you can just list the point cloud file paths.
    
    This will find the outputs of the FSCT processing for all of these point clouds and combine the "plot_reports".
    """
    # point_clouds_to_process = directory_mode()
    point_clouds_to_process = file_mode()

    output_directory = get_lowest_common_directory(point_clouds_to_process)
    combined_dataframe = combine_multiple_output_CSVs(point_clouds_to_process)
    combined_dataframe.to_csv(output_directory+'/combined_processing_reports.csv', sep=',')



