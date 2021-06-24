import laspy
import numpy as np
from sklearn.neighbors import NearestNeighbors
from multiprocessing import Pool, get_context
import pandas as pd


def load_file(filename, plot_centre=None, plot_radius=0, silent=False):
    if not silent:
        print('Loading file...')
    file_extension = filename[-4:]
    if file_extension == '.las' or file_extension == '.laz':
        inFile = laspy.read(file)
        pointcloud = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()

    elif file_extension == '.csv':
        pointcloud = np.array(pd.read_csv(filename, header=None, index_col=None, delim_whitespace=True))

    if plot_centre is None and plot_radius > 0:
        plot_centre = np.mean(pointcloud[:, :2], axis=0)

        distances = np.linalg.norm(pointcloud[:, :2] - plot_centre, axis=1)
        keep_points = distances < plot_radius
        pointcloud = pointcloud[keep_points]

    return pointcloud


def save_file(filename, pointcloud, headers=None, silent=False):
    if not silent:
        print('Saving file...')
    if filename[-4:] == '.las':
        las = laspy.create(file_version="1.4", point_format=0)

        las.header.offsets = np.min(pointcloud, axis=0)
        las.header.scales = [0.0001, 0.0001, 0.0001]

        las.x = pointcloud[:, 0]
        las.y = pointcloud[:, 1]
        las.z = pointcloud[:, 2]

        if headers is not None:
            assert len(headers) == pointcloud.shape[1]

            for header in list(headers)[3:]:
                las.add_extra_dim(laspy.ExtraBytesParams(name=header, type="f8"))
                setattr(las, header, np.random.normal(0, 5, size=pointcloud.shape[0]))

        las.write(filename)
        print("Saved to:", filename)

    elif filename[-4:] == '.csv':
        pd.DataFrame(output).to_csv(filename, header=headers, index=None, sep=' ')
        print("Saved to:", filename)


# if __name__ == '__main__':
#
#     to_process_list = [
#             # 'C:/Users/seank/Documents/GitHub/FSCT/data/original_point_clouds/CULS_plot_1_annotated.csv',
#             'C:/Users/seank/Downloads/CULS/CULS/plot_1_annotated.las']
#
#     for file in to_process_list:
#         # las = np.asarray(laspy.read(file))
#         pointcloud = load_file(filename=file, plot_centre=None, plot_radius=0)
#         print(pointcloud.shape)
#         pointcloud = np.hstack((pointcloud, pointcloud))
#         headers = dict(x=0,
#                        y=1,
#                        z=2,
#                        a=3,
#                        b=4,
#                        c=5)
#
#         save_file('C:/Users/seank/Downloads/CULS/CULS/plot_1_annotated_mod.las', pointcloud, headers=headers)
