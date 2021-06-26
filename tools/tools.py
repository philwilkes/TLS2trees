from sklearn.neighbors import NearestNeighbors
import numpy as np
import glob
import laspy
from sklearn.neighbors import NearestNeighbors
from multiprocessing import Pool, get_context
import pandas as pd
import os
import shutil
from sklearn.cluster import DBSCAN
from scipy.interpolate import griddata


def make_folder_structure(filename):
    filename = filename.replace('\\', '/')
    directory = os.path.dirname(os.path.realpath(filename)) + '/'
    filename = filename.split('/')[-1][:-4]
    output_dir = directory + filename+'_FSCT_output/'
    working_dir = directory + filename+'_FSCT_output/working_directory/'

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if not os.path.isdir(working_dir):
        os.makedirs(working_dir)
    else:
        shutil.rmtree(working_dir, ignore_errors=True)
        os.makedirs(working_dir)

    return output_dir, working_dir


def subsample_point_cloud(X, min_spacing):
    """

    Args:
        X: The input point cloud.
        min_spacing: The minimum allowable distance between two points in the point cloud.

    Returns:
        X: The subsampled point cloud.
    """
    print("Subsampling...")
    neighbours = NearestNeighbors(n_neighbors=2, algorithm='kd_tree', metric='euclidean').fit(X[:, :3])
    distances, indices = neighbours.kneighbors(X[:, :3])
    X_keep = X[distances[:, 1] >= min_spacing]
    i1 = [distances[:, 1] < min_spacing][0]
    i2 = [X[indices[:, 0], 2] < X[indices[:, 1], 2]][0]
    X_check = X[np.logical_and(i1, i2)]

    while np.shape(X_check)[0] > 1:
        neighbours = NearestNeighbors(n_neighbors=2, algorithm='kd_tree', metric='euclidean').fit(X_check[:, :3])
        distances, indices = neighbours.kneighbors(X_check[:, :3])
        X_keep = np.vstack((X_keep, X_check[distances[:, 1] >= min_spacing, :]))
        i1 = [distances[:, 1] < min_spacing][0]
        i2 = [X_check[indices[:, 0], 2] < X_check[indices[:, 1], 2]][0]
        X_check = X_check[np.logical_and(i1, i2)]
    # X = np.delete(X,np.unique(indices[distances[:,1]<min_spacing]),axis=0)
    X = X_keep
    return X


def load_file(filename, plot_centre=None, plot_radius=0, silent=False, headers_of_interest = []):
    if not silent:
        print('Loading file...', filename)
    file_extension = filename[-4:]
    output_headers = []

    if file_extension == '.las' or file_extension == '.laz':
        inFile = laspy.read(filename)
        header_names = list(inFile.point_format.dimension_names)
        print(header_names)
        coord_headers = ['X', 'Y', 'Z']
        pointcloud = np.vstack((inFile.x, inFile.y, inFile.z))
        if len(headers_of_interest) != 0:
            for header in headers_of_interest:
                if header in header_names:
                    print(header)
                    pointcloud = np.vstack((pointcloud, getattr(inFile, header)))
                    output_headers.append(header)
        pointcloud = pointcloud.transpose()

    elif file_extension == '.csv':
        pointcloud = np.array(pd.read_csv(filename, header=None, index_col=None, delim_whitespace=True))

    if plot_centre is None and plot_radius > 0:
        plot_centre = np.mean(pointcloud[:, :2], axis=0)

        distances = np.linalg.norm(pointcloud[:, :2] - plot_centre, axis=1)
        keep_points = distances < plot_radius
        pointcloud = pointcloud[keep_points]
    return pointcloud, coord_headers + output_headers


def save_file(filename, pointcloud, headers=[], silent=False):
    if pointcloud.shape[0] == 0:
        print(filename,'is empty...')
    else:
        if not silent:
            print('Saving file...')
        if filename[-4:] == '.las':
            las = laspy.create(file_version="1.4", point_format=6)
            las.header.offsets = np.min(pointcloud[:, :3], axis=0)
            las.header.scales = [0.0001, 0.0001, 0.0001]

            las.x = pointcloud[:, 0]
            las.y = pointcloud[:, 1]
            las.z = pointcloud[:, 2]

            if len(headers) != 0:
                #  The reverse step below just puts the headings in the preferred order. They are backwards without it.
                headers = headers[3:]  # Trim off X, Y, Z
                headers.reverse()
                col_idxs = list(range(3, pointcloud.shape[1]))
                col_idxs.reverse()
                for header, i in zip(headers, col_idxs):
                    column = pointcloud[:, i]
                    las.add_extra_dim(laspy.ExtraBytesParams(name=header, type="f8"))
                    setattr(las, header, column)
            header_names = list(las.point_format.dimension_names)
            las.write(filename)
            if not silent:
                print("Saved to:", filename)

        elif filename[-4:] == '.csv':
            pd.DataFrame(pointcloud).to_csv(filename, header=None, index=None, sep=' ')
            print("Saved to:", filename)


def get_heights_above_DTM(points, DTM):
    grid = griddata((DTM[:, 0], DTM[:, 1]), DTM[:, 2], points[:, 0:2], method='linear',
                    fill_value=np.median(DTM[:, 2]))
    points[:, -1] = points[:, 2] - grid
    return points


def clustering(points, eps=0.05, min_samples=2):
    print("Clustering...")
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', algorithm='kd_tree', n_jobs=-1).fit(points[:, :3])
    return np.hstack((points, np.atleast_2d(db.labels_).T))



# if __name__=='__main__':
#     pc, headers = load_file('C:/Users/seank/Downloads/CULS/CULS/plot_1_annotated_FSCT_output/full_cyl_array.las')