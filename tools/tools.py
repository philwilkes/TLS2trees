from sklearn.neighbors import NearestNeighbors
import numpy as np
import glob
import laspy
from sklearn.neighbors import NearestNeighbors
from multiprocessing import Pool, get_context
import pandas as pd


def subsample_point_cloud(X, min_spacing):
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


def load_file(filename, plot_centre=None, plot_radius=0, silent=False):
    if not silent:
        print('Loading file...', filename)
    file_extension = filename[-4:]
    if file_extension == '.las' or file_extension == '.laz':
        inFile = laspy.read(filename)
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
