from sklearn.decomposition import PCA 
from scipy import optimize
from scipy.spatial.transform import Rotation 
from scipy.stats import variation
import numpy as np

from matplotlib.patches import Circle
import matplotlib.pyplot as plt

def other_cylinder_fit2(xyz):
    
    from scipy.optimize import leastsq
    
    """
    https://stackoverflow.com/a/44164662/1414831
    
    This is a fitting for a vertical cylinder fitting
    Reference:
    http://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XXXIX-B5/169/2012/isprsarchives-XXXIX-B5-169-2012.pdf
    xyz is a matrix contain at least 5 rows, and each row stores x y z of a cylindrical surface
    p is initial values of the parameter;
    p[0] = Xc, x coordinate of the cylinder centre
    P[1] = Yc, y coordinate of the cylinder centre
    P[2] = alpha, rotation angle (radian) about the x-axis
    P[3] = beta, rotation angle (radian) about the y-axis
    P[4] = r, radius of the cylinder
    th, threshold for the convergence of the least squares
    """   
    
    xm = np.median(xyz.z)
    ym = np.median(xyz.y)
    
    p = np.array([xm, # x centre
                  ym, # y centre
                  0, # alpha, rotation angle (radian) about the x-axis
                  0, # beta, rotation angle (radian) about the y-axis
                  np.ptp(xyz.z) / 2
                  ])

    x = xyz.x
    y = xyz.y
    z = xyz.z

    fitfunc = lambda p, x, y, z: (- np.cos(p[3])*(p[0] - x) - z*np.cos(p[2])*np.sin(p[3]) - np.sin(p[2])*np.sin(p[3])*(p[1] - y))**2 + (z*np.sin(p[2]) - np.cos(p[2])*(p[1] - y))**2 #fit function
    errfunc = lambda p, x, y, z: fitfunc(p, x, y, z) - p[4]**2 #error function 

    est_p, success = leastsq(errfunc, p, args=(x, y, z), maxfev=1000)

    x, y, a, b, rad = est_p
    centre = np.array([x, y])

    return np.abs(rad), centre


def RANSACcylinderFitting3(xyz_, ax, iterations=50, N=100, plot=False):
    
    if plot:
        ax = plt.subplot(111)
    
    bestFit, bestErr = None, np.inf
    
#     try:
    for i in range(iterations):

        xyz = xyz_.copy()
        
        sample = xyz.sample(n=min(max(10, int(len(xyz) / 10)), N))        

        pca = PCA(n_components=3, svd_solver='auto').fit(sample[['x', 'y', 'z']])
        sample[['x', 'y', 'z']] = pca.transform(sample[['x', 'y', 'z']])

        xyz[['x', 'y', 'z']] = pca.transform(xyz[['x', 'y', 'z']])
        xyz = xyz.loc[~xyz.index.isin(sample.index)]

#         xyz.plot.scatter('x', 'z', ax=ax, c=['r', 'g', 'b'][i])

        radius, centre = other_cylinder_fit2(sample)
        if not np.all(np.isclose(centre, 0, atol=radius*1.05)): continue

        xyz.loc[:, 'error'] = np.abs(np.linalg.norm(xyz[['x', 'y']] - centre, axis=1)) / radius
        alsoInliers = xyz.loc[xyz.error.between(.9, 1.1)] # 10% of radius is prob quite large

        if variation(alsoInliers.error) < bestErr and len(alsoInliers) > len(xyz) * .1:

            # for testing uncomment
            c = Circle(centre, radius=radius, facecolor='none', edgecolor='g')

            Centre = np.hstack([centre, sample.z.mean()])
            Centre = pca.inverse_transform(Centre)

            bestFit = [radius, centre, pca, c, alsoInliers]
            bestErr = variation(alsoInliers.error)

#     except:
#         return np.nan, xyz[['x', 'y', 'z']].mean(axis=0).values, np.inf, len(xyz_)

    if bestFit == None: 
        # usually caused by low number of ransac iterations
        return np.nan, xyz[['x', 'y', 'z']].mean(axis=0).values, np.inf, len(xyz_)
    
    # for testing uncomment
    if plot:
        xyzt = bestFit[2].transform(xyz_[['x', 'y', 'z']]) 
        ax.scatter(xyzt[:, 0], xyzt[:, 1], s=1, c='grey')

        allInliers = bestFit[4]
        cbar = ax.scatter(allInliers.x, allInliers.y, s=10, c=allInliers.error)
        plt.colorbar(cbar)

        ax.scatter(bestFit[1][0], bestFit[1][1], marker='+', s=100, c='r')
        ax.add_patch(bestFit[3])

    return [radius, Centre, bestErr, len(xyz_)]

def NotRANSAC(xyz):
    
    try:
        xyz = xyz[['x', 'y', 'z']]
        pca = PCA(n_components=3, svd_solver='auto').fit(xyz)
        xyz[['x', 'y', 'z']] = pca.transform(xyz)
        radius, centre = other_cylinder_fit2(xyz)
        
        if xyz.z.min() - radius < centre[0] < xyz.z.max() + radius or \
           xyz.y.min() - radius < centre[1] < xyz.y.max() + radius: 
            centre = np.hstack([xyz.x.mean(), centre])
        else:
            centre = xyz.mean().values
        
        centre = pca.inverse_transform(centre)
    except:
        radius, centre = np.nan, xyz[['x', 'y', 'z']].mean(axis=0).values
    
    return [radius, centre, np.inf, len(xyz)]

def RANSAC_helper(xyz, ransac_iterations, sample, plot=False):

#     try:
        if len(xyz) == 0: # don't think this is required but....
            cylinder = [np.nan, np.array([np.inf, np.inf, np.inf]), np.inf, len(xyz)]
        elif len(xyz) < 10:
            cylinder = [np.nan, xyz[['x', 'y', 'z']].mean(axis=0).values, np.inf, len(xyz)]
        elif len(xyz) < 50:
            cylinder = NotRANSAC(xyz)
        else:
            cylinder = RANSACcylinderFitting3(xyz, ransac_iterations, sample, plot=plot)
#             if cylinder == None: # again not sure if this is necessary...
#                 cylinder = [np.nan, xyz[['x', 'y', 'z']].mean(axis=0)]
                
#     except:
#         cylinder = [np.nan, xyz[['x', 'y', 'z']].mean(axis=0), np.inf, np.inf]

        return cylinder
