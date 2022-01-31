from sklearn.decomposition import PCA 
from scipy import optimize
from scipy.spatial.transform import Rotation 
from scipy.stats import variation
import numpy as np

from matplotlib.patches import Circle
import matplotlib.pyplot as plt

from tqdm.auto import tqdm


def other_cylinder_fit2(xyz, xm=0, ym=0, xr=0, yr=0, r=1):
    
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
    
    x = xyz.x
    y = xyz.y
    z = xyz.z
    
    p = np.array([xm, ym, xr, yr, r])

    fitfunc = lambda p, x, y, z: (- np.cos(p[3])*(p[0] - x) - z*np.cos(p[2])*np.sin(p[3]) - np.sin(p[2])*np.sin(p[3])*(p[1] - y))**2 + (z*np.sin(p[2]) - np.cos(p[2])*(p[1] - y))**2 #fit function
    errfunc = lambda p, x, y, z: fitfunc(p, x, y, z) - p[4]**2 #error function 

    est_p, success = leastsq(errfunc, p, args=(x, y, z), maxfev=1000)
    
    return est_p

def RANSACcylinderFitting4(xyz_, iterations=50, plot=False):
    
    if plot:
        ax = plt.subplot(111)
    
    bestFit, bestErr = None, np.inf
    xyz_mean = xyz_.mean(axis=0)
    xyz_ -= xyz_mean

#     for i in tqdm(range(iterations), total=iterations, display=plot):
    for i in range(iterations):
        
        xyz = xyz_.copy()
        
        # prepare sample 
        sample = xyz.sample(n=20)  
        xyz = xyz.loc[~xyz.index.isin(sample.index)]
        
        x, y, a, b, radius = other_cylinder_fit2(sample, 0, 0, 0, 0, 0)
        centre = (x, y)
        if not np.all(np.isclose(centre, 0, atol=radius*1.05)): continue
        
        MX = Rotation.from_euler('xy', [a, b]).inv()
        xyz[['x', 'y', 'z']] = MX.apply(xyz)
        xyz.loc[:, 'error'] = np.linalg.norm(xyz[['x', 'y']] - centre, axis=1) / radius
        idx = xyz.loc[xyz.error.between(.8, 1.2)].index # 40% of radius is prob quite large
        
        # select points which best fit model from original dataset
        alsoInliers = xyz_.loc[idx].copy()
        if len(alsoInliers) < len(xyz_) * .2: continue # skip if no enough points chosen
        
        # refit model using new params
        x, y, a, b, radius = other_cylinder_fit2(alsoInliers, x, y, a, b, radius)
        centre = [x, y]
        if not np.all(np.isclose(centre, 0, atol=radius*1.05)): continue

        MX = Rotation.from_euler('xy', [a, b]).inv()
        alsoInliers[['x', 'y', 'z']] = MX.apply(alsoInliers[['x', 'y', 'z']])
        # calculate error for "best" subset
        alsoInliers.loc[:, 'error'] = np.linalg.norm(alsoInliers[['x', 'y']] - centre, axis=1) / radius      

        if variation(alsoInliers.error) < bestErr:
        
            # for testing uncomment
            c = Circle(centre, radius=radius, facecolor='none', edgecolor='g')

            bestFit = [radius, centre, c, alsoInliers, MX]
            bestErr = variation(alsoInliers.error)

    if bestFit == None: 
        # usually caused by low number of ransac iterations
        return np.nan, xyz[['x', 'y', 'z']].mean(axis=0).values, np.inf, len(xyz_)
    
    radius, centre, c, alsoInliers, MX = bestFit
    centre[0] += xyz_mean.x
    centre[1] += xyz_mean.y
    centre = centre + [xyz_mean.z]
    
    # for testing uncomment
    if plot:
        
        radius, Centre, c, alsoInliers, MX = bestFit

        xyz_[['x', 'y', 'z']] = MX.apply(xyz_)
        xyz_ += xyz_mean
        ax.scatter(xyz_.x, xyz_.y,  s=1, c='grey')

        alsoInliers[['x', 'y', 'z']] += xyz_mean
        cbar = ax.scatter(alsoInliers.x, alsoInliers.y, s=10, c=alsoInliers.error)
        plt.colorbar(cbar)

        ax.scatter(Centre[0], Centre[1], marker='+', s=100, c='r')
        ax.add_patch(c)

    return [radius, centre, bestErr, len(xyz_)]

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

def RANSAC_helper(xyz, ransac_iterations, plot=False):

#     try:
        if len(xyz) == 0: # don't think this is required but....
            cylinder = [np.nan, np.array([np.inf, np.inf, np.inf]), np.inf, len(xyz)]
        elif len(xyz) < 10:
            cylinder = [np.nan, xyz[['x', 'y', 'z']].mean(axis=0).values, np.inf, len(xyz)]
        elif len(xyz) < 50:
            cylinder = NotRANSAC(xyz)
        else:
            cylinder = RANSACcylinderFitting4(xyz[['x', 'y', 'z']], iterations=ransac_iterations, plot=plot)
#             if cylinder == None: # again not sure if this is necessary...
#                 cylinder = [np.nan, xyz[['x', 'y', 'z']].mean(axis=0)]
                
#     except:
#         cylinder = [np.nan, xyz[['x', 'y', 'z']].mean(axis=0), np.inf, np.inf]

        return cylinder
