import numpy as np
import traceback
import sys
import faulthandler
import pyvista as pv
from osgeo import gdal
import tifffile
import cutde.fullspace as FS
import scipy.io
import matplotlib.pyplot as plt
import concurrent.futures


array = tifffile.imread('/home/miladmoazezian/Desktop/SRTM_DEM.tif')[0:100 , 0:100]
Data = np.empty([np.shape(array)[0] * np.shape(array)[1] , 3])


for i in range(np.shape(array)[0]):
    for j in range(np.shape(array)[1]):
        Data[np.shape(array)[1] * i + j , :] = [i , j , array[i,j]]


plt.figure(figsize=(3, 10))
plt.tricontourf(Data[:,0], Data[:,1], Data[:,2])
plt.colorbar()
plt.show()


#
#
# nobs = 50
# W = 2000
# zoomx = [-W, W]
# zoomy = [-W, W]
# xs = np.linspace(*zoomx, nobs)
# ys = np.linspace(*zoomy, nobs)
# obsx, obsy = np.meshgrid(xs, ys)
# pts = np.array([obsx, obsy, 0 * obsy]).reshape((3, -1)).T.copy()
#
#
#



fault_L = 1000.0
fault_H = 1000.0
fault_D = 0.0
fault_pts = np.array(
    [
        [-fault_L, 0, -fault_D],
        [fault_L, 0, -fault_D],
        [fault_L, 0, -fault_D - fault_H],
        [-fault_L, 0, -fault_D - fault_H],
    ]
)

fault_tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
plt.triplot(fault_pts[:, 0], fault_pts[:, 2], fault_tris)
plt.xlabel("x")
plt.ylabel("z")
plt.show()


tris=fault_pts[fault_tris]

disp_mat = FS.disp_matrix(obs_pts=Data, tris=fault_pts[fault_tris], nu=0.9)

disp = np.sum(disp_mat[:, :, :, 0], axis=2)
disp = disp.reshape((array.shape[0], array.shape[1], 3))


plt.imshow(disp[:,:,0])




cloud = pv.PolyData(Data)
# cloud.plot(point_size=15)
#
mesh = cloud.delaunay_2d()
# surf.plot(show_edges=True)

points = mesh.points
# mesh = pv.Triangle()
mesh.is_all_triangles()
mesh_ok = mesh.extract_surface()
faces = mesh_ok.faces.reshape((-1, 4))[:,1:4]
faces
def Displacement(Data , faces , nu):
    Data = Data
    faces = faces
    nu = nu
    disp_mat = FS.disp_matrix(obs_pts=Data, tris=Data[faces], nu=nu)
    return disp_mat

disp_mat = Displacement(Data , faces , 0.25)


disp = np.nansum(disp_mat[:, :, :, 0], axis=2)
disp = disp.reshape((array.shape[0], array.shape[1], 3))


plt.figure(figsize=(13, 6))
fig , axe = plt.subplots(1,3)
axe[0].set_title('East Deformation')
axe[1].set_title('North Deformation')
axe[2].set_title('Up Deformation')
axe[0].imshow(disp[:,:,0])
axe[1].imshow(disp[:,:,1])
axe[2].imshow(disp[:,:,2])



tris2=Data[faces]

points = np.asarray(surf.points)

triangle = pv.Triangle([pointa, pointb, pointc])



np.save('surf.npy' , surf)


surf = np.load("surf.npy", allow_pickle=True)


(surf_pts_lonlat, surf_tris), (fault_pts_lonlat, fault_tris) = np.load("surf.npy", allow_pickle=True)


