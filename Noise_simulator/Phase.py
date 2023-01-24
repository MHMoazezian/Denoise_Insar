import glob
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

import sys
sys.path.append('/home/miladmoazezian/.snap/snap-python')
# import snappy

MONTH = ['Jan' , 'Feb' , 'Mar' , 'Apr' , 'May' , 'Jun' , 'Jul' , 'Aug' , 'Sep' , 'Oct' , 'Nov' , 'Dec']

# DEM(xy)
ds = gdal.Open(r"/media/data/Project_subsidence/DEMs/SRTM_DEM.tif")
ulx, xres, xskew, uly, yskew, yres = ds.GetGeoTransform()
lrx = ulx + (ds.RasterXSize * xres)
lry = uly + (ds.RasterYSize * yres)
rasterArray = ds.ReadAsArray()
DEM = np.asarray(rasterArray.flatten())

cols = ds.RasterXSize
rows = ds.RasterYSize
# plt.imshow(rasterArray)
array_ = ds.ReadAsArray()
transform = ds.GetGeoTransform()
xOrigin = transform[0]
yOrigin = transform[3]
pixelWidth = transform[1]
pixelHeight = -transform[5]

LatDem = np.zeros(np.shape(array_))
LonDem = np.zeros(np.shape(array_))
LonDem[:, 0] = xOrigin
for k in range(np.shape(array_)[1] - 1):
    LonDem[:, k + 1] = LonDem[:, k] + pixelWidth
LatDem[0, :] = yOrigin
for l in range(np.shape(array_)[0] - 1):
    LatDem[l + 1, :] = LatDem[l, :] - pixelHeight

pp_Dem = np.vstack((LatDem.flatten(), LonDem.flatten())).T
point_Dem = pp_Dem[~np.isnan(pp_Dem).any(axis=1)]
XY_Dem = list(zip(point_Dem[:, 0], point_Dem[:, 1]))

# SAR(xy)
filenames_SAR = glob.glob(r"/media/miladmoazezian/3B07B9F56DF9B08D/Project_subsidence/Interferograms/*_IW2.tif")
filenames_Range = glob.glob(r"/media/data/Project_subsidence/Interferograms/*_Range.tif")

#plt.imshow(rasterArray)


import os
os.environ['PROJ_LIB'] = '/home/miladmoazazian/anaconda3/envs/Phase/share/proj'
os.environ['GDAL_DATA'] = '/home/miladmoazazian/anaconda3/envs/Phase/share'



for i in range(len(filenames_SAR)):
    product = snappy.ProductIO.readProduct(filenames_SAR[i])
    time = filenames_SAR[i][46:63]
    month_master = MONTH[int(time[4:6]) - 1]
    day_master = time[6:8]
    year_master = time[0:4]
    month_slave = MONTH[int(time[13:15]) - 1]
    day_slave = time[15:17]
    year_slave = time[9:13]

    Perp_Baseline = product.getMetadataRoot().getElement('Abstracted_Metadata').getElement('Baselines').getElement('Master: ' + day_master + month_master + year_master).getElement('Slave: ' + day_slave + month_slave + year_slave).getAttribute('Perp Baseline').getData().getElemFloat()
    Frequency = product.getMetadataRoot().getElement('Abstracted_Metadata').getAttribute('radar_frequency').getData().getElemFloat()
    wave_length = (299792458) / (Frequency * (1e+6))
    ds2 = gdal.Open(filenames_SAR[i])
    ds3 = gdal.Open(filenames_Range[i])
    ulx2, xres2, xskew2, uly2, yskew2, yres2 = ds2.GetGeoTransform()
    ulx3, xres3, xskew3, uly3, yskew3, yres3 = ds3.GetGeoTransform()

    Slant_Range = ds3.GetRasterBand(1).ReadAsArray().flatten() * 0.299792458
    Intensity = ds2.GetRasterBand(1).ReadAsArray().flatten()
    Phase = ds2.GetRasterBand(2).ReadAsArray().flatten()
    Coherency = ds2.GetRasterBand(3).ReadAsArray().flatten()
    Latitude_SAR = ds2.GetRasterBand(4).ReadAsArray().flatten()
    Longitude_SAR = ds2.GetRasterBand(5).ReadAsArray().flatten()
    Incidence_angle = ds2.GetRasterBand(6).ReadAsArray().flatten()
    Amplitude = np.sqrt(Intensity)



    lrx2 = ulx2 + (ds2.RasterXSize * xres2)
    lry2 = uly2 + (ds2.RasterYSize * yres2)
    rasterArray2 = ds2.ReadAsArray()
    cols2 = ds2.RasterXSize
    rows2 = ds2.RasterYSize
    array_2 = ds2.GetRasterBand(1).ReadAsArray()
    transform2 = ds2.GetGeoTransform()
    xOrigin2 = transform2[0]
    yOrigin2 = transform2[3]
    pixelWidth2 = transform2[1]
    pixelHeight2 = -transform2[5]

    LatSar = np.zeros(np.shape(array_2))
    LonSar = np.zeros(np.shape(array_2))
    LonSar[:, 0] = xOrigin2
    for k in range(np.shape(array_2)[1] - 1):
        LonSar[:, k + 1] = LonSar[:, k] + pixelWidth2
    LatSar[0, :] = yOrigin2
    for l in range(np.shape(array_2)[0] - 1):
        LatSar[l + 1, :] = LatSar[l, :] - pixelHeight2

    pp_Sar = np.vstack((LatSar.flatten(), LonSar.flatten())).T
    point_Sar = pp_Sar[~np.isnan(pp_Sar).any(axis=1)]
    XY_Sar = list(zip(point_Sar[:, 0], point_Sar[:, 1]))
    ##############################################################################################

    lrx3 = ulx3 + (ds3.RasterXSize * xres3)
    lry3 = uly3 + (ds3.RasterYSize * yres3)
    rasterArray3 = ds3.ReadAsArray()
    cols3 = ds3.RasterXSize
    rows3 = ds3.RasterYSize
    array_3 = ds3.GetRasterBand(1).ReadAsArray()
    transform3 = ds3.GetGeoTransform()
    xOrigin3 = transform3[0]
    yOrigin3 = transform3[3]
    pixelWidth3 = transform3[1]
    pixelHeight3 = -transform3[5]

    LatRange= np.zeros(np.shape(array_3))
    LonRange = np.zeros(np.shape(array_3))
    LonRange[:, 0] = xOrigin3
    for k in range(np.shape(array_3)[1] - 1):
        LonRange[:, k + 1] = LonRange[:, k] + pixelWidth3
    LatRange[0, :] = yOrigin3
    for l in range(np.shape(array_3)[0] - 1):
        LatRange[l + 1, :] = LatRange[l, :] - pixelHeight3

    pp_Range = np.vstack((LatRange.flatten(), LonRange.flatten())).T
    point_Range = pp_Range[~np.isnan(pp_Range).any(axis=1)]
    XY_Range = list(zip(point_Range[:, 0], point_Range[:, 1]))

    #####################Resampling######################################################
    tree1 = KDTree(np.asarray(XY_Sar))
    dist1, ind = tree1.query(XY_Dem, k=1)
    distance1 = dist1 * 111

    tree2 = KDTree(np.asarray(XY_Range))
    dist2, ind2 = tree2.query(XY_Dem, k=1)
    distance2 = dist2 * 111


    NEW_PHASE = np.zeros(np.shape(DEM))
    NEW_Coherency = np.zeros(np.shape(DEM))
    NEW_Incidence_angle = np.zeros(np.shape(DEM))
    NEW_Amplitude = np.zeros(np.shape(DEM))
    NEW_Intensity = np.zeros(np.shape(DEM))
    NEW_Slant_Range = np.zeros(np.shape(DEM))

    for ii in range(np.shape(NEW_PHASE)[0]):
        NEW_PHASE[ii] = Phase[ind[ii, 0]]
        NEW_Coherency[ii] = Coherency[ind[ii, 0]]
        NEW_Incidence_angle[ii] = Incidence_angle[ind[ii, 0]]
        NEW_Amplitude[ii] = Amplitude[ind[ii, 0]]
        NEW_Intensity[ii] = Intensity[ind[ii, 0]]
        NEW_Slant_Range[ii] = Slant_Range[ind2[ii, 0]]

    NEW_PHASE1 = np.reshape(NEW_PHASE, np.shape(array_))[:,0:407]
    NEW_Coherency1 = np.reshape(NEW_Coherency, np.shape(array_))[:,0:407]
    NEW_Amplitude1 = np.reshape(NEW_Amplitude, np.shape(array_))[:,0:407]
    NEW_Incidence_angle1 = np.reshape(NEW_Incidence_angle, np.shape(array_))[:,0:407]
    NEW_Intensity1 = np.reshape(NEW_Intensity, np.shape(array_))[:,0:407]
    NEW_Slant_Range1 = np.reshape(NEW_Slant_Range, np.shape(array_))[:,0:407]

    landa = wave_length
    #plt.imshow(NEW_Slant_Range1)
    """
    incidence_angle = np.reshape(np.linspace(30 , 46 , 430) , [1 , array_.shape[1]])
    Incidence_angle = np.repeat(incidence_angle , repeats=341 , axis=0)
    plt.imshow(Incidence_angle)
    slant_range = np.reshape(np.linspace(0.00282340207626 , 0.00301340835728 , 430) , [1 , array_.shape[1]]) * 299792458
    Slant_range = np.repeat(slant_range , repeats=341 , axis=0 )
    plt.imshow(Slant_range)
    B_prependecular = np.linspace(-300 , +300  , 1000)

    """
    array_New = array_[:,0:407]
    Perp_Baseline_New = abs(Perp_Baseline) + np.random.randint(200 , 1600)
    Phase_denoise = (((4 * np.pi) / landa) * ((int(Perp_Baseline_New) * array_New) / (NEW_Slant_Range1 * np.sin(np.deg2rad(NEW_Incidence_angle1)))))

    #plt.imshow(Phase_denoise, cmap='jet')


    def write_geotiff(filename, arr1, arr2, arr3, arr4):
        if arr1.dtype == np.float64:
            arr1_type = gdal.GDT_Float64
        else:
            arr1_type = gdal.GDT_Int64

        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(filename, arr1.shape[1], arr1.shape[0], 4, arr1_type)
        out_ds.SetProjection(ds.GetProjection())
        out_ds.SetGeoTransform(ds.GetGeoTransform())
        band1 = out_ds.GetRasterBand(1)
        band1.WriteArray(arr1)
        band1.FlushCache()
        band1.ComputeStatistics(False)

        band2 = out_ds.GetRasterBand(2)
        band2.WriteArray(arr2)
        band2.FlushCache()
        band2.ComputeStatistics(False)

        band3 = out_ds.GetRasterBand(3)
        band3.WriteArray(arr3)
        band3.FlushCache()
        band3.ComputeStatistics(False)

        band4 = out_ds.GetRasterBand(4)
        band4.WriteArray(arr4)
        band4.FlushCache()
        band4.ComputeStatistics(False)


    write_geotiff("/media/data/Project_subsidence/PHASE/" + time + ".tif", NEW_Amplitude1, NEW_Coherency1, Phase_denoise, NEW_PHASE1)




# test
arr = gdal.Open('/media/miladmoazezian/3B07B9F56DF9B08D/Project_subsidence/DEMs/SRTM_DEM.tif').GetRasterBand(1).ReadAsArray()