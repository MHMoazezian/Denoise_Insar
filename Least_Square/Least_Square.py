import numpy as np
import glob
import pandas as pd
from osgeo import gdal
import matplotlib.pyplot as plt
filenames = glob.glob('/media/miladmoazezian/3B07B9F56DF9B08D/Project_subsidence/Interferograms/*IW2.tif')
time1 = []
time2 = []

test_ds = gdal.Open(filenames[0])
test_data = test_ds.GetRasterBand(1).ReadAsArray()
for i in range(len(filenames)):
    time1.append(filenames[i][73:81])
    time2.append(filenames[i][82:90])
time = time1 + time2

Time = np.reshape(np.asarray(time).astype(int),[len(filenames) * 2 , 1])
Unique_Time = np.unique(Time)
Unique_name = Unique_Time.astype(str)
A = np.zeros([len(filenames) , np.shape(Unique_Time)[0]])
Desighn_Matrix = pd.DataFrame(A , columns=Unique_name)
Unique_name = Unique_Time.astype(str)
DELTA_PHASE = np.zeros([len(filenames) , 1])

for k in range(np.size(test_data)):
    for j in range(len(filenames)):
        ds = gdal.Open(filenames[j])
        DELTA_PHASE[j] = np.reshape(ds.GetRasterBand(2).ReadAsArray() , [np.size(test_data) , 1])[k]
        first_time = time1[j]
        second_time = time2[j]
        #ID1 = np.where(Desighn_Matrix.columns == first_time)
        #ID2 = np.where(Desighn_Matrix.columns == second_time)
        Desighn_Matrix[j , first_time] = 1
        Desighn_Matrix[j , second_time] = -1






plt.imshow(DATA)

















