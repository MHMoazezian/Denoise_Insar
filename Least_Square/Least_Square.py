import numpy as np
import glob
import pandas as pd
from osgeo import gdal
import matplotlib.pyplot as plt
import sys
import warnings
warnings.filterwarnings("ignore")
filenames = glob.glob('/media/miladmoazezian/3B07B9F56DF9B08D/Project_subsidence/Interferograms/*IW2.tif')
time1 = []
time2 = []

test_ds = gdal.Open(filenames[0])
test_data = test_ds.GetRasterBand(1).ReadAsArray()
GT_test = test_ds.GetGeoTransform()
for i in range(len(filenames)):
    time1.append(filenames[i][73:81])
    time2.append(filenames[i][82:90])
time = time1 + time2

Time = np.reshape(np.asarray(time).astype(int),[len(filenames) * 2 , 1])
Unique_Time = np.unique(Time)
Unique_name = Unique_Time.astype(str)
A = np.zeros([len(filenames) , np.shape(Unique_Time)[0]])
W = np.zeros([len(filenames) , len(filenames)])
Desighn_Matrix = pd.DataFrame(A , columns=Unique_name)
Unique_name = Unique_Time.astype(str)
DELTA_PHASE = np.zeros([len(filenames) , 1])
Phase = np.zeros([len(Unique_Time), np.shape(test_data)[0] , np.shape(test_data)[1]])

for row in range(np.shape(test_data)[0]):
    for col in range(np.shape(test_data)[1]):
        if test_data[row , col] == 0:
            break
        x_geo = GT_test[1] * col + GT_test[2] * row + GT_test[1] * 0.5 + GT_test[2] * 0.5 + GT_test[0]
        y_geo = GT_test[4] * col + GT_test[5] * row + GT_test[4] * 0.5 + GT_test[5] * 0.5 + GT_test[3]


        for j in range(len(filenames)):
            print(j)
            ds = gdal.Open(filenames[j])
            GT = ds.GetGeoTransform()
            c, a, b, f, d, e = GT
            new_col = int((x_geo - c) / a)
            new_row = int((y_geo - f) / e)
            DELTA_PHASE[j] = ds.GetRasterBand(2).ReadAsArray()[new_row , new_col]
            W[j,j] = ds.GetRasterBand(3).ReadAsArray()[new_row , new_col]
            first_time = time1[j]
            second_time = time2[j]
            ID1 = np.where(Desighn_Matrix.columns == first_time)
            ID2 = np.where(Desighn_Matrix.columns == second_time)
            Desighn_Matrix.iloc[j , ID1] = 1
            Desighn_Matrix.iloc[j , ID2] = -1
        print(row , col)
        Desighn_Matrix_array = np.asarray(Desighn_Matrix)
        Phase[: , row , col] = np.reshape(np.matmul(np.matmul(np.matmul(np.linalg.inv(np.matmul(np.matmul(Desighn_Matrix_array.T, W), Desighn_Matrix_array)), Desighn_Matrix_array.T),W), DELTA_PHASE) , [len(Unique_Time) ,])






#plt.imshow(Phase)
















