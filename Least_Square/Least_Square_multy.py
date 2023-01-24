import numpy as np
import glob
import pandas as pd
from osgeo import gdal
import concurrent.futures
import matplotlib.pyplot as plt
import sys
import time
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
filenames = glob.glob('/media/miladmoazezian/3B07B9F56DF9B08D/Project_subsidence/Interferograms/*IW2.tif')[:90]
time1 = []
time2 = []

test_ds = gdal.Open(filenames[0])
test_data = test_ds.GetRasterBand(1).ReadAsArray()

GT_test = test_ds.GetGeoTransform()

for i in range(len(filenames)):
    time1.append(filenames[i][73:81])
    time2.append(filenames[i][82:90])
timee = time1 + time2

Time = np.reshape(np.asarray(timee).astype(int),[len(filenames) * 2 , 1])
Unique_Time = np.unique(Time)
Unique_name = Unique_Time.astype(str)
A = np.zeros([len(filenames) , np.shape(Unique_Time)[0]])
W = np.zeros([len(filenames) , len(filenames)])
Desighn_Matrix = pd.DataFrame(A , columns=Unique_name)
Unique_name = Unique_Time.astype(str)
DELTA_PHASE = np.zeros([len(filenames) , 1])



def Matrix_Maker(filename , x_geo , y_geo , j):
    start_time = time.time()
    ds = gdal.Open(filename)
    end_time = time.time()
    GT = ds.GetGeoTransform()
    c, a, b, f, d, e = GT
    new_col = int((x_geo - c) / a)
    new_row = int((y_geo - f) / e)
    DELTA_PHASE[j] = ds.GetRasterBand(2).ReadAsArray()[new_row, new_col]
    W[j, j] = ds.GetRasterBand(3).ReadAsArray()[new_row, new_col]
    first_time = time1[j]
    second_time = time2[j]
    ID1 = np.where(Desighn_Matrix.columns == first_time)
    ID2 = np.where(Desighn_Matrix.columns == second_time)
    Desighn_Matrix.iloc[j, ID1] = 1
    Desighn_Matrix.iloc[j, ID2] = -1
    time_function = end_time - start_time
    print(time_function)
    return Desighn_Matrix


def Phase_least_square(Image_array):
    start = time.time()
    A = np.zeros([len(filenames), np.shape(Unique_Time)[0]])
    W = np.zeros([len(filenames), len(filenames)])
    Desighn_Matrix = pd.DataFrame(A, columns=Unique_name)
    DELTA_PHASE = np.zeros([len(filenames), 1])
    for row in range(np.shape(Image_array)[0]):
        for col in range(np.shape(Image_array)[1]):
            if Image_array[row , col] == 0:
                break
            x_geo = GT_test[1] * col + GT_test[2] * row + GT_test[1] * 0.5 + GT_test[2] * 0.5 + GT_test[0]
            y_geo = GT_test[4] * col + GT_test[5] * row + GT_test[4] * 0.5 + GT_test[5] * 0.5 + GT_test[3]
            with concurrent.futures.ThreadPoolExecutor(max_workers=20000) as excuter:
                result = [excuter.submit(Matrix_Maker , filename ,x_geo , y_geo , j) for j , filename in enumerate(filenames)]
                for f in concurrent.futures.as_completed(result):
                    Desighn_Matrix = f.result()

    # print(Desighn_Matrix.iloc[0,0])
    Desighn_Matrix = np.asarray(Desighn_Matrix)
    Phase = np.matmul(np.matmul(np.matmul(np.linalg.inv(np.matmul(np.matmul(Desighn_Matrix.T , W) , Desighn_Matrix)) , Desighn_Matrix.T) , W) , DELTA_PHASE)
    end = time.time()
    duration = start - end
    return Phase , duration





Phase , duration = Phase_least_square(test_data)

ds = gdal.Open('/home/miladmoazezian/Downloads/20220206_20220302.geo.diff_unfiltered_pha.tif')
array = ds.GetRasterBand(1).ReadAsArray()
plt.imshow(array , cmap='jet')


