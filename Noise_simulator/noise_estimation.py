import glob as glob
import numpy as np
import math
from skimage import io
import imagesc as imagesc
from osgeo import gdal
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pyarrow as pa
import pyarrow.parquet as pq

filenames = glob.glob(r"/media/labmanager/3B07B9F56DF9B08D/Project_subsidence/PHASE/*.tif")
scaler1 = preprocessing.MinMaxScaler(feature_range=(-1,1))
scaler2 = preprocessing.MinMaxScaler(feature_range=(0,1))





for i in range(len(filenames)):
    print('process for ' + str(i) + 'th image')
    time = filenames[i][37:54]
    img = io.imread(filenames[i])

    phase = img[:, :, 2]
    coh = img[:, :, 1]
    amp = img[:, :, 0]

    # imagesc.plot(phase, linewidth=0, cmap='jet')
    # imagesc.plot(np.log10(amp), linewidth=0, cmap='jet')
    # imagesc.plot(coh, linewidth=0, cmap='jet')

    Amplitude = amp
    coh = 0.6 * coh
    ## Covariance Matrix Cholesky Decomposition
    C = np.zeros((np.size(phase, 0), np.size(phase, 1), 2, 2), dtype='complex_')
    for ii in range(np.size(phase, 0)):
        for jj in range(np.size(phase, 1)):
            C[ii, jj] = np.array([amp[ii, jj], 0, amp[ii, jj] * coh[ii, jj] * np.exp(complex(0, -phase[ii, jj])),
                                  amp[ii, jj] * np.sqrt(1 - coh[ii, jj] ** 2)]).reshape(2, 2)

    ## Standard Circular Gaussian Random Variables
    U = np.zeros((np.size(phase, 0), np.size(phase, 1), 2, 1), dtype='complex_')
    for iii in range(np.size(phase, 0)):
        for jjj in range(np.size(phase, 1)):
            x = np.random.multivariate_normal([0], [[1]], 4).reshape(4, )
            U[iii, jjj] = np.array([[complex(x[0], x[1])], [complex(x[2], x[3])]]).reshape(2, 1)
    ## Interferometric Pairs
    Z = np.zeros((np.size(phase, 0), np.size(phase, 1), 2, 1), dtype='complex_')
    for l in range(np.size(phase, 0)):
        for k in range(np.size(phase, 1)):
            Z[l, k] = np.matmul(C[l, k, :, :], np.conj(U[l, k, :, :]))

    ## Reconstructed Interferometric
    img_e = np.zeros((np.size(phase, 0), np.size(phase, 1)), dtype='complex_')
    Ampz1 = np.zeros((np.size(phase, 0), np.size(phase, 1)))
    Ampz2 = np.zeros((np.size(phase, 0), np.size(phase, 1)))
    for iiii in range(np.size(phase, 0)):
        for jjjj in range(np.size(phase, 1)):
            z = Z[iiii, jjjj, :, :]
            img_e[iiii, jjjj] = np.matmul(z[0], np.conj(z[1]))
            Ampz1[iiii, jjjj] = np.sqrt(np.imag(z[0]) ** 2 + np.real(z[0]) ** 2)
            Ampz2[iiii, jjjj] = np.sqrt(np.imag(z[1]) ** 2 + np.real(z[1]) ** 2)

    ## Phase Noise
    phase_n = np.zeros((np.size(phase, 0), np.size(phase, 1)), dtype=phase.dtype)
    Amp_n = np.zeros((np.size(phase, 0), np.size(phase, 1)))

    for m in range(np.size(phase, 0)):
        for n in range(np.size(phase, 1)):
            phase_n[m, n] = math.atan2(np.imag(img_e[m, n]), np.real(img_e[m, n]))
            Amp_n[m, n] = np.sqrt(np.imag(img_e[m, n]) ** 2 + np.real(img_e[m, n]) ** 2)

    sin = np.sin(phase_n)
    cos = np.cos(phase_n)

    # imagesc.plot(phase_n, linewidth=0, cmap='jet')

    from patchify import patchify


    def get_patches(img1, img2, img3, img4, img5, patch_size, step):
        patches_img1 = patchify(img1, (patch_size, patch_size), step=step)
        patches_img2 = patchify(img2, (patch_size, patch_size), step=step)
        patches_img3 = patchify(img3, (patch_size, patch_size), step=step)
        patches_img4 = patchify(img4, (patch_size, patch_size), step=step)
        patches_img5 = patchify(img5, (patch_size, patch_size), step=step)

        return patches_img1, patches_img2, patches_img3, patches_img4, patches_img5


    patches_img1, patches_img2, patches_img3, patches_img4, patches_img5 = get_patches(phase_n, Amp_n, coh, phase, amp, 256, 50)
    DATA = np.zeros([256, 256, 5])
    k = 0


    for a in range(np.shape(patches_img1)[0]):
        for b in range(np.shape(patches_img1)[1]):
            DATA[:, :, 0] = scaler1.fit_transform(patches_img1[a, b, :, :])
            DATA[:, :, 1] = scaler2.fit_transform(patches_img2[a, b, :, :])
            DATA[:, :, 2] = scaler2.fit_transform(patches_img3[a, b, :, :])
            DATA[:, :, 3] = scaler1.fit_transform(patches_img4[a, b, :, :])
            DATA[:, :, 4] = scaler2.fit_transform(patches_img5[a, b, :, :])

            import gzip

            f = gzip.GzipFile("x.npy.gz", "w")
            np.save(file=f, arr=DATA)
            f.close()


            DATA_NEW = DATA.reshape(-1 , 5)
            arrays = [
                pa.array(col)  # Create one arrow array per column
                for col in DATA_NEW
            ]
            table = pa.Table.from_arrays(
                arrays,
                names=[str(i) for i in range(len(arrays))]  # give names to each columns
            )
            pq.write_table(table, 'table.pq')

            np.save('' + str(i) + str(4 * a + b) + '.npy', DATA)






filenames_train = glob.glob("/media/data/Project_subsidence/Patches/*.npy")

Patches = np.zeros([len(filenames_train), 256, 256, 5])

for i in range(len(filenames_train)):
    Patches[i, :, :, :] = np.load(filenames_train[i])

np.save('/media/data/Project_subsidence/Patches/Final_Dataset' + '.npy', Patches)


matrix = Patches[0,:,:,0]
arrays = [
    pa.array(col)  # Create one arrow array per column
    for col in matrix
]





#parquet
import pyarrow as pa
pa_table = pa.table({"data" : Patches})
pa.parquet.write_table(pa_table, "/media/data/Project_subsidence/Patches/test.parquet")













from patchify import patchify, unpatchify

# input image

image = io.imread(filenames[i])[:,:,0:3]

# splitting the image into patches
image_height, image_width, channel_count = image.shape
patch_height, patch_width, step = 256, 256, 40
patch_shape = (patch_height, patch_width, channel_count)
patches = patchify(image, patch_shape, step=step)
print(patches.shape)

# processing each patch
output_patches = np.empty(patches.shape).astype(np.float)
for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
        patch = patches[i, j, 0][:,:,0:3]
        output_patch = patch  # process the patch
        output_patches[i, j, 0] = output_patch


# merging back patches
output_height = image_height - (image_height - patch_height) % step
output_width = image_width - (image_width - patch_width) % step
output_shape = (output_height, output_width, channel_count)
output_image = unpatchify(output_patches, output_shape)
####gdal_save_tif_phase






X = np.load(filenames_train[2])[:, :, 0]
Y = np.load(filenames_train[2])[:, :, 3]
Z = np.load(filenames_train[2])[:, :, 2]

plt.imshow(X)
plt.figure()
plt.imshow(Y)
plt.figure()
plt.imshow(Z)





