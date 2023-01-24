import glob

import tensorflow as tf
import numpy as np
import n2v
print(tf.__version__)
print(n2v.__version__)
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from n2v.models import N2VConfig, N2V
import numpy as np
from csbdeep.utils import plot_history
from n2v.utils.n2v_utils import manipulate_val_data
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from matplotlib import pyplot as plt
import urllib
import os
import zipfile


import warnings
warnings.filterwarnings("ignore")

import tifffile

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
datagen = N2V_DataGenerator()
#METHOD 1: Loading images using load_imgs_from_directory method
# We load all the '.png' files from the directory.
# The function will return a list of images (numpy arrays).
filenmaes = glob.glob("/media/miladmoazezian/3B07B9F56DF9B08D/Project_subsidence/Interferograms/*IW2.tif")


imgs = []


for i in range(len(filenmaes) - 3200):

    size = np.shape(tifffile.imread(filenmaes[i]))
    imgs.append((tifffile.imread(filenmaes[i])[1,:,:]).reshape(1,size[1],size[2],1))





# imgs = datagen.load_imgs_from_directory(directory = "/media/miladmoazezian/3B07B9F56DF9B08D/Project_subsidence/test_noise2void", filter='*.tif',dims='YX')  #ZYX for 3D

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


# Let's look at the shape of the image
print('shape of loaded images: ',imgs[0].shape)
# If the image has four color channels (stored in the last dimension): RGB and Aplha.
# We are not interested in Alpha and will get rid of it.
# imgs[0] = imgs[0][...,:3]
print('shape without alpha:    ',imgs[0].shape)
print('The data type of the first image is: ', imgs[0].dtype)



patch_size = 64
patch_shape = (patch_size,patch_size)



patches = datagen.generate_patches_from_list(imgs, shape=patch_shape)



# plt.imshow(patches[0,:,:])


train_val_split = int(patches.shape[0] * 0.8)
X = patches[:train_val_split]
X_val = patches[train_val_split:]

# plt.figure(figsize=(14,7))
# plt.subplot(1,2,1)
# plt.imshow(X[0,:,:,0])
# plt.title('Training Patch');
# plt.subplot(1,2,2)
# plt.imshow(X_val[0,:,:,0])
# plt.title('Validation Patch');



train_batch = 1
config = N2VConfig(X, unet_kern_size=3,
                   unet_n_first=64, unet_n_depth=6, train_steps_per_epoch=int(X.shape[0]/train_batch), train_epochs=200, train_loss='mse',
                   batch_norm=True, train_batch_size=train_batch, n2v_perc_pix=0.198, n2v_patch_shape=(patch_size, patch_size),
                   n2v_manipulator='uniform_withCP', n2v_neighborhood_radius=5, single_net_per_channel=False)


# Let's look at the parameters stored in the config-object.
vars(config)
model_name = 'n2v_2D_insar'
# the base directory in which our model will live
basedir = 'models'
# We are now creating our network model.
model = N2V(config, model_name, basedir=basedir)



history = model.train(X, X_val)



import matplotlib.pyplot as plt
import tifffile
test_img = tifffile.imread('/media/miladmoazezian/3B07B9F56DF9B08D/Project_subsidence/Interferograms/20190112_20190205_IW2.tif')[1,:,:]
# pred = model.predict(test_img , axes='YX')
import numpy as np

test = np.load('/home/miladmoazezian/Desktop/test.npy')

plt.imshow(test , cmap='jet')
plt.figure()
plt.imshow(test_img , cmap='jet')


