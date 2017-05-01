# -*- coding: utf-8 -*-
"""
Side Project: Overlapping chromosomes
"""

import h5py
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import scipy as sp


### Load the dataset
h5f = h5py.File('D:/Dataset/Side_project_Overlapping chromosomes/LowRes_13434_overlapping_pairs.h5','r')
pairs = h5f['dataset_1'][:]
h5f.close()


### Display some examples
grey = pairs[100,:,:,0]
mask = pairs[100,:,:,1]
plt.subplot(121)
plt.imshow(grey)
plt.title('max='+str(grey.max()))
plt.subplot(122)
plt.imshow(mask)

grey = pairs[120,:,:,0]
mask = pairs[120,:,:,1]
plt.subplot(121)
plt.imshow(grey)
plt.title('max='+str(grey.max()))
plt.subplot(122)
plt.imshow(mask)

### Thresholding: otsu
origin_tr = np.copy(pairs_tr)[:,:,:,0]                   

from skimage.filters.thresholding import threshold_otsu
thresh = threshold_otsu(origin_tr)
print(thresh)

origin_tr_thresh = (origin_tr > thresh)*1

plt.imshow(origin_tr[120])
plt.imshow(origin_tr_thresh[120])

### Thresholding: adaptive
from skimage import filters
origin_tr_adp = filters.threshold_adaptive(origin_tr[120], 63, 'gaussian')
plt.imshow(origin_tr[120])
plt.imshow(origin_tr_adp)
plt.imshow(origin_tr_thresh[120])
# 63 looks great


### Normalize the data

# transform the values to 0 to 1
pairs = pairs.astype(float)
for i in range(pairs.shape[0]):
    pairs[i, :, :, 0] = pairs[i, :, :, 0]/255
pairs[:, :, :, 1] = pairs[:, :, :, 1] > 2 # filter out the overlapped section

# check
plt.imshow(pairs[100, :, :, 1])

### Split the data into training and test set
train_ind = np.random.choice(pairs.shape[0], int(pairs.shape[0]*0.8), replace=False)
#pairs = pairs.astype(float)
pairs_tr = np.copy(pairs)[train_ind]
pairs_test = np.copy(pairs)[np.setdiff1d(np.arange(pairs.shape[0]), train_ind)]


### Split the training set for validation set
valid_ind = np.random.choice(pairs_tr.shape[0], int(pairs_tr.shape[0]*0.2), replace=False)
pairs_valid = np.copy(pairs_tr)[valid_ind]
pairs_tr = np.copy(pairs_tr)[np.setdiff1d(np.arange(pairs_tr.shape[0]), valid_ind)]


"""
Use CNN to build the model 
"""

### reshape the data for cnn
h = 94
w = 93

pic_x_tr = pairs_tr[:, :, :, 0].reshape([len(pairs_tr), h, w, 1])
pic_y_tr = pairs_tr[:, :, :, 1].reshape([len(pairs_tr), h, w, 1])
pic_x_valid = pairs_valid[:, :, :, 0].reshape([len(pairs_valid), h, w, 1])
pic_y_valid = pairs_valid[:, :, :, 1].reshape([len(pairs_valid), h, w, 1])

### build the model structure
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.callbacks import TensorBoard

input_img = Input(shape=(h, w, 1))
x = Conv2D(16, (3, 4), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this moment, the shape of the image is (8, 12, 12)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 4), activation='sigmoid')(x)


autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(pic_x_tr, pic_y_tr,
                epochs=100,
                batch_size=10,
                shuffle=True,
                validation_data=(pic_x_valid, pic_y_valid),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

y_tr_pred = autoencoder.predict(pic_x_tr, batch_size=10)
y_valid_pred = autoencoder.predict(pic_x_valid, batch_size=10)

pic = 0
fig = plt.figure(figsize=(18, 8))
ax_1 = fig.add_subplot(1, 3, 1)
plt.imshow(pic_x_tr[pic].reshape([h, w]), cmap='gray')
ax_1.set_title('Raw Picture', fontsize=15)
plt.grid(False)

ax_2 = fig.add_subplot(1, 3, 2)
plt.imshow(y_tr_pred[pic].reshape([h, w]), cmap='gray')
ax_2.set_title('Prediction from Simple CNN model', fontsize=15)
plt.grid(False)

ax_3 = fig.add_subplot(1, 3, 3)
plt.imshow(pic_y_tr[pic].reshape([h, w]), cmap='gray')
ax_3.set_title('Ground Truth', fontsize=15)
plt.grid(False)

pic = 0
fig = plt.figure(figsize=(18, 8))
ax_1 = fig.add_subplot(1, 3, 1)
plt.imshow(pic_x_valid[pic].reshape([h, w]))
ax_1.set_title('Raw Picture', fontsize=15)
plt.grid(False)

ax_2 = fig.add_subplot(1, 3, 2)
plt.imshow(y_valid_pred[pic].reshape([h, w])>0.5, cmap='gray')
ax_2.set_title('Prediction from Simple CNN model', fontsize=15)
plt.grid(False)

ax_3 = fig.add_subplot(1, 3, 3)
plt.imshow(pic_y_valid[pic].reshape([h, w]), cmap='gray')
ax_3.set_title('Ground Truth', fontsize=15)
plt.grid(False)

### use threshold
fig = plt.figure(figsize=[12, 8])
ax_1 = fig.add_subplot(1, 2, 1)
plt.imshow((y_valid_pred[pic].reshape([h, w])>0.5)*pic_x_valid[pic].reshape([h, w]))
plt.grid(False)

ax_2 = fig.add_subplot(1, 2, 2)
plt.imshow(pic_y_valid[pic].reshape([h, w]), cmap='gray')
plt.grid(False)

### Evaluate the model
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

## choose the best threshold based on the validation set on f1 score
th_list = np.arange(0, 1, 0.05)
f1_list = []
# th_best = 0.3
th_best = 0
f1_score_best = 0

for th in th_list:
    f1_temp_list = []
    for i in range(len(pic_y_valid)):
        f1_temp_list.append(f1_score(pic_y_valid[i].flatten(),
                                     y_valid_pred[i].flatten()>th))
    mean_score = np.mean(f1_temp_list)
    f1_list.append(mean_score)
    if mean_score > f1_score_best:
        f1_score_best = mean_score
        th_best = th

print('Best threshold:', th_best)
print('Best f1 socre:', f1_score_best)


### Save the model
## reference: http://machinelearningmastery.com/save-load-keras-deep-learning-models/
# serialize model to YAML
model_yaml = autoencoder.to_yaml()
with open("D:/Project/Side_project_Overlapping_chromosomes/model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
autoencoder.save_weights("D:/Project/Side_project_Overlapping_chromosomes/model.h5")
print("Saved model to disk")

### load and create model
from keras.models import model_from_yaml
yaml_file = open('D:/Project/Side_project_Overlapping_chromosomes/model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("D:/Project/Side_project_Overlapping_chromosomes/model.h5")
print("Loaded model from disk")

loaded_model.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder = loaded_model


### Evaluate on the test dataset
pic_x_test = pairs_test[:, :, :, 0].reshape([len(pairs_test), h, w, 1])
pic_y_test = pairs_test[:, :, :, 1].reshape([len(pairs_test), h, w, 1])

y_pred = autoencoder.predict(pic_x_test, batch_size=10)

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

a_list = []
p_list = []
r_list = []
f_list = []
for i in range(len(y_pred)):
    a_list.append(accuracy_score(pic_y_test[i].flatten(),
                                  y_pred[i].flatten()>th_best))    
    p_list.append(precision_score(pic_y_test[i].flatten(),
                                  y_pred[i].flatten()>th_best))
    r_list.append(recall_score(pic_y_test[i].flatten(),
                                  y_pred[i].flatten()>th_best))
    f_list.append(f1_score(pic_y_test[i].flatten(),
                                  y_pred[i].flatten()>th_best))

print('Results on test dataset:')
print('Accuracy:', np.mean(a_list))
print('Precision:', np.mean(p_list))
print('Recall:', np.mean(r_list))
print('F1 score:', np.mean(f_list))


pic = 1000
fig = plt.figure(figsize=(16, 14))
ax_1 = fig.add_subplot(2, 2, 1)
plt.imshow(pic_x_test[pic].reshape([h, w]))
ax_1.set_title('Raw Picture', fontsize=15)
plt.grid(False)

ax_2 = fig.add_subplot(2, 2, 2)
plt.imshow(y_pred[pic].reshape([h, w]), cmap='gray')
ax_2.set_title('Prediction on Test Set Using Tuned CNN Model', fontsize=15)
plt.grid(False)

ax_3 = fig.add_subplot(2, 2, 3)
plt.imshow((pic_x_test[pic].reshape([h, w]))*(y_pred[pic].reshape([h, w])>th_best))
ax_3.set_title('Map the Raw Data to Predicted Locations Using Trained Threshold', fontsize=15)
plt.grid(False)

ax_4 = fig.add_subplot(2, 2, 4)
plt.imshow(pic_y_test[pic].reshape([h, w]), cmap='gray')
ax_4.set_title('Ground Truth', fontsize=15)
plt.grid(False)


### Load the dataset
h5f = h5py.File('D:/Dataset/Side_project_Overlapping chromosomes/LowRes_13434_overlapping_pairs.h5','r')
pairs = h5f['dataset_1'][:]
h5f.close()
pic_ind = np.setdiff1d(np.arange(13434), train_ind)[1000]

plt.imshow(pairs[pic_ind, :, :, 0])
plt.imshow(pairs[pic_ind, :, :, 1])

ref_exp = np.copy(pairs[pic_ind, :, :, 1])
ref_exp[ref_exp>2] = 2
plt.imshow(ref_exp)

ref_exp[y_pred[pic].reshape([h, w])>th_best] = 3
plt.imshow(ref_exp)

### Show the prediction 

pic = 1000
fig = plt.figure(figsize=(16, 16))
plt.suptitle('Display the Prediction on One Example', fontsize=20, y=1.02)
ax_1 = fig.add_subplot(2, 2, 1)
plt.imshow(pic_x_test[pic].reshape([h, w]))
ax_1.set_title('Raw Picture', fontsize=15)
plt.grid(False)

ax_2 = fig.add_subplot(2, 2, 2)
plt.imshow(ref_exp)
ax_2.set_title('Prediction on Test Set Using Tuned CNN Model', fontsize=15)
plt.grid(False)

ax_3 = fig.add_subplot(2, 2, 3)
ref_mask = np.copy(pic_x_test[pic].reshape([h, w]))
plt.imshow(ref_mask*(y_pred[pic].reshape([h, w])>th_best))
ax_3.set_title('Map the Raw Data to Predicted Locations Using Trained Threshold', fontsize=15)
plt.grid(False)

ax_4 = fig.add_subplot(2, 2, 4)
plt.imshow(pairs[pic_ind, :, :, 1])
ax_4.set_title('Ground Truth', fontsize=15)
plt.grid(False)
plt.tight_layout()
