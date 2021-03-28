# imports
import os
import pandas as pd
import numpy as np
import json
from os.path import join
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_data_format('channels_first')
from keras.preprocessing.image import ImageDataGenerator, load_img
from skimage.transform import downscale_local_mean
from os.path import join
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import cv2
import skimage.io as io

def overall(image):
   
   # define the model
    model = Sequential()    
    model.add(Conv2D(50, (5, 5), input_shape=(1, 102, 110), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
   
    from tensorflow import keras
    model = keras.models.load_model('model/demo_model_4')
    
    model.compile(loss='catagorical_crossentropy', optimizer='adamax', metrics=['acc'])
    
    tf = 'uploads'
    #tf = 'data/processedBuildingLabels/3band'
    test_filenames = os.listdir("uploads")
    
    import glob
    import re

    tif_file = glob.glob('uploads/*.tif') 
    if len(tif_file) != 1:
        raise ValueError('should be only one jif file in the current directory')
    tif_file_new = str(tif_file) 
    characters_to_remove = "] ' ["

    pattern = "[" + characters_to_remove + "]"
    tst = re.sub(pattern, "", tif_file_new)

    #print(test_filenames)
    #tst = test_filenamest
    x = np.array(Image.open(tst))
    #plt.imshow(x)
    images = [np.array(Image.open(join(tf, i)).convert('L')) for i in os.listdir(tf)]
    #plt.imshow(images[0])
    downscaled_images = []
    for i in range(len(images)):
        downscaled_images.append(downscale_local_mean(images[i], (4,4)))
    #plt.imshow(downscaled_images[0])
    shape = list(downscaled_images[0].shape)
    shape[:0] = [len(downscaled_images)]
    array_of_images = np.concatenate(downscaled_images).reshape(shape)
    #array_of_images.shape
    norm_x = array_of_images/255

    nx = norm_x.astype('float32')

    nx2 = nx.reshape(-1, 1, 102, 110)
    #plt.imshow(array_of_images[0], cmap="gray");
    y_pred = model.predict_classes(nx2, batch_size=32)
    #y_pred
    f, ax = plt.subplots()
    plt.imshow(x, cmap='gray')
    #plt.imsave('templates/result.jpg',x)
        
    if y_pred == 1:
        #ax.annotate("found buildings", (5,90), color='white', weight='bold',fontsize=16, bbox={'facecolor':'green', 'alpha':0.9, 'pad':5})
       # imwrite(x,'uploads/yes.jpg')
        plt.imsave('static/result.jpg',x)
        return 'Building found'
    else:
        #ax.annotate("no buildings", (5,90), color='white', weight='bold',fontsize=16, bbox={'facecolor':'red', 'alpha':0.9, 'pad':5})    
        #imwrite(x,'uploads/no.jpg')
        plt.imsave('static/result.jpg',x)
        return 'No buildings found'
        #ax.annotate("prediction:  " + prediction, (5, 10), color='white', weight='bold',fontsize=16)
        #ax.annotate("actual:  " + actual, (5, 20), color='white', weight='bold',fontsize=16)
    
    print('image loaded')        
    #predict = model.predict(test_generator, steps=np.ceil(nb_samples/batch_size))
    #if predict[0,0] > predict[0,1]:
     #   return 'cat'
    #else:
    #    return 'dog'
    # delete file
    os.remove('uploads/filename')
