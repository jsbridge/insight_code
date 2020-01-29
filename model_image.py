from keras.preprocessing.image import image 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras import backend as K
import numpy as np


def predict_class(uploaded_img):
    img_width, img_height = 400, 400


    #load image you want to make prediction for
    img = image.load_img(uploaded_img, target_size = (img_width, img_height))
    img_arr = image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr /= 255

    if K.image_data_format() == 'channels_first': 
        input_shape = (3, img_width, img_height) 
    else: 
        input_shape = (img_width, img_height, 3)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4))
    model.add(Activation('softmax'))

    model.load_weights('model_masks_saved_batch32.h5')

    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    
    pred = model.predict(img_arr)

    index_predict = np.argmax(pred[0])

    if pred[0][index_predict] <= 0.:
        return "unsure"

    dict_labels = {0:'curly', 1:'quite curly', 2:'not very curly', 3:'wavy'}
	
    return dict_labels[index_predict], pred[0]
