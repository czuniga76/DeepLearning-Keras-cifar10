# Christian Zuniga
# July 2018

from keras.datasets import cifar10

import numpy as np
import matplotlib.pyplot as plt
import random

(x_train,y_train),(x_test,y_test) = cifar10.load_data()
num_train, img_rows,img_cols, img_channels = x_train.shape
num_test,_,_,_ = x_test.shape
num_classes = len(np.unique(y_train))

def show_some_examples(names,data,labels):
    plt.figure()
    rows,cols = 4,4
    random_idx = random.sample(range(len(data)),rows*cols)
    for i in range(rows*cols):
        plt.subplot(rows,cols,i+1)
        j = random_idx[i]
        print(j)
        plt.title(names[labels[j][0]])
        img = np.reshape(data[j,:,:,:], (32,32,3))
        plt.imshow(img)
        plt.axis('off')
    #plt.tight_layout()
    
names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog','frog','hors','ship','truck']

show_some_examples(names,x_train,y_train)

# convert target to one hot encoding
# Normalize input images
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


#from keras import layers
from keras import models

# Make model with 3 convolutional layers with MaxPooling. 
# Use batch normalization and Dropout
model = models.Sequential()
model.add(layers.Conv2D(128,(3,3),padding = 'same',data_format = 'channels_last', activation='relu',input_shape=(32,32,3)))

model.add(layers.MaxPooling2D((2,2)))
#model.add(layers.Dropout(0.5))

model.add(layers.Conv2D(256,(3,3), padding ='same')) #,activation='relu'))

model.add(layers.normalization.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.5))

model.add(layers.Conv2D(256,(3,3), padding ='same')) #,activation='relu'))

model.add(layers.normalization.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())

model.add(layers.Dropout(0.5))
model.add(layers.Dense(512))
model.add(layers.normalization.BatchNormalization())
model.add(layers.Activation('relu'))


model.add(layers.Dense(num_classes,activation='softmax'))

# Train with Adam Optimizer
# Start with no data augmentation

import time
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

adam = optimizers.Adam(lr=0.01)
datagen = ImageDataGenerator(zoom_range=0.2,width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True)
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

start_time = time.time()
model_info = model.fit(x_train,y_train,epochs=50,batch_size = 256,validation_data=(x_test,y_test))
#model_info = model.fit_generator(datagen.flow(x_train,y_train,batch_size=256),nb_epoch=50,validation_data=(x_test,y_test))
end_time = time.time()

test_loss,test_acc = model.evaluate(x_test,y_test)
print(test_acc)
print(end_time-start_time)

import matplotlib.pyplot as plt
plt.plot(model_info.history['acc'])
plt.plot(model_info.history['val_acc'])
plt.title("Model Accuracy vs Number Epochs")
plt.legend(["Training","Test"])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()