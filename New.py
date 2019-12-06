#importing the modules

import cv2     # for capturing videos
import math   # for mathematical operations
import matplotlib.pyplot as plt    # for plotting the images%matplotlib inline
import pandas as pd
from keras.preprocessing import image   # for preprocessing the images
import numpy as np    # for mathematical operations
from keras.utils import np_utils
from skimage.transform import resize   # for resizing images



data = pd.read_csv('imggg.csv')     # reading the csv file
print data.head(5)
X = []     # creating an empty array
for img_name in data.Image_ID:
    img = plt.imread('image/' + img_name+'.jpg',0)
    #img=cv2.resize(img,(224,224), interpolation=cv2.INTER_CUBIC)
    X.append(img)  # storing each image in array X

X = np.array(X)  # converting list to array
#print(X.shape)

y = data.classes
dummy_y = np_utils.to_categorical(y)    # one hot encoding Classes

image = []
for i in range(0,X.shape[0]):
    a = resize(X[i], preserve_range=True, output_shape=(224,224)).astype(int)      # reshaping to 224*224*3
    image.append(a)
X = np.array(image)
'''
from keras.applications.vgg16 import preprocess_input
X = preprocess_input(X, mode='tf')      # preprocessing the input data
'''

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, dummy_y, test_size=0.3, random_state=42)    # preparing the validation set


from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
'''
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))    # include_top=False to remove the top layer


X_train = base_model.predict(X_train)
X_valid = base_model.predict(X_valid)

'''
print X_train.shape
print X_valid.shape
X_train = X_train.reshape(733, 224,224,3)      # converting to 1-D
X_valid = X_valid.reshape(315, 224,224,3)

train = X_train/255      # centering the data
X_valid = X_valid/255

# i. Building the model
model = Sequential()

model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(3,activation='softmax'))
'''
model.add(InputLayer((7*7*512,)))    # input layer
model.add(Dense(units=1024, activation='sigmoid')) # hidden layer
model.add(Dense(3, activation='softmax'))    # output layer
'''


#ii. Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



# iii. Training the model
model.fit(train, y_train, epochs=10, validation_data=(X_valid, y_valid))
print (model.summary())
y_pred = model.predict(X_valid)
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(np.argmax(y_valid,axis=1), np.argmax(y_pred,axis=1))
print(confusion)

test = pd.read_csv('test.csv')
sht=list(test.shape)
rt=sht[0]
test_image = []
for img_name in test.Image_ID:
    img = plt.imread('test_image/live/' + img_name)
    test_image.append(img)
test_img = np.array(test_image)

test_image = []
for i in range(0,test_img.shape[0]):
    a = resize(test_img[i], preserve_range=True, output_shape=(224,224)).astype(int)
    test_image.append(a)
test_image = np.array(test_image)



# converting the images to 1-D form
test_image = test_image.reshape(rt, 224,224,3)

# zero centered images
test_image = test_image/255

predictions = model.predict_classes(test_image)

#print("The screen time of POTENTIAL is", predictions[predictions==1].shape[0], "seconds")
#print("The screen time of HARM  is", predictions[predictions==2].shape[0], "seconds")
#print("The screen time of SAFE is", predictions[predictions==0].shape[0], "seconds")

harm=predictions[predictions==2].shape[0]
pot=predictions[predictions==1].shape[0]
safe=predictions[predictions==0].shape[0]

if (harm >= pot) and (harm >= safe):
	print("The video contains Dubious Activity")
elif (pot >= harm) and (pot >= safe):
	print("The video contains Potentially Dubious Activity")
else:
	print("The video contains No Dubious Activity. The video is Safe")

 
