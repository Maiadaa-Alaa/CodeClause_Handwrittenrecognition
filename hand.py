#%%
#libraries
import tensorflow as tf
import numpy as np
import cv2 as cv  
import matplotlib.pyplot as plt
from tensorflow.python.keras.metrics import accuracy
#%%
# split the data in train and test from mnist dataset
(xtrain, ytrain),(xtest,ytest)= tf.keras.datasets.mnist.load_data()
print(xtrain.shape)
print(ytrain.shape)
print(xtest.shape)
print(ytest.shape)
#%%
# image(5)

print(xtrain[5])
#%%
#print shape of xtest[1] --> no.2

plt.imshow(xtest[1])
plt.show()
#%%
# first image in xtest equal first image in ytest
print(ytest[1])
#%%
# inputs range of [0,1]
xtrain = tf.keras.utils.normalize(xtrain , axis = 1)
xtest = tf.keras.utils.normalize(xtest , axis = 1)
# train a neural network model using the Keras API.
model= tf.keras.models.Sequential()
# flatten input layer, two hidden layers with 128 units ,output layer with 10 units 

model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10,activation=tf.nn.softmax))
model.compile(optimizer='Adam' , loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# number of epochs increases beyond 11,chance of overfitting of the model on training data
model.fit(xtrain,ytrain, epochs=3)



#%%
# print accuracy and loss of model
accuracy  =model.evaluate(xtest,ytest)
loss = model.evaluate(xtest,ytest)
print(accuracy)
print(loss)
#%%
# read image and pass it for prediction, image is displayed

for x in range(2,4):
    # now we are going to read images it with open cv

    img=cv.imread(f'{x}.png')[:,:,0]
    img=np.invert(np.array([img]))
    prediction=model.predict(img)
    print("The predicted value is : ",np.argmax(prediction))

    # change the color in black and white
    plt.imshow(img[0],cmap=plt.cm.binary)
    plt.show()

# %%
