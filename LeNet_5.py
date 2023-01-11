import numpy as np
from random import random
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, AveragePooling2D, Flatten
import keras.activations as activation
import keras.losses as loss
from keras.optimizers import SGD
from keras.utils import to_categorical

def getNum(oneHotList: np.ndarray) -> int:
    max = 0.0
    maxI = 0
    for i in range(10):
        weight = oneHotList[i]
        if(weight > max):
            max = weight
            maxI = i
    return maxI

def main():
    X_trainsRaw: np.ndarray
    Y_trainsRaw: np.ndarray
    X_testsRaw: np.ndarray
    Y_testsRaw: np.ndarray
    (X_trainsRaw, Y_trainsRaw), (X_testsRaw, Y_testsRaw) = mnist.load_data()
    
    X_trains: np.ndarray = X_trainsRaw.reshape(60000, 28, 28, 1)/255.0 # 1 channel because gray scale
    X_tests: np.ndarray = X_testsRaw.reshape(10000, 28, 28, 1)/255.0

    Y_trains: np.ndarray = to_categorical(Y_trainsRaw, 10)
    Y_tests: np.ndarray = to_categorical(Y_testsRaw, 10)
    
    model = Sequential()

    # convolution
    # 6 kernels, 5*5 kernel, 1 unit step; result shape: 24*24*6
    model.add(Conv2D(filters=6, kernel_size=(5,5), strides=(1,1), input_shape=(28, 28, 1), padding='valid', activation=activation.relu))
    # sub-sampling with 2*2 average pooling, default 2*2 unit step; result shape: 12*12*6
    model.add(AveragePooling2D(pool_size=(2,2)))
    # result shape: 8*8*16
    model.add(Conv2D(filters=16, kernel_size=(5,5), strides=(1,1), padding='valid', activation=activation.relu))
    # result shape: 4*4*16
    model.add(AveragePooling2D(pool_size=(2,2)))

    # fully connected layers
    model.add(Flatten())
    model.add(Dense(units=120, activation=activation.relu))
    model.add(Dense(units=84, activation=activation.relu))
    model.add(Dense(units=10, activation=activation.softmax))

    model.compile(loss=loss.categorical_crossentropy, optimizer=SGD(learning_rate=0.05), metrics=['accuracy'])
    
    model.fit(x=X_trains, y=Y_trains, epochs=50, batch_size=750)

    error, accuracy = model.evaluate(X_tests, Y_tests)
    print(f'loss: {error}; accuracy: {accuracy}')
    
    # for _ in range(20):
    #     index = int(random()*10000)
    #     image = np.array([X_tests[index]])
    #     print(image.shape)
    #     print(getNum(model(image)[0]))
    #     plt.imshow(X_testsRaw[index], cmap='gray')
    #     plt.show()
    
    model.save('LeNet_5_Model')


if(__name__ == "__main__"):
    main()