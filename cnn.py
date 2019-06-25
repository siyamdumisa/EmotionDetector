# import libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import cv2
import os


class CNN:
    pass

    # define model variables
    epoch_size = 50
    b_size = 100
    num_split = 0.2
    file_name1 = 'CNN_Models_One.h5'
    file_name2 = 'CNN_Models_One.h5'
    image_size = 70

    """
    load seved data for KDEF_Data
    load saved data for FERG_Data 
    """
    img = pickle.load(open("Images.pickle", "rb"))
    lbl = pickle.load(open("Labels.pickle", "rb"))

    # img2 = pickle.load(open("ImagesTwo.pickle", "rb"))
    # lbl2 = pickle.load(open("LabelsTwo.pickle", "rb"))

    # img = pickle.load(open("", "rb"))
    # lbl = pickle.load(open("y.pickle", "rb"))

    # Normalize data
    IMG = img/225.0

    # crating a VGG 16 Convolution Neural network
    def create_cnn(self):

        # create a feed forward model
        model = tf.keras.Sequential()

        # Input layer
        model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(self.image_size, self.image_size, 1)))
        model.add(tf.keras.layers.Activation(tf.nn.relu))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        # hidden layer's
        model.add(tf.keras.layers.Conv2D(64, (3, 3)))
        model.add(tf.keras.layers.Activation(tf.nn.relu))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64))
        model.add(tf.keras.layers.Activation(tf.nn.relu))

        model.add(tf.keras.layers.Dense(1))
        model.add(tf.keras.layers.Activation(tf.nn.sigmoid))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Begin training
        history = model.fit(self.IMG, self.lbl, batch_size=self.b_size, epochs=self.epoch_size, validation_split=self.
                            num_split)

        '''
        the following code plot metrics that the classier save to determine how bad and good does it perform during
        training.
        ---------------------------
        section 1: summarize history for accuracy
        section 2: # summarize history for loss
        '''
        fig = plt.figure()
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Accuracy Curve For CNN')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        fig.savefig("AccuracyCNN.png", dpi=fig.dpi)

        fig2 = plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Loss Curve For CNN ')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        fig2.savefig('LossCNN.png', dpi=fig2.dpi)
        print("finish")



        # saving the model to a json file
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(self.file_name1)
        print("Saved model to disk")


c = CNN()
c.create_cnn()
