import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random
import dataloader as loader
from property import Property
from sklearn import metrics


class RandomForest:

    class_names = ["happy", "sad"]
    data_set = "data"

    num_split = 0.2

    img = "randomX.pickle"
    lbl = "randomY.pickle"

    model_name = "RandomForestModel.h5"

    sets_gets = Property()

    # create Random Forest Classifier
    def create_random(self,):

        # load data for training
        v = loader.DataLoader()
        start_training = v.load_data(self.data_set, self.class_names)

        # shuffling the array
        random.shuffle(start_training)

        (xtrain, ytrain) = ([], [])

        # creating arrays for both the features and the labels
        for d, c in start_training:
            xtrain.append(d)
            ytrain.append(c)

        (xtrain, ytrain) = [np.array(lis) for lis in [xtrain, ytrain]]

        xtrain = np.asarray(xtrain)
        ytrain = np.asarray(ytrain)

        # 80% training and 20% testing
        x_train, x_test, y_train, y_test = train_test_split(xtrain, ytrain, test_size=self.num_split, random_state=0)

        self.sets_gets.xTest = x_test
        self.sets_gets.yTest = y_test
        xt = self.sets_gets.xTest
        yt = self.sets_gets.yTest

        # saving the testing data to pickle
        pickle.dump(xt, open(self.img, 'wb'))
        print("Random Forest X Test Saved")

        pickle.dump(yt, open(self.lbl, 'wb'))
        print("Random Forest Y Test Saved")

        # training the random forest model
        forest_classifier = RandomForestClassifier(n_estimators=2)
        model = forest_classifier.fit(x_train, y_train)

        # accuracy and loss for the model
        a = forest_classifier.score(x_train, y_train)
        print("the value of the training accuracy : " + str(a))

        l = forest_classifier.score(x_test, y_test)
        print("the value of the testing accuracy : " + str(l))

        # confusion matrics
        y_pred = forest_classifier.predict(x_test)

        print("Classification Report:")
        print(metrics.classification_report(y_test, y_pred))

        print("Confusion Matrix:")
        print(metrics.confusion_matrix(y_test, y_pred))

        # saving the random forest model to pickle
        pickle.dump(model, open(self.model_name, 'wb'))
        print("Random Forest Model Saved")


r = RandomForest()
r.create_random()
