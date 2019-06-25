import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
import random
import dataloader as loader
from property import Property
from sklearn import metrics


class SupportVectorMachine:

    class_names = ["happy", "sad"]
    data_set = "data"

    num_split = 0.2

    img = "svmX.pickle"
    lbl = "svmY.pickle"

    model_name = "SVMModel.h5"

    sets_gets = Property()

    # create Random Forest Classifier
    def create_svm(self,):

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

        # 80% training, 20% testing
        x_train, x_test, y_train, y_test = train_test_split(xtrain, ytrain, test_size=self.num_split, random_state=0)

        self.sets_gets.xTest = x_test
        self.sets_gets.yTest = y_test
        xt = self.sets_gets.xTest
        yt = self.sets_gets.yTest

        # saving the testing data to pickle
        pickle.dump(xt, open(self.img, 'wb'))
        print("SVM X Test Saved")

        pickle.dump(yt, open(self.lbl, 'wb'))
        print("SVM Y Test Saved")

        # training the SVM model
        svm_classifier = svm.SVC(kernel='rbf',
                                 probability=True)
        model = svm_classifier.fit(x_train, y_train)

        # accuracy and loss for the model
        a = svm_classifier.score(x_train, y_train)
        print("the value of the training accuracy  :" + str(a))

        l = svm_classifier.score(x_test, y_test)
        print("the value of the testing accuracy :" + str(l))

        # confusion matrics
        y_pred = svm_classifier.predict(x_test)

        print("Classification Report for SVM:")
        print(metrics.classification_report(y_test, y_pred))
        plt.show()
        print("Confusion Matrix For SVM:")
        print(metrics.confusion_matrix(y_test, y_pred))

        # saving the random forest model to pickle
        pickle.dump(model, open(self.model_name, 'wb'))
        print("Support Vector Machine Model Saved")


s = SupportVectorMachine()
s.create_svm()
