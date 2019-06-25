import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

class LoadModels:

    image_width = 70
    image_height = 70

    @staticmethod
    def load_models():
        """
         load all models
         1) load SVM model
         2) load Cnn model
         3 load random forest model
         """

        # 1
        svm_model = pickle.load(open("SVMModel.h5", 'rb'))
        svmx = pickle.load(open("svmX.pickle", 'rb'))
        svmy = pickle.load(open("svmY.pickle", 'rb'))
        svm_pred = svm_model.predict_proba(svmx)[:, 1]

        # 2
        read_jason = open('model.json')
        read_model = read_jason.read()
        read_jason.close()
        cnn_model = tf.keras.models.model_from_json(read_model)
        cnn_model.load_weights("CNN_Models_One.h5")
        print("Cnn model successfully loaded")

        X = pickle.load(open("Images.pickle", "rb"))
        Y = pickle.load(open("Labels.pickle", "rb"))
        cnn_pred = cnn_model.predict(X)

        # 3
        random_model = pickle.load(open("RandomForestModel.h5", 'rb'))
        randomx = pickle.load(open("svmX.pickle", 'rb'))
        randomy = pickle.load(open("svmY.pickle", 'rb'))
        random_pred = random_model.predict_proba(randomx)[:, 1]

        """
        prediction
        for all models"""

        # 1
        fp_svm, tp_svm, th_svm = roc_curve(svmy, svm_pred)
        area_under_svm = auc(fp_svm, tp_svm)

        # 2
        fp_cnn, tp_cnn, th_cnn = roc_curve(Y, cnn_pred)
        area_under_cnn = auc(fp_cnn, tp_cnn)

        # 3
        fp_rft, tp_rft, th_cnn = roc_curve(randomy, random_pred)
        area_under_rft = auc(fp_rft, tp_rft)

        """Plot the curve
        """
        fig = plt.figure()
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fp_svm, tp_svm, label='SVM (area under curve = {:.3f})'.format(area_under_svm))
        plt.plot(fp_cnn, tp_cnn, label='CNN (area under curve = {:.3f})'.format(area_under_cnn))
        plt.plot(fp_rft, tp_rft, label='RFT (area under curve = {:.3f})'.format(area_under_rft))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='best')
        plt.show()
        fig.savefig("Roc Curve", dpi=fig.dpi)
