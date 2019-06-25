import sys

from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QApplication, QDialog
import cv2
import tensorflow as tf
import pickle
import plot_roc as plt


class Main(QDialog):

    image_width = 70
    image_height = 70

    rgb = (225, 51, 51)
    rgb2 = (240, 0, 0)

    def __init__(self):
        super(Main, self).__init__()
        loadUi('Main.ui', self)
        self.setWindowTitle("Emotion Recognition")
        self.image = None
        self.image1 = None
        self.image2 = None


        self.StartEmortionReg.clicked.connect(self.start_webcam)
        self.EndEmotionReg.clicked.connect(self.stop_webcam)
        self.ViewRoc.clicked.connect(self.plot_roc_curve)


        # function to start the webcam
    def start_webcam(self):
            self.capture = cv2.VideoCapture(0)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(5)

    def stop_webcam(self):
        self.timer.stop()

    """ Displaying frames on UI
     1) SVM
     2) CNN
     3) Random forest"""
    def video_displaySVM(self,img,window=1):

        qformat = QImage.Format_Indexed8
        if len(img.shape)==3:
            if img.shape[2]==4 :
                qformat=QImage.Format_RGBA8888
            else:
                qformat=QImage.Format_RGB888
        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        outImage = outImage.rgbSwapped()

        if window == 1:
            self.label_5.setPixmap(QPixmap.fromImage(outImage))
            self.label_5.setScaledContents(True)

    def video_displayCNN(self,img,window=1):

        qformat = QImage.Format_Indexed8
        if len(img.shape)==3:
            if img.shape[2]==4 :
                qformat=QImage.Format_RGBA8888
            else:
                qformat=QImage.Format_RGB888
        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        outImage = outImage.rgbSwapped()

        if window == 1:
            self.label_6.setPixmap(QPixmap.fromImage(outImage))
            self.label_6.setScaledContents(True)

    def video_displayRFT(self, img, window=1):

        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        outImage = outImage.rgbSwapped()

        if window == 1:
            self.label_7.setPixmap(QPixmap.fromImage(outImage))
            self.label_7.setScaledContents(True)

    def update_frame(self):

        # ------------------------------------------------------------------------------------------------------------
        """
        load all models
        1) load SVM model
        2) load Cnn model
        3 load random forest model
        """
        # 1
        svm_model = pickle.load(open("SVMModel.h5", 'rb'))

        # 2
        read_jason = open('model.json')
        read_model = read_jason.read()
        read_jason.close()
        cnn_model = tf.keras.models.model_from_json(read_model)
        cnn_model.load_weights("CNN_Models_One.h5")
        print("Cnn model successfully loaded")

        # 3
        random_model = pickle.load(open("RandomForestModel.h5", 'rb'))

        ret, self.image = self.capture.read()
        ret, self.image1 = self.capture.read()
        ret, self.image2 = self.capture.read()

        # -------------------------------------------------------------------------------------------------------------
        """"
        extract features form the image on webcam
        Perform ROI of interest segmentation
        """
        x1 = int(0.2 * self.image.shape[1])
        y1 = 90
        x2 = self.image.shape[1] - 120
        y2 = int(0.7 * self.image.shape[1])

        cv2.rectangle(self.image, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
        cv2.rectangle(self.image1, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
        cv2.rectangle(self.image2, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)

        face_region = self.image[y1:y2, x1:x2]
        face_region = cv2.resize(face_region, (70, 70))
        face_region = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

        # extracting only the face from the ROI
        ret, img = cv2.threshold(face_region, 120, 255, cv2.THRESH_BINARY)
        # ------------------------------------------------------------------------------------------------------------
        """Prediction
        1} Prediction for SVM
        2) Prediction for CNN
        3) Prediction for Random forest
        """

        # 1
        svm_pred = svm_model.predict(img.reshape(1, self.image_width * self.image_height))
        if (svm_pred == 1):
            cv2.putText(self.image, "Sad", (150, 120), cv2.FONT_ITALIC, 1, self.rgb, 1)
        elif (svm_pred == 0):
            cv2.putText(self.image, "Happy", (150, 120), cv2.FONT_ITALIC, 1, self.rgb, 1)
        else:
            cv2.putText(self.image, " ", (150, 120), cv2.FONT_ITALIC, 1, self.rgb, 1)

        self.video_displaySVM(self.image, 1)

        # 2
        cnn_pred = cnn_model.predict(img.reshape(-1, self.image_width, self.image_height, 1))
        # checking the label and printing the appropriate facial expression
        if cnn_pred >= 0.6 and cnn_pred <= 1:
            cv2.putText(self.image1, "sad", (150, 120), cv2.FONT_ITALIC, 1, self.rgb2, 1)
        elif cnn_pred >= 0 and cnn_pred <= 0.4:
            cv2.putText(self.image1, "Happy", (150, 120), cv2.FONT_ITALIC, 1, self.rgb2, 1)
        else:
            cv2.putText(self.image1, " ", (150, 120), cv2.FONT_ITALIC, 1, self.rgb2, 1)

        self.video_displayCNN(self.image1, 1)

        # 3
        random_pred = random_model.predict(img.reshape(1, self.image_width * self.image_height))

        # checking the label and printing the appropriate facial expression
        if (random_pred == 1):
            cv2.putText(self.image2, "sad", (150, 120), cv2.FONT_ITALIC, 1, (0, 255, 0), 1)
        elif (random_pred == 0):
            cv2.putText(self.image2, "Happy", (150, 120), cv2.FONT_ITALIC, 1, (0, 255, 0), 1)
        else:
            cv2.putText(self.image2, " ", (150, 120), cv2.FONT_ITALIC, 1, (0, 255, 0), 1)

        self.video_displayRFT(self.image2, 1)

    def plot_roc_curve(self):
        plt.LoadModels.load_models()


app = QApplication(sys.argv)
widget = Main()
widget.show()
sys.exit(app.exec_())
