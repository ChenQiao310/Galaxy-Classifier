import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
from PyQt5 import QtWidgets
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QDialog, QApplication, QLabel


class Classifier(QDialog):
    img_path = ""
    def __init__(self):
        super(Classifier, self).__init__()
        loadUi("classifier01.ui", self)
        self.setStyleSheet("background-image:url(\"background1.jpg\") 0 0 0 0 stretch stretch;")
        self.photoView.setAlignment(Qt.AlignCenter)
        self.photoView.setText("\n\n  Drop Image Here  \n\n")
        self.setAcceptDrops(True)

        self.button.clicked.connect(self.classify)

    def dragEnterEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasImage:
            event.setDropAction(Qt.CopyAction)
            file_path = event.mimeData().urls()[0].toLocalFile()
            self.set_image(file_path)
            self.img_path = file_path
            event.accept()
        else:
            event.ignore()

    def set_image(self, file_path):
        self.set_pixmap(QPixmap(file_path))

    def set_pixmap(self, image):
        self.photoView.setPixmap(image)

    def classify(self):
        model = load_model("galaxyClassifier-model.h5")
        categories = {
            0: "Ring",
            1: "Irregular",
            2: "Other",
            3: "Merger"
        }

        SIZE = (224, 224)

        def preprocess_image(path):
            img = plt.imread(path)
            img = tf.image.resize_with_crop_or_pad(img, SIZE[0], SIZE[1])
            img = img / 255
            return img

        image = []
        img = preprocess_image(self.img_path)
        image.append(img)
        image = np.array(image)

        pred = model.predict(image)
        prediction = np.array(pred)
        prediction = np.argmax(prediction, axis=1)

        self.result.setText("\n  Galaxy ID:      "+self.img_path[-10:-4]+
                            "\n  Odd feature:  " + categories[prediction[0]])

app = QApplication(sys.argv)
mainWindow = Classifier()
widget = QtWidgets.QStackedWidget()
widget.addWidget(mainWindow)
widget.setFixedWidth(660)
widget.setFixedHeight(572)
widget.show()

app.exec_()