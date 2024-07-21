import cv2
import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QMessageBox
from PyQt5.uic import loadUi
import numpy as np

class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('../showimage.ui', self)
        self.image = None
        self.loadButton.clicked.connect(self.loadClicked)
        self.grayButton.clicked.connect(self.grayClicked)
        self.actionBrightness.triggered.connect(self.brightness)
        self.actionSimple_Contrast.triggered.connect(self.contrast)  # Connect contrast action
        self.actionContrast_Stretching.triggered.connect(self.contrastStretching)  # Connect contrast stretching action
        self.actionNegative_Image.triggered.connect(self.negativeImage)  # Connect negative image action
        self.actionBiner_Image.triggered.connect(self.binaryImage)  # Connect binary image action
        self.contrast_value = 1.6  # Default contrast value

    @pyqtSlot()
    def loadClicked(self):
        self.loadImage('../koala.jpeg')

    @pyqtSlot()
    def grayClicked(self):
        try:
            if self.image is not None:
                H, W = self.image.shape[:2]
                gray = np.zeros((H, W), np.uint8)
                for i in range(H):
                    for j in range(W):
                        gray[i, j] = np.clip(
                            0.299 * self.image[i, j, 0] + 0.587 * self.image[i, j, 1] + 0.114 * self.image[i, j, 2], 0, 255)
                self.image = gray
                self.displayImage(windows=2)  # Pass 2 to indicate displaying in the second label
            else:
                QMessageBox.critical(self, "Error", "No image loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during grayscale conversion: {str(e)}")

    @pyqtSlot()
    def brightness(self):
        try:
            if self.image is not None:
                brightness = 80
                self.image = np.clip(self.image.astype(int) + brightness, 0, 255).astype(np.uint8)
                self.displayImage(1)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    @pyqtSlot()
    def contrast(self):
        try:
            if self.image is not None:
                # Apply contrast enhancement
                self.image = np.clip(self.image.astype(float) * self.contrast_value, 0, 255).astype(np.uint8)
                self.displayImage(1)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    @pyqtSlot()
    def contrastStretching(self):
        try:
            if self.image is not None:
                # Applying contrast stretching
                min_val = np.min(self.image)
                max_val = np.max(self.image)
                stretched_image = 255 * ((self.image - min_val) / (max_val - min_val))
                self.image = stretched_image.astype(np.uint8)
                self.displayImage(1)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    @pyqtSlot()
    def negativeImage(self):
        try:
            if self.image is not None:
                self.image = 255 - self.image
                self.displayImage(1)
            else:
                QMessageBox.critical(self, "Error", "No image loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during negative transformation: {str(e)}")

    @pyqtSlot()
    def binaryImage(self):
        try:
            if self.image is not None:
                # Convert the image to grayscale
                gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                # Apply binary threshold
                _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
                self.image = binary_image
                self.displayImage(1)
            else:
                QMessageBox.critical(self, "Error", "No image loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during binary transformation: {str(e)}")

    def loadImage(self, flname):
        try:
            self.image = cv2.imread(flname)
            if self.image is None:
                raise FileNotFoundError(f"Could not load image: {flname}")
            self.displayImage()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def displayImage(self, windows=1):
        if self.image is not None:
            qformat = QImage.Format_Indexed8

            if len(self.image.shape) == 3:  # row[0], col[1], channel[2]
                if self.image.shape[2] == 4:
                    qformat = QImage.Format_RGBA8888
                else:
                    qformat = QImage.Format_RGB888
            img = QImage(self.image.data, self.image.shape[1], self.image.shape[0], self.image.strides[0], qformat)

            # OpenCV reads the image in BGR format, while PyQt reads it in RGB format
            img = img.rgbSwapped()
            if windows == 1:
                # Storing the loaded image in imgLabel
                self.imgLabel.setPixmap(QPixmap.fromImage(img))

                # Positioning the image at the center
                self.imgLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

            if windows == 2:
                self.hasilLabel.setPixmap(QPixmap.fromImage(img))
                self.hasilLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
                self.hasilLabel.setScaledContents(True)
        else:
            QMessageBox.critical(self, "Error", "No image loaded.")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ShowImage()
    window.setWindowTitle('Show Image GUI')
    window.show()
    sys.exit(app.exec_())