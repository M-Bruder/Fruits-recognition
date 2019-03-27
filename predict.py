import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array

from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QLineEdit, QPushButton, QHBoxLayout, QSizePolicy,  QLabel, QGridLayout, QMessageBox
from PyQt5.QtCore import Qt

model_path = './models/model.h5'
model_weights_path = './models/weights.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())

class windowFruits(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.left = 10
        self.top = 10
        self.width = 500
        self.height = 500

        self.interface()

    def interface(self):

         # Window background color
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), Qt.white)
        self.setPalette(p)

        # Label name
        label1 = QLabel("Obraz:", self)
        label2 = QLabel("Wynik:", self)
        self.label = QLabel()

        positionLabels = QGridLayout()
        positionLabels.addWidget(label1, 1, 0)
        positionLabels.addWidget(label2, 1, 1)
        

        positionLabels.addWidget(self.label, 0, 0, 1, 3)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label.setAlignment(Qt.AlignCenter)

        pixmap = QPixmap()
        self.label.setPixmap(pixmap)


        # Edit field
        self.fileEdt = QLineEdit()
        self.recognizeEdt = QLineEdit()

        self.recognizeEdt.readonly = True
        self.fileEdt.readonly = True

        positionLabels.addWidget(self.fileEdt, 2, 0)
        positionLabels.addWidget(self.recognizeEdt, 2, 1)

        # Buttons
        loadFileBtn = QPushButton("&Wczytaj obraz", self)
        recognizeBtn = QPushButton("&Rozpoznaj", self)
        endBtn = QPushButton("&Wyjdź", self)
        endBtn.resize(endBtn.sizeHint())

        loadFileBtn.setStyleSheet("color: white; background-color: black; font-size: 16px; padding: 5px;")
        recognizeBtn.setStyleSheet("color: white; background-color: black; font-size: 16px; padding: 5px;")
        endBtn.setStyleSheet("color: white; background-color: red; font-size: 16px; font-weight: bold;")

        positionBtns = QHBoxLayout()
        positionBtns.addWidget(loadFileBtn)
        positionBtns.addWidget(recognizeBtn)
        
        positionLabels.addLayout(positionBtns, 3, 0, 1, 3)
        positionLabels.addWidget(endBtn, 4, 0, 1, 3)

        self.setLayout(positionLabels)

        endBtn.clicked.connect(self.end)
        loadFileBtn.clicked.connect(self.getfiles)
        recognizeBtn.clicked.connect(self.prediction)


        self.setGeometry(300, 300, 700, 400)
        self.setWindowIcon(QIcon('./icon/icon.png'))
        self.setWindowTitle("Rozpoznawanie owoców")
        self.show()

    def end(self):
        self.close()

    def closeEvent(self, event):

        answer = QMessageBox.question(
            self, 'Komunikat',
            "Czy na pewno koniec?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if answer == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()

    def getfiles(self):
        fileName, _ = QFileDialog.getOpenFileName(self,'Open File', '', 'Images (*.png *.jpeg *.jpg)')
        filePath = fileName
        self.file = filePath
        print(self.file)
        self.fileEdt.setText(fileName)
        image = QImage(fileName)
        self.label.setPixmap(QPixmap.fromImage(image))   
   
    
    def prediction(self):   
        img = load_img(self.file, target_size=(100,100))
        x = img_to_array(img)
        array = x / 255
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = model.predict_classes(images)
        pred = model.predict(images)
        if classes  == 0:
            self.recognizeEdt.setText('Borówka')
        elif classes  == 1:
            self.recognizeEdt.setText('Wiśnia')
        elif classes  == 2:
            self.recognizeEdt.setText('Brzoskwinia')
        elif classes  == 3:
            self.recognizeEdt.setText('Malina')    
        else:
            self.recognizeEdt.setText('N/A')

if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    window = windowFruits()
    sys.exit(app.exec_())

