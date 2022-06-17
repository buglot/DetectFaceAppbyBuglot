from PyQt6.QtWidgets import QWidget,QPushButton,QHBoxLayout,QLabel,QLineEdit,QVBoxLayout
from PyQt6.QtGui import QIcon,QImage
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import QTimer
import cv2
from tensorflow.keras.models import load_model
import tensorflow as tf
import os
import numpy as np
class widgetMain(QWidget):
    def __init__(self,name_title :str):
        super().__init__()
        self.setWindowTitle(name_title)
        self.__size =(50,200)
        self.setGeometry(200,30,self.__size[0],self.__size[1])
        #Object widget
        self.__intCam = QLabel('index cam')
        self.__idxcamwithint = QLineEdit('0')
        self.__buttonclick =QPushButton(QIcon(os.path.join('images','buttonclick.jpg')),'เริ่ม')
        self.__images_vi = QLabel()
        self.__time = QTimer()
        #layout set
        self.__layout1 = QHBoxLayout()
        self.__layout1.addWidget(self.__intCam)
        self.__layout1.addWidget(self.__idxcamwithint)
        self.__layout1.addWidget(self.__buttonclick)
        
        self.__layout2 =QVBoxLayout()
        self.__layout2.addLayout(self.__layout1)
        self.__layout2.addWidget(self.__images_vi)
        self.setLayout(self.__layout2)
        #connect etc
        self.__time.timeout.connect(self.viewCam)
        self.__buttonclick.clicked.connect(self.controlview)

    def viewCam(self):
            _,frame = self.__cap.read()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = rgb.shape
            resized = tf.image.resize(rgb, (120,120))

            yhat = self.facetracker.predict(np.expand_dims(resized/255,0))
            sample_coords = yhat[1][0]
            if yhat[0] > 0.5: 
                # Controls the main rectangle
                cv2.rectangle(frame, 
                            tuple(np.multiply(sample_coords[:2], [width,height]).astype(int)),
                            tuple(np.multiply(sample_coords[2:], [width,height]).astype(int)), 
                                    (255,0,0), 2)
                # Controls the label rectangle
                cv2.rectangle(frame, 
                            tuple(np.add(np.multiply(sample_coords[:2], [width,height]).astype(int), 
                                            [0,-30])),
                            tuple(np.add(np.multiply(sample_coords[:2], [width,height]).astype(int),
                                            [80,0])), 
                                    (255,0,0), -1)
                
                # Controls the text rendered
                cv2.putText(frame, 'face', tuple(np.add(np.multiply(sample_coords[:2], [width,height]).astype(int),
                                                    [0,-5])),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = image.shape
            step = channel * width
            # create QImage from image
            qImg = QImage(image.data, width, height, step, QImage.Format.Format_RGB888)
            self.__images_vi.setPixmap(QPixmap.fromImage(qImg))

    def controlview(self):
        if not self.__time.isActive():
            self.facetracker = load_model('face.h5')
            self.__cap = cv2.VideoCapture(int(self.__idxcamwithint.text()))
            self.__time.start(1)
            self.__buttonclick.setText('หยุด')
        else:
            # stop timer
            self.__time.stop()
            self.__images_vi.clear()
            # release video capture
            self.__cap.release()
            cv2.destroyAllWindows()
            # update control_bt text
            self.__buttonclick.setText('เริ่ม')