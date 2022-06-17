from PyQt6.QtWidgets import QApplication
import sys
from newWidget import widgetMain
if __name__=='__main__':
    Qapp = QApplication(sys.argv)
    MainApp = widgetMain('FaceDetect by Buglot')
    MainApp.show()
    sys.exit(Qapp.exec())