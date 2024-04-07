from PyQt5.QtWidgets import *
from PyQt5 import uic
import sys

form_class = uic.loadUiType("TestProgramUI.ui")[0]

class MyWindow(QMainWindow, form_class):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.printFunction)    
        
    def printFunction(self):
        self.lineEdit.setText('안녕하세요')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    app.exec_()