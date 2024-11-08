# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ip1.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.Load_Image = QtWidgets.QPushButton(self.centralwidget)
        self.Load_Image.setGeometry(QtCore.QRect(40, 40, 81, 61))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI Semilight")
        self.Load_Image.setFont(font)
        self.Load_Image.setObjectName("Load_Image")
        self.Smooth_Filter = QtWidgets.QPushButton(self.centralwidget)
        self.Smooth_Filter.setGeometry(QtCore.QRect(40, 130, 81, 61))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI Semilight")
        font.setBold(False)
        font.setWeight(50)
        self.Smooth_Filter.setFont(font)
        self.Smooth_Filter.setObjectName("Smooth_Filter")
        self.Sharp = QtWidgets.QPushButton(self.centralwidget)
        self.Sharp.setGeometry(QtCore.QRect(40, 220, 81, 61))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI Semilight")
        self.Sharp.setFont(font)
        self.Sharp.setObjectName("Sharp")
        self.Gaussian = QtWidgets.QPushButton(self.centralwidget)
        self.Gaussian.setGeometry(QtCore.QRect(40, 310, 81, 61))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI Semilight")
        self.Gaussian.setFont(font)
        self.Gaussian.setObjectName("Gaussian")
        self.Lowerpass = QtWidgets.QPushButton(self.centralwidget)
        self.Lowerpass.setGeometry(QtCore.QRect(40, 400, 81, 61))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI Semilight")
        self.Lowerpass.setFont(font)
        self.Lowerpass.setObjectName("Lowerpass")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(210, 40, 211, 171))
        self.textEdit.setObjectName("textEdit")
        self.textEdit4 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit4.setGeometry(QtCore.QRect(520, 290, 211, 171))
        self.textEdit4.setObjectName("textEdit4")
        self.textEdit2 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit2.setGeometry(QtCore.QRect(520, 40, 211, 171))
        self.textEdit2.setObjectName("textEdit2")
        self.textEdit3 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit3.setGeometry(QtCore.QRect(210, 290, 211, 171))
        self.textEdit3.setObjectName("textEdit3")
        self.Label1 = QtWidgets.QLabel(self.centralwidget)
        self.Label1.setGeometry(QtCore.QRect(210, 220, 211, 31))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI Semilight")
        font.setPointSize(15)
        self.Label1.setFont(font)
        self.Label1.setAlignment(QtCore.Qt.AlignCenter)
        self.Label1.setObjectName("Label1")
        self.Label4 = QtWidgets.QLabel(self.centralwidget)
        self.Label4.setGeometry(QtCore.QRect(520, 470, 211, 31))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI Semilight")
        font.setPointSize(15)
        self.Label4.setFont(font)
        self.Label4.setAlignment(QtCore.Qt.AlignCenter)
        self.Label4.setObjectName("Label4")
        self.Label2 = QtWidgets.QLabel(self.centralwidget)
        self.Label2.setGeometry(QtCore.QRect(520, 220, 211, 31))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI Semilight")
        font.setPointSize(15)
        self.Label2.setFont(font)
        self.Label2.setAlignment(QtCore.Qt.AlignCenter)
        self.Label2.setObjectName("Label2")
        self.Label3 = QtWidgets.QLabel(self.centralwidget)
        self.Label3.setGeometry(QtCore.QRect(210, 470, 211, 31))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI Semilight")
        font.setPointSize(15)
        self.Label3.setFont(font)
        self.Label3.setAlignment(QtCore.Qt.AlignCenter)
        self.Label3.setObjectName("Label3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Load_Image.setText(_translate("MainWindow", "Load Image"))
        self.Smooth_Filter.setText(_translate("MainWindow", "Smooth Filter"))
        self.Sharp.setText(_translate("MainWindow", "Sharp"))
        self.Gaussian.setText(_translate("MainWindow", "Gaussian"))
        self.Lowerpass.setText(_translate("MainWindow", "Lower-pass"))
        self.Label1.setText(_translate("MainWindow", "Original image"))
        self.Label4.setText(_translate("MainWindow", "TextLabel"))
        self.Label2.setText(_translate("MainWindow", "TextLabel"))
        self.Label3.setText(_translate("MainWindow", "TextLabel"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())