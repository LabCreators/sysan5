# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MAIN_WINDOW_lab5.ui'
#
# Created by: PyQt5 UI code generator 5.14.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.setWindowModality(QtCore.Qt.NonModal)
        Form.resize(969, 800)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(214, 207, 207))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 178, 102))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 178, 102))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 178, 102))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(214, 207, 207))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 178, 102))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 178, 102))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 178, 102))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(162, 162, 162))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 178, 102))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 178, 102))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 178, 102))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        Form.setPalette(palette)
        Form.setStyleSheet("background-color: rgba(255, 178, 102, 255);")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.groupBox_5 = QtWidgets.QGroupBox(Form)
        self.groupBox_5.setObjectName("groupBox_5")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.groupBox_5)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox = QtWidgets.QGroupBox(self.groupBox_5)
        self.groupBox.setStyleSheet("background-color: rgba(255, 178, 102, 255);")
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.intervalsTable = QtWidgets.QTableWidget(self.groupBox)
        self.intervalsTable.setGeometry(QtCore.QRect(10, 90, 881, 161))
        self.intervalsTable.setStyleSheet("background-color: rgba(255, 178, 102, 255);")
        self.intervalsTable.setObjectName("intervalsTable")
        self.intervalsTable.setColumnCount(7)
        self.intervalsTable.setRowCount(4)
        item = QtWidgets.QTableWidgetItem()
        self.intervalsTable.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.intervalsTable.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.intervalsTable.setVerticalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.intervalsTable.setVerticalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.intervalsTable.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.intervalsTable.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.intervalsTable.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.intervalsTable.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.intervalsTable.setHorizontalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.intervalsTable.setHorizontalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.intervalsTable.setHorizontalHeaderItem(6, item)
        self.intervalsTable.horizontalHeader().setVisible(False)
        self.intervalsTable.verticalHeader().setVisible(False)
        self.groupBox_2 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_2.setGeometry(QtCore.QRect(10, 280, 881, 271))
        self.groupBox_2.setObjectName("groupBox_2")
        self.label_2 = QtWidgets.QLabel(self.groupBox_2)
        self.label_2.setGeometry(QtCore.QRect(20, 30, 131, 31))
        self.label_2.setObjectName("label_2")
        self.s1Result = QtWidgets.QLineEdit(self.groupBox_2)
        self.s1Result.setGeometry(QtCore.QRect(170, 30, 691, 31))
        self.s1Result.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.s1Result.setObjectName("s1Result")
        self.label_3 = QtWidgets.QLabel(self.groupBox_2)
        self.label_3.setGeometry(QtCore.QRect(20, 80, 151, 41))
        self.label_3.setObjectName("label_3")
        self.s2Result = QtWidgets.QLineEdit(self.groupBox_2)
        self.s2Result.setGeometry(QtCore.QRect(170, 80, 691, 31))
        self.s2Result.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.s2Result.setObjectName("s2Result")
        self.label_4 = QtWidgets.QLabel(self.groupBox_2)
        self.label_4.setGeometry(QtCore.QRect(20, 140, 141, 41))
        self.label_4.setObjectName("label_4")
        self.s3Result = QtWidgets.QLineEdit(self.groupBox_2)
        self.s3Result.setGeometry(QtCore.QRect(170, 140, 691, 31))
        self.s3Result.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.s3Result.setObjectName("s3Result")
        self.label_5 = QtWidgets.QLabel(self.groupBox_2)
        self.label_5.setGeometry(QtCore.QRect(20, 200, 151, 41))
        self.label_5.setObjectName("label_5")
        self.s4Result = QtWidgets.QLineEdit(self.groupBox_2)
        self.s4Result.setGeometry(QtCore.QRect(170, 200, 691, 31))
        self.s4Result.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.s4Result.setObjectName("s4Result")
        self.exec_button = QtWidgets.QPushButton(self.groupBox)
        self.exec_button.setGeometry(QtCore.QRect(640, 30, 111, 51))
        self.exec_button.setObjectName("exec_button")
        self.groupBox_3 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_3.setGeometry(QtCore.QRect(10, 10, 611, 80))
        self.groupBox_3.setObjectName("groupBox_3")
        self.dopusk = QtWidgets.QLineEdit(self.groupBox_3)
        self.dopusk.setGeometry(QtCore.QRect(240, 20, 171, 41))
        self.dopusk.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.dopusk.setObjectName("dopusk")
        self.label = QtWidgets.QLabel(self.groupBox_3)
        self.label.setGeometry(QtCore.QRect(30, 20, 200, 40))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setMaximumSize(QtCore.QSize(200, 40))
        self.label.setStyleSheet("background-color: rgba(255, 178, 102, 255);")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.own = QtWidgets.QCheckBox(self.groupBox_3)
        self.own.setGeometry(QtCore.QRect(490, 20, 101, 20))
        self.own.setObjectName("own")
        self.I_name = QtWidgets.QComboBox(self.groupBox_3)
        self.I_name.setGeometry(QtCore.QRect(430, 50, 73, 22))
        self.I_name.setObjectName("I_name")
        self.I_name.addItem("")
        self.I_name.addItem("")
        self.I_name.addItem("")
        self.index_i = QtWidgets.QComboBox(self.groupBox_3)
        self.index_i.setGeometry(QtCore.QRect(510, 50, 41, 22))
        self.index_i.setObjectName("index_i")
        self.index_i.addItem("")
        self.index_i.addItem("")
        self.index_i.addItem("")
        self.index_i.addItem("")
        self.index_j = QtWidgets.QComboBox(self.groupBox_3)
        self.index_j.setGeometry(QtCore.QRect(560, 50, 41, 22))
        self.index_j.setObjectName("index_j")
        self.index_j.addItem("")
        self.index_j.addItem("")
        self.index_j.addItem("")
        self.index_j.addItem("")
        self.index_j.addItem("")
        self.index_j.addItem("")
        self.index_j.addItem("")
        self.plot_button = QtWidgets.QPushButton(self.groupBox)
        self.plot_button.setGeometry(QtCore.QRect(760, 30, 121, 51))
        self.plot_button.setObjectName("plot_button")
        self.verticalLayout.addWidget(self.groupBox)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.verticalLayout_2.addWidget(self.groupBox_5)

        self.retranslateUi(Form)
        self.exec_button.clicked.connect(Form.exec_clicked)
        self.own.toggled['bool'].connect(Form.own_toggled)
        self.plot_button.clicked.connect(Form.plot_clicked)
        self.I_name.currentIndexChanged['QString'].connect(Form.Iname_modified)
        self.index_i.currentIndexChanged['QString'].connect(Form.Indexi_modified)
        self.index_j.currentIndexChanged['QString'].connect(Form.Indexj_modified)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form1"))
        self.groupBox_5.setTitle(_translate("Form", "Знаходження допустимого інтервалу T0"))
        self.groupBox_2.setTitle(_translate("Form", "Результати обчислень"))
        self.label_2.setText(_translate("Form", "Для ситуації S1:"))
        self.label_3.setText(_translate("Form", "Для ситуації S2:"))
        self.label_4.setText(_translate("Form", "Для ситуації S3:"))
        self.label_5.setText(_translate("Form", "Для ситуації S4:"))
        self.exec_button.setText(_translate("Form", "Виконати"))
        self.groupBox_3.setTitle(_translate("Form", "Параметри"))
        self.dopusk.setText(_translate("Form", "0.6"))
        self.label.setText(_translate("Form", "Величина допуску"))
        self.own.setText(_translate("Form", "Свій варіант"))
        self.I_name.setItemText(0, _translate("Form", "Ip"))
        self.I_name.setItemText(1, _translate("Form", "Id"))
        self.I_name.setItemText(2, _translate("Form", "It"))
        self.index_i.setItemText(0, _translate("Form", "0"))
        self.index_i.setItemText(1, _translate("Form", "1"))
        self.index_i.setItemText(2, _translate("Form", "2"))
        self.index_i.setItemText(3, _translate("Form", "3"))
        self.index_j.setItemText(0, _translate("Form", "0"))
        self.index_j.setItemText(1, _translate("Form", "1"))
        self.index_j.setItemText(2, _translate("Form", "2"))
        self.index_j.setItemText(3, _translate("Form", "3"))
        self.index_j.setItemText(4, _translate("Form", "4"))
        self.index_j.setItemText(5, _translate("Form", "5"))
        self.index_j.setItemText(6, _translate("Form", "6"))
        self.plot_button.setText(_translate("Form", "Графіки"))
