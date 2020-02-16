# coding: utf8
import os
import logging
import sys
from PyQt5.QtWidgets import QTableWidgetItem
from logic import getDecisionTime, graph_plot, get_one_history
import matplotlib.pyplot as plt


def _append_run_path():
    if getattr(sys, 'frozen', False):
        pathlist = []

        # If the application is run as a bundle, the pyInstaller bootloader
        # extends the sys module by a flag frozen=True and sets the app
        # path into variable _MEIPASS'.
        pathlist.append(sys._MEIPASS)

        # the application exe path
        _main_app_path = os.path.dirname(sys.executable)
        pathlist.append(_main_app_path)

        # append to system path enviroment
        os.environ["PATH"] += os.pathsep + os.pathsep.join(pathlist)

    logging.error("current PATH: %s", os.environ['PATH'])
_append_run_path()

from PyQt5.QtCore import pyqtSlot, pyqtSignal
from PyQt5.QtGui import QTextDocument, QFont
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox

app = QApplication(sys.argv)
app.setApplicationName('lab3_sa')
from MAIN_WINDOW import Ui_Form
#from new_main import Ui_Form
#from MAIN_WINDOW import Ui_Form



class MainWindow(QDialog, Ui_Form):

    def __init__(self, *args):
        super(MainWindow, self).__init__(*args)

        # setting up ui
        self.setupUi(self)

        # other initializations
        self.dop = 0.6
        self.isOwn = self.own.isChecked()
        self.tableArray, self.df_history, self.time, self.predictions = getDecisionTime(self.dop)
        self.Iname = 'Ip'
        self.index_i = 0
        self.index_j = 0
        return

    @pyqtSlot(bool)
    def own_toggled(self, val):
        self.isOwn = val

    def _prepare_table(self):
        self.tableArray, self.df_history, self.time, self.predictions = getDecisionTime(float(self.dop), self.isOwn)
        self.s1Result.setText(self.predictions[0])
        self.s2Result.setText(self.predictions[1])
        self.s3Result.setText(self.predictions[2])
        self.s4Result.setText(self.predictions[3])
        for row, val in enumerate(self.tableArray):
            for col, val_cal in enumerate(val):
                temp_cell = QTableWidgetItem()
                temp_cell.setText(val_cal)
                self.intervalsTable.setItem(row, col, temp_cell)

    @pyqtSlot()
    def exec_clicked(self):
        self.exec_button.setEnabled(False)
        try:
            self.dop = self.dopusk.text()
            self._prepare_table()
        except Exception as e:
            QMessageBox.warning(self, 'Error!', 'Error happened during execution: ' + str(e))
        self.exec_button.setEnabled(True)
        return

    @pyqtSlot()
    def plot_clicked(self):
        self.plot_button.setEnabled(False)
        try:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(self.time, get_one_history(self.df_history, self.Iname, self.index_i, self.index_j))
            ax.set(xlabel='time, s', ylabel=self.Iname, title='Physics of {}[{}, {}]'.format(self.Iname, self.index_i, self.index_j))
            ax.grid()
            fig.savefig("Physics of_{}[{}, {}].png".format(self.Iname, self.index_i, self.index_j))
            plt.show()
        except Exception as e:
            QMessageBox.warning(self, 'Error!', 'Error happened during execution: ' + str(e))
        self.plot_button.setEnabled(True)

    @pyqtSlot('QString')
    def Iname_modified(self, val):
        try:
            self.Iname = val
        except Exception as e:
            QMessageBox.warning(self, 'Error!', 'Error happened during execution: ' + str(e))

    @pyqtSlot('QString')
    def Indexi_modified(self, val):
        try:
            self.index_i = int(val)
        except Exception as e:
            QMessageBox.warning(self, 'Error!', 'Error happened during execution: ' + str(e))

    @pyqtSlot('QString')
    def Indexj_modified(self, val):
        try:
            self.index_j = int(val)
        except Exception as e:
            QMessageBox.warning(self, 'Error!', 'Error happened during execution: ' + str(e))

    def _get_params(self):
        return dict(poly_type=self.type, solving_method=self.method, degrees=self.degrees, dimensions=self.dimensions,
                    samples=self.samples_num, input_file="Data/dst(48,2,2,2,2).txt", output_file=self.output_path,
                    weights=self.weight_method, lambda_multiblock=self.lambda_multiblock, custom=self.custom_func_struct,
                    custom_method=self.custom_method, predLength=self.predLength, results_field=self.tableWidget,
                    data=self.data_type, modeltype=self.model, predlength=self.predLength, poldegree=self.degrees)


# -----------------------------------------------------#
form = MainWindow()
form.setWindowTitle('System Analysis - Lab4')
form.show()
sys.exit(app.exec_())
#python -m PyQt5.uic.pyuic main_window.ui -o main_window.py -x
