
from collections import namedtuple, OrderedDict
import sys
import numpy as np
import scipy.sparse as sps
import pickle as pickle
from chemTree import *
from MainLogic import *
from ChemDBView import *

from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QAction, QApplication, QBrush, QCheckBox, QColor, QComboBox, QDialog, QFileDialog, QFormLayout, QGroupBox, QIcon, QInputDialog, QLabel, QLineEdit, QListWidget, QMenu, QMessageBox, QVBoxLayout, QHBoxLayout, QGridLayout, QMainWindow, QPainter, QPlainTextEdit, QProgressBar, QPushButton, QRadioButton, QSizePolicy, QSlider, QSpinBox, QSplitter, QStatusBar, QTabWidget, QTableWidget, QTreeView, QTreeWidget, QTreeWidgetItem, QToolBar, QToolButton, QWidget
from PyQt4.QtCore import Qt, pyqtSignal, QObject, QThread
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams, gridspec
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backend_bases import cursors
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from operator import itemgetter
from os import path
import six
import tabulate
from random import shuffle
import re
from linear_sum_assignment import linear_sum_assignment
import timeit
import time
import math
import os
import nmrglue as ng
from datetime import date

#from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT
from matplotlib.backend_bases import NavigationToolbar2
import matplotlib.backends.backend_qt5 as backend
try:
    import matplotlib.backends.qt_editor.figureoptions as figureoptions
except ImportError:
    figureoptions = None

version = '0.2.0'

cursord = {
    cursors.MOVE: Qt.SizeAllCursor,
    cursors.HAND: Qt.PointingHandCursor,
    cursors.POINTER: Qt.ArrowCursor,
    cursors.SELECT_REGION: Qt.CrossCursor,
    }

colrseq = [(0.1216,    0.4706,    0.7059),\
    (0.8902,    0.1020,    0.1098),\
    (0.2000,    0.6275,    0.1725),\
    (1.0000,    0.4980,         0),\
    (0.4157,    0.2392,    0.6039),\
    (0.6941,    0.3490,    0.1569),\
    (0.6510,    0.8078,    0.8902),\
    (0.6980,    0.8745,    0.5412),\
    (0.9843,    0.6039,    0.6000),\
    (0.9922,    0.7490,    0.4353),\
    (0.7922,    0.6980,    0.8392),\
    (1.0000,    1.0000,    0.6000)]      # Sequence of colors to plot the results

chemDB = readChemDB('chemDB.json')     # Load the chemical database

### Run the DB application ###

if __name__ == '__main__':
    app = 0
    app = QApplication(sys.argv)
    view_db = ChemDBView(chemDB)
    view_db.show()
    app.exec_()
