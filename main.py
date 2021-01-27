import sys
import numpy as np
import dill
from MainLogic import *
from MainLogic import Series, Datum, Workspace
from dataio import *
import config

from PyQt4 import QtGui, QtCore, uic
from PyQt4.QtGui import QAction, QActionGroup, QApplication, QBrush, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QGroupBox, QIcon, QInputDialog, QItemSelectionModel, QItemDelegate, QLabel, QLineEdit, QListWidget, QMenu, QMessageBox, QVBoxLayout, QHBoxLayout, QGridLayout, QMainWindow, QPalette, QPen, QPlainTextEdit, QProgressBar, QPushButton, QRadioButton, QSizePolicy, QSlider, QSpinBox, QSplitter, QStatusBar, QStyle, QTableView, QTabWidget, QTableWidget, QToolButton, QTreeView, QToolBar, QToolTip, QWidget
from PyQt4.QtCore import Qt, pyqtSignal, QObject, QThread, QEvent
import pyqtgraph as pg
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backend_bases import cursors
from matplotlib.figure import Figure
from operator import itemgetter
from os import path
from random import shuffle
import re
import math
import os
import nmrglue as ng
from datetime import date

# Set white background in plots
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)

try:
    import matplotlib.backends.qt_editor.figureoptions as figureoptions
except ImportError:
    figureoptions = None

version = '2.0.1'
compile_standalone = False   # Change to False for debugging/development to output the results into the usual console

cursord = {
    cursors.MOVE: Qt.SizeAllCursor,
    cursors.HAND: Qt.PointingHandCursor,
    cursors.POINTER: Qt.ArrowCursor,
    cursors.SELECT_REGION: Qt.CrossCursor,
    }

global settings
settings = dict()       # A dictionary of settings for processing
steps = []

class EmittingStream(QObject):
    """For printing text in a textEdit."""

    textWritten = pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

class CfunPopup(QWidget):
    """Popup window that shows the objective function"""
    def __init__(self, cfun):
        QWidget.__init__(self)
        self.cfun = cfun

        # a figure instance to plot on
        self.figure = Figure()     #       plt.figure() #

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        # create axes
        self.ax = []
        self.ax.append(self.figure.add_subplot(111))

        # set the layout
        self.mainLayout = QVBoxLayout()
        self.mainLayout.addWidget(self.toolbar)
        self.mainLayout.addWidget(self.canvas)

        # create and set the central widget
        self.setLayout(self.mainLayout)

        res = np.array([self.cfun(x) for x in np.linspace(-1, 1, 30)])

        self.ax[0].clear()    # discards the old graph
        self.ax[0].plot(res[:,0], res[:,1], '-')        # plot data
        self.canvas.draw()             # refresh canvas

    """def paintEvent(self, e):
        #dc = QPainter(self)
        #dc.drawLine(0, 0, 100, 100)
        #dc.drawLine(100, 0, 0, 100)"""

from PyQt4.QtGui import QDialog, QVBoxLayout, QDateTimeEdit, QApplication
from PyQt4.QtCore import Qt, QDateTime

class MyDoubleEdit(QLineEdit):

    valueChanged = pyqtSignal(object)    # Returns the index of the changed freqBlock

    def __init__(self, value=None, parent=None):
        super().__init__(parent)
        self.setValue(value)
        self.editingFinished.connect(self.onEditingFinished)
        self.setValidator(QtGui.QDoubleValidator())

    def value(self):
        return self._value

    def setValue(self, value):
        self._value = value
        if self._value is not None:
            self.setText("{:.5g}".format(self._value))
        else:
            self.setText("")
        self.valueChanged.emit(value)

    def onEditingFinished(self):
        self.setValue(float(self.text()))

    """def focusInEvent(self, event):
        if not self.inFocus:
            if self._value is not None:
                self.setText("{:.10g}".format(self._value))
            self.inFocus = True

    def focusOutEvent(self, event):
        self.inFocus = False"""

class ChooseFromDBDialog(QDialog):
    """A dialog to choose chemicals from the DB to add to the tree."""

    def __init__(self, forbidden_names=None, parent = None):
        super(ChooseFromDBDialog, self).__init__(parent)

        self.forbidden_names = forbidden_names if forbidden_names is not None else []

        layout = QFormLayout(self)
        self.setWindowTitle('Add from database')
        self.setWindowFlags(Qt.WindowTitleHint)
        self.setWindowFlags(Qt.Dialog | Qt.MSWindowsFixedSizeDialogHint | Qt.WindowTitleHint)

        # Add widgets for entering parameters
        self.cmboxLibs, self.cmboxChem = QComboBox(), QComboBox()
        self.cmboxLibs.addItem('All species')
        self.cmboxLibs.addItems( sorted([k for k, _ in chemLib.items()]) )
        self.cmboxLibs.currentIndexChanged.connect(self.onLibrarySelected)
        self.cmboxChem.currentIndexChanged.connect(self.onChemicalSelected)
        self.nameEdit = QLineEdit()
        self.nameEdit.textChanged.connect(self.onTextChanged)

        layout.addRow('Database', self.cmboxLibs)
        layout.addRow('Chemical', self.cmboxChem)
        layout.addRow('Display as', self.nameEdit)

        # OK and Cancel buttons
        self.buttonsBox = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)
        layout.addWidget(self.buttonsBox)

        self.buttonsBox.accepted.connect(self.accept)
        self.buttonsBox.rejected.connect(self.reject)

        # Initialize the comboboxes
        # self.onLibrarySelected(0)
        self.cmboxLibs.setCurrentIndex( self.cmboxLibs.findText('All species') )    # Find the index of the built-in DB
        self.onLibrarySelected(0)

    def onLibrarySelected(self, indx):
        """Sets the items fro the second combo box."""
        self.cmboxChem.clear()
        lib_key = self.cmboxLibs.itemText(indx)
        try:
            keysDB = sorted([key for key, val in chemLib[lib_key].items()])
        except KeyError:
            # List all species
            keysDB = sorted(list(set([key for lib_key in chemLib.keys() for key in chemLib[lib_key].keys()])))
        self.cmboxChem.addItems(keysDB)

    def onChemicalSelected(self, indx):
        """Called when a chemical has been selected from the database."""
        name_in_DB = self.cmboxChem.itemText(indx)   # or self.cmboxChem.currentText()
        self.nameEdit.setText(name_in_DB)

    def onTextChanged(self, text):
        """Called when the text in the Name Edit changes."""

        if text in self.forbidden_names:
            self.setToolTip('A node with this name already exists in the model tree. Would you like to use the same model with a different name?')
            self.nameEdit.setStyleSheet("""
                                            QLineEdit {
                                                        background-color: rgb(249, 193, 203);
                                                        foreground-color: rgb(124, 44, 59);
                                                        border: 1px solid
                                                      }
                                        """)                              # Need to add the border, because otherwise the LineEdit flickers on Windows
            self.buttonsBox.buttons()[0].setEnabled(False)                # Disable the OK button
        else:
            self.setToolTip('')
            self.nameEdit.setStyleSheet("""
                                            QLineEdit {
                                                      }
                                        """)                              # Restore the normal style (white background)
            self.buttonsBox.buttons()[0].setEnabled(True)                 # Enable the OK button

    # get the selection
    def getSelection(self):
        """Returns the chosen display name for the new chemical and the corresponding chemSpec structure from the database."""
        displayName = self.nameEdit.text()
        libsName = self.cmboxLibs.currentText()
        chemName = self.cmboxChem.currentText()

        try:
            return displayName, copy.deepcopy(chemLib[libsName][chemName])
        except KeyError:
            return displayName, copy.deepcopy([spec for lib_key in chemLib.keys() for key, spec in chemLib[lib_key].items() if key==chemName][0])

    # static method to create the dialog and return (name, QDpars, accepted)
    @staticmethod
    def run(forbidden_names=None, parent = None):
        dialog = ChooseFromDBDialog(forbidden_names=forbidden_names, parent=parent)
        result = dialog.exec_()
        if result == QDialog.Accepted:    # If OK was clicked
            return (*dialog.getSelection(), result == QDialog.Accepted)
        else: return (None, None, result == QDialog.Accepted)

class SettingsDialog(QDialog):
    def __init__(self, oldSettings, parent = None):
        super(SettingsDialog, self).__init__(parent)

        self.setWindowTitle('Settings')
        self.setWindowFlags(Qt.WindowTitleHint)
        self.setWindowFlags(Qt.Dialog | Qt.MSWindowsFixedSizeDialogHint | Qt.WindowTitleHint)

        layout = QVBoxLayout(self)

        # ----------- Radio buttons for seleting the starting values -----------
        self.rbtnStartFromCurrent = QRadioButton("Start with current values")
        self.rbtnStartFromPrevious = QRadioButton("Copy values from previous")
        self.rbtnStartFromDefault = QRadioButton("Start with default values")
        if config.OPTIM_startFrom == 'default':
            self.rbtnStartFromDefault.setChecked(True)
        elif config.OPTIM_startFrom == 'previous':
            self.rbtnStartFromPrevious.setChecked(True)
        else: self.rbtnStartFromCurrent.setChecked(True)    # if config.OPTIM_startFrom == 'current'
        self.chkboxAutoPhasing = QCheckBox("Auto phase each spectrum")
        self.chkboxAutoPhasing.setChecked(settings["autoPhase"])
        self.chkboxAutoPicking = QCheckBox("Pick peaks automatically")
        self.chkboxAutoPicking.setChecked(settings["autoPick"])
        groupLayout = QVBoxLayout()
        groupLayout.addWidget(self.rbtnStartFromCurrent)
        groupLayout.addWidget(self.rbtnStartFromPrevious)
        groupLayout.addWidget(self.rbtnStartFromDefault)
        # groupLayout.addWidget(self.chkboxAutoPhasing)
        # groupLayout.addWidget(self.chkboxAutoPicking)
        rbtnGroup = QGroupBox("Processing series of spectra")
        rbtnGroup.setLayout(groupLayout)

        # ----------- Settings for the QD simulations --------------------------
        self.cmboxHCSelector = QComboBox()
        self.cmboxHCSelector.addItem("1H")
        self.cmboxHCSelector.addItem("13C")
        self.cmboxHCSelector.setCurrentIndex( self.cmboxHCSelector.findText(oldSettings['HCmode']) )
        self.editRerunThreshold = MyDoubleEdit(config.QD_RerunQDchshThreshold)
        self.editAggregateThreshold = MyDoubleEdit(config.QD_AggregatePeaksThreshold)
        groupLayout = QFormLayout()
        groupLayout.addRow("Nucleus", self.cmboxHCSelector)
        groupLayout.addRow("Merge resonances closer than, Hz", self.editAggregateThreshold)
        groupLayout.addRow("Update if chsh changed by, Hz", self.editRerunThreshold)
        qdConfigGroup = QGroupBox("QD settings")
        qdConfigGroup.setLayout(groupLayout)

        # ---------- Settings of the fitting algorithm -------------------------
        self.spbxBasinhopping = QSpinBox()
        self.spbxBasinhopping.setMinimum(0)
        self.spbxBasinhopping.setValue(config.OPTIM_maxBasinhoppingSteps)
        optiConfigGroup = QGroupBox('Optimization settings')
        groupLayout = QFormLayout()
        groupLayout.addRow('Number of basinhopping steps', self.spbxBasinhopping)
        optiConfigGroup.setLayout(groupLayout)

        # ---------------------------- LS/TLS ----------------------------------
        """self.rbtnLS = QRadioButton("LS")
        self.rbtnTLS = QRadioButton("TLS")
        if config.SAMPL_funcType == "LS":
            self.rbtnLS.setChecked(True)
        elif config.SAMPL_funcType == "TLS":
            self.rbtnTLS.setChecked(True)
        groupLayout = QHBoxLayout()
        groupLayout.addWidget(self.rbtnLS)
        groupLayout.addWidget(self.rbtnTLS)
        lklfConfigGroup = QGroupBox("Likelihood function")
        lklfConfigGroup.setLayout(groupLayout)"""

        # ---------- Settings for MCMC sampling --------------------------------
        self.chkboxRobustLS = QCheckBox("Use robust LS estimator")
        self.chkboxRobustLS.setChecked(config.SAMPL_robustLS)
        """self.rbtnUsual = QRadioButton("Usual")
        self.rbtnRobust = QRadioButton("Robust")
        self.rbtnLiberal = QRadioButton("Liberal")
        #self.rbtnTight = QRadioButton("Tight")
        if config.SAMPL_varEstimator == "usual":
            self.rbtnUsual.setChecked(True)
        elif config.SAMPL_varEstimator == "robust":
            self.rbtnRobust.setChecked(True)
        elif config.SAMPL_varEstimator == "liberal":
            self.rbtnLiberal.setChecked(True)
        groupLayout = QVBoxLayout()
        groupLayout.addWidget(self.rbtnUsual)
        groupLayout.addWidget(self.rbtnRobust)
        groupLayout.addWidget(self.rbtnLiberal)
        smplConfigGroup = QGroupBox("Variance estimator")
        smplConfigGroup.setLayout(groupLayout)"""

        # ---------- Put all groupboxes together -------------------------------
        layout.addWidget(rbtnGroup)
        layout.addWidget(qdConfigGroup)
        layout.addWidget(optiConfigGroup)
        # layout.addWidget(self.chkboxRobustLS)
        # layout.addWidget(lklfConfigGroup)
        # layout.addWidget(smplConfigGroup)

        # OK and Cancel buttons
        self.buttonsBox = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)
        layout.addWidget(self.buttonsBox)

        self.buttonsBox.accepted.connect(self.accept)
        self.buttonsBox.rejected.connect(self.reject)

    # get current date and time from the dialog
    def getEntries(self):
        newSettings = {}

        # Update the starting values settings
        if self.rbtnStartFromPrevious.isChecked():
            config.OPTIM_startFrom = 'previous'
        elif self.rbtnStartFromDefault.isChecked():
            config.OPTIM_startFrom = 'default'
        else: config.OPTIM_startFrom = 'current'
        settings.update({"autoPhase" : self.chkboxAutoPhasing.isChecked(),
                         "autoPick" : self.chkboxAutoPicking.isChecked()})

        # Update the QD settings
        config.QD_RerunQDchshThreshold = self.editRerunThreshold.value()
        config.QD_AggregatePeaksThreshold = self.editAggregateThreshold.value()
        newSettings['HCmode'] = self.cmboxHCSelector.currentText()

        # Update the optimization settings
        config.OPTIM_maxBasinhoppingSteps = self.spbxBasinhopping.value()

        # Choose the lieklihhod function
        """if self.rbtnLS.isChecked():
            config.SAMPL_funcType = 'LS'
        elif self.rbtnTLS.isChecked():
            config.SAMPL_funcType = 'TLS'"""

        # Update the sampling settings
        """if self.rbtnUsual.isChecked():
            config.SAMPL_varEstimator = 'usual'
        elif self.rbtnRobust.isChecked():
            config.SAMPL_varEstimator = 'robust'
        else: config.SAMPL_varEstimator = 'liberal'"""
        config.SAMPL_robustLS = True if self.chkboxRobustLS.isChecked() else False

        return newSettings

    # static method to create the dialog and return
    @staticmethod
    def run(oldSettings, parent = None):
        dialog = SettingsDialog(oldSettings, parent)
        result = dialog.exec_()
        if result == QDialog.Accepted:    # If OK was clicked
            newSettings = dialog.getEntries()
        else: newSettings = oldSettings
        #else: chkdForAll = False
        return QDialog.Accepted, newSettings

class SpectrumPlotItem(pg.PlotItem):
    """A customized PlotItem with zoomed out view."""

    def __init__(self, parent=None, title=None):
        super().__init__(parent)

    def resizeEvent(self, evt):
        super().resizeEvent(evt)

class PlotStemsItem(pg.PlotCurveItem):
    """Plotting group of stem lines with mouse interaction capabilities."""

    sigStemsDragged = pyqtSignal(object, float)          # Emmited when the chemical shift key changes by the amount delta
    sigStemsHovered = pyqtSignal(object, bool)                   # Key of the stems that are hovered over
    sigStemsClicked = pyqtSignal(object)                 # Key of the stems that are clicked

    def __init__(self, key, chsh_0, chsh_stems, intn_stems, color, *args, **kwargs):

        self.key = key
        self.x_stem = np.repeat(chsh_stems, 2)
        self.y_stem = np.dstack((np.zeros(len(intn_stems)), intn_stems)).flatten()
        self.chsh_0 = chsh_0
        super().__init__(x=self.x_stem, y=self.y_stem,
                         connect='pairs', pen={'color':color, 'width':1}, *kwargs)
        # self.opts['mouseWidth'] = 2
        self.setAcceptHoverEvents(True)

    def mouseClickEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            return
        if self.mouseShape().contains(ev.pos()):
            ev.accept()
            self.sigStemsClicked.emit(self.key)

    def highlight(self, flag=True):
        self.setPen(color=self.opts['pen'].color(), width=2 if flag else 1)

    def mouseDragEvent(self, evt):
        evt.accept()
        # print('Dragging in PlotStemsItem')

        # Main movement
        delta_chsh = evt.pos().x() - evt.buttonDownPos().x()
        self.setData(x=self.x_stem+delta_chsh, y=self.y_stem)

        # End of the event
        if evt.isFinish():
            self.x_stem += delta_chsh
            self.chsh_0 += delta_chsh
            self.sigStemsDragged.emit(self.key, delta_chsh)

    def hoverEnterEvent(self, evt):
        # ev.accept()
        # print('Hovering into PlotStemsItem', key)
        self.highlight(True)
        self.sigStemsHovered.emit(self.key, True)

    def hoverLeaveEvent(self, evt):
        # ev.accept()
        # print('Leaving PlotStemsItem', key)
        self.highlight(False)
        self.sigStemsHovered.emit(self.key, False)

class MainSpectrumWidget(pg.GraphicsLayoutWidget):

    sigFreqRangeChanged = pyqtSignal(int, object)      # Emmited if ranges of a FreqBlock change (indx, lims)
    sigFreqRangeSelected = pyqtSignal(object)          # Boundaries of the new selection (lims)
    sigMouseClicked = pyqtSignal(object)               # Returns the position where the click has occured in the main view (x value is in ppm)
    sigStemsClicked = pyqtSignal(object)
    sigStemsDragged = pyqtSignal(object, float)
    sigPivotDragged = pyqtSignal(float)

    def __init__(self, parent=None, title=None):
        super().__init__(parent)

        # Set arrays to hold the data
        self._f = None
        self._xF = None
        self._yF = None
        self._zF = []
        self._stems = {}
        self._freqBlocks = []         # List of linearRegionItems that indicate the frequency blocks

        self._state = None
        self._stemDragging_flag = False
        self._globalChsh_flag = False              # Set TRUE when the global chemical shift is being set by dragging

        # Create the main plot for the spectrum
        p0 = SpectrumPlotItem()
        self.addItem(p0, row=0, col=0, rowspan=1, colspan=1)                      # Can use p0 = self.addPlot(0, 0)
        p0.showGrid(x = True, y = True, alpha = 0.25)
        p0.invertX(True)
        p0.setLabels(bottom='Chemical shift, ppm', left='Intensity, a.u.')
        p0.sigXRangeChanged.connect(self.onXRangeChanged)

        # Create the plot for the residual
        p1 = self.addPlot(1, 0)
        p1.showGrid(x = True, y = True, alpha = 0.25)
        p1.invertX(True)
        p1.setLabels(bottom='Chemical shift, ppm', left=' ')
        p1.setXLink(p0)
        p0.setXLink(p1)
        p1.hide()

        # Create another vertical axis to plot the stem lines. create a new ViewBox, link the right axis to its coordinate system
        self.p0r = pg.ViewBox()
        self.p0r.invertX(True)
        self.p0r.setLimits(yMin=0.0)
        self.p0r.setRange(yRange=(0.0, 5.0))
        p0.scene().addItem(self.p0r)
        p0.getAxis('right').linkToView(self.p0r)
        self.p0r.setXLink(p0)
        self.p0r.setMouseEnabled(y=False)

        # Handle view resizing
        def updateViews():
            ## view has resized; update auxiliary views to match
            self.p0r.setGeometry(p0.vb.sceneBoundingRect())

            ## need to re-update linked axes since this was called
            ## incorrectly while views had different shapes.
            ## (probably this should be handled in ViewBox.resizeEvent)
            self.p0r.linkedViewChanged(p0.vb, self.p0r.XAxis)

        updateViews()
        p0.vb.sigResized.connect(updateViews)

        # self.layout().setSpacing(0.)
        self.setContentsMargins(0., 0., 0., 0.)

        self.ci.layout.setRowStretchFactor(0, 10)

        # ----------- Phasing pivot -------------
        self._phasingPivot = pg.InfiniteLine(pos=0.0, movable=True, pen=pg.mkPen(color=(0,0,255), width=3.0),
                                             label='Phasing pivot', labelOpts={'angle':90, 'position':0.9, 'anchors':[(0.5, 0), (0.5, 1)] })
        # self._phasingPivot.addMarker('<|>', position=0.5, size=10.0)
        self._phasingPivot.sigPositionChangeFinished.connect(self.onPivotDragged)

        # -------------- Crosshair --------------
        self._crossLines = {'v0' : pg.InfiniteLine(angle=90, movable=False),
                            'v1' : pg.InfiniteLine(angle=90, movable=False),
                            'h0' : pg.InfiniteLine(angle=0, movable=False),
                            'h1' : pg.InfiniteLine(angle=0, movable=False)}
        for key, val in self._crossLines.items():
            if '0' in key:
                p0.addItem(val, ignoreBounds=True)
            else:
                p1.addItem(val, ignoreBounds=True)
            val.hide()

        # --- MouseDrag Event in ViewBox of p0 ---
        p0.vb.mouseDragEvent = self._onMouseDragEvent        ## get the viewbox from the plot item and set its mouseDragEvent

        # Connect mouse signals
        # self.getItem(0,0).scene().sigMouseHover.connect(self._onMouseHover)
        self.getItem(0,0).scene().sigMouseClicked.connect(self._onMouseClicked)

        # --------- Small Insert Plot -----------
        # Add small subplot to the GraphicsScene
        self.FullViewPlotItem = pg.PlotItem()
        p0.scene().addItem(self.FullViewPlotItem)
        self.FullViewPlotItem.setPos(695, 5)
        self.FullViewPlotItem.resize(200, 50)
        self.FullViewPlotItem.setAutoFillBackground(True)
        self.FullViewPlotItem.setContentsMargins(0., 0., 0., 0.)
        # print(self.FullViewPlotItem.pos())
        # print(self.FullViewPlotItem.size().width())

        self.FullViewPlotItem.invertX(True)
        self.FullViewPlotItem.hideAxis('left')
        self.FullViewPlotItem.hideAxis('bottom')
        self.FullViewPlotItem.setMouseEnabled(False, False)

        # Set up the linear region for zooming
        self.lr_zoom = pg.LinearRegionItem(self.FullViewPlotItem.viewRange()[0])
        self.lr_zoom.setZValue(-10)
        self.lr_zoom.sigRegionChanged.connect(self.updatePlot)

        self.setMinimumSize(750, 450)

        self.reset()

    def updateDownsampling(self, range):
        """Updates the downsampling conditions for the main plot."""
        p0, p1 = self.getItem(0,0), self.getItem(1, 0)
        if self._f is not None and len(self._f) > 2**14:
            nf = np.argmax(self._f > range[1]) - np.argmax(self._f > range[0])     # Number of samples within the range
            p0.setDownsampling(ds=nf/2048, auto=False, mode='peak')
            # p1.setDownsampling(ds=nf/2048, auto=False, mode='peak')
        else:
            p0.setDownsampling(ds=1, auto=False)
            p1.setDownsampling(ds=1, auto=False)

    def updatePlot(self):
        p0 = self.getItem(0,0)
        range = self.lr_zoom.getRegion()
        p0.blockSignals(True)
        p0.setXRange(*range, padding=0)
        self.updateDownsampling(range)
        p0.blockSignals(False)

    def onXRangeChanged(self, vb, range):
        self.lr_zoom.setRegion(range)
        self.updateDownsampling(range)

    def plotStemGroup(self, vb, chsh0, chsh_stems, intn_stems):
        """Plots and returns a handle to a group of stem lines for transition peaks."""
        return plot.plot(x=np.repeat(chsh_stems, 2), y=np.dstack((np.zeros(intn_stems.shape[0]), intn_stems)).flatten(), connect='pairs', **kwargs)

    def plot(self, f, yF, xF=None, zF=None, stems=None, freqBlocks=None, phasingPivot=True, indx_colr=None, show_yaxis=True):
        """Plots the data.
        freBlocks is a list of tuples (min, max, bool), where the last position indicates whether the range is active (fitted) or not."""

        self.reset()

        colrseq = [tuple(int(255*c) for c in colr) for colr in config.colrseq]

        p0, p1, pz, p0r = self.getItem(0,0), self.getItem(1, 0), self.FullViewPlotItem, self.p0r
        f = f.ravel()
        self._f = f
        p0.setLimits(xMin=min(f), xMax=max(f))
        p1.setLimits(xMin=min(f), xMax=max(f))
        pz.setLimits(xMin=min(f), xMax=max(f))
        pz.setDownsampling(ds=len(f)/250, auto=False, mode='peak')
        self.lr_zoom.setBounds([f.min(), f.max()])
        self.p0r.setLimits(xMin=min(f), xMax=max(f))

        # Hide the vertical axes
        if not show_yaxis:
            p0.getAxis('left').setStyle(showValues=False)
            p0.getAxis('left').showLabel(False)
            p1.getAxis('left').setStyle(showValues=False)
            p1.getAxis('left').showLabel(False)

        # Plot the experimental spectrum yF
        self._yF = p0.plot(f, yF.ravel().real, pen={'color': colrseq[0], 'width': 2})
        pz.plot(f, yF.ravel().real, pen={'color':'b', 'width':1})

        # Plot the residual
        if xF is not None:
            self._xF = p0.plot(f, xF.ravel().real, pen={'color': colrseq[1], 'width': 1})
            p1.plot(f, np.where(xF.ravel() != 0, (yF-xF).ravel().real, 0), pen={'color':colrseq[5], 'width':1})

        # Plot the components
        if zF is not None:
            if indx_colr is None: indx_colr = list(range(zF.shape[1]))
            self._zF = [None]*zF.shape[1]
            for i in range(zF.shape[1]):
                self._zF[i] = p0.plot(f, zF[:,i].ravel().real, pen={'color':colrseq[indx_colr[i]+2], 'width':1})

        # Plot the ranges
        if freqBlocks is not None:
            self._freqBlocks.clear()
            for i, frqBlk in enumerate(freqBlocks):
                self._add_lr(lims=[frqBlk[0], frqBlk[1]], active=frqBlk[2])

        # Plot the stem lines
        if stems is not None:
            self._stems.clear()
            if indx_colr is None: indx_colr = list(range(len(stems)))
            for i, st in enumerate(stems):
                # Loop over the reported nodes
                scale = np.concatenate([np.array(val[2]) for _, val in st.items()]).max()    # To scale the intensities
                for key, val in st.items():
                    chsh_origin = val[0]
                    chsh_stems=np.array(val[1])
                    intn_stems=np.array(val[2])/scale

                    # Plot groups of stems as PlotCurveItems
                    def setStemDraggingFlag(flag):
                        self._stemDragging_flag = flag

                    self._stems[key] = PlotStemsItem(key, chsh_origin, chsh_stems, intn_stems, colrseq[indx_colr[i]+2])
                    self._stems[key].sigStemsHovered.connect(lambda key, flag : setStemDraggingFlag(flag))
                    self._stems[key].sigStemsClicked.connect(lambda key : self.sigStemsClicked.emit(key))     # Aggregate all clicekd signals fom various stems into a single signal emitted from the MainSpectrumWidget
                    self._stems[key].sigStemsDragged.connect(lambda key, delta : self.sigStemsDragged.emit(key, delta))
                    self.p0r.addItem( self._stems[key] )

        # Show the phasing pivot
        if phasingPivot:
            pos = self._phasingPivot.getXPos()
            if pos > max(f) or pos < min(f):
                pos = (max(f) + min(f))/2
                self._phasingPivot.setPos(pos)
                self.sigPivotDragged.emit(pos)     # Notify that the pivot has changed
            p0.addItem(self._phasingPivot)

        self.updatePlot()

    def replot_yF(self, yF):
        """A slot to be called when the spectrum is being phased. Updates the original spectrum yF."""
        self._yF.setData(x=self._yF.xData, y=yF.ravel().real)

    def _add_lr(self, lims, active=True):
        """Adds a linear region to show a frequency block with certain range."""
        if not np.isinf(lims).all():
            # Define the linear regions
            bounds = [self._f.min(), self._f.max()]
            indx = len(self._freqBlocks)
            lr_p0 = pg.LinearRegionItem(values=lims, bounds=bounds, movable=False, brush=config.colr_freqBlocks if active else config.colr_freqBlocks_inactive)
            lr_p1 = pg.LinearRegionItem(values=lims, bounds=bounds, movable=False, brush=config.colr_freqBlocks if active else config.colr_freqBlocks_inactive)
            lr_p0.setZValue(-100)
            lr_p1.setZValue(-100)
            lr_p0.active = active

            # Add linear regions in each subplot
            self.getItem(0, 0).addItem(lr_p0)    # Add to the main plot
            # self.getItem(1, 0).addItem(lr_p1)    # Add to the residuals plot

            # Connect the frequency ranges in different plots
            def link_lr(master, slave=lr_p1):
                """Links two linearRegionItems to follow the changes in the master."""
                region = master.getRegion()
                slave.setRegion(region)

            lr_p0.sigRegionChanged.connect(link_lr)

            # Define a function to handle the changes in the region
            def update_lr(master):
                # Find the index of this particular region in the list of all regions
                for i, lr_all in enumerate(self._freqBlocks):
                    if lr_all[0] == master:
                        indx = i
                        break

                # Emit the signal that this block has changed
                self.sigFreqRangeChanged.emit(indx, master.getRegion())

            lr_p0.sigRegionChangeFinished.connect(update_lr)

            # Save all linear regions in the internal array
            self._freqBlocks.append([lr_p0, lr_p1])

        else: self._freqBlocks.append([None, None])       # Placeholder for the entire range (will not be displayed)

    def _rem_lr(self, indx):
        """Removes the linear region with certain indx."""
        lr_all = self._freqBlocks.pop(indx)       # Remove from the list of linear regions
        self.getItem(0, 0).removeItem(lr_all[0])         # Remove from the main plot
        self.getItem(1, 0).removeItem(lr_all[1])        # Remove from the residual plot

    def remFreqRange(self, indx=None):
        """Interactive removal of the linear region."""
        if indx is not None:
            self._rem_lr(indx)
        else:
            self.remFreqRange(0)

    def addFreqRange(self, lims, active=True):
        """Adds a frequency range with limits lims to the end of the list."""
        self._add_lr(lims, active)

    def updFreqRange(self, indx, lims=None, active=None):
        """Updates the limits and color of the frequency range indx."""

        if self._freqBlocks[indx][0] is None:
            return

        if lims is not None:
            self._freqBlocks[indx][0].setRegion(lims)      # This will automatically update all three linear ranges

        if active is not None:
            self.fillFreqRange(indx, color = config.colr_freqBlocks if active else config.colr_freqBlocks_inactive)

    def fillFreqRange(self, indx, color):
        """Changes the color for frequency range."""

        if self._freqBlocks[indx][0] is None:
            return

        for lr in self._freqBlocks[indx]:
            lr.setBrush(color)

    def setSelectorFlag(self, flag=True):
        """If the flag is set, MouseDrag event with pressed left button will be interpreted as the frequency region selection (used to add new block)."""
        self.setState(state = 'selectRange' if flag else None)

    def setState(self, state=None):
        """Sets the state of the MainSpectrumWidget. If state is None - return to the default (display) state.
        Possible states include:
         - 'selectRange', in this state, MouseDrag event with pressed left button will be interpreted as the frequency region selection (used to add new block).
         - 'dragSpectrum'
         """
        self._state = state

    def toggleMovableFreqBlocks(self):
        """Enables/disables changes to be made to frequency blocks with mouse events."""
        for blk in self._freqBlocks:
            if blk[0] is not None:
                blk[0].setMovable(not blk[0].movable)

    def setMovableFreqBlocks(self, flag=True):
        """Enables/disables changes to be made to frequency blocks with mouse events."""
        for blk in self._freqBlocks:
            if blk[0] is not None:
                blk[0].setMovable(flag)

    def _onMouseMoved_CrossHair(self, pos):
        p0, p1 = self.getItem(0,0), self.getItem(1, 0)
        # pos = evt[0]  ## using signal proxy turns original arguments into a tuple
        # pos = evt
        if p0.sceneBoundingRect().contains(pos):
            mousePoint = p0.vb.mapSceneToView(pos)
            self._crossLines['v0'].setPos(mousePoint.x())
            self._crossLines['v1'].setPos(mousePoint.x())
            self._crossLines['h0'].setPos(mousePoint.y())
            # index = int(mousePoint.x())
            # # if index > 0:
            # #     label.setText("<span style='font-size: 12pt'>x=%0.1f,   <span style='color: red'>y1=%0.1f</span>,   <span style='color: green'>y2=%0.1f</span>" % (mousePoint.x(), index, index))
        elif p1.sceneBoundingRect().contains(pos):
            mousePoint = p1.vb.mapSceneToView(pos)
            self._crossLines['v0'].setPos(mousePoint.x())
            self._crossLines['v1'].setPos(mousePoint.x())
            self._crossLines['h1'].setPos(mousePoint.y())

    def _onMouseHover(self, obj):
        print('hovered', obj)

    def _onMouseClicked(self, evt):
        """Called when the mouse is clicked in the main graphic scene."""
        if evt.button() == QtCore.Qt.LeftButton:
            p0 = self.getItem(0,0)
            pos = evt.scenePos()
            if p0.sceneBoundingRect().contains(pos):
                mousePoint = p0.vb.mapSceneToView(pos)
                self.sigMouseClicked.emit(mousePoint)

    def _onMouseDragEvent(self, evt, axis=None):
        """To be used instead the mouseDragEvent method in a viewBox. Draws a linear region."""
        global lr_sel           # Selector linear region
        # print('Dragging in vb')

        vb = self.getItem(0,0).vb

        if self._state == 'selectRange' and (evt.button() == QtCore.Qt.LeftButton):   # and (ev.modifiers() & QtCore.Qt.ControlModifier):
            # Adding a new frequency range
            evt.accept()

            # Start of the event
            if evt.isStart():
                # self.setCursor(mode='cross')
                lr_sel = pg.LinearRegionItem()
                vb.addItem(lr_sel)

            # Main movement
            lr_sel.setRegion([vb.mapToView(evt.buttonDownPos()).x(), vb.mapToView(evt.pos()).x()])

            # End of the event
            if evt.isFinish():
                self.sigFreqRangeSelected.emit(lr_sel.getRegion())
                vb.removeItem(lr_sel)
                self.setCursor(mode='normal')
        elif self._stemDragging_flag:
            # Dragging a group of stems. Will be handled by the corresponding PlotStemsItem
            pass
        elif self._state == 'dragSpectrum' and (evt.button() == QtCore.Qt.LeftButton):
            # Moving the experimental spectrum to set a new global chemical shift
            print('Setting global chemical shift')
            evt.accept()
            print(evt)
        else:
            pg.ViewBox.mouseDragEvent(vb, evt, axis)       # Use the standard method

    def setCursor(self, mode='normal'):
        """Sets the cursor to be displayed in the main scene according to the mode."""
        if mode == 'normal' or mode == 'arrow':
            self.getItem(0,0).setCursor(QtCore.Qt.ArrowCursor)
        elif mode == 'cross':
            self.getItem(0,0).setCursor(QtCore.Qt.CrossCursor)
        elif mode == 'hand':
            self.getItem(0,0).setCursor(QtCore.Qt.PointingHandCursor)

    def enableCrosshair(self, enable=True):
        """Enables/disables the crosshair mode."""
        if enable:
            # self._proxy_mouseMoved = pg.SignalProxy(p0.scene().sigMouseMoved, rateLimit=60, slot=self._onMouseMoved_CrossHair)
            self.getItem(0,0).scene().sigMouseMoved.connect(self._onMouseMoved_CrossHair)
            for key, val in self._crossLines.items():
                val.show()
        else:
            self.getItem(0,0).scene().sigMouseMoved.disconnect()
            for key, val in self._crossLines.items():
                val.hide()

    def onPivotDragged(self, pivot):
        """Called when the pivot is being dragged."""
        self.sigPivotDragged.emit(pivot.p[0])      # Returns the position along the x axis

    def showResidual(self, flag=True):
        if flag:
            p0, p1 = self.getItem(0,0), self.getItem(1, 0)
            p0.getAxis('bottom').setStyle(showValues=False)  #this will remove the tick labels and reduces gap b/w plots almost to zero. There will be a double line separating the plot rows
            p0.setLabel('bottom', '')
            p1.show()
        else:
            # Hide
            p0, p1 = self.getItem(0,0), self.getItem(1, 0)
            p0.getAxis('bottom').setStyle(showValues=True)
            p0.setLabel('bottom', 'Chemical shift, ppm')
            p1.hide()

    def showComponents(self, flag=True):
        if len(self._zF) == 0:
            raise Exception('No components exist.')
        for x in self._zF:
            if flag:
                x.show()
            else: x.hide()

    def showStems(self, flag=True):
        if len(self._stems) == 0:
            raise Exception('No components exist.')
        for x in self._stems:
            if flag:
                x.show()
            else:
                x.hide()

    def highlightStems(self, key, flag=True):
        """Highlights/dehighlits stem with a certain key."""
        self._stems[key].highlight(flag)

    def autoRange(self, xlims=None, ylims=None, margin=0.04):
        """Adjust the view to the data (frequency blocks); xlims and ylims overrides the limits imposed by the frequency blocks. margin is the amount of extra space on each side of the plotted region, expressed as a ratio to the region width."""
        self.getItem(0, 0).vb.autoRange()     # Autoscale both x and y

        xmin, xmax, ymin, ymax = np.inf, -np.inf, np.inf, -np.inf
        set_flag = False
        if hasattr(xlims, '__iter__'):
            xmin, xmax = min(xlims), max(xlims)
            if hasattr(ylims, '__iter__'):
                ymin, ymax = min(ylims), max(ylims)
            else:
                yreg = self._yF.yData[np.logical_and(self._f<xmax, self._f>xmin)]
                ymin = min(ymin, min(yreg))
                ymax = max(ymax, max(yreg))
            set_flag = True

        for frqBlk in self._freqBlocks:
            # Check if there is a freq block and if it is active
            if frqBlk[0] is not None and frqBlk[0].active:
                    xreg = frqBlk[0].getRegion()
                    xmin = min(xmin, min(xreg))
                    xmax = max(xmax, max(xreg))

                    yreg = self._yF.yData[np.logical_and(self._f<max(xreg), self._f>min(xreg))]
                    ymin = min(ymin, min(yreg))
                    ymax = max(ymax, max(yreg))
                    set_flag = True

        # self.getItem(0, 0).vb.setRange(xRange=(xmin-(xmax-xmin)*0.04, xmax+(xmax-xmin)*0.04) if set_flag else None)
        if set_flag:
            self.getItem(0, 0).vb.setXRange(xmin-(xmax-xmin)*margin, xmax+(xmax-xmin)*margin)
            self.getItem(0, 0).vb.setYRange(ymin-(ymax-ymin)*(margin/2), ymax+(ymax-ymin)*margin, padding=0)

    def saveImage(self):
        """Saves the spectrum as an image file."""
        # filetypes = self.canvas.get_supported_filetypes_grouped()
        # sorted_filetypes = list(six.iteritems(filetypes))
        # sorted_filetypes.sort()
        # default_filetype = self.canvas.get_default_filetype()

        # startpath = rcParams.get('savefig.directory', '')
        # startpath = path.expanduser(startpath)
        # start = path.join(startpath, self.canvas.get_default_filename())
        # start=None
        # filters = []
        # for name, exts in sorted_filetypes:
        #     exts_list = " ".join(['*.%s' % ext for ext in exts])
        #     filter = '%s (%s)' % (name, exts_list)
        #     if default_filetype in exts:
        #         selectedFilter = filter
        #     filters.append(filter)
        # filters = ';;'.join(filters)
        #
        # fname = QFileDialog.getSaveFileName(parent=self.parent,
        #                                  caption="Choose a filename to save to", directory=start, filter=filters)
        # if fname:
        #     if startpath == '':
        #         # explicitly missing key or empty str signals to use cwd
        #         rcParams['savefig.directory'] = startpath
        #     else:
        #         # save dir for next time
        #         savefig_dir = path.dirname(six.text_type(fname))
        #         rcParams['savefig.directory'] = savefig_dir
        #     try:
        #         #self.canvas.savefig(six.text_type(fname), format='eps', dpi=1200)
        #         fontSizeBefore = rcParams['font.size']
        #         rcParams['font.size'] = 18
        #         self.canvas.print_figure(six.text_type(fname))
        #         rcParams['font.size'] = fontSizeBefore
        #         self.canvas.draw()
        #     except Exception as e:
        #         QMessageBox.critical(
        #             self, "Error saving file", str(e),
        #             QMessageBox.Ok, QMessageBox.NoButton)

        print('Coming soon... Please use the Export option in the context menu (right click on the plot -> Export).')

    def reset(self):
        self.getItem(0,0).vb.clear()
        self.getItem(1, 0).vb.clear()
        self.FullViewPlotItem.vb.clear()
        self.p0r.clear()
        self.FullViewPlotItem.addItem(self.lr_zoom)

        # Reset the signals
        self._f = None
        self._xF = None
        self._yF = None
        self._zF.clear()
        self._stems.clear()
        self._freqBlocks.clear()

        self.setState()

class QCheckableComboBox(QComboBox):
    """Checkable ComboBox"""

    selectionChanged = pyqtSignal()

    def __init__(self, items = [], checkedItems = None, parent=None):
        super().__init__(parent)
        self.view().pressed.connect(self.onItemPressed)
        self._changed = False
        self.setupItems(items, checkedItems)
        self.view().setMinimumWidth(100)

    def setupItems(self, items = [], checkedItems = None):
        """Populates the combobox with items and sets their state"""
        self.clear()
        for indx, item in enumerate(items):
            self.addItem(item)
            if checkedItems is not None:
                if indx in checkedItems:
                    self.setItemChecked(indx, True)
            else:
                self.setItemChecked(indx, False)

    def addItem(self, item, checked = False):
        super().addItem(item)
        newItem = self.model().item(self.model().rowCount()-1, self.modelColumn())
        if checked:
            newItem.setCheckState(Qt.Checked)
        else:
            newItem.setCheckState(Qt.Unchecked)

    def onItemPressed(self, index):
        item = self.model().itemFromIndex(index)
        if item.checkState() == Qt.Checked:
            item.setCheckState(Qt.Unchecked)
        else:
            item.setCheckState(Qt.Checked)
        self._changed = True
        self.selectionChanged.emit() # emit a signal to save the change in the treeWidget

    def hidePopup(self):
        if not self._changed:
            super().hidePopup()
        self._changed = False

    def itemChecked(self, index):
        item = self.model().item(index, self.modelColumn())
        return item.checkState() == Qt.Checked

    def checkedItems(self):
        """Returns a list of checked items' indices or [] if all items are unchecked."""
        return [i for i in range(self.model().rowCount()) if self.itemChecked(i)]

    def setItemChecked(self, index, checked=True):
        item = self.model().item(index, self.modelColumn())
        if checked:
            item.setCheckState(Qt.Checked)
        else:
            item.setCheckState(Qt.Unchecked)

class FreqTableModel(QtCore.QAbstractTableModel):
    """A treeView class for the main navigation view."""

    freqBlockChanged = pyqtSignal(int)    # Returns the index of the changed freqBlock

    def __init__(self, datum, parent = None):
        super().__init__()     # QtCore.QAbstractItemModel.__init__(self)
        self.datum = datum       # A pointer to the workspace

    def setNewDatum(self, datum):
        self.beginResetModel()
        self.datum = datum
        self.endResetModel()

    def headerData(self, section, orientation, role):
        head = ["", "From", "To", "Bsln",  "Re", "Im"]

        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return head[section]

    def columnCount(self, parent = QtCore.QModelIndex()):
        return 4  #5

    def rowCount(self, parent = QtCore.QModelIndex()):
        """Number of rows (children) for each item in the tree. INPUTS: QModelIndex. OUTPUT: int"""
        try:
            return len(self.datum.freqBlocks) + 1
        except AttributeError:
            return 1

    def flags(self, index):
        row = index.row()
        clmn = index.column()

        if row == 0:
            return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsUserCheckable

        if clmn == 0:
            return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsUserCheckable
        else:
            return QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable

    def data(self, index, role):
        row = index.row()
        clmn = index.column()

        if not isinstance(self.datum, Workspace):

            if role == QtCore.Qt.CheckStateRole and clmn == 0:
                if row == 0:
                    if len(self.datum.steps[-1].frqBlkIds) > 0:
                        return QtCore.Qt.Unchecked
                    else: return QtCore.Qt.Checked
                elif row > 0 and (row-1) in self.datum.steps[-1].frqBlkIds:
                    return QtCore.Qt.Checked
                else:
                    return QtCore.Qt.Unchecked

            if role in [QtCore.Qt.DisplayRole, QtCore.Qt.EditRole]:
                if clmn == 0:
                    if row == 0:
                        return "Fit in time domain"
                    elif row == 1:
                        return "Entire frequency range"
                    else:
                        return chr(64 + row-1)
                elif clmn == 1 and row > 1:
                    return "{:.2f}".format(self.datum.freqBlocks[row-1].min)
                elif clmn == 2 and row > 1:
                    return "{:.2f}".format(self.datum.freqBlocks[row-1].max)
                elif clmn == 3 and row > 0:
                    return self.datum.freqBlocks[row-1].bslnOrder[0]
                elif clmn == 4 and row > 0:
                    return self.datum.freqBlocks[row-1].bslnOrder[1]

        else: return None       # If the current datum is the entire Workspace - do not return anything

    # --------------------------------- M A I N   D I S P L A Y   F U N C T I O N ----------------------------------

    def setData(self, index, value, role = QtCore.Qt.EditRole):

        if not index.isValid():
            return False

        node = index.internalPointer()
        clmn = index.column()
        row = index.row()

        if role == QtCore.Qt.EditRole and row > 0:
            if clmn in [1, 2]:
                xmin = float(value) if clmn == 1 else float(self.data(self.index(row, 1), QtCore.Qt.DisplayRole))
                xmax = float(value) if clmn == 2 else float(self.data(self.index(row, 2), QtCore.Qt.DisplayRole))
                self.datum.altFreqBlock(lims=(xmin, xmax), indx=row-1)
                self.freqBlockChanged.emit(row-1)

            elif clmn == 3: # in [3, 4]:
                Br = int(value) if clmn == 3 else int(self.data(self.index(row, 3), QtCore.Qt.DisplayRole))
                #Bi = int(value) if clmn == 4 else int(self.data(self.index(row, 4), QtCore.Qt.DisplayRole))
                #self.datum.altFreqBlock(bslnOrder=(Br, Bi), indx=row-1)    # Use this line to set different values for Br and Bi and set columnCount = 5
                self.datum.altFreqBlock(bslnOrder=(Br, Br), indx=row-1)

            self.dataChanged.emit(index, index)
            return True

        if role == QtCore.Qt.CheckStateRole:
            if row == 0:
                if len(self.datum.steps[-1].frqBlkIds) > 0:
                    for step in self.datum.steps:
                        step.frqBlkIds.clear()
                self.dataChanged.emit(self.index(0,0), self.index(self.rowCount(), 0))
                return True

            elif row > 0:
                try:
                    for step in self.datum.steps:
                        step.frqBlkIds.remove(row-1)
                except KeyError:
                    for step in self.datum.steps:
                        step.frqBlkIds.add(row-1)
                self.dataChanged.emit(self.index(0,0), self.index(self.rowCount(), 0))
                self.freqBlockChanged.emit(row-1)
                return True

        return False

    def notifyDataChanged(self, indx=0):
        """Notifies the view that the row corresponding to the indx freqBlock has changed."""
        self.dataChanged.emit( self.index(indx+1,0), self.index(indx+1,self.columnCount()) )       # (self.index(0,0), self.index(0,0))

    #=====================================================#
    #INSERTING & REMOVING
    #=====================================================#

    def addFreqBlock(self, xmin, xmax, parent = QtCore.QModelIndex()):
        """Adds a new frequency block to the series."""
        self.beginInsertRows(parent, self.rowCount(), self.rowCount())        # Parent node, first and last position
        self.datum.addFreqBlock((xmin, xmax))
        self.endInsertRows()

    def remFreqBlock(self, indx, parent = QtCore.QModelIndex()):
        """Removes a FreqBlock indx from the Series"""
        self.beginRemoveRows(parent, indx+1, indx+1) # Parent node, first and last position
        self.datum.remFreqBlock(indx)
        self.endRemoveRows()

class FreqTableView(QTableView):
    """Model/View based class to display frequency ranges for optimization."""

    def __init__(self, parent=None):
        super().__init__(parent)    # Initialize a QTreeWidget

        self.setAlternatingRowColors(False)

    def setModel(self, model):
        super().setModel(model)

        self.setColumnWidth(0, 40)
        self.setColumnWidth(1, 42)
        self.setColumnWidth(2, 42)
        self.setColumnWidth(3, 30)
        self.setColumnWidth(4, 30)
        self.verticalHeader().setDefaultSectionSize(20)
        self.verticalHeader().hide()
        self.horizontalHeader().setStretchLastSection(True)
        self.setSpan(0, 0, 1, 5)
        self.setSpan(1, 0, 1, 3)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

class NavigationTreeModel(QtCore.QAbstractItemModel):
    """A treeView class for the main navigation view."""

    def __init__(self, wsp, parent = None):
        super().__init__()     # QtCore.QAbstractItemModel.__init__(self)
        self.wsp = wsp       # A pointer to the workspace
        self._copySettingsFromID = None            # Id of the series from which to copy parameter settings

    def resetWorkspace(self):
        self.beginResetModel()
        self.wsp.reset()
        self.endResetModel()

    def headerData(self, section, orientation, role):
        pass

    def columnCount(self, parent):
        return 1

    def rowCount(self, index):
        """Number of rows (children) for each item in the tree. INPUTS: QModelIndex. OUTPUT: int"""
        if index.isValid():
            node = index.internalPointer()
            if isinstance(node, Series) and len(node.data)!=1:
                return len(node.data)
            else: return 0    # If the node is a Datum node
        else: return len(self.wsp.series)    # The node is the Workspace (root) node

    def parent(self, index):
        """Should return QModelIndex of the parent of the node with the given QModelIndex. INPUTS: QModelIndex. OUTPUT: QModelIndex"""
        if index.isValid():
            node = index.internalPointer()
            if isinstance(node, Datum):
                ser = node.parent
                if len(ser.data) > 1:
                    return self.createIndex(self.wsp.series.index(ser), 0, ser)
        return QtCore.QModelIndex()   # The node is a Series, return a pointer to the root

    def index(self, row, column, prnt=None):
        """Should return a QModelIndex that corresponds to the given row, clmn and parent node. INPUTS: int, int, QModelIndex. OUTPUT: QModelIndex"""
        if not prnt or not prnt.isValid():
            parent = self.wsp
        else:
            parent = prnt.internalPointer()

        if isinstance(parent, Workspace):
            if len(parent.series[row].data) != 1:
                child = parent.series[row]
            else:
                child = parent.series[row].data[0]
        elif isinstance(parent, Series):
            child = parent.data[row]
        else:
            child = None

        if child:
            return self.createIndex(row, column, child)
        else:
            return QtCore.QModelIndex()

    # --------------------------------- M A I N   D I S P L A Y   F U N C T I O N ----------------------------------
    def data(self, index, role):
        if not index.isValid():
            return None

        node = index.internalPointer()
        clmn = index.column()
        row = index.row()

        #print("Data", row, clmn, node, node.name)

        if role in [QtCore.Qt.DisplayRole, QtCore.Qt.EditRole]:
            if isinstance(node, Workspace):
                return "Workspace"
            else:
                result = node.name
                if isinstance(node, Series) and len(node.data) == 0: result += ' (empty)'
                return result

        elif role == QtCore.Qt.DecorationRole:
            if isinstance(node, Datum):
                displayIcon = QIcon('icons\icon_gof_none.png')
                # gof = node.goodness_of_fit()
                # if gof is None:
                #     displayIcon = QIcon('icons\icon_gof_none.png')
                # elif gof > 0.9:
                #     displayIcon = QIcon("icons\icon_gof_good.png")
                # elif gof < 0.4:
                #     displayIcon = QIcon('icons\icon_gof_bad.png')
                # else:
                #     displayIcon = QIcon('icons\icon_gof_okay.png')
                return displayIcon

        return None

    def flags(self, index):
        if not index.isValid():
            return None
        return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable

    def setData(self, index, value, role = QtCore.Qt.EditRole):
        """Stores changed name of the node."""
        if not index.isValid():
            return False

        node = index.internalPointer()

        if role in [QtCore.Qt.DisplayRole, QtCore.Qt.EditRole]:
            if not isinstance(node, Workspace) and value != '':
                node.name = value
                return True

        return False

    #=====================================================#
    #INSERTING & REMOVING
    #=====================================================#

    def addSeries(self):
        """Adds a new series to the workspace."""
        # TODO: Additona and removal of series influence the currently copied ID
        self._copySettingsFromID = None

        self.beginInsertRows(QtCore.QModelIndex(), len(self.wsp.series), len(self.wsp.series))        # Parent node, first and last position

        self.wsp.addSeries()

        self.endInsertRows()

    def addDatumFromFile(self, crnt_series, path):
        """Adds new Datum entries specified by the path to the series object."""

        if path[-6:] == '.pyfid':
            with open(path, 'rb') as fp:
                data = [float(x.strip()) if i != 5 else x.strip() for i, x in enumerate(fp.readlines())]

            c0, f0, nt = data[0], data[1], int(data[4])     # Number of time points

            t = np.array(data[6:nt+6]).reshape(-1,1)
            yT = (np.array(data[nt+6:2*nt+6]) + 1j*np.array(data[-nt:])).reshape(-1,1)
            name = path[path.rfind('\\')+1:path.rfind('.')]

        # Read a JCAMP-DX file
        elif path[-3:] == '.dx' or path[-4:] == '.jdx':

            dic, data = ng.jcampdx.read(path)
            #c0 = float(dic['$BF1'][0])
            #fcar = float(dic['$REFERENCEPOINT'][0])
            #swh = float(dic['$SW'][0]) * c0
            #nt = float(dic['$TD'][0])

            udic = ng.jcampdx.guess_udic(dic,data)[0]     # Dictionary of universal parameters
            print(data)

            c0 = float(udic['obs'])
            fcar = float(udic['car'])
            swh = float(udic['sw'])
            nt = float(udic['size'])
            f0 = swh/2-fcar      # Frequency shift in Hz
            dt = 1 / swh;         # Sampling period (dwell time)

            t = np.linspace(start=0, stop=(nt-1)*dt, num=nt).reshape(-1,1)
            yT = (np.array(data[0]) - 1j*np.array(data[1])).reshape(-1,1)

            # Subsample if the frequency range is too large
            k = max(math.floor(swh/c0 / 12), 1)   # Sampling factor to make the sweep width 12 ppm
            t = t[::k]
            yT = yT[::k, :]

            name = os.path.split(os.path.dirname(path))[1]

        # Read a Bruker FID file
        elif path[-3:] == 'fid':

            dic, data = ng.fileio.bruker.read(path[:-3])

            acqus = dic['acqus']
            ntgrp = int(round(acqus['GRPDLY']))    # Number of time samples of the Bruker filter response;
            swh = acqus['SW_h']     # Spectral width in Hz
            f0 = acqus['O1']        # Offset in Hz
            c0 = acqus['SFO1']      # Frequency of the local oscillator in MHz
            dt = 1 / swh         # Sampling period (dwell time)
            tau = acqus['DE'] * (1e-06)   # Ringdown time delay in sec

            yT = data[ntgrp:].reshape(-1, 1)
            # nt = min(16384, len(yT))
            nt = len(yT)
            t = np.linspace(start=0, stop=(nt-1)*dt, num=nt).reshape(-1,1)
            # yT = yT[:nt].reshape(-1, 1)

            ## Subsample if the frequency range is too large
            #k = max(math.floor(swh/c0 / 12), 1)   # Sampling factor to make the sweep width 12 ppm
            #t = t[::k]
            #yT = yT[::k, :]

            name = os.path.split(os.path.dirname(path))[1]

        # Read a JEOL FID file
        elif path[-3:] == 'jdf':
            print(path)
            #
            # dic, data = ng.fileio.bruker.read(path[:-3])
            #
            # acqus = dic['acqus']
            # ntgrp = acqus['GRPDLY']    # Number of time samples of the Bruker filter response;
            # swh = acqus['SW_h']     # Spectral width in Hz
            # f0 = acqus['O1']        # Offset in Hz
            # c0 = acqus['SFO1']      # Frequency of the local oscillator in MHz
            # dt = 1 / swh         # Sampling period (dwell time)
            # tau = acqus['DE'] * (1e-06)   # Ringdown time delay in sec
            #
            # yT = data[ntgrp:].reshape(-1, 1)
            # # nt = min(16384, len(yT))
            # nt = len(yT)
            # t = np.linspace(start=0, stop=(nt-1)*dt, num=nt).reshape(-1,1)
            # # yT = yT[:nt].reshape(-1, 1)
            #
            # ## Subsample if the frequency range is too large
            # #k = max(math.floor(swh/c0 / 12), 1)   # Sampling factor to make the sweep width 12 ppm
            # #t = t[::k]
            # #yT = yT[::k, :]
            #
            # name = os.path.split(os.path.dirname(path))[1]
            pass

        # Read a Spinsolve data.1d file
        elif path[-3:] in ['.1d', '.2d']:

            yT, c0, f0, dt, dic = read_spinsolve(path)

            nt = yT.shape[0]
            t = np.linspace(start=0, stop=(nt-1)*dt, num=nt).reshape(-1,1)
            name = os.path.split(os.path.dirname(path))[1]    # Only the name of the containing directory

        # Read an Mnova corrected FID file
        elif path[-4:] in ['.txt']:
            with open(path, 'rb') as fp:
                # Read the file header line by line
                for line in fp:
                    pair = line.decode().strip().split('=')
                    if 'DataPoints' in pair[0]:
                        break           # The next line will be the first data point -- stop reading the header
                    elif 'Size' in pair[0]:
                        nt = int(pair[1])
                    elif 'SpectrometerFrequency' in pair[0]:
                        c0 = float(pair[1])
                    elif 'Hz' in pair[0]:
                        f0 = -float(pair[1])
                    elif 'SpectralWidth' in pair[0]:
                        dt = 1 / float(pair[1])

                # Read the remainder of the file into a np array
                data = np.fromfile(fp, sep='\t')

            # Form the arrays
            t = np.linspace(0, dt*(nt-1), nt).reshape(-1,1)
            yT = (data[::2] - 1j*data[1::2]).reshape(-1,1)

            ## Subsample if the frequency range is too large
            #k = max(math.floor(swh/c0 / 12), 1)   # Sampling factor to make the sweep width 12 ppm
            #t = t[::k]
            #yT = yT[::k, :]

            name = path[path.rfind('\\')+1:path.rfind('.')]

        # Save the acquisition parameters; these should be the same for all spectra in the series (by convention)
        if crnt_series.c0 is None:
            crnt_series.c0 = c0
            crnt_series.f0 = f0
            crnt_series.t = t
            crnt_series.fullReset()
        elif np.abs(crnt_series.c0 - c0) > 1e-3:
            # Create a new series and put the data into it
            # TODO: Check if new c0/f0 are the same as the old ones when loading the rest of the data
            self.addSeries()
            crnt_series = self.wsp.series[-1]
            crnt_series.c0 = c0
            crnt_series.f0 = f0
            crnt_series.t = t
            crnt_series.fullReset()

        ser_id = self.wsp.series.index(crnt_series) # Position of the Series in the Workspace
        index = self.index(ser_id, 0, None)    # Index corresponding to the Series to which the Datum will be added

        self.layoutAboutToBeChanged.emit()

        if len(crnt_series.data) == 0 and yT.shape[1] == 1:   # Adding only a single first Datum; no rows will be added, but need to replace the existing Series row with this new Datum
            dat = crnt_series.addDatum(yT, name = name)
        else:
            # Start adding rows
            if len(crnt_series.data) == 1:   # Already one node in the series
                self.beginInsertRows(index, 0, yT.shape[1])        # Parent node, first and last position
            else:
                self.beginInsertRows(index, len(crnt_series.data), len(crnt_series.data)+yT.shape[1]-1)        # Parent node, first and last position
            # Add the rows
            for i in range(yT.shape[1]):
                dat = crnt_series.addDatum(yT[:,i].reshape(-1,1), name = name+str(i+1) if yT.shape[1]>1 else name)
            # Finish adding rows
            self.endInsertRows()

        self.layoutChanged.emit()    # Tell the view that we need to recompute persistent indices

        return dat

    def importData(self, crnt_series=None):
        """Opens a dialog to select a new data file to be added to the parent series."""
        # Create new series if working from the workspace itself
        dat = self.wsp

        if crnt_series is None or crnt_series == self.wsp:
            self.addSeries()
            crnt_series = self.wsp.series[-1]
        elif isinstance(crnt_series, Datum):
            crnt_series = crnt_series.parent    # Go one level up to the Series level

        for newFilePath in QFileDialog.getOpenFileNames(None, 'Import file', '.', filter = "All supported files (*.pyfid; *.dx; *.jdx; *.1d; *.2d; *.txt; fid);;Converted FID (*.pyfid);;Spinsolve binary (*.1d; *.2d);;JCAMP (*.dx; *.jdx);;Mnova FID (*.txt);;Bruker FID (fid)"):   # ;;JEOL FID (*.jdf)
            #try:
            dat = self.addDatumFromFile(crnt_series, newFilePath)
            #except:
            #    print("Could not add the file ", newFilePath)

        return dat

    def indexByKey(self, key=(None, None)):
        """Retuens the index of an item (Datum or Series) given its selfID."""
        if key[1] is not None:
            # A Datum
            row = key[1]
            item = self.wsp.series[key[0]].data[row]
        elif key[0] is not None:
            # A Series
            row = key[0]
            item = self.wsp.series[row]
        else:
            row = 0
            item = self.wsp

        column = 0
        return self.createIndex(row, column, item)

    def remItems(self, items):
        """Removes a Series or a Datum"""
        self._copySettingsFromID = None

        for item in items:
            sID = item.selfID()
            if sID[0] is None:
                return None

            # Datum
            if sID[1] is not None:
                ser = self.wsp.series[sID[0]]    # or ser = item.parent
                parent = self.index(sID[0], 0, None)      # Model index of the Series
                index = self.index(sID[1], 0, parent)     # Model index of the Datum
                if len(ser.data) == 1:
                    # If it is a single datum in the Series - remove entire series
                    item = ser
                    sID = item.selfID()      # Now sID[1] is None
                elif len(ser.data) == 2:
                    # If we remove this Datum, there will be only one left in this Series
                    self.layoutAboutToBeChanged.emit()

                    self.beginRemoveRows(self.parent(index), 0, 1) # Remove both rows
                    item.remove()    # Remove itself from the workspace
                    self.endRemoveRows()

                    self.layoutChanged.emit()
                else:
                    self.beginRemoveRows(self.parent(index), index.row(), index.row()) # Parent node, first and last position
                    item.remove()    # Remove itself from the workspace
                    self.endRemoveRows()

            # Series
            if sID[1] is None:
                # It it's a series that needs to be removed
                index = self.index(sID[0], 0, None)    # Index in the model corresponding to the current item (Datum or Series)
                self.beginRemoveRows(self.parent(index), index.row(), index.row()) # Parent node, first and last position
                item.remove()    # Remove itself from the workspace
                self.endRemoveRows()

    def copySettings(self, item):
        """Stores a reference to a Series from which the Settings will be copied."""
        self._copySettingsFrom = item

    def pasteSettings(self, items):
        """Pastes series settings to a selected Series."""
        if self._copySettingsFrom is not None:
            for item in items:
                item.copySettings(self._copySettingsFrom)

class NavigationTreeView(QTreeView):
    """Model/View based class to display loaded datasets."""

    requestPasteCrnt = pyqtSignal(list)
    requestPasteDflt = pyqtSignal(list)
    requestFitSelected = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)    # Initialize a QTreeWidget

        self.setAlternatingRowColors(True)
        self.setHeaderHidden(True)
        self.setSelectionMode(QTreeView.ExtendedSelection)
        self.setIndentation(10)

        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.onCustomContextMenuRequested)

    def onCustomContextMenuRequested(self, pos):
        """Handler of the custom context menu requested signal."""
        popMenu = QMenu(self)
        index = self.indexAt(pos)
        selected = [selind.internalPointer() for selind in self.selectedIndexes() if selind.isValid()]     # List of selected Datums or Series

        actnAddSeries = QAction(QIcon('icons\icon_newSeries.png'), 'Add new series', self)
        actnAddSeries.setStatusTip('Add new series')
        actnAddSeries.triggered.connect(self.model().addSeries)
        actnImportData = QAction(QIcon('icons\icon_addFile.png'), 'Import files', self)
        actnImportData.setStatusTip('Import new data and add them to the current series')
        actnImportData.triggered.connect(lambda : self.onImportData(crnt_series=index.internalPointer() if index.isValid() else None))
        actnPasteCrnt = QAction(QIcon('icons\icon_pasteCrnt.png'), 'Paste as current', self)
        actnPasteCrnt.setStatusTip('Paste as current values')
        actnPasteCrnt.triggered.connect(lambda : self.requestPasteCrnt.emit(selected))     # Emit a list of selected datums to paste the currently copied parameters to them
        actnPasteDflt = QAction(QIcon('icons\icon_pasteDflt.png'), 'Paste as defualt', self)
        actnPasteDflt.setStatusTip('Paste as default values')
        actnPasteDflt.triggered.connect(lambda : self.requestPasteDflt.emit(selected))     # Emit a list of selected datums to paste the currently copied parameters to them
        actnfitSelected = QAction(QIcon('icons\icon_fitSelected.png'), 'Fit selected', self)
        actnfitSelected.setStatusTip('Fit selected datasets')
        actnfitSelected.triggered.connect(lambda : self.requestFitSelected.emit(selected))     # Emit a list of selected datums to fit

        popMenu.addAction(actnfitSelected)
        popMenu.addSeparator()
        popMenu.addAction(actnAddSeries)
        popMenu.addAction(actnImportData)

        if index.isValid():         # If the click was on an item
            actnRemoveData = QAction(QIcon('icons\icon_removeFile.png'), 'Remove files', self)
            actnRemoveData.setStatusTip('Remove file from the workspace')
            actnRemoveData.triggered.connect(lambda : self.model().remItems(items=[selind.internalPointer() for selind in self.selectedIndexes() if selind.isValid()]))
            popMenu.addAction(actnRemoveData)

            # actnCopySettings = QAction(QIcon('icons\icon_blank.png'), 'Copy settings', self)
            # actnCopySettings.setStatusTip('Copies the series settings (frequency ranges, steps, and parameter priors)')
            # actnCopySettings.triggered.connect(lambda : self.model().copySettings(index.internalPointer()))
            # popMenu.addAction(actnCopySettings)
            # actnPasteSettings = QAction(QIcon('icons\icon_blank.png'), 'Apply settings', self)
            # actnPasteSettings.setStatusTip('Applies the copied settings to the current Series')
            # actnPasteSettings.triggered.connect(lambda : self.model().pasteSettings(items=[selind.internalPointer() for selind in self.selectedIndexes() if selind.isValid()]))
            # popMenu.addAction(actnPasteSettings)

        popMenu.addSeparator()
        popMenu.addAction(actnPasteCrnt)
        popMenu.addAction(actnPasteDflt)

        # Show the menu
        popMenu.popup(self.viewport().mapToGlobal(pos))

    def onImportData(self, crnt_series):
        dat = self.model().importData(crnt_series)
        self.selectCurrentDatum(dat.selfID())

    def selectCurrentDatum(self, key=(None, None)):
        """Highlights the current Datum or Series."""
        # print('Selecting ', key)
        index = self.model().indexByKey(key)
        self.selectionModel().setCurrentIndex(index, QItemSelectionModel.SelectCurrent | QItemSelectionModel.Rows)

class NavigationTreeDelegate(QItemDelegate):

    def __init__(self, parent=None, *args):
        super().__init__(parent, *args)

    def paint(self, painter, option, index):
        painter.save()

        # set background color
        painter.setPen(QPen(Qt.NoPen))
        if option.state & QStyle.State_Selected:
            painter.setBrush(QBrush(Qt.red))
        else:
            painter.setBrush(QBrush(Qt.white))
        painter.drawRect(option.rect)

        # set text color
        painter.setPen(QPen(Qt.black))
        value = index.data(Qt.DisplayRole)
        if value:
            text = value
            painter.drawText(option.rect, Qt.AlignLeft, text)

        painter.restore()

def getDisplayTree(T, myOrder = ['ampl', 'chsh', 'alph', 'jcpl']):
    """Returns the tree of parameters P for a chemNode tree T. The variable myOrder defines the order in which the parameters will be sorted. Each node in the parameter tree corresponds to a chemical/group of chemicals or its parameters."""
    P = viewNode(T.name, nodeType='chemNodeDB' if type(T) in [chemNodeDB, chemNodeQM] else 'chemNode')
    #P.addChild(viewNode(name = tuple([T.name] + ['intn'] + [None]), nodeType='intn', alias='intn' ))
    if type(T) is chemNodeQM:
        P.nodeType = 'chemNodeQM'
        P.addChild(viewNode(name = tuple([T.name] + ['ampl'] + [0]), nodeType='param', alias='intn' ))

        if T.childCount() > 1:  # Several chemical shifts; add global parameters
            P.addChild(viewNode(name = tuple([T.name] + ['chsh'] + [0]), nodeType='param' ))
            P.addChild(viewNode(name = tuple([T.name] + ['alph'] + [0]), nodeType='param' ))

        # Add nodes that can become parents for parameters (combinations of spin systems if any and the QM node itself)
        prntNodes = [None] * len(T.spsyComb) + [P]
        parsLists = [[] for _ in range(len(T.spsyComb)+1)]
        for i_comb, comb in enumerate(T.spsyComb):
            prntNodes[i_comb] = viewNode(name=(T.name+'-COMB'+str(i_comb+1)), nodeType='chemNode', alias=comb.name)
            prntNodes[i_comb].addChild(viewNode(name=(T.name+'-COMB'+str(i_comb+1), 'ampl', 0), nodeType='param', alias='intn' ))
            P.addChild(prntNodes[i_comb])

        # Loop over spin systems and collect all parameters that affect it
        for i_spsy in range(len(T.spinTopo)):
            i_comb = -1      # Index to which combination belongs this spin system (the first occurence). Default - the chemNodeQM itself
            for j, comb in enumerate(T.spsyComb):
                if i_spsy in comb.indxSpsy: i_comb = j

            # Add new parameters to the list for each combination
            parsLists[i_comb].extend( [(T.name, 'chshQD', i_parm, T.chshQD[i_parm].label) for i_parm in T._indxChsh_by_spsy[i_spsy]] )
            parsLists[i_comb].extend( [(T.name, 'alphQD', i_parm, T.alphQD[i_parm].label) for i_parm in T._indxChsh_by_spsy[i_spsy]] )
            parsLists[i_comb].extend( [(T.name, 'jcplQD', i_parm, T.jcplQD[i_parm].label) for i_parm in T._indxJcpl_by_spsy[i_spsy]] )

        # Add intensity/amplitude parameters and add the nodes to the tree
        for prnt, parsList_for_prnt in zip(prntNodes, parsLists):
            parsList_for_prnt.extend( [(chld.name, 'ampl', 0, chld.alias) for chld in T[prnt.name].children() if isinstance(chld, chemNodeQT)] )
            for pars in sorted(parsList_for_prnt, key = lambda par : [i for i, x in enumerate(myOrder) if x in par[1]][-1] ):
                try:
                    prnt.addChild(viewNode(name = pars[0:3], alias = pars[3], nodeType='param'))
                except RuntimeError: pass

    elif type(T) is chemNodeQD:    # Spin system defined by itself without a parent chemDB node
        P.addChild(viewNode(name = tuple([T.name] + ['ampl'] + [0]), nodeType='param', alias='intn' ))

        if T.childCount() > 1:  # Several chemical shifts; add global parameters
            P.addChild(viewNode(name = tuple([T.name] + ['chsh'] + [0]), nodeType='param' ))
            P.addChild(viewNode(name = tuple([T.name] + ['alph'] + [0]), nodeType='param' ))

        for par, val in T.default_pars().items():
            for i in range(len(val)):
                if not isinstance(getattr(T, par)[i].label, str):
                    old = getattr(T, par)
                    old[i] = parsSpec(old[i].min, old[i].max, label='', distr=old[i].distr, p1=old[i].p1, p2=old[i].p2)
                    setattr(T, par, old)

        newParsNodes = [tuple([node.name] + [par] + [i] + [getattr(node, par)[i].label]) for node in [T] for par, val in node.default_pars().items() for i in range(len(val)) if "QD" in par]      # All new parameter tuples that will be added as children here; keep the label in the fourth element of the tuple
        newParsNodes += [tuple([node.name] + ['ampl'] + [0] + [node.alias]) for node in T.descendants() if type(node) is chemNodeT]
        for pars in sorted(newParsNodes, key = lambda par : [i for i, x in enumerate(myOrder) if x in par[1]][-1] ):
            P.addChild(viewNode(name = pars[0:3], alias = pars[3], nodeType='param'))    #         + [node.aliasQD[i]]
    elif type(T) is not chemNodeDB:
        # General chemNode (e.g. a group of chemicals)
        P.addChild(viewNode(name = tuple([T.name] + ['ampl'] + [0]), nodeType='param', alias='intn' ))
        P.addChild(viewNode(name = tuple([T.name] + ['chsh'] + [0]), nodeType='param' ))
        P.addChild(viewNode(name = tuple([T.name] + ['alph'] + [0]), nodeType='param' ))
        for node in T.children():
            P.addChild(getDisplayTree(node))
    elif T.childCount() == 1 and T.child(0).childCount() == 1:
        # A singlet (one QD node with one T node as a child)
        P.addChild(viewNode(name = tuple([T.name] + ['ampl'] + [0]), nodeType='param', alias='intn' ))      #
        P.addChild(viewNode(name = tuple([T.child(0).child(0).name] + ['ampl'] + [0]), nodeType='param', alias='intn' ))      # Intensity corresponding to the terminal node
        P.addChild(viewNode(name = tuple([T.child(0).name] + ['chshQD'] + [0]), nodeType='param' ))
        P.addChild(viewNode(name = tuple([T.child(0).name] + ['alphQD'] + [0]), nodeType='param' ))
    else:         # chemNodeDB has several QD systems
        P.addChild(viewNode(name = tuple([T.name] + ['ampl'] + [0]), nodeType='param', alias='intn' ))
        P.addChild(viewNode(name = tuple([T.name] + ['chsh'] + [0]), nodeType='param' ))
        P.addChild(viewNode(name = tuple([T.name] + ['alph'] + [0]), nodeType='param' ))

        # Fix faulty labels of parameters and alises of the terminal nodes
        for node in T.children():
            for par, val in node.default_pars().items():
                for i in range(len(val)):
                    if not isinstance(getattr(node, par)[i].label, str):
                        old = getattr(node, par)
                        old[i] = parsSpec(old[i].min, old[i].max, label='', distr=old[i].distr, p1=old[i].p1, p2=old[i].p2)
                        setattr(node, par, old)
                    elif par == 'chshQD' and node.child(i).alias == '':
                        node.child(i).alias = getattr(node, par)[i].label

        newParsNodes = [tuple([node.name] + [par] + [i] + [getattr(node, par)[i].label]) for node in T.children() for par, val in node.default_pars().items() for i in range(len(val)) if "QD" in par]      # All new parameter tuples that will be added as children here; keep the label in the fourth element of the tuple
        newParsNodes += [tuple([node.name] + ['ampl'] + [0] + [node.alias]) for node in T.descendants() if type(node) is chemNodeT]
        for pars in sorted(newParsNodes, key = lambda par : [i for i, x in enumerate(myOrder) if x in par[1]][-1] ):     # Loop over the list of tuples
            P.addChild(viewNode(name = pars[0:3], alias = pars[3], nodeType='param'))    #         + [node.aliasQD[i]]
    return P

class ChemTreeModel(QtCore.QAbstractItemModel):
    """A treeView representation class"""

    skipColumns = 5    # Number of columns that display numerical values (min, max, default, current) + the name column
    requestParameterChange = pyqtSignal(object, float)    # Emmitted when current values are chenged by the user (parKey, newVal)

    @staticmethod
    def showName(name):
            """Determines how to display the name of an item"""
            if isinstance(name, tuple):
                return name[1]   # return the first element in the tuple
            else: return name

    def __init__(self, datum, parent = None):
        super().__init__()     # QtCore.QAbstractItemModel.__init__(self)
        self.datum = datum             # A pointer to the Workspace, the Series, or the Datum
        self.actvStepIndx = -1         # Currently selected active step id
        self._indxRoot = QtCore.QModelIndex()    # "Invalid" index to point to the root of the display
        self._parsSigma2 = viewNode(('.', 'sigma2', 0), alias=None, nodeType='param')       # alias='Variance of noise, s2'
        self._parsPH0 = viewNode(('.', 'theta', 0), alias=None, nodeType='param')         # alias='Zero-order phase (PH0)'
        self._parsPH1 = viewNode(('.', 'tau', 0), alias=None, nodeType='param')       # alias='Acquisition delay (PH1)'
        # self._ratioTLS = viewNode(('.', 'gamma', 0), alias='TLS ratio', nodeType='param')
        # self._lshape = viewNode('_lshape', alias='Lineshape correction', nodeType='lshape')           # self._lshape = viewNode('lshapeX', alias='Fit custom shape', nodeType='bool')

        self.resetChemTree(flag=False)
        self.resetLshapeTree(flag=False)

    def resetChemTree(self, T=None, flag=True):
        """Updates the chemical tree."""
        if flag: self.beginResetModel()

        if T is not None:
            self.datum.setTree(T)

        # The tree of parameters to be displayed
        self.TP = getDisplayTree(self.datum.T) if self.datum.T is not None else viewNode('')
        self._unfittableParsKeys = None       # A list (or set) of keys that can not be fitted

        if flag: self.endResetModel()

    def resetLshapeTree(self, flag=True):
        """Updates the tree of lineshape correction parameters."""
        if flag: self.beginResetModel()

        # Uncomment to show 2nd and 3rd order coreection parameters
        # # Reset the lineshape correction subtree
        # self._lshape.clearChildren()
        # self._lshape.addChild(viewNode('lshapeX', alias='Fit custom shape', nodeType='bool'))       # Custom lineshape
        # supscr = ['nd', 'rd'] + ['th']*(self.datum.lshapeOrder-2)
        # for i in range(self.datum.lshapeOrder):
        #     self._lshape.addChild(viewNode(('.','lshapeR',i), alias='{}{} order Re'.format(i+2, supscr[i]), nodeType='param'))
        #     self._lshape.addChild(viewNode(('.','lshapeI',i), alias='{}{} order Im'.format(i+2, supscr[i]), nodeType='param'))

        if flag: self.endResetModel()

    def setNewDatum(self, datum):
        if isinstance(self.datum, Workspace) or self.datum.steps != datum.steps:
            # If the list of steps has changed
            oldColumnCount = self.columnCount()
            newColumnCount = self.skipColumns
            try:
                newColumnCount += len(datum.steps)
            except AttributeError: pass

            parent = QtCore.QModelIndex()

            if oldColumnCount < newColumnCount:
                self.beginInsertColumns(parent, oldColumnCount, newColumnCount-1)
            elif oldColumnCount > newColumnCount:
                self.beginRemoveColumns(parent, newColumnCount, oldColumnCount-1)

            self.datum = datum
            self.actvStepIndx = len(self.datum.steps)-1

            if oldColumnCount < newColumnCount:
                self.endInsertColumns()
            elif oldColumnCount > newColumnCount:
                self.endRemoveColumns()
            else: self.notifyDataChanged()

        else:
            # The list of steps is the same (the new Datum is in the same Series)
            self.datum = datum
            self.notifyDataChanged()

    def setHCmode(self, new_HCmode):
        """Sets the HC mode of the workspace."""
        if self.datum.HCmode != new_HCmode:

            self.beginResetModel()

            self.datum.setHCmode(new_HCmode)
            self.resetChemTree(flag=False)
            self.resetLshapeTree(flag=False)

            self.endResetModel()

    def fullReset(self, datum):
        self.beginResetModel()
        self.datum = datum
        self.resetChemTree(flag=False)
        self.resetLshapeTree(flag=False)
        self.endResetModel()

    def notifyDataChanged(self, key=None):
        """Update the entire tree (if index is None); otherwise, only update the row at index."""
        if key is not None:
            index = self.indxByKey(key)
            index_start = self.index(index.row(), 0, self.parent(index))
            index_stop = self.index(index.row(), self.columnCount(self.parent(index)), self.parent(index))
        else:
            index_start, index_stop = self._indxRoot, self._indxRoot

        self._unfittableParsKeys = None              # Will reset the list of fittable parameters

        self.dataChanged.emit(index_start, index_stop)     # Update the entire tree

    def onHeaderSectionPressed(self, clmn):
        """Is called when a user selects a new column. Connected to the slot"""
        if clmn >= self.skipColumns:
            self.actvStepIndx = self.clmn2step(clmn)
            self.dataChanged.emit(self.index(0,clmn), self.index(self.rowCount(self._indxRoot)-1, clmn))                # Update the entire column

    def onHeaderSectionMoved(self, logicalIndex, oldVisualIndex, newVisualIndex):
        """Is called when columns in the tree view are moved."""
        print('Column {} is moved from {} to {}.'.format(logicalIndex, oldVisualIndex, newVisualIndex))

    def headerData(self, section, orientation, role):

        #if role == QtCore.Qt.SizeHintRole:
        #    return QtCore.QSize(20, 20)

        if role == QtCore.Qt.DisplayRole:

            if orientation == QtCore.Qt.Horizontal:
                head = ['Chemicals/parameters', 'min', 'max', 'dflt', 'value']

                if section < 5:
                    return head[section]
                else:
                    # return section - (self.skipColumns-1)  # In normal order
                    return self.columnCount() - section      # In reversed order
            else:
                return None

    def columnCount(self, parent = QtCore.QModelIndex()):
        try:
            return self.skipColumns + len(self.datum.steps)
        except AttributeError:
            return self.skipColumns

    def rowCount(self, index):
        """Number of rows (children) for each item in the tree. INPUTS: QModelIndex. OUTPUT: int"""
        if index.isValid():
            return index.internalPointer().childCount()
        else:
            #print("row count: index is invalid")
            return 4    # Number of rows in the display root

    def parent(self, index):
        """Should return QModelIndex of the parent of the node with the given QModelIndex. INPUTS: QModelIndex. OUTPUT: QModelIndex"""
        if index.isValid():
            p = index.internalPointer().parent()
            if p:
                return self.createIndex(p.siblID(), 0, p)
        #else: print("parent: index is invalid")
        return self._indxRoot

    def index(self, row, column, prnt=QtCore.QModelIndex()):
        """Should return a QModelIndex that corresponds to the given row, clmn and parent node. INPUTS: int, int, QModelIndex. OUTPUT: QModelIndex"""
        if prnt == self._indxRoot:         # Parent is the root
            if row == 0:
                i = self.createIndex(row, column, self.TP)
                return i
            elif row == 1:
                return self.createIndex(row,column,self._parsPH0)
            elif row == 2:
                return self.createIndex(row,column,self._parsPH1)
            elif row == 3:
                return self.createIndex(row,column,self._parsSigma2)
            # elif row == 4:
            #     return self.createIndex(row,column,self._ratioTLS)
            # elif row == 4:
            #     return self.createIndex(row, column, self._lshape)
        else:
            parent = prnt.internalPointer()
            child = parent.child(row)
            if child:
                return self.createIndex(row, column, child)
            else:
                return QtCore.QModelIndex()

    def indexByKey(self, key):
        """Searches for the element specified by its key in the TP tree and returns its index."""
        try:
            item = self.TP[key]
            row = item.siblID()
        except (KeyError, AttributeError):
            if key == ('.', 'theta', 0):
                item = self._parsPH0
                row = 1
            elif key == ('.', 'tau', 0):
                item = self._parsPH1
                row = 2
            elif key == ('.', 'sigma2', 0):
                item = self._parsSigma2
                row = 3
            else: return None
        column = 0
        return self.createIndex(row, column, item)

    def data(self, index, role):
        # Check if the list of fittable parameters needs to be computed again
        if self._unfittableParsKeys is None:
            try:
                # print('Recomputing the list of fittable parameters')
                _, self._unfittableParsKeys = self.datum.fittableParsKeys()
            except AttributeError:
                # If Worksapce or Series do not do anything
                self._unfittableParsKeys = []

        if not index.isValid() or self.datum.T is None:
            return None

        node = index.internalPointer()
        row = index.row()
        clmn = index.column()

        if role == QtCore.Qt.CheckStateRole and clmn >= self.skipColumns:
            # step_indx = clmn-self.skipColumns    # Normal order
            step_indx = self.columnCount() - clmn - 1    # Reversed order

            if not isinstance(self.datum, Datum): return None

            if node.nodeType == 'param':
                if node.name in self.datum.steps[step_indx].parsKeys:
                    return QtCore.Qt.Checked
                elif self.datum.isAutofittable(key=node.name) and node.name in self.datum.steps[step_indx].autoKeys:
                    return QtCore.Qt.PartiallyChecked
                else:
                    return QtCore.Qt.Unchecked

            elif node.name == 'lshapeX':
                return self.datum.steps[step_indx].fitCustomLshape

            else: return None

        # Setup font for the reported nodes (only in the 0-th column)
        if clmn == 0 and role == QtCore.Qt.FontRole:
            font = QtGui.QFont()    # Default font

            if node.nodeType in ['chemNode', 'chemNodeDB', 'chemNodeQM']:     # Chemical node
                if not self.datum.T[node.name].isReported():
                    font.setStyle(QtGui.QFont.StyleItalic)
                elif node.name in self.datum.repRootNames:
                    font.setBold(True)

                if self.datum.isXclRootName(node.name):
                    font.setStrikeOut(True)

            if node.nodeType == 'param':
                if node.name in self._unfittableParsKeys:
                    font.setStrikeOut(True)

            return font

        if role in [QtCore.Qt.DisplayRole, QtCore.Qt.EditRole, QtCore.Qt.ForegroundRole]:
            if node.nodeType == 'param':
                key = node.name
                prior = self.datum.getPrior(key)
                if clmn == 0:
                    return node.alias if node.alias != '' else str(key[1])
                elif clmn == 1:
                    return prior.min
                elif clmn == 2:
                    return prior.max
                elif clmn == 3:
                    return prior.dflt()
                elif clmn == 4:
                    try:
                        crntVal = self.datum.getCrntVal(key)
                        if role == QtCore.Qt.DisplayRole:
                            return "{:.4g}".format(crntVal)
                        elif (role == QtCore.Qt.ForegroundRole) and (self.flags(index) & QtCore.Qt.ItemIsEnabled):
                            if (crntVal-prior.min < 1e-06*(prior.max-prior.min) or prior.max - crntVal < 1e-06*(prior.max-prior.min)) and not np.isinf([prior.min, prior.max]).any():
                                return QtGui.QBrush( QtGui.QColor(255, 0, 0) )
                            elif crntVal != prior.dflt():
                                return QtGui.QBrush( QtGui.QColor(0, 0, 255) )
                        else:
                            return "{:.10g}".format(crntVal)
                    except AttributeError: return None    # If there is no crntParsH attribute (as in the Workspace itself)
                else:
                    pass

            elif node.nodeType in ['chemNode', 'chemNodeDB', 'chemNodeQM'] and clmn == 4:
                if self.datum.T[node.name].isReported() and node.name not in self.datum.repRootNames:
                    crntVal = self.datum.T[node.name].intn

                    if role == QtCore.Qt.DisplayRole:
                        return "{:.4g}".format(crntVal)
                    elif role == QtCore.Qt.EditRole:
                        return "{:.10g}".format(crntVal)

            else:
                if index.column() == 0:
                    # Name of the chemical or parameter
                    return node.alias if node.alias != '' else str(node.name)

        elif role == QtCore.Qt.DecorationRole:
            displayIcon = None
            if clmn == 0:
                if node.nodeType == 'param':
                    if "chshQD" in node.name[1]:
                        displayIcon = QIcon("icons\icon_deltaQD.png")
                    elif node.name[1] == 'ampl':
                        displayIcon = QIcon("icons\icon_ampl.png")
                    elif node.name[1][:6] == "alphQD":
                        displayIcon = QIcon("icons\icon_alphaQD.png")
                    elif node.name[1][:6] == "jcplQD":
                        displayIcon = QIcon("icons\icon_jcplQD.png")
                    elif node.name[1][:4] == "alph":
                        displayIcon = QIcon("icons\icon_alpha.png")
                    elif node.name[1][:4] == "chsh":
                        displayIcon = QIcon("icons\icon_delta.png")
                elif node.nodeType == 'lshape':
                    displayIcon = QIcon('icons\icon_lshape.png')
                elif node.name == 'lshapeX':
                    displayIcon = QIcon('icons\icon_lshape.png')
                else:
                    displayIcon = QIcon("icons\icon_chemMixture.png")
            return displayIcon

        # Display the active step in a different color
        if role == QtCore.Qt.BackgroundRole and clmn >= self.skipColumns and self.clmn2step(clmn) == self.actvStepIndx:
            return QtGui.QBrush(QtGui.QColor(255, 204, 41, 64))

        return None

    def flags(self, index):
        if not index.isValid():
            return None

        node = index.internalPointer()
        clmn = index.column()
        row = index.row()

        # Initialize the result
        if node.nodeType=='param' and node.name[1]=='gamma' and config.SAMPL_funcType!='TLS':
            result = QtCore.Qt.NoItemFlags
        else:
            result = QtCore.Qt.ItemIsEnabled

        if clmn == 0:
            if node.nodeType == 'param':
                return result | QtCore.Qt.ItemIsSelectable
            elif node.nodeType == 'chemNode':
                return result | QtCore.Qt.ItemIsEditable # | QtCore.Qt.ItemIsUserCheckable
            elif node.nodeType in ['chemNodeDB', 'chemNodeQM']:
                return result # | QtCore.Qt.ItemIsUserCheckable
            else:
                return result

        # Allow setting intensities of nodes in the tree
        elif clmn == 4:
            if node.nodeType == 'lshape':
                return result
            elif node.nodeType in ['chemNode', 'chemNodeDB', 'chemNodeQM']:
                return result | QtCore.Qt.ItemIsEditable
            else:
                return result | QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsSelectable
            """elif node.nodeType in ['chemNode', 'chemNodeDB', 'chemNodeQM'] and clmn == 4:
            if self.datum.T[node.name].isReported() and node.name not in self.datum.repRootNames:
                return QtCore.Qt.ItemIsEditable | result"""

        elif node.nodeType == 'param':
            if clmn in [1, 2, 3]:
                return result | QtCore.Qt.ItemIsSelectable
            elif clmn == 4:
                return result | QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsSelectable
            else:
                return result | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsSelectable

        elif node.nodeType == 'bool':
            if clmn < self.skipColumns:
                return QtCore.Qt.ItemIsEnabled
            else:
                return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable

            """elif node.nodeType == 'intn':
            if clmn in [1, 2, 3]:
                return result | QtCore.Qt.ItemIsSelectable
            elif clmn == 4:
                return QtCore.Qt.ItemIsEditable | result | QtCore.Qt.ItemIsSelectable
            else:
                return result |  QtCore.Qt.ItemIsSelectable # | QtCore.Qt.ItemIsUserCheckable"""

        else:
            return result

    def setData(self, index, value, role = QtCore.Qt.EditRole):
        """Stores changed data."""
        if not index.isValid():
            return False

        node = index.internalPointer()
        clmn = index.column()
        row = index.row()

        if role == QtCore.Qt.EditRole:
            if node.nodeType == 'param':
                # Update parameter specification
                if clmn == 4:
                    # try:
                    # self.datum.setCrntVal(node.name, value)
                    self.requestParameterChange.emit(node.name, float(value))
                    # self.dataChanged.emit(index, index)
                    # except: return False

            elif node.nodeType in ['chemNode', 'chemNodeDB', 'chemNodeQM'] and clmn == 4 and self.datum.T[node.name].isReported() and node.name not in self.datum.repRootNames:
                # self.datum.T[node.name].set_intn(float(value))
                self.requestParameterChange.emit((node.name, 'intn', 0), float(value))
                # self.dataChanged.emit(index, index)
            else:
                # Change the node names
                if clmn == 0:
                    success = self.datum.renameTreeNode(node.name, value)
                    if success:
                        for chld in self.TP[node.name].children():
                            if isinstance(chld.name, tuple):
                                # Need to check if this parameter tuple corresponds to the entire spin system or one of its children (e.g. amplitudes of separate peaks represented by terminal nodes)
                                try:
                                    suffix = re.findall('-\d+\.\d+', chld.name[0])[-1]   # Match '-' followed by any (non-zero) number of digits followed by '.' foloowed by any non-zero number of digits
                                except IndexError: suffix = ''        # If no matches have been found with the regular expression and the suffix is empty
                                chld.rename((value+suffix, chld.name[1], chld.name[2]))
                        self.TP[node.name].rename(value)
                    self.dataChanged.emit(index, index)

            return True

        if role == QtCore.Qt.CheckStateRole and clmn >= self.skipColumns:
            # Set the tick boxes
            shiftPressed = (QtGui.QApplication.keyboardModifiers() == QtCore.Qt.ShiftModifier)
            # Find at which step we are now
            step_indx = self.clmn2step(clmn)

            if node.nodeType == 'param':
                try:
                    self.datum.steps[step_indx].autoKeys.remove(node.name)

                    # Repeat for the rest of the steps
                    if shiftPressed:
                        for step in self.datum.steps:
                            try:
                                step.autoKeys.remove(node.name)
                            except KeyError: pass

                    if not self.datum.isAutofittable(key=node.name):
                        self.datum.steps[step_indx].parsKeys.add(node.name)

                        # Repeat for the rest of the steps
                        if shiftPressed:
                            for step in self.datum.steps:
                                step.parsKeys.add(node.name)
                except KeyError:
                    try:
                        self.datum.steps[step_indx].parsKeys.remove(node.name)

                        # Repeat for the rest of the steps
                        if shiftPressed:
                            for step in self.datum.steps:
                                try:
                                    step.parsKeys.remove(node.name)
                                except KeyError: pass

                        if self.datum.isAutofittable(key=node.name):
                            self.datum.steps[step_indx].autoKeys.add(node.name)
                            # Repeat for the rest of the steps
                            if shiftPressed:
                                for step in self.datum.steps:
                                    step.autoKeys.add(node.name)
                    except KeyError:
                        self.datum.steps[step_indx].parsKeys.add(node.name)
                        # Repeat for the rest of the steps
                        if shiftPressed:
                            for step in self.datum.steps:
                                step.parsKeys.add(node.name)

            elif node.nodeType == 'bool':
                if node.name == 'lshapeX':
                    self.datum.steps[step_indx].fitCustomLshape = not self.datum.steps[step_indx].fitCustomLshape

                    # Repeat for the rest of the steps
                    if shiftPressed:
                        for step in self.datum.steps:
                            step.fitCustomLshape = not self.datum.steps[step_indx].fitCustomLshape

            if shiftPressed:
                self.dataChanged.emit(self.index(row,0), self.index(row, self.columnCount()))                # Update the entire current row
            else: self.dataChanged.emit(index, index)              # Update only the current index
            return True

        return False

    def addStep(self):
        try:
            newStep = Step(frqBlkIds = copy.copy(self.datum.steps[-1].frqBlkIds), autoKeys = copy.copy(self.datum.steps[-1].autoKeys))
        except AttributeError:
            return False

        parent = QtCore.QModelIndex()
        self.beginInsertColumns(parent, self.columnCount(), self.columnCount())
        self.datum.steps.append(newStep)
        self.actvStepIndx = len(self.datum.steps) - 1            # Set the last step as active
        self.endInsertColumns()

    def delStep(self):
        # Removes the active step and sets the previous one as active
        try:
            if len(self.datum.steps) > 1:
                parent = QtCore.QModelIndex()
                self.beginRemoveColumns(parent, self.columnCount()-1, self.columnCount()-1)
                self.datum.steps.pop(self.actvStepIndx)              # Remove the step
                self.actvStepIndx = max(0, self.actvStepIndx - 1)    # Set the active step to previous
                self.endRemoveColumns()
        except AttributeError:      # If the datum is the entire Workspace
            return False

    def moveStep(self, oldVisualIndex, newVisualIndex):
        """Moves a step according to indices of columns in the associated View."""
        if oldVisualIndex >= self.skipColumns and newVisualIndex >= self.skipColumns:
            oldStepID = self.clmn2step(oldVisualIndex)
            newStepID = self.clmn2step(newVisualIndex)

            # Move the Step in the datastructure
            print('Step {} is moving to {}.'.format(oldStepID+1, newStepID+1))
            movingStep = self.datum.steps.pop(oldStepID)
            self.datum.steps.insert(newStepID, movingStep)

            # Set it active and update the View
            self.actvStepIndx = newStepID

    def setActiveStep(self, clmn):
        """Is called when a user selects a new column corresponding to a step. Connected to the slot in the View."""
        if clmn >= self.skipColumns:
            self.actvStepIndx = self.clmn2step(clmn)
            self.dataChanged.emit(self.index(0,clmn), self.index(self.rowCount(self._indxRoot)-1, clmn))                # Update the entire column

    def addChemical(self, index, source = 'new'):
        """Adds a new chemical to the tree as a child to node index."""

        # Create new chemical node or load a subtree
        if source == 'new':
            X = chemNode('New group')
        elif source == 'file':
            filename = QFileDialog.getOpenFileName(None, 'Import parameter tree', '.', filter = "Chemical trees (*.ctr)")
            X = loadTree(filename)
            # if filename:
            #     with open(filename, 'rb') as fp:
            #         data = dill.load(fp)
            #     # New tree and its parameters
            #     X = data["tree"]
            #     X.setTreeBook()
            #     pars = data["pars"]
        elif source == 'DB':
            newName, QDpars, accepted = ChooseFromDBDialog.run( forbidden_names=list(self.datum.T.keys()) )
            if accepted:
                X = chemNodeQM(newName, QDpars=QDpars)      # X = chemNodeDB(newName, QDpars=QDpars)
            else:
                return 0
        elif source == 'spsy':
            name = 'New spin system'
            """chsh = [parsSpec()]*2
            jcpl = [parsSpec()]
            chshAsgn = [1, 2]
            jcplAsgn = [[0, 1], [0, 0]]"""
            chsh = [parsSpec(min=-1.0, max=1.0, dval=0.0)]*1
            jcpl = []
            chshAsgn = [1]
            jcplAsgn = None
            spsy = spsySpec(chsh, jcpl, chshAsgn, jcplAsgn, mult=1)
            X = chemNodeQD(name, spsy)  # New spin system node (QD)
            for j in range(len(chsh)):
                X.addChild(chemNodeT(name + '-0.' + str(j+1), intn=chshAsgn.count(j+1),
                                        alias = name+'-'+chsh[j].label if chsh[j].label!='' else ''))      # , intn=spsy.mult

        # Get the node in the parameter tree to which new chemical will be attached
        prnt = index.internalPointer()

        success = self.datum.addTreeNode(X, prnt.name)

        if success:                 # self.datum.T has been updated
            self.beginInsertRows(index, 0, 0) # Parent node, first and last position

            prnt.addChild(getDisplayTree(X))

            # newTP = getDisplayTree(X)   # New parameter tree
            # print(repr(X))
            #
            # newTP = getDisplayTree(self.datum.T)   # New parameter tree
            #
            # # Swap children between the old and new parameter trees
            # existingChildrenNames = [chld.name for chld in prnt.children()]
            # for chld in newTP[prnt.name].children():
            #     if chld.name not in existingChildrenNames:
            #         prnt.addChild(chld.cut())

            self.endInsertRows()

    def remChemical(self, index):
        """Removes a chemical from the tree"""
        node = index.internalPointer()

        self.beginRemoveRows(self.parent(index), index.row(), index.row()) # Parent node, first and last position
        success = self.datum.delTreeNode(node.name)
        if success:
            node.cut()

        self.endRemoveRows()

    def increaseOrder(self):
        """Increases the order of the lineshape correction polynomial."""
        lshapeOrder = self.datum.lshapeOrder           # Current order of the lineshape
        self.beginInsertRows(self.index(4, 0), 2*lshapeOrder, 2*lshapeOrder+1) # Parent node, first and last position
        self.datum.set_lshapeOrder(lshapeOrder+1)
        self.resetLshapeTree(flag=False)
        self.endInsertRows()

    def decreaseOrder(self):
        """Increases the order of the lineshape correction polynomial."""
        lshapeOrder = self.datum.lshapeOrder           # Current order of the lineshape
        self.beginRemoveRows(self.index(4, 0), 2*lshapeOrder-1, 2*lshapeOrder) # Parent node, first and last position
        self.datum.set_lshapeOrder(lshapeOrder-1)
        self.resetLshapeTree(flag=False)
        self.endRemoveRows()

    def clmn2step(self, clmn):
        """A utility function to convert a column index to the corresponding step index."""
        # return clmn-self.skipColumns    # Normal order
        return self.columnCount() - clmn - 1    # Reversed order

class ChemTreeView(QTreeView):
    """Model/View based class to display chemical trees."""

    changedParsList = pyqtSignal(int)              # Signalizes to update the parameters list widget and carries the index of the active step
    changedSelected = pyqtSignal(object, object)           # Emits names of the current and previously selected nodes
    requestAdjustment = pyqtSignal(object)         # Requests the phase correction; object = 'Ph0', 'Ph1', or 'PhX'

    class ParsSpecDialog(QDialog):
        """A dialog to set specification for a parameter."""

        def __init__(self, name, param, crntVal=None, parent = None):
            super().__init__(parent)
            layoutMain = QVBoxLayout(self)
            layoutForm = QFormLayout()

            # Add widgets for entering parameters
            self.editMin = MyDoubleEdit(param.min)
            self.editMax = MyDoubleEdit(param.max)
            self.editDfltVal = MyDoubleEdit(param.dflt())
            self.cmboxPrior = QComboBox()
            self.cmboxPrior.addItems(['Uniform', 'Gaussian', 'Log-Normal', 'Inverse-Gamma', 'Constant'])
            index = self.cmboxPrior.findText(param.distr)
            if index != -1:
                self.cmboxPrior.setCurrentIndex(index)
            self.cmboxPrior.currentIndexChanged.connect(self.setDistrForm)
            self.labelPriorP1, self.labelPriorP2 = QLabel(), QLabel()
            self.editPriorP1, self.editPriorP2 = MyDoubleEdit(param.p1), MyDoubleEdit(param.p2)
            actnDfltFromCrnt = QAction(QIcon('icons\icon_dfltFromCrnt.png'), 'Update from current', self)
            actnDfltFromCrnt.setStatusTip('Update from current')
            self.bttnDfltFromCrnt = QToolButton()
            self.bttnDfltFromCrnt.setDefaultAction(actnDfltFromCrnt)
            self.bttnDfltFromCrnt.setAutoRaise(True)
            dfltLayout = QHBoxLayout()
            dfltLayout.addWidget(self.editDfltVal)
            dfltLayout.addWidget(self.bttnDfltFromCrnt)

            # Set up the QFormLayout
            layoutForm.addRow("Upper bnd.", self.editMax)
            layoutForm.addRow("Lower bnd.", self.editMin)
            layoutForm.addRow("Default val.", dfltLayout)
            layoutForm.addRow(" ", None)
            layoutForm.addRow("Prior dist.", self.cmboxPrior)
            layoutForm.addRow(self.labelPriorP1, self.editPriorP1)
            layoutForm.addRow(self.labelPriorP2, self.editPriorP2)
            self.setDistrForm(distr=self.cmboxPrior.currentText())

            # Checkbutton to select how to save the default parameters
            self.chckSeries = QCheckBox("Set for entire seires")
            if crntVal is None:     # It is a Series, not a Datum
                self.chckSeries.setChecked(True)
                self.chckSeries.setEnabled(False)
                self.bttnDfltFromCrnt.setEnabled(False)
            else:
                actnDfltFromCrnt.triggered.connect(lambda : self.editDfltVal.setValue(crntVal))

            # OK and Cancel buttons
            self.buttonsBox = QDialogButtonBox(
                QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
                Qt.Horizontal, self)
            layoutMain.addLayout(layoutForm)
            layoutMain.addWidget(self.chckSeries)
            layoutMain.addWidget(self.buttonsBox)

            self.buttonsBox.accepted.connect(self.accept)
            self.buttonsBox.rejected.connect(self.reject)

            # Resize and change the caption
            prnt_pos = self.parent().mapToGlobal(QtCore.QPoint(0,0))
            prnt_siz = self.parent().size()
            w, h = 180, 210
            x = prnt_pos.x() + (prnt_siz.width()-w)/2
            y = prnt_pos.y() + (prnt_siz.height()-h)/2
            self.setGeometry(x,y, w, h)
            self.setWindowFlags(QtCore.Qt.Tool)
            self.setWindowTitle(str(name))

        def setDistrForm(self, distr):
            """Sets up a form to enter parameters of the distribution."""
            if distr in [0, 'Uniform']:
                self.labelPriorP1.setText('')
                self.labelPriorP2.setText('')
                self.editPriorP1.setVisible(False)
                self.editPriorP2.setVisible(False)
            elif distr in [1, 'Gaussian']:
                self.labelPriorP1.setText('Mean, \u03BC')
                self.labelPriorP2.setText('Std. dev., \u03C3')
                self.editPriorP1.setVisible(True)
                self.editPriorP2.setVisible(True)
            elif distr in [2, 'Log-Normal']:
                self.labelPriorP1.setText('Location')
                self.labelPriorP2.setText('Scale')
                self.editPriorP1.setVisible(True)
                self.editPriorP2.setVisible(True)
            elif distr in [3, 'Inverse-Gamma']:
                self.labelPriorP1.setText('Shape, \u03B1')     # Unicode alpha
                self.labelPriorP2.setText('Scale, \u03B2')     # Unicode beta
                self.editPriorP1.setVisible(True)
                self.editPriorP2.setVisible(True)
            elif distr in [4, 'Constant']:
                self.labelPriorP1.setText('')
                self.labelPriorP2.setText('')
                self.editPriorP1.setVisible(False)
                self.editPriorP2.setVisible(False)

        # get the selection
        def getSelection(self):
            xmin = self.editMin.value() if abs(self.editMin.value()-(-np.pi))>0.001 else -np.pi
            xmax = self.editMax.value() if abs(self.editMax.value()-np.pi)>0.001 else np.pi
            xmin, xmax = min(xmin, xmax), max(xmin, xmax)
            return {'min':xmin, 'max':xmax,\
                    'distr':self.cmboxPrior.currentText(),\
                    'p1':self.editPriorP1.value(),\
                    'p2':self.editPriorP2.value(),\
                    'dval':self.editDfltVal.value()},\
                    self.chckSeries.isChecked()

    class LabelAndButton(QWidget):
        """A widget consisting of a label and a small button, e.g. used to display phase adjustment in the tree."""

        clicked = pyqtSignal()

        def __init__(self, caption, parent = None):
            super().__init__(parent)

            layout = QHBoxLayout()
            label = QLabel(caption)
            #label.setMaximumSize(250, 18)
            button = QPushButton('A')
            button.setMaximumSize(18, 18)
            button.clicked.connect(lambda : self.clicked.emit())            # Emit the clicked signal
            layout.addWidget(label, Qt.AlignLeft|Qt.AlignBottom)
            layout.addWidget(button, Qt.AlignRight|Qt.AlignVCenter)
            layout.setContentsMargins(3,0,2,0)     # void QLayout::setContentsMargins(int left, int top, int right, int bottom)
            self.setLayout(layout)

    def __init__(self, parent=None):
        super().__init__(parent)    # Initialize a QTreeWidget

        self.setAlternatingRowColors(True)
        self.setHeaderHidden(False)

        # Add actions and setup the context menu
        self.toggleDfltsAction = QAction(QIcon('icons\icon_blank.png'), 'Show default values', self)
        self.toggleDfltsAction.setStatusTip('Show default values')
        self.toggleDfltsAction.setCheckable(True)
        self.toggleDfltsAction.setChecked(False)        # By default, default values are not shown
        self.toggleDfltsAction.toggled.connect(self.toggleDflts)
        self.toggleRangesAction = QAction(QIcon('icons\icon_blank.png'), 'Show parameter ranges', self)
        self.toggleRangesAction.setStatusTip('Show parameter ranges')
        self.toggleRangesAction.setCheckable(True)
        self.toggleRangesAction.setChecked(False)        # By default, ranges are not shown
        self.toggleRangesAction.toggled.connect(self.toggleRanges)
        # set up the menu and the policy
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.onCustomContextMenuRequested)
        self.doubleClicked.connect(self.onDoubleClicked)

        self._copy_buffer = {}

    def setModel(self, model):

        """# Define a proxy model for filtering
        self.proxy = QSortFilterProxyModel()
        self.proxy.setSourceModel(model)
        super().setModel(self.proxy)"""

        super().setModel(model)

        # Set a selection model
        selectionModel = QItemSelectionModel(self.model())
        self.setSelectionModel(selectionModel)
        self.setSelectionMode(QTreeView.ExtendedSelection)

        self.header().sectionCountChanged.connect(self.onSectionCountChanged)
        self.header().setClickable(True)
        self.header().sectionPressed.connect(self.model().setActiveStep)
        self.header().sectionMoved.connect(self.onHeaderSectionMoved)

        #self.header().setDefaultSectionSize(20)
        self.setColumnWidth(0, 150)
        self.setColumnWidth(1, 50)
        self.setColumnWidth(2, 50)
        self.setColumnWidth(3, 50)
        self.setColumnWidth(4, 50)
        self.header().setResizeMode(0, QtGui.QHeaderView.ResizeToContents)

        self.setColumnHidden(1, True)
        self.setColumnHidden(2, True)
        self.setColumnHidden(3, True)

    def currentChanged(self, current, previous):
        """Is called when the current item changes."""
        super().currentChanged(current, previous)
        key_crnt = current.internalPointer().name if current.isValid() else None
        key_prev = previous.internalPointer().name if previous.isValid() else None
        self.changedSelected.emit(key_crnt, key_prev)

    def rowsInserted(self, parent, start, end):
        """Is called to update the view when rows have been inserted."""
        super().rowsInserted(parent, start, end)
        self.hideExcessiveRows()

    def reset(self):
        """Subclassing the reset slot."""
        super().reset()

        self.hideExcessiveRows()
        self._copy_buffer.clear()

        # Create buttons for phase correction
        labelWidgetPH0, labelWidgetPH1 = self.LabelAndButton('Zero-order phase, PH0'), self.LabelAndButton('First-order phase, PH1')
        labelWidgetSig2 = self.LabelAndButton('Var. of noise, \u03C3\u00B2')
        labelWidgetPH0.clicked.connect(lambda : self.requestAdjustment.emit('Ph0'))
        labelWidgetPH1.clicked.connect(lambda : self.requestAdjustment.emit('Ph1'))
        labelWidgetSig2.clicked.connect(lambda : self.requestAdjustment.emit('Rsd'))
        self.setIndexWidget(self.model().indexByKey(key=('.', 'theta', 0)), labelWidgetPH0)
        self.setIndexWidget(self.model().indexByKey(key=('.', 'tau', 0)), labelWidgetPH1)
        self.setIndexWidget(self.model().indexByKey(key=('.', 'sigma2', 0)), labelWidgetSig2)

    def onSectionCountChanged(self, oldCount, newCount):
        """Called by the model after the number of columns is changed."""
        self.setColumnWidth(4, 50)
        for i in range(5, newCount):
            self.setColumnWidth(i, 18)

    def onHeaderSectionMoved(self, logicalIndex, oldVisualIndex, newVisualIndex):
        """Is called when columns in the tree view are moved. Moves them back, but also calls a function in the model to update the list of Steps."""
        self.header().blockSignals(True)
        self.header().moveSection(newVisualIndex, oldVisualIndex)    # Move back
        self.header().blockSignals(False)

        self.model().moveStep(oldVisualIndex, newVisualIndex)

    def onCustomContextMenuRequested(self, pos):
        """Handler of the custom context menu requested signal."""
        popMenu = QMenu(self)

        # Form an array of all selected parameter keys
        #slctdKeys = [index.internalPointer().name for index in self.selectedIndexes() if index.isValid() and index.column() == 0]     # List of all selected parameter keys
        slctdKeys = []
        for index in self.selectedIndexes():
            if index.isValid() and index.column() == 0:
                node = index.internalPointer()
                if node.nodeType == 'param':
                    slctdKeys.append(node.name)

        index = self.indexAt(pos)
        if index.isValid():         # If the click was on an item
            node = index.internalPointer()
            clmn = index.column()
            row = index.row()

            if node.nodeType in ['chemNode', 'chemNodeDB', 'chemNodeQM']:       # Chemical node
                key = node.name

                # Add/remove node actions
                actnToggleReported = QAction(QIcon('icons\icon_blank.png'), 'Reported', self)
                actnToggleReported.setStatusTip('Change the reported state')
                actnToggleReported.setCheckable(True)
                actnToggleReported.setChecked( self.model().datum.T[key].isReported() )
                actnToggleReported.toggled.connect(lambda : self.toggleReported(key) )
                actnToggleExcluded = QAction(QIcon('icons\icon_blank.png'), 'Exclude from fit', self)
                actnToggleExcluded.setStatusTip('Excludes the signature model from the the fit')
                actnToggleExcluded.setCheckable(True)
                actnToggleExcluded.setChecked( self.model().datum.isXclRootName(key) )
                actnToggleExcluded.toggled.connect(lambda : self.toggleExcluded(key) )
                actnNewGroup = QAction(QIcon('icons\icon_blank.png'), 'Add new group', self)
                actnNewGroup.setStatusTip('Add new group')
                actnNewGroup.triggered.connect(lambda : self.model().addChemical(index, source='new'))
                actnAddChemical = QAction(QIcon('icons\icon_addChemical.png'), 'Add new chemical', self)
                actnAddChemical.setStatusTip('Add new chemical')
                actnAddChemical.triggered.connect(lambda : self.model().addChemical(index, source='DB'))
                actnAddSpsy= QAction(QIcon('icons\icon_spin.png'), 'Add spin system', self)
                actnAddSpsy.setStatusTip('Add spin system')
                actnAddSpsy.triggered.connect(lambda : self.model().addChemical(index, source='spsy'))
                actnAddSubtree= QAction(QIcon('icons\icon_blank.png'), 'Insert subtree', self)
                actnAddSubtree.setStatusTip('Insert subtree')
                actnAddSubtree.triggered.connect(lambda : self.model().addChemical(index, source='file'))
                actnRemoveChemical = QAction(QIcon('icons\icon_blank.png'), 'Remove the node', self)
                actnRemoveChemical.setStatusTip('Remove the node')
                actnRemoveChemical.triggered.connect(lambda : self.model().remChemical(index))
                actnSaveSubtree = QAction(QIcon('icons\icon_saveTree.png'), 'Save subtree', self)
                actnSaveSubtree.setStatusTip('Save subtree')
                actnSaveSubtree.triggered.connect(lambda : self.saveSubtree(index))
                actnShowRows = QAction(QIcon('icons\icon_blank.png'), 'Show hidden rows', self)
                actnShowRows.setStatusTip('Show hidden rows')
                actnShowRows.triggered.connect(lambda : self.showRows(key))

                popMenu.addAction(actnToggleReported)
                if actnToggleReported.isChecked():
                    popMenu.addAction(actnToggleExcluded)
                popMenu.addSeparator()
                popMenu.addAction(actnSaveSubtree)
                popMenu.addAction(actnAddSubtree)
                popMenu.addAction(actnNewGroup)
                popMenu.addAction(actnAddChemical)
                popMenu.addAction(actnAddSpsy)
                popMenu.addAction(actnRemoveChemical)
                popMenu.addSeparator()
                popMenu.addAction(actnShowRows)

            elif node.nodeType == 'lshape':
                actnIncreaseOrder = QAction(QIcon('icons\icon_increaseOrder.png'), 'Increase order', self)
                actnIncreaseOrder.setStatusTip('Increase the order of lineshape correction polynomial.')
                actnIncreaseOrder.triggered.connect(self.model().increaseOrder)
                actnDecreaseOrder = QAction(QIcon('icons\icon_decreaseOrder.png'), 'Decrease order', self)
                actnDecreaseOrder.setStatusTip('Decrease the order of lineshape correction polynomial.')
                actnDecreaseOrder.triggered.connect(self.model().decreaseOrder)

                popMenu.addAction(actnIncreaseOrder)
                popMenu.addAction(actnDecreaseOrder)
                popMenu.addSeparator()

            elif node.nodeType == 'param' and len(slctdKeys) > 0:              # Parameter node(s)
                # Single selected row
                if len(slctdKeys) == 1:
                    key = node.name
                    cfunAction = QAction(QIcon('icons\icon_blank.png'), 'Display cost function', self)
                    cfunAction.setStatusTip('Display cost function')
                    cfunAction.triggered.connect(lambda : self.showCfunPopup(key))
                    popMenu.addAction(cfunAction)

                    if key[1] in ['chsh', 'chshQD']:
                        actnScaleToRef = QAction(QIcon('icons\icon_none.png'), 'Set reference', self)
                        actnScaleToRef.setStatusTip('Reset all chemical shifts in the model to the reference')
                        actnScaleToRef.triggered.connect( lambda _ : self.model().datum.shiftToRef(refKey=key) )
                        popMenu.addAction(actnScaleToRef)
                    popMenu.addSeparator()

                # Define parameter setting actions
                actnPromotePriors = QAction(QIcon('icons\icon_globalPriors.png'), 'Set prior as global' if len(slctdKeys) == 1 else 'Set priors as global', self)
                actnPromotePriors.setStatusTip('Use this prior for all datasets in the Workspace')
                actnPromotePriors.triggered.connect(lambda : self.promotePriors(keys=slctdKeys))

                actnResetToDflt = QAction(QIcon('icons\icon_none.png'), 'Reset to default', self)
                actnResetToDflt.setStatusTip('Reset current value(s) to default')
                actnResetToDflt.triggered.connect(lambda : self.resetToDflt(keys=slctdKeys))

                actnHideRows = QAction(QIcon('icons\icon_none.png'), 'Hide parameter' if len(slctdKeys) == 1 else 'Hide parameters', self)
                actnHideRows.setStatusTip('Hide parameters')
                actnHideRows.triggered.connect(lambda : self.hideRows(slctdKeys) )

                actnRemovePars = QAction(QIcon('icons\icon_none.png'), 'Remove parameter' if len(slctdKeys) == 1 else 'Remove parameters', self)
                actnRemovePars.setStatusTip('Remove parameters')
                actnRemovePars.triggered.connect(lambda : self.removePars(slctdKeys) )

                actnCopyCrnt = QAction(QIcon('icons\icon_copy.png'), 'Copy value' if len(slctdKeys) == 1 else 'Copy values', self)
                actnCopyCrnt.setStatusTip('Copy current values')
                actnCopyCrnt.triggered.connect(lambda : self.copyCrntPars(slctdKeys) )

                actnPasteCrnt = QAction(QIcon('icons\icon_pasteCrnt.png'), 'Paste as current', self)
                actnPasteCrnt.setStatusTip('Paste copied to current values')
                actnPasteCrnt.triggered.connect(lambda : self.pasteCrntPars() )
                actnPasteCrnt.setEnabled(len(self._copy_buffer) > 0)

                actnPasteDflt = QAction(QIcon('icons\icon_pasteDflt.png'), 'Paste as default', self)
                actnPasteDflt.setStatusTip('Paste copied to default values')
                actnPasteDflt.triggered.connect(lambda : self.pasteDfltPars() )
                actnPasteDflt.setEnabled(len(self._copy_buffer) > 0)

                popMenu.addAction(actnPromotePriors)
                if isinstance(self.model().datum, Datum):
                    popMenu.addAction(actnCopyCrnt)
                    popMenu.addAction(actnPasteCrnt)
                    popMenu.addAction(actnPasteDflt)
                    popMenu.addAction(actnResetToDflt)
                if isinstance(self.model().datum, Series):
                    popMenu.addAction(actnPasteCrnt)
                    popMenu.addAction(actnPasteDflt)
                popMenu.addSeparator()
                popMenu.addAction(actnHideRows)
                popMenu.addAction(actnRemovePars)

                """elif node.nodeType == 'intn' and len(slctdKeys) == 1:              # Intensity node(s)
                key = ('.', 'ampl', self.model().datum.repRootNames.index(node.name[0]))
                cfunAction = QAction(QIcon('icons\icon_blank.png'), 'Display cost function', self)
                cfunAction.setStatusTip('Display cost function')
                cfunAction.triggered.connect(lambda : self.showCfunPopup(key))
                popMenu.addAction(cfunAction)
                popMenu.addSeparator()"""

        # Add toggle buttons
        popMenu.addAction(self.toggleRangesAction)
        popMenu.addAction(self.toggleDfltsAction)

        # Show the menu
        popMenu.popup(self.viewport().mapToGlobal(pos))

    def onDoubleClicked(self, index):
        """Displays a dialog to edit parameter specifications."""
        node = index.internalPointer()
        clmn = index.column()
        row = index.row()

        if node.nodeType in ['param'] and clmn in [0, 1, 2, 3]:
            key = node.name    # Usual parameter
            param = self.model().datum.getPrior(key)
            try:
                crntVal = self.model().datum.getCrntVal(key)
            except AttributeError:
                crntVal = None

            dialog = self.ParsSpecDialog(key, param, crntVal, parent=self)
            result = dialog.exec_()
            if result == QDialog.Accepted:    # If OK was clicked
                newParSpec, resetSeries = dialog.getSelection()
                crnt = self.model().datum
                if isinstance(crnt, Datum) and resetSeries:
                    crnt = crnt.parent
                else: pass # It is either a Datum and no series flag was set or it is a Series
                crnt.setPrior(key, **newParSpec, reset=True)

    def saveSubtree(self, index):
        """Saves the subtree starting with the node index."""
        node = index.internalPointer()

        filename = QFileDialog.getSaveFileName(parent=self, caption='Select output file', directory='.', filter='NMR worksapce (*.ctr)')
        if filename:
            if filename[-4:] == '.ctr': filename = filename[:-4]

            T = self.model().datum.getTree(node.name)           # Extract the subtree starting with the node key and set its parent to None

            # Save the tree and the parameters
            saveTree(filename, T)

    def showCfunPopup(self, key):
        """Plots the cost function with respect to the particular variable specified by a tuple key."""
        DDD = self.model().datum
        evalPars = copy.deepcopy(DDD.crntParsH)
        actvStep = DDD.steps[-1]

        if key[1] == 'chsh':
            allowShift, shiftingRange = True, DDD.getPrior(key).max - DDD.getPrior(key).min
        elif key[1] == 'alph':
            allowShift, shiftingRange = True, 0.0
        else: allowShift, shiftingRange = False, 0.0

        # allowShift, shiftingRange = False, 0.0

        costFuncOpti = lambda x : (DDD.getPrior(key).abs(x), DDD.evaluate(evalParsH=updateFromFlat(evalPars, [key], [DDD.getPrior(key).abs(x)]), autoKeys=actvStep.autoKeys, \
                                                                          frqBlkIds=actvStep.frqBlkIds, allowShift=allowShift, shiftingRange=shiftingRange)[0])
        self.popupWindow = CfunPopup(costFuncOpti)
        self.popupWindow.show()

    def toggleDflts(self, checked):
        """Show/hide default parameter values."""
        self.setColumnHidden(3, not checked)

    def toggleRanges(self, checked):
        """Show/hide parameter ranges."""
        self.setColumnHidden(1, not checked)
        self.setColumnHidden(2, not checked)

    def toggleReported(self, key):
        self.model().datum.toggleRepRoot(key)
        self.hideExcessiveRows()

    def toggleExcluded(self, key):
        """Sets the key to ignored/not ignored."""
        if not isinstance(self.model().datum, Workspace):
            self.model().datum.toggleXclRootName(key)
            self.model().notifyDataChanged()

    def dataChanged(self, topLeft, btmRight):
        """Reimplemented dataChanged slot."""
        super().dataChanged(topLeft, btmRight)
        self.hideExcessiveRows()

    def hideExcessiveRows(self):
        """Traverse the tree and hide some rows"""
        def traverse(index):
            for row in range(self.model().rowCount(index)):
                chld = self.model().index(row=row, column=0, prnt=index)     # Child of the index
                yield from traverse(chld)
            yield index

        root = self.rootIndex()

        for index in traverse(root):
            if index.isValid():
                node = index.internalPointer()
                if self.model().datum.isXclRootName(node.name):
                    # Hide all children of excluded nodes
                    for row in range(self.model().rowCount(index)):
                        self.setRowHidden(row, index, True)
                elif node.name[1] == 'ampl':
                    self.setRowHidden(index.row(), index.parent(), node.name[0] not in self.model().datum.repRootNames)
                else:
                    self.setRowHidden(index.row(), index.parent(), node.hidden)

    def promotePriors(self, keys):
        """Sets the priors for selected parameters as the global priors in the workspace."""
        for key in keys:
            prior = self.model().datum.getPrior(key)
            self.model().datum.setGlobalPrior(key, prior)

    def resetToDflt(self, keys):
        """Resets current values of parameters in the keys list to their defaults."""
        for key in keys:
            val = self.model().datum.getPrior(key).dflt()
            self.model().datum.setCrntVal(key, val)
        self.model().notifyDataChanged()

    def copyCrntPars(self, keys=None):
        """Copy current values of selected parameters."""
        if keys is None:
            # Copy all parameters
            keys = self.model().datum.allParsKeys()
        self._copy_buffer.clear()
        self._copy_buffer = {key : self.model().datum.getCrntVal(key) for key in keys}

    def pasteCrntPars(self, datums=None):
        """Updates current parameters based on the copied values."""
        # Find which Datums will be updated (either the current Datum or all Datums in the current Series)
        if datums is None:
            datums = [self.model().datum] if isinstance(self.model().datum, Datum) else self.model().datum.data

        for key, val in self._copy_buffer.items():
            try:
                for DDD in datums:
                    DDD.setCrntVal(key, val)
            except KeyError:
                print("Non-existing parameter", key)
        self.model().notifyDataChanged()

    def pasteDfltPars(self, datums=None):
        """Updates default parameters based on the copied values."""
        # Datums is a list of Datum or Series whose parameters need to be updated
        # Find which Datums will be updated (either the current Datum or the entire Series and some Datums that have specific distributions predefined)
        if datums is None:
            datums = [self.model().datum]      # Create a list that contains either the current Datum or the current Series

        for key, val in self._copy_buffer.items():
            for XXX in datums:
                XXX.setPrior(key, dval=val)
                if isinstance(XXX, Series):
                    for DDD in XXX.data:
                        if key in DDD.parsSpecDict:
                            DDD.setPrior(key, dval=val)

        self.model().notifyDataChanged()

    def showRows(self, key):
        """Shows all children rows of the node key."""
        for node in self.model().TP[key].children():
            node.hidden = False

        self.hideExcessiveRows()

    def hideRows(self, keys):
        """Hides the selected rows."""
        for key in keys:
            if key[0] != '.':
                self.model().TP[key].hidden = True
        self.hideExcessiveRows()

    def removePars(self, keys):
        """Removes parameters from the QD system."""
        print(keys)
        print(self.model().datum)

    def selectPickedParameter(self, stemKey):
        """Selects an active paramter for a picked peak."""
        self.selectionModel().clear()

        index = self.model().indexByKey(stemKey)
        self.selectionModel().setCurrentIndex(index, QItemSelectionModel.SelectCurrent | QItemSelectionModel.Rows)

        # Select alpha as well
        QD = stemKey[1][4:]      # Either 'QD' or ''
        index = self.model().indexByKey((stemKey[0], 'alph'+QD, stemKey[2]))
        self.selectionModel().select(index, QItemSelectionModel.Select | QItemSelectionModel.Rows)

    def updateValue(self, stemKey, delta=0.0):
        """Updates the curreent value at key by changing it by the amount delta. Used with the stem dragging functions."""
        # Highlight the changing chemical shift in the tree
        self.selectPickedParameter(stemKey)

        # Update the value
        val = self.model().datum.getCrntVal(stemKey)
        # self.model().datum.setCrntVal(stemKey, val + delta)
        # self.model().notifyDataChanged()

        # Emit the signal to recompute the model
        self.model().requestParameterChange.emit(stemKey, val+delta)

class PeakPickingWidget(QWidget):

    RANGE_MAX = 64
    RANGE_MIN = -64
    assigned = pyqtSignal()

    class PeakTableModel(QtCore.QAbstractTableModel):

        def __init__(self, colors = [[]], headers = [], parent = None):
            QtCore.QAbstractTableModel.__init__(self, parent)
            self.__colors = colors
            self.__headers = headers

        def rowCount(self, parent):
            return len(self.__colors)

        def columnCount(self, parent):
            return len(self.__colors[0])

        def flags(self, index):
            return QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable

        def data(self, index, role):

            if role == QtCore.Qt.EditRole:
                row = index.row()
                clmn = index.column()
                return self.__colors[row][column].name()

            if role == QtCore.Qt.ToolTipRole:
                row = index.row()
                clmn = index.column()
                return "Hex code: " + self.__colors[row][column].name()

            if role == QtCore.Qt.DecorationRole:

                row = index.row()
                clmn = index.column()
                value = self.__colors[row][column]

                pixmap = QtGui.QPixmap(26, 26)
                pixmap.fill(value)

                icon = QtGui.QIcon(pixmap)

                return icon

            if role == QtCore.Qt.DisplayRole:

                row = index.row()
                clmn = index.column()
                value = self.__colors[row][column]

                return value.name()

        def setData(self, index, value, role = QtCore.Qt.EditRole):
            if role == QtCore.Qt.EditRole:

                row = index.row()
                clmn = index.column()

                color = QtGui.QColor(value)

                if color.isValid():
                    self.__colors[row][column] = color
                    self.dataChanged.emit(index, index)
                    return True
            return False

        def headerData(self, section, orientation, role):

            if role == QtCore.Qt.DisplayRole:

                if orientation == QtCore.Qt.Horizontal:

                    if section < len(self.__headers):
                        return self.__headers[section]
                    else:
                        return "not implemented"
                else:
                    return "Color {}".format(section)    # QtCore.QString("Color %1").arg(section)

        #=====================================================#
        #INSERTING & REMOVING
        #=====================================================#
        def insertRows(self, position, rows, parent = QtCore.QModelIndex()):
            self.beginInsertRows(parent, position, position + rows - 1)

            for i in range(rows):

                defaultValues = [QtGui.QColor("#000000") for i in range(self.columnCount(None))]
                self.__colors.insert(position, defaultValues)

            self.endInsertRows()

            return True

        def insertColumns(self, position, columns, parent = QtCore.QModelIndex()):
            self.beginInsertColumns(parent, position, position + columns - 1)

            rowCount = len(self.__colors)

            for i in range(columns):
                for j in range(rowCount):
                    self.__colors[j].insert(position, QtGui.QColor("#000000"))

            self.endInsertColumns()

            return True

    class PeakTreeModel(QtCore.QAbstractItemModel):
        """A treeView representation class"""

        def __init__(self, parent = None):
            super().__init__()     # QtCore.QAbstractItemModel.__init__(self)
            self._indxRoot = QtCore.QModelIndex()    # "Invalid" index to point to the root of the display
            self._tree = stepClass.mdldPeaks         # A hierarchical dictionary of modelled peaks

        def headerData(self, section, orientation, role):
            pass

        def columnCount(self, parent):
            return 4 + len(steps)

        def rowCount(self, index):
            """Number of rows (children) for each item in the tree. INPUTS: QModelIndex. OUTPUT: int"""
            if index.isValid():
                return len(index.internalPointer())
            else:
                return len(self._tree)    # Number of rows in the display root

        def parent(self, index):
            """Should return QModelIndex of the parent of the node with the given QModelIndex. INPUTS: QModelIndex. OUTPUT: QModelIndex"""
            if index.isValid():
                item = index.internalPointer()
                if isinstance(item, dict):
                    if item in self._tree.values():
                        return self._indxRoot
                elif isinstance(item, list):
                    for rep in self._tree.values():
                        if item in rep.values():
                            parent = rep
                            row = [k for k in self._tree.keys()].index(parent)
                            return self.createIndex(row, 0, parent)
                else:   # If it's a peakSpec instance
                    for rep in self._tree.values():
                        for leaf in rep.values():
                            if item in leaf.values():    # Iterate over groups of peaks
                                parent = leaf
                                row = [k for k in rep.keys()].index(parent)
                                return self.createIndex(row, 0, parent)

            #else: print("parent: index is invalid")"""
            return self._indxRoot

        def index(self, row, column, prnt=None):
            """Should return a QModelIndex that corresponds to the given row, clmn and parent node. INPUTS: int, int, QModelIndex. OUTPUT: QModelIndex"""
            if prnt == self._indxRoot:         # Parent is the root
                key = [k for k in self._tree.keys()][row]
                if key:
                    return self.createIndex(row, column, self._tree[key])      # An inner dictionary; parent is the display root
            elif prnt.isValid():    # Inner dictionaries
                parent = prnt.internalPointer()
                if isinstance(parent, dict):                    # Parent is an inner dictionary
                    key = [k for k in parent.keys()][row]
                    if key:
                        return self.createIndex(row, column, parent[key])
                else:    # parent is a list
                    if len(parent) > row:
                        return self.createIndex(row, column, parent[row])      # return index of a peakSpec

            return QtCore.QModelIndex()     # Return invalid index by default

            #if not self.hasIndex(row, column, prnt):
            #    print("doesn't have this index")
            #    return self._rootIndex

        # --------------------------------- M A I N   D I S P L A Y   F U N C T I O N ----------------------------------
        def data(self, index, role):
            if not index.isValid():
                return None

            #print(self.TP)



            node = index.internalPointer()
            if role in [QtCore.Qt.DisplayRole, QtCore.Qt.EditRole, QtCore.Qt.CheckStateRole]:
                if node.nodeType == 'param':
                    if index.column() == 0:
                        return node.alias if node.alias != '' else str(node.name[1])
                    #elif index.column() == 1:
                    #    return "{:.4f}".format(getattr(self.D[node.name[0]], node.name[1])[node.name[2]].min)
                    #elif index.column() == 2:
                    #    return "{:.4f}".format(getattr(self.D[node.name[0]], node.name[1])[node.name[2]].max)
                    elif index.column() > 0:
                        return index.column()
                        #return QtCore.Qt.Checked
                else:
                    if index.column() == 0:
                        return node.alias if node.alias != '' else str(node.name)
                    else: return None

                if index.column() > 2:
                        return index.column()

            return None

        def flags(self, index):

            if index.column() == 0:
                return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
            elif index.column() in [1, 2]:
                return QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
            else:
                return QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable

        def setData(self, index, value, role = QtCore.Qt.EditRole):
            """Stores changed data."""
            if index.isValid():

                if role == QtCore.Qt.EditRole:

                    node = index.internalPointer()
                    if node.nodeType == 'param':
                        # update parameter ranges
                        absRange = list(getattr(self.T[node.name[0]], node.name[1])[node.name[2]])
                        if index.column() in [1, 2]:
                            absRange[index.column()-1] = float(value)
                            getattr(self.T[node.name[0]], node.name[1])[node.name[2]] = parsSpec(*absRange)
                    else:
                        pass




                    return True
            return False

        #=====================================================#
        #INSERTING & REMOVING
        #=====================================================#

        """INPUTS: int, int, QModelIndex"""
        def insertRows(self, position, rows, parent=QtCore.QModelIndex()):

            parentNode = self.getNode(parent)

            self.beginInsertRows(parent, position, position + rows - 1)

            for row in range(rows):

                childCount = parentNode.childCount()
                childNode = Node("untitled" + str(childCount))
                success = parentNode.insertChild(position, childNode)

            self.endInsertRows()

            return success

        """INPUTS: int, int, QModelIndex"""
        def removeRows(self, position, rows, parent=QtCore.QModelIndex()):

            parentNode = self.getNode(parent)
            self.beginRemoveRows(parent, position, position + rows - 1)

            for row in range(rows):
                success = parentNode.removeChild(position)

            self.endRemoveRows()

            return success

        def insertColumns(self, pos=None, n_clm=1, parent = QtCore.QModelIndex()):

            if pos is None: pos = self.columnCount(parent)
            self.beginInsertColumns(parent, pos, pos+n_clm-1)

            self.steps.append('123')

            self.endInsertColumns()

    class PeakTreeView(QTreeView):

        def __init__(self, parent=None):
            super().__init__(parent)    # Initialize a QTreeWidget

            self.setAlternatingRowColors(True)
            self.setHeaderHidden(False)
            self.setColumnWidth(0, 150)

    def __init__(self, canvas, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self.ax = self.canvas.figure.get_axes()
        self.pthres = None
        self.drange = ()

        self.bttnAutoPick = QPushButton('Pick automatically')
        self.bttnAutoPick.clicked.connect(self.autoPick)
        self.bttnManuPick = QPushButton('Pick manually')
        self.bttnManuPick.clicked.connect(self.manuPick)
        self.bttnAssignPeaks = QPushButton('Assign')
        self.bttnAssignPeaks.clicked.connect(self.assignPeaks)
        self.sliderThresh = QSlider()
        self.sliderThresh.setRange(self.RANGE_MIN, self.RANGE_MAX)
        self.sliderThresh.setValue(0)
        self.sliderThresh.valueChanged.connect(self.onThreshChanged)
        self.sliderThresh.sliderPressed.connect(self.startPlotting)

        self.pickedTable = QTableWidget()
        layout = QGridLayout()
        layout.addWidget(QLabel("Peak picking"), 0, 0, 1, 3, Qt.AlignCenter|Qt.AlignTop)
        #layout.addWidget(self.pickedTable, 1, 0, 1, 2)
        layout.addWidget(self.bttnAutoPick, 3, 0, Qt.AlignHCenter)
        layout.addWidget(self.bttnManuPick, 3, 1, Qt.AlignHCenter)
        layout.addWidget(self.bttnAssignPeaks, 3, 2, Qt.AlignHCenter)
        layout.addWidget(self.sliderThresh, 1, 3, 3, 1, Qt.AlignHCenter)

        listView = QtGui.QListView()
        listView.setAcceptDrops(True);
        listView.setDragEnabled(True);
        listView.setDragDropMode(QtGui.QAbstractItemView.InternalMove);

        comboBox = QtGui.QComboBox()
        self.mdldPeaksTree = self.PeakTreeView()

        layout.addWidget(listView, 1, 0, 1, 2)
        layout.addWidget(self.mdldPeaksTree, 1, 2)
        #layout.addWidget(comboBox, 2, 0, 1, 3)

        red   = QtGui.QColor(255,0,0)
        green = QtGui.QColor(0,255,0)
        blue  = QtGui.QColor(0,0,255)

        rowCount = 4
        columnCount = 6

        headers = ["Pallete0", "Colors", "Brushes", "Omg", "Technical", "Artist"]
        tableData0 = [ [ QtGui.QColor("#FFFF00") for i in range(columnCount)] for j in range(rowCount)]

        model = self.PeakTableModel(tableData0, headers)
        #model.insertColumns(0, 5)

        #listView.setModel(model)
        #comboBox.setModel(model)
        #tableView.setModel(model)

        self.setMaximumWidth(500)
        self.setLayout(layout)

        self.reset()    # Sets the values for the current file/step

    def autoPick(self):
        """Automatic peak picking"""
        print("Automatic peak picking")
        f = stepClass.f
        yFph = stepClass.yFph
        pos, ids, width, amps = ng.analysis.peakpick.pick(yFph.real, pthres = self.pthres, algorithm='downward', cluster='False', table=False)      # , algorithm='connected'
        stepClass.pckdPeaks = [peakSpec(freq=f[p[0]], intn = a, fwhm=w[0] / np.pi) for p, w, a in zip(pos, width, amps) if w[0] > 0]
        self.startPlotting()

    def startPlotting(self):
        # remember the axis settings
        if settings["ax0Limits"] is not None:
            settings["ax0Limits"] = {"xlim":self.ax[0].get_xlim(), "ylim":self.ax[0].get_ylim()}
            indxPlot = np.flatnonzero((stepClass.f<=max(settings["ax0Limits"]["xlim"]))*(stepClass.f>=min(settings["ax0Limits"]["xlim"])))
            supsRatio = math.ceil(indxPlot.size / (2**13))   # Subsampling ratio; plot no more than 2^12 points
            indxPlot = np.append(indxPlot[:-1:supsRatio], indxPlot[-1])  # Make sure that the first and the last indices of each group are included
            self.f = stepClass.f[indxPlot]
            self.yFph = stepClass.yFph[indxPlot]
        else:
            self.f = stepClass.f
            self.yFph = stepClass.yFph

        self.plot()

    def plot(self):
        """Plots the phased spectrum and picked peaks on top of it."""
        self.ax[0].clear()     # discards the old graph
        self.ax[0].plot(self.f, self.yFph.real, '-', color=(0,0.58,0.86), linewidth=1.5, label='Measured data')

        # Plot the threshold
        self.ax[0].plot([self.f[0], self.f[-1]], [self.pthres]*2, '-', color='r')

        # Plot the peaks
        if len(stepClass.pckdPeaks) > 0:
            markerline, stemlines, baseline = self.ax[0].stem([peak.shft for peak in stepClass.pckdPeaks], [peak.intn/peak.fwhm/np.pi for peak in stepClass.pckdPeaks], basefmt=" ")     # , label=node.name if j==0 else ''
            plt.setp(stemlines, linewidth=1, color=(0.5, 0.5, 0.5), picker = 2)    # Picking tolerance in px
            plt.setp(markerline, markerfacecolor = (0.5, 0.5, 0.5), linestyle='None', color=(0.5, 0.5, 0.5), markersize=2)      # , picker=self.onStemPick

        self.ax[0].legend(loc=0)

        # Set the updated limits
        if settings["ax0Limits"] is not None:
            self.ax[0].set_xlim(settings["ax0Limits"]["xlim"])
            self.ax[0].set_ylim(settings["ax0Limits"]["ylim"])
        else:
            self.ax[0].relim()    # recompute the ax.dataLim
            self.ax[0].margins(0, 0.05)    # x and y margins in percentages
            self.ax[0].autoscale()    # update ax.viewLim using the new dataLim
            #self.ax[0].autoscale_view(tight=True, scalex=True, scaley=True)
            settings["ax0Limits"] = {"xlim":self.ax[0].get_xlim(), "ylim":self.ax[0].get_ylim()}
        self.ax[0].ticklabel_format(scilimits=(-3,3))
        self.ax[0].set_xlabel('Chemical shift, ppm', horizontalalignment='right', x=1.0)

        self.canvas.draw()    # refresh canvas
        return 0

    def plotPicked(self):
        settings["ax1Limits"] = {"xlim":self.ax[1].get_xlim(), "ylim":self.ax[1].get_ylim()}
        self.ax[1].clear()
        self.allStems = {}     # Dictionary that stores references to all stem lines

        markerline, stemlines, baseline = self.ax[1].stem([peak.shft for peak in stepClass.pckdPeaks], [peak.intn for peak in stepClass.pckdPeaks], basefmt=" ")     # , label=node.name if j==0 else ''
        plt.setp(stemlines, linewidth=1, color=(0.5, 0.5, 0.5), picker = 2)    # Picking tolerance in px
        plt.setp(markerline, markerfacecolor = (0.5, 0.5, 0.5), linestyle='None', color=(0.5, 0.5, 0.5), markersize=2)      # , picker=self.onStemPick

        self.ax[1].set_xlim(settings["ax1Limits"]["xlim"])     # Set the saved limits
        self.ax[1].set_ylim(settings["ax1Limits"]["ylim"])     # Set the saved limits
        self.ax[1].set_navigate(False)

        self.canvas.draw()

    def manuPick(self):
        """Manual peak picking"""
        print("Manual peak picking")
        pass

    def assignPeaks(self):
        """Assigns model peaks to one of the picked peaks."""
        #self.autoPick()
        steps[0].assign()
        self.assigned.emit()    # Tell the parent form to update the tree widget

    def onThreshChanged(self, val):
        """Reads new value from the slider sets the threshold, picks new peaks, and updates the plot."""
        pthres_rel = (val - self.RANGE_MIN) / (self.RANGE_MAX - self.RANGE_MIN)   # Relative range (0, 1)
        self.pthres = pthres_rel * (self.drange[1] - self.drange[0]) + self.drange[0]

        # Automatic peak picking
        f = stepClass.f
        yFph = stepClass.yFph
        pos, ids, width, amps = ng.analysis.peakpick.pick(yFph.real, pthres = self.pthres, algorithm='downward', cluster='False', table=False)      # , algorithm='connected'
        stepClass.pckdPeaks = [peakSpec(freq=f[p[0]], intn = a, fwhm=w[0] / np.pi) for p, w, a in zip(pos, width, amps) if w[0] > 0]

        self.plot()

    def reset(self):
        """Resets the sliders to display the phasing parameters for the currently open file/step."""
        self.sliderThresh.blockSignals(True)

        #self.mdldPeaksTree.setModel(self.PeakTreeModel())
        print(stepClass.mdldPeaks)

        # Range of displayed values
        if stepClass.yF is not None:
            self.drange = np.percentile(abs(stepClass.yF), (90, 99.99))
            self.pthres = self.drange[0] + 0.05 * (self.drange[1] - self.drange[0])

            # Set the slider
            pthres_rel = (self.pthres - self.drange[0]) / (self.drange[1] - self.drange[0])
            val = pthres_rel * (self.RANGE_MAX - self.RANGE_MIN) + self.RANGE_MIN
            self.sliderThresh.setValue(val)
        else:
            self.drange = self.ax[0].get_ylim()    # Display range
            self.pthres = None
            self.sliderThresh.setValue((self.RANGE_MAX - self.RANGE_MIN)/2)

        self.sliderThresh.blockSignals(False)

class PhasingWidget(QWidget):
    """A widget that contains scrollers/buttons for phasing and that interacts with a matplotlib canvas to plot the results."""

    RANGE_MAX = 64
    RANGE_MIN = -64
    sigPhasingProgress = pyqtSignal(object)
    sigPhasingComplete = pyqtSignal(float, float)          # Emmited when phasing is completed; outputs the values of ph0 and ph1 in degrees

    def __init__(self, orientation='Vertical', parent=None):

        super().__init__(parent)
        self.p0deg = 0.0     # Phasing parameters in degrees
        self.p1deg = 0.0
        self.pivot = 0.5     # Pivot point for phasing, float in the range (0.0, 1.0)

        self.sliderPh0, self.sliderPh1 = QSlider(), QSlider()
        self.sliderPh0.setRange(self.RANGE_MIN, self.RANGE_MAX)
        self.sliderPh1.setRange(self.RANGE_MIN, self.RANGE_MAX)
        self.sliderPh0.setValue(0)
        self.sliderPh1.setValue(0)
        self.sliderPh0.valueChanged.connect(self.onPh0SliderChanged)
        self.sliderPh1.valueChanged.connect(self.onPh1SliderChanged)
        self.sliderPh0.sliderReleased.connect(self.phasingComplete)
        self.sliderPh1.sliderReleased.connect(self.phasingComplete)
        self.bttnSetPivot = QPushButton('Pivot')
        self.bttnAutoPhase = QPushButton('Auto')
#        self.bttnAutoPhase.clicked.connect(self.autoPhase)
        layout = QGridLayout()
        if orientation == 'Vertical':
            layout.setVerticalSpacing(0)
            self.sliderPh0.setOrientation(Qt.Vertical)
            self.sliderPh1.setOrientation(Qt.Vertical)
            self.sliderPh0.setMinimumHeight(120)
            self.sliderPh1.setMinimumHeight(120)
            layout.addWidget(QLabel("Phasing"), 0, 0, 1, 2, Qt.AlignCenter|Qt.AlignTop)
            layout.addWidget(QLabel("ph0"), 1, 0, Qt.AlignCenter|Qt.AlignTop)
            layout.addWidget(QLabel("ph1"), 1, 1, Qt.AlignCenter|Qt.AlignTop)
            layout.addWidget(self.sliderPh0, 2, 0, Qt.AlignCenter|Qt.AlignTop)
            layout.addWidget(self.sliderPh1, 2, 1,  Qt.AlignCenter|Qt.AlignTop)
            #layout.addWidget(self.bttnSetPivot, 3, 0, 1, 2, Qt.AlignCenter)
            layout.addWidget(self.bttnAutoPhase, 4, 0, 1, 2, Qt.AlignCenter)
            self.setMaximumWidth(80)
            self.setMaximumHeight(250)
        else:
            self.sliderPh0.setOrientation(Qt.Horizontal)
            self.sliderPh1.setOrientation(Qt.Horizontal)
            self.sliderPh0.setMinimumWidth(120)
            self.sliderPh1.setMinimumWidth(120)
            layout.addWidget(QLabel("Phasing"), 0, 0, 1, 2, Qt.AlignCenter|Qt.AlignTop)
            layout.addWidget(QLabel("ph0"), 1, 0, Qt.AlignCenter)
            layout.addWidget(QLabel("ph1"), 2, 0, Qt.AlignCenter)
            layout.addWidget(self.sliderPh0, 1, 1, Qt.AlignCenter)
            layout.addWidget(self.sliderPh1, 2, 1,  Qt.AlignCenter)
            # layout.addWidget(self.bttnSetPivot, 2, 2, Qt.AlignCenter)
            # layout.addWidget(self.bttnAutoPhase, 1, 2, Qt.AlignCenter)
            self.setMaximumHeight(100)

        # Set event filters on the sliders to disallow scrolling with the mouse wheel
        class scrollEventFilter(QObject):

            def __init__(self, parent=None):
                super().__init__(parent)

            def eventFilter(self, source, event):
                if event.type() == QEvent.Wheel:
                    return True    # True will not propagate the event to the source widget

                return super().eventFilter(source, event)

        self.scrollEventFilterInstance = scrollEventFilter()
        self.sliderPh0.installEventFilter(self.scrollEventFilterInstance)
        self.sliderPh1.installEventFilter(self.scrollEventFilterInstance)

        self.setLayout(layout)

    def setData(self, f, yF, xF, freqBlocks, frange=None, pivot=None):
        """freBlocks is a list of tuples (min, max, bool), where the last position indicates whether the range is active (fitted) or not."""
        self.yF = yF
        self.xF = xF
        self._frange = (min(f), max(f)) if frange is None else frange
        self.f_norm = (f - self._frange[0]) / (self._frange[1] - self._frange[0])    # Frequency scale normalized to [0, 1]

        if pivot is not None:
            self.setPivot(pivot)

    def setPivot(self, val):
        """Sets a pivoting point for phase correction."""
        self.pivot = np.asscalar( (val - self._frange[0]) / (self._frange[1] - self._frange[0]) )

    def onPh0SliderChanged(self, val):
        """Reads new values from the sliders ph0 and ph1 and updates the plot"""
        ph0_rel = 2*(val - self.RANGE_MIN) / (self.RANGE_MAX - self.RANGE_MIN) - 1
        self.p0deg = ph0_rel * 5.0

        yFph = self.yF * np.exp(-1j*(self.p0deg + self.p1deg*self.f_norm)*np.pi/180)
        self.sigPhasingProgress.emit(yFph)

    def onPh1SliderChanged(self, val):
        """Reads new values from the sliders ph0 and ph1 and updates the plot"""
        ph1_rel = 2*(val - self.RANGE_MIN) / (self.RANGE_MAX - self.RANGE_MIN) - 1
        p1deg_new = ph1_rel * 30.0

        if self.pivot != 0.0:
            self.sliderPh0.blockSignals(True)
            self.p0deg += self.pivot*(self.p1deg - p1deg_new)     # find the new ph0 value given the current non-zero pivoting point
            self.p0deg = (self.p0deg + 180.0) % 360.0 - 180.0     # make sure the phase stays in the (-180.0, 180.0) interval
            self.sliderPh0.setValue((self.p0deg/180.0+1)*(self.RANGE_MAX - self.RANGE_MIN)/2 + self.RANGE_MIN)
            self.sliderPh0.blockSignals(False)
        self.p1deg = p1deg_new

        yFph = self.yF * np.exp(-1j*(self.p0deg + self.p1deg*self.f_norm)*np.pi/180)
        self.sigPhasingProgress.emit(yFph)

    # def reset(self):
    #     """Resets the sliders to display the phasing parameters for the currently open file/step."""
    #     try:
    #         theta = self.datum.getCrntVal(('.', 'theta', 0))
    #         tau = self.datum.getCrntVal(('.', 'tau', 0))
    #     except AttributeError:
    #         theta, tau = 0., 0.             # If the datum is the entire Workspace
    #
    #     self.pivot = 0.0
    #     self.p0deg = 180 * theta / np.pi
    #     try:
    #         self.p1deg = np.asscalar( tau*(self.datum.f[-1]*self.datum.c0-self.datum.f0)*360. )
    #     except (IndexError, AttributeError) as e:      # If self.datum.f == []
    #         self.p1deg = 0.
    #
    #     # Set the sliders
    #     self.sliderPh0.blockSignals(True)
    #     self.sliderPh1.blockSignals(True)
    #     self.sliderPh0.setValue((self.p0deg/180.0+1)*(self.RANGE_MAX - self.RANGE_MIN)/2 + self.RANGE_MIN)
    #     self.sliderPh1.setValue((self.p1deg/180.0+1)*(self.RANGE_MAX - self.RANGE_MIN)/2 + self.RANGE_MIN)
    #     self.sliderPh0.blockSignals(False)
    #     self.sliderPh1.blockSignals(False)

    # def autoPhase(self):
    #     """Autophasing"""
    #     p0, p1 = ng.process.proc_autophase.automatic_ps(self.datum.yF.ravel(), 'acme', p0=-self.p0deg, p1=-self.p1deg)     # 'peak_minima'
    #     self.p0deg, self.p1deg = -p0, -p1
    #     self.p0deg = (self.p0deg + 180.0) % 360.0 - 180.0     # make sure the phase stays in the (-180.0, 180.0) interval
    #     self.pivot = 0.0
    #     print(self.p0deg, self.p1deg)
    #
    #     theta, tau = self.deg2tau(self.p0deg, self.p1deg)
    #     print(theta, tau)
    #
    #     self.phasingComplete()

    def phasingComplete(self):
        # print("Phasing complete:", self.p0deg, self.p1deg)
        self.sigPhasingComplete.emit(self.p0deg, self.p1deg)

        returnToZero = True
        if returnToZero:
            self.sliderPh0.blockSignals(True)
            self.sliderPh0.setValue(0)
            self.sliderPh0.blockSignals(False)
            self.sliderPh1.blockSignals(True)
            self.sliderPh1.setValue(0)
            self.sliderPh1.blockSignals(False)

            self.yF = self.yF * np.exp(-1j*(self.p0deg + self.p1deg*self.f_norm)*np.pi/180)

            self.p0deg, self.p1deg = 0.0, 0.0

class PreprocessingWidget(QWidget):
    """Handles basic preprocessing operations, e.g. zero-filling and apodization."""

    parsChanged = pyqtSignal(object)          # A dictionary of changed parameters (keys: zff, apod, flagAdapFreq)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Add widgets for entering parameters
        self._resetting = False
        self.editNZF = QSpinBox()
        self.editNZF.setMinimum(0)
        self.editNZF.editingFinished.connect( lambda : self.onParsChanged( key='zff' ) )
        self.editApod = MyDoubleEdit()
        self.editApod.valueChanged.connect( lambda : self.onParsChanged( key='apod' ) )
        self.chkboxadaptiveFreqFlags = QCheckBox('Adaptive frequency scale')
        self.chkboxadaptiveFreqFlags.toggled.connect( lambda : self.onParsChanged( key='flagAdapFreq' ) )
        self.lbl_numTimePoints = QLabel('')
        self.lbl_numFreqPoints = QLabel('')
        self.lbl_numOptimRange = QLabel('')


        formLayout = QFormLayout()
        formLayout.addRow("Zero-filling factor", self.editNZF)
        formLayout.addRow("Line-broadening", self.editApod)
        formLayout.addRow(self.chkboxadaptiveFreqFlags)
        formLayout.addRow(self.lbl_numTimePoints)
        formLayout.addRow(self.lbl_numFreqPoints)
        formLayout.addRow(self.lbl_numOptimRange)
        self.setLayout(formLayout)

        self.setMaximumWidth(300)
        #self.setLayout(layout)

    def setNewDatum(self, dat=None):

        self._resetting = True

        if isinstance(dat, Datum):
            stats = dat.stats()
            self.editNZF.setValue(dat.zff)
            self.editApod.setValue(dat.apod)
            self.chkboxadaptiveFreqFlags.setChecked(dat.isAdapFreq())
            self.lbl_numTimePoints.setText('Original FID size: {:d}'.format(stats['nT']))
            self.lbl_numFreqPoints.setText('Spectrum size: {:d}'.format(stats['nF']) if stats['nF_adap'] == stats['nF']
                                            else 'Spectrum size: {:d} ({:d})'.format(stats['nF'], stats['nF_adap']))
            self.lbl_numOptimRange.setText('Optimization range: {:d} ({:.0f}%)'.format(stats['nF_opti'], 100*stats['nF_opti']/stats['nF']))
        else:
            self.editNZF.setValue(0)
            self.editApod.setValue(0)
            self.chkboxadaptiveFreqFlags.setChecked(False)
            self.lbl_numTimePoints.setText('')
            self.lbl_numFreqPoints.setText('')
            self.lbl_numOptimRange.setText('')

        self._resetting = False

    def onParsChanged(self, key):
        """Updates the settings (linebroadening, zero-filling, etc.)"""
        if not self._resetting:
            if key == 'zff':
                val = self.editNZF.value()
            elif key == 'apod':
                val = self.editApod.value()
            elif key == 'flagAdapFreq':
                val = self.chkboxadaptiveFreqFlags.isChecked()

            self.parsChanged.emit( {key : val} )

class ParameterDisplayWidget(QWidget):
    """A widget to display, modify, and sample parameters"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # ----------------- set up the histogram figure
        self.histFigure = Figure(facecolor='w', edgecolor='k')     # a figure instance to plot on
        self.histCanvas = FigureCanvas(self.histFigure)# this is the Canvas Widget that displays the `figure`; it takes the `figure` instance as a parameter to __init__
        self.axDistr = self.histFigure.add_subplot(111)    # create axes
        self.axDistr2 = self.axDistr.twinx()    # Separate vertical axis for a histogram plot

        # ----------------- create a table for sampling
        self.parsListWidget = QListWidget()
        #self.parsListWidget.itemClicked.connect(self.onSelectParList)

        # ------------ Set up a Block of Widgets for the results ---------------
        self.editMin = QLineEdit()
        #self.editMin.editingFinished.connect(self.saveParsForm)
        self.editMax = QLineEdit()
        #self.editMax.editingFinished.connect(self.saveParsForm)
        self.editCrntVal = QLineEdit()
        #self.editCrntVal.editingFinished.connect(self.saveParsForm)
        self.cmboxPrior = QComboBox()
        self.cmboxPrior.addItems(['Uniform', 'Gaussian', 'Log-Normal'])
        #self.cmboxPrior.activated.connect(self.onPriorNameChanged)
        self.editPriorMode = QLineEdit()
        #self.editPriorMode.editingFinished.connect(self.saveParsForm)
        self.editPriorStdv = QLineEdit()
        #self.editPriorStdv.editingFinished.connect(self.saveParsForm)
        self.chkboxUseCrnt = QCheckBox('Use crnt.')
        #self.chkboxUseCrnt.stateChanged.connect(self.saveParsForm)
        self.resForm1Layout = QFormLayout()
        self.resForm1Layout.addRow("Lower bnd.", self.editMin)
        self.resForm1Layout.addRow("Upper bnd.", self.editMax)
        self.resForm1Layout.addRow("Current val.", self.editCrntVal)
        self.resForm1Layout.addRow(" ", None)
        self.resForm1Layout.addRow("Prior dist.", self.cmboxPrior)
        self.resForm1Layout.addRow("Mode", self.editPriorMode)
        self.resForm1Layout.addRow(" ", self.chkboxUseCrnt)
        self.resForm1Layout.addRow("Deviation", self.editPriorStdv)

        self.bttnSample = QPushButton('Sample')
        self.bttnSample.clicked.connect(lambda : self.sampleStep(self.treeWidget.indxActvStep))
        layout = QGridLayout()
        layout.addWidget(self.parsListWidget, 0, 0, 2, 1)
        layout.addLayout(self.resForm1Layout, 0, 1)
        layout.addWidget(self.bttnSample, 1, 1)
        layout.addWidget(self.histCanvas, 0, 2, 2, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 2)

        self.setLayout(layout)

        self.reset()

    def reset(self):
        pass

class FittingThread(QThread):
    """A worker thread used to fit models to data in several steps."""

    def __init__(self):
        super().__init__()
        self._exiting = False      # The exiting attribute is used to tell the thread to stop processing.
        self._queueFiles = []        # Sequence of Datums to fit
        self._queueActns = []        # Sequence of Step IDs to fit

    def __del__(self):
        """Before a Worker object is destroyed, we need to ensure that it stops processing. For this reason, we implement the following method in a way that indicates to the part of the object that performs the processing that it must stop, and waits until it does so."""
        self._exiting = True
        self.wait()

    def setExitFlag(self, flag=True):
        """Setter of the _exiting flag."""
        self._exiting = flag

    def isExiting(self):
        return self._exiting

    def fit(self, fileToFit, actnToRun, evalStepID=-1):
        self._fileToFit = fileToFit
        self._actnToRun = actnToRun
        self._evalStep = self._fileToFit.steps[evalStepID]        # Step that will be evaluated

        self.start()    # calls self.run()   (should be called as self.start() anyway)

    def run(self):

        fileToFit = self._fileToFit
        actnToRun = self._actnToRun

        if isinstance(actnToRun, Step):
            # The action code is a Step
            actnToRun.run(fileToFit)         # Run the step on the given datum
        else:
            step = self._evalStep

            if actnToRun in ['Ph0', 'Ph1', 'PhX']:
                # Adjust the phasing parameters
                fileToFit.adjust_phase(frqBlkIds=step.frqBlkIds, mode=actnToRun)
                # Re-evaluate the step to update the (marginalized) amplitudes and the signals to be plotted
                fileToFit.evaluate(frqBlkIds=step.frqBlkIds, autoKeys = set([key for key in step.autoKeys if key != ('.', 'theta', 0)]), returnSignals=True)
            elif actnToRun == 'Rsd':
                # Adjusting the residual
                fileToFit.adjust_residual(frqBlkIds=step.frqBlkIds)
            elif actnToRun == 'Lsh':
                # Adjust the lineshape
                fileToFit.adjust_shape(frqBlkIds=step.frqBlkIds)
                # Re-evaluate the step to update the signals to be plotted
                fileToFit.evaluate(frqBlkIds=step.frqBlkIds, autoKeys=step.autoKeys, returnSignals=True)
            elif actnToRun == 'Lor':
                # Reset the lineshape to the default (Lorentzian)
                fileToFit.reset_shape()
                fileToFit.evaluate(frqBlkIds=step.frqBlkIds, autoKeys=step.autoKeys, returnSignals=True)
            elif actnToRun == 'Evl':
                # Only evaluatethe last step
                fileToFit.evaluate(frqBlkIds=step.frqBlkIds, autoKeys=step.autoKeys, returnSignals=True)
            elif actnToRun == 'PhA':
                # Autophasing
                fileToFit.auto_phase()
            elif actnToRun == 'PhA0':
                # Autophasing, only Ph0
                fileToFit.auto_phase(fit_Ph1=False)

class MySpecPlot(FigureCanvas):

    def __init__(self, figure):
        pass

class MainView(QMainWindow):
    """Main GUI form class."""

    def __init__(self, wsp, expiryTime = np.inf, parent = None):
        # Set the expiry time/date
        self._expiryTime = expiryTime

        # Show the warning window if the license is about to expire (less than 5 days left)
        if (self._expiryTime - time.time()) < 5*24*3600:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)

            msg.setText("The license expires on {:s}.".format(time.strftime('%B %d, %Y', time.gmtime(self._expiryTime))) )
            msg.setInformativeText("Please consider renewing it." )
            msg.setWindowTitle("Expiring license")
            # msg.setDetailedText("The details are as follows:")
            msg.setStandardButtons(QMessageBox.Close)
            msg.exec_()

        # Initialize with some workspace
        self.wsp = wsp    # The Workspace; main class that holds all logic
        self._crnt = self.wsp     # Currently opened Series/Datum/or the entire Workspace
        self._undoStack = []      # The stack of previous actions; each entry is a 2-tuple with the first element = the datum, second elent = flat dictionary of previous parameter values
        self._redoStack = []

        # Global settings
        settings.update({"ax0Limits": None,
                        "ax1Limits": {'ylim':(0, 5)},
                        "ax2Limits": None,
                        "startFromPars" : "current",          # Starting values of parameters when fitting multiple files (current, previous, default) - will be copied from crntParsH of this file, previous file or dfltParsH
                        "autoPhase" : False,
                        "autoPick" : False})

        # initialize the main window
        super(MainView, self).__init__(parent)
        self.resize(1400, 800)
        self.setupGUI()

        # Start the fitting thread
        self.fittingThread = FittingThread()
        self.fittingThread.finished.connect(self.onThreadFinished)
        self.fittingThread.terminated.connect(self.onThreadFinished)

        ## Install the custom output stream
        if compile_standalone:
            sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)
            sys.stderr = EmittingStream(textWritten=self.errorOutputWritten)

    def __del__(self):
        # Restore sys.stdout (if used to collect output to the console)
        if compile_standalone:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

    def __getattr__(self, attr):
        """Called with the dot notation for attributes not found in the class. Access the Workspace directly."""
        return getattr(self.wsp, attr)

    def normalOutputWritten(self, text):
        # Used to show text that was written to the console
        #self.statusBar.showMessage(text)
        cursor = self.printoutEdit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.printoutEdit.setTextCursor(cursor)
        self.printoutEdit.ensureCursorVisible()

    def errorOutputWritten(self, text):
        # Used to show text that was written to the console
        #self.statusBar.showMessage(text)
        cursor = self.printoutEdit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        #if err: cursor.insertText("------------ ERROR -------------")
        cursor.insertText(text)
        self.printoutEdit.setTextCursor(cursor)
        self.printoutEdit.ensureCursorVisible()

    def _icon(self, name):
        return QIcon(path.join('icons', name))

    def setupGUI(self):
        """Sets the layout for the main window."""
        self.setWindowTitle("Model-based quantitative NMR analysis ver. {} ({})".format(version, str(date.today())) )

        # ------------- set the main diagram --------------
        #rc('font', **{'family' : 'sans-serif', 'weight' : 'normal', 'size'   : 10})
        rcParams['pdf.fonttype'] = 42 # pdf.fonttype : 42 # Output Type 3 (Type3) or Type 42 (TrueType)
        rcParams['ps.fonttype'] = 42
        SMALL_SIZE = 11
        MEDIUM_SIZE = 12
        BIGGER_SIZE = 14

        rc('font', size=SMALL_SIZE)          # controls default text sizes
        rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        #rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        #rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        #rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        #rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        #rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        # ------------------------- Set up the toolbars ------------------------
        menubar = self.menuBar()
        tbMain = self.addToolBar("File")               # Main toolbar
        tbTree = QToolBar("Tree")                      # Tree toolbar
        tbTree.setIconSize(QtCore.QSize(18,18))
        tbTree.setFloatable(False)
        tbTree.setMovable(False)

        # ------------------- Printout for the console -------------------------
        self.printoutEdit = QPlainTextEdit()
        self.printoutEdit.setReadOnly(True)
        self.printoutEdit.resize(50, 50)

        # ----------------- set up the pie chart figure
        self.pieFigure = Figure(facecolor='w', edgecolor='k')     # a figure instance to plot on
        self.pieCanvas = FigureCanvas(self.pieFigure)# this is the Canvas Widget that displays the `figure`; it takes the `figure` instance as a parameter to __init__
        self.pieCanvas.setMaximumHeight(275)
        self.ax_pie = self.pieFigure.add_subplot(111)    # create axes

        # ------------------ 2. Set up the chemical tree ----------------------
        self.treeView = ChemTreeView()
        self.treeModel = ChemTreeModel(self.wsp)
        self.treeView.setModel(self.treeModel)
        self.treeModel.requestParameterChange.connect( self.onParameterChange )       # Updates the parameters and evaluates the active step if current values are changed by the user
        self.treeView.changedSelected.connect(self.selectStems)                       # If new parameter is selected by the user
        self.treeView.requestAdjustment.connect( lambda actnToRun : self.startThread(queueActns=[actnToRun]) )     # Adjust the phase

        # create a text edit widget to choose the optimization sequence
        self.stepsEdit = QPlainTextEdit('Please enter a sequence of steps to fit. All steps will be fitted consecutively by default.')  # , e.g.: 1, A, 5, (3, 4, A, 1), 2
        self.stepsEdit.setMaximumHeight(50)

        # ----------------- Spectrum figure in pyqtgraph -----------------------
        self.mainFigureWidget = MainSpectrumWidget()
        self.mainFigureWidget.sigFreqRangeSelected.connect(self.onFreqRangeSelected)
        self.mainFigureWidget.sigFreqRangeChanged.connect(self.onFreqRangeChanged)
        self.mainFigureWidget.sigMouseClicked.connect(self.onMouseClicked)
        self.mainFigureWidget.sigStemsClicked.connect(self.treeView.selectPickedParameter)
        self.mainFigureWidget.sigStemsDragged.connect(self.treeView.updateValue)
        self.mainFigureWidget.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))

        # Create the parameters tab
        tabParam = QWidget()
        layParam = QVBoxLayout()
        tabParam.setLayout(layParam)
        layParam.addWidget(self.treeView)
        layParam.addWidget(tbTree)
        layParam.addWidget(self.stepsEdit)


        # ----------------------------------------------------------------------
        #                            Navigation tab
        # ----------------------------------------------------------------------
        # Create the navigation tree
        self.naviTreeView = NavigationTreeView()
        self.naviTreeModel = NavigationTreeModel(self.wsp)
        self.naviSelection = QItemSelectionModel(self.naviTreeModel)
        self.naviTreeView.setModel(self.naviTreeModel)
        self.naviTreeView.setSelectionModel(self.naviSelection)
        # self.naviTreeDelegate = NavigationTreeDelegate()
        # self.naviTreeView.setItemDelegate(self.naviTreeDelegate)
        self.naviSelection.currentChanged.connect(self.onCurrentSelectedChanged)
        self.naviTreeView.requestPasteCrnt.connect(self.treeView.pasteCrntPars)     # Paste copied parameter values to all selected Datums in the naviTreeView
        self.naviTreeView.requestPasteDflt.connect(self.treeView.pasteDfltPars)
        self.naviTreeView.requestFitSelected.connect(self.fitAllSteps)

        # Set up the tab
        tabNavi = QWidget()
        layNavi = QVBoxLayout()
        tabNavi.setLayout(layNavi)
        layNavi.addWidget(self.naviTreeView)
        layNavi.addWidget(self.pieCanvas)
        layNavi.setContentsMargins(1,1,1,1)


        # ----------------------------------------------------------------------
        #                            Settings tab
        # ----------------------------------------------------------------------
        # A table to display frequency ranges
        self.freqTableView = FreqTableView()
        self.freqTableModel = FreqTableModel(self.wsp)
        self.freqTableView.setModel(self.freqTableModel)
        self.freqTableModel.freqBlockChanged.connect(self.updFreqBlock)

        # Add the preprocessing parameters tool
        self.preprocTool = PreprocessingWidget()
        self.preprocTool.parsChanged.connect(lambda x : self.resetSignals(**x))
        # self.phasingTool = PhasingWidget(self._crnt, self.canvas, orientation='Horizontal') # set to 'Horizontal' if displayed on the right
        # self.phasingTool.sigPhasingProgress.connect(self.onPhased)

        # Add Phasing tool
        self.mainPhasingWidget = PhasingWidget(orientation='Horizontal')

        self.mainFigureWidget.sigPivotDragged.connect(self.mainPhasingWidget.setPivot)
        self.mainPhasingWidget.sigPhasingProgress.connect(self.mainFigureWidget.replot_yF)
        self.mainPhasingWidget.sigPhasingComplete.connect(self.onPhased)

        # Set up the tab
        tabSettings = QWidget()
        laySettings = QVBoxLayout()
        tabSettings.setLayout(laySettings)
        laySettings.addWidget(self.freqTableView)
        laySettings.addWidget(self.mainPhasingWidget)
        laySettings.addWidget(self.preprocTool)

        # # ----------------------------------------------------------------------
        # #                            Phasing tab
        # # ----------------------------------------------------------------------
        # self.mainPhasingWidget = PhasingWidget(orientation='Horizontal')
        #
        # self.mainFigureWidget.sigPivotDragged.connect(self.mainPhasingWidget.setPivot)
        # self.mainPhasingWidget.sigPhasingProgress.connect(self.mainFigureWidget.replot_yF)
        # self.mainPhasingWidget.sigPhasingComplete.connect(self.onPhased)
        #
        # # Set up the tab
        # tabPhasing = QWidget()
        # layPhasing = QVBoxLayout()
        # tabPhasing.setLayout(layPhasing)
        # layPhasing.addWidget(self.mainPhasingWidget)
        # # layNavi.addWidget(self.pieCanvas)
        # # layNavi.setContentsMargins(1,1,1,1)

        # ---------------------------- Actions ---------------------------------
        self.setupActions(menubar, tbMain, tbTree)

        # --------------- Left --------------------
        widgetLeft = QTabWidget()
        widgetLeft.setMinimumSize(200, 0)
        widgetLeft.setMaximumWidth(250)
        # set up the tabs
        widgetLeft.setTabPosition(QTabWidget.North)
        widgetLeft.addTab(tabNavi, 'Navigation')
        widgetLeft.addTab(tabSettings, 'Settings')
        # widgetLeft.addTab(tabPhasing, 'Phasing')
        widgetLeft.setCurrentIndex(0)

        # -------------- Center -------------------
        # spectrum on top and console on the bottom
        widgetCenter = QSplitter()
        widgetCenter.setOrientation(Qt.Vertical)
        widgetCenter.addWidget(self.mainFigureWidget)
        widgetCenter.addWidget(self.printoutEdit)
        widgetCenter.setSizes([1000, 10])

        # --------------- Right -------------------
        widgetRight = QTabWidget()
        widgetRight.setMinimumSize(320, 0)
        widgetRight.setTabPosition(QTabWidget.North)
        widgetRight.addTab(tabParam, 'Parameters')

        # Put everything together in a QSplitter
        mainSplitter = QSplitter()
        mainSplitter.addWidget(widgetLeft)
        mainSplitter.addWidget(widgetCenter)
        mainSplitter.addWidget(widgetRight)
        mainSplitter.setStretchFactor(0, 1)
        mainSplitter.setStretchFactor(1, 3)
        mainSplitter.setStretchFactor(2, 1)
        # mainSplitter.setHandleWidth(1)

        # Set the result in the center of the form
        self.setCentralWidget(mainSplitter)

        # ---------------------- Set up the status bar -------------------------
        self.statusBar= QStatusBar()
        self.statusBar.setMaximumHeight(16)
        self.setStatusBar(self.statusBar)
        self.progressBarFiles = QProgressBar()
        self.progressBarFiles.setMaximumHeight(16)
        self.progressBarFiles.setMaximumWidth(400)
        self.statusBar.addPermanentWidget(self.progressBarFiles)

    def setupActions(self, menubar, tbMain, tbTree):
        # Add clear action
        clearAction = QAction(QIcon('icons\icon_new.png'), 'Clear workspace', self)
        clearAction.setStatusTip('Clear the workspace')
        clearAction.triggered.connect(lambda : self.onResetWspAction(newSettings=None, newWorkspace=None))
        # Add import datafile action
        actnImportData = QAction(self._icon('icon_addFile.png'), 'Import files', self)
        actnImportData.setStatusTip('Import new data and add them to the current series')
        actnImportData.triggered.connect(lambda : self.naviTreeView.onImportData(crnt_series=self._crnt))
        actnRemoveCurrent = QAction(self._icon('icon_removeFile.png'), 'Remove file', self)
        actnRemoveCurrent.setStatusTip('Remove file from the workspace')
        actnRemoveCurrent.triggered.connect(self.removeCurrent)
        showSettingsAction = QAction(self._icon('icon_settings.png'), 'Settings', self)
        showSettingsAction.setStatusTip('Show settings dialog')
        showSettingsAction.triggered.connect(self.showSettingsDialog)
        # Add load Tree action
        actnLoadTree = QAction(self._icon('icon_hierarchy.png'), 'Load new tree', self)
        actnLoadTree.setStatusTip('Load a new chemical tree')
        actnLoadTree.triggered.connect(self.loadChemTree)
        # Add load action
        loadAction = QAction(self._icon('icon_load.png'), 'Load workspace', self)
        loadAction.setStatusTip('Load the workspace')
        loadAction.triggered.connect(self.onLoadWspAction)
        # Add save action
        saveAction = QAction(self._icon('icon_save.png'), 'Save workspace', self)
        saveAction.setShortcut('Ctrl+S')
        saveAction.setStatusTip('Save the workspace')
        saveAction.triggered.connect(self.onSaveWspAction)
        # Add exit action
        exitAction = QAction(QIcon('icons\icon_exit.png'), 'Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.close)
        # Show the information dialog action
        actnAbout = QAction(QIcon('icons\icon_info.png'), 'About', self)
        actnAbout.setStatusTip('Information about the program')
        actnAbout.triggered.connect(self.showAboutMessage)

        # ----------------------- Actions for the plot -------------------------
        # Add freqBlock action
        self.actnAddFreqBlock = QAction(self._icon('add_range.png'), 'Add optimization range', self)
        self.actnAddFreqBlock.setShortcut('Ctrl+A')
        self.actnAddFreqBlock.setStatusTip('Add optimization range')
        self.actnAddFreqBlock.setCheckable(True)
        # Remove freqBlock action
        self.actnRemoveFreqBlock = QAction(self._icon('remove_range.png'), 'Remove optimization range', self)
        self.actnRemoveFreqBlock.setShortcut('Ctrl+D')
        self.actnRemoveFreqBlock.setStatusTip('Remove optimization range')
        self.actnRemoveFreqBlock.setCheckable(True)
        # Modify freqBlocks action
        self.actnChangeFreqBlock = QAction(self._icon('modify_range.png'), 'Modify optimization ranges', self)
        self.actnChangeFreqBlock.setStatusTip('Modify optimization ranges')
        self.actnChangeFreqBlock.setCheckable(True)
        # Add all frequency ranges actions to the group
        self.actnGroupFreqBlocks = QActionGroup(self)
        self.actnGroupFreqBlocks.addAction(self.actnAddFreqBlock)
        self.actnGroupFreqBlocks.addAction(self.actnRemoveFreqBlock)
        self.actnGroupFreqBlocks.addAction(self.actnChangeFreqBlock)
        self.actnGroupFreqBlocks.triggered.connect(self.onFreqRangeControl)
        self.actnGroupFreqBlocks._previuosAction = None          # Indicator of the previously selected action

        # Show stems
        self.actnToggleStems = QAction(self._icon('icon_showStems.png'), 'Show transition lines', self)
        self.actnToggleStems.setStatusTip('Show transition lines')
        self.actnToggleStems.setCheckable(True)
        self.actnToggleStems.triggered.connect(self.toggleStems)
        # Show components
        self.actnToggleComps = QAction(self._icon('icon_showComponents.png'), 'Show fitted components', self)
        self.actnToggleComps.setStatusTip('Show fitted components')
        self.actnToggleComps.setCheckable(True)
        self.actnToggleComps.triggered.connect(self.toggleComps)
        # Show residual
        self.actnToggleResid = QAction(self._icon('icon_plotResidual.png'), 'Show residual', self)
        self.actnToggleResid.setStatusTip('Show residual')
        self.actnToggleResid.setCheckable(True)
        self.actnToggleResid.triggered.connect(self.toggleResid)
        # Autoscale
        self.actnAutoRange = QAction(self._icon('icon_autoScale.png'), 'Autoscale', self)
        self.actnAutoRange.setStatusTip('Scale plot to data')
        self.actnAutoRange.triggered.connect(self.mainFigureWidget.autoRange)
        # Save Image
        self.actnSaveImage = QAction(self._icon('icon_saveImage.png'), 'Save image', self)
        self.actnSaveImage.setStatusTip('Saves the spectrum as image')
        self.actnSaveImage.triggered.connect(self.mainFigureWidget.saveImage)

        # ------------------------ Processing Actions --------------------------
        # Autophase
        self.actnAutoPhase = QAction(self._icon('icon_autoPhase.png'), 'Autophase', self)
        self.actnAutoPhase.setStatusTip('Apply a phase correction algorithm')
        self.actnAutoPhase.triggered.connect( lambda : self.startThread(queueActns=['PhA']) )
        # Set custom lineshape
        self.actnSetLshape = QAction(self._icon('icon_customShape.png'), 'Set custom lineshape', self)
        self.actnSetLshape.setStatusTip('Apply a custom lineshape derived by deconvolution')
        self.actnSetLshape.triggered.connect( lambda _ : self.startThread(queueActns=['Lsh']) )
        self.actnResetLshape = QAction(self._icon('icon_resetShape.png'), 'Reset the lineshape', self)
        self.actnResetLshape.setStatusTip('Remove the custom lineshape - reset to default (Lorentzian)')
        self.actnResetLshape.triggered.connect( lambda _ : self.startThread(queueActns=['Lor']) )

        # ----------------------- Actions for the tree -------------------------
        # Add step
        actnAddStep = QAction(self._icon('icon_addStep.png'), 'Add step', self)
        actnAddStep.setStatusTip('Add an optimization step')
        actnAddStep.triggered.connect(self.treeModel.addStep)
        # Delete step
        actnDelStep = QAction(self._icon('icon_delStep.png'), 'Remove active step', self)
        actnDelStep.setStatusTip('Remove an optimization step')
        actnDelStep.triggered.connect(self.treeModel.delStep)
        # Copy All parameters
        actnCopyAllPars = QAction(self._icon('icon_copy.png'), 'Copy all parameters', self)
        actnCopyAllPars.setStatusTip('Copy current values of all parameters')
        actnCopyAllPars.triggered.connect(lambda : self.treeView.copyCrntPars(keys=None))
        # Paste copied parameters
        actnPastePars = QAction(self._icon('icon_pasteCrnt.png'), 'Paste parameters', self)
        actnPastePars.setStatusTip('Paste values of copied parameters as current')
        actnPastePars.triggered.connect(lambda : self.treeView.pasteCrntPars(datums=None))
        # Toggle LS/TLS
        self.actnToggleTLS = QAction(self._icon('icon_TLS.png'), 'Use TLS algorithm', self)
        self.actnToggleTLS.setStatusTip('Use the Total Least Squares algorithm')
        self.actnToggleTLS.setCheckable(True)
        def onToggleTLS():
            config.SAMPL_funcType = 'TLS' if self.actnToggleTLS.isChecked() else 'LS'
            self.treeModel.notifyDataChanged()     # Update the TLS line in the chemTree view
        self.actnToggleTLS.triggered.connect(onToggleTLS)
        # Fit the last step action
        self.actnFitLastStep = QAction(self._icon('icon_fitOneStep.png'), 'Fit active step', self)
        self.actnFitLastStep.setStatusTip('Fit the last step')
        self.actnFitLastStep.triggered.connect( lambda : self.startThread(queueActns=['Fit']) )
        # Sample action
        self.actnSample = QAction(self._icon('icon_sample.png'), 'Sample last step with MCMC', self)
        self.actnSample.setStatusTip('Sample parameters checked on the last step with the MCMC algorithm')
        self.actnSample.triggered.connect( self.sampleStep )
        # Report without sampling action
        self.actnReport = QAction(self._icon('icon_report.png'), 'Report results', self)
        self.actnReport.setStatusTip('Report results without MCMC sampling')
        self.actnReport.triggered.connect(lambda:self.sampleStep(onlyAutoKeys=True))
        # Fit all steps action
        self.actnFitAllSteps = QAction(self._icon('icon_fitAllSteps.png'), 'Fit all steps', self)
        self.actnFitAllSteps.setStatusTip('Fit all steps for this file')
        self.actnFitAllSteps.triggered.connect(lambda _ : self.fitAllSteps(selectedFiles = None))
        # Fit all files action
        self.actnFitAllFiles = QAction(self._icon('icon_fitAllFiles.png'), 'Fit all files', self)
        self.actnFitAllFiles.setStatusTip('Fit all steps for this file')
        self.actnFitAllFiles.triggered.connect(self.fitAllFiles)
        # Undo
        self.actnUndo = QAction(self._icon('icon_undo.png'), 'Undo', self)
        self.actnUndo.setStatusTip('Undo the previous change to model parameters')
        self.actnUndo.triggered.connect(self.pullUndo)
        # Redo
        self.actnRedo = QAction(self._icon('icon_redo.png'), 'Return', self)
        self.actnRedo.setStatusTip('Return the undone change to model parameters')
        self.actnRedo.triggered.connect(self.pullRedo)
        # Stop fitting action
        actnstopThread = QAction(self._icon('icon_stopFitting.png'), 'Stop fitting', self)
        actnstopThread.setStatusTip('Stop fitting')
        actnstopThread.triggered.connect(self.stopThread)
        # Phase correction actions
        """self.actnCorrectPh0 = QAction(self._icon('icon_correctPh0.png'), 'Correct ph0', self)
        self.actnCorrectPh0.setStatusTip('Correct zero-order phasing.')
        self.actnCorrectPh0.triggered.connect(self.correct_phase)"""
        # Save current results
        actnSaveResults = QAction(self._icon('icon_saveResults.png'), 'Save results to file', self)
        actnSaveResults.setStatusTip('Save all current results to file')
        actnSaveResults.triggered.connect(self.saveResults)

        # ------------------------- set the menubar ----------------------------
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(actnImportData)
        fileMenu.addAction(actnRemoveCurrent)
        fileMenu.addAction(loadAction)
        fileMenu.addAction(saveAction)
        fileMenu.addAction(exitAction)
        helpMenu = menubar.addMenu('&Help')
        helpMenu.addAction(actnAbout)

        # ------------------------- Set the toolbar ----------------------------
        tbMain.addAction(clearAction)
        tbMain.addAction(actnImportData)
        tbMain.addAction(actnRemoveCurrent)
        tbMain.addAction(showSettingsAction)
        tbMain.addSeparator()
        tbMain.addAction(loadAction)
        tbMain.addAction(saveAction)
        tbMain.addSeparator()
        tbMain.addAction(self.actnAddFreqBlock)
        tbMain.addAction(self.actnRemoveFreqBlock)
        tbMain.addAction(self.actnChangeFreqBlock)
        tbMain.addAction(self.actnToggleStems)
        tbMain.addAction(self.actnToggleComps)
        tbMain.addAction(self.actnToggleResid)
        tbMain.addAction(self.actnAutoRange)
        # tbMain.addAction(self.actnSaveImage)
        tbMain.addSeparator()
        tbMain.addAction(self.actnUndo)
        tbMain.addAction(self.actnRedo)
        tbMain.addSeparator()
        tbMain.addAction(self.actnAutoPhase)
        tbMain.addAction(self.actnSetLshape)
        tbMain.addAction(self.actnResetLshape)
        tbMain.addSeparator()
        tbMain.addAction(self.actnFitLastStep)
        tbMain.addAction(self.actnFitAllSteps)
        tbMain.addAction(self.actnFitAllFiles)
        tbMain.addSeparator()
        # tbMain.addAction(self.actnSample)
        tbMain.addAction(self.actnReport)
        tbMain.addAction(actnSaveResults)

        # ---------------------- Toolbar for the tree --------------------------
        tbTree.addAction(actnLoadTree)
        tbTree.addAction(actnAddStep)
        tbTree.addAction(actnDelStep)
        tbTree.addSeparator()
        tbTree.addAction(actnCopyAllPars)
        tbTree.addAction(actnPastePars)
        tbTree.addSeparator()
        # tbTree.addAction(self.actnToggleTLS)
        tbTree.addAction(self.actnFitLastStep)
        tbTree.addAction(self.actnFitAllSteps)
        tbTree.addAction(self.actnFitAllFiles)
        tbTree.addAction(actnstopThread)
        # tbTree.addAction(self.actnSample)

    # ----------------- Signals from the main Spectrum Widget ------------------

    def onFreqRangeSelected(self, lims):
        # print('Selected a frequency block', lims)
        self.actnAddFreqBlock.setChecked(False)           # Make sure that the buton is not pressed
        self.mainFigureWidget.setSelectorFlag(False)      # Exit the selection mode
        self.addFreqBlock(*lims)                          # Add new Frequency block

        self.actnGroupFreqBlocks._previuosAction = None

    def onFreqRangeChanged(self, indx, lims):
        # print('Frequency block has changed', indx, lims)
        self.actnChangeFreqBlock.setChecked(False)
        self.mainFigureWidget.setCursor(mode='normal')
        self.updFreqBlock(indx, lims)

        self.actnGroupFreqBlocks._previuosAction = None

    def onMouseClicked(self, pos):
        # print('Mouse clicked', pos)
        # Process the mouse click differently according to the current state
        if self.actnRemoveFreqBlock.isChecked():    # Remove freqBlock request
            self.mainFigureWidget.setCursor(mode='normal')
            # Find which block was clicked on (the narrowest block that covers the clicked position)
            scores = [blk.max-blk.min if blk.min < pos.x() < blk.max else np.inf for blk in self._crnt.freqBlocks]
            indx = min(enumerate(scores), key=itemgetter(1))[0]      # Find the index of the minimum
            if indx > 0:    # Don't remove the allFrequencies block
                self.actnRemoveFreqBlock.setChecked(False)     # Unpress the button
                self.remFreqBlock(indx)

            self.actnGroupFreqBlocks._previuosAction = None

    def onFreqRangeControl(self, actn):
        """Called when the action group of freqRange controls is triggered"""
        if self.actnGroupFreqBlocks._previuosAction == actn:
            # Turn off the buttons
            actn.setChecked(False)
            self.mainFigureWidget.setMovableFreqBlocks(False)
            self.mainFigureWidget.setSelectorFlag(False)
            self.mainFigureWidget.setCursor(mode='normal')
            self.actnGroupFreqBlocks._previuosAction = None
            return 0

        elif self.actnChangeFreqBlock.isChecked():
            # Changing frequency blocks
            self.mainFigureWidget.setMovableFreqBlocks(True)
            self.mainFigureWidget.setSelectorFlag(False)
            self.mainFigureWidget.setCursor(mode='hand')

        elif self.actnAddFreqBlock.isChecked():
            # Adding new frequency block
            self.mainFigureWidget.setSelectorFlag(True)
            self.mainFigureWidget.setMovableFreqBlocks(False)
            self.mainFigureWidget.setCursor(mode='hand')

        elif self.actnRemoveFreqBlock.isChecked():
            # Removing a frequency block
            self.mainFigureWidget.setMovableFreqBlocks(False)
            self.mainFigureWidget.setSelectorFlag(False)
            self.mainFigureWidget.setCursor(mode='hand')

        self.actnGroupFreqBlocks._previuosAction = actn

    # -------------------- Processing keyboard interactions --------------------

    def keyPressEvent(self, ev):
        # self.scene().keyPressEvent(ev)
        # self.sigKeyPress.emit(ev)
        # print('Key pressed ', ev.key())
        pass

    # ------------------------- Other utility methods --------------------------

    def removeCurrent(self):
        """Removes current selection (Series or Datum)."""
        if self._crnt != self.wsp:
            self.naviTreeModel.remItems([self._crnt])
            self.naviSelection.clear()    # Select the entire workspace

    def setCurrent(self, newCrnt=None):
        """Sets the _crnt Datum and updates the plot, tables, etc. accordingly."""

        # Set self._crnt to the new Series/Datum or the entire workspace
        if newCrnt is None:
            try:
                newCrnt = self.wsp.series[0].data[0]
            except IndexError:
                self._crnt = self.wsp
                return None

        # Compute the model signal if there is None
        try:
            if newCrnt.zF is None:
                step = newCrnt.steps[-1]
                newCrnt.evaluate(frqBlkIds=step.frqBlkIds, autoKeys=step.autoKeys, returnSignals=True)
                #TODO: Possibly check which step to evaluate if steps use different frequency ranges
        except AttributeError:
            pass

        # Update the necessary display widgets
        self.treeModel.setNewDatum(newCrnt)
        self.freqTableModel.setNewDatum(newCrnt)
        self.preprocTool.setNewDatum(newCrnt)
        # self.phasingTool.setNewDatum(newCrnt)

        # Highlight the current Datum in the Navigation widget if it was seletected programmatically
        self.naviTreeView.selectCurrentDatum(newCrnt.selfID())

        if isinstance(self._crnt, Datum) and isinstance(newCrnt, Datum) and self._crnt.parent == newCrnt.parent:
            resetView = False
        else: resetView = True
        self._crnt = newCrnt
        self.plotCurrent(autoRange=resetView)

    def onCurrentSelectedChanged(self, index):
        """Slot for the signal indicating the change in the currently selected Series/Datum"""

        self.setCurrent(newCrnt = index.internalPointer() if index.isValid() else None)

    def loadChemTree(self):
        """Calls a dialog and loads a new chemical tree from file."""
        filename = QFileDialog.getOpenFileName(self, 'Import parameter tree', '.', filter = "Chemical trees (*.ctr)")

        T = loadTree(filename)

        self.treeModel.resetChemTree(T)

    def onSaveWspAction(self):
        """Saves the workspace including the stepClass class and the steps array."""
        filename = QFileDialog.getSaveFileName(parent=self, caption='Select output file', directory='.', filter='NMR worksapce (*.wsp)')
        if filename:
            if filename[-4:] != '.wsp': filename += '.wsp'

            # Pack the logic of the workspace
            dataPack = self.wsp.pack()

            # Pack the settings
            stngPack = copy.deepcopy(settings)
            stngView = {'stepsEditText' : self.stepsEdit.toPlainText(),\
                        'hiddenTreeViewNodes' : [node.name for node in self.treeModel.TP.items() if node.hidden]}
            stngConfig = config.as_dict()

            stngPack['_view'] = stngView
            stngPack['_config'] = stngConfig

            with open(filename, 'wb') as fp:
                dill.dump([dataPack, stngPack], fp)

    def onLoadWspAction(self):
        """Loads the workspace including the stepClass class and the steps array."""
        filename = QFileDialog.getOpenFileName(self, 'Open workspace', '.', filter = "NMR worksapce (*.wsp)")
        if filename:
            with open(filename, 'rb') as fp:
                dataUnPack = dill.load(fp)

            # Reset the settings and the Workspace
            self.onResetWspAction(newWorkspace=dataUnPack[0], newSettings=dataUnPack[1])

    def onResetWspAction(self, newWorkspace=None, newSettings=None):
        """Clears the workspace including the stepClass, signals and the tree."""
        self.naviTreeModel.beginResetModel()

        # Update the global settings
        settings.clear()
        settings.update({"ax0Limits": None,
                        "ax1Limits": {'ylim':(0, 5)},
                        "ax2Limits": None,
                        "startFromPars" : "current",          # Starting values of parameters when fitting multiple files (current, previous, default) - will be copied from crntParsH of this file, previous file or dfltParsH
                        "autoPhase" : False,
                        "autoPick" : False})
        if newSettings is not None:
            try:
                stngView = newSettings.pop('_view')
            except KeyError:
                stngView = None
            try:
                stngConfig = newSettings.pop('_config')
            except KeyError:
                stngConfig = None
            settings.update(newSettings)
        else:
            settings.update({"HCmode":"1H"})
            stngView, stngConfig = None, None
        settings.update({"ax0Limits":None, "ax1Limits":None, "ax2Limits":None})

        # Reset the undo and redo stacks
        self._undoStack.clear()
        self.actnUndo.setEnabled(False)
        self._redoStack.clear()
        self.actnRedo.setEnabled(False)

        # Reset the workspace
        self.wsp.reset()
        if newWorkspace is not None:
            self.wsp.unpack(newWorkspace)

        # Reset the tree model
        self.treeModel.fullReset(self._crnt)

        self.naviTreeModel.endResetModel()

        self.setCurrent()          # Sets the current display to the first Datum or the entire workspace if there is no Datum

        # Update the views (e.g. shown/hidden rows, etc.)
        if stngView is not None:
            self.stepsEdit.setPlainText(stngView['stepsEditText'])
            for node in stngView['hiddenTreeViewNodes']:
                self.treeModel.TP[node].hidden = True
            self.treeView.hideExcessiveRows()

        # Update the global config
        if stngConfig is not None:
            config.from_dict(config, stngConfig)
        self.actnToggleTLS.setChecked(config.SAMPL_funcType == 'TLS')

    def resetSignals(self, zff=None, apod=None, flagAdapFreq=None):
        self._crnt.resetFreqs(zff, apod)

        if flagAdapFreq is not None:
            for dat in self._crnt.parent.data:
                dat.resetSignals(flagAdapFreq)           # Or simply dat.resetSignals(flagAdapFreq) to reset the adaptive flag for a single (current) Datum only

        self.preprocTool.setNewDatum(self._crnt)         # Update the statistics display
        self.plotCurrent(autoRange=(apod is None))       # Do not autorange if what has changed is only apodization

    def showSettingsDialog(self):
        """Shows an input dialog and updates settings"""
        accepted, newSettings = SettingsDialog.run(oldSettings={'HCmode': self.wsp.HCmode})

        if accepted:
            self.treeModel.setHCmode(newSettings['HCmode'])

    def showAboutMessage(self):
        """Displays the About message."""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)

        msg.setText("Quantitative NMR analysis with quantum mechanical models.")
        msg.setInformativeText("The license expires on {:s}.".format(time.strftime('%B %d, %Y', time.gmtime(self._expiryTime))) )
        msg.setWindowTitle("About qNMR")
        # msg.setDetailedText("The details are as follows:")
        msg.setStandardButtons(QMessageBox.Close)

        msg.exec_()            # Returns the values of pressed button

    def onAssigned(self):
        """Updates the tree and plots when model peaks have been assigned to picked peaks."""
        self.plotCurrent(autoRange=False)
        self.treeWidget.showStep()
        #self.phasingTool.setVals(self.steps[0])
        #dataFiles[self.indxCrntFile].update()
        pass

    def pullUndo(self):
        """Restores the last state from the Undo stack."""
        dats_list, pars_list = self._undoStack.pop()       # Outputs list of Datums and list of parsF dictionaries
        self._redoStack.append([dats_list, [dat.getCrntVals(keys=list(par.keys())) for dat, par in zip(dats_list, pars_list)]])

        # Set the stored values to the Datums
        for dat, par in zip(dats_list, pars_list):
            dat.setCrntVals(par)

        self.startThread(queueFiles=dats_list, pushUndo=False)   # This will also update the undo/redo buttons

    def pullRedo(self):
        """Restores the last state from the Redo stack."""
        dats_list, pars_list = self._redoStack.pop()       # Outputs list of Datums and list of parsF dictionaries
        self._undoStack.append([dats_list, [dat.getCrntVals(keys=list(par.keys())) for dat, par in zip(dats_list, pars_list)]])

        # Set the stored values to the Datums
        for dat, par in zip(dats_list, pars_list):
            dat.setCrntVals(par)

        self.startThread(queueFiles=dats_list, pushUndo=False)   # This will also update the undo/redo buttons

    # ------------------ Working with the fitting thread -----------------------

    def startThread(self, queueFiles=None, queueActns=None, pushUndo=True):
        """Fits the files in the queueFiles list."""
        # Disable controls that can start fitting
        self.actnFitAllSteps.setDisabled(True)
        self.actnFitAllFiles.setDisabled(True)
        self.actnFitLastStep.setDisabled(True)
        self.actnUndo.setEnabled(False)
        self.actnRedo.setEnabled(False)
        self.setCursor(Qt.BusyCursor)

        if queueFiles is None: queueFiles = [self._crnt]
        if pushUndo:
            self._undoStack.append([queueFiles, [dat.getCrntVals() for dat in queueFiles]])
            self._redoStack.clear()

        if queueActns is None: queueActns = ['Evl']        # Only evaluate the active step by default

        queueActns = [self.treeModel.actvStepIndx if x == 'Fit' else x for x in queueActns]
        if len(queueFiles) > 1: queueActns.insert(0, 'Init')

        # Set up the fitting queue
        self._fittingQueue = [[file, actn] for file in queueFiles for actn in queueActns]

        # Set up the progress bars
        self.progressBarFiles.setRange(0, len(self._fittingQueue))
        self.progressBarFiles.setValue(0)

        self.fittingThread.setExitFlag(False)

        self.continueThread()

    def continueThread(self):
        """Continues fitting the thread if there are any files left."""
        if len(self._fittingQueue) > 0:
            fileToFit, actnToRun = self._fittingQueue.pop(0)

            if actnToRun == 'Init':
                # We are starting to fit a new file and will be running through the list of steps from 0 again... Need to set up the initial values.
                # Determine the starting values of parameters for the next file in the fittingQueueFiles and KEEP the current values if necessary
                if config.OPTIM_startFrom == "previous":
                    sid = fileToFit.selfID()
                    # Check if the current file is not the first one in the Series. If possible use parameters of the previous file, otherwise keep the current parameters.
                    if sid[1] > 0:
                        fileToFit.resetCrntPars(crntParsH = copy.deepcopy(fileToFit.series[sid[0]].data[sid[1]-1].crntParsH) )
                elif config.OPTIM_startFrom == "default":
                    fileToFit.resetCrntPars()   # Reset to defaults
                else: # i.e. settings["startgFromPars"] == "current"
                    pass     # Don't do anything; the file will be loaded with its current parameters, and the optimization will start from them
                self.progressBarFiles.setValue(self.progressBarFiles.value()+1)

                self.continueThread()

            elif isinstance(actnToRun, int):
                # If it is a step with too many parameters, split it into several steps and add them to the queue
                print("\nOptimizing step {:d}".format(actnToRun+1))
                actnToRun = fileToFit.steps[actnToRun]

                if len(actnToRun.parsKeys) > 3:
                    newSteps = split_steps(fileToFit, actnToRun)
                    self._fittingQueue[:0] = [[fileToFit, step] for step in newSteps]         # Insert all new steps from the beginning of the queue

                    # Update the progress bar accordingly
                    oldVal, oldMax = self.progressBarFiles.value(), self.progressBarFiles.maximum()
                    if oldMax != oldVal:
                        newVal = oldVal*int(math.ceil((oldMax-oldVal+len(newSteps)-1)/(oldMax-oldVal)))           # newVal = int(math.ceil(oldVal*(oldMax+len(newSteps)-1)/oldMax))
                        newMax = newVal + (oldMax-oldVal) + len(newSteps) - 1

                        self.progressBarFiles.setMaximum(newMax)
                        self.progressBarFiles.setValue(newVal)
                else:
                    # TODO: Possibly check here that the parsKeys in the step are fittable
                    self._fittingQueue.insert(0, [fileToFit, actnToRun] )          # Substitute the integer with the step

                self.continueThread()

            else:
                # It is a usual optimization step or an action (e.g. phasing)
                self.fittingThread.fit(fileToFit, actnToRun)

        else:
            # All done. Reset the widgets
            # This will be executed always when the thread is finished, either normally or by termination.
            self.plotCurrent(autoRange=False)
            self.treeModel.notifyDataChanged()

            # Enable controls that can start fitting again
            self.actnFitAllSteps.setEnabled(True)
            self.actnFitAllFiles.setEnabled(True)
            self.actnFitLastStep.setEnabled(True)
            self.actnUndo.setEnabled(len(self._undoStack) > 0)
            self.actnRedo.setEnabled(len(self._redoStack) > 0)
            self.unsetCursor()
            self.progressBarFiles.setValue(self.progressBarFiles.maximum())

    def onThreadFinished(self):
        """Called when the fittingThread finishes processing each step. Depending if there are files/steps in queue, may call the startThread/fitqueueActns function again or just display the results."""
        self.progressBarFiles.setValue(self.progressBarFiles.value()+1)

        if self.fittingThread.isExiting(): self._fittingQueue.clear()

        self.continueThread()

    def stopThread(self):
        """Stops fitting in the thread."""
        self.fittingThread.setExitFlag(True)
        self.fittingThread.quit()

    def sampleStep(self, indx=None, onlyAutoKeys=False):
        if indx is None: indx = self.treeModel.actvStepIndx        # Fit the active step by default
        step = self._crnt.steps[indx]
        samples = self._crnt.sample(frqBlkIds=step.frqBlkIds, parsKeys=None if onlyAutoKeys else step.parsKeys, autoKeys=step.autoKeys, evaluatePriors=True, nwalkers=None, nsteps=250)     # parsKeys=step.parsKeys
        reportMCMC(samples)

    def onParameterChange(self, key, val):
        """Sets a new value to the parameter key."""

        oldVal = self._crnt.getCrntVal(key)

        if not np.isclose(val, oldVal):

            self._undoStack.append([ [self._crnt], [{key : oldVal}] ])
            self._redoStack.clear()
            self._crnt.setCrntVal(key, val)

        self.startThread(pushUndo=False)

    def onPhased(self, p0deg, p1deg):
        """Gets the phasing values from the phasing tool widget and sets current parameters accordingly."""
        nf = next_pow_of_2( 2**self._crnt.zff * len(self._crnt.t) )     # Determine the number of samples in the FULL signal spectrum (possibly including zero-filling). zff and t are taken from the Series level
        dt = self._crnt.t[1]-self._crnt.t[0]

        d_theta, d_tau = deg2tau(dt, nf, p0deg, p1deg)
        if not ( np.isclose(d_theta, 0) and np.isclose(d_tau, 0) ):
            theta = self._crnt.getCrntVal(key=('.', 'theta', 0))
            tau = self._crnt.getCrntVal(key=('.', 'tau', 0))

            self._undoStack.append([ [self._crnt], [{('.', 'theta', 0):theta, ('.', 'tau', 0):tau}] ])
            self._redoStack.clear()
            self._crnt.setCrntVal(key=('.', 'theta', 0), val = (theta+d_theta + np.pi) % np.pi - np.pi )     # make sure the phase stays in the (-180.0, 180.0) interval  # p0deg = (p0deg + 180.0) % 360.0 - 180.0
            self._crnt.setCrntVal(key=('.', 'tau', 0), val = tau + d_tau)

        self.startThread(pushUndo=False)

    def fitAllSteps(self, selectedFiles = None):
        """Fits all steps in selected files; if no files are selected, uses the current file/series. The starting values on the next step are copied from the current found values."""
        # Form the list of steps to Fit
        s = self.stepsEdit.toPlainText()
        if re.search('[0-9]|(A[ ,A])|(Ph0)|(Ph1)|(PhX)|(Rsd)|(Lsh)|(PhA)|(PhA0)', s) is None: s = 'A'    # Fit all steps if the string is missing any numerical characters or A's
        s = "A ".join(re.split("A", s ))      # Prevent any consecutive A's from occuring in the string; separate them with spaces
        while s.find('(') != -1:    # Randomize all elements in all parentheses
            beg, end = s.find('('), s.find(')')
            R = re.split("[ ,]+", s[beg+1:end])    # R = re.split("[^0-9A]+", s[beg+1:end])
            R = [i for i in R if i != 'A'] + [str(i+1) for i in range(len(self._crnt.steps))]*R.count('A')   # Turn A's into lists of numbers and add them to the array
            shuffle(R)
            s = ", ".join((s[:beg], *R, s[end+1:]))
        L = re.split("[ ,]+",s)     #  L = re.split("[^0-9A]+", s)   # regex matches any non-digit character followed by any number (1+) of non-digit characters
        stepIdsToFit = [int(j)-1 if re.match('[0-9]+', j) else j for c in L for j in {'A':[str(i+1) for i in range(len(self._crnt.steps))]}.get(c, [c]) if j != '']    # replace 'A' with the list of all items

        # Set up the fitting queue making sure that there are no repeated files
        if selectedFiles is None:
            selectedFiles = [self._crnt]         # Fit all steps of the current file only
        selectedIDs = [ddd.selfID() for ddd in selectedFiles if isinstance(ddd, Datum)] \
                    + [ddd.selfID() for sss in selectedFiles for ddd in sss.data if isinstance(sss, Series)]      # Expand all Series
        selectedIDs = sorted(list(set(selectedIDs)))
        queueFiles = [self._crnt.series[sid[0]].data[sid[1]] for sid in selectedIDs]

        # Call the fitting function
        self.startThread(queueFiles, stepIdsToFit)

    def fitAllFiles(self):
        """Fits all steps for all Files in the current Series. The starting values on the next step are copied from the current found values. Starting values for each file are determined by the settings and are set by the FittingThread."""

        if isinstance(self._crnt, Series):
            selectedFiles = [i for i in self._crnt.data]

        elif isinstance(self._crnt, Datum):
            selectedFiles = [i for i in self._crnt.parent.data]

        else: return 0
        self.fitAllSteps(selectedFiles)

    def saveResults(self):
        """Saves the current results of computation into a file."""
        filename = QFileDialog.getSaveFileName(parent=self, caption='Select output file', directory='.', filter='(*.xlsx)')
        if filename:
            if filename == '' : filename = 'results.xlsx'
            if filename[-5:] != '.xlsx': filename += '.xlsx'

        # # Write the results to a file
        # tab = []
        # for sss in self._crnt.series:
        #     for ddd in sss.data:
        #         row = [sss.name, ddd.name, '{:.6f}'.format(np.linalg.norm(ddd.yT))]
        #         ampl, vars, names = [], [], []
        #         for name in ddd.repRootNames:
        #             if name not in ddd.xclRootNames:
        #                 try:
        #                     row += ['{:.6f}'.format(ddd.smplDistF[(name, 'ampl', 0)].mean), '{:.6f}'.format(ddd.smplDistF[(name, 'ampl', 0)].var)]
        #                     if name != 'Water':
        #                         ampl.append(ddd.smplDistF[(name, 'ampl', 0)].mean)
        #                         vars.append(ddd.smplDistF[(name, 'ampl', 0)].var)
        #                         names.append(name)
        #                 except KeyError:
        #                     row += ['{:.6f}'.format(ddd.crntParsH[name]['ampl'][0]), '---']
        #                     if name != 'Water':
        #                         ampl.append(ddd.crntParsH[name]['ampl'][0])
        #                         vars.append(0.0)
        #                         names.append(name)
        #
        #         # Compute mole fractions
        #         ampl = np.array(ampl).ravel()
        #         vars = np.array(vars).ravel()
        #         m_tot = np.sum(ampl)      # Total intensity
        #         v_tot = np.sum(vars)      # Total variance of the intensity estimate
        #         mfrac = ampl / m_tot
        #         confi = 2 * mfrac * np.sqrt(vars/(ampl**2) + v_tot/(m_tot**2))
        #         row += [f for mf, ci in zip(mfrac, confi) for f in ( '{:.6f}'.format(mf), '{:.6f}'.format(ci) )]
        #
        #         tab.append(row)
        # head = ['Series', 'Filename', 'Signal norm'] + [f for name in self.repRootNames for f in (name, 'var')] + [f for name in names for f in ('x_'+name, '95% cred.i.')]
        # try:
        #     with open(filename, 'w') as fout:
        #         print(tabulate.tabulate(tab, headers=head), file=fout)        # write results to a text file ...
        # except PermissionError:
        #     print('Can not save the results to file.')

        self._crnt.saveResults(filename)

# ------------------------ Parameter list --------------------------------------
    def updateParsList(self, indxStep = None):
        """Updates and displays the list of optimizaed parameters on the current step."""
        self.parsListWidget.clear()
        items = [node.name for node in stepClass.T.repRoots()]
        if indxStep is not None:
            items.extend([str(v) for v in self.steps[indxStep].parsKeys])
        self.parsListWidget.addItems(items)

    def onSelectParList(self, item):
        """Handles the selection event of a parameter in the list."""
        indxRow = self.parsListWidget.row(item)
        numRepRoots = len([node for node in stepClass.T.repRoots()])    # Number of reported nodes
        if indxRow < numRepRoots:
            key = self.parsListWidget.currentItem().text()
        else:
            key = self.steps[-1].parsKeys[indxRow - numRepRoots]    # Parameter name
            slctParSpec = stepClass.getPrior(stepClass, key)
            self.editMin.setText("{:.4g}".format(slctParSpec.min))
            self.editMax.setText("{:.4g}".format(slctParSpec.max))
            self.editCrntVal.setText("{:.4g}".format(slctParSpec.abs(self.steps[-1].getCrntVal(key))) )
            self.cmboxPrior.setCurrentIndex(self.cmboxPrior.findText(slctParSpec.prior['name']))
            if slctParSpec.prior['name'] in ['Gaussian', 'Log-Normal']:
                self.editPriorMode.setEnabled(True)
                self.editPriorStdv.setEnabled(True)
                self.chkboxUseCrnt.setEnabled(True)
                if slctParSpec.prior['mode'] is None:
                    self.chkboxUseCrnt.setCheckState(Qt.Checked)
                    self.editPriorMode.setText(self.editCrntVal.text())
                    self.editPriorMode.setReadOnly(True)
                else:
                    self.chkboxUseCrnt.setCheckState(Qt.Unchecked)
                    self.editPriorMode.setReadOnly(False)
                    self.editPriorMode.setText("{:.4g}".format(rel2abs(slctParSpec, slctParSpec.prior['mode'])))
                self.editPriorStdv.setText("{:.4g}".format(slctParSpec.prior['stdv']))
            else:   # Uniform prior
                    self.editPriorMode.setDisabled(True)
                    self.editPriorStdv.setDisabled(True)
                    self.chkboxUseCrnt.setDisabled(True)
        self.plotDistr(key)    # plot the prior distribution

    def onPriorNameChanged(self, itemIndx):
        """Handles the event of changing the name of the prior distribution in the combobox."""
        priorName = self.cmboxPrior.currentText()
        if priorName in ['Gaussian', 'Log-Normal']:      # or if itemIndx in [1, 2]
            self.editPriorMode.setEnabled(True)
            self.editPriorStdv.setEnabled(True)
            self.chkboxUseCrnt.setEnabled(True)
            if self.chkboxUseCrnt.isChecked:
                self.editPriorMode.setText(self.editCrntVal.text())
                self.editPriorMode.setReadOnly(True)
            else:
                self.editPriorMode.setReadOnly(False)
                self.editPriorMode.setText("{:.4g}".format( (float(self.editMin.text()) + float(self.editMax.text()))/2 ))
            self.editPriorStdv.setText("{:.4g}".format(0.5))
        else:   # Uniform prior
                self.editPriorMode.setDisabled(True)
                self.editPriorStdv.setDisabled(True)
                self.chkboxUseCrnt.setDisabled(True)
        self.saveParsForm()

    def saveParsForm(self):
        """Saves the parsSpec entered in the resForm1."""
        indxRow = self.parsListWidget.currentRow()
        numRepRoots = len([node for node in stepClass.T.repRoots()])    # Number of reported nodes
        if indxRow < numRepRoots:
            key = self.parsListWidget.currentItem().text()
        else:
            key = self.steps[0].parsKeys[indxRow - numRepRoots]    # Parameter name
            priorName = self.cmboxPrior.currentText()
            if priorName == 'Uniform':
                prior = {'name':priorName, 'mode':None, 'stdv':None}
            elif self.chkboxUseCrnt.isChecked():
                self.editPriorMode.setText(self.editCrntVal.text())
                self.editPriorMode.setReadOnly(True)
                prior = {'name':priorName, 'mode':None, 'stdv':float(self.editPriorStdv.text())}
            else:
                self.editPriorMode.setReadOnly(False)
                prior = {'name':priorName, 'mode':abs2rel(parsSpec(float(self.editMin.text()), float(self.editMax.text())), float(self.editPriorMode.text())), 'stdv':float(self.editPriorStdv.text())}

            # Save the parSpec
            stepClass.setPrior(stepClass, key, min=float(self.editMin.text()), max=float(self.editMax.text()), distr=priorName)
            # Update the current parameter values
            self.steps[0].setCrntVal(key, float(self.editCrntVal.text()))
            """self.refreshStep(len(self.steps)-1)     # Updqate the last step in the tree table
            # Display new min/max values in the tree
            self.treeWidget.blockSignals(True)     # don't call the onTreeItemChanged function
            self.treeItems[key].setText(1, "{:.4g}".format(slctParSpec.min))
            self.treeItems[key].setText(2, "{:.4g}".format(slctParSpec.max))
            self.treeWidget.blockSignals(False)
            # Update the rest of parameters based on their values in the tree table
            for indx, stp in enumerate(self.steps):
                clmn = indx+3
                stp.setCrntVal(key, float(self.treeItems[key].text(clmn)))"""
        # plot the prior distribution
        self.plotDistr(key)

    def plotDistr(self, key=None):
        """Plots a prior probability distribution for the parameter key on the middle plot."""
        # Determine which parameter is selected in the list and plot its samples
        if key is None:
            numRepRoots = len([node for node in stepClass.T.repRoots()])    # Number of reported nodes
            if self.parsListWidget.currentRow() < numRepRoots:
                key = self.parsListWidget.currentItem().text()
            else:
                key = self.steps[-1].parsKeys[self.parsListWidget.currentRow() - numRepRoots]    # Parameter name
        # Plot the piror distribution and samples
        self.axDistr.clear()
        self.axDistr2.clear()
        if type(key) is tuple:
            # Handle adjustible parameters
            slctParSpec = self.steps[-1].getPrior(key)
            # Plot the samples
            if self.steps[-1].sHat is not None:
                indx = self.steps[-1].sHat['smplKeys'].index(key)       # Index of the sampled parameter in the array of samples
                smpl_abs = self.steps[-1].sHat['smplVals'][indx, :]
                self.axDistr2.hist(smpl_abs, bins=50, range=(slctParSpec.min, slctParSpec.max), color='y', edgecolor=(0.96, 0.53, 0.20), alpha=0.5)
            # Plot the prior
            vals = [slctParSpec.evalPrior(x) for x in np.linspace(-1,1,25)]
            self.axDistr.plot(np.linspace(slctParSpec.min, slctParSpec.max, 25), vals, color=(0.96, 0.53, 0.20))
            self.axDistr.set_xlim((slctParSpec.min, slctParSpec.max))
            self.axDistr.axvline(x=slctParSpec.abs(self.steps[-1].getCrntVal(key)), ymin=0, ymax=0.05, color='r')
        else:
            # Handle the amplitudes
            if self.steps[-1].sHat is not None:
                indx = self.steps[-1].sHat['labels'].index(key)       # Index of the sampled parameter in the array of samples
                smpl_abs = np.abs(self.steps[-1].sHat['smplAmpl'][indx,:])
                self.axDistr2.hist(smpl_abs, bins=50, color='y', edgecolor=(0.96, 0.53, 0.20), alpha=0.5)
        self.histCanvas.draw()

# ------------------------ Functions for plotting ------------------------------
    def plotCurrent(self, autoRange=True):
        """Plots signals corresponding to the currently opened file and current parameters."""
        if isinstance(self._crnt, Series):
            self.mainFigureWidget.reset()
            self.plotPieChart(reset=True)

        elif isinstance(self._crnt, Datum):
            f, yFph, xF, zF, bF = self._crnt.signals_for_plot()

            # Compute the stems
            if self.actnToggleStems.isChecked():
                allStems = self._crnt.stems_for_plot()
            else: allStems = None

            # Plotting function
            self.mainFigureWidget.plot(f, yFph, xF,
                zF = zF if self.actnToggleComps.isChecked() else None,
                stems = allStems if self.actnToggleStems.isChecked() else None,
                freqBlocks=[(blk.min, blk.max, (i in self._crnt.steps[-1].frqBlkIds) ) for i, blk in enumerate(self._crnt.freqBlocks)],
                indx_colr=[i for i, name in enumerate(self._crnt.repRootNames) if not self._crnt.isXclRootName(name)])   # if i in self._crnt.steps[-1].frqBlkIds])

            if autoRange:
                self.mainFigureWidget.autoRange()

            # Output the found results
            self.plotPieChart()

            # Update the phasing widget
            self.mainPhasingWidget.setData(f, yFph, xF,
                freqBlocks=[(blk.min, blk.max, (i in self._crnt.steps[-1].frqBlkIds) ) for i, blk in enumerate(self._crnt.freqBlocks)])

        else:
            self.mainFigureWidget.reset()
            self.plotPieChart(reset=True)

    def toggleStems(self):
        """Plots stem lines to indicate modeled peaks."""
        try:
            self.mainFigureWidget.showStems(flag=self.actnToggleStems.isChecked())
        except:
            # if self.actnToggleStems.isChecked():
            self.plotCurrent(autoRange=False)

    def toggleComps(self):
        """Plots the constituent peaks for each model component"""
        # If no components have been computed so far, will evaluate them first.
        # TODO! Use customized exceptions
        try:
            self.mainFigureWidget.showComponents(flag=self.actnToggleComps.isChecked())
        except:
            if self.actnToggleComps.isChecked():
                self.plotCurrent(autoRange=False)

    def toggleResid(self):
        """Show or hide the residual plot."""
        self.mainFigureWidget.showResidual(flag=self.actnToggleResid.isChecked())

    def selectStems(self, key_crnt, key_prev=None):
        """Shows which stems are affected when a row is selected in the tree."""
        if self.actnToggleStems.isChecked():
            if key_prev is not None and key_prev[1] in ['chshQD', 'alphQD']:
                self.mainFigureWidget.highlightStems((key_prev[0], 'chshQD', key_prev[2]), False)
            if key_crnt is not None and key_crnt[1] in ['chshQD', 'alphQD']:
                self.mainFigureWidget.highlightStems((key_crnt[0], 'chshQD', key_crnt[2]), True)

    def plotPieChart(self, reset=False):
        """Plots a pie chart that represents the found component concentrations."""

        def hover(evt):
            if evt.inaxes == self.ax_pie:
                # Find which bar contains the event
                for indx, patch in enumerate(bars.patches):
                    if patch.contains(evt)[0]:
                        newMessage = '{:s}    {:.3g}%'.format(labels[indx], 100*cnct[indx])
                        if self.statusBar.currentMessage() != newMessage:
                            self.statusBar.showMessage(newMessage)
                        return

            self.statusBar.clearMessage()

        self.ax_pie.clear()

        # Remove the reference to the hovering event
        try:
            self.pieCanvas.mpl_disconnect(self._cid_hover)
        except AttributeError: pass

        if not reset:
            data = [(self._crnt.crntParsH[lbl]['ampl'][0], str(self.wsp.T[lbl])) for lbl in self.wsp.repRootNames if lbl not in ['Water', 'Chlorophorm'] and not self._crnt.isXclRootName(lbl)]     # All concentrations expcept water, chlorophorm, etc...
            cnct = np.abs([d[0] for d in data])
            cnct = np.where(np.isnan(cnct), 0.0, cnct)
            if sum(cnct) != 0:
                cnct = cnct / sum(cnct)
            labels = [d[1] for d in data]
            bars = self.ax_pie.bar(np.arange(len(labels)), 100*cnct, tick_label=labels, align='center',
                color=[col for col, name in zip(config.colrseq[2:], self._crnt.repRootNames) if name not in ['Water', 'Chlorophorm'] and not self._crnt.isXclRootName(name)])
            self.ax_pie.set_xticklabels(labels, rotation='vertical' if len(labels) > 3 else 'horizontal')
            ttl = self.ax_pie.set_title('Relative concentrations, %', fontsize=12)
            ttl.set_position((0.5, 1.03))
            # wedges, texts, autotexts = self.ax_pie.pie(cnct, labels=labels, explode=[0.05]*len(cnct), shadow=True, autopct='%0.2f', colors=config.colrseq)
            # self.ax_pie.legend(wedges, labels,
            #   loc="bottom",
            #   bbox_to_anchor=(0, 0.1, 0.5, 1))
            # self.ax_pie.axis('equal')

            self._cid_hover = self.pieCanvas.mpl_connect("motion_notify_event", hover)

        self.pieCanvas.draw()

# --------------------- Adding and removing frequency blocks -------------------
    def addFreqBlock(self, xmin, xmax):
        """Adds new optimization range to the current Series and updates the plot."""
        dref_chsh = 0.0 # self._crnt.getGlobalChshVal() if config.DISPL_ShiftToReference else 0.0
        self.freqTableModel.addFreqBlock(xmin + dref_chsh, xmax + dref_chsh)
        self.mainFigureWidget.addFreqRange((xmin, xmax))

        self.startThread(pushUndo=False)

        self.preprocTool.setNewDatum(self._crnt)

    def remFreqBlock(self, indx):
        """Removes a frequency block that covers a location xdata in ppm and updates the plot."""
        # Find which block (if any) covers the passed location and which one to remove if there are multiple blocks.
        self.mainFigureWidget.remFreqRange(indx)
        self.freqTableModel.remFreqBlock(indx)

        self.startThread(pushUndo=False)

        self.preprocTool.setNewDatum(self._crnt)

    def updFreqBlock(self, indx, lims=None, active=None):
        """Updates the ranges of a frequency block in the model."""
        if lims is None:
            lims = (self._crnt.freqBlocks[indx].min, self._crnt.freqBlocks[indx].max)
        if active is None:
            active = indx in self._crnt.steps[-1].frqBlkIds

        self._crnt.altFreqBlock(indx=indx, lims=lims)

        # In general, calling both functions below is unnecessary as the changes heve likely been caused by either freqTableModel or mainFigureWidget
        self.freqTableModel.notifyDataChanged(indx)
        self.mainFigureWidget.updFreqRange( indx, lims, active )

        self.startThread(pushUndo=False)

        self.preprocTool.setNewDatum(self._crnt)

    def reduceRange(self):
        """Selects and cuts a region of interest in the spectrum and adjusts the underlying data accordingly."""
        print('In reduce range')

if __name__ == '__main__':
    app = 0
    app = QApplication(sys.argv)

    expiryTime, options = readLicenseFile()

    if expiryTime is None:
        # No license file found
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)

        msg.setText("Please place a valid license *.lic file in the program directory.")
        # msg.setInformativeText("This is additional information")
        msg.setWindowTitle("Missing license file")
        # msg.setDetailedText("The details are as follows:")
        msg.setStandardButtons(QMessageBox.Close)

        msg.show()            # Returns the values of pressed button

    elif time.time() > expiryTime:
        # Checks whether the current time is less than the expiry time (in sec from the beginning of the epoch).
        # The license was found but has expired
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)

        msg.setText("The license has expired.")
        msg.setInformativeText("Please place a valid license *.lic file in the program directory.")
        msg.setWindowTitle("Expired license file")
        # msg.setDetailedText("The details are as follows:")
        msg.setStandardButtons(QMessageBox.Close)

        msg.show()         # msg.exec_()            # Returns the values of pressed button
    else:
        wsp = Workspace()
        main_view = MainView(wsp, expiryTime)
        main_view.show()

    app.exec_()
