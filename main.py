
import sys
import numpy as np
import pickle as pickle
from MainLogic import *
from MainLogic import Series, Datum, Workspace
import config

from PyQt4 import QtGui, QtCore, uic
from PyQt4.QtGui import QAction, QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QGroupBox, QIcon, QInputDialog, QItemSelectionModel, QLabel, QLineEdit, QListWidget, QMenu, QMessageBox, QVBoxLayout, QHBoxLayout, QGridLayout, QMainWindow, QPlainTextEdit, QProgressBar, QPushButton, QRadioButton, QSizePolicy, QSlider, QSpinBox, QSplitter, QStatusBar, QTableView, QTabWidget, QTableWidget, QToolButton, QTreeView, QToolBar, QWidget
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
import math
import os
import nmrglue as ng
from datetime import date

#from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT
from matplotlib.backend_bases import NavigationToolbar2
try:
    import matplotlib.backends.qt_editor.figureoptions as figureoptions
except ImportError:
    figureoptions = None

version = '0.16.0'
compile_standalone = True   # Change to False for debugging/development to output the results into the usual console

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

class CustomToolbar(NavigationToolbar2, QToolBar):

    def __init__(self, canvas, parent, coordinates=True):
        """ coordinates: should we show the coordinates on the right? """
        self.canvas = canvas
        self.parent = parent
        self.coordinates = coordinates
        self._actions = {}    # A mapping of toolitem method names to their QActions

        self.toolitems = (
            ('Home', 'Reset original view', 'home', 'home'),
            ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
            ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
            (None, None, None, None),
            #('My action', 'Description of my action', 'icon_import', 'doSomething'),        # text, tooltip_text, image_file, callback
            ('AddRange', 'Add optimization range', 'add_range', 'addRange'),
            ('RemoveRange', 'Remove optimization range', 'remove_range', 'remRange'),
            ('ShowStems', 'Show models for all chemical species', 'show_all_chems', 'showStems'),
            ('ShowComponents', 'Show computed model components', 'show_components', 'showStems'),
            ('PlotResidual', 'Plot residual', 'plot_residual', 'showStems'),
            (None, None, None, None),
            #('Subplots', 'Configure subplots', 'subplots', 'configure_subplots'),
            ('Save', 'Save the current image', 'icon_saveImage', 'save_figure'),
            )

        QToolBar.__init__(self, parent)
        NavigationToolbar2.__init__(self, canvas)

    def _icon(self, name):
        """Use the _icon function from the main form."""
        return self.parent._icon(name)

    def _init_toolbar(self):
        self.basedir = path.join(rcParams['datapath'], 'images')

        for text, tooltip_text, image_file, callback in self.toolitems:
            if text is None:
                self.addSeparator()
            elif text == 'ShowStems':
                self.addAction(self.parent.actnShowStems)
            elif text == 'ShowComponents':
                self.addAction(self.parent.actnShowComponents)
            elif text == 'PlotResidual':
                self.addAction(self.parent.actnPlotResidual)
            else:
                a = self.addAction(self._icon(image_file + '.png'),
                                         text, getattr(self, callback))
                self._actions[callback] = a
                if callback in ['zoom', 'pan', 'addRange', 'remRange']:
                    a.setCheckable(True)
                if tooltip_text is not None:
                    a.setToolTip(tooltip_text)

        if figureoptions is not None:
            a = self.addAction(self._icon("qt4_editor_options.png"),
                               'Customize', self.edit_parameters)
            a.setToolTip('Edit curves line and axes parameters')

        self.buttons = {}

        # Add the x,y location widget at the right side of the toolbar
        # The stretch factor is 1 which means any resizing of the toolbar
        # will resize this label instead of the buttons.
        if self.coordinates:
            self.locLabel = QLabel("", self)
            self.locLabel.setAlignment(
                    Qt.AlignRight | Qt.AlignTop)
            self.locLabel.setSizePolicy(
                QSizePolicy(QQSizePolicy.Expanding,
                                  QSizePolicy.Ignored))
            labelAction = self.addWidget(self.locLabel)
            labelAction.setVisible(True)

        # reference holder for subplots_adjust window
        self.adj_window = None

    if figureoptions is not None:
        def edit_parameters(self):
            allaxes = self.canvas.figure.get_axes()
            if len(allaxes) == 1:
                axes = allaxes[0]
            else:
                titles = []
                for axes in allaxes:
                    title = axes.get_title()
                    ylabel = axes.get_ylabel()
                    label = axes.get_label()
                    if title:
                        fmt = "%(title)s"
                        if ylabel:
                            fmt += ": %(ylabel)s"
                        fmt += " (%(axes_repr)s)"
                    elif ylabel:
                        fmt = "%(axes_repr)s (%(ylabel)s)"
                    elif label:
                        fmt = "%(axes_repr)s (%(label)s)"
                    else:
                        fmt = "%(axes_repr)s"
                    titles.append(fmt % dict(title=title,
                                         ylabel=ylabel, label=label,
                                         axes_repr=repr(axes)))
                item, ok = QInputDialog.getItem(
                    self.parent, 'Customize', 'Select axes:', titles, 0, False)
                if ok:
                    axes = allaxes[titles.index(six.text_type(item))]
                else:
                    return

            figureoptions.figure_edit(axes, self)

    def _update_buttons_checked(self):
        # sync button checkstates to match active mode
        self._actions['pan'].setChecked(self._active == 'PAN')
        self._actions['zoom'].setChecked(self._active == 'ZOOM')
        self._actions['addRange'].setChecked(self._active == 'ADDRANGE')
        self._actions['remRange'].setChecked(self._active == 'REMRANGE')

    def home(self):
        """Subclassed home button. Resets the ranges to the best view."""
        axes = self.canvas.figure.get_axes()   # Get the main axes
        for ax in axes:
            if ax.get_visible():
                ax.relim()    # recompute the ax.dataLim
                ax.margins(0, 0.05)    # x and y margins in percentages
                ax.autoscale()    # update ax.viewLim using the new dataLim
        self.canvas.draw()

    def pan(self, *args):
        super(CustomToolbar, self).pan(*args)
        self.parent.freqRangeSelector.active = False
        self._update_buttons_checked()

    def zoom(self, *args):
        super(CustomToolbar, self).zoom(*args)
        self.parent.freqRangeSelector.active = False
        self._update_buttons_checked()

    def dynamic_update(self):
        self.canvas.draw()

    def _set_cursor(self, event):
        # from backend_bases.NavigationToolbar2
        if not event.inaxes or not self._active:
            if self._lastCursor != cursors.POINTER:
                self.set_cursor(cursors.POINTER)
                self._lastCursor = cursors.POINTER
        else:
            if self._active == 'ZOOM':
                if self._lastCursor != cursors.SELECT_REGION:
                    self.set_cursor(cursors.SELECT_REGION)
                    self._lastCursor = cursors.SELECT_REGION
            elif (self._active == 'PAN' and
                  self._lastCursor != cursors.MOVE):
                self.set_cursor(cursors.MOVE)
                self._lastCursor = cursors.MOVE
            elif (self._active == 'ADDRANGE' and
                  self._lastCursor != cursors.HAND):
                self.set_cursor(cursors.HAND)
                self._lastCursor = cursors.HAND
            elif (self._active == 'REMRANGE' and
                  self._lastCursor != cursors.HAND):
                self.set_cursor(cursors.HAND)
                self._lastCursor = cursors.HAND

    def set_cursor(self, cursor):
        self.canvas.setCursor(cursord[cursor])

    def draw_rubberband(self, event, x0, y0, x1, y1):
        height = self.canvas.figure.bbox.height
        y1 = height - y1
        y0 = height - y0

        w = abs(x1 - x0)
        h = abs(y1 - y0)

        rect = [int(val)for val in (min(x0, x1), min(y0, y1), w, h)]
        self.canvas.drawRectangle(rect)

    def save_figure(self, *args):
        filetypes = self.canvas.get_supported_filetypes_grouped()
        sorted_filetypes = list(six.iteritems(filetypes))
        sorted_filetypes.sort()
        default_filetype = self.canvas.get_default_filetype()

        startpath = rcParams.get('savefig.directory', '')
        startpath = path.expanduser(startpath)
        start = path.join(startpath, self.canvas.get_default_filename())
        filters = []
        for name, exts in sorted_filetypes:
            exts_list = " ".join(['*.%s' % ext for ext in exts])
            filter = '%s (%s)' % (name, exts_list)
            if default_filetype in exts:
                selectedFilter = filter
            filters.append(filter)
        filters = ';;'.join(filters)

        fname = QFileDialog.getSaveFileName(parent=self.parent,
                                         caption="Choose a filename to save to", directory=start, filter=filters)
        if fname:
            if startpath == '':
                # explicitly missing key or empty str signals to use cwd
                rcParams['savefig.directory'] = startpath
            else:
                # save dir for next time
                savefig_dir = path.dirname(six.text_type(fname))
                rcParams['savefig.directory'] = savefig_dir
            try:
                #self.canvas.savefig(six.text_type(fname), format='eps', dpi=1200)
                fontSizeBefore = rcParams['font.size']
                rcParams['font.size'] = 18
                self.canvas.print_figure(six.text_type(fname))
                rcParams['font.size'] = fontSizeBefore
                self.canvas.draw()
            except Exception as e:
                QMessageBox.critical(
                    self, "Error saving file", str(e),
                    QMessageBox.Ok, QMessageBox.NoButton)

    def addRange(self):
        #self.parent.addFreqBlock(-1., 1.)
        """Activate the adding a new range mode."""
        if self._active == 'ADDRANGE':
            self._active = None
        else:
            self._active = 'ADDRANGE'
            # Turn off previous mode (zoom or pan)
            self.canvas.widgetlock.release(self)
        for a in self.canvas.figure.get_axes():
            a.set_navigate_mode(None)

        """Deactivate previous mode (zoom/pan)"""
        if self._idPress is not None:
            self._idPress = self.canvas.mpl_disconnect(self._idPress)
            self.mode = ''

        if self._idRelease is not None:
            self._idRelease = self.canvas.mpl_disconnect(self._idRelease)
            self.mode = ''

        if self._active:
            #self._idPress = self.canvas.mpl_connect('button_press_event',
            #                                        self.press_zoom)
            #self._idRelease = self.canvas.mpl_connect('button_release_event',
            #                                          self.release_zoom)
            self.parent.freqRangeSelector.active = True
            self.mode = 'add range'
        else:
            self.parent.freqRangeSelector.active = False
            self.mode = ''

        self._update_buttons_checked()

    def remRange(self):
        """Removes an optimization range by clicking on it."""
        if self._active == 'REMRANGE':
            self._active = None
        else:
            self._active = 'REMRANGE'
            self.canvas.widgetlock.release(self)            # Turn off previous mode (zoom or pan)
            self.parent.freqRangeSelector.active = False    # Deactivate the range adding mode
        for a in self.canvas.figure.get_axes():
            a.set_navigate_mode(None)

        """Deactivate previous mode (zoom/pan)"""
        if self._idPress is not None:
            self._idPress = self.canvas.mpl_disconnect(self._idPress)
            self.mode = ''

        if self._idRelease is not None:
            self._idRelease = self.canvas.mpl_disconnect(self._idRelease)
            self.mode = ''

        if self._active:
            self._idPress = self.canvas.mpl_connect('button_press_event',
                                                    self.parent.remFreqBlock)
            #self._idRelease = self.canvas.mpl_connect('button_release_event',
            #                                          self.release_zoom)
            self.mode = 'remove range'
        else:
            self.mode = ''

        self._update_buttons_checked()

    def showStems(self):
        print("This is my action in the toolbar.")

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

    def __init__(self, parent = None):
        super(ChooseFromDBDialog, self).__init__(parent)
        layout = QVBoxLayout(self)

        # Add widgets for entering parameters
        keysDB = sorted([k for k, v in chemDB.items()])
        self.cmbox = QComboBox()
        self.cmbox.addItems(keysDB)
        layout.addWidget(self.cmbox)

        # OK and Cancel buttons
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)
        layout.addWidget(self.buttons)

        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    # get the selection
    def getSelection(self):
        return self.cmbox.currentText()

    # static method to create the dialog and return (date, time, accepted)
    @staticmethod
    def run(parent = None):
        dialog = ChooseFromDBDialog(parent)
        result = dialog.exec_()
        if result == QDialog.Accepted:    # If OK was clicked
            return (dialog.getSelection(), result == QDialog.Accepted)
        else: return (None, result == QDialog.Accepted)

class SettingsDialog(QDialog):
    def __init__(self, parent = None):
        super(SettingsDialog, self).__init__(parent)

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
        groupLayout.addWidget(self.chkboxAutoPhasing)
        groupLayout.addWidget(self.chkboxAutoPicking)
        rbtnGroup = QGroupBox("Processing series of spectra")
        rbtnGroup.setLayout(groupLayout)

        # ----------- Settings for the QD simulations --------------------------
        self.editRerunThreshold = MyDoubleEdit(config.QD_RerunQDchshThreshold)
        self.editAggregateThreshold = MyDoubleEdit(config.QD_AggregatePeaksThreshold)
        groupLayout = QFormLayout()
        groupLayout.addRow("Merge resonances closer than, Hz", self.editAggregateThreshold)
        groupLayout.addRow("Update if chsh changed by, ppm", self.editRerunThreshold)
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
        layout.addWidget(self.chkboxRobustLS)
        # layout.addWidget(lklfConfigGroup)
        # layout.addWidget(smplConfigGroup)

        # OK and Cancel buttons
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)
        layout.addWidget(self.buttons)

        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    # get current date and time from the dialog
    def getEntries(self):
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

    # static method to create the dialog and return
    @staticmethod
    def run(parent = None):
        dialog = SettingsDialog(parent)
        result = dialog.exec_()
        if result == QDialog.Accepted:    # If OK was clicked
            dialog.getEntries()
        #else: chkdForAll = False
        return QDialog.Accepted

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
                    if len(self.datum.steps[0].frqBlkIds) > 0:
                        return QtCore.Qt.Unchecked
                    else: return QtCore.Qt.Checked
                elif row > 0 and (row-1) in self.datum.steps[0].frqBlkIds:
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
                if len(self.datum.steps[0].frqBlkIds) > 0:
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
                return True

        return False

    #=====================================================#
    #INSERTING & REMOVING
    #=====================================================#

    def addFreqBlock(self, xmin, xmax, parent = QtCore.QModelIndex()):
        """Adds a new frequency block to the series."""
        self.beginInsertRows(parent, self.rowCount(), self.rowCount())        # Parent node, first and last position
        self.datum.addFreqBlock((xmin, xmax))
        for step in self.datum.steps:
            step.frqBlkIds.add(len(self.datum.freqBlocks)-1)
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

        self.beginInsertRows(QtCore.QModelIndex(), len(self.wsp.series), len(self.wsp.series))        # Parent node, first and last position

        self.wsp.addSeries()

        self.endInsertRows()

    def addDatum(self, crnt_series, path):
        """Adds new Datum entries specified by the path to the series object."""

        ser_id = self.wsp.series.index(crnt_series) # Position of the Series in the Workspace
        index = self.index(ser_id, 0, None)    # Index corresponding to the Series to which the Datum will be added

        if path[-6:] == '.pyfid':
            with open(path, 'rb') as fp:
                data = [float(x.strip()) if i != 5 else x.strip() for i, x in enumerate(fp.readlines())]

            c0, f0, nt = data[0], data[1], int(data[4])     # Number of time points

            t = np.array(data[6:nt+6]).reshape(-1,1)
            yT = (np.array(data[nt+6:2*nt+6]) + 1j*np.array(data[-nt:])).reshape(-1,1)
            name = path[path.rfind('\\')+1:path.rfind('.')]

        elif path[-3:] == '.dx':
            # Read a JCAMP-DX file
            dic, data = ng.jcampdx.read(path)
            #c0 = float(dic['$BF1'][0])
            #fcar = float(dic['$REFERENCEPOINT'][0])
            #swh = float(dic['$SW'][0]) * c0
            #nt = float(dic['$TD'][0])

            udic = ng.jcampdx.guess_udic(dic,data)[0]     # Dictionary of universal parameters

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

            name = os.path.split(os.path.dirname(path))[1]    # Only the name of the containing directory

        elif path[-3:] == '.1d':

            # Read a Spinsolve data.1d file
            dic, data = ng.fileio.spinsolve.read(path)
            c0 = dic['b1Freq']
            fcar = -dic['lowestFrequency']
            dt = dic['dwellTime'] * 1e-6
            nt = dic['nrPnts']

            swh = 1 / dt
            f0 = swh/2-fcar      # Frequency shift in Hz

            t = np.linspace(start=0, stop=(nt-1)*dt, num=nt).reshape(-1,1)
            yT = data.reshape(-1, 1)

            # Subsample if the frequency range is too large
            k = max(math.floor(swh/c0 / 12), 1)   # Sampling factor to make the sweep width 12 ppm
            t = t[::k]
            yT = yT[::k, :]

            name = os.path.split(os.path.dirname(path))[1]    # Only the name of the containing directory

        # Save the acquisition parameters; these should be the same for all spectra in the series (by convention)
        if crnt_series.c0 is None:
            crnt_series.c0 = c0
            crnt_series.f0 = f0
            crnt_series.t = t
            crnt_series.fullReset()
            # TODO: Check if new c0/f0 are the same as the old ones when loading the rest of the data

        self.layoutAboutToBeChanged.emit()

        if len(crnt_series.data) == 0 and yT.shape[1] == 1:   # Adding only a single first Datum; no rows will be added, but need to replace the existing Series row with this new Datum

            crnt_series.addDatum(yT, name = name)
        else:
            # Start adding rows
            if len(crnt_series.data) == 1:   # Already one node in the series
                self.beginInsertRows(index, 0, yT.shape[1])        # Parent node, first and last position
            else:
                self.beginInsertRows(index, len(crnt_series.data), len(crnt_series.data)+yT.shape[1]-1)        # Parent node, first and last position
            # Add the rows
            for i in range(yT.shape[1]):
                crnt_series.addDatum(yT[:,i], name = name+str(i+1) if yT.shape[1]>1 else name)
            # Finish adding rows
            self.endInsertRows()

        self.layoutChanged.emit()    # Tell the view that we need to recompute persistent indices

    def importData(self, parent=None):
        """Opens a dialog to select a new data file to be added to the parent series."""
        # Create new series if working from the workspace itself
        if parent is None or parent == self.wsp:
            self.addSeries()
            parent = self.wsp.series[-1]
        elif isinstance(parent, Datum):
            parent = parent.parent    # Go one level up to the Series level

        for newFilePath in QFileDialog.getOpenFileNames(None, 'Import file', '.', filter = "All supported files (*.pyfid; *.dx; *.1d);;Converted FID (*.pyfid);;Spinsolve binary (*.1d);;JCAMP (*.dx)"):
            #try:
            self.addDatum(parent, newFilePath)
            #except:
            #    print("Could not add the file ", newFilePath)

        # Set current index to the newly updated Series
        #ser_indx = self.wsp.series.index(self._crnt) # Position of the Series in the Workspace
        #index = self.naviTreeModel.index(ser_indx, 0, None)    # Index corresponding to the Series to which the Datum will be added
        #self.naviSelection.setCurrentIndex(index, QItemSelectionModel.Select)

    def remItems(self, items):
        """Removes a Series or a Datum"""
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

class NavigationTreeView(QTreeView):
    """Model/View based class to display loaded datasets."""

    requestPasteCrnt = pyqtSignal(list)
    requestPasteDflt = pyqtSignal(list)
    requestfitSelected = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)    # Initialize a QTreeWidget

        self.setAlternatingRowColors(True)
        self.setHeaderHidden(True)
        self.setSelectionMode(QTreeView.ExtendedSelection)

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
        actnImportData.triggered.connect(lambda : self.model().importData(parent=index.internalPointer() if index.isValid() else None))
        actnPasteCrnt = QAction(QIcon('icons\icon_pasteCrnt.png'), 'Paste as current', self)
        actnPasteCrnt.setStatusTip('Paste as current values')
        actnPasteCrnt.triggered.connect(lambda : self.requestPasteCrnt.emit(selected))     # Emit a list of selected datums to paste the currently copied parameters to them
        actnPasteDflt = QAction(QIcon('icons\icon_pasteDflt.png'), 'Paste as defualt', self)
        actnPasteDflt.setStatusTip('Paste as default values')
        actnPasteDflt.triggered.connect(lambda : self.requestPasteDflt.emit(selected))     # Emit a list of selected datums to paste the currently copied parameters to them
        actnfitSelected = QAction(QIcon('icons\icon_fitSelected.png'), 'Fit selected', self)
        actnfitSelected.setStatusTip('Fit selected datasets')
        actnfitSelected.triggered.connect(lambda : self.requestfitSelected.emit(selected))     # Emit a list of selected datums to fit

        popMenu.addAction(actnfitSelected)
        popMenu.addSeparator()
        popMenu.addAction(actnAddSeries)
        popMenu.addAction(actnImportData)

        if index.isValid():         # If the click was on an item
            actnRemoveData = QAction(QIcon('icons\icon_removeFile.png'), 'Remove files', self)
            actnRemoveData.setStatusTip('Remove file from the workspace')
            actnRemoveData.triggered.connect(lambda : self.model().remItems(items=[selind.internalPointer() for selind in self.selectedIndexes() if selind.isValid()]))
            popMenu.addAction(actnRemoveData)

        popMenu.addSeparator()
        popMenu.addAction(actnPasteCrnt)
        popMenu.addAction(actnPasteDflt)

        # Show the menu
        popMenu.popup(self.viewport().mapToGlobal(pos))

def getParsTree(T, myOrder = ['ampl', 'chsh', 'alph', 'jcpl']):
    """Returns the tree of parameters P for a chemNode tree T. The variable myOrder defines the order in which the parameters will be sorted. Each node in the parameter tree corresponds to a chemical/group of chemicals or its parameters."""
    P = viewNode(T.name, nodeType='chemNodeDB' if isinstance(T, chemNodeDB) else 'chemNode')
    #P.addChild(viewNode(name = tuple([T.name] + ['intn'] + [None]), nodeType='intn', alias='intn' ))
    if type(T) is chemNodeQD:    # Spin system defined by itself without a parent chemDB node
        P.addChild(viewNode(name = tuple([T.name] + ['ampl'] + [0]), nodeType='param', alias='intn' ))

        if T.childCount() > 1:  # Several chemical shifts; add global parameters
            P.addChild(viewNode(name = tuple([T.name] + ['chsh'] + [0]), nodeType='param' ))
            P.addChild(viewNode(name = tuple([T.name] + ['alph'] + [0]), nodeType='param' ))

        for node in [T]:
            for par, val in node.default_pars().items():
                for i in range(len(val)):
                    if not isinstance(getattr(node, par)[i].label, str):
                        old = getattr(node, par)
                        old[i] = parsSpec(old[i].min, old[i].max, label='', distr=old[i].distr, p1=old[i].p1, p2=old[i].p2)
                        setattr(node, par, old)

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
            P.addChild(getParsTree(node))
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

        # Fix faulty labels of parameters and alaises of the terminal nodes
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

    """
    # Full parameter tree
    P = treeNode(T.name)
    # add the parameters nodes
    for par, val in sorted(T.default_pars().items(), key = lambda par : [i for i, x in enumerate(myOrder) if x in par[0]][-1] ):
        print(par, val)
        for i in range(len(val)):
            P.addChild(treeNode(name = tuple([T.name] + [par] + [i]) ))
    # add the children nodes
    for node in T.children():
        P.addChild(getParsTree(node))
    return P
    """

class ChemTreeModel(QtCore.QAbstractItemModel):
    """A treeView representation class"""

    skipColumns = 5    # Number of columns that display numerical values (min, max, default, current) + the name column
    crntChanged = pyqtSignal()    # Emmitted when current values are chenged by the user

    @staticmethod
    def showName(name):
            """Determines how to display the name of an item"""
            if isinstance(name, tuple):
                return name[1]   # return the first element in the tuple
            else: return name

    def __init__(self, datum, parent = None):
        super().__init__()     # QtCore.QAbstractItemModel.__init__(self)
        self.datum = datum       # A pointer to the workspace
        self._indxRoot = QtCore.QModelIndex()    # "Invalid" index to point to the root of the display
        self._buttons = viewNode("_buttons", alias=None)
        self._ranges = viewNode("_ranges", alias="")
        self._parsSigma2 = viewNode(('.', 'sigma2', 0), alias='Variance of noise, s2', nodeType='param')
        self._parsPH0 = viewNode(('.', 'theta', 0), alias='Zero-order phase (PH0)', nodeType='param')
        self._parsPH1 = viewNode(('.', 'tau', 0), alias='Acquisition delay (PH1)', nodeType='param')
        self._ratioTLS = viewNode(('.', 'gamma', 0), alias='TLS ratio', nodeType='param')
        self._lshape = viewNode('_lshape', alias='Lineshape correction', nodeType='lshape')

        self.resetChemTree(flag=False)
        self.resetLshapeTree(flag=False)

    def resetChemTree(self, flag=True):
        """Updates the chemical tree."""
        if flag: self.beginResetModel()

        # The tree of parameters to be displayed
        self.TP = getParsTree(self.datum.T) if self.datum.T is not None else viewNode('')

        if flag: self.endResetModel()

    def resetLshapeTree(self, flag=True):
        """Updates the tree of lineshape correction parameters."""
        if flag: self.beginResetModel()

        # Reset the lineshape correction subtree
        self._lshape.clearChildren()
        self._lshape.addChild(viewNode('lshapeX', alias='Fit custom shape', nodeType='bool'))       # Custom lineshape
        supscr = ['nd', 'rd'] + ['th']*(self.datum.lshapeOrder-2)
        for i in range(self.datum.lshapeOrder):
            self._lshape.addChild(viewNode(('.','lshapeR',i), alias='{}{} order Re'.format(i+2, supscr[i]), nodeType='param'))
            self._lshape.addChild(viewNode(('.','lshapeI',i), alias='{}{} order Im'.format(i+2, supscr[i]), nodeType='param'))

        if flag: self.endResetModel()

    def setNewDatum(self, datum):
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

        if oldColumnCount < newColumnCount:
            self.endInsertColumns()
        elif oldColumnCount > newColumnCount:
            self.endRemoveColumns()
        else: self.notifyDataChanged()

    def fullReset(self, datum):
        self.beginResetModel()
        self.datum = datum
        self.resetChemTree(flag=False)
        self.resetLshapeTree(flag=False)
        self.endResetModel()

    def notifyDataChanged(self):
        self.dataChanged.emit(self._indxRoot, self._indxRoot)     # Update the entire tree

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
                    return self.columnCount() - section    # In reversed order
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
            return 6    # Number of rows in the display root

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
            #if row == 0:
            #    return self.createIndex(row,column,self._buttons)
            #elif row == 1:
            #    return self.createIndex(row,column,self._ranges)
            if row == 0:
                i = self.createIndex(row, column, self.TP)
                return i
            elif row == 1:
                return self.createIndex(row,column,self._parsPH0)
            elif row == 2:
                return self.createIndex(row,column,self._parsPH1)
            elif row == 3:
                return self.createIndex(row,column,self._parsSigma2)
            elif row == 4:
                return self.createIndex(row,column,self._ratioTLS)
            elif row == 5:
                return self.createIndex(row, column, self._lshape)
        else:
            parent = prnt.internalPointer()
            child = parent.child(row)
            if child:
                return self.createIndex(row, column, child)
            else:
                return QtCore.QModelIndex()

        #if not self.hasIndex(row, column, prnt):
        #    print("doesn't have this index")
        #    return self._rootIndex

    def indexByKey(self, key):
        """Searches for the element specified by its key in the TP tree and returns its index."""
        item = self.TP[key]
        row = item.siblID()
        column = 0
        return self.createIndex(row, column, item)

    def data(self, index, role):
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

            if node.nodeType in ['chemNode', 'chemNodeDB']:     # Chemical node
                if not self.datum.T[node.name].isReported():
                    font.setStyle(QtGui.QFont.StyleItalic)
                elif node.name in self.datum.repRootNames:
                    font.setBold(True)

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

            elif node.nodeType in ['chemNode', 'chemNodeDB'] and clmn == 4:
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
                    """elif node.nodeType == 'intn':
                    key = node.name[0]
                    if self.datum.T[key].isReported():
                        if key in self.datum.repRootNames:
                            displayIcon = QIcon("icons\icon_ampl.png")
                        else: displayIcon = QIcon("icons\icon_intn.png")
                    else: displayIcon = QIcon("icons\icon_blank.png")"""
                elif node.nodeType == 'lshape':
                    displayIcon = QIcon('icons\icon_lshape.png')
                elif node.name == 'lshapeX':
                    displayIcon = QIcon('icons\icon_lshape.png')
                else:
                    displayIcon = QIcon("icons\icon_chemMixture.png")
            return displayIcon

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
            elif node.nodeType == 'chemNodeDB':
                return result # | QtCore.Qt.ItemIsUserCheckable
            else:
                return result

        # Allow setting intensities of nodes in the tree
        elif clmn == 4:
            if node.nodeType == 'lshape':
                return result
            elif node.nodeType in ['chemNode', 'chemNodeDB']:
                return result | QtCore.Qt.ItemIsEditable
            else:
                return result | QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsSelectable
            """elif node.nodeType in ['chemNode', 'chemNodeDB'] and clmn == 4:
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
                    try:
                        self.datum.setCrntVal(node.name, value)
                        self.crntChanged.emit()
                        self.dataChanged.emit(index, index)
                    except: return False

            elif node.nodeType in ['chemNode', 'chemNodeDB'] and clmn == 4 and self.datum.T[node.name].isReported() and node.name not in self.datum.repRootNames:
                self.datum.T[node.name].set_intn(float(value))
                self.crntChanged.emit()
                self.dataChanged.emit(index, index)
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
            modifiers = QtGui.QApplication.keyboardModifiers()
            """if modifiers == QtCore.Qt.ShiftModifier:
                print('Shift+Click')
            elif modifiers == QtCore.Qt.ControlModifier:
                print('Control+Click')
            elif modifiers == (QtCore.Qt.ControlModifier |
                               QtCore.Qt.ShiftModifier):
                print('Control+Shift+Click')
            else:
                print('Click')"""
            # step_indx = clmn-self.skipColumns    # Normal order
            step_indx = self.columnCount() - clmn - 1    # Reversed order

            if node.nodeType == 'param':
                try:
                    self.datum.steps[step_indx].autoKeys.remove(node.name)
                    if not self.datum.isAutofittable(key=node.name):
                        self.datum.steps[step_indx].parsKeys.add(node.name)
                except KeyError:
                    try:
                        self.datum.steps[step_indx].parsKeys.remove(node.name)
                        if self.datum.isAutofittable(key=node.name):
                            self.datum.steps[step_indx].autoKeys.add(node.name)
                    except KeyError:
                        self.datum.steps[step_indx].parsKeys.add(node.name)

            elif node.nodeType == 'bool':
                if node.name == 'lshapeX':
                    self.datum.steps[step_indx].fitCustomLshape = not self.datum.steps[step_indx].fitCustomLshape

            self.dataChanged.emit(index, index)
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
        self.endInsertColumns()

    def delStep(self):
        try:
            if len(self.datum.steps) > 1:
                parent = QtCore.QModelIndex()
                self.beginRemoveColumns(parent, self.columnCount()-1, self.columnCount()-1)
                self.datum.steps.pop(-1)
                self.endRemoveColumns()
        except AttributeError:
            return False

    def addChemical(self, index, source = 'new'):
        """Adds a new chemical to the tree as a child to node index."""

        # Create ne chemical node or load a subtree
        if source == 'new':
            X = chemNode('New group')
            pars = defaultTreePars(X)
        elif source == 'file':
            filename = QFileDialog.getOpenFileName(None, 'Import file', '.', filter = "Chemical trees (*.ctr)")
            if filename:
                with open(filename, 'rb') as fp:
                    data = pickle.load(fp)
                # New tree and its parameters
                X = data["tree"]
                X.setTreeBook()
                pars = data["pars"]
        elif source == 'DB':
            newName, accepted = ChooseFromDBDialog.run()
            if accepted:
                X = chemNodeDB(newName)
                pars = defaultTreePars(X)
            else:
                return 0
        elif source == 'spsy':
            name = 'New spin system'
            """chsh = [parsSpec()]*2
            jcpl = [parsSpec()]
            chshAsgn = [1, 2]
            jcplAsgn = [[0, 1], [0, 0]]"""
            chsh = [parsSpec()]*1
            jcpl = []
            chshAsgn = [1]
            jcplAsgn = None
            spsy = spsySpec(chsh, jcpl, chshAsgn, jcplAsgn, mult=1)
            X = chemNodeQD(name, spsy)  # New spin system node (QD)
            for j in range(len(chsh)):
                X.addChild(chemNodeT(name + '-0.' + str(j+1), intn=chshAsgn.count(j+1),
                                        alias = name+'-'+chsh[j].label if chsh[j].label!='' else ''))      # , intn=spsy.mult
            pars = defaultTreePars(X)

        # Get the node in the parameter tree to which new chemical will be attached
        prnt = index.internalPointer()

        success = self.datum.addTreeNode(prnt.name, X, pars)

        if success:                 # self.datum.T has been updated
            self.beginInsertRows(index, 0, 0) # Parent node, first and last position

            newTP = getParsTree(self.datum.T)   # New parameter tree

            # Swap children between the old and new parameter trees
            for chld in prnt.children():
                chld.cut()
            for chld in newTP[prnt.name].children():
                prnt.addChild(chld.cut())

            self.endInsertRows()

    def remChemical(self, index):
        """Removes a chemical from the tree"""
        node = index.internalPointer()

        success = self.datum.delTreeNode(node.name)

        if success:                 # self.datum.T has been updated
            self.beginRemoveRows(self.parent(index), index.row(), index.row()) # Parent node, first and last position

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

    def resetShape(self):
        """Resets the custom lineshape."""
        self.datum.reset_shape()
        self.crntChanged.emit()

class ChemTreeView(QTreeView):
    """Model/View based class to display chemical trees."""

    changedParsList = pyqtSignal(int)        # Signalizes to update the parameters list widget and carries the index of the active step
    changedSelected = pyqtSignal(object)     # Supports signals with any data types

    class ParsSpecDialog(QDialog):
        """A dialog to set specification for a parameter."""

        def __init__(self, name, param, crntVal=None,  parent = None):
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
            self.buttons = QDialogButtonBox(
                QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
                Qt.Horizontal, self)
            layoutMain.addLayout(layoutForm)
            layoutMain.addWidget(self.chckSeries)
            layoutMain.addWidget(self.buttons)

            self.buttons.accepted.connect(self.accept)
            self.buttons.rejected.connect(self.reject)

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
        if current.isValid():
            node = current.internalPointer()
            self.changedSelected.emit(node.name)

    def rowsInserted(self, parent, start, end):
        """Is called to update the view when rows have been inserted."""
        super().rowsInserted(parent, start, end)
        self.hideExcessiveRows()

    def reset(self):
        """Subclassing the reset slot."""
        super().reset()

        self.hideExcessiveRows()
        self._copy_buffer.clear()

    def onSectionCountChanged(self, oldCount, newCount):
        """Called by the model after the number of columns is changed."""
        self.setColumnWidth(4, 50)
        for i in range(5, newCount):
            self.setColumnWidth(i, 18)

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

            if node.nodeType in ['chemNode', 'chemNodeDB']:       # Chemical node
                key = node.name

                # Add/remove node actions
                actnToggleReported = QAction(QIcon('icons\icon_blank.png'), 'Reported', self)
                actnToggleReported.setStatusTip('Change the reported state')
                actnToggleReported.setCheckable(True)
                actnToggleReported.setChecked( self.model().datum.T[key].isReported() )
                actnToggleReported.toggled.connect(lambda : self.toggleReported(key) )
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
                actnResetShape = QAction(QIcon('icons\icon_resetShape.png'), 'Reset custom shape', self)
                actnResetShape.setStatusTip('Remove the custom lineshape.')
                actnResetShape.triggered.connect(self.model().resetShape)
                actnIncreaseOrder = QAction(QIcon('icons\icon_increaseOrder.png'), 'Increase order', self)
                actnIncreaseOrder.setStatusTip('Increase the order of lineshape correction polynomial.')
                actnIncreaseOrder.triggered.connect(self.model().increaseOrder)
                actnDecreaseOrder = QAction(QIcon('icons\icon_decreaseOrder.png'), 'Decrease order', self)
                actnDecreaseOrder.setStatusTip('Decrease the order of lineshape correction polynomial.')
                actnDecreaseOrder.triggered.connect(self.model().decreaseOrder)

                popMenu.addAction(actnResetShape)
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

            T = copy.deepcopy(self.model().datum.T[node.name])           # Extract the subtree starting with the node key and set its parent to None
            T.makeRoot()
            for node in T.items(): node.reset()

            # Set tree parameters to the default values from the workspace
            pars = {}

            # Save the tree and the parameters
            saveTree(filename, T, pars)

    def showCfunPopup(self, key):
        """Plots the cost function with respect to the particular variable specified by a tuple key."""
        DDD = self.model().datum
        evalPars = copy.deepcopy(DDD.crntParsH)
        actvStep = DDD.steps[-1]
        costFuncOpti = lambda x : (DDD.getPrior(key).abs(x), DDD.evaluate(evalParsH=updateFromFlat(evalPars, [key], [DDD.getPrior(key).abs(x)]), autoKeys=actvStep.autoKeys, frqBlkIds=actvStep.frqBlkIds)[0])
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

    def hideExcessiveRows(self):
        """Traverse the tree and hide some rows"""
        def traverse(index):
            yield index
            for row in range(self.model().rowCount(index)):
                chld = self.model().index(row=row, column=0, prnt=index)     # Child of the index
                yield from traverse(chld)

        root = self.rootIndex()

        for index in traverse(root):
            if index.isValid():
                node = index.internalPointer()
                if node.name[1] == 'ampl':
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

    def copyCrntPars(self, keys):
        """Copy current values of selected parameters."""
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

    def selectActive(self, stemKey):
        """Selects an active paramter for a picked peak."""
        key = (stemKey[:stemKey.rfind('-')]+'-SPSY'+stemKey[stemKey.rfind('-')+1:stemKey.rfind('.')], 'chshQD', int(stemKey[stemKey.rfind('.')+1:])-1)
        index = self.model().indexByKey(key)
        self.selectionModel().setCurrentIndex(index, QItemSelectionModel.SelectCurrent | QItemSelectionModel.Rows)

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
        stepClass.pckdPeaks = [peakSpec(chsh=f[p[0]], fwhm=w[0] / np.pi, intn = a) for p, w, a in zip(pos, width, amps) if w[0] > 0]
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

        """if checked:
            # Plot stem diagrams
            self._T.findRoot().propPoles()     # Propagate all poles
            for i, node in enumerate(self._T.repRoots()):
                for j, leaf in enumerate(node.leaves()):
                    if leaf.uPoles.size > 0:
                        markerline, stemlines, baseline = self.ax[1].stem(leaf.uPoles.imag/(stepClass.c0*np.pi*2), leaf.qPolesIntn*leaf.intn, basefmt=" ")     # , label=node.name if j==0 else ''
                        plt.setp(stemlines, linewidth=1, color=config.colrseq[i], picker = 2)    # Picking tolerance in px
                        plt.setp(markerline, markerfacecolor = config.colrseq[i], linestyle='None', color=config.colrseq[i], markersize=2)      # , picker=self.onStemPick
                        self.allStems[leaf.name] = (markerline, stemlines)
            self.ax[1].set_xlim(settings['ax1Limits']['xlim'])     # Set the saved limits
            self.ax[1].set_ylim(settings['ax1Limits']['ylim'])     # Set the saved limits
            self.ax[1].set_navigate(False)
            self.ax[1].legend(loc=0)
            # Connect events
            self.cidScroll = self.canvas.mpl_connect('scroll_event', self.onSpectrumZoom)
            self.cidPick = self.canvas.mpl_connect('pick_event', self.onStemPick)
        elif self.cidScroll and self.cidPick:
            self.canvas.mpl_disconnect(self.cidScroll)
            self.canvas.mpl_disconnect(self.cidPick)"""

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
        stepClass.pckdPeaks = [peakSpec(chsh=f[p[0]], fwhm=w[0] / np.pi, intn = a) for p, w, a in zip(pos, width, amps) if w[0] > 0]

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

PhasingForm, PhasingFormBC = uic.loadUiType("qtFormPhasingWidget.ui")    # Load the predesigned PhasingWidget Form

class PhasingWidget(QWidget):
    """A widget that contains scrollers/buttons for phasing and that interacts with a matplotlib canvas to plot the results."""

    RANGE_MAX = 64
    RANGE_MIN = -64
    phased = pyqtSignal(float, float)        # Emmited when phasing is completed; outputs the values of ph0 and ph1 in degrees

    def __init__(self, datum, canvas, orientation='Vertical', parent=None):
        #QWidget.__init__(self)
        #PhasingForm.__init__(self)
        #

        super().__init__(parent)

        #uic.loadUi("qtFormPhasingWidget.ui", self)
        #self.setupUi(self)

        #print(self.__dict__)
        #self.sliderPh0 = self.ui.findChild(QtGui.QSlider, "sliderPh0")
        self.datum = datum
        self.canvas = canvas
        self.ax = self.canvas.figure.get_axes()

        self.p0deg = 0.0     # Phasing parameters in degrees
        self.p1deg = 0.0
        self.pivot = 0.0     # Pivot point for phasing, float in the range (0.0, 1.0)

        self.sliderPh0, self.sliderPh1 = QSlider(), QSlider()
        self.sliderPh0.setMinimumHeight(120)
        self.sliderPh1.setMinimumHeight(120)
        self.sliderPh0.setRange(self.RANGE_MIN, self.RANGE_MAX)
        self.sliderPh1.setRange(self.RANGE_MIN, self.RANGE_MAX)
        self.sliderPh0.setValue(0)
        self.sliderPh1.setValue(0)
        self.sliderPh0.valueChanged.connect(self.onPh0SliderChanged)
        self.sliderPh1.valueChanged.connect(self.onPh1SliderChanged)
        self.sliderPh0.sliderReleased.connect(self.phasingComplete)
        self.sliderPh1.sliderReleased.connect(self.phasingComplete)
        self.sliderPh0.sliderPressed.connect(self.startPlotting)
        self.sliderPh1.sliderPressed.connect(self.startPlotting)
        self.bttnSetPivot = QPushButton('Pivot')
        self.bttnSetPivot.clicked.connect(self.setPivot)
        self.bttnAutoPhase = QPushButton('Auto')
        self.bttnAutoPhase.clicked.connect(self.autoPhase)
        layout = QGridLayout()
        if orientation == 'Vertical':
            layout.setVerticalSpacing(0)
            self.sliderPh0.setOrientation(Qt.Vertical)
            self.sliderPh1.setOrientation(Qt.Vertical)
            layout.addWidget(QLabel("Phasing"), 0, 0, 1, 2, Qt.AlignCenter|Qt.AlignTop)
            layout.addWidget(QLabel("ph0"), 1, 0, Qt.AlignCenter|Qt.AlignTop)
            layout.addWidget(QLabel("ph1"), 1, 1, Qt.AlignCenter|Qt.AlignTop)
            layout.addWidget(self.sliderPh0, 2, 0, Qt.AlignCenter|Qt.AlignTop)
            layout.addWidget(self.sliderPh1, 2, 1,  Qt.AlignCenter|Qt.AlignTop)
            #layout.addWidget(self.bttnSetPivot, 3, 0, 1, 2, Qt.AlignCenter)
            layout.addWidget(self.bttnAutoPhase, 4, 0, 1, 2, Qt.AlignCenter)
            self.setMaximumWidth(80)
        else:
            self.sliderPh0.setOrientation(Qt.Horizontal)
            self.sliderPh1.setOrientation(Qt.Horizontal)
            layout.addWidget(QLabel("Phasing"), 0, 0, 1, 2, Qt.AlignCenter|Qt.AlignTop)
            layout.addWidget(QLabel("ph0"), 1, 0, Qt.AlignCenter)
            layout.addWidget(QLabel("ph1"), 2, 0, Qt.AlignCenter)
            layout.addWidget(self.sliderPh0, 1, 1, Qt.AlignCenter)
            layout.addWidget(self.sliderPh1, 2, 1,  Qt.AlignCenter)
            layout.addWidget(self.bttnSetPivot, 2, 2, Qt.AlignCenter)
            layout.addWidget(self.bttnAutoPhase, 1, 2, Qt.AlignCenter)

        self.setMaximumHeight(250)
        #self.resize(50, 200)
        self.setLayout(layout)

    def setNewDatum(self, datum):
        self.datum = datum
        self.reset()

    def onPh0SliderChanged(self, val):
        """Reads new values from the sliders ph0 and ph1 and updates the plot"""
        ph0_rel = 2*(val - self.RANGE_MIN) / (self.RANGE_MAX - self.RANGE_MIN) - 1
        self.p0deg = ph0_rel * 180.0

        self.plot()

    def onPh1SliderChanged(self, val):
        """Reads new values from the sliders ph0 and ph1 and updates the plot"""
        ph1_rel = 2*(val - self.RANGE_MIN) / (self.RANGE_MAX - self.RANGE_MIN) - 1
        p1deg_new = ph1_rel * 180.0
        if self.pivot != 0.0:
            self.sliderPh0.blockSignals(True)
            self.p0deg += self.pivot*(self.p1deg - p1deg_new)     # find the new ph0 value given the current non-zero pivoting point
            self.p0deg = (self.p0deg + 180.0) % 360.0 - 180.0     # make sure the phase stays in the (-180.0, 180.0) interval
            self.sliderPh0.setValue((self.p0deg/180.0+1)*(self.RANGE_MAX - self.RANGE_MIN)/2 + self.RANGE_MIN)
            self.sliderPh0.blockSignals(False)
        self.p1deg = p1deg_new

        self.plot()

    def setPivot(self):
        """Sets a pivoting point for phase correction."""
        self.pivot = 0.5

    def reset(self):
        """Resets the sliders to display the phasing parameters for the currently open file/step."""
        try:
            theta = self.datum.getCrntVal(('.', 'theta', 0))
            tau = self.datum.getCrntVal(('.', 'tau', 0))
        except AttributeError:
            theta, tau = 0., 0.             # If the datum is the entire Workspace

        self.pivot = 0.0
        self.p0deg = 180 * theta / np.pi
        try:
            self.p1deg = np.asscalar( tau*(self.datum.f[-1]*self.datum.c0-self.datum.f0)*360. )
        except (IndexError, AttributeError) as e:      # If self.datum.f == []
            self.p1deg = 0.

        # Set the sliders
        self.sliderPh0.blockSignals(True)
        self.sliderPh1.blockSignals(True)
        self.sliderPh0.setValue((self.p0deg/180.0+1)*(self.RANGE_MAX - self.RANGE_MIN)/2 + self.RANGE_MIN)
        self.sliderPh1.setValue((self.p1deg/180.0+1)*(self.RANGE_MAX - self.RANGE_MIN)/2 + self.RANGE_MIN)
        self.sliderPh0.blockSignals(False)
        self.sliderPh1.blockSignals(False)

    def startPlotting(self):
        self.f = self.datum.f
        self.yF = self.datum.yF
        # remember the axis settings
        if settings["ax0Limits"] is not None:
            settings["ax0Limits"] = {"xlim":self.ax[0].get_xlim(), "ylim":self.ax[0].get_ylim()}
            indxPlot = np.flatnonzero((self.datum.f<=max(settings["ax0Limits"]["xlim"]))*(self.datum.f>=min(settings["ax0Limits"]["xlim"])))
            supsRatio = math.ceil(indxPlot.size / (2**13))   # Subsampling ratio; plot no more than 2^12 points
            indxPlot = np.append(indxPlot[:-1:supsRatio], indxPlot[-1])  # Make sure that the first and the last indices of each group are included
            self.f = self.f[indxPlot]
            self.yF = self.yF[indxPlot]

        self.plot()

    def plot(self):
        """Plots the phased spectrum"""
        self.ax[0].clear()     # discards the old graph

        theta = self.p0deg * np.pi/180.
        tau = np.asscalar( self.p1deg / (self.datum.f[-1]*self.datum.c0-self.datum.f0) / 360. )
        ph = np.exp(-1j*2*np.pi * tau * (self.f*self.datum.c0-self.datum.f0) - 1j*theta ).reshape((-1,1))
        yFph = self.yF * ph
        self.ax[0].plot(self.f, yFph.real, '-', color=(0,0.58,0.86), linewidth=1.5, label='Measured data')

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

    def autoPhase(self):
        """Autophasing"""
        p0, p1 = ng.process.proc_autophase.automatic_ps(self.datum.yF.ravel(), 'acme', p0=-self.p0deg, p1=-self.p1deg)     # 'peak_minima'
        self.p0deg, self.p1deg = -p0, -p1
        self.p0deg = (self.p0deg + 180.0) % 360.0 - 180.0     # make sure the phase stays in the (-180.0, 180.0) interval
        self.pivot = 0.0
        print(self.p0deg, self.p1deg)

        theta = self.p0deg * np.pi/180.
        tau = np.asscalar( self.p1deg / (self.datum.f[-1]*self.datum.c0-self.datum.f0) / 360. )

        # Set the sliders
        self.sliderPh0.blockSignals(True)
        self.sliderPh1.blockSignals(True)
        self.sliderPh0.setValue((self.p0deg/180.0+1)*(self.RANGE_MAX - self.RANGE_MIN)/2 + self.RANGE_MIN)
        self.sliderPh1.setValue((self.p1deg/180.0+1)*(self.RANGE_MAX - self.RANGE_MIN)/2 + self.RANGE_MIN)
        self.sliderPh0.blockSignals(False)
        self.sliderPh1.blockSignals(False)

        self.startPlotting()
        self.phasingComplete()

    def phasingComplete(self):
        print("Phasing complete:", self.p0deg, self.p1deg)
        self.phased.emit(self.p0deg, self.p1deg)

class PreprocessingWidget(QWidget):
    """Handles basic preprocessing operations, e.g. zero-filling and apodization."""

    parsChanged = pyqtSignal(int, float)

    class N2SpinBox(QSpinBox):

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setMaximum(2**17)

    def __init__(self, datum, parent=None):
        super().__init__(parent)
        # Add widgets for entering parameters
        self._resetting = False
        self.editNZF = self.N2SpinBox()
        self.editNZF.editingFinished.connect(self.onParsChanged)
        self.editApod = MyDoubleEdit()
        self.editApod.valueChanged.connect(self.onParsChanged)

        formLayout = QFormLayout()
        formLayout.addRow("Zero-filling", self.editNZF)
        formLayout.addRow("Line-broadening", self.editApod)
        self.setLayout(formLayout)

        self.setMaximumWidth(300)
        #self.setLayout(layout)

        # Set the current datum and update the widgets
        self.setNewDatum(datum)

    def setNewDatum(self, datum):

        self._resetting = True
        try:
            self.datum = datum
            self.editNZF.setMinimum(self.datum.t.size)
            self.editNZF.setValue(self.datum.f.size)
            self.editApod.setValue(self.datum.apod)
        except AttributeError:
            pass
        self._resetting = False

    def onParsChanged(self):
        """Updates the settings (linebroadening, zero-filling, etc.)"""
        if not self._resetting:
            self.parsChanged.emit(self.editNZF.value(), self.editApod.value())

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

    result = pyqtSignal(object)          # Outputs the results as fileIndx, crntParsH, hat
    notify = pyqtSignal(str)             # Emits notification

    def __init__(self):
        super().__init__()
        self._exiting = False      # The exiting attribute is used to tell the thread to stop processing.
        self.stepIdsToFit = []       # Sequence of steps from the textEdit

    def __del__(self):
        """Before a Worker object is destroyed, we need to ensure that it stops processing. For this reason, we implement the following method in a way that indicates to the part of the object that performs the processing that it must stop, and waits until it does so."""
        self._exiting = True
        self.wait()

    def setExitFlag(self, flag=True):
        """Setter of the _exiting flag."""
        self._exiting = flag

    def isExiting(self):
        return self._exiting

    def fit(self, fileToFit, stepIdsToFit):
        self.fileToFit = fileToFit         # Should remember on which Datum we are currently working
        self.stepIdsToFit = stepIdsToFit   # if stepIdsToFit is not None else [i for i in range(len(self.steps))]
        self.start()    # calls self.run()   (should be called as self.start() anyway)

    def run(self):
        # Optimize
        if not self._exiting:
            for i, indx in enumerate(self.stepIdsToFit):
                print("Optimizing step No. {:d} ({:d}/{:d})".format(indx+1, i+1, len(self.stepIdsToFit)))
                step = self.fileToFit.steps[indx]
                # Update the custom lineshape if requested
                if step.fitCustomLshape:
                    self.fileToFit.set_shape(frqBlkIds=step.frqBlkIds)
                # Fit the model parameters
                self.fileToFit.optimize(parsKeys=step.parsKeys, autoKeys=step.autoKeys, frqBlkIds=step.frqBlkIds, evaluatePriors=False)
                # Finish fitting and return the results
                self.result.emit(self.fileToFit.crntParsH)

class MySpecPlot(FigureCanvas):

    def __init__(self, figure):
        pass

class MainView(QMainWindow):
    """Main GUI form class."""

    def __init__(self, wsp, parent = None):
        # Initialize with some workspace
        self.wsp = wsp    # The Workspace; main class that holds all logic
        self._crnt = self.wsp     # Currently opened Series/Datum/or the entire Workspace

        # Global settings
        settings.update({"ax0Limits": None,
                        "ax1Limits": {'ylim':(0, 5)},
                        "ax2Limits": None,
                        "startFromPars" : "current",          # Starting values of parameters when fitting multiple files (current, previous, default) - will be copied from crntParsH of this file, previous file or dfltParsH
                        "autoPhase" : False,
                        "autoPick" : False})
        self.fittingQueue = []    # List of file indices to be fitted in that order from last to first; if [], the current file will be fitted
        self.indxCrntFile = None  # Currently opened and displayed file
        self.cidScroll = None
        self.cidPick = None

        # initialize the main window
        super(MainView, self).__init__(parent)
        self.resize(1300, 800)

        self.setupGUI()

        # Start the fitting thread
        self.fittingThread = FittingThread()
        self.fittingThread.finished.connect(self.onFittingFinished)
        self.fittingThread.terminated.connect(self.onFittingFinished)
        self.fittingThread.result.connect(self.onReceivedResults)
        self.fittingThread.notify.connect(self.onNotification)

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
        self.setWindowTitle("Model-based quantitative NMR analysis ver. {} ({})".format(version, str(date.today())) )

        # ------------- set the main diagram --------------
        #rc('font', **{'family' : 'sans-serif', 'weight' : 'normal', 'size'   : 10})
        rcParams['pdf.fonttype'] = 42 # pdf.fonttype : 42 # Output Type 3 (Type3) or Type 42 (TrueType)
        rcParams['ps.fonttype'] = 42
        SMALL_SIZE = 11
        MEDIUM_SIZE = 12
        BIGGER_SIZE = 14

        rc('font', size=SMALL_SIZE)          # controls default text sizes
        #rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        #rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        #rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        #rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        #rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        #rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        # ----------------- set up the spectrum figure
        self.figure = Figure(facecolor='w', edgecolor='k')     # a figure instance to plot on    # figsize=(1, 1), dpi=80,
        self.canvas = FigureCanvas(self.figure) # this is the Canvas Widget that displays the `figure`; it takes the `figure` instance as a parameter to __init__
        self.figureGrid  = gridspec.GridSpec(2, 1, hspace = 0.03, left = 0.05, right = 0.95, height_ratios=[3, 1])
        self.ax = [None, None, None]
        self.ax[0] = self.figure.add_subplot(111)
        self.ax[0].invert_xaxis()
        # Set the second axis on the first subplot
        self.ax[1] = self.ax[0].twinx()    # Separate vertical axis for a histogram plot
        self.ax[1].set_ylim((0, 5))
        #self.ax[1].set_visible(False)
        # Set the second subplot for the residues
        self.ax[2] = self.figure.add_subplot(self.figureGrid[1], sharex = self.ax[0])    # create axes
        self.ax[2].set_visible(False)
        # Make the main plot span the entire figure when the residuals are not shown
        self.ax[0].set_position(self.figureGrid[0:2].get_position(self.figure))
        self.ax[1].set_position(self.figureGrid[0:2].get_position(self.figure))
        # Set span selector
        self.freqRangeSelector = SpanSelector(self.ax[1], self.addFreqBlock, 'horizontal', useblit=True, minspan=0.01,
                     rectprops=dict(alpha=0.15, facecolor='yellow'))     # set useblit True on gtkagg for enhanced performance
        self.freqRangeSelector.active = False

        # ----------------- set up the pie chart figure
        self.pieFigure = Figure(facecolor='w', edgecolor='k')     # a figure instance to plot on
        self.pieCanvas = FigureCanvas(self.pieFigure)# this is the Canvas Widget that displays the `figure`; it takes the `figure` instance as a parameter to __init__
        self.ax_pie = self.pieFigure.add_subplot(111)    # create axes

        # ------------------ 2. Set up the chemical tree ----------------------
        self.treeView = ChemTreeView()
        self.treeModel = ChemTreeModel(self.wsp)
        self.treeView.setModel(self.treeModel)
        self.treeModel.crntChanged.connect(lambda:self.tryStep(None))       # If current values are changed by the user
        self.treeView.changedSelected.connect(self.selectStems)             # If new parameter is selected by the user

        # create a text edit widget to choose the optimization sequence
        self.stepsEdit = QPlainTextEdit('Please enter a sequence of steps to fit. ALL steps will be fitted consecutively by default.')  # , e.g.: 1, A, 5, (3, 4, A, 1), 2
        self.stepsEdit.setMaximumHeight(50)

        # ------------------ 3. Navigation and processing plane ---------------
        # Navigation tree in the navigation tab
        self.naviTreeView = NavigationTreeView()
        self.naviTreeModel = NavigationTreeModel(self.wsp)
        self.naviSelection = QItemSelectionModel(self.naviTreeModel)
        self.naviTreeView.setModel(self.naviTreeModel)
        self.naviTreeView.setSelectionModel(self.naviSelection)
        self.naviSelection.currentChanged.connect(self.onCurrentSelectedChanged)
        self.naviTreeView.requestPasteCrnt.connect(self.treeView.pasteCrntPars)     # Paste copied parameter values to all selected Datums in the naviTreeView
        self.naviTreeView.requestPasteDflt.connect(self.treeView.pasteDfltPars)
        self.naviTreeView.requestfitSelected.connect(self.fitAllSteps)

        # A table to display frequency ranges
        self.freqTableView = FreqTableView()
        self.freqTableModel = FreqTableModel(self.wsp)
        self.freqTableView.setModel(self.freqTableModel)
        self.freqTableModel.freqBlockChanged.connect(self.plotCurrent)

        # Add the preprocessing parameters tool
        ortnNaviTab = 'Vertical'    # set to 'Horizontal' if displayed on the right
        self.preprocTool = PreprocessingWidget(self._crnt)
        self.preprocTool.parsChanged.connect(self.resetSignals)
        self.phasingTool = PhasingWidget(self._crnt, self.canvas, orientation=ortnNaviTab)
        self.phasingTool.phased.connect(self.onPhased)

        # Create the tab and put everything together
        lay0 = QHBoxLayout()
        lay1 = QVBoxLayout()
        naviLayout = QVBoxLayout() if ortnNaviTab == 'Horizontal' else QHBoxLayout()
        self.naviTreeView.setMinimumWidth(150)
        lay0.addWidget(self.naviTreeView)
        lay0.addLayout(lay1)
        lay1.addWidget(QLabel("Parameters of the series"), alignment=Qt.AlignCenter)
        lay1.addWidget(self.freqTableView)
        lay1.addWidget(self.preprocTool)
        naviLayout.addLayout(lay0)
        #naviLayout.addWidget(self.phasingTool)
        tabNavi = QWidget()
        tabNavi.setLayout(naviLayout)
        tabNavi.setMaximumWidth(400)
        tabNavi.setMaximumHeight(250)


        self.setupActions()

        # Create the parameters tab

        self.paramLayout = QVBoxLayout()
        self.paramLayout.addWidget(self.treeView)
        self.paramLayout.addWidget(self.stepsEdit)
        self.paramWidget = QWidget()
        self.paramWidget.setLayout(self.paramLayout)
        self.printoutEdit = QPlainTextEdit()
        self.printoutEdit.setReadOnly(True)
        tabParam = QSplitter()
        tabParam.addWidget(self.paramWidget)
        tabParam.addWidget(self.printoutEdit)
        tabParam.setSizes([500, 120])
        tabParam.setOrientation(Qt.Vertical)

        # ----------------------- Tabs on the right ----------------------------
        self.tabsNavi = QTabWidget()
        self.tabsNavi.setTabPosition(QTabWidget.North)
        #self.tabsNavi.addTab(tabNavi, 'Navigation')
        self.tabsNavi.addTab(tabParam, 'Parameters')
        self.tabsNavi.setCurrentIndex(0)

        # ---------------------- Set up the status bar -------------------------
        self.statusBar= QStatusBar()
        self.statusBar.setMaximumHeight(16)
        self.setStatusBar(self.statusBar)
        self.progressBarFiles = QProgressBar()
        self.progressBarFiles.setMaximumHeight(16)
        self.progressBarFiles.setMaximumWidth(400)
        self.statusBar.addPermanentWidget(self.progressBarFiles)

        # Setup the analysis tab
        self.paramsTool = ParameterDisplayWidget()
        self.resLayout = QGridLayout()
        #self.resLayout.addWidget(self.paramsTool, 0, 0)
        self.resLayout.addWidget(tabNavi, 0, 0)
        self.resLayout.addWidget(self.phasingTool, 0, 1)
        self.resLayout.addWidget(self.pieCanvas, 0, 2)
        self.resLayout.setColumnStretch(0, 3)
        self.resLayout.setColumnStretch(1, 1)

        # Setup the processing tab (phase correction/baseline/etc...)
        #self.pickingTool = PeakPickingWidget(self.canvas)
        #self.pickingTool.assigned.connect(self.onAssigned)
        self.procLayout = QGridLayout()
        #self.procLayout.addWidget(self.phasingTool, 1, 0, alignment=Qt.AlignLeft)
        #self.procLayout.addWidget(self.pickingTool, 0, 1, 2, 1, alignment=Qt.AlignLeft)

        # set tabs
        self.tabsMain = QTabWidget()
        self.tabsMain.setTabPosition(QTabWidget.West)
        tab1, tab2 = QWidget(), QWidget()
        tab1.setLayout(self.procLayout)
        tab2.setLayout(self.resLayout)
        #self.tabsMain.addTab(tab1, 'Processing')
        self.tabsMain.addTab(tab2, 'Analysis')
        self.tabsMain.setCurrentIndex(1)

        # set the Left layout (VBox: spectrum figure and tabs)
        self.leftLayout = QVBoxLayout()
        self.leftLayout.addWidget(self.canvas)
        self.leftLayout.addWidget(self.tabsMain)
        #self.leftLayout.addLayout(self.resLayout)
        self.leftLayout.setStretch(0, 5)
        self.leftLayout.setStretch(1, 3)

        # With QSplitter
        self.mainWidgetLeft = QWidget()
        self.mainWidgetLeft.setLayout(self.leftLayout)
        self.mainSplitter = QSplitter()
        self.mainSplitter.addWidget(self.mainWidgetLeft)
        self.mainSplitter.addWidget(self.tabsNavi)
        self.mainSplitter.setSizes([450, 200])
        self.mainSplitter.setHandleWidth(1)
        self.setCentralWidget(self.mainSplitter)

    def setupActions(self):
        # Add clear action
        clearAction = QAction(QIcon('icons\icon_new.png'), 'Clear workspace', self)
        clearAction.setStatusTip('Clear the workspace')
        clearAction.triggered.connect(lambda : self.onResetWspAction(newSettings=None, newWorkspace=None))
        # Add import datafile action
        actnImportData = QAction(self._icon('icon_addFile.png'), 'Import files', self)
        actnImportData.setStatusTip('Import new data and add them to the current series')
        actnImportData.triggered.connect(lambda : self.naviTreeModel.importData(parent=self._crnt))
        actnRemoveCurrent = QAction(self._icon('icon_removeFile.png'), 'Remove file', self)
        actnRemoveCurrent.setStatusTip('Remove file from the workspace')
        actnRemoveCurrent.triggered.connect(self.removeCurrent)
        showSettingsAction = QAction(self._icon('icon_settings.png'), 'Settings', self)
        showSettingsAction.setStatusTip('Show settings dialog')
        showSettingsAction.triggered.connect(self.showSettingsDialog)
        # Add load Tree action
        actnLoadTree = QAction(self._icon('icon_hierarchy.png'), 'Load tree', self)
        actnLoadTree.setStatusTip('Load a chemical tree')
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
        # ----------------------- Actions for the tree -------------------------

        # Add step
        actnAddStep = QAction(self._icon('icon_addStep.png'), 'Add step', self)
        actnAddStep.setStatusTip('Add an optimization step')
        actnAddStep.triggered.connect(self.treeModel.addStep)
        # Delete step
        actnDelStep = QAction(self._icon('icon_delStep.png'), 'Remove active step', self)
        actnDelStep.setStatusTip('Remove an optimization step')
        actnDelStep.triggered.connect(self.treeModel.delStep)
        # Toggle LS/TLS
        self.actnToggleTLS = QAction(self._icon('icon_TLS.png'), 'Use TLS algorithm', self)
        self.actnToggleTLS.setStatusTip('Use the Total Least Squares algorithm')
        self.actnToggleTLS.setCheckable(True)
        def onToggleTLS():
            config.SAMPL_funcType = 'TLS' if self.actnToggleTLS.isChecked() else 'LS'
            self.treeModel.notifyDataChanged()     # Update the TLS line in the chemTree view
        self.actnToggleTLS.triggered.connect(onToggleTLS)
        # Fit the last step action
        self.actnFitLastStep = QAction(self._icon('icon_fitOneStep.png'), 'Fit last step', self)
        self.actnFitLastStep.setStatusTip('Fit the last step')
        self.actnFitLastStep.triggered.connect(lambda:self.fitStep(indx = -1))
        # Sample action
        self.actnSample = QAction(self._icon('icon_sample.png'), 'Sample last step with MCMC', self)
        self.actnSample.setStatusTip('Sample parameters checked on the last step with the MCMC algorithm')
        self.actnSample.triggered.connect(lambda:self.sampleStep(indx = -1))
        # Report without sampling action
        self.actnReport = QAction(self._icon('icon_report.png'), 'Report results without sampling', self)
        self.actnReport.setStatusTip('Report results without sampling')
        self.actnReport.triggered.connect(lambda:self.sampleStep(indx = -1, onlyAutoKeys=True))
        # Fit all steps action
        self.actnFitAllSteps = QAction(self._icon('icon_fitAllSteps.png'), 'Fit all steps', self)
        self.actnFitAllSteps.setStatusTip('Fit all steps for this file')
        self.actnFitAllSteps.triggered.connect(lambda _ : self.fitAllSteps(selectedFiles = None))
        # Fit all files action
        actnStopFitting = QAction(self._icon('icon_stopFitting.png'), 'Stop fitting', self)
        actnStopFitting.setStatusTip('Stop fitting')
        actnStopFitting.triggered.connect(self.stopFitting)
        # Stop fitting action
        self.actnFitAllFiles = QAction(self._icon('icon_fitAllFiles.png'), 'Fit all files', self)
        self.actnFitAllFiles.setStatusTip('Fit all steps for this file')
        self.actnFitAllFiles.triggered.connect(self.fitAllFiles)
        # Save current results
        actnSaveResults = QAction(self._icon('icon_saveResults.png'), 'Save results to file', self)
        actnSaveResults.setStatusTip('Save all current results to file')
        actnSaveResults.triggered.connect(self.saveResults)
        # ------------------- Plotting actions ---------------------------------
        self.actnShowStems = QAction(self._icon('icon_showStems.png'), 'Show model peaks', self)
        self.actnShowStems.setStatusTip('Show model peaks')
        self.actnShowStems.setCheckable(True)
        self.actnShowStems.toggled.connect(self.showStems)
        self.actnShowComponents = QAction(self._icon('icon_showComponents.png'), 'Show model components', self)
        self.actnShowComponents.setStatusTip('Show model components')
        self.actnShowComponents.setCheckable(True)
        self.actnShowComponents.toggled.connect(self.plotCurrent)
        self.actnPlotResidual = QAction(self._icon('icon_plotResidual.png'), 'Plot residual', self)
        self.actnPlotResidual.setStatusTip('Plot residual')
        self.actnPlotResidual.setCheckable(True)
        self.actnPlotResidual.toggled.connect(self.plotCurrent)

        # ------------------------- set the menubar ----------------------------
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(actnImportData)
        fileMenu.addAction(actnRemoveCurrent)
        fileMenu.addAction(loadAction)
        fileMenu.addAction(saveAction)
        fileMenu.addAction(exitAction)

        # ------------------------- Set the toolbar ----------------------------
        tbMain = self.addToolBar("File")
        tbMain.addAction(clearAction)
        tbMain.addAction(actnImportData)
        tbMain.addAction(actnRemoveCurrent)
        tbMain.addAction(showSettingsAction)
        tbMain.addSeparator()
        tbMain.addAction(loadAction)
        tbMain.addAction(saveAction)
        self.addToolBar(CustomToolbar(self.canvas, self, coordinates=False))    # Figure toolbar
        tbTree = self.addToolBar("Tree")               # Tree toolbar
        tbTree.addAction(actnLoadTree)
        self.cmboxHCSelector = QComboBox()
        self.cmboxHCSelector.addItem("1H")
        self.cmboxHCSelector.addItem("13C")
        self.cmboxHCSelector.currentIndexChanged.connect(self.onHCSelect)
        #tbTree.addWidget(self.cmboxHCSelector)
        tbTree.addAction(actnAddStep)
        tbTree.addAction(actnDelStep)
        tbTree.addSeparator()
        tbTree.addAction(self.actnToggleTLS)
        tbTree.addAction(self.actnFitLastStep)
        tbTree.addAction(self.actnFitAllSteps)
        tbTree.addAction(self.actnFitAllFiles)
        tbTree.addAction(actnStopFitting)
        tbTree.addAction(self.actnSample)
        tbTree.addAction(self.actnReport)
        tbTree.addSeparator()
        tbTree.addAction(actnSaveResults)

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
                newCrnt = self.wsp

        self._crnt = newCrnt

        # Compute the model signal if there is None
        try:
            if self._crnt.zF is None:
                step = self._crnt.steps[-1]
                self._crnt.evaluate(frqBlkIds=step.frqBlkIds, autoKeys=step.autoKeys, returnSignals=True)
                #TODO: Possibly check which step to evaluate if steps use different frequency ranges
        except AttributeError:
            pass

        # Update the necessary display widgets
        # TODO!
        self.treeModel.setNewDatum(self._crnt)
        self.freqTableModel.setNewDatum(self._crnt)
        self.preprocTool.setNewDatum(self._crnt)
        self.phasingTool.setNewDatum(self._crnt)

        # TODO!
        # Highlight the current Datum in the Navigation widget if it was seletec programmatically
        #ids = self._crnt.selfID()
        #print(ids)
        #ser_index = self.naviTreeModel.index(ids[0], 0, None)    # Index corresponding to the Series
        #dat_index = self.naviTreeModel.index(ids[1], 0, ser_index)    # Index corresponding to the Datum
        #self.naviSelection.setCurrentIndex(dat_index, QItemSelectionModel.Select)

        settings["ax0Limits"] = None
        self.plotCurrent()

    def onCurrentSelectedChanged(self, index):
        """Slot for the signal indicating the change in the currently selected Series/Datum"""

        self.setCurrent(newCrnt = index.internalPointer() if index.isValid() else None)

    def loadChemTree(self):
        """Calls a dialog and loads a new chemical tree from file."""
        filename = QFileDialog.getOpenFileName(self, 'Import file', '.', filter = "Chemical trees (*.ctr)")
        if filename:
            with open(filename, 'rb') as fp:
                data = pickle.load(fp)

        #data["pars"]["."] = self.wsp.dfltParsH["."]     # Keep the values for tau and theta used before
        T = data["tree"]
        T.setTreeBook()
        pars = data["pars"]

        # NOTE: Technically, this is unsafe. The tree should be better reset from the treeModel rather than in the workspace and then the module updated.....
        self.wsp.setTree(T)
        self.treeModel.resetChemTree()

    def onHCSelect(self, val):
        """Selects a mode 1H/13C."""
        newMode = '1H' if val == 0 else '13C'

        self.wsp.setHCmode(newMode)
        self.treeModel.resetChemTree()

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
                pickle.dump([dataPack, stngPack], fp)

    def onLoadWspAction(self):
        """Loads the workspace including the stepClass class and the steps array."""
        filename = QFileDialog.getOpenFileName(self, 'Import file', '.', filter = "NMR worksapce (*.wsp)")
        if filename:
            with open(filename, 'rb') as fp:
                dataUnPack = pickle.load(fp)

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

        # Reset the workspace
        self.wsp.reset()
        if newWorkspace is not None:
            self.wsp.unpack(newWorkspace)

        # Reset the tree model
        self.treeModel.fullReset(self._crnt)

        self.naviTreeModel.endResetModel()

        self.setCurrent()          # Sets the current display to the first Datum or the entire workspace if there is no Datum

        # Update the plots
        self.plotCurrent()

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

    def resetSignals(self, nf, apod):
        self._crnt.resetFreqs(nf, apod)
        self.plotCurrent()

    def onPhased(self, ph0, ph1):
        """Gets the phasing values from the phasing tool widget and sets current parameters accordingly."""
        # Update current parameter; convert phases from degrees to radians for ph0 (theta) and sec for ph1 (tau)
        theta = ph0*np.pi/180.
        tau = np.asscalar( ph1/(self._crnt.f[-1]*self._crnt.c0-self._crnt.f0)/360. )
        self._crnt.setCrntVal(('.', 'theta', 0), theta)
        self._crnt.setCrntVal(('.', 'tau', 0), tau)

        self.treeModel.notifyDataChanged()
        self.plotCurrent()

    def showSettingsDialog(self):
        """Shows an input dialog and updates settings"""
        accepted = SettingsDialog.run()

    def onAssigned(self):
        """Updates the tree and plots when model peaks have been assigned to picked peaks."""
        self.plotCurrent()
        self.treeWidget.showStep()
        #self.phasingTool.setVals(self.steps[0])
        #dataFiles[self.indxCrntFile].update()

    # ------------------ Working with the fitting thread -----------------------

    def fitQueue(self, stepIdsToFit = None, autoPhase=False, autoPick=False):
        """Fits the files in the self.fittingQueue list. Must be called only when appropriate self.fittingQueue is set."""
        # Disable controls that can start fitting
        #self.actnFitAllSteps.setDisabled(True)
        #self.actnFitAllFiles.setDisabled(True)
        #self.actnFitLastStep.setDisabled(True)

        # Import new datafile if necessary
        fileToFit = self.fittingQueue.pop(0)         # Selects a file to fit fro the beginning of fittingQueue. Possibly empties the fittingQueue

        # Determine the starting values of parameters for the next file in the fittingQueue and KEEP the current values if necessary
        if config.OPTIM_startFrom == "previous":
            sid = fileToFit.selfID()
            # Check if the current file is not the first one in the Series. If possible use parameters of the previous file, otherwise keep the current parameters.
            if sid[1] > 0:
                fileToFit.resetCrntPars(crntParsH = copy.deepcopy(self._crnt.series[sid[0]].data[sid[1]-1].crntParsH) )
        elif config.OPTIM_startFrom == "default":
            fileToFit.resetCrntPars()   # Reset to defaults
        else: # i.e. settings["startgFromPars"] == "current"
            pass     # Don't do anything; the file will be loaded with its current parameters, and the optimization will start from them

        # Phase and pick peaks if necessary
        if autoPhase:
            self.phasingTool.autoPhase()
        if autoPick:
            pass
            #self.pickingTool.autoPick()
            #self.pickingTool.assignPeaks()

        # Select which steps to fit
        if stepIdsToFit is None: stepIdsToFit = [-1]        # Fit the last step by default

        # Call the fitting thread
        self.fittingThread.setExitFlag(False)
        self.fittingThread.fit(fileToFit, stepIdsToFit)

    def onNotification(self, text):
        """Displays a notifications from the fitting thread."""
        #self.progressBarSteps.setValue(self.progressBarSteps.value()+1)
        pass

    def onReceivedResults(self, crntParsH):
        """A function that recieves results from the fittingThread. Called before onFittingFinished"""
        pass
        #dataFiles[indxFile].foundParsH = crntParsH
        #dataFiles[indxFile].xFph = hat['xFph']
        #dataFiles[indxFile].mdldPeaks = copy(stepClass.mdldPeaks
        #dataFiles[indxFile].pckdPeaks = copy(stepClass.pckdPeaks

    def onFittingFinished(self):
        """Called when the fittingThread finishes processing. Depending if there are files in queue, may call the fitQueue function again or just display the results."""

        self.saveResults()    # Writes results to the file and shows them on screen

        if self.fittingQueue != [] and not self.fittingThread.isExiting():
            # Fitting several files
            self.progressBarFiles.setValue(self.progressBarFiles.value()+1)
            self.fitQueue(stepIdsToFit = self.fittingThread.stepIdsToFit)    # Continue fitting with the next file using the same fitting order
        else:
            # Fitting only a single (or the last) file; all done now. Reset the widgets
            self.plotCurrent()
            self.treeModel.notifyDataChanged()
            self.phasingTool.reset()
            #self.pickingTool.reset()

            # Enable controls that can start fitting again
            self.actnFitAllSteps.setEnabled(True)
            self.actnFitAllFiles.setEnabled(True)
            self.actnFitLastStep.setEnabled(True)
            self.progressBarFiles.setValue(self.progressBarFiles.maximum())

    def stopFitting(self):
        """Stops fitting in the thread."""
        self.fittingThread.setExitFlag(True)
        self.fittingThread.quit()

    def tryStep(self, indx=None):
        if indx is None: indx = -1        # Fit the active step by default
        step = self._crnt.steps[indx]
        if step.fitCustomLshape:
            self._crnt.set_shape(frqBlkIds=step.frqBlkIds)
        self._crnt.evaluate(frqBlkIds=step.frqBlkIds, autoKeys=step.autoKeys, returnSignals=True)
        self.plotCurrent()
        self.treeModel.notifyDataChanged()

    def sampleStep(self, indx=None, onlyAutoKeys=False):
        if indx is None: indx = -1        # Fit the active step by default
        step = self._crnt.steps[indx]
        samples = self._crnt.sample(frqBlkIds=step.frqBlkIds, parsKeys=None if onlyAutoKeys else step.parsKeys, autoKeys=step.autoKeys, evaluatePriors=True, nwalkers=None, nsteps=250)     # parsKeys=step.parsKeys
        reportMCMC(samples)

    def fitStep(self, indx=None):
        """Fits a single step specified by its indx or the active column. By default, fit the last step."""
        stepIdsToFit = [-1] if indx is None else [indx]       # Fit the active step by default

        # Set up the progress bars
        self.progressBarFiles.setRange(0, 1)
        self.progressBarFiles.setValue(0)

        # Call the fitting function
        self.fittingQueue = [self._crnt]
        self.fitQueue(stepIdsToFit = stepIdsToFit)

    def fitAllSteps(self, selectedFiles = None):
        """Fits all steps in selected files; if no files are selected, uses the current file/series. The starting values on the next step are copied from the current found values."""
        # Form the list of steps to Fit
        s = self.stepsEdit.toPlainText()
        if re.search('[0-9A]', s) is None: s = 'A'    # Fit all steps if the string is missing any numerical characters or A's
        s = " ".join(re.split("(A)", s ))      # Prevent any consecutive A's from occuring in the string; separate them with spaces
        while s.find('(') != -1:    # Randomize all elements in all parentheses
            beg, end = s.find('('), s.find(')')
            R = re.split("[^0-9A]+", s[beg+1:end])
            R = [i for i in R if i != 'A'] + [str(i+1) for i in range(len(self._crnt.steps))]*R.count('A')   # Turn A's into lists of numbers and add them to the array
            shuffle(R)
            s = ", ".join((s[:beg], *R, s[end+1:]))
        L = re.split("[^0-9A]+", s)   # regex matches any non-digit character followed by any number (1+) of non-digit characters
        stepIdsToFit = [int(j)-1 for c in L for j in {'A':[str(i+1) for i in range(len(self._crnt.steps))]}.get(c, [c]) if j != '']    # replace 'A' with the list of all items

        # Set up the fitting queue making sure that there are no repeated files
        if selectedFiles is None:
            selectedFiles = [self._crnt]         # Fit all steps of the current file only
        selectedIDs = [ddd.selfID() for ddd in selectedFiles if isinstance(ddd, Datum)] \
                    + [ddd.selfID() for sss in selectedFiles for ddd in sss.data if isinstance(sss, Series)]      # Expand all Series
        selectedIDs = sorted(list(set(selectedIDs)))
        self.fittingQueue = [self._crnt.series[sid[0]].data[sid[1]] for sid in selectedIDs]

        # Set up the progress bar
        self.progressBarFiles.setRange(0, len(self.fittingQueue))
        self.progressBarFiles.setValue(0)

        # Call the fitting function
        self.fitQueue(stepIdsToFit = stepIdsToFit)

    def fitAllFiles(self):
        """Fits all steps for all Files. The starting values on the next step are copied from the current found values. Starting values for each file are determined by the settings and are set in the self.fitQueue function."""
        # Call the fitting function. It is important to make fittingQue as a copy of self._crnt.data, because items will be popped from it
        if isinstance(self._crnt, Series):
            selected = [i for i in self._crnt.data]
        elif isinstance(self._crnt, Datum):
            selected = [i for i in self._crnt.parent.data]
        else: return 0
        self.fitAllSteps(selected)

    def saveResults(self):
        """Saves the current results of computation into the file and prints them on screen."""
        # Write the results to a file
        tab = []
        for sss in self._crnt.series:
            for ddd in sss.data:
                row = [sss.name, ddd.name, '{:.6f}'.format(np.linalg.norm(ddd.yT))]
                ampl, vars, names = [], [], []
                for name in self.repRootNames:
                    try:
                        row += ['{:.6f}'.format(ddd.smplDistF[(name, 'ampl', 0)].mean), '{:.6f}'.format(ddd.smplDistF[(name, 'ampl', 0)].var)]
                        if name != 'Water':
                            ampl.append(ddd.smplDistF[(name, 'ampl', 0)].mean)
                            vars.append(ddd.smplDistF[(name, 'ampl', 0)].var)
                            names.append(name)
                    except KeyError:
                        row += ['{:.6f}'.format(ddd.crntParsH[name]['ampl'][0]), '---']
                        if name != 'Water':
                            ampl.append(ddd.crntParsH[name]['ampl'][0])
                            vars.append(0.0)
                            names.append(name)

                # Compute mole fractions
                ampl = np.array(ampl).ravel()
                vars = np.array(vars).ravel()
                m_tot = np.sum(ampl)      # Total intensity
                v_tot = np.sum(vars)      # Total variance of the intensity estimate
                mfrac = ampl / m_tot
                confi = 2 * mfrac * np.sqrt(vars/(ampl**2) + v_tot/(m_tot**2))
                row += [f for mf, ci in zip(mfrac, confi) for f in ( '{:.6f}'.format(mf), '{:.6f}'.format(ci) )]

                tab.append(row)
        head = ['Series', 'Filename', 'Signal norm'] + [f for name in self.repRootNames for f in (name, 'var')] + [f for name in names for f in ('x_'+name, '95% cred.i.')]
        with open('_results.txt', 'w') as fout:
            print(tabulate.tabulate(tab, headers=head), file=fout)        # write results to a text file ...
        #print(tabulate.tabulate(tab[self._crnt.data.index(self._crnt)], headers=head))    # ... and show on the screen

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
    def plotCurrent(self):
        """Plots signals corresponding to the currently opened file and current parameters."""
        # remember the axis settings
        if settings["ax0Limits"] is not None:
            settings["ax0Limits"] = {"xlim":self.ax[0].get_xlim(), "ylim":self.ax[0].get_ylim()}
        self.ax[0].clear()     # discards the old graph
        self.ax[2].clear()     # discards the old residuals graph

        if isinstance(self._crnt, Series):
            pass
        elif isinstance(self._crnt, Datum):
            DDD = self._crnt
            ph = np.exp(-1j*2*np.pi * DDD.crntParsH["."]["tau"][0] * (DDD.f*DDD.c0-DDD.f0) - 1j*DDD.crntParsH["."]["theta"][0] ).reshape((-1,1))
            yFph = DDD.yF * ph
            if DDD.zF is not None:
                zF = DDD.zF * np.array([DDD.crntParsH[name]['ampl'][0] for name in DDD.repRootNames]).reshape(1, -1)
                xF = zF.sum(1).reshape(-1,1)
                if DDD.bF is not None:
                    bF = DDD.bF
                    xF += bF

            else: zF, xF, bF = None, None, None

            inRange, outRange = splitFreq([DDD.freqBlocks[blk] for blk in DDD.steps[0].frqBlkIds], f=DDD.f)
            rmsResidual = 0.0
            if outRange:
                supsRatio = math.ceil(yFph.size / (2**13))   # Subsampling ratio; take no more than 2^13 points
                allIndx = [np.append(r.indxFreq[:-1:supsRatio], r.indxFreq[-1]) for r in outRange]   # Make sure that the first and the last indices of each group are included
                gapsPos = np.cumsum([r.size for r in allIndx])        # Positions of gaps
                allIndx = np.concatenate(allIndx)
                f_outR = np.insert(DDD.f[allIndx], gapsPos, None)

                # Plot measured data
                yF_outR = np.insert(yFph[allIndx], gapsPos, None)
                self.ax[0].plot(f_outR, yF_outR.real, '-', color=(0,0.58,0.86), linewidth=1.5, label='Measured data')

                # Plot the fitted model
                if xF is not None:
                    xF_outR = np.insert(xF[allIndx], gapsPos, None)
                    self.ax[0].plot(f_outR, xF_outR.real, '-', color='r', label='Fitted model')

                    # Plot the residuals
                    if self.actnPlotResidual.isChecked():
                        self.ax[2].plot(f_outR, yF_outR.real - xF_outR.real, '-', color='darkkhaki')

            if inRange:
                allIndx = np.concatenate([r.indxFreq for r in inRange])
                gapsPos = np.cumsum([r.indxFreq.size for r in inRange])
                f_inR = np.insert(DDD.f[allIndx], gapsPos, None)

                # Plot measured data
                yF_inR = np.insert(yFph[allIndx], gapsPos, np.nan)     #  - 1*step.bFph[allIndx]
                self.ax[0].plot(f_inR, yF_inR.real, '-', color=(0,0.58,0.86), linewidth=1.5, label='')

                # Plot the model components
                if self.actnShowComponents.isChecked():
                    if zF is not None:
                        zF = (zF + 1*bF)
                        zF_inR = np.insert(zF[allIndx, :], gapsPos, None, axis=0)
                        for i, node in enumerate(DDD.repRootNames):
                            self.ax[0].plot(f_inR, zF_inR[:, i], '-', linewidth=0.5, color=config.colrseq[i], label=node)

                # Plot the fitted model
                if xF is not None:
                    xF_inR = np.insert(xF[allIndx], gapsPos, None)       #  - 1*step.bFph[allIndx]
                    self.ax[0].plot(f_inR, xF_inR.real, '-', color='r', label='')

                    # Plot the residuals
                    if self.actnPlotResidual.isChecked():
                        self.ax[2].plot(f_inR, yF_inR.real - xF_inR.real, '-', color='darkkhaki')
                        rmsResidual += np.sqrt(np.nanmean(np.abs(yF_inR - xF_inR)**2))

            # Show or hide stems depending on the state of the checkable action self.actnShowStems
            if self.actnShowStems.isChecked():
                self.showStems(True)

            # Show the residuals plot below the graph
            if self.actnPlotResidual.isChecked():
                self.ax[2].set_visible(True)
                # Rearrange the canvas (resize the main plot)
                self.ax[0].set_position(self.figureGrid[0].get_position(self.figure))
                self.ax[1].set_position(self.figureGrid[0].get_position(self.figure))
                # Set ticks and labels
                plt.setp(self.ax[0].get_xticklabels(), visible=False)
                self.ax[0].set_xlabel('')
                self.ax[0].ticklabel_format(scilimits=(-3,3))
                self.ax[2].ticklabel_format(scilimits=(-3,3))
                self.ax[2].set_xlabel('Chemical shift, ppm', horizontalalignment='right', x=1.0)
                # Show RMS of the residual
                self.ax[2].text(0.01,0.92, "RMS = {:.4g}".format(rmsResidual), fontsize=10,
                                horizontalalignment='left', verticalalignment='top', transform = self.ax[2].transAxes)
            else:
                self.ax[2].set_visible(False)
                self.ax[0].set_position(self.figureGrid[0:2].get_position(self.figure))
                self.ax[1].set_position(self.figureGrid[0:2].get_position(self.figure))
                self.ax[0].ticklabel_format(scilimits=(-3,3))
                self.ax[0].set_xlabel('Chemical shift, ppm', horizontalalignment='right', x=1.0)

            # Plot optimization limits
            for i, blk in enumerate(DDD.freqBlocks):
                self.ax[0].axvspan(blk.min, blk.max, alpha=0.2 if i in DDD.steps[0].frqBlkIds else 0.05, facecolor='yellow')

            self.ax[0].legend(loc=0)

            # First try rescaling the graph
            self.ax[0].relim()    # recompute the ax.dataLim
            self.ax[0].margins(0, 0.05)    # x and y margins in percentages
            self.ax[0].autoscale()    # update ax.viewLim using the new dataLim
            new_ax0Limits = {"xlim":self.ax[0].get_xlim(), "ylim":self.ax[0].get_ylim()}

            # Set the updated limits
            if settings["ax0Limits"] is not None:
                self.ax[0].set_xlim(settings["ax0Limits"]["xlim"])
                self.ax[0].set_ylim(settings["ax0Limits"]["ylim"])
            else:
                #self.ax[0].relim()    # recompute the ax.dataLim
                #self.ax[0].margins(0, 0.05)    # x and y margins in percentages
                #self.ax[0].autoscale()    # update ax.viewLim using the new dataLim
                ##self.ax[0].autoscale_view(tight=True, scalex=True, scaley=True)
                settings["ax0Limits"] = new_ax0Limits   # {"xlim":self.ax[0].get_xlim(), "ylim":self.ax[0].get_ylim()}

            self.figure.suptitle(str(DDD))

            # Output the found results
            self.plotPieChart()

        else:
            print("Nothing to plot here.")
            return 0

        self.canvas.draw()    # refresh canvas

    def showStems(self, checked = True):
        """Plots stem lines to indicate modeled peaks."""
        # remember the axis settings
        settings["ax1Limits"] = {"xlim":self.ax[1].get_xlim(), "ylim":(0, self.ax[1].get_ylim()[1])}
        self.ax[1].clear()
        self.allStems = {}     # Dictionary that stores references to all stem lines

        if checked:
            # Plot stem diagrams
            mdldPeaks = collectPeaks(self._crnt.T, self._crnt.c0, self._crnt.crntParsH)
            #print(mdldPeaks)
            """self._crnt.T.findRoot().propPoles()     # Propagate all poles
            for i, node in enumerate([node for node in self._crnt.T.repRoots() if node.name not in ['Water', 'Chloroform'] ]):
                for j, leaf in enumerate(node.leaves()):
                    if leaf.uPoles.size > 0:
                        markerline, stemlines, baseline = self.ax[1].stem(leaf.uPoles.imag/(self._crnt.c0*np.pi*2), leaf.qPolesIntn*leaf.intn, basefmt=" ")     # , label=node.name if j==0 else ''
                        plt.setp(stemlines, linewidth=1, color=config.colrseq[i], picker = 2)    # Picking tolerance in px
                        plt.setp(markerline, markerfacecolor = config.colrseq[i], linestyle='None', color=config.colrseq[i], markersize=2)      # , picker=self.onStemPick
                        self.allStems[leaf.name] = (markerline, stemlines)"""
            for i, name in enumerate([name for name in self._crnt.repRootNames if name not in ['Water', 'Chloroform'] ]):
                for key, val in mdldPeaks[name].items():   # Loop over the leaves
                    markerline, stemlines, baseline = self.ax[1].stem([pk.chsh for pk in val], [np.abs(pk.intn) for pk in val], basefmt=" ")     # , label=node.name if j==0 else ''
                    plt.setp(stemlines, linewidth=1, color=config.colrseq[i], picker = 2)    # Picking tolerance in px
                    plt.setp(markerline, markerfacecolor = config.colrseq[i], linestyle='None', color=config.colrseq[i], markersize=2)      # , picker=self.onStemPick
                    self.allStems[key] = (markerline, stemlines)
            self.ax[1].set_xlim(settings["ax1Limits"]["xlim"])     # Set the saved limits
            self.ax[1].set_ylim(settings["ax1Limits"]["ylim"])     # Set the saved limits
            self.ax[1].set_ylabel('Number of atoms', horizontalalignment='right', x=1.0, verticalalignment='top', y=1.0)
            self.ax[1].set_navigate(False)
            self.ax[1].legend(loc=0)
            # Connect events
            self.cidScroll = self.canvas.mpl_connect('scroll_event',self.onSpectrumZoom)
            self.cidPick = self.canvas.mpl_connect('pick_event', self.onStemPick)
        else:
            self.ax[1].clear()
            #self.ax[1].set_visible(False)
            try:
                self.canvas.mpl_disconnect(self.cidScroll)
                self.canvas.mpl_disconnect(self.cidPick)
            except:
                pass

        self.canvas.draw()

    def showComponents(self, checked = True):
        """Plots the constituent peaks for each model component"""
        pass

    def onStemPick(self, event):
        """Called on picking event; selects the corresponding row in chem tree."""
        for k, v in self.allStems.items():
            if event.artist in v[1]:
                self.treeView.selectActive(k)
                self.selectStems(k)

    def selectStems(self, key):
        """Shows which stems are affected when a new row is selected in the tree."""
        if self.actnShowStems.isChecked():
            if isinstance(key, tuple):
                # Convert the item name to teh name of stems
                if 'SPSY' in key[0]:
                    nameStem = key[0].rsplit('-', 1)[0] + '-' + key[0][key[0].rfind('SPSY')+4:] + '.' + str(key[2]+1)
                else:
                    nameStem = key[0]
            else:
                nameStem = key

            for k, v in self.allStems.items():
                for stemline in v:
                    plt.setp(stemline, linewidth = 2 if nameStem == k else 1)     # if nameStem is not None and nameStem in k

            self.canvas.draw()    # refresh canvas

    def onSpectrumZoom(self,event):
        #self.canvas.setFocus()    # Need to set focus to be able to process keyboard events
        # get the current x and y limits
        cur_ylim = self.ax[1].get_ylim()
        cur_yrange = (cur_ylim[1] - cur_ylim[0])*.5

        # Set scale factor
        base_scale = 1.6

        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1/base_scale
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1

        # set new limits
        self.ax[1].set_ylim([0, cur_ylim[1]*scale_factor])
        self.canvas.draw() # force re-draw

    def plotPieChart(self):
        """Plots a pie chart that represents the found component concentrations."""
        self.ax_pie.clear()
        #if '.' in self._crnt.crntParsH.keys() and 'ampl' in self._crnt.crntParsH['.'].keys():
        data = [(self._crnt.crntParsH[lbl]['ampl'][0], str(self.wsp.T[lbl])) for lbl in self.wsp.repRootNames if lbl not in ['Water', 'Chlorophorm']]     # All concentrations expcept water, chlorophorm, etc...
        cnct = np.abs([d[0] for d in data])
        cnct = np.where(np.isnan(cnct), 0.0, cnct)
        if sum(cnct) != 0:
            cnct = cnct / sum(cnct)
        labels = [d[1] for d in data]
        self.ax_pie.pie(cnct, labels=labels, explode=[0.05]*len(cnct), shadow=True, autopct='%0.2f', colors=config.colrseq)
        self.ax_pie.axis('equal')
        self.pieCanvas.draw()

    def addFreqBlock(self, xmin, xmax):
        """Adds new optimization range to the current Series andf updates the plot."""
        self.freqTableModel.addFreqBlock(xmin, xmax)

        self.ax[0].axvspan(xmin, xmax, alpha=0.1, facecolor='yellow')
        self.canvas.draw()

    def remFreqBlock(self, event):
        """Removes a frequency block that covers a location xdata in ppm and updates the plot."""
        # Find which block (if any) covers the passed location and which one to remove if there are multiple blocks. Event is a button_press_event passed from the canvas/CustomToolbar
        if event.xdata:   # If the click was in axes
            scores = [blk.max-blk.min if blk.min < event.xdata < blk.max else np.inf for blk in self._crnt.freqBlocks]    # Find the narrowest block that covers teh clicked position
            indx = min(enumerate(scores), key=itemgetter(1))[0]      # Find the index of the minimum

            if indx > 0:    # Don't remove the allFrequencies block
                self.freqTableModel.remFreqBlock(indx)
                # Update comboboxes
                self.plotCurrent()

    def plotFreqRanges(self):
        """Plots all frequency ranges"""

    def plot(self):
        self.ax[0].autoscale()



if __name__ == '__main__':
    app = 0
    app = QApplication(sys.argv)

    wsp = Workspace()
    main_view = MainView(wsp)
    main_view.show()
    app.exec_()
