import sys
from PyQt4 import QtCore, QtGui, uic
from PyQt4.QtGui import QAction, QApplication, QBrush, QCheckBox, QColor, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QGroupBox, QIcon, QInputDialog, QLabel, QLineEdit, QListWidget, QMenu, QMessageBox, QVBoxLayout, QHBoxLayout, QGridLayout, QMainWindow, QPainter, QPlainTextEdit, QProgressBar, QPushButton, QRadioButton, QSizePolicy, QSlider, QSpinBox, QSplitter, QStatusBar, QTabWidget, QTableWidget, QTreeView, QTreeWidget, QTreeWidgetItem, QToolBar, QToolButton, QWidget
from PyQt4.QtGui import QAction, QIcon, QMenu
from PyQt4.QtCore import Qt, pyqtSignal, QObject
import copy
from main import MyDoubleEdit
from chemTree import writeChemDB, chemSpec

qtCreatorFile = "form_DB.ui" # Enter file created with the QT designer here.

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

def mult2asgn(mult):
    """Transforms a vector of chemical shift multiplicities to a vector of assignments."""
    pass

class ListModel_DB(QtCore.QAbstractListModel):

    def __init__(self, chemDB, parent = None):
        QtCore.QAbstractListModel.__init__(self, parent)
        self.__chemDB = chemDB
        self._names = sorted([val.name for val in self.__chemDB.values()])

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self._names)

    def flags(self, index):
        return  QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable     # QtCore.Qt.ItemIsEditable |

    def data(self, index, role):
        pass

        if role == QtCore.Qt.ToolTipRole:
            row = index.row()
            column = index.column()
            return self._names[row]


        if role == QtCore.Qt.DecorationRole:

            row = index.row()
            column = index.column()
            value = 15 # self.__colors[row][column]

            pixmap = QtGui.QPixmap(26, 26)
            pixmap.fill()     # fill(self, color: QColor = Qt.white)

            icon = QtGui.QIcon(pixmap)

            return icon


        if role == QtCore.Qt.DisplayRole:

            row = index.row()
            column = index.column()
            return self._names[row]

    def setData(self, index, value, role = QtCore.Qt.EditRole):
        if role == QtCore.Qt.EditRole:

            row = index.row()
            column = index.column()

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
                return QtCore.QString("Color %1").arg(section)

    def addEntry(self, name):
        """Add new entry to the database."""
        self.beginInsertRows(QtCore.QModelIndex(), self.rowCount(), self.rowCount())

        self._names.append(name)
        self.__chemDB[name] = chemSpec(name=name)

        self.endInsertRows()


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

        rowCount = len(self.__chemDB)

        for i in range(columns):
            for j in range(rowCount):
                pass
                #self.__colors[j].insert(position, QtGui.QColor("#000000"))

        self.endInsertColumns()

        return True

class Model_chsh(QtCore.QAbstractTableModel):

    updated = pyqtSignal()     # Emmited when something has changed to update connected views

    def __init__(self, chem, parent = None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self._chem = chem             # Chemical specie
        self._chsh = []
        self._mult = []
        # Subclass this method and define self._chsh and self._asgn

    def reset(self, chem):
        pass

    def rowCount(self, parent = QtCore.QModelIndex()):
        return len(self._chsh)

    def columnCount(self, parent = QtCore.QModelIndex()):
        return 4

    def flags(self, index):
        return QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable

    def data(self, index, role):

        if role in [QtCore.Qt.EditRole, QtCore.Qt.DisplayRole]:
            row = index.row()
            column = index.column()
            if column == 0:
                return self._chsh[row].label
            elif column == 1:
                return self._chsh[row].dflt()
            elif column == 2:
                return self._nSpin[row]
            elif column == 3:
                return self._mult[self._spsyAsgn[row]]

        if role == QtCore.Qt.ToolTipRole:
            row = index.row()
            column = index.column()
            return "Chemical shift"

        if role == QtCore.Qt.DecorationRole:
            pass

    def setData(self, index, value, role = QtCore.Qt.EditRole):
        if role == QtCore.Qt.EditRole:

            row = index.row()
            column = index.column()

            if column == 0:
                self._chsh[row] = self._chsh[row]._replace(label=value)
                self.updated.emit()
            elif column == 1:
                self._chsh[row] = self._chsh[row]._replace(min=value-0.1, max = value+0.1)
                # TODO: handle the change in default values of parameters
            elif column == 2:
                self._nSpin[row] = value
            elif column == 3:
                self._mult[self._spsyAsgn[row]] = value
            return True
        return False

    def headerData(self, section, orientation, role):

        if role == QtCore.Qt.DisplayRole:

            if orientation == QtCore.Qt.Horizontal:
                if section == 0:
                    return "name"
                elif section == 1:
                    return "ppm"
                elif section in [2, 3]:
                    return "mult"
            else:
                return section+1

class ModelH(Model_chsh):

    def __init__(self, chem, parent = None):
        Model_chsh.__init__(self, chem, parent)
        self.reset(chem)

    def reset(self, chem):
        self.beginResetModel()
        self._chem = chem             # Chemical specie
        self._chsh = self._chem.chshH # if self._chem.chshH is not None else []
        self._nSpin = self._chem.nSpinH
        self._mult = self._chem.multH    # Multiplicity of each chemical shift
        self._spsyAsgn = self._chem._spsyAsgnH
        self.endResetModel()

    def addEntry(self):
        self.beginInsertRows(QtCore.QModelIndex(), self.rowCount(), self.rowCount())
        self._chem.add_chshH()
        self.endInsertRows()

    def delEntry(self, row):
        self.beginRemoveRows(QtCore.QModelIndex(), row, row)
        self._chem.del_chshH(row)
        self.endRemoveRows()
        self.updated.emit()

class ModelC(Model_chsh):

    def __init__(self, chem, parent = None):
        Model_chsh.__init__(self, chem, parent)
        self.reset(chem)

    def reset(self, chem):
        self.beginResetModel()
        self._chem = chem             # Chemical specie
        self._chsh = self._chem.chshC # if self._chem.chshC is not None else []
        self._nSpin = [1]*len(self._chsh)
        self._mult = self._chem.multC       # Multiplicity of each chemical shift
        self._spsyAsgn = [*range(len(self._chem.chshC))]    # Each carbon assigned to its own spin system
        self.endResetModel()

    def addEntry(self):
        self.beginInsertRows(QtCore.QModelIndex(), self.rowCount(), self.rowCount())
        self._chem.add_chshC()
        self._nSpin = [1]*len(self._chsh)
        self._spsyAsgn = [*range(len(self._chem.chshC))]    # Each carbon assigned to its own spin system
        self.endInsertRows()

    def delEntry(self, row):
        self.beginRemoveRows(QtCore.QModelIndex(), row, row)
        self._chem.del_chshC(row)
        self._nSpin = [1]*len(self._chsh)
        self._spsyAsgn = [*range(len(self._chem.chshC))]    # Each carbon assigned to its own spin system
        self.endRemoveRows()

class DoubleDelegate(MyDoubleEdit):

    def __init__(self, parent=None):
        super().__init__(parent)

class ComboDelegate(QtGui.QItemDelegate):
    height = 20
    width = 50

    def __init__(self, ListViewModel, parent=None):
        super(ComboDelegate, self).__init__(parent)
        self.ListViewModel = ListViewModel

    def createEditor(self, parent, option, index):
        row = index.row()
        column = index.column()

        if column in [0, 1]:
            editor = QtGui.QListView(parent)
            editor.setModel(self.ListViewModel)
            editor.clicked.connect(lambda : self.commitData.emit(editor))
            return editor
        else:
            pass

    def setEditorData(self,editor,index):
        row = index.row()
        column = index.column()
        editor.setModel(self.ListViewModel)
        editor.setGeometry(column*self.width, row*self.height if row < 8 else 0, self.width, 3*self.height)

    def setModelData(self, editor, model, index):
        row=editor.currentIndex().row()      # Return the selected row number
        model.setData(index, row)
        self.closeEditor.emit(editor, QtGui.QAbstractItemDelegate.NoHint)

class ModelJ(QtCore.QAbstractTableModel):

    updated = pyqtSignal()     # Emmited when something has changed to update connected views

    def __init__(self, chem, parent = None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self.reset(chem)

    def reset(self, chem):
        self.beginResetModel()
        self._chem = chem             # Chemical specie
        self._chsh = self._chem.chshH if self._chem.chshH is not None else []
        self._jcpl = self._chem.jcplHH if self._chem.jcplHH is not None else []
        self._fromto = self._chem.pairHH if self._chem.pairHH is not None else []
        self.endResetModel()

    def addEntry(self):
        self.beginInsertRows(QtCore.QModelIndex(), self.rowCount(), self.rowCount())
        self._chem.add_jcplHH()
        self.endInsertRows()

    def delEntry(self, row):
        self.beginRemoveRows(QtCore.QModelIndex(), row, row)
        self._chem.del_jcplHH(row)
        self.endRemoveRows()
        self.updated.emit()

    def notifyDataChanged(self):
        self.dataChanged.emit(QtCore.QModelIndex(), QtCore.QModelIndex())     # Update the entire tree

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self._fromto)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 3

    def flags(self, index):
        return QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable

    def data(self, index, role):

        if role == QtCore.Qt.EditRole:
            row = index.row()
            column = index.column()
            if column == 0:
                return self._chsh[self._fromto[row][0]].label
            elif column == 1:
                return self._chsh[self._fromto[row][1]].label
            elif column == 2:
                return self._jcpl[row].dflt()

        if role == QtCore.Qt.ToolTipRole:
            row = index.row()
            column = index.column()
            return "J couplings"

        if role == QtCore.Qt.DecorationRole:
            pass

        if role == QtCore.Qt.DisplayRole:
            row = index.row()
            column = index.column()
            if column == 0:
                if self._fromto[row][0] is not None:
                    return self._chsh[self._fromto[row][0]].label
                else: return None
            elif column == 1:
                if self._fromto[row][1] is not None:
                    return self._chsh[self._fromto[row][1]].label
                else: return None
            elif column == 2:
                return self._jcpl[row].dflt()

    def setData(self, index, value, role = QtCore.Qt.EditRole):
        if role == QtCore.Qt.EditRole:

            row = index.row()
            column = index.column()

            if column in [0, 1]:
                pair = copy.copy(self._fromto[row])
                pair[column] = value
                self._chem.set_jcplHH(row, pair)
                self.updated.emit()
            elif column == 2:
                self._jcpl[row] = self._jcpl[row]._replace(min=value-1.0, max = value+1.0, dval=value)
            return True
        return False

    def headerData(self, section, orientation, role):

        if role == QtCore.Qt.DisplayRole:

            if orientation == QtCore.Qt.Horizontal:
                if section == 0:
                    return ""
                elif section == 1:
                    return ""
                elif section == 2:
                    return "J, Hz"
            else:
                return section+1

class ModelSpsy(QtCore.QAbstractTableModel):

    def __init__(self, chem, parent = None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self.reset(chem)

    def reset(self, chem=None):
        self.beginResetModel()
        if chem is not None:
            self._chem = chem             # Chemical specie
        self.endResetModel()

    def notifyDataChanged(self):
        self.dataChanged.emit(QtCore.QModelIndex(), QtCore.QModelIndex())     # Update the entire tree

    def rowCount(self, parent):
        return self._chem._nSpsyH

    def columnCount(self, parent):
        return 2

    def flags(self, index):
        column = index.column()
        if column == 0:
            return QtCore.Qt.ItemIsEnabled # | QtCore.Qt.ItemIsSelectable
        elif column == 1:
            return QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled # | QtCore.Qt.ItemIsSelectable

    def data(self, index, role):
        row = index.row()
        column = index.column()

        if role == QtCore.Qt.ToolTipRole:
            return "Spin systems"

        if role == QtCore.Qt.DecorationRole:
            pass

        if role in [QtCore.Qt.DisplayRole, QtCore.Qt.EditRole]:
            if column == 0:
                return "-".join([ch.label for ch, j in zip(self._chem.chshH, self._chem._spsyAsgnH) if j==row])
            elif column == 1:
                return self._chem.multH[row]

    def setData(self, index, value, role = QtCore.Qt.EditRole):
        if role == QtCore.Qt.EditRole:
            row = index.row()
            column = index.column()

            if column == 1:
                self._chem.multH[row] = value
                return True
        return False

class ChemDBView(QtGui.QMainWindow, Ui_MainWindow):

    def __init__(self, chemDB):
        QtGui.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.chemDB = chemDB
        self.setupUi(self)

        # Setup the main database view
        self.model_DB = ListModel_DB(self.chemDB)
        self.ListView_DB.setModel(self.model_DB)
        self.ListView_DB.clicked.connect(self.onSelectFromDB)

        slct = "Shikimic acid"

        # Setup the chsh QTableView
        self.modelH = ModelH(self.chemDB[slct])
        self.tableView_chshH.setModel(self.modelH)
        self.tableView_chshH.setColumnWidth(0, 50)
        self.tableView_chshH.setColumnWidth(1, 50)
        self.tableView_chshH.setColumnWidth(2, 35)
        self.tableView_chshH.verticalHeader().setDefaultSectionSize(20)
        self.tableView_chshH.setColumnHidden(3, True)
        self.tableView_chshH.customContextMenuRequested.connect(lambda pos : self.onCustomContextMenuRequested(widget = self.tableView_chshH, pos=pos))
        self.tableView_chshH.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.tableView_chshH.horizontalHeader().setStretchLastSection(True)

        self.modelC = ModelC(self.chemDB[slct])
        self.tableView_chshC.setModel(self.modelC)
        self.tableView_chshC.setColumnWidth(0, 50)
        self.tableView_chshC.setColumnWidth(1, 45)
        self.tableView_chshC.setColumnWidth(2, 35)
        self.tableView_chshC.setColumnWidth(3, 35)
        self.tableView_chshC.verticalHeader().setDefaultSectionSize(20)
        self.tableView_chshC.setColumnHidden(2, True)
        self.tableView_chshC.customContextMenuRequested.connect(lambda pos : self.onCustomContextMenuRequested(widget = self.tableView_chshC, pos=pos))
        self.tableView_chshC.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.tableView_chshC.horizontalHeader().setStretchLastSection(True)

        self.modelJ = ModelJ(self.chemDB[slct])
        self.tableView_jcplHH.setModel(self.modelJ)
        delegate = ComboDelegate(self.modelH)          # A delegate to display list of chem shifts in comboboxes
        self.tableView_jcplHH.setItemDelegateForColumn(0, delegate)
        self.tableView_jcplHH.setItemDelegateForColumn(1, delegate)
        self.tableView_jcplHH.setColumnWidth(0, 50)
        self.tableView_jcplHH.setColumnWidth(1, 50)
        self.tableView_jcplHH.setColumnWidth(2, 50)
        self.tableView_jcplHH.verticalHeader().setDefaultSectionSize(20)
        self.tableView_jcplHH.customContextMenuRequested.connect(lambda pos : self.onCustomContextMenuRequested(widget = self.tableView_jcplHH, pos=pos))
        self.tableView_jcplHH.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.tableView_jcplHH.horizontalHeader().setStretchLastSection(True)

        self.modelSpsy = ModelSpsy(self.chemDB[slct])
        self.tableView_spsyH.setModel(self.modelSpsy)
        self.tableView_spsyH.setColumnWidth(0, 155)
        self.tableView_spsyH.setColumnWidth(1, 30)
        self.tableView_spsyH.verticalHeader().setDefaultSectionSize(20)
        self.tableView_spsyH.horizontalHeader().hide()
        self.tableView_spsyH.verticalHeader().hide()
        self.tableView_spsyH.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.tableView_spsyH.horizontalHeader().setStretchLastSection(True)

        self.modelH.updated.connect(self.modelJ.notifyDataChanged)
        self.modelH.updated.connect(self.modelSpsy.reset)
        self.modelJ.updated.connect(self.modelSpsy.reset)

        self.bttn_newChemical.clicked.connect(self.addNewChemical)
        self.bttn_save.clicked.connect(lambda : self.saveDB(fname=None))

    def onSelectFromDB(self, index):
        slct = index.data()           # New selection
        self.modelH.reset(self.chemDB[slct])
        self.modelC.reset(self.chemDB[slct])
        self.modelJ.reset(self.chemDB[slct])
        self.modelSpsy.reset(self.chemDB[slct])

    def onCustomContextMenuRequested(self, widget, pos):
        """Handler of the custom context menu requested signal."""
        popMenu = QMenu(self)
        index = widget.indexAt(pos)
        model = widget.model()

        actnAdd = QAction(QIcon('icons\icon_plus.png'), 'Add', self)
        actnAdd.setStatusTip('Add new entry')
        actnAdd.triggered.connect(model.addEntry)
        actnDel = QAction(QIcon('icons\icon_minus.png'), 'Remove', self)
        actnDel.setStatusTip('Remove this entry')
        if index.isValid():
            actnDel.triggered.connect(lambda : model.delEntry(index.row()))
        else: actnDel.setEnabled(False)

        popMenu.addAction(actnAdd)
        popMenu.addAction(actnDel)

        # Show the menu
        popMenu.popup(widget.viewport().mapToGlobal(pos))

    def addNewChemical(self):
        print("Adding new chemical.")

        class NewChemicalDialog(QDialog):
            """A dialog to request a name of the new chemical."""

            def __init__(self, parent = None):
                super().__init__(parent)
                layoutMain = QVBoxLayout(self)

                # Add widgets for entering parameters
                self.editName = QLineEdit()

                # OK and Cancel buttons
                self.buttons = QDialogButtonBox(
                    QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
                    Qt.Horizontal, self)
                #layoutMain.addLayout(layoutForm)
                layoutMain.addWidget(self.editName)
                layoutMain.addWidget(self.buttons)

                self.buttons.accepted.connect(self.accept)
                self.buttons.rejected.connect(self.reject)

                # Resize and change the caption
                """prnt_pos = self.parent().mapToGlobal(QtCore.QPoint(0,0))
                prnt_siz = self.parent().size()
                w, h = 180, 210
                x = prnt_pos.x() + (prnt_siz.width()-w)/2
                y = prnt_pos.y() + (prnt_siz.height()-h)/2
                self.setGeometry(x,y, w, h)"""
                self.setWindowFlags(QtCore.Qt.Tool)
                self.setWindowTitle("Enter new chemical name")

            # get the selection
            def getSelection(self):
                name = self.editName.text()
                print(name)
                return name

        dialog = NewChemicalDialog(parent=self)
        result = dialog.exec_()
        if result == QDialog.Accepted:    # If OK was clicked
            newName = dialog.getSelection()
            self.model_DB.addEntry(newName)

    def saveDB(self, fname=None):
        if fname is None:
            fname = 'chemDB'
        writeChemDB(self.chemDB, fname)


    # properties to read/write widget value
    @property
    def running(self):
        return self.ui.pushButton_running.isChecked()
    @running.setter
    def running(self, value):
        self.ui.pushButton_running.setChecked(value)





from PyQt4 import QtGui, QtCore, uic
import sys






class PaletteTableModel(QtCore.QAbstractTableModel):

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
            column = index.column()
            return self.__colors[row][column].name()


        if role == QtCore.Qt.ToolTipRole:
            row = index.row()
            column = index.column()
            return "Hex code: " + self.__colors[row][column].name()


        if role == QtCore.Qt.DecorationRole:

            row = index.row()
            column = index.column()
            value = self.__colors[row][column]

            pixmap = QtGui.QPixmap(26, 26)
            pixmap.fill(value)

            icon = QtGui.QIcon(pixmap)

            return icon


        if role == QtCore.Qt.DisplayRole:

            row = index.row()
            column = index.column()
            value = self.__colors[row][column]

            return value.name()


    def setData(self, index, value, role = QtCore.Qt.EditRole):
        if role == QtCore.Qt.EditRole:

            row = index.row()
            column = index.column()

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
                return QtCore.QString("Color %1").arg(section)



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





"""

if __name__ == '__main__':

    app = QtGui.QApplication(sys.argv)
    app.setStyle("plastique")


    #ALL OF OUR VIEWS
    listView = QtGui.QListView()
    listView.show()

    comboBox = QtGui.QComboBox()
    comboBox.show()

    tableView = QtGui.QTableView()
    tableView.show()



    red   = QtGui.QColor(255,0,0)
    green = QtGui.QColor(0,255,0)
    blue  = QtGui.QColor(0,0,255)





    listView.setModel(model)
    comboBox.setModel(model)
    tableView.setModel(model)


    sys.exit(app.exec_())"""
