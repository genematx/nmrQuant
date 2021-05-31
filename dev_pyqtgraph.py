# -*- coding: utf-8 -*-
"""
This example demonstrates many of the 2D plotting capabilities
in pyqtgraph. All of the plots may be panned/scaled by dragging with
the left/right mouse buttons. Right click on any plot to show a context menu.
"""

import sys
from PyQt4 import QtGui  # (the example applies equally well to PySide)
import pyqtgraph as pg
import numpy as np
import config
import dill, copy
from main import MainSpectrumWidget, PhasingWidget

colrseq = [tuple(int(255*c) for c in colr) for colr in config.colrseq]

# Set white background in plots
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)


def test():
    with open('test_data.dill', 'rb') as fp:
        data = dill.load(fp)
    f, yF, yFph, zF, xF, stems, freqBlocks = data['f'].ravel(), data['yF'].ravel(), data['yFph'].ravel(), data['zF'], data['xF'].ravel(), data['stems'], data['freqBlocks']

    mainSpectrumWidget = MainSpectrumWidget(title='Spectrum #1 in Series #3')
    mainPhasingWidget = PhasingWidget()

    mainSpectrumWidget.plot(f, yFph, xF, zF, stems, freqBlocks)
    mainPhasingWidget.setData(f, yFph, xF, freqBlocks)
    mainSpectrumWidget.sigPivotDragged.connect(mainPhasingWidget.setPivot)

    def update(indx, lims):
        print(indx, lims)

    def onPhased(p0deg, p1deg):
        print(p0deg, p1deg)

    mainSpectrumWidget.sigFreqRangeChanged.connect(update)
    mainPhasingWidget.sigPhasingProgress.connect(mainSpectrumWidget.replot_yF)
    mainPhasingWidget.sigPhasingComplete.connect(onPhased)



    # Plot the full spectrum in a small viewBox

#    mainFigureWidget.getPlotItem().setDownsampling(ds=len(f)/250, auto=False, mode='peak')

    # def updatePlot():
    #     mainFigureWidget.getPlotItem().blockSignals(True)
    #     mainFigureWidget.getPlotItem().setXRange(*lr_zoom.getRegion(), padding=0)
    #     mainFigureWidget.getPlotItem().blockSignals(False)
    # def updateRegion():
    #     currentRange = mainFigureWidget.getPlotItem().viewRange()[0]
    #     lr_zoom.setRegion([max(currentRange[0], f.min()), min(currentRange[1], f.max())])
    # lr_zoom.sigRegionChanged.connect(updatePlot)
    # mainFigureWidget.getPlotItem().sigXRangeChanged.connect(updateRegion)
    # updatePlot()


    return mainSpectrumWidget, mainPhasingWidget



#x2 = np.linspace(-100, 100, 1000)
#data2 = np.sin(x2) / x2
#p8 = win.addPlot(title="Region Selection")
#p8.plot(data2, pen=(255,255,255,200))
#lr = pg.LinearRegionItem([400,700])
#lr.setZValue(-10)
#p8.addItem(lr)
#
#p9 = win.addPlot(title="Zoom on selected region")
#p9.plot(data2)







if __name__ == '__main__':
    app = 0
    app = QtGui.QApplication(sys.argv)

    mainSpectrumWidget, mainPhasingWidget = test()

    ## Define a top-level widget to hold everything
    w = QtGui.QWidget()

    ## Create some widgets to be placed inside
    btn1 = QtGui.QPushButton('Show Residual')
    btn2 = QtGui.QPushButton('Hide Residual')
    btn3 = QtGui.QPushButton('Add region')
    btn4 = QtGui.QPushButton('Rem region')
    btn5 = QtGui.QPushButton('Enable CH')
    btn6 = QtGui.QPushButton('Disable CH')
    btn7 = QtGui.QPushButton('AutoRange')
    btn8 = QtGui.QPushButton('Save Figure')
    btn9 = QtGui.QPushButton('Set global chsh')
    listw = QtGui.QListWidget()

    ## Create a grid layout to manage the widgets size and position
    layout = QtGui.QGridLayout()
    w.setLayout(layout)

    ## Add widgets to the layout in their proper positions
    layout.addWidget(btn1, 0, 0)   # button goes in upper-left
    layout.addWidget(btn2, 1, 0)   # text edit goes in middle-left
    layout.addWidget(btn3, 2, 0)
    layout.addWidget(btn4, 3, 0)
    layout.addWidget(btn5, 4, 0)
    layout.addWidget(btn6, 5, 0)
    layout.addWidget(btn7, 6, 0)
    layout.addWidget(btn8, 7, 0)
    layout.addWidget(btn9, 8, 0)
    layout.addWidget(mainPhasingWidget, 0, 1, 10, 1)
#    layout.addWidget(listw, 2, 0)  # list widget goes in bottom-left
#    layout.addWidget(fullSpectrumViewBox, 2, 0)
    layout.addWidget(mainSpectrumWidget, 0, 2, 10, 2)  # plot goes on right side, spanning several rows

    btn1.pressed.connect(mainSpectrumWidget.showResidual)
    btn2.pressed.connect(lambda:mainSpectrumWidget.showResidual(False))
    # btn3.pressed.connect(mainSpectrumWidget.toggleSelectorFlag)
    btn4.pressed.connect(lambda : mainSpectrumWidget.remFreqRange(indx=0))
    btn5.pressed.connect(mainSpectrumWidget.enableCrosshair)
    btn6.pressed.connect(lambda : mainSpectrumWidget.enableCrosshair(False))
    btn7.pressed.connect(mainSpectrumWidget.autoRange)
    btn8.pressed.connect(mainSpectrumWidget.saveImage)
    btn9.pressed.connect(mainSpectrumWidget.setGlobalChsh)

    ## Display the widget as a new window
    w.show()

    app.exec_()
