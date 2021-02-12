import sys
import numpy as np
import dill
from MainLogic import *
from MainLogic import Series, Datum, Workspace
from dataio import *
import config

from PyQt4 import QtGui, QtCore, uic
from PyQt4.QtGui import QAction, QActionGroup, QApplication, QBrush, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QGroupBox, QIcon, QInputDialog, QItemSelectionModel, QItemDelegate, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMenu, QMessageBox, QVBoxLayout, QHBoxLayout, QGridLayout, QMainWindow, QPalette, QPen, QPlainTextEdit, QProgressBar, QPushButton, QRadioButton, QSizePolicy, QSlider, QSpinBox, QSplitter, QStatusBar, QStyle, QTableView, QTabWidget, QTableWidget, QToolButton, QTreeView, QToolBar, QToolTip, QWidget
from PyQt4.QtCore import Qt, pyqtSignal, QObject, QThread, QEvent
import pyqtgraph as pg
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backend_bases import cursors
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from operator import itemgetter
from os import path
from random import shuffle
import re
import math
import os
import nmrglue as ng
from datetime import date
from types import MethodType

from main import MainSpectrumWidget, FittingThread

# Set white background in plots
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
pg.setConfigOptions(antialias=True)       # Enable antialiasing for prettier plots

version = '0.0.3'
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

# class EmittingStream(QObject):
#     """For printing text in a textEdit."""
#
#     textWritten = pyqtSignal(str)
#
#     def write(self, text):
#         self.textWritten.emit(str(text))

from PyQt4.QtGui import QDialog, QVBoxLayout, QDateTimeEdit, QApplication
from PyQt4.QtCore import Qt, QDateTime

# Some definitions
labels = ['Ethanol','Glucose','Fructose','Sucrose','Sorbitol',
        'Glycerol', 'Methanol', '2,3-Butanediol',
        'Acetic acid', 'Citric acid', 'Lactic acid', 'Malic acid', 'Succinic acid', 'Tartaric acid', 'Peak 1', 'Peak 2']    # Ordred keys/labels
labels_volatile = ['Ethanol', 'Methanol', '2,3-Butanediol', 'Acetic acid']
labels_fromdry = ['Succinic acid', '2,3-Butanediol', 'Tartaric acid', 'Citric acid', 'Malic acid', 'Glucose', 'Fructose', 'Sucrose', 'Sorbitol']       # Which chemicals to estimate from evaporated samples
abbrev = {'Ethanol':'EthOH', 'Glucose':'Glu', 'Fructose':'Fru', 'Sucrose':'Suc', 'Sorbitol':'SrbOH',
        'Glycerol':'GlyOH', 'Methanol':'MetOH', '2,3-Butanediol':'BudOH',
        'Lactic acid':'LacAc', 'Acetic acid':'AceAc', 'Malic acid':'MalAc', 'Citric acid':'CitAc', 'Succinic acid':'SccAc', 'Tartaric acid':'TrtAc', 'Peak 1':'noAsgn_1', 'Peak 2':'noAsgn_2'}     # Concentrations in g/kg
prefcol = {'Ethanol':None, 'Glucose':None, 'Fructose':None, 'Sucrose':None, 'Sorbitol':None,
           'Glycerol':None, 'Methanol':None, '2,3-Butanediol':'BudOH',
           'Lactic acid':None, 'Acetic acid':'AceAc', 'Malic acid':'MalAc', 'Citric acid':'CitAc', 'Succinic acid':'SccAc', 'Tartaric acid':None, 'Peak 1':None, 'Peak 2':None}     # Preferred colors for each species
molWeight = {'Ethanol':46.07, 'Glucose':180.16, 'Fructose':180.16, 'Sucrose':342.2965, 'Sorbitol':182.17,
           'Glycerol':92.09382, 'Methanol':32.04, '2,3-Butanediol':90.121,
           'Lactic acid':90.08, 'Acetic acid':60.052, 'Malic acid':134.0874, 'Maleic acid':116.07, 'Citric acid':192.124,
           'Water':18.01, 'Succinic acid':118.09, 'Alanine':89.09, 'Proline':115.13, 'TMSP':146.26, 'Tartaric acid':150.087, 'Peak 1':100.0, 'Peak 2':100.0}     # Molar weights for each species
nH_labile = {'Ethanol':1, 'Glucose':5, 'Fructose':5, 'Sucrose':8, 'Sorbitol':6,
             'Glycerol':3, 'Methanol':1, '2,3-Butanediol':2,
             'Lactic acid':2, 'Acetic acid':1, 'Malic acid':3, 'Maleic acid':2, 'Citric acid':4,
             'Water':2, 'Succinic acid':2, 'Alanine':2, 'Proline':2, 'TMSP':0, 'Tartaric acid':4, 'Peak 1':1, 'Peak 2':1}     # Number of labile protons (OH, NH2, NH3)
def cww2pvv(wconc):
    """Converts a dictionary of concentrations expressed in g/kg to %v/v of ethanol, actual alcoholic strength, and total alcoholic strength."""
    act_alc_vv = 1.2241*(wconc['Ethanol']+wconc['Methanol']+wconc['2,3-Butanediol'])*100+0.1182
    tot_alc_vv = act_alc_vv + 0.06*(wconc['Fructose']+wconc['Glucose']+wconc['Sucrose'])*est_density(wconc)

    return act_alc_vv, tot_alc_vv

def est_density(wconc):
    """Estimates density (expressed in g/L) of an aqueous solution having wconc weight concentrations of solutes."""
    # Mass fractions of each component
    x_Eth, x_Gly, x_Glu, x_Fru, x_Suc = wconc['Ethanol'], wconc['Glycerol'], wconc['Glucose'], wconc['Fructose'], wconc['Sucrose']
    x_H2O = 1 - (x_Eth + x_Gly + x_Glu + x_Fru + x_Suc)

    if x_H2O == 1:
        return 1000.0
    else:
        # Compute the densities if only each spcies were present in the given amount of water
        d_Eth = 43.8*((x_Eth/x_H2O)**2) - 158.3*(x_Eth/x_H2O) + 997.8
        d_Gly = 56.0*((x_Gly/x_H2O)**2) + 227.3*(x_Gly/x_H2O) + 998.2
        d_Glu = 173.7*((x_Glu/x_H2O)**2) + 373.4*(x_Glu/x_H2O) + 998.3
        d_Fru = 141.5*((x_Fru/x_H2O)**2) + 388.7*(x_Fru/x_H2O) + 998.2
        d_Suc = 152.7*((x_Suc/x_H2O)**2) + 383.4*(x_Suc/x_H2O) + 998.2

        # Find the resulting total density by weighting according to the mass fractions
        d_Tot = (d_Eth*x_Eth + d_Gly*x_Gly + d_Glu*x_Glu + d_Fru*x_Fru + d_Suc*x_Suc) / (1-x_H2O)

        return d_Tot

def reset_parameters(DDD):
    """To be removed. Left for compatibility."""
    start_fit(DDD)

def start_fit(DDD):
    """Resets the current parameters in a Datum."""
    # Reset the definitions of the extra parameters
    DDD.extra.update({'mass_frac_MalAc':0.0, 'masses_au': {}, 'mass_total_au':0.0, 'density':1000.0, 'fitted':False})

    # Clear priors set in the Datum, if any
    DDD.parsSpecDict.clear()

    # Reset the current parameters to their default values
    for key in DDD.allParsKeys():
        val = DDD.getPrior(key).dflt()    # Default parameter value from the prior
        DDD.setCrntVal(key, val)

    # Reset all intensities
    for name in DDD.repRootNames:
        DDD.setCrntVal(key=(name, 'ampl', 0), val=0.0)

    # Measure the noise level in the spectrum
    DDD.extra['sigma_est'] = np.sqrt( DDD.measure_noise(lims=(-10, -2)) )

def fit_global_chsh(DDD):
    """Aligns the spectrum by adjusting its gloabl chemical shift."""
    # Turn on autofitting for some components
    autoKeys = [('.', 'sigma2', 0)]
    for name in ['Glycerol', 'Fructose', 'Glucose', 'Sucrose', 'Sorbitol']:
        autoKeys.append((name, 'ampl', 0))
    if 'DRY' not in DDD.name:
        autoKeys.append( ('Ethanol', 'ampl', 0) )

    # Use apodization
    for apod in [10, 5, 3]:
        DDD.resetFreqs(apod=apod, zff=0)
        DDD.setCrntVal(key=('Mixture', 'alph', 0), val=apod+2.0)

        # Adjust the global chemical shift
        DDD.optimize(parsKeys=[('Mixture', 'chsh', 0)], autoKeys=autoKeys+[('Maleic acid', 'ampl', 0)], frqBlkIds=[2,3])
        # DDD.setPrior(key=('Mixture', 'chsh', 0), fromCurrent=True)

        # Refine the global shift, now without maleic acid
        DDD.optimize(parsKeys=[('Mixture', 'chsh', 0)], autoKeys=autoKeys, frqBlkIds=[3], nhop=5)
        # Adjust the chemical shift of maleic acid alone
        DDD.optimize(parsKeys=[('Maleic acid', 'chshQD', 0)], autoKeys=[('.', 'sigma2', 0), ('Maleic acid', 'ampl', 0)], frqBlkIds=[2])

    DDD.setPrior(key=('Mixture', 'chsh', 0), fromCurrent=True)

    # If there is strong ethanol signal, fit global position based on the CH3 peaks of ethanol
    mfrac = DDD.report_moleFrac(names = ['Ethanol', 'Glycerol', 'Fructose', 'Glucose', 'Sucrose', 'Sorbitol'])
    if DDD.getCrntVal(key=('Ethanol', 'ampl', 0)) > 0.5*DDD.extra['sigma_est']:
        DDD.optimize(parsKeys=[('Mixture', 'chsh', 0)], autoKeys=autoKeys, frqBlkIds=[4])

    # Reset apodization
    DDD.resetFreqs(apod=00.0, zff=1)
    DDD.setCrntVal(key=('Mixture', 'alph', 0), val=2.0)

def fit_ethanol_CH3(DDD):
    if 'DRY' in DDD.name:
        # Fit butanediol and ethanol
        DDD.altFreqBlock(indx=4, lims=(1.0, 1.33))          # Use narrow range
        autoKeys = [('.', 'sigma2', 0), ('2,3-Butanediol', 'ampl', 0), ('Ethanol', 'ampl', 0)]
        for _ in range(2):
            DDD.optimize(parsKeys=[('2,3-Butanediol', 'chsh', 0), ('Ethanol', 'chshQD', 0)], autoKeys=autoKeys, frqBlkIds=[4])
            DDD.optimize(parsKeys=[('2,3-Butanediol', 'chsh', 0)], autoKeys=autoKeys, frqBlkIds=[4], nhop=5)
            DDD.optimize(parsKeys=[('Ethanol', 'chshQD', 0)], autoKeys=autoKeys, frqBlkIds=[4])
            DDD.optimize(parsKeys=[('2,3-Butanediol', 'alph', 0), ('Ethanol', 'alph', 0)], autoKeys=autoKeys, frqBlkIds=[4])
        DDD.altFreqBlock(indx=4, lims=(0.5, 1.75))         # Restore the range
    else:
        # Fit only ethanol
        autoKeys = [('.', 'sigma2', 0), ('Ethanol', 'ampl', 0)]
        for _ in range(2):
            DDD.optimize(parsKeys=[('Ethanol', 'chshQD', 0)], autoKeys=autoKeys, frqBlkIds=[4])
            DDD.optimize(parsKeys=[('Ethanol', 'alph', 0)], autoKeys=autoKeys, frqBlkIds=[4])

    # Use the CH3 peak as the reference (only if the estimated signal of ethanol is strong enough)
    if DDD.getCrntVal(key=('Ethanol', 'ampl', 0)) > 0.5*DDD.extra['sigma_est']:
        DDD.shiftToRef(refKey=('Ethanol', 'chshQD', 0), keepFitted=False)

def fit_sugars(DDD):
    # Step 4. Fit Glucose and Sucrose using the anomeric peaks
    autoKeys = [('.', 'sigma2', 0), ('Glucose', 'ampl', 0), ('Sucrose', 'ampl', 0)]
    for _ in range(2):
        DDD.optimize(parsKeys=[('Sugars', 'chsh', 0)], autoKeys=autoKeys, frqBlkIds=[5])
        DDD.optimize(parsKeys=[('Sugars', 'alph', 0)], autoKeys=autoKeys, frqBlkIds=[5])

    # Step 5. Fit the rest of sugars, glycerol, and CH2 peak of ethanol
    mfrac = DDD.report_moleFrac(names = ['Ethanol', 'Glycerol', 'Fructose', 'Glucose', 'Sucrose', 'Sorbitol'])
    mfrac['Sugars'] = mfrac['Fructose']+mfrac['Glucose']+mfrac['Sucrose']+mfrac['Sorbitol']
    mfrac_sorted = sorted(['Sugars', 'Ethanol', 'Glycerol'], key=lambda x : mfrac[x], reverse=True)

    if mfrac['Fructose'] > 1.1*mfrac['Glucose'] and DDD.getCrntVal(key=('Fructose', 'ampl', 0)) > DDD.extra['sigma_est']:
        for _ in range(2):
            DDD.optimize(parsKeys=[('Sugars', 'chsh', 0)], autoKeys=[('.', 'sigma2', 0), ('Fructose', 'ampl', 0)], frqBlkIds=[14])
            DDD.optimize(parsKeys=[('Sugars', 'alph', 0)], autoKeys=[('.', 'sigma2', 0), ('Fructose', 'ampl', 0)], frqBlkIds=[14])

    autoKeys = [('.', 'sigma2', 0), ('Fructose', 'ampl', 0), ('Sorbitol', 'ampl', 0), ('Glycerol', 'ampl', 0)]
    for _ in range(2):
        for name in mfrac_sorted:
            if name == 'Ethanol':
                DDD.optimize(parsKeys=[('Ethanol', 'chshQD', 1)], autoKeys=autoKeys, frqBlkIds=[3])
                DDD.optimize(parsKeys=[('Ethanol', 'alphQD', 1)], autoKeys=autoKeys, frqBlkIds=[3])
            else:
                DDD.optimize(parsKeys=[(name, 'chsh', 0)], autoKeys=autoKeys, frqBlkIds=[3])
                DDD.optimize(parsKeys=[(name, 'alph', 0)], autoKeys=autoKeys, frqBlkIds=[3])

    mfrac = DDD.report_moleFrac(names = ['Ethanol', 'Glycerol', 'Fructose', 'Glucose', 'Sucrose', 'Sorbitol'])
    mfrac['Sugars'] = mfrac['Fructose']+mfrac['Glucose']+mfrac['Sucrose']+mfrac['Sorbitol']
    mfrac_sorted = sorted(['Sugars', 'Ethanol', 'Glycerol'], key=lambda x : mfrac[x], reverse=True)
    # if mfrac['Glycerol'] > mfrac['Sugars'] + mfrac['Ethanol']:
    #     DDD.shiftToRef(refKey=('Glycerol', 'chsh', 0), keepFitted=True)
    if mfrac['Glucose'] > 0.05: autoKeys.append( ('Glucose', 'ampl', 0) )
    if mfrac['Sucrose'] > 0.05: autoKeys.append( ('Sucrose', 'ampl', 0) )

    for _ in range(2):
        for name in mfrac_sorted:
            if name == 'Ethanol':
                DDD.optimize(parsKeys=[('Ethanol', 'chshQD', 1)], autoKeys=autoKeys, frqBlkIds=[3])
                DDD.optimize(parsKeys=[('Ethanol', 'alphQD', 1)], autoKeys=autoKeys, frqBlkIds=[3])
            else:
                DDD.optimize(parsKeys=[(name, 'chsh', 0)], autoKeys=autoKeys, frqBlkIds=[3])
                DDD.optimize(parsKeys=[(name, 'alph', 0)], autoKeys=autoKeys, frqBlkIds=[3])

def fit_acids(DDD):
    # Step 7. Fit the rest of acids
    autoKeys=[('.', 'sigma2', 0), ('Citric acid', 'ampl', 0), ('Malic acid', 'ampl', 0)]
    ampl_MalAc = DDD.getCrntVal(key=('Maleic acid', 'ampl', 0))
    if ampl_MalAc > DDD.extra['sigma_est']:
        print('Maleic acid present, intensity = {:.4g}.'.format(ampl_MalAc))
        # Reset the parameters for the case when maleic acid is present
        DDD.altFreqBlock(indx=6, lims=(2.75, 3.05))

        # Shifts of individual acid peaks
        DDD.setCrntVals(parsF={('Citric acid', 'chshQD', 0):2.966, ('Citric acid', 'chshQD', 1):2.793,
                               ('Malic acid', 'chshQD', 0): 4.595, ('Malic acid', 'chshQD', 1):2.852, ('Malic acid', 'chshQD', 2):2.804,
                               ('Lactic acid', 'chshQD', 0):1.415, ('Lactic acid', 'chshQD', 1):4.370})     # Reset the chemical shift values to defaults for the case of maleic acid present

        # Global shift for the acids
        DDD.setPrior(key=('Acids', 'chsh', 0), min=-0.01, max=0.1, dval=0.00)
        DDD.setCrntVal(key=('Acids', 'chsh', 0), val=0.05)
    else:
        print('No maleic acid detected.')

    # Peaks of citric and malic acids shift together.
    DDD.optimize(parsKeys=[('Acids', 'chsh', 0)], autoKeys=autoKeys, frqBlkIds=[6])    # Adjust the global shift
    # Update the parameters of shifting peaks
    chsh_acids = DDD.getCrntVal(key=('Acids', 'chsh', 0))
    for key in [('Citric acid', 'chshQD', 0), ('Citric acid', 'chshQD', 1), ('Malic acid', 'chshQD', 1), ('Malic acid', 'chshQD', 2)]:
        DDD.setCrntVal(key, val=DDD.getCrntVal(key)+chsh_acids)
        DDD.setPrior(key, fromCurrent=True)
    DDD.setCrntVal(key=('Acids', 'chsh', 0), val=0.0)
    DDD.setPrior(key=('Malic acid', 'chshQD', 0), fromCurrent=True, chshRange=0.025)

    if DDD.getCrntVal(key=('Citric acid', 'ampl', 0)) > DDD.extra['sigma_est'] or DDD.getCrntVal(key=('Malic acid', 'ampl', 0)) > DDD.extra['sigma_est']:
        DDD.optimize(parsKeys=[('Acids', 'alph', 0)], autoKeys=autoKeys, frqBlkIds=[6])

    # Adjust citric and malic acids individually
    for _ in range(2):
        if DDD.getCrntVal(key=('Citric acid', 'ampl', 0)) > DDD.getCrntVal(key=('Malic acid', 'ampl', 0)):
            DDD.optimize(parsKeys=[('Citric acid', 'chshQD', 0), ('Citric acid', 'chshQD', 1)], autoKeys=autoKeys, frqBlkIds=[6])
            DDD.optimize(parsKeys=[('Malic acid', 'chshQD', 1), ('Malic acid', 'chshQD', 2)], autoKeys=autoKeys, frqBlkIds=[6])
        else:
            DDD.optimize(parsKeys=[('Malic acid', 'chshQD', 1), ('Malic acid', 'chshQD', 2)], autoKeys=autoKeys, frqBlkIds=[6])
            DDD.optimize(parsKeys=[('Citric acid', 'chshQD', 0), ('Citric acid', 'chshQD', 1)], autoKeys=autoKeys, frqBlkIds=[6])

        if DDD.getCrntVal(key=('Citric acid', 'ampl', 0)) > DDD.extra['sigma_est'] or DDD.getCrntVal(key=('Malic acid', 'ampl', 0)) > DDD.extra['sigma_est']:
            DDD.optimize(parsKeys=[('Acids', 'alph', 0)], autoKeys=autoKeys, frqBlkIds=[6])

    # Return to the original frequency blocks
    DDD.altFreqBlock(indx=6, lims=(2.5, 3.1))     # Return to the original frequency block

    # Fit tartaric acid, if possible
    if ampl_MalAc > DDD.extra['sigma_est']:
        # With maleic acid
        DDD.setCrntVal(key=('Tartaric acid', 'chshQD', 0), val=4.671)
        DDD.setPrior(key=('Tartaric acid', 'chshQD', 0), min=4.59, max=4.75)
        frqBlkID_tartaric = 11
    else:
        # Without maleic acid
        DDD.setCrntVals(parsF={('Tartaric acid', 'chshQD', 0):4.514})
        DDD.setPrior(key=('Tartaric acid', 'chshQD', 0), min=4.45, max=4.6)
        frqBlkID_tartaric = 12

    if 'DRY' in DDD.name or 'PRESAT' in DDD.name:
        for _ in range(2):
            DDD.optimize(parsKeys=[('Tartaric acid', 'chshQD', 0)], autoKeys=[('.', 'sigma2', 0), ('Tartaric acid', 'ampl', 0)], frqBlkIds=[frqBlkID_tartaric])
            DDD.optimize(parsKeys=[('Tartaric acid', 'chshQD', 0), ('Tartaric acid', 'alphQD', 0)], autoKeys=[('.', 'sigma2', 0), ('Tartaric acid', 'ampl', 0)], frqBlkIds=[frqBlkID_tartaric])
            DDD.optimize(parsKeys=[('Tartaric acid', 'alphQD', 0)], autoKeys=[('.', 'sigma2', 0), ('Tartaric acid', 'ampl', 0)], frqBlkIds=[frqBlkID_tartaric])

def fit_lactic(DDD):
    """Fits the region related to lactic acid peaks H3 (incl. lactic acid and alanine)."""
    DDD.setPrior(key=('Lactic acid', 'chshQD', 0), fromCurrent=True)        # Depends on presense/absense of Maleic acid in the sample
    DDD.setPrior(key=('Lactic acid', 'chshQD', 1), fromCurrent=True)
    autoKeys=[('.', 'sigma2', 0), ('Lactic acid', 'ampl', 0), ('Alanine', 'ampl', 0)]
    for _ in range(2):
        DDD.optimize(parsKeys=[('Lactic acid', 'chshQD', 0)], autoKeys=autoKeys, frqBlkIds=[8])
        DDD.optimize(parsKeys=[('Alanine', 'chshQD', 1)], autoKeys=autoKeys, frqBlkIds=[8])
        DDD.optimize(parsKeys=[('Lactic acid', 'alph', 0), ('Alanine', 'alph', 0)], autoKeys=autoKeys, frqBlkIds=[8])
        DDD.optimize(parsKeys=[('Lactic acid', 'alph', 0)], autoKeys=autoKeys, frqBlkIds=[8])

    if 'DRY' in DDD.name:
        # Fit butanediol and ethanol
        autoKeys.extend([('2,3-Butanediol', 'ampl', 0), ('Ethanol', 'ampl', 0)])
        for _ in range(2):
            DDD.optimize(parsKeys=[('2,3-Butanediol', 'chsh', 0), ('Ethanol', 'chshQD', 0)], autoKeys=autoKeys, frqBlkIds=[4, 8])
            DDD.optimize(parsKeys=[('2,3-Butanediol', 'chsh', 0)], autoKeys=autoKeys, frqBlkIds=[4, 8])
            DDD.optimize(parsKeys=[('Ethanol', 'chshQD', 0)], autoKeys=autoKeys, frqBlkIds=[4, 8])
            DDD.optimize(parsKeys=[('2,3-Butanediol', 'alph', 0), ('Ethanol', 'alph', 0)], autoKeys=autoKeys, frqBlkIds=[4, 8])
    else:
        # Fit only ethanol
        autoKeys.append( ('Ethanol', 'ampl', 0) )
        for _ in range(2):
            DDD.optimize(parsKeys=[('Ethanol', 'chshQD', 0)], autoKeys=autoKeys, frqBlkIds=[4, 8])
            DDD.optimize(parsKeys=[('Ethanol', 'alph', 0)], autoKeys=autoKeys, frqBlkIds=[4, 8])

def fit_autoPhase(DDD):
    DDD.auto_phase(fit_Ph1=False)

def fit_volatile(DDD):
    """Fits butanediol (non-volatile) in the DRY samples or unidentified volatile compounds in the usual samples."""
    if 'DRY' in DDD.name:
        # Fit 2,3-butanediol
        pass
    else:
        # Fit unidentified peaks
        autoKeys = [('.', 'sigma2', 0), ('Peak 1', 'ampl', 0), ('Peak 2', 'ampl', 0)]
        for _ in range(2):
            # DDD.optimize(parsKeys=[('Unidentified', 'chsh', 0), ('Unidentified', 'alph', 0)],
            #              autoKeys=autoKeys, frqBlkIds=[10])
            DDD.optimize(parsKeys=[('Peak 1', 'chshQD', 0), ('Peak 2', 'chshQD', 0)],
                         autoKeys=autoKeys, frqBlkIds=[10])
            DDD.optimize(parsKeys=[('Peak 1', 'chshQD', 0), ('Peak 1', 'alphQD', 0)],
                         autoKeys=autoKeys, frqBlkIds=[10])
            DDD.optimize(parsKeys=[('Peak 2', 'chshQD', 0), ('Peak 2', 'alphQD', 0)],
                         autoKeys=autoKeys, frqBlkIds=[10])
            DDD.optimize(parsKeys=[('Unidentified', 'alph', 0)],
                         autoKeys=autoKeys, frqBlkIds=[10])

def fit_water_neighborhood(DDD):
    """Adjust the peaks of malic and lactic acids that are close to water without reestimating their intensities"""
    parsKeys = []
    if DDD.getCrntVal(key=('Malic acid', 'ampl', 0)) > DDD.extra['sigma_est']: parsKeys.append(('Malic acid', 'chshQD', 0))
    if DDD.getCrntVal(key=('Lactic acid', 'ampl', 0)) > DDD.extra['sigma_est']: parsKeys.append(('Lactic acid', 'chshQD', 1))
    DDD.optimize(parsKeys=parsKeys, autoKeys=[], frqBlkIds=[12, 13])

def use_presat(DDD):
    """Tries to find an already fitted (PRESAT) experiment in the same Series and copies all its parameters and distributions to the current (PROTON) datum."""
    if 'PROTON' not in DDD.name:
        return 0

def finish_fit(DDD):
    """Finishing the fitting of a wine sample."""
    DDD.extra.update({'fitted':True})

def wine_results(data, massFracIS_grav=None):
    """Expresses the results in %w/w. Takes into account all files in an array 'data'; all files must correspond to the same _original_ with the same amount of internal standard, possibly including spectra without ht einternal standard. massFracIS_grav controls how to estimate the concentration of internal standard and can be eitehr None (will be determined from water) or a number corresponding to gravimetric mass fraction of maleic acid wrt the sample."""
    result = {key : 0.0 for key in labels+['Maleic acid', 'Water', 'Alanine']}          # Initialize the results
    result['data_names'], result['acqu_time'], result['acqu_time_abs'] = [], None, None
    dat_FULL, dat_MAIN, dat_DRY = None, None, None         # Datums for the full (PROTON, non-PRESAT) spectrum, a spectrum from which the concentrations would be determined (usually, PRESAT), and a spectrum of dried sample (either presat or proton)
    dat_FULL_DRY_D2O, dat_PRES_DRY_D2O, dat_FULL_DRY, dat_PRES_DRY, dat_FULL, dat_PRES = None, None, None, None, None, None
    for DDD in data:
        result['data_names'].append(DDD.name)
        # Determine the type of each dataset
        # TODO: Reorganize this
        if 'DRY' in DDD.name:
            if 'D2O' in DDD.name:
                if 'PROTON' in DDD.name:
                    dat_FULL_DRY_D2O = DDD
                elif 'PRESAT' in DDD.name:
                    dat_PRES_DRY_D2O = DDD
            else:
                if 'PROTON' in DDD.name:
                    dat_FULL_DRY = DDD
                elif 'PRESAT' in DDD.name:
                    dat_PRES_DRY = DDD
        else:
            if 'PROTON' in DDD.name:
                dat_FULL = DDD
            elif 'PRESAT' in DDD.name:
                dat_PRES = DDD

        if 'DRY' in DDD.name and 'PRESAT' in DDD.name:
            dat_DRY = DDD
        elif 'PROTON' in DDD.name and not ('DRY' in DDD.name):
            if dat_MAIN is None:
                dat_MAIN = DDD
            dat_FULL = DDD
        elif 'PRESAT' in DDD.name:
            dat_MAIN = DDD

        # Compute the masses of species expressed in (arbitrary units)
        DDD.extra['masses_au'].clear()
        DDD.extra['masses_au'].update({lbl:DDD.getCrntVal(key=(lbl, 'ampl', 0))*molWeight[lbl] for lbl in labels+['Maleic acid']})
        mass_H2O_au = molWeight['Water']*(DDD.getCrntVal(key=('Water', 'ampl', 0))*2 - sum([DDD.getCrntVal(key=(lbl, 'ampl', 0))*nH_labile[lbl] for lbl in DDD.extra['masses_au'].keys()]) )/2    # Mass of water (excluding all labile protons)
        DDD.extra['masses_au'].update({'Water': max(mass_H2O_au, 0.0)})

    # Use the first PROTON spectrum with internal standard to estimate the mass fraction of maleic acid
    massFracIS_estm = None
    if massFracIS_grav is None:
        for DDD in data:
            if 'PROTON' in DDD.name and '-IS' in DDD.name:
                DDD.extra['mass_total_au'] = sum([val for _, val in DDD.extra['masses_au'].items()])
                massFracIS_estm = DDD.extra['masses_au']['Maleic acid'] / DDD.extra['mass_total_au']            # Shared among all samples
                break

    # Find the total mass (expressed in arbitrary units). Preferrably - from the internal standard, alternatively, as a sum of all components
    for DDD in data:
        if '-IS' in DDD.name:
            DDD.extra['mass_total_au'] = DDD.extra['masses_au']['Maleic acid']
            if massFracIS_estm is not None:
                DDD.extra['mass_total_au'] /= massFracIS_estm
            elif massFracIS_grav is not None:
                DDD.extra['mass_total_au'] /= massFracIS_grav
        elif 'PROTON' in DDD.name:
            DDD.extra['mass_total_au'] = sum([val for _, val in DDD.extra['masses_au'].items()])
        else:
            # If it is a PRESAT experiment without internal standard, use the total mass of non-water components as a reference of use an external standard
            if dat_FULL is not None:
                # TODO: Use constant scaling factor
                DDD.extra['mass_total_au'] = sum([val for _, val in dat_FULL.extra['masses_au'].items()]) * sum([val for key, val in DDD.extra['masses_au'].items() if key != 'Water']) / sum([val for key, val in dat_FULL.extra['masses_au'].items() if key != 'Water'])

    # Find the mass fractions of all chemicals
    wconc, brix = {key : 0.0 for key in labels+['Maleic acid', 'Water', 'Alanine']}, 0
    if dat_MAIN is not None and dat_MAIN.extra['mass_total_au'] != 0:
        wconc.update({key:val/dat_MAIN.extra['mass_total_au'] for key, val in dat_MAIN.extra['masses_au'].items()})
        brix = 100*(dat_MAIN.extra['masses_au']['Glucose']+dat_MAIN.extra['masses_au']['Fructose']+dat_MAIN.extra['masses_au']['Sucrose'])/dat_MAIN.extra['mass_total_au']
        if dat_MAIN.extra['acqu_time'] is not None:
            timeString = dat_MAIN.extra['acqu_time']
        else:
            timeString = '-'.join(dat_MAIN.name.split('-')[-2:])
        result['acqu_time'] = timeString
        result['acqu_time_abs'] = time.mktime(time.strptime(timeString, '%Y%m%d-%H%M%S'))         # Time in sec from the start of epoch

    if dat_DRY is not None and dat_DRY.extra['mass_total_au'] != 0:
        wconc.update({key:dat_DRY.extra['masses_au'][key]/dat_DRY.extra['mass_total_au'] for key in labels_fromdry})
        brix = 100*(dat_DRY.extra['masses_au']['Glucose']+dat_DRY.extra['masses_au']['Fructose']+dat_DRY.extra['masses_au']['Sucrose'])/dat_DRY.extra['mass_total_au']

    # Estimate tartaric acid from DRY, PROTON, D2O, if available
    for DDD in data:
        if 'DRY' in DDD.name and 'PROTON' in DDD.name and 'D2O' in DDD.name:
            wconc.update({'Tartaric acid': DDD.extra['masses_au']['Tartaric acid']/DDD.extra['mass_total_au']})

    # Compute the total metrics


    # Convert the units for alcohol and estimate the density of the sample
    act_alc_vv, tot_alc_vv = cww2pvv(wconc)
    density = est_density(wconc)

    result.update({key:density*val for key, val in wconc.items()})
    result.update({'BRIX':brix, 'Total Alcohol, %v\v':tot_alc_vv, 'Actual Alcohol, %v\v':act_alc_vv,
                   'Density':density, 'mass_frac_MalAc_estm':massFracIS_estm, 'mass_frac_MalAc_grav':massFracIS_grav})

    # Save results from all other datasets
    for DDD, DDD_type in [(dat_FULL, 'dat_FULL'), (dat_PRES, 'dat_PRES'), (dat_FULL_DRY, 'dat_FULL_DRY'), (dat_PRES_DRY, 'dat_PRES_DRY'), (dat_FULL_DRY_D2O, 'dat_FULL_DRY_D2O'), (dat_PRES_DRY_D2O, 'dat_PRES_DRY_D2O')]:
        if DDD is not None:
            result[DDD_type] = {key:density*val/DDD.extra['mass_total_au'] for key, val in DDD.extra['masses_au'].items()}
        else: result[DDD_type] = None

    return result

def get_sample_name(DDD):
    """Assigns a specific Datum to a particular wine sampl, based on its name and presence/absence of an internal standard."""
    if DDD.extra['Sample'] is not None:
        sample_name = DDD.extra['Sample']
    else:
        data_name = DDD.name.split('-')
        sample_name = data_name[0][:-2] if data_name[0][-2:] == '.0' else data_name[0]
        if data_name[1][:2] == 'IS':
            sample_name = '-'.join([sample_name, data_name[1]])

    return sample_name

def sort_names(names):
    """Sorts a list of wine names taking into account the name of the samples, presence of internal standard, if the sample was evaporated or not, addition of D2O, etc."""
    pass

def init_freqBlocks_autoWine(SSS):
    """Initializes frequency blocks in a series for wine analysis."""
    # Remove all frequency blocks
    for i in range(len(SSS.freqBlocks)-1, 0, -1):
        SSS.remFreqBlock(indx=i)
    # Add new blocks
    SSS.addFreqBlock(lims=(-0.5, 7.0), select=True)           # Block 1. The main range (mainly for plotting)
    SSS.addFreqBlock(lims=(6.0, 6.8), select=False)            # Block 2. Maleic acid
    SSS.addFreqBlock(lims=(3.25, 4.20), bslnOrder=(1,1), select=False)            # Block 3. Sugars
    SSS.addFreqBlock(lims=(0.5, 1.75), select=False)           # Block 4. Ethanol CH3
    SSS.addFreqBlock(lims=(5.1, 5.55), bslnOrder=(5,5), select=False)           # Block 5. Anomeric protons of Glucose
    SSS.addFreqBlock(lims=(2.5, 3.1), bslnOrder=(3,3), select=False)           # Block 6. Citric/Malic acid
    SSS.addFreqBlock(lims=(2.0, 2.2), bslnOrder=(2,2), select=False)           # Block 7. Acetic acid
    SSS.addFreqBlock(lims=(1.33, 1.6), bslnOrder=(3,3), select=False)          # Block 8. Lactic acid/Alanine
    SSS.addFreqBlock(lims=(3.30, 3.40), bslnOrder=(2,2), select=False)         # Block 9. Methanol
    SSS.addFreqBlock(lims=(0.75, .975), bslnOrder=(3,3), select=False)         # Block 10. Unidedentified volatile compounds
    SSS.addFreqBlock(lims=(4.59, 4.75), bslnOrder=(2,2), select=False)         # Block 11. Tartaric acid _with_ maleic acid
    SSS.addFreqBlock(lims=(4.45, 4.60), bslnOrder=(2,2), select=False)         # Block 12. Tartaric acid _without_ maleic acid
    SSS.addFreqBlock(lims=(4.00, 4.62), select=False)                          # Block 13. Adjusting peaks of lactic acid and malic acid close to water -- without reestimating their concentrations
    SSS.addFreqBlock(lims=(3.90, 4.20), bslnOrder=(1,1), select=False)         # Block 14. Peaks of fructose

def init_Steps_autoWine(SSS):
    SSS.steps.clear()
    SSS.steps.append(Step(script=start_fit))
    SSS.steps.append(Step(script=fit_autoPhase))
    SSS.steps.append(Step(script=fit_global_chsh))
    SSS.steps.append(Step(script=fit_ethanol_CH3))
    SSS.steps.append(Step(parsKeys=[('Water', 'chshQD', 0), ('Water', 'alphQD', 0)],
                          autoKeys = [('.', 'sigma2', 0), ('Water', 'ampl', 0)], frqBlkIds=[1], nrep=3, fitEach=True))     # Fit water
    SSS.steps.append(Step(parsKeys=[('Maleic acid', 'chshQD', 0), ('Maleic acid', 'alphQD', 0)],
                          autoKeys = [('.', 'sigma2', 0), ('Maleic acid', 'ampl', 0)], frqBlkIds=[2], nrep=3, fitEach=True))
    # Set up and fit the acids
    SSS.steps.append(Step(script=fit_acids))
    SSS.steps.append(Step(parsKeys=[('Acetic acid', 'chshQD', 0), ('Acetic acid', 'alphQD', 0)],
                          autoKeys = [('.', 'sigma2', 0), ('Acetic acid', 'ampl', 0)], frqBlkIds=[7], nrep=3, fitEach=True))     # Fit acetic acid
    SSS.steps.append(Step(parsKeys=[('Succinic acid', 'chshQD', 0), ('Succinic acid', 'alphQD', 0)],
                         autoKeys = [('.', 'sigma2', 0), ('Succinic acid', 'ampl', 0)], frqBlkIds=[6], nrep=3, fitEach=True))     # Fit succinic acid
    SSS.steps.append(Step(script=fit_lactic))        # Fits lactic acid and alanine
    # SSS.steps.append(Step(parsKeys=[('Acids', 'alph', 0)],
    #                       autoKeys = [('.', 'sigma2', 0), ('Acetic acid', 'ampl', 0), ('Succinic acid', 'ampl', 0), ('Malic acid', 'ampl', 0), ('Citric acid', 'ampl', 0), ('Lactic acid', 'ampl', 0)], frqBlkIds=[6, 7, 8]))
    # # Fit the sugars
    SSS.steps.append(Step(script=fit_sugars))
    # Fit methanol and butanediol
    SSS.steps.append(Step(parsKeys=[('Methanol', 'chshQD', 0), ('Methanol', 'alphQD', 0)],
                         autoKeys = [('.', 'sigma2', 0), ('Methanol', 'ampl', 0)], frqBlkIds=[9], nrep=3, fitEach=True))     # Fit methanol
    # SSS.steps.append(Step(parsKeys=[('2,3-Butanediol', 'chsh', 0), ('2,3-Butanediol', 'alph', 0)],
    #                      autoKeys = [('.', 'sigma2', 0), ('2,3-Butanediol', 'ampl', 0)], frqBlkIds=[10], fitEach=True))     # Fit butanediol
    SSS.steps.append(Step(script=fit_volatile))
    SSS.steps.append(Step(script=fit_water_neighborhood))

    SSS.steps.append(Step(script=finish_fit))
    SSS.steps.append(Step(parsKeys=[], autoKeys = [('.', 'sigma2', 0)], frqBlkIds=[1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14]))   # The last step to evaluate and plot the result # Add an empty step (no autofitting for amplitudes is selected)

def init_autoWine(wsp, resetSeries=True, resetTree=True, resetFreqBlks=True, resetSteps=True):
    """Initializes the workspace wsp for beverage analysis."""

    def saveResults_wine(self, filename='results.xlsx', parsKeys=None):
        """Saves the reults to an excel file."""

        # Create a workbook and add a worksheet.
        workbook = xlsxwriter.Workbook(filename)
        fmt_center = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'text_wrap':True})
        fmt_cenrot = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'rotation': 90})
        fmt_num3f = workbook.add_format()
        fmt_num3f.set_num_format('0.000')
        fmt_num2f = workbook.add_format()
        fmt_num2f.set_num_format('0.00')
        fmt_num1f = workbook.add_format()
        fmt_num1f.set_num_format('0.0')
        fmt_num0f = workbook.add_format()
        fmt_num0f.set_num_format('0')

        # ----------------------------------------------------------------------
        # Save only summary for wines
        worksheet = workbook.add_worksheet('SUMMARY')
        data_combs = {}        # Combinations of spectra to consider for the wine analysis
        for DDD in self.series[0].data:
            wine_name = get_sample_name(DDD)
            try:
                data_combs[wine_name].append(DDD.selfID())
            except KeyError: data_combs[wine_name] = [DDD.selfID()]

        # Write the header. Rows and columns are zero indexed.
        ncol_ampl = len(labels) + 1      # Number of columns for amplitudes (incl. Maleic acid)
        for i, (text, col_width) in enumerate(zip(['','Sample ID', 'Aqusition Date-Time', 'Acquisition time (absolute)', 'Estimated Density, g/L', 'Actual Alcohol, %v/v', 'Potential Alcohol, %v/v', 'BRIX', 'MassFrac of Maleic Acid (grav.), w/w', 'MassFrac of Maleic acid (estm.), w/w'], [3, 12, 5, 5, 10, 10, 10, 7, 10, 10])):
            worksheet.merge_range(0, i, 2, i, text, fmt_center)
            worksheet.set_column(i, i, col_width)
        worksheet.merge_range(0, 10, 1, 10+ncol_ampl+1-1, 'Absolute Concentrations, g/L', fmt_center)

        col = 10
        # Write the amplitude names
        for lbl in ['Maleic acid']+labels:
            worksheet.write( 2, col, lbl )
            col += 1
        worksheet.write( 2, col, 'Comment')
        worksheet.write( 2, col+1, 'IDs of used Data')
        col += 2

        # write the header for the individual datums
        for i, dat_type in enumerate(['dat_FULL', 'dat_PRES', 'dat_FULL_DRY', 'dat_PRES_DRY', 'dat_FULL_DRY_D2O', 'dat_PRES_DRY_D2O']):
            worksheet.merge_range(0, 12+ncol_ampl*(i+1), 0, 12+ncol_ampl*(i+2)-1, 'Absolute Concentrations, g/L', fmt_center)
            worksheet.merge_range(1, 12+ncol_ampl*(i+1), 1, 12+ncol_ampl*(i+2)-1, dat_type, fmt_center)
            for lbl in ['Maleic acid']+labels:
                worksheet.write( 2, col, lbl )
                col += 1

        # Write the amplitudes and parameters
        row = 3
        for name in sorted(data_combs.keys(), key=lambda x : (float( re.sub('[^0-9.]','', x.split('-')[0][1:]) ), x) ):          # Turn the name into number for sorting
            results = wine_results(data = [self.series[id[0]].data[id[1]] for id in data_combs[name]])

            worksheet.write(row, 0, row-1)
            worksheet.write(row, 1, name)
            worksheet.write(row, 2, results['acqu_time'])
            worksheet.write(row, 3, results['acqu_time_abs'])
            worksheet.write(row, 4, results['Density'], fmt_num3f)
            worksheet.write(row, 5, results['Actual Alcohol, %v\v'], fmt_num2f)
            worksheet.write(row, 6, results['Total Alcohol, %v\v'], fmt_num2f)
            worksheet.write(row, 7, results['BRIX'], fmt_num1f)
            worksheet.write(row, 8, results['mass_frac_MalAc_grav'], fmt_num3f)
            worksheet.write(row, 9, results['mass_frac_MalAc_estm'], fmt_num3f)

            # Write the results
            col = 10
            for lbl in ['Maleic acid']+labels:
                worksheet.write( row, col, results[lbl], fmt_num3f if results[lbl] != 0 else fmt_num0f )
                col += 1

            # Write a comment
            col += 1

            # Write the indices of used Datums
            worksheet.write(row, col, repr([data_combs[name]]) )
            col += 1

            # ----------------------------------------------------------------------
            # Save results for each Datum
            for DDD_type in ['dat_FULL', 'dat_PRES', 'dat_FULL_DRY', 'dat_PRES_DRY', 'dat_FULL_DRY_D2O', 'dat_PRES_DRY_D2O']:
                for lbl in ['Maleic acid']+labels:
                    if results[DDD_type] is not None:
                        worksheet.write( row, col, results[DDD_type][lbl], fmt_num3f if results[DDD_type][lbl] != 0 else fmt_num0f )
                    col += 1

            row += 1

        # ----------------------------------------------------------------------
        # Save complete results
        worksheet = workbook.add_worksheet('Complete results')

        # Keys of parameters to output
        if parsKeys is None:
            parsKeys = self.allParsKeys(parsKind=['chshQD', 'jcplQD'])
        chshKeys = sorted( [key for key in parsKeys if key[1]=='chshQD'] )
        jcplKeys = sorted( [key for key in parsKeys if key[1]=='jcplQD'] )

        # Write the header. Rows and columns are zero indexed.
        ncol_ampl = 2*len(self.repRootNames)       # Number of columns for amplitudes
        ncol_chsh = len(chshKeys)
        ncol_jcpl = len(jcplKeys)
        for i, (text, col_width) in enumerate(zip(['','ID', 'Series Name', 'Data Name', 'Arrayed Value'], [3, 5, 3, 15, 6])):
            worksheet.merge_range(0, i, 3, i, text, fmt_center)
            worksheet.set_column(i, i, col_width)
        worksheet.merge_range(0, 5, 1, 5+ncol_ampl-1, 'Absolute intensities of mixture components, a.u.', fmt_center)
        if ncol_chsh > 0:
            worksheet.merge_range(0, 5+ncol_ampl, 0, 5+ncol_ampl+ncol_chsh-1, 'Chemical shifts of spins, ppm', fmt_center)
        if ncol_jcpl > 0:
            worksheet.merge_range(0, 5+ncol_ampl+ncol_chsh, 0, 5+ncol_ampl+ncol_chsh+ncol_jcpl-1, 'J-coupling values, Hz', fmt_center)

        col = 5
        # Write the amplitude names
        for name in self.repRootNames:
            worksheet.merge_range(2, col, 2, col+1, name, fmt_center)
            worksheet.write_row(3, col, ['Intensity, a.u.', 'Variance'])
            col += 2
        # Write the chemical shift names
        for key in chshKeys:
            worksheet.write( 2, col, self.getPrior(key).label )
            worksheet.write( 3, col, str(key) )
            col += 1
        # Write the J-coupling names
        for key in jcplKeys:
            worksheet.write( 2, col, self.getPrior(key).label )
            worksheet.write( 3, col, str(key) )
            col += 1


        # Write the Series names in merged rows
        row_start = 4
        for ser in self.series:
            row_stop = row_start + len(ser.data) - 1
            if len(ser.data) > 1:
                # Merge rows if there are several Datum files in the Series
                worksheet.merge_range(row_start, 2, row_stop, 2, ser.name, fmt_cenrot)
            else:
                # Write horizontally, if tehre is just one Datum
                worksheet.write(row_start, 2, ser.name)
            row_start = row_stop + 1

        # Write the amplitudes and parameters
        row = 4
        for ser in self.series:
            for dat in ser.data:
                # Write the Datum ID and Name
                worksheet.write(row, 0, row-2)
                worksheet.write(row, 1, repr(dat.selfID()))
                worksheet.write(row, 3, dat.name)
                worksheet.write(row, 4, dat.arrVal)

                # Write the results
                col = 5
                for name in self.repRootNames:
                    key=(name, 'ampl', 0)
                    if not dat.isXclRootName(name):
                        ampl = dat.getCrntVal(key)
                        var = dat.smplDistF[key].var if key in dat.smplDistF.keys() else 0.0
                    else:
                        ampl, var = [0.0, 0.0]
                    worksheet.write_row(row, col, [ampl, var])
                    col += 2
                # Write the chemical shift and J-couplings values
                for key in chshKeys + jcplKeys:
                    val = dat.getCrntVal(key)
                    worksheet.write( row, col, val if not np.isnan(val) else 0.0 )
                    worksheet.write( 3, col, str(key) )
                    col += 1
                row += 1

        workbook.close()

    config.QD_AggregatePeaksThreshold = 0.0
    config.QD_RerunQDchshThreshold = 0.0
    config.OPTIM_startFrom = "default"
    wsp.extra['autoWine'] = True
    wsp.saveResults = MethodType(saveResults_wine, wsp)      # Update the saving function

    # Reset the series
    if resetSeries or len(wsp.series) == 0:
        wsp.series.clear()
        SSS = wsp.addSeries()
    else: SSS = wsp.series[0]

    # Set the chemical tree
    if resetTree:
        T = loadTree('autoWineTree.ctr')
        wsp.setTree(T)

        # # Set distributions' parameters (to be done in the tree)
        # # TODO: Update in the tree
        wsp.setGlobalPrior(key=('Mixture', 'chsh', 0), min=-0.35, max=0.35, dval=0.0)
        wsp.setGlobalPrior(key=('Acids', 'alph', 0), min=-1.0, max=5.0, dval=0.00)
        # wsp.setGlobalPrior(key=('Succinic acid', 'chshQD', 0), min=2.625, max=2.675, dval=2.655)
        # wsp.setGlobalPrior(key=('Succinic acid', 'alphQD', 0), min=-2.0, max=5.0, dval=0.0)
        # wsp.setGlobalPrior(key=('Acetic acid', 'chshQD', 0), min=2.05, max=2.1, dval=2.08)
        # wsp.setGlobalPrior(key=('Acetic acid', 'alphQD', 0), min=-2.5, max=5.0, dval=0.0)
        # wsp.setGlobalPrior(key=('Lactic acid', 'chshQD', 0), min=1.365, max=1.39, dval=1.381)
        # wsp.setGlobalPrior(key=('Lactic acid', 'alph', 0), min=-2.0, max=5.0, dval=0.00)
        # wsp.setGlobalPrior(key=('Alanine', 'alph', 0), min=-1.0, max=5.0, dval=0.00)
        # wsp.setGlobalPrior(key=('2,3-Butanediol', 'alph', 0), min=-2.0, max=5.0, dval=0.0)
        # wsp.setGlobalPrior(key=('Ethanol', 'chshQD', 0), min=1.165, max=1.19, dval=1.177)
        # wsp.setGlobalPrior(key=('Maleic acid', 'chshQD', 0), min=6.30, max=6.41, dval=6.390)

    # Set the frequency blocks
    if resetFreqBlks:
        init_freqBlocks_autoWine(SSS)

    if resetSteps:
        init_Steps_autoWine(SSS)

    return wsp

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

        # initialize the main window
        super(MainView, self).__init__(parent)
        self.resize(1200, 700)
        self.setAcceptDrops(True)      # Allow drag-and-drop
        self.setupGUI()

        # Initialize with an empty workspace
        self.wsp = wsp    # The Workspace; main class that holds all logic
        self.onResetWspAction()

        # Start the fitting thread
        self.fittingThread = FittingThread()
        self.fittingThread.finished.connect(self.onThreadFinished)
        self.fittingThread.terminated.connect(self.onThreadFinished)

    #     ## Install the custom output stream
    #     if compile_standalone:
    #         sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)
    #         sys.stderr = EmittingStream(textWritten=self.errorOutputWritten)
    #
    # def __del__(self):
    #     # Restore sys.stdout (if used to collect output to the console)
    #     if compile_standalone:
    #         sys.stdout = sys.__stdout__
    #         sys.stderr = sys.__stderr__

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
        self.setWindowTitle("Automated qNMR analysis of fermented beverages ver. {} ({})".format(version, str(date.today())) )

        # ----------------- set up the pie chart figure
        self.resFigure = Figure(facecolor='w', edgecolor='k')     # a figure instance to plot on
        self.resCanvas = FigureCanvas(self.resFigure)# this is the Canvas Widget that displays the `figure`; it takes the `figure` instance as a parameter to __init__
        self.resCanvas.setMinimumHeight(400)

        # ----------------- Spectrum figure in pyqtgraph -----------------------
        self.mainFigureWidget = MainSpectrumWidget()
        self.mainFigureWidget.setMinimumSize(550, 300)
        self.mainFigureWidget.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))

        # ----------------------------------------------------------------------
        #                            Navigation tab
        # ----------------------------------------------------------------------
        self.dataListWidget = QListWidget()
        self.dataListWidget.currentRowChanged.connect(self.setCurrent)

        # ---------------------------- Actions and toolbars ------------------------
        tbMain = self.addToolBar("File")               # Main toolbar
        tbTree = QToolBar("Tree")                      # Tree toolbar
        tbTree.setIconSize(QtCore.QSize(18,18))
        tbTree.setFloatable(False)
        tbTree.setMovable(False)
        self.setupActions(tbMain, tbTree)

        # Set up the navigation and results widgets
        tabNavi, tabRest = QWidget(), QWidget()
        layNavi, layRest = QHBoxLayout(), QHBoxLayout()
        tabNavi.setLayout(layNavi)
        tabRest.setLayout(layRest)
        layNavi.addWidget(self.dataListWidget)
        layRest.addWidget(self.resCanvas)
        layNavi.setContentsMargins(1,1,1,1)
        layRest.setContentsMargins(1,1,1,1)
        tabNavi.setMaximumHeight(150)                 # Set to 0 to hide the navigation widget

        # --------------- Left --------------------
        widgetLeft = QWidget()
        widgetLeft.setMinimumSize(250, 0)
        widgetLeft.setMaximumWidth(300)
        layLeft = QVBoxLayout()
        widgetLeft.setLayout(layLeft)
        layLeft.addWidget(tabNavi)
        layLeft.addWidget(tabRest)
        layLeft.setContentsMargins(1,1,1,1)

        # --------------- Main widget and layout ----------------
        widgetMain = QWidget()
        layMain = QHBoxLayout()
        layMain.addWidget(widgetLeft)
        layMain.addWidget(self.mainFigureWidget)
        widgetMain.setLayout(layMain)
        self.setCentralWidget(widgetMain)    # Set the result in the center of the window

        # ---------------------- Set up the status bar -------------------------
        self.statusBar= QStatusBar()
        self.statusBar.setMaximumHeight(16)
        self.setStatusBar(self.statusBar)
        self.progressBarFiles = QProgressBar()
        self.progressBarFiles.setMaximumHeight(16)
        self.progressBarFiles.setMaximumWidth(400)
        self.statusBar.addPermanentWidget(self.progressBarFiles)

    def setupActions(self, tbMain, tbTree):
        # Add clear action
        clearAction = QAction(QIcon('icons\icon_new.png'), 'Clear workspace', self)
        clearAction.setStatusTip('Clear the workspace')
        clearAction.triggered.connect(lambda : self.onResetWspAction(newWorkspace=None, newSettings=None))
        # Add import datafile action
        actnImportData = QAction(self._icon('icon_addFile.png'), 'Import files', self)
        actnImportData.setStatusTip('Import new data and add them to the current series')
        actnImportData.triggered.connect(self.onImportData)
        actnRemoveCurrent = QAction(self._icon('icon_removeFile.png'), 'Remove file', self)
        actnRemoveCurrent.setStatusTip('Remove file from the workspace')
        actnRemoveCurrent.triggered.connect(self.removeCurrent)
        # Add load action
        loadAction = QAction(self._icon('icon_load.png'), 'Load workspace', self)
        loadAction.setStatusTip('Load the workspace')
        loadAction.triggered.connect(self.onLoadWspAction)
        # Add save action
        saveAction = QAction(self._icon('icon_save.png'), 'Save workspace', self)
        saveAction.setShortcut('Ctrl+S')
        saveAction.setStatusTip('Save the workspace')
        saveAction.triggered.connect(self.onSaveWspAction)
        # Show the information dialog action
        actnAbout = QAction(QIcon('icons\icon_info.png'), 'About', self)
        actnAbout.setStatusTip('Information about the program')
        actnAbout.triggered.connect(self.showAboutMessage)

        # ----------------------- Actions for the plot -------------------------
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

        # ----------------------- Actions for the tree -------------------------
        # Fit the last step action
        self.actnFitLastStep = QAction(self._icon('icon_fitOneStep.png'), 'Fit active step', self)
        self.actnFitLastStep.setStatusTip('Fit the last step')
        self.actnFitLastStep.triggered.connect( lambda : self.startThread(queueActns=['Fit']) )
        # Fit all steps action
        self.actnFitAllSteps = QAction(self._icon('icon_fitAllSteps.png'), 'Fit all steps', self)
        self.actnFitAllSteps.setStatusTip('Fit all steps for this file')
        self.actnFitAllSteps.triggered.connect(lambda _ : self.fitAllSteps(selectedFiles = None))
        # Fit all files action
        self.actnFitAllFiles = QAction(self._icon('icon_fitAllFiles.png'), 'Fit all files', self)
        self.actnFitAllFiles.setStatusTip('Fit all steps for this file')
        self.actnFitAllFiles.triggered.connect(self.fitAllFiles)
        # Reset the fit but keep the files
        self.actnResetFit = QAction(self._icon('icon_magic.png'), 'Reset the fit', self)
        self.actnResetFit.setStatusTip('Resets the fitted parameters to default values but keeps loaded spectra in the workspace')
        self.actnResetFit.triggered.connect(lambda _ : init_autoWine(self.wsp, resetSeries=False))
        # Stop fitting action
        actnstopThread = QAction(self._icon('icon_stopFitting.png'), 'Stop fitting', self)
        actnstopThread.setStatusTip('Stop fitting')
        actnstopThread.triggered.connect(self.stopThread)
        # Save current results
        actnSaveResults = QAction(self._icon('icon_saveResults.png'), 'Save results to file', self)
        actnSaveResults.setStatusTip('Save all current results to file')
        actnSaveResults.triggered.connect(self.saveResults)

        # ------------------------- Set the toolbar ----------------------------
        tbMain.addAction(clearAction)
        tbMain.addAction(self.actnResetFit)
        tbMain.addAction(actnImportData)
        tbMain.addAction(actnRemoveCurrent)
        tbMain.addSeparator()
        tbMain.addAction(loadAction)
        tbMain.addAction(saveAction)
        tbMain.addSeparator()
        tbMain.addAction(self.actnToggleComps)
        tbMain.addAction(self.actnToggleResid)
        tbMain.addAction(self.actnAutoRange)
        tbMain.addSeparator()
        tbMain.addAction(actnSaveResults)

        # ---------------------- Toolbar for the tree --------------------------
        tbMain.addAction(self.actnFitAllSteps)
        tbMain.addAction(self.actnFitAllFiles)
        tbMain.addAction(actnstopThread)

    # -------------------- Processing keyboard interactions --------------------

    def keyPressEvent(self, ev):
        # self.scene().keyPressEvent(ev)
        # self.sigKeyPress.emit(ev)
        # print('Key pressed ', ev.key())
        pass

    # ------------------------- Other utility methods --------------------------
    def addDatumFromFile(self, path):
        """Imports a new spectrum and adds it to the workspace and the list widget of data."""
        crnt_series = self.wsp.series[0]
        dic = None

        if path[-3:] == '.1d':

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

        dat = crnt_series.addDatum(yT, name = name, extra=dic)

        # Add new entry to the data List
        newItem = QListWidgetItem(self._icon('icon_gof_none.png'), name, parent=self.dataListWidget)
        self.setCurrent(resetView=True if len(crnt_series.data)==1 else False)     # Set the last spectrum as current

        return dat

    def onImportData(self):
        """Runs a dialog to select a new file."""
        for newFilePath in QFileDialog.getOpenFileNames(None, 'Import file', '.', filter = "Spinsolve FID (*.1d)"):   # ;;JEOL FID (*.jdf)
            dat = self.addDatumFromFile(newFilePath)
        return dat

    def removeCurrent(self):
        """Removes currently selected spectrum."""
        crntID = self._crnt.selfID()
        if crntID[0] == 0 and crntID[1] is not None:
            self.wsp.series[0].data[crntID[1]].remove()
            self.dataListWidget.takeItem(crntID[1])

    def setCurrent(self, indx_crnt=-1, resetView=False):
        """Sets the _crnt Datum and updates the plot, tables, etc. accordingly."""
        try:
            self._crnt = self.wsp.series[0].data[indx_crnt]
            self.dataListWidget.setCurrentRow(self.dataListWidget.count()-1 if indx_crnt == -1 else indx_crnt)
        except IndexError:
            # If there are no Series/Datums
            self._crnt = self.wsp
            self.dataListWidget.clear()

        self.plotCurrent(autoRange=resetView)

    def onSaveWspAction(self):
        """Saves the workspace including the stepClass class and the steps array."""
        filename = QFileDialog.getSaveFileName(parent=self, caption='Select output file', directory='.', filter='NMR worksapce (*.wsp)')
        if filename:
            if filename[-4:] != '.wsp': filename += '.wsp'

            # Pack the logic of the workspace
            dataPack = self.wsp.pack()

            # Pack the settings
            stngPack = copy.deepcopy(settings)
            stngConfig = config.as_dict()
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

        self.wsp.reset() # Reset the workspace
        self.setCurrent(resetView=True)     # Set the last spectrum as current
        if newWorkspace is None:
            self.wsp = init_autoWine(self.wsp)
        else:
            try:
                if newWorkspace['extra']['autoWine'] == True:
                    self.wsp.unpack(newWorkspace)
                    for dat in self.wsp.series[0].data:
                        newItem = QListWidgetItem(self._icon('icon_gof_none.png'), dat.name, parent=self.dataListWidget)
                    # Update the global config
                    if newSettings is not None:
                        try:
                            stngConfig = newSettings.pop('_config')
                            config.from_dict(config, stngConfig)
                        except KeyError: pass
            except KeyError: self.onResetWspAction()      # If autoWine is not the loaded workspace

        self.setCurrent(resetView=True)     # Set the last spectrum as current

    def resetSignals(self, zff=None, apod=None, flagAdapFreq=None):
        self._crnt.resetFreqs(zff, apod)

        if flagAdapFreq is not None:
            for dat in self._crnt.parent.data:
                dat.resetSignals(flagAdapFreq)           # Or simply dat.resetSignals(flagAdapFreq) to reset the adaptive flag for a single (current) Datum only

        self.preprocTool.setNewDatum(self._crnt)         # Update the statistics display
        self.plotCurrent(autoRange=(apod is None))       # Do not autorange if what has changed is only apodization

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

    # ------------------ Working with the fitting thread -----------------------

    def startThread(self, queueFiles=None, queueActns=None):
        """Fits the files in the queueFiles list."""
        # Disable controls that can start fitting
        self.setCursor(Qt.BusyCursor)

        if queueFiles is None: queueFiles = [self._crnt]

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

    def onParameterChange(self, key, val):
        """Sets a new value to the parameter key."""

        oldVal = self._crnt.getCrntVal(key)

        if not np.isclose(val, oldVal):

            self._crnt.setCrntVal(key, val)

        self.startThread()

    def onPhased(self, p0deg, p1deg):
        """Gets the phasing values from the phasing tool widget and sets current parameters accordingly."""
        nf = next_pow_of_2( 2**self._crnt.zff * len(self._crnt.t) )     # Determine the number of samples in the FULL signal spectrum (possibly including zero-filling). zff and t are taken from the Series level
        dt = self._crnt.t[1]-self._crnt.t[0]

        d_theta, d_tau = deg2tau(dt, nf, p0deg, p1deg)
        if not ( np.isclose(d_theta, 0) and np.isclose(d_tau, 0) ):
            theta = self._crnt.getCrntVal(key=('.', 'theta', 0))
            tau = self._crnt.getCrntVal(key=('.', 'tau', 0))

            self._crnt.setCrntVal(key=('.', 'theta', 0), val = (theta+d_theta + np.pi) % np.pi - np.pi )     # make sure the phase stays in the (-180.0, 180.0) interval  # p0deg = (p0deg + 180.0) % 360.0 - 180.0
            self._crnt.setCrntVal(key=('.', 'tau', 0), val = tau + d_tau)

        self.startThread()

    def fitAllSteps(self, selectedFiles = None):
        """Fits all steps in selected files; if no files are selected, uses the current file/series. The starting values on the next step are copied from the current found values."""
        # Form the list of steps to Fit
        stepIdsToFit = list(range(len(self._crnt.steps)))

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

        # Reset the workspace/current parameters without resetting the data in the series
        init_autoWine(self.wsp, resetSeries=False)

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
        """Plots the spectrum/results for a specific datum dat."""
        if isinstance(self._crnt, Datum):
            if self._crnt.zF is None: self._crnt.evaluate(autoKeys=[], returnSignals=True)
            f, yFph, xF, zF, bF = self._crnt.signals_for_plot()

            # Plotting function
            self.mainFigureWidget.plot(f, yFph, xF,
                zF = zF if self.actnToggleComps.isChecked() else None, stems =  None, phasingPivot=False, freqBlocks=None, show_yaxis=False,
                # freqBlocks=[(blk.min, blk.max, (i in self._crnt.steps[-1].frqBlkIds) ) for i, blk in enumerate(self._crnt.freqBlocks)],
                indx_colr=[i for i, name in enumerate(self._crnt.repRootNames) if not self._crnt.isXclRootName(name)])   # if i in self._crnt.steps[-1].frqBlkIds])

            if autoRange:
                self.mainFigureWidget.autoRange(xlims=(0.0, 7.0))

            # self.mainFigureWidget.getItem(0, 0).vb.setMouseEnabled(x=False)

            # Output the found results
            try:
                self.plotResultsChart()
            except KeyError: pass
        else:
            self.mainFigureWidget.reset()
            self.plotResultsChart(reset=True)

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

    def plotResultsChart(self, reset=False):
        """Plots a pie chart that represents the found component concentrations."""

        self.resFigure.clear()

        def hover(evt):
            if evt.inaxes in ax_bar:
                # Find which bar contains the event
                for indx, patch in enumerate(bars.patches):
                    if patch.contains(evt)[0]:
                        newMessage = '{:s}    {:.3g} g/L'.format(labels[indx], vals[indx])
                        if self.statusBar.currentMessage() != newMessage:
                            self.statusBar.showMessage(newMessage)
                        return
            elif evt.inaxes == ax_pie:
                # Find which wedge contains the event
                return

            self.statusBar.clearMessage()

        def axes_intervals(vals):
            """Determine the ranges for the subplots"""

            def round_lims(val):
                """Return min and max values of vertical limits of the plot need to correctly represent the value."""
                if val < 2:
                    min_lim, max_lim = math.floor(0.95*val*5)/5, math.ceil(1.01*val*5)/5
                elif val < 20:
                    min_lim, max_lim = math.floor(0.95*val/2)*2, math.ceil(1.01*val/2)*2
                elif val < 200:
                    min_lim, max_lim = math.floor(0.95*val/5)*5, math.ceil(1.01*val/5)*5
                else:
                    min_lim, max_lim = math.floor(0.95*val/10)*10, math.ceil(1.01*val/10)*10

                return minmaxTuple(max(0.0, min_lim), max(max_lim, 0.1))

            intervals = merge_intervals([round_lims(val) for val in vals])
            gaps_size = (np.array([interval.min for interval in intervals])[1:] - np.array([interval.max for interval in intervals])[:-1]) / np.array([interval.max for interval in intervals])[:-1]      # Relative gap sizes between the intervals

            boundaries = [intervals[0].min, intervals[-1].max]       # Selected boundaries for the plotting
            for gap_indx in np.argsort(gaps_size)[-2:]:
                # Considr only the largest two gaps
                if gaps_size[gap_indx] > 1:
                    boundaries.extend([intervals[gap_indx].max, intervals[gap_indx+1].min])
            boundaries = sorted(boundaries)

            return [(x,y) for x, y in zip(boundaries[::2], boundaries[1::2])]

        # Remove the reference to the hovering event
        try:
            self.resCanvas.mpl_disconnect(self._cid_hover)
        except AttributeError: pass

        if not reset:
            # Get the concentrations in g/L; choose only related Datums
            wine_name = get_sample_name(self._crnt)
            conc_gL = wine_results(data = [DDD for DDD in self._crnt.parent.data if wine_name == get_sample_name(DDD)])       # Use all datums in the series

            vals = np.array([conc_gL[lbl] for lbl in labels if 'Peak' not in lbl])
            vals = np.where(np.isnan(vals), 0.0, vals)

            # Define the subplots
            intervals = axes_intervals(vals)         # Determine the ranges for the subplots (broken axes)
            n_ax = len(intervals)      # Number of axes in the broken graph
            gs_top = gridspec.GridSpec(n_ax+1, 2, bottom=0.0, top=0.99, hspace=0.0, wspace=0.0)
            gs_bot = gridspec.GridSpec(n_ax+1, 2, hspace=0.03)
            ax_pie = self.resFigure.add_subplot(gs_top[0, 0], aspect="equal")
            # ax_tab = self.resFigure.add_subplot(gs_top[0, 1], aspect="equal")
            ax_bar = [self.resFigure.add_subplot(gs_bot[n_ax,:])]
            if n_ax > 1:
                for i in range(n_ax-1, 0, -1):
                    ax_bar.append(self.resFigure.add_subplot(gs_bot[i,:], sharex=ax_bar[0]) )


            # ---------------------------- Doughnut chart --------------------------------
            data_pie = [sum([conc_gL[key] for key in ['Ethanol', 'Glycerol', 'Methanol', '2,3-Butanediol']]),
                    sum([conc_gL[key] for key in ['Glucose', 'Fructose', 'Sucrose', 'Sorbitol']]),
                    sum([conc_gL[key] for key in ['Lactic acid', 'Acetic acid', 'Malic acid', 'Citric acid', 'Succinic acid']])]
            act_alc_vv, tot_alc_vv = cww2pvv(conc_gL)        # Actual and total alcoholic strength

            wedges, texts = ax_pie.pie(data_pie, wedgeprops=dict(width=0.3), startangle=45)
            ax_pie.text(0, 0, '{:.1f}%'.format(conc_gL['Total Alcohol, %v\v']), fontsize=18, family='cursive',
                         horizontalalignment='center', verticalalignment='center')

            # -------------------------------- Table -------------------------------------
            # ax_tab.clear()
            # ax_tab.axis('off')
            # cellText = [['Tot. sugars, g/L', '1'], ['Tot. alcohol, g/L', '2'], ['Tot. acidity, g/L','3']]
            # table = ax_tab.table(cellText=cellText,
            #             colWidths=[0.75, 0.25], loc='center')
            # table.auto_set_font_size(False)
            # table.set_fontsize(14)

            # ------------------------------- Bar chart ----------------------------------
            bar_labels = [lbl for lbl in labels if 'Peak' not in lbl]
            for i in range(n_ax-1, -1, -1):
                ax = ax_bar[i]
                x = np.arange(len(bar_labels))  # the label locations
                bars = ax.bar(x, vals, align='center', width=0.75,
                              tick_label=[abbrev[lbl] for lbl in bar_labels])
                ax.spines['top'].set_visible(False)        # Don't show the top spine
                ax.tick_params(length=3, labelsize=10, pad=2)
                if i == 0:
                    # Bottom plot
                    ax.set_xticks(x)
                    ax.tick_params(top=False, right=False)
                    ax.set_xlim(-0.5, len(bar_labels)-0.5)
                    for tick in ax.get_xticklabels():
                        tick.set_rotation('vertical')
                else:
                    # The rest of the plots
                    ax.tick_params(bottom=False, top=False, right=False)
                    for tick in ax.get_xticklabels():
                        tick.set_visible(False)
                    ax.spines['bottom'].set_color((0.8, 0.8, 0.8))
                    ax.spines['bottom'].set_linestyle('--')
                    if i == n_ax-1:
                        # The top plot
                        ax.spines['top'].set_visible(True)

            for ax, lims in zip(ax_bar, intervals):
                ax.set_ylim(*lims)

            ttl = ax_bar[-1].set_title('Concentrations, g/L', fontsize=16)
            ttl.set_position((0.5, 1.03))



            self._cid_hover = self.resCanvas.mpl_connect("motion_notify_event", hover)

        self.resCanvas.draw()

# ----------------------- Handling Drag-and-Drop events ------------------------
    def dragEnterEvent(self, evt):
        if evt.mimeData().hasUrls():
            evt.acceptProposedAction()

    def dragMoveEvent(self, evt):
        if evt.mimeData().hasUrls():
            evt.acceptProposedAction()

    def dropEvent(self, evt):

        def read_folders(rootPath, pathList=None):
            """Recursively opens folders and returns paths to data.1d files, if found."""
            if pathList is None: pathList = []

            if os.path.isdir(rootPath):
                if 'data.1d' in os.listdir(rootPath):
                    pathList.append(os.path.join(rootPath, 'data.1d'))
                else:
                    for file in os.listdir(rootPath):
                        read_folders(os.path.join(rootPath, file), pathList)

            return pathList

        # Load the files
        filePathList = []
        for url in evt.mimeData().urls():
            filePathList = read_folders(url.toLocalFile(), filePathList)

        for filePath in filePathList:
            dat = self.addDatumFromFile(filePath)

        # Run the optimization
        pass

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
