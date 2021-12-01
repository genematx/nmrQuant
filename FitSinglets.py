# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 09:43:54 2021

@author: djh208
"""

# Initialization - hopefully none of these needed
#%reset -f  
#%load_ext autoreload
#%autoreload 2
#%precision 4

# Add the main software directory to the path
#import sys, os
#sys.path.insert(0, os.path.abspath('.'))  # By default it is the parent directory

# Import necessary modules
import numpy as np
#import pandas as pd
from chemTree import *
from MainLogic import *
from MainLogic import Workspace#, Series, Datum
from dataio import read_any_file, saveFID

#from pprint import pprint

from optparse import OptionParser

#SCRIPT_PATH = os.path.abspath(os.path.dirname(sys.argv[0]))
#CALLED_PATH = os.getcwd()                                          # Where it has been executed from
#DATA_PATH = None

def read_InputFile(filename):       # move to appropriate spot if needed

    """
    Read an InputFile file into a dictionary.


    Parameters
    ----------
    filename : str
        Filename of the input file .in file.

    Returns
    -------
    fitpars : dict
        Dictionary of parameters in file.


    """
    fitpars = {}  # create empty dictionary

    with open(filename, 'r') as f:
        cnt = 0         # counter to track number of components to be fitted
        while True:     # loop until end of file is found

            line = f.readline().rstrip()    # read a line
            if line == '':      # end of file found
                break

            else:
                line = line.split('==')
                key = line[0].strip()
                val = line[1].strip()
                if key == 'Fitting Block' :
                    fitpars['FrqBlk']=val.strip('[]')
                    
                elif key == 'Component':
                    param = {}       
                    comp = val.split(';')
                    param['peakname'] = comp[0].strip('[]\'')
                    param['peaktype'] = comp[1].strip().strip('\'')
                    if param['peaktype'] == 'Singlet':
                        par1 = comp[2].split('>')
                        par2 = par1[0].split('=')                                           #todo here need to modify for multiple peaks/chsh values
        
                        key = (param['peakname'], par2[0].strip().strip('\''), par2[1].strip('\' '))      # define parameters as belonging to peak1 chsh.
     
                        if par2[0].strip().strip('\'')=='chsh':                                                 #todo not right here. will not handle multiple chsh vals. may need to modify structure
                            vals = par1[1].split(',')
                            chsh = {}
                            for index in range(len(vals)):
                                item = vals[index].split('=')
                                chsh[item[0].strip()]=item[1].strip()
                            param['chsh']=chsh
                    fitpars[cnt] = param
                    cnt += 1
    
    return fitpars


def saveResultsXLS(wsp, filename='results.xlsx', parsKeys=None):
    """Saves the reults to an excel file."""

    # Create a workbook and add a worksheet.
    workbook = xlsxwriter.Workbook(filename)
    fmt_center = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'text_wrap':True})
    fmt_cenrot = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'rotation': 90})
    fmt_num3f = workbook.add_format()
    fmt_num3f.set_num_format('0.000')
    fmt_num3f_ita = workbook.add_format()
    fmt_num3f_ita.set_num_format('0.000')
    fmt_num3f_ita.set_italic()
    fmt_num2f = workbook.add_format()
    fmt_num2f.set_num_format('0.00')
    fmt_num1f = workbook.add_format()
    fmt_num1f.set_num_format('0.0')
    fmt_num0f = workbook.add_format()
    fmt_num0f.set_num_format('0')


    # ----------------------------------------------------------------------
    # Save complete results
    
    worksheet = workbook.add_worksheet('Complete results')

    # Keys of parameters to output
    if parsKeys is None:
        parsKeys = wsp.allParsKeys(parsKind=['chshQD', 'jcplQD'])
    chshKeys = sorted( [key for key in parsKeys if key[1]=='chshQD'] )
    jcplKeys = sorted( [key for key in parsKeys if key[1]=='jcplQD'] )

    # Write the header. Rows and columns are zero indexed.
    ncol_ampl = 2*len(wsp.repRootNames)       # Number of columns for amplitudes
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
    for name in wsp.repRootNames:
        worksheet.merge_range(2, col, 2, col+1, name, fmt_center)
        worksheet.write_row(3, col, ['Intensity, a.u.', 'Variance'])
        col += 2
    # Write the chemical shift names
    for key in chshKeys:
        worksheet.write( 2, col, wsp.getPrior(key).label )
        worksheet.write( 3, col, str(key) )
        col += 1
    # Write the J-coupling names
    for key in jcplKeys:
        worksheet.write( 2, col, wsp.getPrior(key).label )
        worksheet.write( 3, col, str(key) )
        col += 1


    # Write the Series names in merged rows
    row_start = 4
    for ser in wsp.series:
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
    for ser in wsp.series:
        for dat in ser.data:
            # Write the Datum ID and Name
            worksheet.write(row, 0, row-2)
            worksheet.write(row, 1, repr(dat.selfID()))
            worksheet.write(row, 3, dat.name)
            worksheet.write(row, 4, dat.arrVal)

            # Write the results
            col = 5
            for name in wsp.repRootNames:
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

def saveResultsCSV(wsp, filename='results.csv', parsKeys=None):
    """Saves the reults to a csv file."""


    # Keys of parameters to output
    if parsKeys is None:
        parsKeys = wsp.allParsKeys(parsKind=['chshQD', 'jcplQD'])

    # ----------------------------------------------------------------------
    # Save complete results
    with open(filename, 'w') as fid:
        # Write the amplitude names
        for name in wsp.repRootNames:
            fid.write(name)
            key=(name, 'ampl', 0)
            if not dat.isXclRootName(name):
                ampl = dat.getCrntVal(key)
                var = dat.smplDistF[key].var if key in dat.smplDistF.keys() else 0.0
            else:
                ampl, var = [0.0, 0.0]
            fid.write(', ampl = {}, '.format(ampl))  
            fid.write('{}, '.format(var))  

            chshKeys = sorted( [key for key in parsKeys if key[1]=='chshQD' and key[0]==name] )
            jcplKeys = sorted( [key for key in parsKeys if key[1]=='jcplQD'] )

            fid.write('chsh = ')
            for key in chshKeys:
                val = dat.getCrntVal(key)
                fid.write('{}, '.format(val if not np.isnan(val) else 0.0))
            
            if len(jcplKeys)>0:
                fid.write('jcpl = ')
                for key in jcplKeys:
                    val = dat.getCrntVal(key)
                    fid.write('{}, '.format(val if not np.isnan(val) else 0.0))
            fid.write('\n')    


def make_parser():
    """Construct an option parser"""

    usage = """%prog: args

    """

    parser = OptionParser(usage)

    # add_options takes at least one long option
    # you can optionally include a short option.
    # - are one character (short) options (e.g. -h)
    #
    # -- are long options, the name is also used as the
    #    variable name attached that holds the option

    # parser.add_option("-e", "--error", help="Set error code")

    parser.add_option("-i", "--inputfile", help="The input file specifying peak positions")

    parser.add_option('-d', '--datadir', help='The FID data file')

    parser.add_option('-s', '--save', help='Save the workspace', action="store_true")
    
    parser.add_option('-o', '--outputfolder', help='Folder to save the FID and results')

    # parser.add_option('-v', '--verbose', action="store_true", help='Output the model fitting progress')

    # # opt_parse can be configured to store different kinds of values
    # # like filenames, and boolean options
    # parser.add_option("-b", "--bad-option", action="store_true",
    #                   help="trigger an option error")
    #
    # # you can also do simple type checking on parameters
    # parser.add_option("-n", "--number", help="set a number", type="int")
    #
    # # if needed you can tell optparse to use a different variable name.
    # # with the dest argument.
    # parser.add_option("--index", dest="createRDSIndex", action="store_true")
    #
    # parser.add_option("--make-template", action="store_true",
    #                   help="print a simplified template")

    parser.set_defaults(verbose=False,
                        createRDSIndex=False,
                        #datadir= 'blank',
                        datadir = 'docs\data_tutorial\S20.10\data.1d',
                        #os.path.join( SCRIPT_PATH, 'docs\data_tutorial\S20.10\data.1d'),
                        inputfile= 'ExampleInputFile.in', 
                        #os.path.join( SCRIPT_PATH, 'ExampleInputFile.in'),
                        outputfolder= 'simResults', 
                        #os.path.join( SCRIPT_PATH, 'simResults'),
                        error=None,
                        save=True,
                        number=0)

    return parser


parser = make_parser()
opts, args = parser.parse_args(sys.argv[1:])

#filename = 'ExampleInputFile.in'

fitpars = read_InputFile(opts.inputfile)  ## once fixed should be able to use this line
fitpars['filename'] = opts.inputfile

# Read the FID (yT), array of time stamps (t), and dictionary of acquisition parameters (dic)
t, yT, dic = read_any_file(opts.datadir)          # pass spinsolve data directory name here


            
# Create a new Workspace
wsp = Workspace(HCmode='1H')

# 1. Add first Series with two Spinsolve spectra
ser1 = wsp.addSeries(name = 'Series 1')

# Create an instance of a chemNode
for x in range(0,len(fitpars)-2):
    comp = fitpars[x]
    peakname = comp['peakname'].strip()
    peaktype = comp['peaktype'].strip()
    newNode = chemNodeQM(peakname,nameDB=peaktype)                      
    wsp.addTreeNode(newNode)
    wsp.setRepRoot(peakname, True)          # set peak as reportable

# Set the parameters in the Series
ser1.c0, ser1.f0, ser1.t = dic['c0'], dic['f0'], t
ser1.fullReset()

# Add the first spectrum
ser1.addDatum(yT, name=dic['name'])


# Add new frequency blocks
ranges = fitpars['FrqBlk']
ranges = ranges.split(';')
for x in range(len(ranges)):
    lim=ranges[x].split(',')
    ser1.addFreqBlock(lims=(float(lim[0]), float(lim[1])), select=True)       # Block 2. TMSP
 
# zero fill, apodization and initial phasing
ser1.resetFreqs(zff=1, apod=0.5)
dat = wsp[0,0]
dat.auto_phase(fit_Ph1=False)



# Display all frequency blocks in a Series
#ser1.showFreqBlocks()

for x in range(0,len(fitpars)-2):
    comp = fitpars[x]
    peakname = comp['peakname'].strip()
    peaktype = comp['peaktype'].strip()
    # Set default value and alter the range for a specific Datum
                    #todo here need to modify for multiple peaks/chsh values
    for mod in chemLib:
        if peaktype in chemLib[mod]:
            if peaktype == 'Singlet':
                key = (peakname, 'chshQD', 0)      # define parameters as belonging to peak1 chsh.
                chshval=float(comp['chsh']['dval'].strip('\''))
                minval=float(comp['chsh']['min'].strip('\''))
                maxval=float(comp['chsh']['max'].strip('\''))
                dat.setPrior(key, dval=chshval, min=minval, max=maxval, label=peakname)
                dat.setCrntVal(key, val=chshval)
        
                # Set up lists of parameters to optimize
                parsKeys = [(peakname, 'chshQD', 0), (peakname, 'alphQD', 0)]
                # Check which frequency block to use
                for y in range(len(ranges)):
                    lim=ranges[y].split(',')
                    if chshval > float(lim[0]) and chshval < float(lim[1]):
                        id = y + 1
                        break
                
        
                # Run the optimization algorithm with the selected autoKeys
                dat.optimize(parsKeys=parsKeys, autoKeys='All', frqBlkIds=[id], nhop=3);
                break
            
            else :
                for val in range(len(chemLib[mod][peaktype].chshH)):
                    key = (peakname, 'chshQD', val)      # define parameters as belonging to peak1 chsh.
#                    dat.setCrntVal(key, val=chshval)
                    chshval=dat.getCrntVal(key)
                    # Set up lists of parameters to optimize
                    parsKeys = [(peakname, 'chshQD', val), (peakname, 'alphQD', val)]
                    # Check which frequency block to use
                    for y in range(len(ranges)):
                        lim=ranges[y].split(',')
                        if chshval > float(lim[0]) and chshval < float(lim[1]):
                            id = y + 1
                            break
            
                    # Run the optimization algorithm with the selected autoKeys
                    dat.optimize(parsKeys=parsKeys, autoKeys='All', frqBlkIds=[id], nhop=3);

                for val in range(len(chemLib[mod][peaktype].jcplH)):
#                    key = (peakname, 'jcplQD', val)      # define parameters as belonging to peak1 chsh.
#                    dat.setCrntVal(key, val=chshval)
        
                    # Set up lists of parameters to optimize
                    parsKeys = [(peakname, 'jcplQD', val)]
                    # Check which frequency block to use: How to do this for jcpl?
#                    for y in range(len(ranges)):    
#                        lim=ranges[y].split(',')
#                        if chshval > float(lim[0]) and chshval < float(lim[1]):
#                            id = y + 1
#                            break
            
                    # Run the optimization algorithm with the selected autoKeys
                    dat.optimize(parsKeys=parsKeys, autoKeys='All', frqBlkIds=[0], nhop=3);
    

        else:
            continue
        break


xT, xF = dat.modelled_signal(phased=True, bl_corr=True)
dt=np.asscalar(dat.t[1]-dat.t[0])
saveFID(xT, dat.c0, dat.f0, dt, tau=0., fname=opts.outputfolder)
saveResultsCSV(wsp, os.path.join(opts.outputfolder, 'results.csv'))


## Plot the spectrum and the residual
## Re-evaluate to get the values of amplitudes without recomputing them (set autoKeys=[])
#dat.evaluate(autoKeys=[], frqBlkIds=[0], returnSignals=True)
#
#import matplotlib.pyplot as plt
#from matplotlib import cm, rcParams
#fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
#dat.plot(ax_main=ax[0], ax_residual=ax[1], showRanges=None, showComponents=True)
#ax[0].set_xlim(9.0, -1.0)
#ax[0].set_ylim(-200, 4000)
#fig.show()




