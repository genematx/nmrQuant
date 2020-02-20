#!/usr/bin/env python

# #!/usr/bin/env python - will search for the first python
#                         interpreter on your path
# unlike
# #!/usr/bin/python2.5  - which will only run if there is a file
#                         python 2.5 is installed at /usr/bin
"""Module summary

If you import this module and do help (module) you'll see this.
The first line of a docstring is the "summary", and should be
a one line description.

You can go into more detail after the summary if needed.
See http://www.python.org/dev/peps/pep-0257/
for the python docstring style guide.
"""
from optparse import OptionParser
import sys, os, time
from io import BytesIO
import config
import subprocess

import numpy as np
from chemTree import loadTree
from MainLogic import Workspace, next_pow_of_2, save_workspace
from dataio import read_spinsolve
import matplotlib.pyplot as plt
from matplotlib import rcParams
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, Flowable
from reportlab.lib.styles import getSampleStyleSheet

from PIL import ImageFile

rcParams.update({'font.size': 10})
rcParams['ps.fonttype'] = 42               # Needed to be able to save figures in .eps format
# SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))       # Where the exe is located
SCRIPT_PATH = os.path.abspath(os.path.dirname(sys.argv[0]))
CALLED_PATH = os.getcwd()                                          # Where it has been executed from

def script_sugars(dirName, showFigure=True, showResidual=False):
    # Load the data
    yT, c0, f0, dt, pars = read_spinsolve(dirName)

    try:
        acquScore = 1.0 if pars['acqDelay'] > 10.0 else 0.0
    except KeyError:
        acquScore = 0.5

    # Define the array of time samples
    nt = yT.shape[0]
    t = np.linspace(start=0, stop=(nt-1)*dt, num=nt).reshape(-1,1)

    # Create a new Workspace and add the data there
    wsp = Workspace()
    ser = wsp.addSeries(c0=c0, f0=f0, t=t)
    DDD = ser.addDatum(yT)

    # Define and set the chemical tree
    tree_fname = 'GluSucFruMaCaEthWater.ctr'
    try:
        T = loadTree(tree_fname)
    except FileNotFounderror:
        T = loadTree(os.path.join(SCRIPT_PATH, tree_fname))
    wsp.setTree(T)

    # Align the spectrum based on the frequency of the water peak
    DDD.alignToSolventPeak(chshTo=4.75)

    # Define regions for optimization
    DDD.addFreqBlock(lims=(0.0, 10.0))                     # Block #1
    DDD.addFreqBlock(lims=(3.0, 4.3), bslnOrder=(1,0))    # Block #2 - main sugars
    DDD.addFreqBlock(lims=(5.0, 5.6), bslnOrder=(3,0))     # Block #3 - sugars anomeric protons
    DDD.addFreqBlock(lims=(2.4, 3.1), bslnOrder=(0,0))     # Block #4 - acids
    DDD.addFreqBlock(lims=(0.6, 1.75), bslnOrder=(1,0))    # Block #5 - ethanol

    # Define lists of autoKeys (for convenience)
    autoKeys = {'Water' : [('.', 'sigma2', 0), ('Water', 'ampl', 0)],
                'WaterSugars': [('.', 'sigma2', 0), ('Water', 'ampl', 0), ('Glucose', 'ampl', 0), ('Fructose', 'ampl', 0), ('Sucrose', 'ampl', 0)],
                'Sugars' : [('.', 'sigma2', 0), ('Glucose', 'ampl', 0), ('Fructose', 'ampl', 0), ('Sucrose', 'ampl', 0)],
                'Acids' : [('.', 'sigma2', 0), ('Citric acid', 'ampl', 0), ('Malic acid', 'ampl', 0)],
                'Ethanol' : [('.', 'sigma2', 0), ('Ethanol', 'ampl', 0)]}

    # # --------------------------- Run optimization -----------------------------

    # Optimize global chemical shift
    DDD.optimize(frqBlkIds=[1], autoKeys=autoKeys['WaterSugars'],
                 parsKeys=[('Mixture', 'chsh', 0), ('Water-SPSY1', 'chshQD', 0)])

    # Optimize the peak width
    DDD.optimize(frqBlkIds=[1], autoKeys=autoKeys['WaterSugars'], parsKeys=[('Water-SPSY1', 'alphQD', 0)])
    # DDD.optimize(frqBlkIds=[1], parsKeys=[('Mixture', 'alph', 0), ('Water-SPSY1', 'alphQD', 0)])

    DDD.optimize(frqBlkIds=[2, 3], autoKeys=autoKeys['WaterSugars'], parsKeys=[('Sugars', 'chsh', 0)], nhop=5)
    DDD.optimize(frqBlkIds=[2, 3], autoKeys=autoKeys['WaterSugars'], parsKeys=[('Sugars', 'alph', 0)])
    DDD.optimize(frqBlkIds=[1], autoKeys=autoKeys['WaterSugars'], parsKeys=[('Water-SPSY1', 'chshQD', 0)])
    DDD.optimize(frqBlkIds=[1], autoKeys=autoKeys['WaterSugars'], parsKeys=[('Water-SPSY1', 'alphQD', 0)])
    DDD.optimize(frqBlkIds=[2, 3], autoKeys=autoKeys['Sugars'], parsKeys=[('Sugars', 'chsh', 0)])
    DDD.optimize(frqBlkIds=[2, 3], autoKeys=autoKeys['Sugars'], parsKeys=[('Sugars', 'alph', 0)])

    # # Correct the phasing
    # DDD.setPrior(key=('.', 'theta', 0), distr='Constant')
    # DDD.adjust_phase(mw=512, mode='PhA')

    DDD.optimize(frqBlkIds=[4], autoKeys=autoKeys['Acids'], parsKeys=[('Acids', 'chsh', 0)], nhop=5)
    DDD.optimize(frqBlkIds=[4], autoKeys=autoKeys['Acids'], parsKeys=[('Acids', 'alph', 0)])
    DDD.optimize(frqBlkIds=[4], autoKeys=autoKeys['Acids'], parsKeys=[('Malic acid', 'chsh', 0), ('Citric acid', 'chsh', 0)], nhop=5)
    DDD.optimize(frqBlkIds=[4], autoKeys=autoKeys['Acids'], parsKeys=[('Acids', 'alph', 0)])

    # Optimize ethanol
    DDD.setPrior(key=('Ethanol', 'alph', 0), min=-DDD.getCrntVal(key=('Mixture', 'alph', 0)))    # Set the minimum bound on alpha
    DDD.optimize(frqBlkIds=[5], autoKeys=autoKeys['Ethanol'], parsKeys=[('Ethanol', 'chsh', 0)], nhop=3)
    DDD.optimize(frqBlkIds=[5], autoKeys=autoKeys['Ethanol'], parsKeys=[('Ethanol', 'alph', 0)])

    val, meta = DDD.evaluate(frqBlkIds=[2, 3, 4, 5], autoKeys=[], returnSignals=True)


    # ------------------------- Output the results -----------------------------

    intns = meta['ampl'][0]           # Array of component intensities
    # Compute mole fractions
    total = sum([np.asscalar(intn) for name, intn in zip(DDD.repRootNames, intns) if name not in ['Water', 'Chloroform']])
    mfracs = [np.asscalar(intn)/total if name not in ['Water', 'Chloroform'] else None for name, intn in zip(DDD.repRootNames, intns)]
    # Compute absolute concentrations
    Mw = {'Fructose':180.156, 'Sucrose':342.30, 'Glucose':180.156, 'Citric acid':192.123, 'Malic acid':134.087, 'Water':18.01528, 'Ethanol':46.069}
    I_H2O = np.asscalar(intns[DDD.repRootNames.index('Water')])
    aconcs = [np.asscalar(intn)*Mw[name]*1000/(I_H2O*Mw['Water']) for name, intn in zip(DDD.repRootNames, intns)]
    # Combine all results
    data = {'names': DDD.repRootNames,
            'intns': intns,
            'aconcs': aconcs,
            'mfracs': mfracs,
            'image': get_figure(DDD, residual=showResidual) if showFigure else None,
            'path': os.path.abspath(dirName),
            'acqu': acquScore,
            'timeSaved': pars['timeSaved'],
            'score': DDD.goodness_of_fit(frqBlkIds=[2])}

    return wsp, data

def main(cmdline=None):
    """Example main function.

    If cmdline is none, parser.parse_args will look at
    sys.argv[1:] by default

    However if import this module in python call this main function
    like this:

    main(["-n", "3", "asdf", "jkl"])

    in addition to running it from the shell.
    """
    parser = make_parser()

    opts, args = parser.parse_args(cmdline)

    # if opts.error is not None:
    #     return opts.error
    # elif opts.bad_option:
    #     # you can call parser.error, which will show an error message
    #     # displays the help, and then exits the program
    #     parser.error("you called a bad option")
    # elif opts.make_template:
    #     pass
    #     return 0
    #
    # # # args is now just a list, of everything that wasn't an
    # # # "option". AKA everything that started with - or --
    # for i in range(len(args)):
    #     print("arg {:d}: {:s}".format(i, args[i]))
    # # print("the number is:", opts.number)

    # -------------------- Execute the script ----------------------
    # dirName =     # 'C:\\Users\\yma80\\Data\\UWA_Sugars\\Raw Data\\1\\1Pulse-H (1 0% - 1)\\1\\'

    wsp, data = script_sugars(opts.directory, opts.residual)

    report(data)

    # ------------------------- Save the workspace -----------------------------
    if opts.save:
        fileName = os.path.join(CALLED_PATH, "workspace_{:s}.wsp".format(time.strftime('%d%m%Y_%H%M%S')))
        save_workspace(fileName, wsp)

    print('\nDONE!')

    # Wait for an input
    # print( CALLED_PATH)
    # print( SCRIPT_PATH )
#    input()

    return 0

def get_figure(DDD, residual=False):
    # Plot the signals and save the figure to a temporary fill
    xlims = (5.5, 2.5) if True else 0     # (DDD.freqBlocks[1].max, DDD.freqBlocks[1].min)

    if residual:
        fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True, sharey=True)
        f, yFph, xF, _ = DDD.plot(ax_main=ax[0], ax_residual=ax[1], showRanges=None, showComponents=True, returnSignals=True)

        # Scale the displayed range to the sugars region
        ymin, ymax = min(0, yFph[(3.0 < f) & (f < 4.2)].min()), yFph[(3.0 < f) & (f < 4.2)].max()
        ylims = (ymin-0.05*(ymax-ymin), ymax+0.05*(ymax-ymin))
        ax[0].set_xlim(*xlims)
        ax[0].set_ylim(*ylims)
    else:
        fig = plt.figure(figsize=(13, 4))
        f, yFph, xF, _ = DDD.plot(ax_main=fig.gca(), showRanges=None, showComponents=True, returnSignals=True)

        # Scale the displayed range to the sugars region
        ymin, ymax = min(0, yFph[(3.0 < f) & (f < 4.2)].min()), yFph[(3.0 < f) & (f < 4.2)].max()
        ylims = (ymin-0.05*(ymax-ymin), ymax+0.05*(ymax-ymin))
        fig.gca().set_xlim(*xlims)
        fig.gca().set_ylim(*ylims)

    # temp_figure_path = os.path.join(SCRIPT_PATH, '_image.eps')
    # fig.savefig(temp_figure_path, bbox_inches='tight')

    imgdata = BytesIO()
    fig.savefig(imgdata, bbox_inches='tight', format='png')
    imgdata.seek(0)  # rewind the data

    return imgdata

def report(data):
    """Generates a pdf report with the results."""

    class OKCircle(Flowable):
        """
        Circle flowable --- draws a circle of a specifci color in a flowable
        """
        #----------------------------------------------------------------------
        def __init__(self, size=4, score=0.5):
            Flowable.__init__(self)
            self.size = size
            if score > 0.9:
                self.color = (0.0, 0.66, 0.35)
            elif score < 0.25:
                self.color = (0.93, 0.195, 0.215)
            else:
                self.color = (0.95, 0.52, 0.20)
        #----------------------------------------------------------------------
        def __repr__(self):
            return "Circle"
        #----------------------------------------------------------------------
        def draw(self):
            """
            draw the circle
            """
            self.canv.setFillColorRGB(*self.color)
            self.canv.circle(0, 5, self.size, stroke=0, fill=1)

    Story=[]

    fileName = os.path.join(CALLED_PATH, "results_{:s}.pdf".format(time.strftime('%d%m%Y_%H%M%S')))
    doc = SimpleDocTemplate(fileName,
                            rightMargin=72,leftMargin=72,
                            topMargin=72,bottomMargin=18)
    stylesheet=getSampleStyleSheet()

    ptext = '<font size=14>{:s}</font>'.format('qNMR Analysis of a Fruit Juice Sample')
    Story.append(Paragraph(ptext, stylesheet['Title']))
    Story.append(Spacer(1, 24))

    tab = []
    tab.append(['Data directory:', '{:s}'.format(data['path'])])
    tab.append(['Acquisition time:', '{:s}'.format(data['timeSaved'])])
    tab.append(['Analysis time:', '{:s}'.format(time.ctime())])
    tab.append(['Acquisition parameters:', OKCircle(score=data['acqu'])])
    tab.append(['Residual after fit:', OKCircle(score=data['score'])])
    tab = Table(tab, style=[('LEFTPADDING', (0,0), (1,4), 0), ('FONTSIZE', (0,0), (1,4), 9)], colWidths=(130, None) )
    Story.append(tab)
    Story.append(Spacer(1, 12))

    # Insert the table
    Story.append(Spacer(1, 18))
    names, intns, mfracs, aconcs = data['names'], data['intns'], data['mfracs'], data['aconcs']
    tab = [['', 'Intensity, a.u.', 'Mole frac., mol/mol', 'Absolute conc., g/L']]
    mfracs = ['{:.3f}'.format(mfrac) if name not in ['Water', 'Chloroform'] else '' for name, mfrac in zip(names, mfracs)]
    aconcs = ['{:.3f}'.format(aconc) if name not in ['Water', 'Chloroform'] else '' for name, aconc in zip(names, aconcs)]
    for name, intn, mfrac, aconc in zip(names, intns, mfracs, aconcs):
        tab.append([name, '{:.3f}'.format(np.asscalar(intn)), mfrac, aconc])
    tab = Table(tab, style=[('LINEBELOW', (0,0), (3,0), 1, (0,0,0))], colWidths=None )
    Story.append(tab)
    Story.append(Spacer(1, 12))

    # Insert the figure
    imgdata = data['image']
    p = ImageFile.Parser()
    p.feed(imgdata.read())
    img = p.close()
    width, height = img.size
    Story.append(Image(imgdata, width=450, height=height*450/width))
    Story.append(Spacer(1, 12))

    # ptext = '<font size=12>Thank you very much and we look forward to serving you.</font>'
    # Story.append(Paragraph(ptext, stylesheet['Normal']))

    doc.build(Story)

    subprocess.Popen(fileName,shell=True)

def make_parser():
    """Construct an option parser"""

    usage = """%prog: args

    Sometimes you might explain the purpose of this program as well.
    """

    parser = OptionParser(usage)

    # add_options takes at least one long option
    # you can optionally include a short option.
    # - are one character (short) options (e.g. -h)
    #
    # -- are long options, the name is also used as the
    #    variable name attached that holds the option

    # parser.add_option("-e", "--error", help="Set error code")

    parser.add_option("-r", "--residual", help="Plot the residual", action="store_true")

    parser.add_option('-d', '--directory', help='The directory with FID data')

    parser.add_option('-s', '--save', help='Save the workspace', action="store_true")

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
                        directory=os.path.join( SCRIPT_PATH, 'data\\juice'),   #Juices\\S13.35_Orange_small\\001-cold'),
                        residual=False,
                        error=None,
                        save=False,
                        number=0)

    return parser


if __name__ == "__main__":
    # this runs when the application is run from the command
    # it grabs sys.argv[1:] which is everything after the program name
    # and passes it to main
    # the return value from main is then used as the argument to
    # sys.exit, which you can test for in the shell.
    # program exit codes are usually 0 for ok, and non-zero for something
    # going wrong.
    sys.exit(main(sys.argv[1:]))
