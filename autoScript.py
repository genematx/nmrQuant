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

import numpy as np
from chemTree import loadTree
from MainLogic import Workspace, next_pow_of_2
from dataio import read_spinsolve
import matplotlib.pyplot as plt
from matplotlib import rcParams
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet

from PIL import ImageFile

rcParams.update({'font.size': 10})
rcParams['ps.fonttype'] = 42               # Needed to be able to save figures in .eps format
# SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))       # Where the exe is located
SCRIPT_PATH = os.path.abspath(os.path.dirname(sys.argv[0]))
CALLED_PATH = os.getcwd()                                          # Where it has been executed from

def script(dirName, residual=False):
    # Load the data
    yT, c0, f0, dt = read_spinsolve(dirName)

    # Define the array of time samples
    nt = yT.shape[0]
    t = np.linspace(start=0, stop=(nt-1)*dt, num=nt).reshape(-1,1)

    # Create a new Workspace and add the data there
    wsp = Workspace()
    ser = wsp.addSeries(c0=c0, f0=f0, t=t, nf=1*next_pow_of_2(len(t)) )
    DDD = ser.addDatum(yT)

    # Define and set the chemical tree
    T = loadTree(os.path.join(SCRIPT_PATH, 'GluSucFruWater.ctr'))
    wsp.setTree(T)

    # Align the spectrum based on the frequency of the water peak
    DDD.alignToSolventPeak(chshTo=4.75)

    # Define regions for optimization
    DDD.addFreqBlock(lims=(2.5, 6.0))


    # --------------------------- Run optimization -----------------------------

    # Optimize global chemical shift
    DDD.optimize(frqBlkIds=[1], parsKeys=[('Mixture', 'chsh', 0), ('Water-SPSY1', 'chshQD', 0)])

    # Optimize the peak width
    DDD.optimize(frqBlkIds=[1], parsKeys=[('Water-SPSY1', 'alphQD', 0)])
    DDD.optimize(frqBlkIds=[1], parsKeys=[('Mixture', 'alph', 0), ('Water-SPSY1', 'alphQD', 0)])

    # Check if the solution is in H2O
    val, meta = DDD.evaluate(frqBlkIds=[1], returnSignals=True)
    ampl = meta['ampl'][0]
    if ampl[0] / sum(ampl) > 0.95:
        DDD.addFreqBlock(lims=(3.1, 4.2))      # Add new frequency block

        DDD.optimize(frqBlkIds=[2], parsKeys=[('Water-SPSY1', 'alphQD', 0)])
        DDD.optimize(frqBlkIds=[2], parsKeys=[('Water-SPSY1', 'chshQD', 0)])
        DDD.optimize(frqBlkIds=[2], parsKeys=[('Sugars', 'chsh', 0)])
        DDD.optimize(frqBlkIds=[2], parsKeys=[('Sugars', 'alph', 0)])

        val, meta = DDD.evaluate(frqBlkIds=[2], returnSignals=True)

    # ------------------------- Output the results -----------------------------

    # val, meta = DDD.evaluate(frqBlkIds=[1], returnSignals=True)
    data = {'names': DDD.repRootNames,
            'intns': meta['ampl'][0],
            'image': get_figure(DDD, residual=residual),
            'path': os.path.abspath(dirName)}
    report(data)

    print('\nDONE!')

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
    script(opts.directory, residual=opts.residual)

    # Wait for an input
    # print( CALLED_PATH)
    # print( SCRIPT_PATH )
#    input()

    return 0

def get_figure(DDD, residual=False):
    # Plot the signals and save the figure to a temporary fill
    if residual:
        fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
        f, yFph, xF, _ = DDD.plot(ax_main=ax[0], ax_residual=ax[1], showRanges=None, showComponents=True, returnSignals=True)

        # Scale the displayed range to the sugars region
        xlims = (DDD.freqBlocks[1].max, DDD.freqBlocks[1].min)
        ymin, ymax = min(0, yFph[(3.0 < f) & (f < 4.2)].min()), yFph[(3.0 < f) & (f < 4.2)].max()
        ylims = (ymin-0.05*(ymax-ymin), ymax+0.05*(ymax-ymin))
        ax[0].set_xlim(*xlims)
        ax[0].set_ylim(*ylims)
    else:
        fig = plt.figure(figsize=(13, 4))
        f, yFph, xF, _ = DDD.plot(ax_main=fig.gca(), showRanges=None, showComponents=True, returnSignals=True)

        # Scale the displayed range to the sugars region
        xlims = (DDD.freqBlocks[1].max, DDD.freqBlocks[1].min)
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

    Story=[]

    doc = SimpleDocTemplate(os.path.join(CALLED_PATH, "results.pdf"),
                            rightMargin=72,leftMargin=72,
                            topMargin=72,bottomMargin=18)
    stylesheet=getSampleStyleSheet()

    formatted_time = time.ctime()
    ptext = '<font size=10>{:s}</font>'.format(formatted_time)
    Story.append(Paragraph(ptext, stylesheet['Normal']))
    Story.append(Spacer(1, 12))

    ptext = '<font size=10>{:s}</font>'.format(data['path'])
    Story.append(Paragraph(ptext, stylesheet['Normal']))
    Story.append(Spacer(1, 12))

    # Insert the figure
    imgdata = data['image']
    p = ImageFile.Parser()
    p.feed(imgdata.read())
    img = p.close()
    width, height = img.size
    Story.append(Image(imgdata, width=450, height=height*450/width))
    Story.append(Spacer(1, 12))

    # Insert the table
    names, intns = data['names'], data['intns']
    tab = [['', 'Intensity, a.u.', 'Mole frac., mol/mol']]
    total = sum([np.asscalar(intn) for name, intn in zip(names, intns) if name not in ['Water', 'Chloroform']])
    mfracs = ['{:.4f}'.format(np.asscalar(intn)/total) if name not in ['Water', 'Chloroform'] else '' for name, intn in zip(names, intns)]
    for name, intn, mfrac in zip(names, intns, mfracs):
        tab.append([name, '{:.4f}'.format(np.asscalar(intn)), mfrac])
    tab = Table(tab, style=[('LINEBELOW', (0,0), (2,0), 1, (0,0,0))] )
    Story.append(tab)
    Story.append(Spacer(1, 12))

    # ptext = '<font size=12>Thank you very much and we look forward to serving you.</font>'
    # Story.append(Paragraph(ptext, stylesheet['Normal']))


    doc.build(Story)

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
                        directory=os.path.join( SCRIPT_PATH, 'data\\1'),
                        residual=False,
                        error=None,
                        make_template=False,
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
