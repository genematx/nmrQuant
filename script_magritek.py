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
import sys
import config

import numpy as np
from chemTree import loadTree
from MainLogic import Workspace
from dataio import read_spinsolve
import matplotlib.pyplot as plt

def script(dirName):
    # Load the data
    yT, c0, f0, dt = read_spinsolve(dirName)

    # Define the array of time samples
    nt = yT.shape[0]
    t = np.linspace(start=0, stop=(nt-1)*dt, num=nt).reshape(-1,1)

    # Create a new Workspace and add the data there
    wsp = Workspace()
    ser = wsp.addSeries(c0=c0, f0=f0, t=t)
    DDD = ser.addDatum(yT)

    # Define and set the chemical tree
    T = loadTree('GluSucFruWater.ctr')
    wsp.setTree(T)

    # Align the spectrum based on the frequency of the water peak
    DDD.alignToSolventPeak(chshTo=4.75)

    # Define regions for optimization
    DDD.addFreqBlock(lims=(2.5, 6.0))

    # ------------------------------- Run optimization ------------------------------

    # Optimize global chemical shift
    DDD.optimize(parsKeys=[('Mixture', 'chsh', 0), ('Water-SPSY1', 'chshQD', 0)])

    # Optimize the peak width
    DDD.optimize(parsKeys=[('Water-SPSY1', 'alphQD', 0)])
    DDD.optimize(parsKeys=[('Mixture', 'alph', 0), ('Water-SPSY1', 'alphQD', 0)])

    # Plot the signals and save to pdf
    val, meta = DDD.evaluate(frqBlkIds=[1], returnSignals=True)

    fig = plt.figure(figsize=(13, 4))
    f, yFph, xF, _ = DDD.plot(ax_main=fig.gca(), showRanges=None, showComponents=True, returnSignals=True)

    # Scale the displayed range to the sugars region
    xlims = (DDD.freqBlocks[1].max, DDD.freqBlocks[1].min)
    ymin, ymax = yFph[(3.0 < f) & (f < 4.2)].min(), yFph[(3.0 < f) & (f < 4.2)].max()
    ylims = (ymin-0.05*(ymax-ymin), ymax+0.05*(ymax-ymin))
    fig.gca().set_xlim(*xlims)
    fig.gca().set_ylim(*ylims)

    fig.savefig("image.pdf")

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

    if opts.error is not None:
        return opts.error
    elif opts.bad_option:
        # you can call parser.error, which will show an error message
        # displays the help, and then exits the program
        parser.error("you called a bad option")
    elif opts.make_template:
        pass
        return 0

    # # args is now just a list, of everything that wasn't an
    # # "option". AKA everything that started with - or --
    # for i in range(len(args)):
    #     print("arg {:d}: {:s}".format(i, args[i]))
    #
    # print("the number is:", opts.number)

    # Execute the script
    dirName = 'C:\\Users\\yma80\\Data\\UWA_Sugars\\Raw Data\\1\\1Pulse-H (1 0% - 1)\\1\\'
    script(dirName)



    input()

    return 0

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

    parser.add_option("-e", "--error", help="set error code")

    parser.add_option('-d', '--directory', help='the directory with FID data')

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

    parser.set_defaults(bad_option=False,
                        createRDSIndex=False,
                        directory='data\\1',
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
