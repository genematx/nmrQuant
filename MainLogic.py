import numpy as np
from scipy.linalg import block_diag
import dill
import copy
import emcee
from chemTree import *
from leastsquares import *
import sys
import tabulate
from math import ceil
import pywt
import xlsxwriter

# Functions for generating FIDs and optimization
from scipy import optimize
from scipy.optimize import minimize
import scipy as sp
import scipy.sparse
import scipy.linalg
import scipy.signal
from collections import OrderedDict, MutableMapping
import os, time
import nmrglue

# Functions needed only for Matlab
from operator import getitem

# Classes and functions for the basinhopping algorithm
class RandomDisplacementBounds(object):
    """random displacement with bounds"""
    def __init__(self, xmin, xmax, stepsize=0.5):
        self.xmin = xmin
        self.xmax = xmax
        self.stepsize = stepsize

    def __call__(self, x):
        """take a random step but ensure the new position is within the bounds"""
        while True:
            # this could be done in a much more clever way, but it will work for example purposes
            xnew = x + np.random.uniform(-self.stepsize, self.stepsize, np.shape(x))
            if np.all(xnew < self.xmax) and np.all(xnew > self.xmin):
                break
        return xnew

class MyTakeStep(object):
    # Using a custom step taking routine
    def __init__(self, stepsize=0.5):
        self.stepsize = stepsize
    def __call__(self, x):
        s = self.stepsize
        x = np.random.uniform(-1, 1, x.shape)
        return x

##### ------------ Main classes for the general program logic ------------ #####

minmaxTuple = namedtuple('minmaxTuple', 'min, max')
minmaxTuple.__new__.__defaults__ = (-np.inf, np.inf)
minmaxTuple.imin = lambda self, f : np.searchsorted(f.ravel(), self.min)
minmaxTuple.imax = lambda self, f : np.searchsorted(f.ravel(), self.max)

class freqSpec():
    """A class to store the specifications of frequency blocks along with their baselines"""

    def __init__(self, min=-np.inf, max=np.inf, bslnOrder=(None, None)):
        self.min = min
        self.max = max
        self.bslnOrder = bslnOrder
        self._indxFreq = None
        self._bF = None          # An array of baselines
        self._fhash = None       # Hash value for the previously computed f

    def repr(self):
        return '{:.2f} ... {:.2f}'.format(self.min, self.max) if not (self.min == -float('inf') and self.max == float('inf')) else 'Entire range'

    def bline(self, nf, numberField='Cm'):
        """Creates a set of base polynomial functions to store the baseline of length nf."""
        if self._bF is None or self._bF.shape[1] != nf:
            # Define baseline in the frequency domain
            bFr = [np.linspace(-1,1,nf).reshape(-1,1)**i for i in range(self.bslnOrder[0]+1)] if (self.bslnOrder[0] is not None) and (numberField in ['Re', 'Cm']) else []
            bFi = [1j*np.linspace(-1,1,nf).reshape(-1,1)**i for i in range(self.bslnOrder[1]+1)] if (self.bslnOrder[1] is not None) and (numberField in ['Im', 'Cm']) else []
            self._bF = np.hstack(bFr+bFi) if len(bFr)+len(bFi) > 0 else None
        return self._bF

    def imin(self, f):
        """Starting index of the range in the array f."""
        return np.searchsorted(f.ravel(), self.min)

    def imax(self, f):
        """Last index of the range in the array f."""
        return np.searchsorted(f.ravel(), self.max)

    def indxFreq(self, f, nw2=0):
        """Returns the indices of array f that fall into the range defined by the block. Optionally can include padding with nw samples on both ends of the range."""
        # Compute the hash value for the frequency array to determine wethwer update is necessary
        newHash = arrhash(f) + nw2
        if self._indxFreq is None or self._fhash != newHash:
            self._indxFreq = np.arange(self.imin(f) - nw2, self.imax(f) + nw2) % len(f)
            self._fhash = newHash
        return self._indxFreq

    def update(self, lims=None, bslnOrder=None):
        """Updates the parameters of a frequency block."""
        if lims is not None:
            self.min = min(lims)
            self.max = max(lims)
        if bslnOrder is not None:
            self.bslnOrder = bslnOrder

        self._bF = None                # Remove all precomputed baselines
        self._indxFreq = None

    def pars(self):
        """Returns a named tuple of the main blok parameters."""
        pars = namedtuple('pars', 'min, max, bslnOrder')
        return(pars(self.min, self.max, self.bslnOrder))

class Step():

    def __init__(self, frqBlkIds = None, parsKeys = None, autoKeys=None, repRootNames=None, fitCustomLshape = False):
        self.frqBlkIds = frqBlkIds if frqBlkIds is not None else set()
        self.parsKeys = set(parsKeys) if parsKeys is not None else set()      # Parameters to fit on this step
        self.autoKeys = set(autoKeys) if autoKeys is not None else set( [('.', 'sigma2', 0)] )      # ('.', 'theta', 0),                 # Potentially autofittable parameters that will be excluded from fitting on this step (may contain, theta, gamma, sigma2, and any amplitudes)
        if repRootNames is not None:
            self.autoKeys.update([(name, 'ampl', 0) for name in repRootNames])
        self.fitCustomLshape = fitCustomLshape

class Workspace():

    def __init__(self, HCmode='1H'):
        self.reset(HCmode=HCmode)

    def __repr__(self):
        """A function to display the workspace."""
        output = "Workspace" + '\n'
        if len(self.series) > 0:
            output += "|------"
            for SSS in self.series:
                output += SSS.log(depth=1)

        return output

    def selfID(self):
        return (None, None)

    def reset(self, HCmode='1H', lshapeOrder=2):
        # Resets the entire workspace
        self.series = []
        self.HCmode = HCmode
        self.repRootNames = []
        self.lshapeOrder = 0

        # Define default priors. All Series and Datums will be affected by these priors if not explicitely overwritten.
        self.parsSpecDict = {('.', 'mult', 0): parsSpec(dval=1.), \
                            ('.', 'tau', 0): parsSpec(-1e-04, 1e-04), \
                            ('.', 'sigma2', 0): parsSpec(distr='Inverse-Gamma', p1=2., p2=10., dval=0.), \
                            ('.', 'theta', 0): parsSpec(distr='Uniform', min=-np.pi, max=np.pi), \
                            ('.', 'gamma', 0): parsSpec(distr='Uniform', min=0.0, max=1.0-1e-09, dval=0.0)}
        if lshapeOrder is not None: self.set_lshapeOrder(lshapeOrder)
        self.setTree(chemNode('Mixture', chsh = [parsSpec(min=-0.1, max=0.1)], \
                              alph=[parsSpec(min=-5., max=25., dval=2.0)]))

    def getPrior(self, key):
        """Returns the specification of a parameter in the current tree or tau. Start by looking for the specification in the current datum structure, then proceed to the series level andthe tree if the parameter is not found."""
        # TODO: Make it nicer...
        try:
            return self.parsSpecDict[key]
        except KeyError:
            if key[1] == 'ampl':
                # Default prior for amplitudes
                return parsSpec(distr='Gaussian', p1=0, p2=np.inf, dval=0.)
            elif key[1] == 'phase':
                # Default prior for amplitudes
                return parsSpec(distr='Uniform', min=-np.pi, max=np.pi, dval=0.0)
            else:
                return getattr(self.T[key[0]], key[1])[key[2]]

    def setGlobalPrior(self, key, val, reset=True):
        """Sets the specification of a parameter in the current tree or tau; val is of parsSpec type."""
        if key[0] == '.':
            self.parsSpecDict[key] = val
        else:
            # Set a prior in the tree
            getattr(self.T[key[0]], key[1])[key[2]] = getattr(self.T[key[0]], key[1])[key[2]]._replace(min=val.min, max=val.max, p1=val.p1, p2=val.p2, distr=val.distr, dval=val.dval)
        # Remove this parSpec from all datasets and series
        if reset:
            for ser in self.series:
                try:
                    ser.parsSpecDict.pop(key)
                except KeyError:
                    pass
                for dat in ser.data:
                    try:
                        dat.parsSpecDict.pop(key)
                    except KeyError:
                        pass

    def setHCmode(self, HCmode):

        if HCmode != self.HCmode:

            self.HCmode = HCmode

        if self.T is not None:
            # add QD nodes to the tree based on the mode of the current workspace
            for node in self.T.items():
                if isinstance(node, chemNodeDB) and node.HCmode != self.HCmode: node.dendrolize(self.HCmode)

            self._updateParameters()    # Also sets self.repRootNames

    def set_lshapeOrder(self, newOrder):
        """Sets a new lineshape correction order."""
        if self.lshapeOrder < newOrder:
            # Need to add new parameters
            for i in range(self.lshapeOrder, newOrder):
                self.parsSpecDict[('.', 'lshapeR', i)] = parsSpec(-2.5, 2.5, dval=0.0)
                self.parsSpecDict[('.', 'lshapeI', i)] = parsSpec(-2.5, 2.5, dval=0.0)
                for parsH in [dat.crntParsH for ser in self.series for dat in ser.data]:
                    try:
                        parsH['.']['lshapeR'].append(0.0)
                        parsH['.']['lshapeI'].append(0.0)
                    except KeyError:
                        parsH['.']['lshapeR'] = [0.0]
                        parsH['.']['lshapeI'] = [0.0]

        elif self.lshapeOrder > newOrder:
            # Need to remove some parameters
            for i in range(newOrder, self.lshapeOrder):
                for psdict in [self.parsSpecDict]+[ser.parsSpecDict for ser in self.series]+[dat.parsSpecDict for ser in self.series for dat in ser.data]:
                    try: psdict.pop(('.', 'lshapeR', i))
                    except KeyError: pass
                    try: psdict.pop(('.', 'lshapeI', i))
                    except KeyError: pass

                for parsH in [dat.crntParsH for ser in self.series for dat in ser.data]:
                    parsH['.']['lshapeR'].pop(i)
                    parsH['.']['lshapeI'].pop(i)

        self.lshapeOrder = newOrder

    def setTree(self, T, priors = None):
        # Load the tree
        self.T = T
        #self.T.setTreeBook()

        # add QD nodes to the tree based on the mode of the current workspace
        for node in T.items():
            #if isinstance(node, chemNodeDB): node.dendrolize(self.HCmode)
            if isinstance(node, chemNodeDB) and node.HCmode != self.HCmode: node.dendrolize(self.HCmode)

        self._updateParameters()

        if priors is not None:
            self.parsSpecDict.update(priors)

    def getTree(self, node=None):
        """Creates a copy of the tree rooted at node with name node."""

        # Extract the subtree starting with the node key and set its parent to None
        T = copy.deepcopy(self.T[node]) if node is not None else copy.deepcopy(self.T)
        T.makeRoot()

        # Reset the nodes (to remove any curently stored signals)
        for node in T.items():
            node.reset()

        return T

    def allParsKeys(self, node_name=None, parsKind=None):
        """Returns all parameter keys for a (sub)tree starting from a specific node."""
        if node_name is None:
            node = self.T.findRoot()
        else: node = self.T[node_name]

        if parsKind is None:
            parsKind = ['chsh', 'chshQD', 'alph', 'alphQD', 'jcplQD', '.']

        parsKeys = [key for key in flatten(defaultTreePars(node, startFromRoot=False)).keys() \
                    if key[1] in parsKind]      # List of all parameter keys that affect the subtree

        if '.' in parsKind:
            parsKeys.extend([('.', 'theta', 0), ('.', 'tau', 0), ('.', 'sigma2', 0), ('.', 'gamma', 0)] + \
                            [('.', 'lshapeR', i) for i in range(self.lshapeOrder)] + \
                            [('.', 'lshapeI', i) for i in range(self.lshapeOrder)])

        return parsKeys

    def _updateParameters(self):
        """Updates the existing dictionaries of parameters after the tree has changed (e.g. when adding/removing nodes or setting new root nodes). Updates the structure to match with the new default parameters but keeps the old values."""
        oldRoots = self.repRootNames

        newParsH = defaultTreePars(self.T, lshapeOrder=self.lshapeOrder)
        newRoots = [node.name for node in self.T.repRoots()]

        # Update parameter dictionaries
        for oldParsH in [dat.crntParsH for ser in self.series for dat in ser.data]:
            # Find which nodes parameters have changed
            addKeys = set(newParsH.keys()) - set(oldParsH.keys())    # Added keys
            remKeys = set(oldParsH.keys()) - set(newParsH.keys())    # Removed keys

            # Add new parameters
            for key in addKeys:
                oldParsH[key] = copy.deepcopy(newParsH[key])

            # Remove parameters
            for key in remKeys:
                oldParsH.pop(key)

        # Update parsKeys and autoKeys for parameters in each step
        def keyExists(key):
            """Determines if a parameter with a certain key exists in the newParsH dictionary."""
            try:
                newParsH[key[0]][key[1]][key[2]]
                return True
            except KeyError: return False

        addRoots = set(newRoots) - set(oldRoots)    # Added keys
        for step in [step for ser in self.series for step in ser.steps]:
            step.parsKeys = set(filter(keyExists, step.parsKeys))         # Remove keys that no longer exist
            step.autoKeys = set(filter(keyExists, step.autoKeys))         # Remove keys that no longer exist
            for root in addRoots:
                step.autoKeys.add((root, 'ampl', 0))

        # Set the new reported roots
        self.repRootNames = newRoots

    def addTreeNode(self, prnt, X, priors = None):
        """Grafts a subtree X to the parent node of self.T and updates the parameters and priors."""

        # Dendrolize nodes if necessary
        for node in X.items():
            #print(self.HCmode, node.HCmode)
            if type(node) is chemNodeDB and node.HCmode != self.HCmode: node.dendrolize(self.HCmode)

        ## Update the tree book
        #X.setTreeBook()

        # Insert the node
        #try:
        self.T[prnt].addChild(X)
        #except:
        #    print("Can not add the node.")
        #    return 0

        # Update the parameters of all series/datasets
        self._updateParameters()

        return True

    def delTreeNode(self, node):
        """Removes a node (and the entire subtree) from the tree."""
        try:
            self.T[node].cut()
        except:
            print("Can not remove the node.")
            return False

        # Update the parameters of all series/datasets
        self._updateParameters()

        return True

    def renameTreeNode(self, oldName, newName):
        # Rename all children, if it's a spin system node
        if isinstance(self.T[oldName], chemNodeQD):
            for chld in self.T[oldName].children():
                suffix = chld.name[chld.name.rfind('-'):]
                self.renameTreeNode(chld.name, newName + suffix)

        try:
            self.T[oldName].rename(newName)
        except:
            print("Could not rename the node {} to {}.".format(oldName, newName))
            return 0

        # Update the parameters of all series/datasets
        for oldParsH in [dat.crntParsH for ser in self.series for dat in ser.data]:
            oldParsH[newName] = oldParsH[oldName]
            oldParsH.pop(oldName)

        # Update parameter specification dictionaries
        for psdict in [self.parsSpecDict]+[ser.parsSpecDict for ser in self.series]+[dat.parsSpecDict for ser in self.series for dat in ser.data]:
            for key, val in psdict.items():
                if key[0] == oldName:
                    psdict[(newName, key[1], key[2])] = val
                    psdict.pop(key)

        # Update parameter names in the specification of steps
        for ser in self.series:
            for step in ser.steps:
                for key in step.parsKeys:
                    if key[0] == oldName:
                        step.parsKeys.remove(key)
                        step.parsKeys.add((newName, key[1], key[2]))
                for key in step.autoKeys:
                    if key[0] == oldName:
                        step.autoKeys.remove(key)
                        step.autoKeys.add((newName, key[1], key[2]))

        # Update the list of reported roots
        try:
            self.repRootNames[self.repRootNames.index(oldName)] = newName
        except ValueError: pass

        # TODO: Update the list of Ignored roots

        return True

    def toggleRepRoot(self, key):
        """Toggles the reportability of a certain root node and updates the parameters accordingly."""
        self.T[key].toggleReported()
        self._updateParameters()

    def setRepRoot(self, key, flag=True):
        """Sets the reportability of a certain root node and updates the parameters accordingly."""
        self.T[key].setReported(flag)
        self._updateParameters()

    def isXclRootName(self, name):
        """Checks whether the node with the a certain name is ignored in the analysis."""
        flags = [DDD.isXclRootName(name) for SSS in self.series for DDD in SSS.data]
        return len(flags)>0 and all(flags)

    def addSeries(self, **kwargs):

        SSS = Series(parent=self, **kwargs)
        self.series.append(SSS)

        return SSS

    #@profile
    def _optimize(self, costFuncOpti, bounds, initVals, nhop=None, respectBounds=True, verbose=True):
        """Core optimization routine; used by all Series and Datums in this Workspace"""
        eps_range = np.mean([np.abs(bnd[1]-bnd[0]) for bnd in bounds])     # Find the range of optomiztion (needed to set the step size for Jacobian)
        if len(initVals) > 2 or (nhop is not None and nhop > 0):
            res = optimize.basinhopping(costFuncOpti, initVals, \
                  niter = nhop if nhop is not None else config.OPTIM_maxBasinhoppingSteps, \
                  niter_success = nhop if nhop is not None else config.OPTIM_niterSuccess, T = 10, disp = verbose, \
                  minimizer_kwargs=dict(method=config.OPTIM_method, bounds=bounds, tol=1e-12) )     #, \
            #      #take_step=MyTakeStep())
        else:
            res = optimize.minimize(costFuncOpti, x0=initVals, bounds=bounds, method=config.OPTIM_method, \
                  options={'eps':eps_range*1e-05, 'ftol':1e-12})       # Step-size for computing the Jacobian
            #print(res['message'])

        # Discard the found parameters if any of them lies close to its range and run the optimization again
        if respectBounds:
            closeToBounds = np.isclose(bounds, res.x.reshape(-1,1)).any(axis=1)
            if closeToBounds.all():
                print('Optimization failed. All found values are close to bounds.')
                return None
            elif closeToBounds.any():
                # Sustitute the initial values for the bad parameter in the optimization
                P = np.delete(np.eye(len(initVals)), np.where(closeToBounds)[0], axis=1)
                print('Start inner optimization loop with {} parameters'.format( len(np.where(closeToBounds)[0]) ) )
                res = self._optimize(lambda x : costFuncOpti( P.dot(x).ravel()+np.where(closeToBounds, initVals, 0.0)), \
                                    [bnd for bnd, flg in zip(bounds, ~ closeToBounds) if flg], \
                                    [val for val, flg in zip(initVals, ~ closeToBounds) if flg], nhop, verbose=verbose )
                if res is not None:
                    res.x = P.dot(res.x.reshape(-1,1)).ravel()+np.where(closeToBounds, initVals, 0.0)

        return res

    def _sample(self, costFuncSmpl, bounds, initVals, nwalkers=None, nsteps=None, verbose=True):
        """Core sampling routine; used by all Series and Datums in this Workspace"""
        def progressBar(value, endvalue, text='', bar_length=20):
            percent = float(value) / endvalue
            arrow = '-' * int(round(percent * bar_length)-1) + '>'
            spaces = ' ' * (bar_length - len(arrow))

            sys.stdout.write("\r{0}[{1}] {2}% ".format(text, arrow + spaces, int(ceil(percent * 100))))
            sys.stdout.flush()

        # Set parameters of the sampling algorithm
        ndim = len(initVals)          # Number of dimensions (number of parameters to sample over)
        if nwalkers is None: nwalkers = 4*ndim             # Number of walkers
        if nsteps is None: nsteps = min(int(1000/nwalkers), 250)      # Number of steps
        nsteps = max(1, nsteps)       # Make sure at least one step is taken
        bounds = np.array(bounds)
        delta = np.abs(bounds[:,1] - bounds[:,0])
        p0 = 0.1*delta*(np.random.rand(nwalkers, ndim)-1/2) + np.array(initVals).reshape(1,-1)    # Starting points
        p0 = np.minimum(np.maximum(p0, bounds[:, 0].reshape(1,-1)), bounds[:, 1].reshape(1,-1))

        """# Run burn-in iterations (separately for each dimension)
        print('Burning in...')
        for i in range(ndim):
            arg = np.array(initVals)
            def costFuncSmplReduced(x):
                arg[i] = x
                return costFuncSmpl(arg)

            sampler = emcee.EnsembleSampler(nwalkers, 1, costFuncSmplReduced, a=2.0)
            pos0 = 0.1*np.random.rand(nwalkers, 1) + initVals[i]
            pos0 = np.random.rand(nwalkers, 1)*(bounds[i][1]-bounds[i][0]) + bounds[i][0]
            pos, _, _, _ = sampler.run_mcmc( pos0=pos0, N=max(1, int(nsteps/10)) )    # burn-in
            p0[:,i] = pos.ravel()"""

        # Define the sampler
        sampler = emcee.EnsembleSampler(nwalkers, ndim, costFuncSmpl, a=2.0)

        # Run burn-in iterations (jointly for all dimensions)
        print("Burning in...")
        pos, _, _, _ = sampler.run_mcmc(p0, N=max(1, int(nsteps/10)) )    # burn-in
        sampler.reset()

        # Final sampling starting from the parameter values found during burn-in
        print("Sampling...")
        for i, result in enumerate(sampler.sample( pos, iterations=nsteps )):
            progressBar(i, nsteps)
        sys.stdout.write("\n")

        #sampler.run_mcmc(pos, nsteps)
        #print(sampler.blobs)

        if verbose:
            print("Mean acceptance ratio: {0:.3f}"
                  .format(np.mean(sampler.acceptance_fraction)))

        return sampler

    def saveResults(self, filename='results.xlsx', parsKeys=None):
        """Saves the reults to an excel file."""

        # Create a workbook and add a worksheet.
        workbook = xlsxwriter.Workbook(filename)
        worksheet = workbook.add_worksheet()

        # Which cell to start writing the data from. Rows and columns are zero indexed.
        datarow = 0
        datacol = 0

        # Write the header
        fmt_center = workbook.add_format({'align': 'center', 'valign': 'vcenter'})
        fmt_cenrot = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'rotation': 90})
        for i, (text, col_width) in enumerate(zip(['','ID', 'Series Name', 'Data Name'], [3, 5, 3, 15])):
            worksheet.merge_range(0, i, 2, i, text, fmt_center)
            worksheet.set_column(i, i, col_width)
        worksheet.merge_range(0, 4, 0, 4+2*len(self.repRootNames)-1, 'Absolute intensities of mixture components', fmt_center)

        col = 4
        for name in self.repRootNames:
            worksheet.merge_range(1, col, 1, col+1, name, fmt_center)
            worksheet.write_row(2, col, ['Intensity, a.u.', 'Variance'])
            col += 2


        # Write the Series names in merged rows
        row_start = 3
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
        row = 3
        for ser in self.series:
            for dat in ser.data:
                # Write the Datum ID and Name
                worksheet.write(row, 0, row-2)
                worksheet.write(row, 1, repr(dat.selfID()))
                worksheet.write(row, 3, dat.name)

                # Write the results
                col = 4
                for name in self.repRootNames:
                    key=(name, 'ampl', 0)
                    if not dat.isXclRootName(name):
                        ampl = dat.getCrntVal(key)
                        var = dat.smplDistF[key].var if key in dat.smplDistF.keys() else 0.0
                    else:
                        ampl, var = [0.0, 0.0]
                    worksheet.write_row(row, col, [ampl, var])
                    col += 2
                row += 1

        # # Iterate over the data and write it out row by row.
        # for item, cost in (expenses):
        #     worksheet.write(row, col,     item)
        #     worksheet.write(row, col + 1, cost)
        #     row += 1
        #
        # # Write a total using a formula.
        # worksheet.write(row, 0, 'Total')
        # worksheet.write(row, 1, '=SUM(B1:B4)')

        workbook.close()

    def pack(self):
        """Packs the workspace into a dictionary to be stored.
        !!!!!!!
        WARNING: The results of the function may need to be deepcopied if they are needed unchanged later.
        !!!!!!!
        """
        T = copy.deepcopy(self.T)
        if T is not None:
            for node in T.items(): node.reset()
        result = {'HCmode' : self.HCmode,
                'T' : T,
                'parsSpecDict' : self.parsSpecDict,
                'lshapeOrder' : self.lshapeOrder,
                'series' : []}
        for ser in self.series:
            result['series'].append({'name' : ser.name,
                                    'c0' : ser.c0,
                                    'f0' : ser.f0,
                                    'nt' : len(ser.t),
                                    'dt' : ser.t[1]-ser.t[0] if len(ser.t) > 1 else 0.0,
                                    'steps' : ser.steps,
                                    'freqBlocks' : [blk.pars() for blk in ser.freqBlocks],
                                    'parsSpecDict' : ser.parsSpecDict,
                                    'crntMetaF' : ser.crntMetaF,
                                    'smplDistF' : ser.smplDistF,
                                    'meta_function' : ser._meta,
                                    'jointPrior' : ser._joint,
                                    'apod' : ser.apod,
                                    'zff' : ser.zff,
                                    'data' : []})
            for dat in ser.data:
                result['series'][-1]['data'].append({
                                                    'name' : dat.name,
                                                    'yT' : dat.yT,
                                                    'arrVal' : dat.arrVal,
                                                    'crntParsH' : dat.crntParsH,
                                                    'smplDistF' : dat.smplDistF,
                                                    'parsSpecDict' : dat.parsSpecDict,
                                                    'mdldPeaks' : dat.mdldPeaks,
                                                    'pckdPeaks' : dat.pckdPeaks,
                                                    'sF' : dat.sF,
                                                    'sT' : dat.sT,
                                                    'jointPrior' : dat._joint,
                                                    'refChshKey' : dat.refChshKey,
                                                    'flagAdapFreq' : dat._flagAdapFreq,
                                                    'xclRootNames' : dat.xclRootNames
                                                    })

        return result

    def unpack(self, packed):
        """Unpackes a saved workspace from the dictionary."""
        try: lshapeOrder = packed['lshapeOrder']    # Needed for older packed Workspaces that did not include lineshape correction parameters
        except KeyError: lshapeOrder = None

        self.reset(packed['HCmode'], lshapeOrder)
        T = packed['T']
        if T is not None:
            T.setTreeBook()
        # Make sure that each node in the tree has an ampl and a phase attributes
        for node in T.items():
            if not hasattr(node, 'ampl'): node.ampl = [parsSpec(min=0., max=np.inf, distr='Gaussian', p1=0.0, p2=np.inf, dval=1.0)]
            if not hasattr(node, 'phase'): node.phase = [parsSpec(distr='Uniform', min=-np.pi, max=np.pi, dval=0.0)]
        #self.repRootNames = [node.name for node in self.T.repRoots()]    # Need to set the repRootNames before to refer to them later in the _updateParameters function
        self.setTree(T, packed['parsSpecDict'])
        for ser in packed['series']:
            ser['t'] = np.linspace(0, ser['dt']*(ser['nt']-1), ser['nt']).reshape(-1, 1)
            newSeries = self.addSeries(priors=ser['parsSpecDict'], **ser)
            # newSeries = self.addSeries(name=ser['name'], c0=ser['c0'], f0=ser['f0'],
            #                         t=t, apod=ser['apod'], zff=ser['zff'] if 'zff' in ser.keys() else 0,
            #                         priors=ser['parsSpecDict'])
            for blk in ser['freqBlocks']:
                if blk.min == -np.inf and blk.max == np.inf:
                    newSeries.altFreqBlock(indx=0, bslnOrder=blk.bslnOrder)      # Set the order of baseline for the All frequencies block
                else: newSeries.addFreqBlock(lims=(blk.min, blk.max), bslnOrder=blk.bslnOrder)
            #newSeries.steps = ser['steps']
            newSeries.steps = []
            try:
                newSeries.crntMetaF = ser['crntMetaF']
            except KeyError: pass
            if 'meta_function' in ser.keys():
                newSeries.setMetaFunction(ser['meta_function'])
            if 'smplDistF' in ser.keys():
                newSeries.smplDistF.update(ser['smplDistF'])
            if 'jointPrior' in ser.keys():
                newSeries.setJointPrior(ser['jointPrior'])
            for stp in ser['steps']:
                newStep = Step()
                newStep.frqBlkIds, newStep.parsKeys = stp.frqBlkIds, stp.parsKeys
                try: newStep.autoKeys = stp.autoKeys
                except AttributeError: pass
                try: newStep.fitCustomLshape = stp.fitCustomLshape
                except AttributeError: pass
                newSeries.steps.append(newStep)
            for dat in ser['data']:
                newDatum = newSeries.addDatum(priors=dat['parsSpecDict'], **dat)
                # newDatum = newSeries.addDatum(dat['yT'], name=dat['name'], arrVal=dat['arrVal'],   #/np.linalg.norm(dat['yT'])
                #                     crntParsH=dat['crntParsH'], priors=dat['parsSpecDict'])
                newDatum.mdldPeaks = dat['mdldPeaks']
                newDatum.pckdPeaks = dat['pckdPeaks']
                newDatum.sF = dat['sF'] if 'sF' in dat.keys() else None
                newDatum.sT = dat['sT'] if 'sT' in dat.keys() else None
                if 'smplDistF' in dat.keys():
                    newDatum.smplDistF.update(dat['smplDistF'])
                if 'refChshKey' in dat.keys():
                    newDatum.setReferenceChshKey(dat['refChshKey'])
                if 'jointPrior' in dat.keys():
                    newDatum.setJointPrior(dat['jointPrior'])

        if lshapeOrder is None: self.set_lshapeOrder(2)

        # Add missing parameters (needed for back-compatibility)
        for parsH in [dat.crntParsH for ser in self.series for dat in ser.data]:
            if 'gamma' not in parsH['.'].keys():
                parsH['.']['gamma'] = [0.0]

class Series():
    """Class for the data series (e.g. in reaction monitoring)."""

    def __init__(self, parent, name = None, c0=None, f0=None, t=None, zff=0, apod=0, priors=None, **kwargs):
        self.parent = parent     # The workspace that contains the tree
        self.name = name if name is not None else 'Series ' + str(len(self.parent.series)+1)
        self.c0 = c0
        self.f0 = f0
        self.t = t.reshape(-1,1) if t is not None else np.array([])
        self.f = np.array([])
        self.data = []               # A list of Datum structures
        self.steps = [Step(repRootNames=self.repRootNames)]               # Fitting steps; each entry is a set of parsKeys tuples and set of frqBlkIds
        self.freqBlocks = []         # a list of optimization frequency ranges
        self.apod = apod
        self.zff = zff               # zero-filling factor (exponent of 2)
        self.wT = 1
        self.crntMetaF = dict()       # A dictionary of current values of meta-parameters
        self.smplDistF = dict()
        self._meta = None             # A function that chnages the Series parameters controlled by the meta-parameters
        self._joint = None
        self.fullReset(zff, apod, priors)           # Setup the frequency range and compute the spectra

    def __getattr__(self, attr):
        """Called with the dot notation for attributes not found in the class (e.g. parameters shared between many spectra in the series, c0, f0, etc.)."""
        return getattr(self.parent, attr)

    def log(self, depth=0):
        """A function to display the workspace."""
        s = self.selfID()[0]       # ID of this Seies in the Workspace
        output = '\t'*depth + self.name + '\n'
        if len(self.data) > 0:
            output += "\t"*(0+depth) + "|------<{},{}> ".format(s, 0) + self.data[0].name + '\n'
            for i, DDD in enumerate(self.data[1:]):
                output += '\t'*(0+depth) + "|------<{},{}> ".format(s, i+1) + DDD.name + "\n"
        return output

    def __repr__(self, depth=0):
        return self.log(depth)

    def selfID(self):
        return (self.parent.series.index(self), None)

    def getCrntVal(self, key):
        """Returns the relative or absolute value of the parameter key."""
        # TODO: Will be deprecated.
        if len(key) == 4:
            return self.data[key[0]].getCrntVal(key[-3:])
        elif len(key) == 2 and key[0] == 'meta':
            return self.crntMetaF[key]
        else: raise(AttributeError("Series does not have 3-tuple parameters."))

    def setCrntVal(self, key, val):
        """Updates the value of the parameter key."""
        # TODO: Will be removed.
        #self.crntParsH[key[0]][key[1]][key[2]] = float(val)
        #self.smplDistF.clear()
        pass

    def getDfltParsH(self):
        """Returns a complete hierarchical dictionary of default parameters."""
        result = defaultTreePars(self.T, lshapeOrder=self.lshapeOrder)

        # Update the default parameters using priors defined for the Series, if there are any
        for node_name in result.keys():
            for par_name, par_array in result[node_name].items():
                for i in range(len(par_array)):
                    key = (node_name, par_name, i)
                    result[key[0]][key[1]][key[2]] = self.getPrior(key).dflt()
        return result

    def setPrior(self, key, customPriors=None, reset=True, **kwargs):
        """Updates the prior key with parameters passed in kwargs (other parameters are left unchanged). Sets a new prior if no prior has been defined for this Series."""
        par = self.getPrior(key, customPriors)
        par = par._replace(**kwargs)

        if customPriors is not None:
            customPriors[key] = par
        else:
            if len(key) == 4:
                # Set for the specific Datum only
                self.data[key[0]].setPrior(key[-3:], **kwargs)
            else:
                # Set the prior
                self.parsSpecDict[key] = par
                # Remove this parSpec from all datasets in this series
                if reset:
                    for DD in self.data:
                        try:
                            DD.parsSpecDict.pop(key)
                        except KeyError:
                            pass

    def getPrior(self, key, customPriors=None):
        """Returns the specification of a parameter in the current tree or tau. Start by looking for the specification in the current datum structure, then proceed to the series level andthe tree if the parameter is not found."""
        # Try looking for the prior in the dictionary of custom priors first
        if customPriors is not None:
            try: return customPriors[key]
            except KeyError: pass

        # Handle 4-tuple keys
        if len(key) == 4:
            return(self.data[key[0]].getPrior(key[-3:]))

        # Handle 2- and 3-tuple keys
        try:
            return self.parsSpecDict[key]
        except KeyError:
            try:
                return self.parent.parsSpecDict[key]
            except KeyError:
                return getattr(self.T[key[0]], key[1])[key[2]]

    def setReferenceChshKey(self, key=None):
        """Sets the reference chemical shift and updates the global chemical shift accordingly."""
        for DDD in self.data:
            DDD.setReferenceChshKey(key)

    def addMetaParameter(self, **kwargs):
        """Adds a new meta parameter to the current Series. Additional key word argumrnts may include standard parsSpec arguments: label, min, max, distr, p1, p2, dval."""
        # Check maybe there is already a parameter with such label
        par_new = parsSpec(**kwargs)     # New specification of the parameter
        for key, par_old in self.parsSpecDict.items():
            if par_old.label == par_new.label and key[-2] == 'meta':
                print("A meta-parameter with label \"{}\" already exists. Please refer to it by its key, {}".format(par_new.label, key))
                self.parsSpecDict[key] = par_new
                self.crntMetaF[key[-2:]] = par_new.dflt()
                return 0

        new_indx = max([key[-1] for key in self.parsSpecDict.keys() if key[-2] == 'meta']+[-1]) + 1
        key = ('meta', new_indx)
        #print("Adding new meta-parameter to the Series. Please refer to it by its key,", key[-2:])
        self.parsSpecDict[key] = par_new
        self.crntMetaF[key[-2:]] = par_new.dflt()

    def setMetaFunction(self, func):
        """Sets an externally defined function func to self._meta"""
        self._meta = func

    def remMetaFunction(self):
        """Removes the meta function from the Series."""
        self._meta = None

    def setJointPrior(self, func):
        """Sets an externally defined function func to self._joint"""
        self._joint = func

    def remJointPrior(self):
        """Removes the jointPrior from the Series."""
        self._joint = None

    def isXclRootName(self, name):
        """Checks whether the node with the a certain name is ignored in the analysis."""
        flags = [DDD.isXclRootName(name) for DDD in self.data]
        return all(flags) and len(flags)>0

    def toggleXclRootName(self, name):
        """Sets the node to the igonred state."""
        if len(self.data) > 0:
            flag = self.data[0].isXclRootName(name)      # The current state in the first Datum
            for DDD in self.data:
                if flag:
                    DDD.xclRootNames.discard(name)
                else:
                    DDD.xclRootNames.add(name)
                DDD.resetSignals()

    def addDatum(self, yT, **kwargs):
        """Adds a Datum to the Series."""
        # Create new Datum structure and add it to the Series
        yT = yT.reshape(-1,1)     # Make sure the data is reshaped properly
        name = kwargs.pop('name', 'Datum #'+str(len(self.data)+1))
        DDD = Datum(yT, parent=self, name=name, **kwargs)
        self.data.append(DDD)

        return DDD

    def copySettings(self, serFrom):
        """Copies main settings (freq blocks, steps, parameters distributions) from another series, serFrom."""
        if serFrom != self:
            # Remove existing frequency blocks
            for i in range(1, len(self.freqBlocks)):
                self.remFreqBlock(i)

            # Remove all Steps
            self.steps.clear()

            # Remove all Frequency blocks (only after the steps have been removed)
            self.freqBlocks.clear()

            # Add new frequency blocks
            for blk in serFrom.freqBlocks:
                self.addFreqBlock(lims=(blk.min, blk.max), bslnOrder=blk.bslnOrder)

            # Add new steps
            for step in serFrom.steps:
                self.steps.append(copy.deepcopy(step))

            # Update the parameter distributions
            for key, par in serFrom.parsSpecDict.items():
                try:
                    self.setPrior(key, min=par.min, max=par.max, label=par.label, distr=par.distr, p1=par.p1, p2=par.p2, dval=par.dval)
                except (KeyError, IndexError): pass

            # Reset apodization and zero-filling
            self.resetFreqs(zff=serFrom.zff, apod=serFrom.apod)

    def import_data(self, path):
        """Loads the data stored in the file. Updates the array of Datum structures; each Datum points onto this file."""
        with open(path, 'rb') as fp:
            data = [float(x.strip()) if i != 5 else x.strip() for i, x in enumerate(fp.readlines())]

        # Save the acquisition parameters; these should be the same for all spectra in the series (by convention)
        if self.c0 is None:
            self.c0 = data[0]
            self.f0 = data[1]
            nt = int(data[4])     # Number of time points
            self.t = np.array(data[6:nt+6]).reshape(-1,1)
            self.fullReset()
            # TODO: Check if new c0/f0 are the same as the old ones when loading the rest of the data

        nt = len(self.t)
        yT = (np.array(data[nt+6:2*nt+6]) + 1j*np.array(data[-nt:])).reshape(-1,1)

        # Create new Datum structure and add it to the Series
        DDD = self.addDatum(yT, name = path[path.rfind('\\')+1:path.rfind('.')])

        return DDD

    def synth_data(self, name='', evalParsH=None, parsDict=None, ampl=None, phase=None, tau=None, theta=None, sigma2=None, sT=None, arrVal=None):
        """Generates synthetic data based on the evalParsH parameters (use dfltParsH by default). ampl is an nz_x_1 vector, where nz is the number of model signals."""

        # Set the default parameters
        if evalParsH is None:
            evalParsH = self.getDfltParsH()

        # Overwrite some default parameters with specifically supplied values in parsDict (flat dictionary of parsKey:value)
        if parsDict is not None:
            for k, v in parsDict.items():
                evalParsH[k[0]][k[1]][k[2]] = v

        # Take the amplitudes from evalParsH if none is explicitely supplied
        if ampl is None or len(ampl) != len(self.repRootNames):
            ampl = np.array([evalParsH[name]['ampl'][0] for name in self.repRootNames])

        if phase is None or len(phase) != len(self.repRootNames):
            phase = np.array([evalParsH[name]['phase'][0] for name in self.repRootNames])

        if tau is None:
            tau = evalParsH['.']['tau'][0]

        if theta is None:
            theta = evalParsH['.']['theta'][0]

        if sigma2 is None:
            sigma2 = evalParsH['.']['sigma2'][0]

        if sT is None or len(sT) != len(self.t):
            sT = 1.0

        # Generate the signal
        zT, _ = getFID(self.T, self.t, self.c0, self.f0, evalParsH, tau)
        xT = np.dot(zT, np.array(ampl).reshape(-1,1)) * np.exp(1j*theta)
        xT *= sT
        yT = xT + np.sqrt(sigma2/2)*(np.random.randn(*xT.shape) + 1j*np.random.randn(*xT.shape)) if sigma2 > 0 else xT

        # Add new data to the series
        DDD = Datum(yT, parent=self, name = name if name != '' else 'data_'+str(len(self.data)+1), arrVal = arrVal )
        self.data.append(DDD)

        return DDD

    def resetFreqs(self, zff=None, apod=None):
        """Resets the frequency scale for the entire Series and computed spectra."""

        # Reset the zero-filling factor if it has been supplied
        if zff is None:
            zff = self.zff
        else:
            self.zff = zff

        nf = next_pow_of_2( 2**zff * len(self.t) )     # Determine the number of samples in the full signal spectrum (possibly including zero-filling)

        if nf > 0:
            dt = self.t[1]-self.t[0]
            self.f = (np.fft.fftshift(np.fft.fftfreq(nf, dt))+self.f0).reshape(-1,1) / self.c0

            # Update the frequency blocks
            for i in range(len(self.freqBlocks)):
                self.altFreqBlock(indx=i)
            if len(self.freqBlocks) == 0 : self.addFreqBlock()   # Add the "all frequencies" block

            # Compute a window in the time domain
            if apod is not None : self.apod = apod
            self.wT = np.exp(-self.apod*self.t) if self.apod > 0 else 1

            for D in self.data:
                D.resetSignals()

    def fullReset(self, zff=None, apod=None, priors=None):
        self.resetFreqs(zff, apod)
        self.parsSpecDict = priors if priors is not None else {}
        self.crntMetaF.clear()
        self.remMetaFunction()

    def clear_data(self):
        """Removes al datasets from the series."""
        self.data.clear()

    def addFreqBlock(self, lims=None, bslnOrder=(0, 0)):
        """Adds a frequency block for optimization at certain in the self.freqBlocks arrays."""
        if lims is None:
            self.freqBlocks = []         # Reset the frequency blocks and add the entire frequency range
            lims = (-1*float('inf'), float('inf'))

        self.freqBlocks.append(freqSpec(min(lims), max(lims), bslnOrder))

        # Include the new block in all steps
        for step in self.steps:
            if 0 in step.frqBlkIds: step.frqBlkIds.clear()    # Make it impossible to optimize over the entire frequency range and some specific smaller ranges
            step.frqBlkIds.add(len(self.freqBlocks)-1)

    def altFreqBlock(self, lims=None, bslnOrder=None, indx=-1):
        """Alters a frequency block at position indx in self.freqBlocks (the last block by default)."""
        if indx == 0:         # Can't change the limits of the first block
            lims = (self.freqBlocks[indx].min, self.freqBlocks[indx].max)

        self.freqBlocks[indx].update(lims, bslnOrder)

    def remFreqBlock(self, indx):
        """Removes a frequency block from the series and updates all steps accordingly."""
        if indx > 0 and indx < len(self.freqBlocks):
            # Remove this block from all steps and decrement the indices of the rest of the references
            for step in self.steps:
                step.frqBlkIds = set([i if i < indx else i-1 for i in step.frqBlkIds if i != indx])
            self.freqBlocks.pop(indx)
            return True
        else:
            return False

    def reduce_range(self, lims):
        """Reduces the frequency range of the signals to new limits lims (in ppm)."""

        # Filter and cut each Datum
        dt = self.t[1]-self.t[0]
        for DDD in self.data:
            DDD.yT, f0_new, dt_new = cut_roi(DDD.yT, [min(lims), max(lims)], self.c0, self.f0, dt)
        nt_new = len(self.data[0].yT)
        self.f0 = f0_new
        self.t = np.linspace(0.0, dt_new*(nt_new-1), nt_new).reshape(-1, 1)     # Update the vector of sampling times

        # Check if all frequency blocks are within the new range limits. Update/remove if necessary
        indx_to_remove = []
        for indx, blk in enumerate(self.freqBlocks):
            if blk.min < min(lims) and blk.max > max(lims):
                indx_to_remove.append(indx)       # Take a note to remove this block later
            elif blk.min < min(lims):
                self.altFreqBlock(lims=[min(lims), blk.max], indx=indx)
            elif blk.max > max(lims):
                self.altFreqBlock(lims=[blk.min, max(lims)], indx=indx)
        for indx in indx_to_remove:
            self.remFreqBlock(indx)

        # Reset the signals
        self.resetFreqs()

    def _fnc_prior(self, evalParsH, evalMetaF, parsKeys, customPriors=None):
        """Custom prior probability function. Can be used to describe dependencies among parameters in different planes. Use parsKeys to determine if the prior needs to be computed for the specific keys."""
        if len(parsKeys) == 0:
            return 0
        else:
            # Evaluate for ALL meta parameters
            #if parsKeys is None:    # All meta parameters
            parsKeys = [key for key in self.parsSpecDict.keys() if len(key)==2]
            return sum([self.getPrior(key, customPriors).evalPrior(arg=evalMetaF[key]) if len(key)==2 else 0 for key in set(parsKeys) ])

    def evaluate(self, evalParsH=None, evalMetaF=None, parsKeys=None, autoKeys=None, frqBlkIds=None, freqMask=None, funcType=None, evaluatePriors=False, customPriors=None, robust=None, returnSignals=False, evaluateAll=True):
        """Evaluates the objective function (sum of logLikelihoods for each datum + sum of logPriors).
           Inputs:
           evalParsH - a list of hierarchical dictionaries one for each Datum
           evalMetaF - flat dictionary of meta-parameters' values"""

        # Prepare keys and starting parameters. Expand parameter keys (if 3-tuples were provided, they will be substituted with 4-tuples for all datasets) and make sure there are no repeats
        if parsKeys is None or autoKeys is None:
            parsKeys, autoKeys = self._prepareKeys(parsKeys, autoKeys, verbose=verbose)

        if evalParsH is None:
            evalParsH = [DDD.crntParsH for DDD in self.data]      # evalParsH=[None]*len(self.data)      # All planes will be evaluated with their current parameters
        if evalMetaF is None:
            evalMetaF = copy.copy(self.crntMetaF)

        # Initialize the result and meta data
        result = 0
        m_ampl = np.zeros((len(self.repRootNames), len(self.data)))
        S_ampl = np.zeros((len(self.repRootNames), len(self.repRootNames), len(self.data)))
        theta, a_sigma2, b_sigma2 = np.zeros(len(self.data)), np.zeros(len(self.data)), np.zeros(len(self.data))

        # Evaluate the metafnction to update values/distributions based on custom meta parameters. This could possibly change evalParsH and customPriors
        if self._meta is not None:
            self._meta(self, evalParsH, evalMetaF, customPriors)

        # Evaluate cost functions for all data planes
        for i, DDD in enumerate(self.data):
            parsKeysDatum = set([key[-3:] for key in parsKeys if key[0]==i or len(key)==3]) if parsKeys is not None else None    # Select only keys of non-linear parameters. This will exclude all meta-parameters' keys
            autoKeysDatum = set([key[-3:] for key in autoKeys if key[0]==i or len(key)==3]) if autoKeys is not None else None
            if (not evaluateAll) and (parsKeysDatum == set([])): continue          # Skip some datasets that we don't need to evaluate (there are no keys relating to the i-th dataset)
            _res, meta = DDD.evaluate(evalParsH[i], parsKeysDatum, autoKeysDatum, frqBlkIds, freqMask, funcType, evaluatePriors, customPriors, robust=robust, returnSignals=returnSignals)
            result += _res
            m_ampl[..., i] = meta['ampl'][0].ravel()
            S_ampl[...,i] = meta['ampl'][1]
            theta[i] = meta['theta']
            a_sigma2[i], b_sigma2[i] = meta['sigma2']

        if evaluatePriors:
            result += self._fnc_prior(evalParsH, evalMetaF, parsKeys, customPriors=customPriors)      # Add prior on the series level

        return result, {"ampl":(m_ampl, S_ampl), "theta":theta, "sigma2":(a_sigma2, b_sigma2)}

    def optimize(self, parsKeys, autoKeys=None, frqBlkIds=None, freqMask=None, funcType=None, evaluatePriors=False, nhop=None, respectBounds=True, verbose=True):
        """Optimization over the tree parameters selected in the parsKeys (list of tuples of the form: (datum_id, node_name, parameter_name, parameter_id), e.g. (2, 'Sucrose-F', 'chshQD', 5) )."""

        # Prepare keys and starting parameters. Expand parameter keys (if 3-tuples were provided, they will be substituted with 4-tuples for all datasets) and make sure there are no repeats
        parsKeys, autoKeys = self._prepareKeys(parsKeys, autoKeys, verbose=verbose)

        if len(parsKeys) > 0:
            evaluateAll = any([key[-2]=='meta' for key in parsKeys])     # If optimizing over any meta parameters, will need to evaluate all Datums, else can skip some

            # Define optimization ranges and initial values of parameters
            evalParsH = [copy.deepcopy(DDD.crntParsH) for DDD in self.data]    # Make a copy of parameters which will be used to evaluate the function
            evalMetaF = copy.deepcopy(self.crntMetaF)
            bounds = tuple((self.getPrior(key).min, self.getPrior(key).max) for key in parsKeys)
            initVals = np.array([evalParsH[k[0]][k[1]][k[2]][k[3]] if len(k) == 4 else evalMetaF[k] for k in parsKeys])
            customPriors = dict()
            # Define the objective function
            def costFuncOpti(x):
                if np.isnan(x).any():
                    return -np.inf
                for k, v in zip(parsKeys, x):
                    if len(k) == 4:
                        evalParsH[k[0]][k[1]][k[2]][k[3]] = v
                    elif len(k) == 2:
                        evalMetaF[k] = v
                # Evaluate the function skipping the datasets that are not present in parsKeys
                return -self.evaluate(evalParsH, evalMetaF, parsKeys, autoKeys, frqBlkIds, freqMask, funcType, evaluatePriors, customPriors, robust=False, evaluateAll=evaluateAll)[0]

            res = self._optimize(costFuncOpti, bounds, initVals, nhop=nhop, respectBounds=respectBounds, verbose=verbose)

            if res is not None:
                # Update the structure of all parameters
                for k, v in zip(parsKeys, res.x):
                    if len(k) == 4:
                        self.data[k[0]].setCrntVal(k[1:], v) #   crntParsH[k[1]][k[2]][k[3]] = v
                    elif len(k) == 2:
                        self.crntMetaF[k] = v

        # Re-evaluatethe posterior
        result, meta = self.evaluate(None, None, parsKeys, autoKeys, frqBlkIds, freqMask, funcType, evaluatePriors, returnSignals=True)

        if verbose:
            if len(parsKeys) > 0:
                print("Optimization finished. Posterior={:.4g}".format(result))
                print('Found values:')
                for key in parsKeys:
                    print("     {} = {:.5g}".format(str(key), self.getCrntVal(key)))

        return result, meta

    def sample(self, parsKeys, autoKeys=None, frqBlkIds=None, freqMask=None, funcType=None, evaluatePriors=False, nwalkers=None, nsteps=None):
        """Samples the posterior distribution using the MCMC algorithm."""

        parsKeys, autoKeys = self._prepareKeys(parsKeys, autoKeys, verbose=False)
        evaluateAll = any([key[-2]=='meta' for key in parsKeys])     # If optimizing over any meta parameters, will need to evaluate all Datums, else can skip some
        result = {}
        self.smplDistF.clear()             # Clear the characteristics of marginal distributions

        # If there are no parameters to sample
        # First evaluate the cost function with current parameters. If there is nothing to sample, this will be output as the result (at least for some datasets).
        value, meta = self.evaluate(autoKeys=autoKeys, frqBlkIds=frqBlkIds, freqMask=freqMask, funcType=funcType, evaluatePriors=evaluatePriors, returnSignals=True)
        for j in range(len(self.data)):
            for i, a in enumerate(meta['ampl'][0]):
                result[(j, self.repRootNames[i], 'ampl', 0)] = np.array([a[j]])
            result[(j, '.', 'ampl', 'mean')] = meta['ampl'][0][..., j,np.newaxis]
            result[(j, '.', 'ampl', 'covr')] = meta['ampl'][1][..., j,np.newaxis]     # Add the third (singular) dimension corresponding to the number of samples
            sigma2 = np.array([[x[j]] for x in meta['sigma2']])
            result[(j, '.', 'sigma2', 0)] = sigma2[1] / (sigma2[0]-1)     # Mean estimator for sigma
            result[(j, '.', 'sigma2', 'distr')] = sigma2
            result[(j, '.', 'theta', 0)] = np.array([meta['theta'][j]])

        # If no parameters are set for sampling, just evaluate the marginal posterior
        if len(parsKeys) == 0:
            print("Done. No non-marginalized parameters were requested.")
            return result

        # If there are some parameters to sample
        # Define sampling ranges and initial values of parameters
        evalParsH = [copy.deepcopy(DDD.crntParsH) for DDD in self.data]    # Make a copy of parameters which will be used to evaluate the function
        evalMetaF = copy.deepcopy(self.crntMetaF)
        bounds = tuple((self.getPrior(key).min, self.getPrior(key).max) for key in parsKeys)
        initVals = [evalParsH[k[0]][k[1]][k[2]][k[3]] if len(k) == 4 else evalMetaF[k] for k in parsKeys]
        customPriors = dict()
        # Define the objective function
        def costFuncSmpl(x):
            for k, v in zip(parsKeys, x):
                if len(k) == 4:
                    evalParsH[k[0]][k[1]][k[2]][k[3]] = v
                elif len(k) == 2:
                    evalMetaF[k] = v
            # Evaluate the function skipping the datasets that are not present in parsKeys
            return self.evaluate(evalParsH, evalMetaF, parsKeys, autoKeys, frqBlkIds, freqMask, funcType, evaluatePriors, customPriors, evaluateAll=evaluateAll)

        sampler = self._sample(costFuncSmpl, bounds, initVals, nwalkers, nsteps)

        # Form a table of results for the output
        # sampler.blobs has size nsteps x nwalkers
        # sampler.chain in nwalkers x nsteps x ndim
        # Want an output in the form nsamples x ndim
        flatchain = sampler.flatchain
        for i, key in enumerate(parsKeys):
            if key[1] not in ['theta', 'ampl', 'sigma2']:
                result[key] = flatchain[:, i]
                self.smplDistF[key] = smplSpec_from_data(result[key])

        sigma2 = np.array([blbWlkr['sigma2'] for blbSmpl in sampler.blobs for blbWlkr in blbSmpl]).T
        theta = np.array([blbWlkr['theta'] for blbSmpl in sampler.blobs for blbWlkr in blbSmpl]).T
        for j in set([k[0] for k in parsKeys if len(k)==4]):      #   range(len(self.data)):
            m_ampl = np.array([blbWlkr['ampl'][0][:,j].ravel() for blbSmpl in sampler.blobs for blbWlkr in blbSmpl]).T
            for i in range(m_ampl.shape[0]):
                result[(j, self.repRootNames[i], 'ampl', 0)] = m_ampl[i,:]

            m_ampl = np.array([blbWlkr['ampl'][0][...,j].ravel() for blbSmpl in sampler.blobs for blbWlkr in blbSmpl]).T     # Means of the amplitudes
            S_ampl = np.array([blbWlkr['ampl'][1][...,j] for blbSmpl in sampler.blobs for blbWlkr in blbSmpl]).T             # Covariance matrices of the amplitudes
            #result[(j, '.', 'ampl', 'covr')] = np.array([blbWlkr['ampl'][1][...,j] for blbSmpl in sampler.blobs for blbWlkr in blbSmpl]).T
            result[(j, '.', 'ampl', 'mean')] = m_ampl
            result[(j, '.', 'ampl', 'covr')] = S_ampl
            result[(j, '.', 'sigma2', 0)] = sigma2[j, 1,:] / (sigma2[j, 0,:]-1)     # Mean estimator for sigma
            result[(j, '.', 'sigma2', 'distr')] = sigma2[j,...]

            # Theta
            key = (j, '.', 'theta', 0)
            result[key] = theta[j,...].ravel()
            self.smplDistF[key] = smplSpec_from_data(result[key])

        return(result)

    def _prepareKeys(self, parsKeys, autoKeys=None, verbose=True, print_parameters=False):
        """Convert parameter Keys from Datum to Series representations and make sure there are no repetitions."""
        parsKeys = set([]) if parsKeys is None else set(parsKeys)
        if autoKeys is not None: autoKeys = set(autoKeys)

        for key in list(parsKeys):
            if len(key) == 3:
                parsKeys.remove(key)
                parsKeys.update([(i, *key) for i in range(len(self.data))])        # Repeat the same key for all Datums
        parsKeys = sorted(list(parsKeys))

        if verbose:
            # print('\n')
            npar_auto = len(autoKeys)
            npar_fit = len(parsKeys)
            if npar_fit == 0:
                print("Nothing to fit; {} parameters inferred in closed form...".format(npar_auto))
            elif npar_fit == 1:
                print("Fitting one parameter; {} parameters inferred in closed form...".format(npar_auto))
            else:
                print("Fitting {} parameters; {} parameters inferred in closed form...".format(npar_fit, npar_auto))
            if print_parameters:
                print("Optimized parameters:")
                for key in parsKeys:
                    print("     {}".format(str(key)))
                print("Parameters inferred automatically:")
                for key in autoKeys:
                    print("     {}".format(str(key)))

        return parsKeys, autoKeys

    def remove(self):
        """Removes itself from the Workspace"""
        self.parent.series.remove(self)

    def copy(self, name=None):
        """Creates a copy of the Series in the workspace"""
        newSSS = self.addSeries(name = name if name is not None else self.name+'_COPY', c0=self.c0, f0=self.f0, \
                                t=self.t.copy(), priors=copy.deepcopy(self.parsSpecDict))
        for blck in self.freqBlocks:
            newSSS.addFreqBlock(lims=(blck.min, blck.max), bslnOrder=blck.bslnOrder)
        for DDD in self.data:
            newSSS.addDatum(yT=DDD.yT.copy(), name=DDD.name, crntParsH=copy.deepcopy(DDD.crntParsH), priors=copy.deepcopy(DDD.parsSpecDict))
        return newSSS

    def evalForPlot2D(self, keys, frqBlkIds=None, lims=None, npts=25):
        ndim = 2

        if len(keys) != ndim: raise RuntimeError('Only 2 dimensional inputs are supported.')
        if any([len(key) != 2 for key in keys]): raise RuntimeError('Plotting for only meta-parameters is supported. Please use evalForPlot2D on the data level to plot distributions of usual parameters')

        pars = [self.getPrior(key) for key in keys]

        if lims is None or len(lims) != ndim:
            lims = [(par.min, par.max) for par in pars]

        # Form a grid of points at which the function will be evaluated
        x_arr = np.linspace(min(lims[0]), max(lims[0]), npts)
        y_arr = np.linspace(min(lims[1]), max(lims[1]), npts)

        evalParsH = [copy.deepcopy(DDD.crntParsH) for DDD in self.data]    # Make a copy of parameters which will be used to evaluate the function
        evalMetaF = copy.deepcopy(self.crntMetaF)      # Make a copy of the parameter dictionary that will be used for evaluation
        def func(x, y):
            """for k, v in zip(keys, (x, y)):
                if len(k) == 4:
                    evalParsH[k[0]][k[1]][k[2]][k[3]] = v
                elif len(k) == 2:
                    evalMetaF[k] = v"""
            evalMetaF[keys[0]], evalMetaF[keys[1]] = x, y
            # Evaluate the function skipping the datasets that are not present in parsKeys
            lpst = self.evaluate(evalParsH, evalMetaF, parsKeys=keys, frqBlkIds=frqBlkIds, funcType=None, evaluatePriors=True, customPriors=dict(), robust=False)[0]
            lpri = pars[0].evalPrior(arg=x) + pars[1].evalPrior(arg=y)
            return lpst, lpri

        lpst_arr, lpri_arr = np.vectorize(func)(*np.meshgrid(x_arr, y_arr, sparse=True))     # Return a table of f(x, y)
        llkl_arr = lpst_arr - lpri_arr

        crntVal = (self.crntMetaF[keys[0]], self.crntMetaF[keys[1]])

        return (x_arr, y_arr), llkl_arr, lpri_arr, lpst_arr, crntVal

    def evalForPlot(self, key, frqBlkIds=None, npts=75):
        """Returns an array of argument values and the values of log likelihood, prior, and posterior."""
        par = self.getPrior(key)     # Settings for the prior distribution of this parameter key
        x_arr = np.linspace(par.min, par.max, npts)
        lpst_arr = np.zeros(x_arr.shape)
        lpri_arr = np.zeros(x_arr.shape)
        evalParsH = [copy.deepcopy(DDD.crntParsH) for DDD in self.data]    # Make a copy of parameters which will be used to evaluate the function
        evalMetaF = copy.deepcopy(self.crntMetaF)                          # Make a copy of the parameter dictionary that will be used for evaluation
        for i, x in enumerate(x_arr):
            if len(key) == 2:
                evalMetaF[key] = x
            elif len(key) == 4:
                evalParsH[key[0]][key[1]][key[2]][key[3]] = x
            lpst_arr[i], _ = self.evaluate(evalParsH, evalMetaF, parsKeys=[key], frqBlkIds=frqBlkIds, funcType=None, evaluatePriors=True, customPriors=dict(), robust=False)
            lpri_arr[i] = par.evalPrior(arg=x)

        llkl_arr = lpst_arr - lpri_arr
        crntVal = self.data[key[0]].crntParsH[key[1]][key[2]][key[3]] if len(key) == 4 else self.crntMetaF[key]

        return x_arr, llkl_arr, lpri_arr, lpst_arr, crntVal

class Datum():
    """A single data instance. Contains signals of a single NMR experiment."""

    def __init__(self, yT, parent, name='', arrVal=None, crntParsH=None, flagAdapFreq=None, priors=None, xclRootNames=None, **kwargs):
        self.name = name
        self.parent = parent               # A series object that will contain this Datum
        self._f = None                     # Subsampled (adaptive) array of frequencies
        self.yT = yT    # The acquired signal in time domain (FID) without any preprocessing
        self.arrVal = arrVal if arrVal is not None else len(self.parent.data)+1     # Value of the arrayed parameter in the serial experiment (e.g., extent of reaction)
        self.parsSpecDict = {}
        self.crntParsH = None
        self.smplDistF = dict()          # A flat dictionary of sampled (or marginalized) parameters
        self.mdldPeaks = {}
        self.pckdPeaks = []
        self.refChshKey = None           # A key of the chemical shift that will be used as a reference (will be set to its default value and the rest of the spectrum shifted accordingly)
        self.xclRootNames = set(xclRootNames) if xclRootNames is not None else set([])       # Excluded RootNames
        self._joint = None            # A joint prior of all parameters
        self._flagAdapFreq = flagAdapFreq if flagAdapFreq is not None else (len(self.yT) > 2**16)
        self._gof = None              # Computed goodness of fit
        self.fullReset(crntParsH, priors)

    # @profile
    def __getattr__(self, attr):
        """Called with the dot notation for attributes not found in the class (e.g. parameters shared between many spectra in the series, c0, f0, etc.)."""

        # Return the adaptive frequency range
        if attr == 'f' and self._f is not None:
            return self._f

        # All other attributes
        return getattr(self.parent, attr)

    def __str__(self):
        return 'Dataset ' + self.name + ' in ' + self.parent.name

    def __repr__(self):
        return self.name

    def selfID(self):
        """Returns indices of the Series in the Workspace and the Datum in the Series"""
        return (self.parent.parent.series.index(self.parent), self.parent.data.index(self))

    def update(self):
            """Computes the spectral representation of the signal yT and updates the class parameters."""
            # nf = len(self.f)
            #
            # # Compute the spectrum of the input signal if necessary
            # if self.yF is None:
            #     self.yF = np.fft.fftshift(np.fft.fft(self.yT * self.wT, nf, axis=0), axes=0) / np.sqrt(nf)
            #
            # # Compute (possibly new) phasing terms
            # ph = np.exp(-1j*2*np.pi * self.crntParsH["."]["tau"][0] * (self.f*self.c0-self.f0) - 1j*self.crntParsH["."]["theta"][0] ).reshape((-1,1))
            #
            # # Get the data spectrum
            # self.yFph = self.yF * ph
            # if np.mean(self.yFph.ravel().real) < np.median(self.yFph.ravel().real):   # If the distribution is skewed to the left; i.e. only a few points are less than the most of them #sum(yFph.ravel().real) < 0:
            #     self.yFph = -1 * self.yFph
            #
            # # Still need the model spectrum if the evaluation was in time domain (xT is known but xFph is not)
            # if self.xFph is None and self.xT is not None:
            #     xF = np.fft.fftshift(np.fft.fft(self.xT * self.wT, nf, axis=0), axes=0) / np.sqrt(nf)
            #     self.xFph = self.xF * ph
            #     if sum(self.xFph.ravel().real) < 0:
            #         self.xFph = -1 * self.xFph
            raise RuntimeError      # This method to be deprecated

    def isAdapFreq(self):
        return self._flagAdapFreq

    def resetSignals(self, flagAdapFreq=None):
        self._f = None
        self.yF = np.fft.fftshift(np.fft.fft(self.yT * self.wT, len(self.f), axis=0), axes=0) / np.sqrt(len(self.f))
        self.zF = None        # A matrix of component signals
        self.bF = None        # A baseline
        self.zF_corr, self.bF_corr = None, None           # Corrections for the model matrix and the baseline
        self.sF, self.sT = None, None        # A lineshape kernel
        self.Gz = None
        self._gof = None

        # Reset the adaptive frequencies flag, if supplied
        if flagAdapFreq is not None:
            self._flagAdapFreq = flagAdapFreq

        if self._flagAdapFreq:
            # Sample more densely around the peaks
            yFabs = np.abs(self.yF)

            n = 128
            yFconv = scipy.signal.fftconvolve(yFabs, np.ones((n, 1))/n, mode='same')
            yFnorm = (yFconv - yFconv.min()) / (yFconv.max() - yFconv.min())

            # Compute a score for each sample 0 <= score <= 1
            # def score_fun(x):
            #     """A sigmoid-like function. Takes a number in the range 0<=x<=1 and outputs a score that determines whether this number should be kept in the subsampled array."""
            #     return np.where(x>0.1, 1.0, x*10.0)

            yFscore = np.where(yFnorm>0.1, 1.0, yFnorm*10.0)     # score_fun(yFnorm)

            indx = np.where(np.random.random((len(yFscore), 1)) < yFscore)[0]       # indx = np.sort(np.random.randint(0, len(yFabs), 2**15))
            self._f = self.parent.f[indx, :].reshape(-1,1)
            self.yF = self.yF[indx, :].reshape(-1,1)

    def resetCrntPars(self, crntParsH=None, priors=None):
        """Resets ALL current parameters."""
        # Reset the dictionary of priors
        self.parsSpecDict.clear()
        if priors is not None:
            self.parsSpecDict.update(priors)

        self.crntParsH = self.getDfltParsH()
        if crntParsH is not None:
            for key, val in crntParsH.items():
                try:
                    self.crntParsH[key].update(val)
                except KeyError:
                    print('Key {} is absent in the list of parameters.'.format(key))
        self.smplDistF.clear()          # A flat dictionary of sampled (or marginalized) parameters
        self.mdldPeaks.clear()
        self.pckdPeaks.clear()
        self._gof = None

    def fullReset(self, crntParsH=None, priors=None, flagAdapFreq=None):
        self.resetSignals(flagAdapFreq)
        self.resetCrntPars(crntParsH, priors)

    def alignToSolventPeak(self, chshTo=4.75):
        """Shifts the entire spectrum to align the highest peak (assumed to be the solvent peak) with a specified chemical shift."""
        chshFrom = self.f[np.argmax(np.abs(self.yF))]
        df = (chshTo - chshFrom)*self.c0
        self.yT *= np.exp(2*np.pi*1j*df*self.t)
        self.resetSignals()

    def fittableParsKeys(self, frqBlkIds=None, inRange=None, customPriors=None, node_name=None, considerRange=False):
        """Returns all _individually_ fittable parameters for a (sub)tree starting from a specific node."""
        # TODO: Need to check jcplQD
        if frqBlkIds is None:
            frqBlkIds = self.steps[-1].frqBlkIds

        # If in time domain, all parameters are good
        if len(frqBlkIds) == 0:
            return set(self.allParsKeys(node_name)), set([])

        if inRange is None:
            inRange, _ = splitFreq([ minmaxTuple(self.freqBlocks[blk].min, self.freqBlocks[blk].max) for blk in frqBlkIds ])

        # Construct the chemical shift tree
        if node_name is None:
            rootNode = self.T.findRoot()
        else: rootNode = self.T[node_name]

        P = rootNode.getChshTree()      # A subtree of chemical shifts

        # Set the chsh offset from the root node (if any)
        P.ancs = sum( [self.getCrntVal((rootNode.name, 'chsh', 0)) for node in rootNode.ancestors()] ) - self.getGlobalChshVal()    # Chemical shift due to all ancestors

        # Set the current values of the chemical shifts; remove ignored nodes
        for keynode in P.descendants(include_self=True):
            if keynode.name[0] in self.xclRootNames.intersection(self.repRootNames):
                # This will mark all parameters of the descending nodes as 'bad'
                keynode.crnt = np.inf
            else: keynode.crnt = self.getCrntVal(keynode.name)     # Set the current value

        # Propagate the limits
        P.propLims()

        # Loop over all possibly changing chemical shifts
        good_keys, bad_keys = [], []
        for keynode in P.descendants(include_self=True):
            # Find the offset given by the currently set value of a parameter, its descendants and possibly its range
            # Check if the peak is currently in the range
            if any([leaf.isInRange(inRange) for leaf in keynode.leaves()]):
                good_keys.extend(keynode.keys)

                # Decide what to do with the chemical shift parameter
                if not considerRange:
                    # The range doesn't matter; include the chemical shift as well
                    good_keys.append(keynode.name)
                else:
                    # Take into account not only the current value of a parameter but also its range, which may possibly be out of optimization bounds
                    distr = self.getPrior(keynode.name, customPriors)
                    offset = np.array([distr.min, distr.max]) - self.getCrntVal(keynode.name)
                    # Try, maybe even changing the chemical shift wouldn't matter
                    if any([leaf.isInRange(inRange, offset) for leaf in keynode.leaves()]):
                        good_keys.append(keynode.name)        # Add the chemical shift key
                    else: bad_keys.append(keynode.name)
            else:
                bad_keys.extend(keynode.keys)
                bad_keys.append(keynode.name)

        return set(good_keys), set(bad_keys)

    def isAutofittable(self, key, customPriors=None):
        """Checks if a parameter can be fitted algebraically/marginalized based on the definition of its prior distribution."""
        prior = self.getPrior(key, customPriors)
        if (key[1]=='sigma2' and prior.distr=='Inverse-Gamma') \
            or (key[1]=='theta' and prior.distr=='Uniform' and prior.min==-np.pi and prior.max==np.pi) \
            or (key[1]=='ampl' and key[0] in self.repRootNames and prior.distr=='Gaussian') \
            or (key[1]=='gamma' and prior.distr != 'Constant'):
            return True
        else: return False

    def isFittable(self, key, frqBlkIds=None, inRange=None, customPriors=None):
        """Checks if the parameter key can be fitted with the current settings of distributions/ranges."""
        # TODO: can be made much faster by not checking all nodes
        return key in self.fittableParsKeys(frqBlkIds, inRange, customPriors)

    def pickFittable(self, keys, frqBlkIds=None, inRange=None, customPriors=None):
        """Selects only (jointly) fittable parameters from the list of keys."""

        # Allways fit jcpl and general parameters
        # TODO: Need to check them as well
        good_keys = [key for key in keys if key[1]=='jcplQD' or key[0]=='.']
        keys = [key for key in keys if not ( key[1]=='jcplQD' or key[0]=='.' ) ]

        def equiv_chshKey(key):
            # Define a key that corresponds to a chemical shift (or chshQD) instead of alph or ampl
            if key[1] == 'ampl' and isinstance(self.T[key[0]], chemNodeT):
                rootName = self.T[key[0]].parent().name
                sfx = 'QD'
                indx = int(key[0].split('.')[-1])-1
            else:
                rootName, sfx, indx = key[0], key[1][4:], key[2]

            return (rootName, 'chsh'+sfx, indx)

        if frqBlkIds is None:
            frqBlkIds = self.steps[-1].frqBlkIds

        if inRange is None:
            inRange, _ = splitFreq([ minmaxTuple(self.freqBlocks[blk].min, self.freqBlocks[blk].max) for blk in frqBlkIds ])

        # Sort the nodes of the tree in the depth-first order
        nodes_in_tree = [node.name for node in self.T.iterDepth(method='in-order') if not isinstance(node, chemNodeT)]                           # Tree nodes' names sorted depth first
        keys = sorted(keys, key = lambda x : (nodes_in_tree.index(equiv_chshKey(x)[0]), equiv_chshKey(x)[1] ) )                              # Sort the keys such that all 'chsh' parameters come before 'chshQD'

        for key in keys:
            key_chsh = equiv_chshKey(key)
            # Build a chemical shift tree
            try: P[key_chsh]       # Check if this parameter is alreadty in the tree
            except (KeyError, UnboundLocalError):         # If no tree has been defined, or this parameter is not in the tree
                P = self.T[ key_chsh[0] ].getChshTree( key_chsh[1][4:], key_chsh[2] )      # A subtree of chemical shifts
                P.ancs = sum( [self.getCrntVal((node.name, 'chsh', 0)) for node in self.T[ key_chsh[0] ].ancestors()] ) - self.getGlobalChshVal()    # Chemical shift due to all ancestors

                # Remove all ignored roots
                for name in self.xclRootNames.intersection(self.repRootNames):
                    P[(name, 'chsh', 0)].cut()

                # For chemical shifts that are in the list of keys, use ranges of parameters; for all other chemcial shifts, use their current values
                for keynode in P.descendants(include_self=True):
                    if keynode.name[0] not in self.xclRootNames.intersection(self.repRootNames):
                        # This will mark all parameters of the descending nodes as 'bad'
                        keynode.crnt = np.inf
                    elif keynode.name in keys:
                        distr = self.getPrior(keynode.name)
                        keynode.crnt = np.array([distr.min, distr.max])
                    else: keynode.crnt = self.getCrntVal(keynode.name)

                P.propLims()     # Propagate the limits

            # Check if the corresponding chsh parameter would fall into the optimization range
            if any( [any([node.lims[0] > range.min and node.lims[1] < range.max for range in inRange]) for node in P[key_chsh].leaves()] ):
                good_keys.append(key)

        return set(good_keys)

    def isXclRootName(self, name):
        """Checks whether the node with the a certain name is ignored in the analysis."""
        return name in self.xclRootNames.intersection(self.repRootNames)

    def toggleXclRootName(self, name):
        """Sets the node to the igonred state."""
        try:
            self.xclRootNames.remove(name)
        except KeyError:
            self.xclRootNames.add(name)
        self.resetSignals()

    def setXclRootName(self, name, flag=True):
        """Sets the reported root name to excluded."""
        if not name in self.repRootNames:
            return 0

        if flag:
            self.xclRootNames.add(name)
        else:
            try:
                self.xclRootNames.remove(name)
            except KeyError:
                pass

        self.resetSignals()

    def getTree(self, node_name=None):
        """Return a copy of the tree rooted in the node with name node. All default distributions and parameter values are replaced with the current values in this Datum."""

        # Get the copy of the tree
        T = self.parent.parent.getTree(node_name)

        # Set default tree parameters to the current values from the Datum
        # allParsKeys = [key for key in flatten(defaultTreePars(T)).keys() \
        #                if key[1] in ['chsh', 'chshQD', 'alph', 'alphQD', 'jcplQD'] ]       # All parameters from the subtree
        for key in self.allParsKeys(node_name):
            par = copy.deepcopy(self.getPrior(key))       # Prior distribution in the Datum
            par = par._replace(dval=self.getCrntVal(key))
            T.setPrior(key, par)

        return T

    def getCrntVal(self, key):
        """Returns the relative or absolute value of the parameter key."""
        # TODO: Will be deprecated.
        if key[1] == 'intn':
            return self.T[key[0]].intn
        else:
            return self.crntParsH[key[0]][key[1]][key[2]]

    def setCrntVal(self, key, val):
        """Updates the current value of the parameter key."""
        self._gof = None           # Need to update the goodness of fit

        if key[1] == 'intn':
            self.T[key[0]].set_intn(val)
            return 0

        if key == self.refChshKey:
            self.setGlobalChshVal(self.getGlobalChshVal() + (float(val) - self.getPrior(key).dflt()) )
            val = self.getPrior(key).dflt()
        self.crntParsH[key[0]][key[1]][key[2]] = float(val)
        self.smplDistF.clear()

    def getCrntVals(self, node_name=None, keys=None):
        """Returns a flat dictionary of all parameters that affect nodes in the tree below and including the given node."""
        if keys is None:
            keys = self.allParsKeys(node_name) + [('.', 'theta', 0), ('.', 'tau', 0), ('.', 'sigma2', 0), ('.', 'gamma', 0)]
        parsF = {key:self.getCrntVal(key) for key in keys }
        return parsF

    def setCrntVals(self, parsF):
        """Updates maultiple parameters from a flat dictionary parsF."""
        for key, val in parsF.items():
            self.setCrntVal(key, val)

    def getGlobalChshVal(self):
        """Returns the value of the top-level chemical shift in the parameter tree."""
        return self.getCrntVal(key = (self.T.findRoot().name, 'chsh', 0))

    def setGlobalChshVal(self, val):
        """Sets the value of the top-level chemical shift in the parameter tree."""
        self.setCrntVal(key = (self.T.findRoot().name, 'chsh', 0), val=val)

    def setReferenceChshKey(self, key=None):
        """Sets the reference chemical shift and updates the global chemical shift accordingly."""
        self.refChshKey = key
        if key is not None:
            self.setGlobalChshVal(self.getGlobalChshVal() + (self.getCrntVal(key) - self.getPrior(key).dflt()) )
            self.setCrntVal(key, self.getPrior(key).dflt())

    def getDfltParsH(self):
        """Returns a complete hierarchical dictionary of default parameters."""
        result = defaultTreePars(self.T, lshapeOrder=self.lshapeOrder)
        for node_name in result.keys():
            for par_name, par_array in result[node_name].items():
                for i in range(len(par_array)):
                    key = (node_name, par_name, i)
                    result[key[0]][key[1]][key[2]] = self.getPrior(key).dflt()
        return result

    def getPrior(self, key, customPriors=None):
        """Returns the specification of a parameter in the current tree or tau. Start by looking for the specification in the current datum structure, then proceed to the series level andthe tree if the parameter is not found."""
        # TODO: Make it nicer...
        # Try looking for the prior in the dictionary of custom priors first
        if customPriors is not None:
            try:
                return customPriors[(self.parent.data.index(self), *key)]
            except KeyError:
                try:
                    return customPriors[key]
                except KeyError: pass

        try:
            return self.parsSpecDict[key]
        except KeyError:
            try:
                return self.parent.parsSpecDict[key]
            except KeyError:
                try:
                    return self.parent.parent.parsSpecDict[key]
                except KeyError:
                    try:
                        # Use the prior from the tree
                        return getattr(self.T[key[0]], key[1])[key[2]]
                    except KeyError:
                        print("Something is wrong with {}".format(key))

    def setPrior(self, key, customPriors=None, reset=True, **kwargs):
            """Updates the prior key with parameters passed in kwargs (other parameters are left unchanged). Sets a new prior if no prior has been defined for this Datum."""
            par = self.getPrior(key, customPriors)
            par = par._replace(**kwargs)

            # Set this prior
            if customPriors is not None:
                customPriors[key] = par
            else:
                self.parsSpecDict[key] = par

    def setJointPrior(self, func):
        """Sets an externally defined function func to self._joint"""
        self._joint = func

    def remJointPrior(self):
        """Removes the jointPrior from the Datum."""
        self._joint = None

    # @profile
    def _get_indxFreq(self, i, nw2=0):
        """Returns the indices of the frequency scale covered by the block i; takes into account possible padding by nw2 on both sides of the range."""

        dref_chsh = self.getGlobalChshVal() if config.DISPL_ShiftToReference else 0.0   # Reference chemical shift
        f = self.f - np.array(dref_chsh)

        return self.freqBlocks[i].indxFreq(f, nw2)

    def _get_signals_in_time(self, evalParsH, wnd=None):
        # 1. Compute model signals in time domain
        zT, _ = getFID(self.T, self.t, self.c0, self.f0, evalParsH, xclRootNames=self.xclRootNames)            # 1. Compute the model signals

        # 1. Apply custom lineshape correction if defined
        if self.sT is not None:
            zT *= self.sT

        # 2. Apply window in the time domain if needed
        yTw, zTw = (self.yT * self.wT * wnd, zT * wnd) if wnd is not None else (self.yT * self.wT, zT)

        ## Define modelled and measured signals
        return zTw[0:,:], yTw[0:, :]

    # @profile
    def _get_signals_in_freq(self, evalParsH, frqBlkIds=None, freqMask=None, wnd=None, numberField=None, convolve=True):
        """Returns a matrix of modelled signals and the y vector in frequency domain."""

        if frqBlkIds is None:
            frqBlkIds = self.steps[-1].frqBlkIds
        if numberField is None:
            numberField = config.SAMPL_numberField

        nw = len(self.sF) if self.sF is not None else 0         # Length of the adaptive lineshape window (in frequency domain)
        nw2 = int(nw/2)

        # Find indices for each frequency block (including padding)
        indxFreqByBlock = [ self._get_indxFreq(i, nw2) for i in frqBlkIds ]
        indxInRange = np.concatenate(indxFreqByBlock)

        if ( 'lshapeR' in evalParsH['.'].keys() and (any(evalParsH['.']['lshapeR']) or any(evalParsH['.']['lshapeI'])) ) or wnd is not None:
            zT, _ = getFID(self.T, self.t, self.c0, self.f0, evalParsH, tau=0.0, xclRootNames=self.xclRootNames)            # 1. Compute the model signals

            # 1. Apply custom lineshape correction if defined
            if self.sT is not None:
                zT *= self.sT

            # 2. Apply window in the time domain if needed
            yTw, zTw = (self.yT * self.wT * wnd, zT * wnd) if wnd is not None else (self.yT * self.wT, zT)

            # 3. Compute the spectra
            zF = np.fft.fftshift(np.fft.fft(zTw, len(self.f), axis=0), axes=0) / np.sqrt(len(self.f))
            yF = np.fft.fftshift(np.fft.fft(yTw, len(self.f), axis=0), axes=0) / np.sqrt(len(self.f))

            # 4. Take only the valid frequency ranges
            zFinRange = zF[indxInRange, :]
            yFinRange = yF[indxInRange, :]
        else:
            zFinRange, _ = evalTreeF(self.T, self.f[ indxInRange ], self.t[1]-self.t[0], self.c0, self.f0, evalParsH, xclRootNames=self.xclRootNames)

            # Apply custom lineshape correction (this reduces the range)
            if nw2 > 0 and convolve:
                indxSplit = np.cumsum([len(indx) for indx in indxFreqByBlock])[:-1]     # Indices showing how to split the concatenated arrays xF, yF, zF, etc.
                zFinRange = np.vstack([scipy.signal.fftconvolve(z, self.sF, 'valid') for z in np.split(zFinRange, indxSplit)]) / np.sqrt(len(self.f))
                indxInRange = np.concatenate([indx[nw2:-nw2] for indx in indxFreqByBlock])

            yFinRange = self.yF[indxInRange, :]

        if freqMask is not None:
            pass
            # # Control which frequencies should be excluded from optimization
            # unmaskedIndx = np.array([[f>msk[0] and f<msk[1] for msk in freqMask] for f in self.f[indxInRange]-dref_chsh ]).any(axis=1).ravel()
            # indxInRange = indxInRange[unmaskedIndx]

        # Possibly update the phased signal if the first-order phasing parameter has changed
        phFinRange = np.exp(-1j*2*np.pi * evalParsH["."]["tau"][0] * (self.f[indxInRange]*self.c0-self.f0) - 1j*0 ).reshape((-1,1))   # The phasing term
        yFinRange *= phFinRange

        # Choose only components that are in the optimization range
        # TODO!

        # Include the baseline
        bslnPoly = block_diag(*[self.freqBlocks[i].bline(len(indx)-2*nw2, numberField) \
                                for i, indx in zip(frqBlkIds, indxFreqByBlock)])     # All baseline models padded with zeros; use only real-valued baselines if the model is real-valued
        # if numberField == 'Re':
        #     bslnPoly = bslnPoly[:, np.isreal(bslnPoly).all(axis=0)]
        # #else: bslnPoly *= phFinRange
        if freqMask is not None:
            bslnPoly = bslnPoly[unmaskedIndx, :]
        nb = bslnPoly.shape[1]     # Total number of baseline terms

        return zFinRange, bslnPoly, yFinRange, indxInRange

    # @profile
    def _fnc_lklhd(self, evalParsH, frqBlkIds=None, autoKeys=None, freqMask=None, funcType=None, wnd=None, customPriors=None, returnSignals=False, robust=None, numberField=None):
        """Computes the value of the likelihood function. If evaluatePriors == True, will also add values of prior distributions for amplitudes, theta, and sigma2, if those parameters can not be integrated out."""
        #funcType = 'TLS'

        # 1. Update the settings
        if autoKeys is None:
            autoKeys = []
        if frqBlkIds is None:
            frqBlkIds = self.steps[-1].frqBlkIds
        if robust is None:
            robust = config.SAMPL_robustLS
        if funcType is None:
            funcType=config.SAMPL_funcType
        if numberField is None:
            numberField = config.SAMPL_numberField
        if funcType is 'TLS':
            numberField = 'Re'

        # 2. Compute a matrix of model signals Z, either in time or frequency domain
        inTimeDomain = (len(frqBlkIds) == 0)
        if inTimeDomain:
            numberField = 'Cx'
            Z, y = self._get_signals_in_time(evalParsH, wnd)
        else:
            zFinRange, bFinRange, yFinRange, indxInRange = self._get_signals_in_freq(evalParsH, frqBlkIds, freqMask, wnd, numberField)
            Z, y = np.hstack((zFinRange, bFinRange)), yFinRange

        ns, nz = Z.shape     # Number of samples and (model signals + baselines)
        reportedNames = [name for name in self.repRootNames if name not in self.xclRootNames]
        na = len( reportedNames )    # Number of model signals, and hence the resulting amplitudes
        if np.isnan(Z).any() or np.isinf(Z).any():             # This can happen if some chemical shifts are set to None
            return 0.0, {}

        # 3. Collect current values of the amplitudes, phase, and the variance of noise
        # 3.1. Amplitudes
        #ampl = np.array([evalParsH[self.repRootNames[i]]['ampl'][0] for i in range(na)]).reshape(-1,1) \
        #     * np.exp(1j*np.array([evalParsH[self.repRootNames[i]]['phase'][0] for i in range(na)])).reshape(-1,1)
        ampl = np.array([None]*nz)        # By default, if no amplitudes are set, they will be found as ML estimates
        # Set the corresponding priors
        m0 = np.zeros((nz, 1))         # Prior amplitudes
        S0 = np.where(np.identity(nz)>0, np.inf, 0)           # Prior covariance matrix of amplitudes (vague priors)
        for i in range(na):
            key = (reportedNames[i], 'ampl', 0)
            if key in autoKeys:
                # Set a Gaussian prior with the supplied mean and variance
                spec = self.getPrior(key, customPriors=customPriors)
                m0[i], S0[i,i] = spec.p1, spec.p2
            else:
                ampl[i] = evalParsH[reportedNames[i]]['ampl'][0]
                if numberField == 'Cx':
                    ampl[i] *= np.exp(1j*evalParsH[reportedNames[i]]['phase'][0])    # Set possibly different phases for each amplitude
        iS0 = np.linalg.inv(S0)

        # 3.2. Global phase shift
        key = ('.', 'theta', 0)
        if key in autoKeys:
            # Estimate theta using the closed form expression
            Zy = Z.conj().T.dot(y)
            ZZ = Z.conj().T.dot(Z)
            if ZZ.size > 0 and np.linalg.matrix_rank(ZZ.real) < ZZ.shape[0]:
                ZZ += (1e-09)*np.identity(ZZ.shape[0])          # Make sure ZZ is invertible if it is low rank
            Sc = np.linalg.inv(ZZ.real)
            theta = np.asscalar( 0.5*np.angle(Zy.T.dot(np.dot(Sc, Zy))) )
        else: theta = evalParsH['.']['theta'][0]
        if numberField == 'Re':
            # Use only the real part
            Z, y = Z.real, (y*np.exp(-1j*theta)).real
        elif numberField == 'ReIm':
            # Concatenate the real and imaginary parts
            Z, y = np.vstack([Z.real, Z.imag]), np.vstack([(y*np.exp(-1j*theta)).real, (y*np.exp(-1j*theta)).imag])
        elif numberField == 'Cx':
            # Use the complex signals
            y = y*np.exp(-1j*theta)

        # 3.3. Variance of noise
        key = ('.', 'sigma2', 0)
        if key in autoKeys:
            spec = self.getPrior(key, customPriors=customPriors)
            a_sigma2, b_sigma2 = spec.p1, spec.p2
            sigma2 = None
        else:
            a_sigma2, b_sigma2 = None, None
            sigma2 = evalParsH['.']['sigma2'][0]

        # 3.4. TLS ratio, gamma
        key = ('.', 'gamma', 0)
        if key in autoKeys:
            gamma = None          # Will fit gamma
        else:
            gamma = evalParsH['.']['gamma'][0]

        # 4. Compute the log-likelihood function
        if self.Gz is not None:
            Gz = self.Gz
        else:
            Gz = 0.000001*np.ones((ns, nz))
            Gz = 1.0*np.abs(Z)

        Gz[:, na:] = 0
        result, ampl, sigma2, meta = log_likelihood(Z, y, ampl=ampl, sigma2=sigma2, \
            Gz=Gz, Gy=None, gamma=gamma, m0=m0, iS0=iS0, a_sigma2=a_sigma2, b_sigma2=b_sigma2, \
            funcType=funcType, robust=robust)
        diff_theta = np.asscalar( 1/2*np.angle(ampl[:na].T.dot(ampl[:na])) )   # Global phase estimated from the complex valued amplitudes
        theta = (theta + diff_theta)   # + np.pi) % (2 * np.pi) - np.pi             # Updated value of theta
        m_ampl = ampl*np.exp(-1j*diff_theta)
        #m_ampl[:na] = m_ampl[:na].real
        # TODO: This needs revision
        if m_ampl[:na].real.sum() < 0:      # Make sure that all amplitudes are positive
            m_ampl = - m_ampl
            theta = (theta + np.pi + np.pi) % (2 * np.pi) - np.pi
            # print('Flippping the phase by 180 degrees...')
            if numberField == 'Re': evalParsH['.']['theta'][0] = (evalParsH['.']['theta'][0] + np.pi + np.pi) % (2 * np.pi) - np.pi  # Always update the phase if it needs to be flipped
        gamma = meta['gamma']
        S_ampl = meta['ampl'][1]
        a_sigma2, b_sigma2 = meta['sigma2']
        #print(m_ampl)

        mult = 1   # sum(m_ampl)     # Multiplier (can be used to output normalized amplitudes)
        for lbl, val in zip(reportedNames, m_ampl[:na]):
            evalParsH[lbl]['ampl'][0] = np.asscalar(np.abs(val)) / mult
            evalParsH[lbl]['phase'][0] = np.asscalar(np.angle(val)) if numberField == 'Cx' else 0.0
        evalParsH['.']['mult'][0] = mult
        evalParsH['.']['theta'][0] = theta            # Update the phase
        evalParsH['.']['sigma2'][0] = sigma2
        if gamma is not None: evalParsH['.']['gamma'][0] = gamma

        # Save and output the resulting signals zF and bF
        self.zF, self.bF = None, None                     # Reset the signals
        self.zF_corr, self.bF_corr = None, None           # Reset the corrections for the model matrix and the baseline
        if returnSignals:
            # Save the estimated signals
            if inTimeDomain:
                self.zF = np.fft.fftshift(np.fft.fft(Z, len(self.f), axis=0), axes=0) / np.sqrt(len(self.f))
            else:
                self.zF = np.zeros((len(self.f),na), dtype=complex)
                self.zF[indxInRange,:] = zFinRange ### / Znrm[:, 0:na]

            # Save the characteristics of the marginalized distributions
            for i in range(na):
                key=(reportedNames[i], 'ampl', 0)
                if key in autoKeys:
                    self.smplDistF[key] = smplSpec_Gaussian(np.asscalar(np.abs(m_ampl[i])), np.asscalar(np.abs(S_ampl[i,i])))

            key=('.', 'sigma2', 0)
            if key in autoKeys:
                self.smplDistF[key] = smplSpec_invGamma( a_sigma2, b_sigma2 )

        # Always save the baseline; it may be needed for the adjustmwnt algorithms
        if not inTimeDomain and nz-na>0:
            self.bF = np.zeros((len(self.f),1), dtype=complex)
            self.bF[indxInRange] = np.dot(bFinRange, m_ampl[-(nz-na):])

        meta['sigma2'] = (2.0, sigma2)
        meta['theta'] = theta       # distr = {"ampl":(m_ampl, S_ampl), "theta":theta, "sigma2":(a_sigma2, b_sigma2)}
        meta['ampl'] = (np.abs(m_ampl[:na]), S_ampl[:na, :na].real)

        return result, meta         # Output the log value and parameters of the marginalized distributions

    def _fnc_prior(self, evalParsH, parsKeys, customPriors=None):
        """Computes the prior functions. Priors will be computed only for parameters in the list parsKeys (if parsKeys is None -- all parameters will be included). Assumes that parameters that can be marginalized are not included in parsKeys."""
        if len(parsKeys) == 0:
            return 0
        else:
            # if parsKeys is None:    # All parameters
            #     parsKeys = flatten(self.crntParsH).keys()
            return sum([self.getPrior(key, customPriors).evalPrior(arg=evalParsH[key[0]][key[1]][key[2]]) for key in set(parsKeys) if key[1] not in ['ampl', 'theta', 'sigma2']])

    def _fnc_joint(self, evalParsH):
        """Evaluates the joint prior."""
        if self._joint is not None:
            return self._joint(evalParsH)
        elif self.parent._joint is not None:
            return self.parent._joint(evalParsH)
        else: return 0.0

    def measure_noise(self, lims, lmda=5.0):
        """Measures the standard deviation of noise in the spectrum within the limits lims in ppm."""

        indxFreq = np.arange(np.searchsorted(self.f.ravel(), min(lims)), \
                             np.searchsorted(self.f.ravel(), max(lims)))     # Indices of frequency points in the range

        yF = self.yF[indxFreq].ravel()
        yFbsln = whitsm(yF, lmda)
        yFnoise = yF - yFbsln
        #yFbsln[indxFreq] = self.yF[indxFreq] - whitsm(self.yF[indxFreq], 7.0)
        sigma2_est = np.sum(np.abs(yFnoise)**2) * (self.f.size/indxFreq.size) / self.t.size
        print("sigma2_est = {:.6f}".format(np.asscalar(sigma2_est)))
        return yF, yFbsln

    def set_shape(self, frqBlkIds=None, wnd=None):
        """Sets the custom lineshape sF and sT."""
        # TODO! Check this function when using an adaptive frequency scale
        if self.isAdapFreq():
            raise RuntimeError('ACustom lineshapes are not supported with adaptive frequency scale.')

        self._gof = None
        nt, nf = len(self.t), len(self.f)
        nw = config.MODEL_ShapeKernelSize         # Length of the adaptive lineshape window (in frequency domain)
        nw2 = int(nw/2)

        evalParsH = self.crntParsH

        if frqBlkIds is None:
            frqBlkIds = self.steps[-1].frqBlkIds

        # Compute a matrix of model signals Z, either in time or frequency domain
        if len(frqBlkIds) == 0:
            # -------------------------- TIME ----------------------------
            pass
        else:
            indxFreqByBlock = [ self._get_indxFreq(i, nw2) for i in frqBlkIds ]
            indxPadded = np.concatenate(indxFreqByBlock)
            indxInRange = np.concatenate([indx[nw2:-nw2] for indx in indxFreqByBlock])

            if ( 'lshapeR' in evalParsH['.'].keys() and (any(evalParsH['.']['lshapeR']) or any(evalParsH['.']['lshapeI'])) ) or wnd is not None:
                zT, _ = getFID(self.T, self.t, self.c0, self.f0, evalParsH, tau=0.0, xclRootNames=self.xclRootNames)            # 1. Compute the model signals

                # 1. Apply custom lineshape correction if defined
                if self.sT is not None:
                    zT *= self.sT

                # 2. Apply window in the time domain if needed
                yTw, zTw = (self.yT * self.wT * wnd, zT * wnd) if wnd is not None else (self.yT * self.wT, zT)

                # 3. Compute the spectra
                zF = np.fft.fftshift(np.fft.fft(zTw, len(self.f), axis=0), axes=0) / np.sqrt(len(self.f))
                yF = np.fft.fftshift(np.fft.fft(yTw, len(self.f), axis=0), axes=0) / np.sqrt(len(self.f))

                # 4. Take only the valid frequency ranges
                zFPadded = zF[indxPadded, :]
                yFinRange = yF[indxInRange, :]
            else:
                zFPadded, _ = evalTreeF(self.T, self.f[ indxPadded ], self.t[1]-self.t[0], self.c0, self.f0, evalParsH, xclRootNames=self.xclRootNames)
                indxSplit = np.cumsum([len(indx) for indx in indxFreqByBlock])[:-1]
                zFPadded = [z for z in np.split(zFPadded, indxSplit)]

            mc = np.array([evalParsH[name]['ampl'][0] for name in self.repRootNames if name not in self.xclRootNames])           # First na results correspond to the actual amplitudes of components, the rest, if any, correspond to the baselines
            theta = evalParsH['.']['theta'][0]

            xFPadded = [np.dot(z, mc*np.exp(1j*theta)).reshape(-1,1) for z in zFPadded]
            yFinRange = self.yF[indxInRange, :]
            bFinRange = self.bF[indxInRange, :]

            # Form the Toeplitz matrix of shifted arrays
            S = np.vstack([np.hstack([x[i:i+len(x)-2*nw2] for i in np.arange(2*nw2, -1, -1, dtype='int')]) for x in xFPadded])

            # Solve the system of equations
            SS = np.dot(S.T.conj(), S) + 0.00*np.eye(nw)
            sF = np.linalg.solve(SS, np.dot(S.T.conj(), yFinRange-bFinRange))
            sF /= sum(sF) / np.sqrt(len(self.f))

            # Compute the iFFT of the lineshape
            sF_padded = np.pad(sF.ravel(), (math.ceil((nf-nw)/2), math.floor((nf-nw)/2)), 'constant', constant_values=0).reshape(-1,1) # Zero-pad sF before taking the iFFT
            sT = np.fft.ifft(np.fft.ifftshift(sF_padded, axes=0), axis=0)[:nt] * np.sqrt(nf)
            self.sF, self.sT = sF, sT

    def reset_shape(self):
        """Resets the custom lineshape to its default values (None)."""
        self.sT = None
        self.sF = None
        self._gof = None

    def evaluate(self, evalParsH=None, parsKeys=None, autoKeys=None, frqBlkIds=None, freqMask=None, funcType=None, evaluatePriors=False, customPriors=None, robust=None, returnSignals=False):
        """Evaluates the objective function (logLikelihood + sum of logPriors).
           Inputs:
           evalParsH - hierarchical dictionary of parameters (node name -> parameter name -> list of parameters); use crntParsH by default
           parsKeys - list of parameter tuples (node name, parameter name, parameter index)
           frqBlkIds - list of indices of frequency blocks over which to evaluate the function; evaluate in time domain by default, []
           evaluatePriors - if True, will add values of priors to the likelihood function to compute the posterior. Only those priors specified by parsKeys will be evaluated. """

        if parsKeys is None or autoKeys is None:
            parsKeys, autoKeys = self._prepareKeys(parsKeys, autoKeys, frqBlkIds, customPriors, verbose=False)

        if evalParsH is None:
            evalParsH = self.crntParsH

        result, meta = self._fnc_lklhd(evalParsH, frqBlkIds, autoKeys, freqMask, funcType, customPriors=customPriors, returnSignals=returnSignals, robust=robust)

        if evaluatePriors:
            result += self._fnc_prior(evalParsH, parsKeys, customPriors=customPriors)
        result += self._fnc_joint(evalParsH)

        return result, meta

    def optimize(self, parsKeys, autoKeys=None, frqBlkIds=None, freqMask=None, funcType=None, evaluatePriors=False, nhop=None, respectBounds=True, verbose=True):
        """Optimization over the tree parameters selected in the parsKeys (list of tuples)."""

        parsKeys, autoKeys = self._prepareKeys(parsKeys, autoKeys, frqBlkIds, verbose=verbose)

        if len(parsKeys) > 0:
            # Define the objective function using a copy of the parameters dictionary
            evalParsH = copy.deepcopy(self.crntParsH)
            bounds = tuple((self.getPrior(key).min, self.getPrior(key).max) for key in parsKeys)
            initVals = [evalParsH[k[0]][k[1]][k[2]] for k in parsKeys]
            costFuncOpti = lambda x : -self.evaluate(updateFromFlat(evalParsH, parsKeys, x), parsKeys, autoKeys, frqBlkIds, freqMask, funcType, evaluatePriors, robust=False)[0]

            # Call the optimization routine
            res = self._optimize(costFuncOpti, bounds, initVals, nhop=nhop, respectBounds=respectBounds, verbose=verbose)

            if res is not None:
                # Update the stored parameters
                updateFromFlat(self.crntParsH, parsKeys, res.x)    # Updated structure of all parameters
                if self.refChshKey in parsKeys: self.setCrntVal(key = self.refChshKey, val = res.x[parsKeys.index(self.refChshKey)])
                self.smplDistF.clear()

        # Re-evaluate the posterior
        result, meta = self.evaluate(None, parsKeys, autoKeys, frqBlkIds, freqMask, funcType, evaluatePriors, returnSignals=True)

        if verbose:
            if len(parsKeys) > 0:
                print("Optimization finished. Posterior={:.4g}".format(result))
                print('Found values:')
                for key in parsKeys:
                    print("     {} = {:.5g}".format(str(key), self.getCrntVal(key)))

        return result, meta

    def goodness_of_fit(self, frqBlkIds=None):
        """Evaluates how well the model is fitted to the data on the scale from 0.0 (bad) to 1.0 (good)."""
        if self._gof is None:
            # print('Calculating GOF')
            self.evaluate(autoKeys=[], returnSignals=True)
            f, yFph, xF, _, bF = self.signals_for_plot(frqBlkIds=frqBlkIds, onlyInRange=True)
            rFabs = np.abs(yFph.real - xF.real)
            yFabs = np.abs(yFph.real)
            xFabs = np.abs(xF.real)

            # print(yFabs.sum(), xFabs.sum(), rFabs.sum(), np.median(rFabs), np.max(rFabs))

            score1 = np.abs(yFabs.sum()-xFabs.sum()) / yFabs.sum()
            score2 = rFabs.sum() / yFabs.sum()
            score3 = max(np.max(rFabs)-np.median(rFabs), 0.0) / np.median(rFabs)

            # print('GOF scores = ', score1, score2, score3)

            if score1 < 0.05 and score3 < 20.0:
                self._gof = 1.0
            elif 0.05 <= score1 and score1 < 0.2:
                self._gof = 0.5
            else:
                self._gof = 0.0

        return self._gof

    def auto_phase(self):
        """Run the autophasing algorithm."""

        def deg2tau(dt, nf, p0deg, p1deg):
            """Converts the phasing parameters from degrees to their tau and theta representations."""
            fmin, fmax = flims(dt, nf)
            df = (fmax-fmin)/(nf-1)     # The same as f[1]-f[0]

            theta = np.asscalar( p0deg - p1deg*fmin/df/nf )*np.pi/180.0
            tau = np.asscalar( p1deg / df / 360. / nf )

            return theta, tau

        def tau2deg(dt, nf, theta, tau):
            """Converts from theta/tau representation to phase angles in degrees."""
            _, fmax = flims(dt, nf)

            p0deg = theta * 180.0 / np.pi
            p1deg = tau*fmax*360.

            return p0deg, p1deg

        # Get the spectrum (or compute it if the signal is too long and adaptive FFT has been used)
        dt = self.t[1]-self.t[0]
        if self._f is None or len(self.yF) > 2**14:
            yF = self.yF
            nf = len(yF)
        else:
            # If an adaptive frequency scale has been used, compute a new (shorter) spectrum and autophase it instead
            nf = min(len(self.yT), 2**14)
            yT = self.yT[:nf, :] * np.exp( -5*np.linspace(0, 1, nf) ).reshape(-1,1)
            yF = np.fft.fftshift(np.fft.fft(yT, nf, axis=0), axes=0) / np.sqrt(nf)

        # Get the initial phasing parameters (in degrees)
        p0deg, p1deg = tau2deg(dt, nf, self.getCrntVal(key = ('.', 'theta', 0)), \
                               self.getCrntVal(key = ('.', 'tau', 0)) )

        p0, p1 = nmrglue.process.proc_autophase.automatic_ps(yF.ravel(), 'acme', p0=-p0deg, p1=-p1deg)     # 'peak_minima'
        p0deg, p1deg = -p0, -p1

        theta, tau = deg2tau(dt, nf, p0deg, p1deg)
        theta = (theta + np.pi) % np.pi - np.pi    # make sure the phase stays in the (-180.0, 180.0) interval  # p0deg = (p0deg + 180.0) % 360.0 - 180.0

        self.setCrntVal(key = ('.', 'theta', 0), val = theta)
        self.setCrntVal(key = ('.', 'tau', 0), val = tau)

    def adjust_phase(self, evalParsH=None, frqBlkIds=None, freqMask=None, mode='PhA', mw=512, cfun='LS', nhop=0, verbose=True):
        """Phase correction by adjusting the residual.
        Inputs:
        mode - choose which phase parameters to adjust ('PhA', 'Ph0', 'Ph1')
        """

        if verbose:
            print('Adjusting the phasing parameters, {}'.format(mode))

        # 1. Update the settings
        if evalParsH is None:
            evalParsH = self.crntParsH
        if frqBlkIds is None:
            frqBlkIds = self.steps[-1].frqBlkIds

        indxFreqByBlock = [ self._get_indxFreq(i) for i in frqBlkIds ]
        indxInRange = np.concatenate(indxFreqByBlock)
        # indxInRange = np.concatenate(tuple(self.freqBlocks[i].indxFreq(self.f) for i in frqBlkIds))
        dt = np.asscalar(self.t[1]-self.t[0])             # Dwell time

        # 2. Compute the model spectrum if necessary
        if self.zF is None:
            self.evaluate(evalParsH=evalParsH, frqBlkIds=frqBlkIds, freqMask=freqMask, autoKeys=[], returnSignals=True)
        self.zF_corr, self.bF_corr = None, None           # Reset the corrections for the model matrix and the baseline

        # Find the model signal
        ampl = np.array([self.getCrntVal(key=(name, 'ampl', 0)) for name in self.repRootNames if name not in self.xclRootNames])
        xF = self.zF[indxInRange, :].dot(ampl).reshape(-1,1) + self.bF[indxInRange]

        # Phase the measured data according to the values in the parameters
        theta = self.getCrntVal(key=('.', 'theta', 0))
        tau = self.getCrntVal(key=('.', 'tau', 0))
        ph = np.exp(-1j*2*np.pi * tau * (self.f[indxInRange]*self.c0-self.f0) - 1j*theta ).reshape((-1,1))   # The phasing term

        # Define the cost function to optimize (in terms of ph0 and ph1)
        costFuncPhase = lambda x : ph_cost(yF=self.yF[indxInRange]*ph, xF=xF, \
                        ph0=x[0], ph1=x[1], mw=mw, \
                        f=(self.f[indxInRange]*self.c0-self.f0)*dt, cfun=cfun )         # Frequency scale in fractions of the sampling frequrncy
        if mode == 'PhA':
            costFuncOpti = lambda x : costFuncPhase(x)[0]
            bounds, initVals = ((-0.5, 0.5), (-0.5, 0.5)), [0.0, 0.0]
        elif mode == 'Ph0':
            costFuncOpti = lambda x : costFuncPhase([x, 0.0])[0]
            bounds, initVals = ((-0.5, 0.5), ), [0.0]
        elif mode == 'Ph1':
            costFuncOpti = lambda x : costFuncPhase([0.0, x])[0]
            bounds, initVals = ((-0.5, 0.5), ), [0.0]

        # Call the optimization routine
        res = self._optimize(costFuncOpti, bounds, initVals, nhop=nhop, respectBounds=False, verbose=verbose)

        # Interpret the results
        if mode == 'PhA':
            ph0, ph1 = res.x
        elif mode == 'Ph0':
            ph0, ph1 = res.x[0], 0.0
        elif mode == 'Ph1':
            ph0, ph1 = 0.0, res.x[0]

        # Update and save the phasing parameters
        self.setCrntVal(key=('.', 'theta', 0), val=theta+ph0)
        self.setCrntVal(key=('.', 'tau', 0), val=tau+ph1*dt/(2*np.pi))

        #
        if verbose:
            print('Found values: ph0 = {:.4f}, ph1 = {:.4f}'.format(ph0, ph1))

    def adjust_residual(self, evalParsH=None, frqBlkIds=None, freqMask=None, mw=2048, verbose=True):
        """Correction of the model signals and the baseline to make the residual noise-like."""

        if verbose:
            print('Adjusting the baseline and residual.')

        # 1. Update the settings
        if evalParsH is None:
            evalParsH = self.crntParsH
        if frqBlkIds is None:
            frqBlkIds = self.steps[-1].frqBlkIds

        indxFreqByBlock = [ self._get_indxFreq(i) for i in frqBlkIds ]
        indxInRange = np.concatenate(indxFreqByBlock)
        # indxInRange = np.concatenate(tuple(self.freqBlocks[i].indxFreq(self.f) for i in frqBlkIds))
        dt = np.asscalar(self.t[1]-self.t[0])             # Dwell time

        # 2. Compute the model spectrum if necessary
        if self.zF is None or self.bF is None:
            self.evaluate(evalParsH=evalParsH, frqBlkIds=frqBlkIds, freqMask=freqMask, autoKeys=[], returnSignals=True)

        # Find the model signal
        ampl = np.array([self.getCrntVal(key=(name, 'ampl', 0)) for name in self.repRootNames if name not in self.xclRootNames])
        xF = self.zF[indxInRange, :].dot(ampl).reshape(-1,1) + self.bF[indxInRange]

        # Phase the measured data according to the values in the parameters
        theta = self.getCrntVal(key=('.', 'theta', 0))
        tau = self.getCrntVal(key=('.', 'tau', 0))
        ph = np.exp(-1j*2*np.pi * tau * (self.f[indxInRange]*self.c0-self.f0) - 1j*theta ).reshape((-1,1))   # The phasing term
        yFph=self.yF[indxInRange]*ph

        # Evaluate the phasing cost function to find the residual and baseline
        result, yFph, res, bln = ph_cost(yFph, xF, mw=mw)

        # Find the corrected amplitudes
        zT0, _ = getFID(self.T, [0.0], self.c0, self.f0, evalParsH, tau=0.0, xclRootNames=self.xclRootNames)           # Values of the first time-domain points for each model signal
        zF0 = np.sum(self.zF[indxInRange, :] - zT0.ravel()/(2*np.sqrt(len(self.f))), axis=0).real / np.sqrt(len(self.f))     # What the (restricted) models sum to; should be 1/2*zT0 if the entire frequency range
        corr_comp = np.where(zF0 > 0.01*sum(zF0))[0]           # Indices of components to correct, choose only large components

        bF0 = ampl.reshape(1,-1)*zT0.reshape(1,-1)/(2*np.sqrt(len(self.f)))             # Zero-order baselines
        Za = self.zF[indxInRange, :] * ampl.reshape(1,-1)
        posZa = Za - bF0      # Remove the constant baseline from the model signals
        absZa = np.abs(Za)**2
        C = absZa/np.sum(absZa, axis=1).reshape(-1,1)                     # Weights for redistributing the residual

        posZa_corr = posZa + res.real*C           # Corrected models without the constant baselines
        ampl_corr = ampl
        ampl_corr[corr_comp] = np.sum(posZa_corr[:, corr_comp].real, axis=0)/zF0[corr_comp].real.ravel()
        ampl_corr[corr_comp] *= np.nanmean(ampl[corr_comp].ravel()/np.sum(posZa[:, corr_comp].real, axis=0))     # Corrected amplitudes. Introduces a scaling factor to make the sum of Za approximately equal the intensities
        Za_corr = self.zF[indxInRange, :] * ampl_corr.reshape(1,-1)

        # Save the corrections and amplitudes
        reportedNames = [name for name in self.repRootNames if name not in self.xclRootNames]

        for name, val in zip(reportedNames, ampl_corr):
            self.setCrntVal(key=(name, 'ampl', 0), val=val)
        self.zF_corr, self.bF_corr = np.zeros(self.zF.shape), np.zeros(self.bF.shape)
        self.zF_corr[indxInRange, :] = (posZa_corr - Za_corr)               # Additive correction for the model signals
        self.zF_corr[np.ix_(indxInRange, ampl_corr.nonzero()[0])] /= ampl_corr[ampl_corr.nonzero()]       # Scale by the amplitudes. Only those where ampl_corr != 0
        self.bF_corr[indxInRange] = bln + np.sum(bF0)                       # Additive correction for the baseline

    def sample(self, parsKeys=None, autoKeys=None, frqBlkIds=None, freqMask=None, funcType=None, evaluatePriors=False, nwalkers=None, nsteps=None):
        """Samples the posterior distribution using the MCMC algorithm."""

        parsKeys, autoKeys = self._prepareKeys(parsKeys, autoKeys, frqBlkIds)
        result = {}
        self.smplDistF.clear()             # Clear the characteristics of marginal distributions
        reportedNames = [name for name in self.repRootNames if name not in self.xclRootNames]

        # If no parameters are set for sampling, just evaluate the marginal posterior
        if len(parsKeys) == 0:
            # Nothing to sample; just evaluate the function
            value, meta = self.evaluate(autoKeys=autoKeys, frqBlkIds=frqBlkIds, freqMask=freqMask, funcType=funcType, evaluatePriors=evaluatePriors, returnSignals=True)

            m_ampl = np.array(meta['ampl'][0]).reshape(-1, 1)
            S_ampl = meta['ampl'][1]
            nrep = len(m_ampl)*250     # Number of repeats for each case to sample the amplitudes from the Gaussian distributions
            indx = [i for i, name in enumerate(reportedNames) if self.getPrior(key=(name, 'ampl', 0)).distr == 'Gaussian' and (name, 'ampl', 0) not in parsKeys]       # Indices of amplitudes that were not sampled
            smpl = np.linalg.cholesky(S_ampl[np.ix_(indx, indx)]).dot(np.random.randn(len(indx), nrep)) + m_ampl[indx].reshape(-1,1)                  # Multivariate Gaussian random samples
            for i, id in enumerate(indx):
                key = (reportedNames[id], 'ampl', 0)
                result[key] = smpl[i,:]
                self.smplDistF[key] = smplSpec_from_data(result[key])
            sigma2 = np.array([[x] for x in meta['sigma2']])
            result[('.', 'sigma2', 0)] = sigma2[1] / (sigma2[0]-1)     # Mean estimator for sigma
            result[('.', 'sigma2', 'distr')] = sigma2
            result[('.', 'theta', 0)] = np.array([meta['theta']])

            print("Done. No non-marginalized parameters were requested.")
            return result

        # Define the objective function using a copy of the parameters dictionary
        evalParsH = copy.deepcopy(self.crntParsH)
        bounds = tuple((self.getPrior(key).min, self.getPrior(key).max) for key in parsKeys)
        initVals = [evalParsH[k[0]][k[1]][k[2]] for k in parsKeys]
        costFuncSmpl = lambda x : self.evaluate(updateFromFlat(evalParsH, parsKeys, x), parsKeys, autoKeys, frqBlkIds, freqMask, funcType, evaluatePriors)

        sampler = self._sample(costFuncSmpl, bounds, initVals, nwalkers, nsteps)

        # Form a table of results for the output
        # sampler.blobs has size nsteps x nwalkers
        # sampler.chain in nwalkers x nsteps x ndim
        # Want an output in the form nsamples x ndim
        flatchain = sampler.flatchain
        for i, key in enumerate(parsKeys):
            result[key] = flatchain[:, i]
            self.smplDistF[key] = smplSpec_from_data(result[key])

        m_ampl = np.array([blbWlkr['ampl'][0].ravel() for blbSmpl in sampler.blobs for blbWlkr in blbSmpl]).T     # Means of the amplitudes
        S_ampl = np.array([blbWlkr['ampl'][1] for blbSmpl in sampler.blobs for blbWlkr in blbSmpl]).T             # Covariance matrices of the amplitudes
        # Generate random samples of amplitudes
        nrep = 3     # Number of repeats for each case to sample the amplitudes from the Gaussian distributions
        indx = [i for i, name in enumerate(reportedNames) if self.getPrior(key=(name, 'ampl', 0)).distr == 'Gaussian' and (name, 'ampl', 0) not in parsKeys]       # Indices of amplitudes that were not sampled
        if len(indx) > 0:
            smpl = np.hstack([np.linalg.cholesky(np.squeeze(S_ampl[np.ix_(indx, indx, [i])])).dot(np.random.randn(len(indx), nrep)) \
                             + m_ampl[indx,i].reshape(-1,1) for i in range(S_ampl.shape[-1])])              # Multivariate Gaussian random samples
            for i, id in enumerate(indx):
                key = (reportedNames[id], 'ampl', 0)
                if key not in parsKeys:
                    result[key] = smpl[i,:]
                    self.smplDistF[key] = smplSpec_from_data(result[key])

        # Sigma2
        sigma2 = np.array([blbWlkr['sigma2'] for blbSmpl in sampler.blobs for blbWlkr in blbSmpl]).T
        result[('.', 'sigma2', 0)] = sigma2[1,:] / (sigma2[0,:]-1)     # Sample sigma
        result[('.', 'sigma2', 'distr')] = sigma2

        # Theta
        key = ('.', 'theta', 0)
        result[key] = np.array([blbWlkr['theta'] for blbSmpl in sampler.blobs for blbWlkr in blbSmpl]).ravel()
        self.smplDistF[key] = smplSpec_from_data(result[key])

        return(result)

    def sweep(self, key, lims=None, npts=50, reoptimize=False, frqBlkIds=None, freqMask=None, evaluatePriors=False):
        # Evaluates the posterior and computes the amplitudes while sweeping the parameter parKey in the range lims
        par = self.getPrior(key)     # Settings for the prior distribution of this parameter key
        if lims is None: lims = (par.min, par.max)
        x_arr = np.linspace(min(lims), max(lims), npts)
        lpst_arr = np.zeros(x_arr.shape)
        lpri_arr = np.zeros(x_arr.shape)
        evalParsH = copy.deepcopy(self.crntParsH)      # Make a copy of the parameter dictionary that will be used for evaluation
        for i, x in enumerate(x_arr):
            evalParsH[key[0]][key[1]][key[2]] = x
            lpst_arr[i], meta = self.evaluate(evalParsH=evalParsH, frqBlkIds=frqBlkIds, freqMask=freqMask, funcType=None, evaluatePriors=True, parsKeys=[key], autoKeys=None, robust=False)
            lpri_arr[i] = par.evalPrior(arg=x)

        llkl_arr = lpst_arr - lpri_arr

        crntVal = self.crntParsH[key[0]][key[1]][key[2]]

        return x_arr, llkl_arr, lpri_arr, lpst_arr, crntVal

    def _prepareKeys(self, parsKeys, autoKeys=None, frqBlkIds=None, customPriors=None, verbose=True, print_parameters=False):
        """Make sure that all parameter keys are relevant for the current Datum (e.g. no 4-tuple keys)."""

        # --------------------- Set the autoKeys ----------------------
        if autoKeys is None:
            # All possible autofittable parameters
            autoKeys = [('.', 'theta', 0), ('.', 'sigma2', 0), ('.', 'gamma', 0)] + \
                       [(name, 'ampl', 0) for name in self.repRootNames if name not in self.xclRootNames]

        # Keep only those parameters than can be autofitted because they have appropriate distributions
        autoKeys = set([key for key in autoKeys if self.isAutofittable(key, customPriors)])

        # -------------------- Set the parsKeys -----------------------
        parsKeys = set([]) if parsKeys is None else set(parsKeys)
        for key in list(parsKeys):
            if key[-2] == 'meta':
                parsKeys.remove(key)
                continue
            if len(key) == 4:
                parsKeys.remove(key)
                if key[0] in [self.parent.data.index(self), '.', None]:
                    parsKeys.update([key[1:]])

        # print('before', autoKeys, parsKeys)
        if len(autoKeys) + len(parsKeys) > 0 and len(frqBlkIds) > 0:
            good, bad = self.fittableParsKeys(frqBlkIds=frqBlkIds, customPriors=customPriors)

            # Remove non-fittabel parameters
            parsKeys = parsKeys.difference(bad)           # Set diffetence, elements in 'parsKeys' but not in 'bad'
            autoKeys = autoKeys.difference(bad)

            # Detremine which parameters can be marginalized and remove them from the list of fitted values
            parsKeys = parsKeys.difference(autoKeys)

        # print('after', autoKeys, parsKeys)
        parsKeys = sorted(list(parsKeys))

        if verbose:
            # print('\n')
            npar_auto = len(autoKeys)
            npar_fit = len(parsKeys)
            if npar_fit == 0:
                print("Nothing to fit; {} parameters inferred in closed form...".format(npar_auto))
            elif npar_fit == 1:
                print("Fitting one parameter; {} parameters inferred in closed form...".format(npar_auto))
            else:
                print("Fitting {} parameters; {} parameters inferred in closed form...".format(npar_fit, npar_auto))
            if print_parameters:
                print("Optimized parameters:")
                for key in parsKeys:
                    print("     {}".format(str(key)))
                print("Parameters inferred automatically:")
                for key in autoKeys:
                    print("     {}".format(str(key)))

        return parsKeys, autoKeys

    def modelled_signal(self, phased=True, baseline=False):
        # TODO: will be removed
        nt, nf = len(self.t), len(self.f)
        evalParsH = self.crntParsH
        tau, theta = 0.0, 0.0
        ampl = np.array([evalParsH[name]['ampl'][0] for name in self.repRootNames if name not in self.xclRootNames])
        if not phased:
            tau = evalParsH['.']['tau'][0]
            theta = evalParsH['.']['theta'][0]

        # Generate the signal
        zT, _ = getFID(self.T, self.t, self.c0, self.f0, evalParsH, tau)
        zT *= np.exp(1j*theta)

        # 1. Apply custom lineshape correction if defined
        if self.sT is not None:
            zT *= self.sT

        xT = np.dot(zT, np.array(ampl).reshape(-1,1))

        # Add the baseline
        if baseline and self.bF is not None:
            bT = np.fft.ifft(np.fft.ifftshift(self.bF, axes=0), axis=0)[:nt] * np.sqrt(nf)
            xT += bT

        """# Apply the inferred adaptive window
        if self.sF is not None:
            nw = len(self.sF)
            wF = self.sF * np.sqrt(nf)
            wFull = np.pad(wF.ravel(), (math.ceil((nf-nw)/2), math.floor((nf-nw)/2)), 'constant', constant_values=0).reshape(-1,1) # Zero-pad wF before taking the iFFT
            wT = np.fft.ifft(np.fft.ifftshift(wFull, axes=0), axis=0)[:nt]
            xT = xT * wT"""

        # Find the spectrum
        xF = np.fft.fftshift(np.fft.fft(xT, nf, axis=0), axes=0) / np.sqrt(nf)

        return xT, xF

    def residual_spectrum(self):
        """Computes the residual spectrum after model fitting."""
        # Phase the data
        ph = np.exp(-1j*2*np.pi * self.crntParsH["."]["tau"][0] * (self.f*self.c0-self.f0) - 1j*self.crntParsH["."]["theta"][0] ).reshape(-1,1)
        yFph = self.yF * ph

        return yFph - self.modelled_signal(baseline=True)[1]

    def remove(self):
        """Removes itself from the Series"""
        self.parent.data.remove(self)

    def report(self, full=True):
        """Prints out the current values and results."""
        s = self.selfID()    # Self ID of the Datum
        if full:
            pass
        else:
            tab = [[self.name] + [np.linalg.norm(fff.yT)] + fff.crntParsH["."]["ampl"] for fff in self._crnt.data]   # Update the table of results
            head = ['Filename'] + ['Total intensity'] + [node.name for node in self.wsp.T.repRoots()]
            with open('_results.txt', 'w') as fout:
                print(tabulate.tabulate(tab, headers=head), file=fout)        # write results to a text file ...
            print(tabulate.tabulate(tab, headers=head))    # ... and show on the screen

    def plot_distributions(self, keys, ax, frqBlkIds=None, lims=None, npts=None, levels=10):
        """Plots the 1D or 2D distributions and likelihood function."""

        if len(keys) != 2:
            if npts is None: npts = 75
            x_arr, llkl_arr, lpri_arr, lpst_arr, crntVal = self.evalForPlot(keys, frqBlkIds, lims, npts)
            ax.plot(x_arr, llkl_arr, label="Likelihood fnc.")
            if np.any(lpri_arr-np.mean(lpri_arr)):
                ax.plot(x_arr, lpri_arr, label="Prior distribution")
                ax.plot(x_arr, lpst_arr, label="Posterior distr.")
            ax.axvline(x=crntVal, linestyle='dashed', color='red')
            ax.set_title("Distributions for {}".format(str(keys)))
            ax.legend(loc=0)
        else:
            if npts is None: npts = 25
            grid, llkl_arr, lpri_arr, lpst_arr, crntVal = self.evalForPlot2D(keys, frqBlkIds, lims, npts)

            ax[0].contour(*np.meshgrid(grid[0], grid[1]), llkl_arr)
            ax[0].set_title('Log-likelihood')
            if np.any(lpri_arr-np.mean(lpri_arr)):
                ax[1].contour(*np.meshgrid(grid[0], grid[1]), lpri_arr)
            ax[1].set_title('Prior')
            ax[2].contour(*np.meshgrid(grid[0], grid[1]), lpst_arr)
            ax[2].set_title('Posterior')
            ax[2].plot(*crntVal, 'ro')
            ax[1].set_xlabel(str(keys[0]))
            ax[0].set_ylabel(str(keys[1]))

    def plot(self, ax_main, ax_residual=None, showRanges='all', showComponents=False, showLegend=True, returnSignals=False, real=True):
        """Plots the dataset using a matplotlib Figure."""

        f, yFph, xF, zF, bF = self.signals_for_plot()

        # Plot measured data
        ax_main.plot(f, yFph.real if real else yFph.imag, '-', color=(0,0.58,0.86), linewidth=1.5, label='')

        # Plot the model components
        reportedNames = [name for name in self.repRootNames if name not in self.xclRootNames]
        if showComponents and zF is not None:
            zF = (zF + 0*bF)
            for i, node in enumerate(reportedNames):
                ax_main.plot(f, zF[:, i].real if real else zF[:, i].imag, '-', linewidth=0.5, color=config.colrseq[i], label=node)

        # Plot the fitted model
        if xF is not None:
            ax_main.plot(f, xF.real if real else xF.imag, '-', color='r', label='')

            # Plot the residuals
            if ax_residual is not None:
                rF = np.where(xF != 0, yFph-xF, 0)
                rF = rF.real if real else rF.imag
                ax_residual.plot(f, rF, '-', color='darkkhaki')
                # rmsResidual += np.sqrt(np.nanmean(np.abs(rF)**2))

        # Show the residuals plot below the graph
        # Set ticks and labels
        #plt.setp(ax_main.get_xticklabels(), visible=False)
        ax_main.set_xlabel('')
        ax_main.ticklabel_format(scilimits=(-3,3))
        if ax_residual is not None:
            ax_residual.ticklabel_format(scilimits=(-3,3))
            ax_residual.set_xlabel('Chemical shift, ppm', horizontalalignment='right', x=1.0)
            # # Show RMS of the residual
            # ax_residual.text(0.01, 0.92, "RMS = {:.4g}".format(rmsResidual), fontsize=10,
            #             horizontalalignment='left', verticalalignment='top', transform = ax_residual.transAxes)
        else:
            ax_main.set_xlabel('Chemical shift, ppm', horizontalalignment='right', x=1.0)

        # Plot optimization limits
        if showRanges:
            for i, blk in enumerate(self.freqBlocks):
                if showRanges == 'all':
                    ax_main.axvspan(blk.min - dref_chsh, blk.max - dref_chsh, alpha=0.2 if i in self.steps[-1].frqBlkIds else 0.05, facecolor='yellow')
                elif showRanges == 'active' and i in self.steps[-1].frqBlkIds:
                    ax_main.axvspan(blk.min - dref_chsh, blk.max - dref_chsh, alpha=0.2, facecolor='yellow')

        if showLegend: ax_main.legend(loc=0)

        # Set the updated limits
        ax_main.relim()    # recompute the ax.dataLim
        ax_main.margins(0, 0.05)    # x and y margins in percentages
        ax_main.autoscale()    # update ax.viewLim using the new dataLim
        if ax_main.get_xlim()[1] > ax_main.get_xlim()[0]: ax_main.invert_xaxis()

        if returnSignals: return f, yFph, xF, zF

    def signals_for_plot(self, frqBlkIds=None, showComponents=False, onlyInRange=False):
        """Computes the signals for plotting. Applies subsampling to the parts of the signals that are out ouf the fitting ranges."""

        if frqBlkIds is None:
            frqBlkIds = self.steps[-1].frqBlkIds

        # Phase the data
        theta, tau = self.getCrntVal(key=('.', 'theta', 0)), self.getCrntVal(key=('.', 'tau', 0))
        ph = np.exp(-1j*2*np.pi * tau * (self.f*self.c0-self.f0) - 1j*theta ).reshape(-1,1)
        yFph = self.yF * ph
        # Flip the phase if needed
        if np.median(yFph.real) - np.min(yFph.real) > np.max(yFph.real) - np.median(yFph.real):
            # Possibly also change the value in the model
            # theta = (theta + np.pi + np.pi) % (2 * np.pi) - np.pi
            # self.setCrntVal(key=('.', 'theta', 0), val=theta)
            yFph *= -1

        # Subsample out-of-range parts of the spectrum
        inRange, outRange = splitFreq([ minmaxTuple(self.freqBlocks[blk].min, self.freqBlocks[blk].max) for blk in frqBlkIds ])
        if onlyInRange: outRange.clear()
        dref_chsh = self.getGlobalChshVal() if config.DISPL_ShiftToReference else 0.0           # Find global chemical shift that will be used to shift the ppm scale on the graph
        f = self.f - dref_chsh
        rmsResidual = 0.0
        allRange = sorted(inRange+outRange, key=lambda x : x[0])
        supsRatio = ceil(yFph.size / (2**12))   # Subsampling ratio; take no more than 2^12 points
        allIndx = np.concatenate( [np.arange(r.imin(f), r.imax(f), supsRatio if r in outRange else 1) for r in allRange] )

        f, yFph = f[allIndx, :], yFph[allIndx, :]

        zF, xF, bF = None, None, None
        if self.zF is not None:
            ampl = np.array([self.crntParsH[name]['ampl'][0] for name in self.repRootNames if name not in self.xclRootNames]).reshape(1, -1)
            zF = self.zF * ampl if self.zF_corr is None else (self.zF + self.zF_corr)*ampl
            xF = zF.sum(1).reshape(-1,1)
            xF, zF = xF[allIndx, :], zF[allIndx, :]
            if self.bF is not None:
                bF = self.bF[allIndx, :] if self.bF_corr is None else self.bF[allIndx, :] + self.bF_corr[allIndx, :]
                xF += bF

        return f, yFph, xF, zF, bF

    def stems_for_plot(self):
        """Computes all stem lines to be displayed on a plot. The result is a list of dictionaries with number of elements == number of reported nodes. In each dictionary: key - (name, chshQD, i), val - a tuple, (chshQD, list of stems chsh for the specific chshQD, list of corresponding stems intensities)."""

        allStems = []     # Dictionary that stores references to all stem lines
        mdldPeaks = collectPeaks(self.T, self.c0, self.crntParsH)
        dref_chsh = self.getGlobalChshVal() if config.DISPL_ShiftToReference else 0.0           # Find global chemical shift that will be used to shift the ppm scale on the graph

        # for i, name in enumerate([name for name in self.repRootNames if name not in ['Water', 'Chloroform'] ]):         #
        for i, name in enumerate(self.repRootNames):
            if name not in self.xclRootNames:
                stems_i = {}
                for stemKey, val in mdldPeaks[name].items():   # Loop over the leaves
                    parsKey = peakName2parsKey(stemKey)
                    freq = [pk.chsh - dref_chsh for pk in val]
                    intn = [np.abs(pk.intn) for pk in val]
                    stems_i[parsKey] = (self.getCrntVal(key=parsKey), freq, intn)
                allStems.append(stems_i)

        return allStems

    def evalForPlot(self, key, frqBlkIds=None, lims=None, npts=75):
        """Returns an array of argument values and the values of log likelihood, prior, and posterior."""
        result = {}
        reportedNames = [name for name in self.repRootNames if name not in self.xclRootNames]
        na = len(reportedNames)
        par = self.getPrior(key)     # Settings for the prior distribution of this parameter key
        if lims is None: lims = (par.min, par.max)
        x_arr = np.linspace(min(lims), max(lims), npts)
        lpst_arr = np.zeros(x_arr.shape)
        lpri_arr = np.zeros(x_arr.shape)
        m_ampl_arr = np.zeros((na, npts))
        S_ampl_arr = np.zeros((na, na, npts))
        evalParsH = copy.deepcopy(self.crntParsH)      # Make a copy of the parameter dictionary that will be used for evaluation
        for i, x in enumerate(x_arr):
            evalParsH[key[0]][key[1]][key[2]] = x
            lpst_arr[i], meta = self.evaluate(evalParsH=evalParsH, frqBlkIds=frqBlkIds, funcType=None, evaluatePriors=True, parsKeys=[key], robust=False)
            lpri_arr[i] = par.evalPrior(arg=x)
            m_ampl_arr[:, i] = np.array(meta['ampl'][0]).ravel()
            S_ampl_arr[..., i] = meta['ampl'][1]     # Add the third (singular) dimension corresponding to the number of samples

        llkl_arr = lpst_arr - lpri_arr

        crntVal = self.crntParsH[key[0]][key[1]][key[2]]

        return x_arr, llkl_arr, lpri_arr, lpst_arr, crntVal

    def evalForPlot2D(self, keys, frqBlkIds=None, lims=None, npts=25):
        ndim = 2

        if len(keys) != ndim: raise RuntimeError('Only 2 dimensional inputs are supported.')

        pars = [self.getPrior(key) for key in keys]

        if lims is None or len(lims) != ndim:
            lims = [(par.min, par.max) for par in pars]

        # Form a grid of points at which the function will be evaluated
        x_arr = np.linspace(min(lims[0]), max(lims[0]), npts)
        y_arr = np.linspace(min(lims[1]), max(lims[1]), npts)

        evalParsH = copy.deepcopy(self.crntParsH)
        def func(x, y):
            evalParsH[keys[0][0]][keys[0][1]][keys[0][2]] = x
            evalParsH[keys[1][0]][keys[1][1]][keys[1][2]] = y
            # Evaluate the function skipping the datasets that are not present in parsKeys
            lpst = self.evaluate(evalParsH, parsKeys=keys, frqBlkIds=frqBlkIds, funcType=None, evaluatePriors=True, robust=False)[0]
            lpri = self._fnc_prior(evalParsH, parsKeys=keys) + self._fnc_joint(evalParsH)
            return lpst, lpri

        lpst_arr, lpri_arr = np.vectorize(func)(*np.meshgrid(x_arr, y_arr, sparse=True))     # Return a table of f(x, y)
        llkl_arr = lpst_arr - lpri_arr
        crntVal = (self.crntParsH[keys[0][0]][keys[0][1]][keys[0][2]],
                   self.crntParsH[keys[1][0]][keys[1][1]][keys[1][2]])

        return (x_arr, y_arr), llkl_arr, lpri_arr, lpst_arr, crntVal

    def stats(self):
        """Returns a dictionary of statistics about the Datum."""

        return {'nT' : len(self.t),
                'nF' : len(self.parent.f),
                'nF_adap' : len(self.f),
                'nF_opti' : sum( [len(self._get_indxFreq(i)) for i in self.steps[-1].frqBlkIds] )}   # Find the number of points in the active optimization ranges

#### Utility functions #####

def gmm_pdf(x, m, S, w=None):
    """Returns an analytical fucntion for a mixture of Gaussians (nd<=2)
    m - matrix of means, ndim_x_ncmp
    S - covariance matrices, ndim_x_ndim_x_ncmp
    w - vector of weights (optional), 1_x_ncmp,
    lims - limits for plotting
    npts - number of points to evaluatethe function at
    """

    try:
        ndim, ncmp = m.shape     # Number of dimensions and number of components
    except ValueError:
        # If m and S are flattened arrays
        ndim, ncmp = 1, m.size
        m, S = m.reshape(1, -1), S.reshape(1, 1, -1)

    if ndim > 2:
        raise Exception('Only 1- or 2-dimensional arrays are curently supported.')

    if w is None:
        w = 1/ncmp * np.ones((ncmp,1))

    # Compute the values of the distribution on the grid points
    try:
        F = np.zeros(x.shape)
    except AttributeError:
        F = 0.0

    for i in range(ncmp):
        mi = m[...,i]
        Si = S[...,i]
        diff = x-mi.reshape(-1,*[1]*ndim)
        F += w[i] * 1/np.sqrt(np.linalg.det(np.pi*Si)) * np.exp(-(np.inner(diff.T, np.linalg.inv(Si)).T * diff).sum(0))

    return F

def gmm_hdr(m, S, w=None):
    # Computes the highest density regions for the Gaussian mixture model given a density function func and the precision level alpha
    pass

def plotGaussianMixture(m, S, w=None, lims=None, npts=100):
    """Plots a pdf for a mixture of nc Gaussian nd-dimensional components (nd<=2)
    m - matrix of means, ndim_x_ncmp
    S - covariance matrices, ndim_x_ndim_x_ncmp
    w - vector of weights (optional), 1_x_ncmp,
    lims - limits for plotting
    npts - number of points to evaluatethe function at
    """

    try:
        ndim, ncmp = m.shape
    except ValueError:
        # If m and S are flattened arrays
        ndim, ncmp = 1, m.size
        m, S = m.reshape(1, -1), S.reshape(1, 1, -1)

    if ndim > 2:
        raise Exception('Only 1- or 2-dimensional arrays are curently supported.')

    if w is None:
        w = 1/ncmp * np.ones((ncmp,1))

    if lims is None or len(lims) != ndim:
        lims = [None]*ndim
        for i in range(ndim):
            stdv = np.sqrt(np.amax(S[i,i,...]))
            lims[i] = (np.amin(m[i,...])-3*stdv, np.amax(m[i,...])+3*stdv)

    # Form a grid of points at which the function will be evaluated
    grid = np.array(np.meshgrid(*[np.linspace(lim[0], lim[1], npts) for lim in lims]))

    # Compute the values of the distribution on the grid points
    F = np.zeros((npts,)*ndim)
    for i in range(ncmp):
        mi = m[...,i]
        Si = S[...,i]
        diff = grid-mi.reshape(-1,*[1]*ndim)
        F += w[i] * 1/np.sqrt(np.linalg.det(np.pi*Si)) * np.exp(-(np.inner(diff.T, np.linalg.inv(Si)).T * diff).sum(0))

    return grid, F

def plotInverseGammaMixture(alpha, beta, w=None, lims=None, npts=100):
    """Plots a pdf for a mixture of nc Gaussian nd-dimensional components (nd<=2)
    alpha - matrix of the alpha parameter, ndim_x_ncmp
    beta - matrix of the beta parameters, ndim_x_ndim_x_ncmp
    w - vector of weights (optional), 1_x_ncmp,
    lims - limits for plotting
    npts - number of points to evaluatethe function at
    """

    ncmp = len(alpha)

    if w is None:
        w = 1/ncmp * np.ones((ncmp,1))

    # Form a grid of points at which the function will be evaluated
    mean = beta/(alpha-1)
    stdv = np.where(alpha>2, beta**2 / (alpha-1)**2 / (alpha-1), alpha)
    if lims is None:
        lims = (max(1e-245, np.amin(mean-stdv)), np.amax(mean+stdv))
    grid = np.linspace(lims[0], lims[1], npts)

    # Compute the values of the distribution on the grid points
    F = np.zeros((npts,))
    for i in range(ncmp):
        a, b = alpha[i], beta[i]
        F += w[i] * np.exp(a*np.log(b) - np.log(scipy.special.gamma(a)) - (a+1)*np.log(grid) - b/grid)

    return grid, F

def reportMCMC(samples):
    ampl, vars, names = [], [], []
    for key, val in samples.items():
        x = samples[key]
        if key[-2] == 'ampl' and key[-3] not in ['Water', 'Chloroform']:
            ampl.append(np.mean(x))
            vars.append(np.var(x))
            names.append(key[-3])
    ampl = np.array(ampl).ravel()
    vars = np.array(vars).ravel()

    m_tot = np.sum(ampl)      # Total intensity
    v_tot = np.sum(vars)      # Total variance of the intensity estimate
    mfrac = ampl / m_tot
    confi = 1.96 * mfrac * np.sqrt(vars/(ampl**2) + v_tot/(m_tot**2))

    tab = [[n, "{:.4f}".format(m), "{:.2f}".format(np.sqrt(v)), " {:.4f} {}{:.4f}".format(mf, chr(177), ci)] for n, m, v, mf, ci in sorted(zip(names, ampl, vars, mfrac, confi))]
    tab.append(['Total', "{:.4f}".format(np.asscalar(m_tot)), "{:.2f}".format(np.asscalar(np.sqrt(v_tot))), ""])
    head = ['name', 'mean', 'stdv', 'mole frac.']
    print('\n')
    print(tabulate.tabulate(tab, headers=head))

def load_workspace(filename):
    """Loads a Worksapce saved in file fname."""
    with open(filename, 'rb') as fp:
        dataPack, GUIsettings = dill.load(fp)
    wsp = Workspace()       # Define a new Workspace object
    wsp.unpack(dataPack)    # Unpack the loaded data into it

    # set the global config settings
    if GUIsettings is not None:
        try:
            stngConfig = GUIsettings.pop('_config')
            config.from_dict(config, stngConfig)
        except KeyError:
            print('Using default global settings.')

    return wsp, GUIsettings

def save_workspace(filename, wsp, GUIsettings=None):
    """Saves the workspace wsp into file filename."""
    if GUIsettings is None:
        GUIsettings = {'ax0Limits': {'ylim': (0.0, 1000.0),
                                     'xlim': (10.0, 0.0)},
                       'autoPhase': False, 'startFromPars': 'current',
                       '_view': {'hiddenTreeViewNodes': [], 'stepsEditText': 'AAAAAA'},
                       '_config': {'SAMPL_varEstimator': 'robust', 'OPTIM_method': 'L-BFGS-B', 'QD_RerunQDchshThreshold': 0.1, 'OPTIM_startFrom': 'current', 'SAMPL_funcType': 'LS', 'OPTIM_niterSuccess': 5, 'MODEL_ShapeKernelSize': 13, 'OPTIM_maxBasinhoppingSteps': 3, 'QD_AggregatePeaksThreshold': 0.5, 'DISPL_ShiftToReference': True},
                       'autoPick': False, 'ax1Limits': None, 'ax2Limits': None}
    dataPack = wsp.pack()
    with open(filename, 'wb') as fp:
        dill.dump([dataPack, GUIsettings], fp)

def saveFID(xT, c0, f0, dt, tau=0., fname='fid'):
    """Saves FID signal y in the Spinsolve format in a newly created folder."""
    # Create a new directory of it does not exist
    if not os.path.exists(fname):
        os.makedirs(fname)

    # Reshape and reorganize the data as T-T-T-...-T-R-I-R-I-...-R-I
    n1, n2, n3, n4 = xT.size, 1, 1, 1   # Dimensions of the data array
    xT = np.hstack([xT.real.reshape(-1,1), -xT.imag.reshape(-1,1)]).ravel()
    data = np.concatenate( [np.linspace(0., (n1-1)*dt, n1), xT.ravel()] ).astype('single')

    # Save the binary data
    with open(os.path.join(fname, 'data.1d'), 'wb') as fid:
        fid.write(np.array([1347571539, 1145132097, 1446063665, 504, n1, n2, n3, n4], dtype='int32').tobytes())
        fid.write(data.tobytes())

    # Save the parameter file
    with open(os.path.join(fname, 'acqu.par'), 'w') as fid:
        fid.write('Solvent                   = ""\n')
        fid.write('Sample                    = ""\n')
        fid.write('startTime                 = {}\n'.format(time.ctime()))
        fid.write('acqDelay                  = {:0.16f}\n'.format(tau*1e+06))     # Ringdown delay in ms
        fid.write('b1Freq                    = {:0.16f}\n'.format(c0))            # B1 frequency in MHz
        fid.write('bandwidth                 = {:0.16f}\n'.format((1/dt)/1000))       # Sweep bandwidth in kHz
        fid.write('dwellTime                 = {:0.16f}\n'.format(1000*dt))       # Dwell time in ms
        fid.write('experiment                = "1D"\n')
        fid.write('expName                   = "1D"\n')
        fid.write('nrPnts                    = {:d}\n'.format(n1))
        fid.write('nrScans                   = 1\n')
        fid.write('repTime                   = 0\n')                              # Repetiotion time in ms
        fid.write('rxChannel                 = "1H"\n')
        fid.write('rxGain                    = 0\n')                              # Reciever gain in dB
        fid.write('lowestFrequency           = {:0.16f}\n'.format(-(1/dt)/2+f0))  # Lowest frequency in Hz
        fid.write('totalAcquisitionTime      = 42\n')                             # Total acquisition time in sec
        fid.write('graphTitle                = "1D-1H-"StandardScan""\n')
        fid.write('userData                  = ""\n')
        fid.write('90Amplitude               = 0\n')                              # Amplitude of the 90-degree pulse in dB
        fid.write('pulseLength               = 0\n')                              # Pulse length in ms
        fid.write('Protocol                  = "1D PROTON"\n')
        fid.write('Options                   = "Scan(StandardScan)"\n')
        fid.write('Spectrometer              = "Python"\n')
        fid.write('Software                  = "Python"')

def cut_roi_ver2(xT, lims, c0, f0, dt, subsample=True):
    """Applies a bandpass filter to the signal to cut a region of interest within the limits lims.
    Returns the resulting filtered signal along with new fo and dt parameters if the signal was subsampled.
    A second version of the cut_roi function."""
    nt = len(xT)
    t = np.linspace(0.0, dt*(nt-1), nt).reshape(xT.shape)

    # Center the input signal wrt to the specified lims
    f0_new = c0*np.sum(lims) / 2.0
    cutoff = c0*(np.max(lims) - np.sum(lims)/2.0) / (1/(2*dt))           # Boundary of the limit in chsh as a fraction of the total chsh range
    xT = xT * np.exp(1j*2*np.pi * (f0-f0_new) * t)

    # Filter and subsample all at once
    subs_factor = int(np.floor(0.99 / cutoff))
    dt_new = dt*subs_factor
    yT = scipy.signal.decimate(xT, subs_factor, axis=0).reshape(-1,1)

    return yT, f0_new, dt_new

def cut_roi(xT, lims, c0, f0, dt, subsample=True):
    """Applies a bandpass filter to the signal to cut a region of interest within the limits lims. Returns the resulting filtered signal along with new fo and dt parameters if the signal was subsampled."""
    nt = len(xT)
    t = np.linspace(0.0, dt*(nt-1), nt).reshape(xT.shape)

    # Center the input signal wrt to the specified lims
    f0_new = c0*np.sum(lims) / 2.0
    cutoff = c0*(np.max(lims) - np.sum(lims)/2.0) / (1/(2*dt))           # Boundary of the limit in chsh as a fraction of the total chsh range
    xT = xT * np.exp(1j*2*np.pi * (f0-f0_new) * t)

    ## Define the LP filter and apply it to the signal
    b, a = scipy.signal.butter(10, cutoff, 'low')
    yT = scipy.signal.filtfilt(b, a, xT, axis=0).reshape(-1,1)

    # Subsample
    up_factor, down_factor = 100, int(np.floor(100 / cutoff))
    dt_new = dt/up_factor*down_factor
    yT = scipy.signal.resample_poly(yT, up_factor, down_factor).reshape(-1,1)

    return yT, f0_new, dt_new

def splitFreq(inRange):
    """Given a list of freqSpec tuples, divides the frequency range -inf to +inf into lists of disjoint intervals: inRange and outRange by merging overlapping optimization ranges."""

    if inRange:
        inRange = mergeFreq(inRange)      # Merged and sorted list of freqRanges
        outRange = []
        if not np.isinf(inRange[0].min):
            lwr = -np.inf
            upr = inRange[0].min
            outRange.append(minmaxTuple(min=lwr, max=upr))
        for i in range(len(inRange)-1):
            lwr = inRange[i].max
            upr = inRange[i+1].min
            outRange.append(minmaxTuple(min=lwr, max=upr))
        if not np.isinf(inRange[-1].max):
            lwr = inRange[-1].max
            upr = np.inf
            outRange.append(minmaxTuple(min=lwr, max=upr))
    else:
        outRange = [minmaxTuple(-np.inf, np.inf)]  # Infinite interval

    return inRange, outRange

def mergeFreq(intervals):
    """
    Merge oevrlapping intervals. Based on https://codereview.stackexchange.com/questions/69242/merging-overlapping-intervals.
    A simple algorithm can be used:
    1. Sort the intervals in increasing order
    2. Push the first interval on the stack
    3. Iterate through intervals and for each one compare current interval
       with the top of the stack and:
       A. If current interval does not overlap, push on to stack
       B. If current interval does overlap, merge both intervals in to one
          and push on to stack
    4. At the end return stack
    """
    sorted_by_lower_bound = sorted(intervals, key=lambda tup: tup.min)
    merged = []

    for higher in sorted_by_lower_bound:
        if not merged:
            merged.append(higher)
        else:
            lower = merged[-1]
            # test for intersection between lower and higher:
            # we know via sorting that lower[0] <= higher[0]
            if higher.min <= lower.max:
                upper_bound = max(lower.max, higher.max)
                merged[-1] = minmaxTuple(min=lower.min, max=upper_bound)  # replace by merged interval
            else:
                merged.append(higher)
    return merged

#@profile
def whitsm(y, lmda=5.0):
    """Whittaker smoother. See: https://gist.github.com/zmeri/3c43d3b98a00c02f81c2ab1aaacc3a49"""
    m = len(y)

    x = np.arange(1, m+1, 1)
    dx0 = (x[1:-1]-x[:-2]).ravel()
    dx1 = (x[2:]-x[1:-1]).ravel()
    D = sp.sparse.diags([2/dx0/(dx0+dx1), np.append(-2/dx0/dx1, 0.0), 2/dx1/(dx0+dx1)], np.array([0, 1, 2]), shape=(m-2, m)) * np.mean(np.abs(dx0)**2)

    E = sp.sparse.identity(m)
    #d1 = -1 * np.ones((m-3),dtype='d')
    #d2 = 3 * np.ones((m-2),dtype='d')
    #d3 = -3 * np.ones((m-2),dtype='d')
    #d4 = np.ones((m-3),dtype='d')
    #D = sp.sparse.diags([d1,d2,d3,d4],[0,1,2,3], shape=(m-3, m))
    #z = sp.sparse.linalg.cg(E + (10**lmda) * (D.conj().transpose()).dot(D), y, maxiter=2500)
    z = sp.sparse.linalg.cg(E + (10**lmda) * (D.conj().T).dot(D), y, maxiter=5000)

    return z[0]

def make_causal(xF):
    """Returns the causal part of a frequency domain data xF."""
    # Upsample the spectrum
    xF = scipy.signal.resample_poly(xF, up=2, down=1)

    # Compute the Hilbert transform
    xFr_h = scipy.signal.hilbert(xF.real, axis=0).conj()
    xFi_h = 1j*(scipy.signal.hilbert(xF.imag, axis=0).conj())
    xF_h = xFi_h.real + 1j*xFr_h.imag

    # Compute the time-domain signal and keep only its first half
    xTc = np.fft.ifft(np.fft.ifftshift(xF_h, axes=0), axis=0)[:len(xF_h)/2]
    xFc = np.fft.fftshift(np.fft.fft(xTc, axis=0), axes=0)
    return xTc, xFc

#@profile
def leastSquares(Z, y, m0=None, S0=None, Gy=None, lockedPhase=True, indxPositive=None, robust=True):
    """Solves a phased-constrained complex-valued least-squares problem, y=Zx for x; nb - number of baseline terms (columns in the end of Z). theta=None - the phase will be determined from the data. Gy - covariance matrix of noise (or the diagonal vecotr of that matrix)"""
    nz = Z.shape[1]      # Number of dimensions
    if lockedPhase and indxPositive is None:
        indxPositive = list(range(nz))
    ampl = np.zeros((nz,1))

    # 2. Set up the (Gaussian) priors
    if m0 is None: m0 = np.zeros((k, 1))
    if S0 is None: S0 = 1e+42 * np.identity(k)

    # Find which dimensions have priors with infinite or zero variance and invert the covariance matrix
    indx_inf = np.where(np.diag(S0) == np.inf)[0]        # Indices of components with infinite-variance (non-informative) intensities
    indx_fixed = np.where(np.diag(S0) == 0)[0]           # Indices of components with fixed (zero-variance) intensities
    indx_variable = np.where(np.diag(S0) != 0)[0]        # Indices of components with variable intensities
    indx_zero = []                                       # Indices of components with zero intensities
    iS0 = np.zeros((nz, nz))
    iS0[indx_variable[:,None], indx_variable] = np.linalg.inv(S0[indx_variable[:, None], indx_variable])           # Invert the part that that doesn't have zeros on the diagonal as usual
    iS0[indx_fixed, indx_fixed] = 1e+42   # np.inf                                                             # Substitute the rest of diagonal values with a very large number

    # 3.
    if Gy is None:
        # No weighting matrix (assume identity)
        ZG = Z.conj().T
        yG = y.conj().T
    elif Gy.ndim==1 or (Gy.ndim==2 and (Gy.shape[0]==1 or Gy.shape[1]==1)):
        # Weighting matrix G is diagonal and is defined by the vector
        iGy = 1 / Gy.reshape(1, -1)
        ZG = Z.conj().T * iGy
        yG = y.conj().T * iGy
    else:
        # G is a full matrix
        iGy = np.linalg.inv(Gy)
        ZG = Z.conj().T.dot(iGy)
        yG = y.conj().T.dot(iGy)
    ZZ = ZG.dot(Z)
    Zy = ZG.dot(y)
    if np.linalg.matrix_rank(ZZ) < nz:
        ZZ += 0.000001*np.identity(nz)

    if lockedPhase:                 # Locked phase - real amplitudes
        while True:
            iSc = (iS0 + ZZ).real
            Sc = np.linalg.inv(iSc)
            # Estimate theta to maximize the posterior
            theta = np.asscalar( 0.5*np.angle(np.dot(Zy.T, np.dot(Sc, Zy))) )
            # TODO!!! This should also depend on priors over amplitudes (i.e. S0 and m0)
            mc = Sc.dot( (Zy*np.exp(-1j*theta) + np.dot(iS0,m0)).real )

            # Find which components (if any) have negative intensities and set them to 0.0
            mc[indxPositive] = np.maximum(mc[indxPositive], 0.0) if mc[indxPositive].sum() > 0 else np.minimum(mc[indxPositive], 0.0)     # Discard negative values in mc but keep the sign for baseline components
            indx_zero = [ indxPositive[i] for i in np.where(mc[indxPositive] == 0.0)[0] ]     # Indices of elements currently set to zeros
            if len(indx_zero) == 0:
                # If all components have the same sign
                if mc[indxPositive].sum() < 0:      # Make sure that all amplitudes are positive
                    mc = - mc
                    theta = theta + np.pi
                mc[indx_fixed] = m0[indx_fixed]     # Replace intensities with fixed vcalues if neecessary (where variance is 0)
                break
            else:
                # Exclude the components with negative intensities from computation and update their priors accordingly
                indx_fixed = np.append(indx_fixed, indx_zero)
                indx_variable = np.setdiff1d(indx_variable, indx_zero)
                indxPositive = np.setdiff1d(indxPositive, indx_zero)
                m0[indx_zero] = 0.0
                iS0[indx_zero, indx_zero] = 1e+42
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        mc = mc* np.exp(1j*theta)
    else:
        # LS problem with unconstrained phase (if complex)
        theta = None
        iSc = iS0 + ZZ
        Sc = np.linalg.inv(iSc)
        mc = np.dot(Sc, (Zy+np.dot(iS0, m0)))
        mc[indx_fixed] = m0[indx_fixed]     # Replace values if neecessary (where variance is 0)

    # Find robust variance estimators
    if robust:
        r = y - Z.dot(mc)    # the residual
        r2 = np.abs(r.reshape(1,-1))**2
        #print( "sum r2 = {}".format(np.asscalar(r2.sum(axis=1))/r2.size) )
        ZrZ = (ZG * r2).dot(ZG.conj().T)
        if np.linalg.matrix_rank(ZrZ) < nz:
            ZrZ = ZrZ + 0.000001*np.identity(nz)
        Sr = Sc.dot( (iS0 + ZrZ).real ).dot(Sc)
    else: Sr = None

    Q = yG.dot(y) + np.dot(np.dot(m0[indx_variable].conj().T, iS0[indx_variable[:,None], indx_variable]), m0[indx_variable]) - np.dot(np.dot(mc[indx_variable].conj().T, iSc[indx_variable[:,None], indx_variable]), mc[indx_variable])
    Q = max(np.asscalar(Q.real), 0.0)
    return mc, Sc, Sr, Q

def setattrs(_self, **kwargs):
    """Sets multiple attributes for a class instance"""
    """Use this function like this: setattrs(obj, a = 1, b = 2,  #... )"""
    for k,v in kwargs.items():
        setattr(_self, k, v)

def flatten(h, new_key=[]):
    """Flattens a hierarchical dictionary of parameters (h) into an ordered dictionary of single parameter values."""
    items = []
    for k, v in h.items():
        if isinstance(v, MutableMapping):    # v is another dictionary
            items.extend(flatten(v, new_key + [k]).items())
        elif isinstance(v, list):                        # v is a list of parameters
            for i, vv in enumerate(v):
                items.append((tuple(new_key + [k] + [i]), vv))
    items.sort()
    return OrderedDict(items)

def updateFromFlat(hierDict, flatKeys, flatVals):
    """Updates a hierarchical parameter dictionary with values in its flattened representation."""
    for k, v in zip(flatKeys, flatVals):
        hierDict[k[0]][k[1]][k[2]] = v
    return hierDict

def next_pow_of_2(x):
    # Returns the smallest power of 2 greater than x
    return 0 if x == 0 else 2**(x - 1).bit_length()

# Denoising function
def wden(x_in, wname = 'sym8', tptr='sqtwolog', sorh='hard', scal='mln', wsize=15):
    """Wavelet denoising of a 1D signal. Inspired by the Matlab's wden function."""
    # tptr is the threshold selection rule specified as a string. Supported options for TPTR are:
    #      TODO: 'modwtsqtwolog' uses the maximal overlap discrete wavelet transform (MODWT) to denoise the signal with Donoho and Johnstone's universal threshold and level-dependent thresholding.
    #      'rigrsure' uses the principle of Stein's Unbiased Risk.
    #      'heursure' is a heuristic variant of Stein's Unbiased Risk.
    #      'sqtwolog' uses Donoho and Johnstone's universal threshold with the DWT.
    #      'minimaxi' uses minimax thresholding.
    # sorh specifies soft or hard thresholding with 's' or 'h'.
    # scal defines the type of threshold rescaling:
    #      'one' for no rescaling.
    #      'sln' for rescaling using a noise estimate based on the first-level coefficients.
    #      'mln' for rescaling using level-dependent estimates of the noise. This is the only option supported for MODWT denoising.
    # wsize = size of the Wiener filter; 0 for no filtering

    # 1. Multiscale wavelet decomposition
    wC = pywt.wavedec(x_in, wname, axis=0)
    nlev = len(wC)-1    # Number of decomposition levels
    nwcf = sum([len(c) for c in wC])          # Total number of wavelet coefficients

    # 2. Estimate standard deviation of noise on each level of the decomposition
    if scal == 'one':
        scl = [1.] * nlev
    elif scal == 'sln':
        scl = [np.median(np.abs(wC[-1]))/0.6745] * nlev           # Use the finest scale coefficients
    elif scal == 'mln':
        scl = [np.median(np.abs(c))/0.6745 for c in wC[1:]]       # See wnoisest in Matlab
    else: raise Exception('Unsupported value of parameter scal.')

    # 3. Detrmine the size of the threshold for each coefficient level
    if tptr == 'modwtsqtwolog':
        raise NotImplementedError('Not yet implemented for tptr = \'modwtsqtwolog\'.')
    elif tptr == 'rigrsure':
        """sx = sort(abs(x),1);
        sx2 = sx.^2;
        N1 = repmat((n-2*(1:n))',1,m);
        N2 = repmat((n-1:-1:0)',1,m);
        CS1 = cumsum(sx2,1);
        risks = (N1+CS1+N2.*sx2)./n;
        [~,best] = min(risks,[],1);
        % thr will be row vector
        thr = sx(best);"""
        pass
    elif tptr == 'heursure':
        hthr = np.sqrt(2*np.log(nwcf))
        eta = np.sum(np.abs(np.concatenate(wC))**2) / nwcf - 1
        crit = np.log2(nwcf)**1.5 / np.sqrt(nwcf)
        #thr = thselect(x,'rigrsure');
        #thr(thr > hthr) = hthr;
        #thr(eta < crit) = hthr;
        pass
    elif tptr == 'minimaxi':
        thr = [0.0 if nwcf <= 32 else 0.3936+0.1829*np.log2(nwcf)] * nlev
    elif tptr == 'sqtwolog':
        thr = [np.sqrt(2*np.log(nwcf))] * nlev
    else: raise Exception('Unsupported value of parameter tptr.')

    # 5. Apply the thresholding (hard or soft)
    wC_thr = [wC[0]] + [pywt.threshold(c, t*s, sorh) for c, t, s in zip(wC[1:], thr, scl)]

    # 6. Wavelet reconstruction
    x_out = pywt.waverec(wC_thr, wname, axis=0)

    x_out = x_out[:len(x_in)]

    # 7. Apply the Wiener filter to the difference
    if wsize > 0:
        diff = x_in - x_out
        diff = scipy.signal.wiener(diff.ravel(), mysize=wsize).reshape(x_in.shape)
        x_out += diff

    return x_out

# Phasing cost
def ph_cost(yF, xF, ph0=0.0, ph1=0.0, f=None, mw=2*512, cfun='LS'):
    """Calculate the cost function for phasing the data.
    yF - measured (unphased) spectrum
    xF - fitted model
    ph0, ph1 - zero- and first-order phasing terms
    mw - size of the median filter
    f - frequency scale in the fractions of angular frequency (i.e. f = [-1/2...(n-2)/2n] for even number of samples n and f = [-(n-1)/2n...(n-1)/2n] for odd n). If yF and xF contain non-contiguous frequency ranges, f should be specified explicitely.
    """

    # Define the frequency scale if it's not given
    if f is None:
        f = np.fft.fftshift(np.fft.fftfreq(len(yF), 1))

    # Compute the phasing term
    #### ph = np.exp(-1j*2*np.pi * tau * f_Hz - 1j*theta ).reshape((-1,1))   # The phasing term
    ph = np.exp(-1j* ph1 * f - 1j*ph0 ).reshape((-1,1))   # The phasing term
    yFph = yF * ph

    # Find and denoise the residual spectrum
    den = wden(yFph.real - xF.real, tptr='sqtwolog', scal='mln', wsize=25)

    # Remove the baseline with median filter
    res = nmrglue.process.proc_bl.med(den.ravel(), mw).reshape(-1,1)
    bln = (den - res).reshape(-1,1)

    # Compute the cost function
    if cfun == 'LS':
        val = np.linalg.norm(bln - np.mean(bln), 2)
    elif cfun == 'TV':
        val = np.linalg.norm(bln[1:] - bln[:-1], 1)

    return val, yFph, res, bln
    #return np.linalg.norm(res - np.mean(res), 2), yFph, res, bln

def flims(dt, nf):
    """Computes the values (in Hz) of the first and last sample in the spectrum with nf samples corresponding to a signal with sampling time dt. See https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fftfreq.html"""
    if nf%2 == 0:
        return (-nf/(2*dt*nf), (nf/2-1)/(dt*nf))
    else:
        return (-(nf-1)/(2*dt*nf), (nf-1)/(2*nf*dt))

def split_steps(dat, step):
    """Splits the list of fited parameters and returns a list of corresponding steps."""
    return [Step(frqBlkIds = step.frqBlkIds, parsKeys = [key], autoKeys=step.autoKeys, fitCustomLshape = step.fitCustomLshape) for key in step.parsKeys]

# ------------------------------ License files ---------------------------------

def makeLicenseFile(expiryDate=None, filename='license.lic', options=None):
    """Creates a license file, license.lic. Expiry date should have the format '%d-%m-%Y', e.g. '25-11-1986'."""
    if options is None:
        options = {}

    if expiryDate is not None:
        expiryTime = time.mktime(time.strptime(expiryDate, '%d-%m-%Y'))         # Time in sec from the start of epoch
    else: expiryTime = time.time() + 30 * 24*3600                               # Give thirty days

    # Save to file
    with open(filename, 'wb') as fp:
        dill.dump([expiryTime, options], fp)

def readLicenseFile(path=None):
    """Tries to locate a license file *.lic in the current directory, reads it, and returns the expiry date."""

    if path is None:
        path = os.getcwd()

    expiryTime, options = None, None

    # List all files in a directory using os.listdir
    for entry in os.listdir(path):
        fullpath = os.path.join(path, entry)
        if os.path.isfile(fullpath):
            if fullpath[-4:] == '.lic':
                try:
                    with open(fullpath, 'rb') as fp:
                        expiryTime_new, options_new = dill.load(fp)
                        if expiryTime is None or expiryTime_new > expiryTime:
                            # This license is better than one loaded previously
                            expiryTime = expiryTime_new
                            options = options_new
                except UnpicklingError: pass

    return expiryTime, options
