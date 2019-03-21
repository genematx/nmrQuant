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

# Functions for generating FIDs and optimization
from scipy import optimize
from scipy.optimize import minimize
import scipy as sp
import scipy.sparse
import scipy.linalg
import scipy.signal
from collections import OrderedDict, MutableMapping
import os, time
import marshal, inspect

# Functions needed only for Matlab
from operator import getitem

# ------- Setup predefined mixtures ----------
def setupSugars():
    # Create the sugars subtree
    S = chemNode("Sugars")
    Fr = chemNode("Fructose", chsh = [parsSpec(-0.05, 0.05)])
    Fr.addChild(chemNodeDB("alpha-D-Fructofuranose", intn=0.0517))
    Fr.addChild(chemNodeDB("beta-D-Fructofuranose", intn=0.2246))
    Fr.addChild(chemNodeDB("alpha-D-Fructopyranose", intn=0.0149))
    Fr.addChild(chemNodeDB("beta-D-Fructopyranose", intn=0.7088))
    Gl = chemNode("Glucose", chsh = [parsSpec(-0.05, 0.05)])
    Gl.addChild(chemNodeDB("alpha-D-Glucopyranose", intn=0.3750))
    Gl.addChild(chemNodeDB("beta-D-Glucopyranose", intn=0.6250))
    Su = chemNode("Sucrose", chsh = [parsSpec(-0.05, 0.05)])
    Su.addChild(chemNodeDB("Sucrose-F", intn=1))
    Su.addChild(chemNodeDB("Sucrose-G", intn=1))
    S.addChild(Fr)
    S.addChild(Gl)
    S.addChild(Su)

    # add QD nodes to the tree
    for node in S.items():
        if isinstance(node, chemNodeDB): node.dendrolize()
    S.setReported(False)

    # Assign the fitted parameters
    pars = defaultTreePars(S)
    pars["alpha-D-Fructofuranose-SPSY1"]["chshQD_rel"] = [-0.2309,0.0909]
    pars["alpha-D-Fructofuranose-SPSY1"]["jcplQD_rel"] = [-0.0230]
    pars["alpha-D-Fructofuranose-SPSY2"]["chshQD_rel"] = [-0.0194,-0.2070,-0.7459,-0.2558,-0.0761]
    pars["alpha-D-Fructofuranose-SPSY2"]["jcplQD_rel"] = [-0.0762,0.0881,0.1501,0.0055,-0.2295,0]
    pars["beta-D-Fructofuranose-SPSY1"]["chshQD_rel"] = [-0.1184,-0.0875]
    pars["beta-D-Fructofuranose-SPSY1"]["jcplQD_rel"] = [-0.2046]
    pars["beta-D-Fructofuranose-SPSY2"]["chshQD_rel"] = [-0.1743,-0.4962,-0.1394,-0.0733,-0.0881]    # [-0.1743,-0.1716,-0.5394,-0.0733,-0.0881]
    pars["beta-D-Fructofuranose-SPSY2"]["jcplQD_rel"] = [-0.6218,0.2476,0.9005,0.1568,-0.2148,0]
    pars["alpha-D-Fructopyranose-SPSY1"]["chshQD_rel"] = [-0.025,-0.0500]
    pars["alpha-D-Fructopyranose-SPSY1"]["jcplQD_rel"] = [0.3619]
    pars["alpha-D-Fructopyranose-SPSY2"]["chshQD_rel"] = [-0.0370,-0.0219,-0.4281,-0.3327,0.1398]
    pars["alpha-D-Fructopyranose-SPSY2"]["jcplQD_rel"] = [0.0287,-0.1771,-0.5064,0.2440,-0.5597,0]
    pars["beta-D-Fructopyranose-SPSY1"]["chshQD_rel"] = [-0.1210,-0.1350]
    pars["beta-D-Fructopyranose-SPSY1"]["jcplQD_rel"] = [-0.0329]
    pars["beta-D-Fructopyranose-SPSY2"]["chshQD_rel"] = [-0.1513,-0.1842,-0.1016,-0.2692,-0.1756]
    pars["beta-D-Fructopyranose-SPSY2"]["jcplQD_rel"] = [-0.0516,-0.1640,-0.0649,-0.2223,-0.1310,0]
    pars["alpha-D-Glucopyranose-SPSY1"]["chshQD_rel"] = [0.1231,0.1128,0.0962,0.1172,0.0520,0.1289,0.1466]
    pars["alpha-D-Glucopyranose-SPSY1"]["jcplQD_rel"] = [0.0169,-0.1084,0.1663,0.3671,-0.4402,-0.0515,0.2412]
    pars["beta-D-Glucopyranose-SPSY1"]["chshQD_rel"] = [0.1074,0.1005,0.1071,0.1194,0.0897,0.1309,0.1375]
    pars["beta-D-Glucopyranose-SPSY1"]["jcplQD_rel"] = [0.3562,-0.3018,0.2189,-0.3157,-0.2695,0.0124,-0.1188]
    pars["Sucrose-F-SPSY1"]["chshQD_rel"] = [0.1279, 0.1279]
    pars["Sucrose-F-SPSY1"]["jcplQD_rel"] = [0]
    pars["Sucrose-F-SPSY2"]["chshQD_rel"] = [0.0192,0.0337,0.0894,0.2890,-0.0665]
    pars["Sucrose-F-SPSY2"]["jcplQD_rel"] = [-0.0994,-0.2246,-0.4012,0.3173,-0.0479]
    pars["Sucrose-G-SPSY1"]["chshQD_rel"] = [0.1845,0.3602,0.1281,0.1974,-0.2918,0.6373,0.7598]
    pars["Sucrose-G-SPSY1"]["jcplQD_rel"] = [-0.0283,0.1973,-0.2828,0.1321,-0.8838,0.2803,-0.3675]

    # Save the tree and the parameters
    for node in S.items(): node.reset()

    # Convert from relative values to absolute ones
    pars = rel2abs(S, pars)

    # Update default tree parameters
    for key, val in flatten(pars).items():
        try:
            getattr(S[key[0]], key[1])[key[2]] = getattr(S[key[0]], key[1])[key[2]]._replace(dval=val)
        except KeyError:
            pass

    saveTree('Sugars', S, pars)
    S, pars = loadTree('Sugars')

    return S, pars

def setupJuices():
    X = chemNode("Juice")
    X.addChild(chemNodeDB("Water", alph = [parsSpec(0, 50)]))

    # Load the sugars subtree
    S, sugarsPars = loadTree('Sugars')
    X.addChild(S)

    # Create the acids subtree
    A = chemNode("Acids")
    X.addChild(A)
    A.addChild(chemNodeDB("Citric acid"))
    A.addChild(chemNodeDB("Malic acid"))

    # add QD nodes to the tree
    for node in X.items():
        if isinstance(node, chemNodeDB): node.dendrolize()

    # Set the reported flags (must be done in the end when the leaves are added to the tree)
    S.setReported(False)
    A.setReported(False)

    # Set up the model parameters
    pars = defaultTreePars(X)
    pars.update(sugarsPars)     # use the predefined QD parameters for sugars
    pars["Juice"]["alph_rel"] = [-0.85]

    # Convert from relative values to absolute ones
    pars = rel2abs(X, pars)

    # Update default tree parameters
    for key, val in flatten(pars).items():
        try:
            getattr(X[key[0]], key[1])[key[2]] = getattr(X[key[0]], key[1])[key[2]]._replace(dval=val)
        except KeyError:
            pass

    saveTree('Juices', X, pars)
    X, pars = loadTree('Juices')

    return X, pars

def setupACN_Diox():
    # Create a subtree for caffeine / maleic acid
    X = chemNode("Mixture")
    X.addChild(chemNodeDB("Acetonitrile"))
    X.addChild(chemNodeDB("Dioxane"))

    # add QD nodes to the tree
    for node in X.items():
        if isinstance(node, chemNodeDB): node.dendrolize()

    X.setReported(False)

    # Assign the fitted parameters
    pars = defaultTreePars(X)

    # Update default tree parameters
    for key, val in flatten(pars).items():
        try:
            getattr(X[key[0]], key[1])[key[2]] = getattr(X[key[0]], key[1])[key[2]]._replace(dval=val)
        except KeyError:
            pass

    # Save the tree and the parameters
    for node in X.items(): node.reset()
    saveTree('ACN_Diox', X, pars)
    X, pars = loadTree('ACN_Diox')

    return X, pars

def setupCafMA():
    # Create a subtree for caffeine / maleic acid
    X = chemNode("Mixture")
    #X.addChild(chemNodeDB("Water", alph = [parsSpec(0, 75)]))
    #S = chemNode("Solutes")
    #X.addChild(S)
    X.addChild(chemNodeDB("Caffeine"))
    X.addChild(chemNodeDB("Maleic acid"))

    # add QD nodes to the tree
    for node in X.items():
        if isinstance(node, chemNodeDB): node.dendrolize()

    X.setReported(False)

    # Assign the fitted parameters
    pars = defaultTreePars(X)
    pars["Mixture"]["alph_rel"] = [-0.85]
    pars["Caffeine-SPSY1"]["chshQD_rel"] = [0.0847330953486094]
    pars["Caffeine-SPSY2"]["chshQD_rel"] = [-0.000906643455866547]
    pars["Caffeine-SPSY3"]["chshQD_rel"] = [-0.0935097134208091, -0.270221261778036]
    pars["Caffeine-SPSY3"]["jcplQD_rel"] = [0.515823922069580]

    # Convert from relative values to absolute ones
    pars = rel2abs(X, pars)

    # Update default tree parameters
    for key, val in flatten(pars).items():
        try:
            getattr(X[key[0]], key[1])[key[2]] = getattr(X[key[0]], key[1])[key[2]]._replace(dval=val)
        except KeyError:
            pass

    # Save the tree and the parameters
    for node in X.items(): node.reset()
    saveTree('CafMA', X, pars)
    X, pars = loadTree('CafMA')

    return X, pars

def setupScionExp2():
    X = chemNode("Mixture")
    #X.addChild(chemNodeDB("Water", alph = [parsSpec(0, 75)]))

    # add sugars
    Fr = chemNode("Fructose", chsh = [parsSpec(-0.05, 0.05)])
    Fr.addChild(chemNodeDB("alpha-D-Fructofuranose", intn=0.0517))
    Fr.addChild(chemNodeDB("beta-D-Fructofuranose", intn=0.2246))
    Fr.addChild(chemNodeDB("alpha-D-Fructopyranose", intn=0.0149))
    Fr.addChild(chemNodeDB("beta-D-Fructopyranose", intn=0.7088))
    Gl = chemNode("Glucose", chsh = [parsSpec(-0.05, 0.05)])
    Gl.addChild(chemNodeDB("alpha-D-Glucopyranose", intn=0.3750))
    Gl.addChild(chemNodeDB("beta-D-Glucopyranose", intn=0.6250))
    Su = chemNode("Sucrose", chsh = [parsSpec(-0.05, 0.05)])
    Su.addChild(chemNodeDB("Sucrose-F", intn=0.5))
    Su.addChild(chemNodeDB("Sucrose-G", intn=0.5))
    #X.addChild(Fr)
    X.addChild(Gl)
    X.addChild(Su)

    # Create the acids subtree
    X.addChild(chemNodeDB("Quinic acid"))
    X.addChild(chemNodeDB("Shikimic acid"))

    # Add TMS
    X.addChild(chemNodeDB("TMS"))

    # add QD nodes to the tree
    for node in X.items():
        if isinstance(node, chemNodeDB): node.dendrolize()

    # Set the reported flags (must be done in the end when the leaves are added to the tree)
    X.setReported(False)

    # Set up the model parameters
    pars = defaultTreePars(X)
    # Update parameters for acids
    # Assign the fitted parameters
    pars["Quinic acid-SPSY1"]["chshQD_rel"] = [-0.0115,-0.0609,-0.0026,0.0200,-0.0066,-0.0968,-0.0422]
    pars["Quinic acid-SPSY1"]["jcplQD_rel"] = [0.0361,0.3928,-0.2112,0.0105,0.0029,-0.0664,0.0023,0.0607,0.0040]
    pars["Shikimic acid-SPSY1"]["chshQD_rel"] = [-0.0495,-0.0055,0.0246,-0.0459,-0.0097,0.0243]
    pars["Shikimic acid-SPSY1"]["jcplQD_rel"] = [0.0430,-0.0310,-0.0195,-0.0126,-0.0103,0.1441,0.0255,0.0396,0.0413,0.3911]
    # Assign the fitted parameters for sugars
    """pars["alpha-D-Fructofuranose-SPSY1"]["chshQD_rel"] = [-0.2309,0.0909]
    pars["alpha-D-Fructofuranose-SPSY1"]["jcplQD_rel"] = [0.0230]
    pars["alpha-D-Fructofuranose-SPSY2"]["chshQD_rel"] = [-0.0194,-0.2070,-0.7459,-0.2558,-0.0761]
    pars["alpha-D-Fructofuranose-SPSY2"]["jcplQD_rel"] = [-0.0762,0.0881,0.1501,0.0055,-0.2295,0]
    pars["beta-D-Fructofuranose-SPSY1"]["chshQD_rel"] = [-0.1184,-0.0875]
    pars["beta-D-Fructofuranose-SPSY1"]["jcplQD_rel"] = [0.2046]
    pars["beta-D-Fructofuranose-SPSY2"]["chshQD_rel"] = [-0.1743,-0.1716,-0.5394,-0.0733,-0.0881]
    pars["beta-D-Fructofuranose-SPSY2"]["jcplQD_rel"] = [-0.6218,0.2476,0.9005,0.1568,-0.2148,0]
    pars["alpha-D-Fructopyranose-SPSY1"]["chshQD_rel"] = [-0.025,-0.0500]
    pars["alpha-D-Fructopyranose-SPSY1"]["jcplQD_rel"] = [-0.3619]
    pars["alpha-D-Fructopyranose-SPSY2"]["chshQD_rel"] = [-0.0370,-0.0219,-0.4281,-0.3327,0.1398]
    pars["alpha-D-Fructopyranose-SPSY2"]["jcplQD_rel"] = [0.0287,-0.1771,-0.5064,0.2440,-0.5597,0]
    pars["beta-D-Fructopyranose-SPSY1"]["chshQD_rel"] = [-0.1210,-0.1350]
    pars["beta-D-Fructopyranose-SPSY1"]["jcplQD_rel"] = [0.0329]
    pars["beta-D-Fructopyranose-SPSY2"]["chshQD_rel"] = [-0.1513,-0.1842,-0.1016,-0.2692,-0.1756]
    pars["beta-D-Fructopyranose-SPSY2"]["jcplQD_rel"] = [-0.0516,-0.1640,-0.0649,-0.2223,-0.1310,0]"""
    pars["alpha-D-Glucopyranose-SPSY1"]["chshQD_rel"] = [0.1231,0.1128,0.0962,0.1172,0.0520,0.1289,0.1466]
    pars["alpha-D-Glucopyranose-SPSY1"]["jcplQD_rel"] = [0.0169,-0.1084,0.1663,0.3671,-0.4402,-0.0515,0.2412]
    pars["beta-D-Glucopyranose-SPSY1"]["chshQD_rel"] = [0.1074,0.1005,0.1071,0.1194,0.0897,0.1309,0.1375]
    pars["beta-D-Glucopyranose-SPSY1"]["jcplQD_rel"] = [0.3562,-0.3018,0.2189,-0.3157,-0.2695,0.0124,-0.1188]
    pars["Sucrose-F-SPSY1"]["chshQD_rel"] = [0.1279, 0.1279]
    pars["Sucrose-F-SPSY1"]["jcplQD_rel"] = [0]
    pars["Sucrose-F-SPSY2"]["chshQD_rel"] = [0.0192,0.0337,0.0894,0.2890,-0.0665]
    pars["Sucrose-F-SPSY2"]["jcplQD_rel"] = [-0.0994,-0.2246,-0.4012,0.3173,-0.0479]
    pars["Sucrose-G-SPSY1"]["chshQD_rel"] = [0.1845,0.3602,0.1281,0.1974,-0.2918,0.6373,0.7598]
    pars["Sucrose-G-SPSY1"]["jcplQD_rel"] = [-0.0283,0.1973,-0.2828,0.1321,-0.8838,0.2803,-0.3675]

    pars["Mixture"]["alph_rel"] = [-0.85]

    # Convert from relative values to absolute ones
    pars = rel2abs(X, pars)

    # Update default tree parameters
    for key, val in flatten(pars).items():
        try:
            getattr(X[key[0]], key[1])[key[2]] = getattr(X[key[0]], key[1])[key[2]]._replace(dval=val)
        except KeyError:
            pass

    saveTree('ScionExp2', X, pars)
    X, pars = loadTree('ScionExp2')

    return X, pars

def setupMixture():
    X = chemNode("Mixture")
    #X.addChild(chemNodeDB("Water", alph = [parsSpec(0, 75)]))
    #X.addChild(chemNodeDB("Ethanol"))
    #X.addChild(chemNodeDB("Citric acid"))
    X.addChild(chemNodeDB("Thiamine"))

    # add QD nodes to the tree
    for node in X.items():
        if isinstance(node, chemNodeDB): node.dendrolize()

    # Set the reported flags (must be done in the end when the leaves are added to the tree)
    X.setReported(False)
    #Fr.setReported(False)
    #Gl.setReported(False)

    # Set up the model parameters
    pars = defaultTreePars(X)

    pars["Mixture"]["alph"] = [2]

    # Update default tree parameters
    for key, val in flatten(pars).items():
        try:
            getattr(X[key[0]], key[1])[key[2]] = getattr(X[key[0]], key[1])[key[2]]._replace(dval=val)
        except KeyError:
            pass

    saveTree('Mixture', X, pars)
    X, pars = loadTree('Mixture')

    return X, pars

def setupElmar():
    X = chemNode("Mixture")

    # add components
    X.addChild(chemNodeDB("EvA34"))
    X.addChild(chemNodeDB("b-EvA34"))
    X.addChild(chemNodeDB("g-EvA34"))
    X.addChild(chemNodeDB("b,g-EvA34"))
    X.addChild(chemNodeDB("HxCO3-"))

    # add QD nodes to the tree
    for node in X.items():
        if isinstance(node, chemNodeDB): node.dendrolize('13C')

    # Set the reported flags (must be done in the end when the leaves are added to the tree)
    X.setReported(False)

    # Set up the model parameters
    pars = defaultTreePars(X)
    pars["Mixture"]["alph"] = [2]

    # Update default tree parameters
    for key, val in flatten(pars).items():
        try:
            getattr(X[key[0]], key[1])[key[2]] = getattr(X[key[0]], key[1])[key[2]]._replace(dval=val)
        except KeyError:
            pass

    # Save the tree and the parameters
    for node in X.items(): node.reset()
    saveTree('Elmar', X, pars)
    X, pars = loadTree('Elmar')

    return X, pars

def setupEvA02():
    X = chemNode("Mixture")

    # add components
    X.addChild(chemNodeDB("EvA02"))
    #X.addChild(chemNodeDB("b-EvA02"))
    X.addChild(chemNodeDB("HxCO3-"))
    #X.addChild(chemNodeDB("CO2"))

    # add QD nodes to the tree
    for node in X.items():
        if isinstance(node, chemNodeDB): node.dendrolize('13C')

    # Set the reported flags (must be done in the end when the leaves are added to the tree)
    X.setReported(False)

    # Set up the model parameters
    pars = defaultTreePars(X)
    pars["Mixture"]["alph"] = [3]

    # Save the tree and the parameters
    for node in X.items(): node.reset()

    # Update default tree parameters
    for key, val in flatten(pars).items():
        try:
            getattr(X[key[0]], key[1])[key[2]] = getattr(X[key[0]], key[1])[key[2]]._replace(dval=val)
        except KeyError:
            pass

    saveTree('EvA02', X, pars)
    X, pars = loadTree('EvA02')

    return X, pars

def rel2abs(tree, relParsH):
    """Converts a hierarchical structure of relative parameters to the corresponding absolute values for a given tree."""
    absParsH = defaultTreePars(tree)
    for name, pars in relParsH.items():
        if name != '.':
            for key, val_rel_arr in pars.items():
                if key[-4:] == '_rel':
                    val_abs_arr = []
                    for i, val_rel in enumerate(val_rel_arr):
                        spec = getattr(tree[name], key[:-4])[i]
                        val_abs = (spec.max-spec.min)*(val_rel+1)/2 + spec.min
                        val_abs_arr.append(val_abs)
                    absParsH[name][key[:-4]] = val_abs_arr
        #else: absParsH[name] = copy.copy(relParsH['.'])

    return absParsH

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

def print_fun(x, f, accepted):
    print(x)
    print("at minima %.4f accepted %d" % (f, int(accepted)))

##### ------------ Main classes for the general program logic ------------ #####

class Step():

    def __init__(self, frqBlkIds = None, parsKeys = None, autoKeys=None, repRootNames=None, fitCustomLshape = False):
        self.frqBlkIds = frqBlkIds if frqBlkIds is not None else set()
        self.parsKeys = set(parsKeys) if parsKeys is not None else set()      # Parameters to fit on this step
        self.autoKeys = set(autoKeys) if autoKeys is not None else set( [('.', 'theta', 0), ('.', 'sigma2', 0)] )                      # Potentially autofittable parameters that will be excluded from fitting on this step (may contain, theta, gamma, sigma2, and any amplitudes)
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
        self.setTree(chemNode('Mixture'))

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
            if isinstance(node, chemNodeDB) and node.HCmode != self.HCmode: node.dendrolize(self.HCmode)

        self._updateParameters()

        if priors is not None:
            self.parsSpecDict.update(priors)

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
            return 0

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

        return True

    def toggleRepRoot(self, key):
        """Toggles the reportability of a certain root node and updates the parameters accordingly."""
        self.T[key].toggleReported()
        self._updateParameters()

    def addSeries(self, **kwargs):

        SSS = Series(parent=self, **kwargs)
        self.series.append(SSS)

        return SSS

    #@profile
    def _optimize(self, costFuncOpti, bounds, initVals, nhop=None, verbose=True):
        """Core optimization routine; used by all Series and Datums in this Workspace"""
        if len(initVals) > 2 or (nhop is not None and nhop > 0):
            res = optimize.basinhopping(costFuncOpti, initVals, \
                  niter = nhop if nhop is not None else config.OPTIM_maxBasinhoppingSteps, \
                  niter_success = nhop if nhop is not None else config.OPTIM_niterSuccess, T = 10, disp = verbose, \
                  minimizer_kwargs=dict(method=config.OPTIM_method, bounds=bounds, tol=1e-12) )     #, \
            #      #take_step=MyTakeStep())
        else:
            res = optimize.minimize(costFuncOpti, x0=initVals, bounds=bounds, method='L-BFGS-B')
            #print(res['message'])

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
                                    'dt' : ser.t[1]-ser.t[0],
                                    'nf' : len(ser.f),
                                    'steps' : ser.steps,
                                    'freqBlocks' : [blk._replace(indxFreq=None, bF=None) for blk in ser.freqBlocks],
                                    'parsSpecDict' : ser.parsSpecDict,
                                    'crntMetaF' : ser.crntMetaF,
                                    'smplDistF' : ser.smplDistF,
                                    'meta_function' : ser._meta,
                                    'apod' : ser.apod,
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
                                                    'refChshKey' : dat.refChshKey
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
            t = np.linspace(0, ser['dt']*(ser['nt']-1), ser['nt']).reshape(-1, 1)
            newSeries = self.addSeries(name=ser['name'], c0=ser['c0'], f0=ser['f0'],
                                    t=t, nf=ser['nf'], apod=ser['apod'],
                                    priors=ser['parsSpecDict'])
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
            for stp in ser['steps']:
                newStep = Step()
                newStep.frqBlkIds, newStep.parsKeys = stp.frqBlkIds, stp.parsKeys
                try: newStep.autoKeys = stp.autoKeys
                except AttributeError: pass
                try: newStep.fitCustomLshape = stp.fitCustomLshape
                except AttributeError: pass
                newSeries.steps.append(newStep)
            for dat in ser['data']:
                newDatum= newSeries.addDatum(dat['yT'], name=dat['name'], arrVal=dat['arrVal'],   #/np.linalg.norm(dat['yT'])
                                    crntParsH=dat['crntParsH'], priors=dat['parsSpecDict'])
                newDatum.mdldPeaks = dat['mdldPeaks']
                newDatum.pckdPeaks = dat['pckdPeaks']
                newDatum.sF = dat['sF'] if 'sF' in dat.keys() else None
                newDatum.sT = dat['sT'] if 'sT' in dat.keys() else None
                if 'smplDistF' in dat.keys():
                    newDatum.smplDistF.update(dat['smplDistF'])
                if 'refChshKey' in dat.keys():
                    newDatum.setReferenceChshKey(dat['refChshKey'])

        if lshapeOrder is None: self.set_lshapeOrder(2)

        # Add missing parameters (needed for back-compatibility)
        for parsH in [dat.crntParsH for ser in self.series for dat in ser.data]:
            if 'gamma' not in parsH['.'].keys():
                parsH['.']['gamma'] = [0.0]

class Series():
    """Class for the data series (e.g. in reaction monitoring)."""

    def __init__(self, parent, name = None, c0=None, f0=None, t=None, nf=None, apod=0, priors=None):
        self.parent = parent     # The workspace that contains the tree
        self.name = name if name is not None else 'Series ' + str(len(self.parent.series)+1)
        self.c0 = c0
        self.f0 = f0
        self.t = t.reshape(-1,1) if t is not None else []
        self.f = []
        self.data = []               # A list of Datum structures
        self.steps = [Step(repRootNames=self.repRootNames)]               # Fitting steps; each entry is a set of parsKeys tuples and set of frqBlkIds
        self.freqBlocks = []         # a list of optimization frequency ranges
        self.apod = apod
        self.wT = 1
        self.crntMetaF = dict()       # A dictionary of current values of meta-parameters
        self.smplDistF = dict()
        self._meta = None             # A function that chnages the Series parameters controlled by the meta-parameters
        self.fullReset(nf, apod, priors)           # Setup the frequency range and compute the spectra

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

    def addDatum(self, yT, **kwargs):
        """Adds a Datum to the Series."""
        # Create new Datum structure and add it to the Series
        yT = yT.reshape(-1,1)     # Make sure the data is reshaped properly
        DDD = Datum(yT, parent=self, **kwargs)
        self.data.append(DDD)

        return DDD

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
        zT, repRootNames = getFID(self.T, self.t, self.c0, self.f0, evalParsH, tau)
        xT = np.dot(zT, np.array(ampl).reshape(-1,1)) * np.exp(1j*theta)
        xT *= sT
        yT = xT + np.sqrt(sigma2/2)*(np.random.randn(*xT.shape) + 1j*np.random.randn(*xT.shape)) if sigma2 > 0 else xT

        # Add new data to the series
        DDD = Datum(yT, parent=self, name = name if name != '' else 'data_'+str(len(self.data)+1), arrVal = arrVal )
        self.data.append(DDD)

        return DDD

    def resetFreqs(self, nf=None, apod=None):
        """Resets the frequency scale for the entire Series and computed spectra."""
        if nf is None:
            nf = next_pow_of_2(len(self.t))     # Determine the number of samples in the full signal spectrum (possibly including zero-filling)
        if nf > 0:
            self.f = (np.fft.fftshift(np.fft.fftfreq(nf, self.t[1]-self.t[0]))+self.f0).reshape(-1,1)/self.c0

            # Update the frequency blocks
            for i in range(len(self.freqBlocks)):
                self.altFreqBlock(indx=i)
            if len(self.freqBlocks) == 0 : self.addFreqBlock()   # Add the "all frequencies" block

            # Compute a window in the time domain
            if apod is not None : self.apod = apod
            self.wT = np.exp(-self.apod*self.t) if self.apod > 0 else 1

            for D in self.data:
                D.resetSignals()

    def fullReset(self, nf=None, apod=None, priors=None):
        self.resetFreqs(nf, apod)
        self.parsSpecDict = priors if priors is not None else {}
        self.crntMetaF.clear()
        self.remMetaFunction()

    def clear_data(self):
        """Removes al datasets from the series."""
        self.data.clear()

    def addFreqBlock(self, lims=None, bslnOrder=(2, 2)):
        """Adds a frequency block for optimization at certain in the self.freqBlocks arrays."""
        if lims is None:
            self.freqBlocks = []         # Reset the frequency blocks and add the entire signal
            lims = (-1*float('inf'), float('inf'))

        # Choose only samples that are in the optimization range
        indxFreq = np.flatnonzero((self.f<=max(lims))*(self.f>=min(lims)))     # Indices of frequency points in the range
        nf = indxFreq.size
        # Define baseline in the frequency domain
        bFr = [np.linspace(-1,1,nf).reshape(-1,1)**i for i in range(bslnOrder[0]+1)] if bslnOrder[0] is not None else []
        bFi = [1j*np.linspace(-1,1,nf).reshape(-1,1)**i for i in range(bslnOrder[1]+1)] if bslnOrder[1] is not None else []
        bF = np.hstack(bFr+bFi) if len(bFr)+len(bFi) > 0 else None

        self.freqBlocks.append(freqSpec(min(lims), max(lims), indxFreq, bslnOrder, bF))

    def altFreqBlock(self, lims=None, bslnOrder=None, indx=-1):
        """Alters a frequency block at position indx in self.freqBlocks (the last block by default)."""
        if lims is None or indx == 0:         # Can't change the limits of the first block
            lims = (self.freqBlocks[indx].min, self.freqBlocks[indx].max)
        if bslnOrder is None:
            bslnOrder = self.freqBlocks[indx].bslnOrder

        # Choose only samples that are in the optimization range
        indxFreq = np.flatnonzero((self.f<=max(lims))*(self.f>=min(lims)))     # Indices of frequency points in the range
        nf = indxFreq.size
        # Define baseline in the frequency domain
        bFr = [np.linspace(-1,1,nf).reshape(-1,1)**i for i in range(bslnOrder[0]+1)] if bslnOrder[0] is not None else []
        bFi = [1j*np.linspace(-1,1,nf).reshape(-1,1)**i for i in range(bslnOrder[1]+1)] if bslnOrder[1] is not None else []
        bF = np.hstack(bFr+bFi)

        self.freqBlocks[indx] = freqSpec(min(lims), max(lims), indxFreq, bslnOrder, bF)

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

    def _fnc_prior(self, evalParsH, evalMetaF, parsKeys=None, customPriors=None):
        """Custom prior probability function. Can be used to describe dependencies among parameters in different planes. Use parsKeys to determine if the prior needs to be computed for the specific keys."""
        if parsKeys == []:
            return 0
        else:
            # Evaluate for ALL meta parameters
            #if parsKeys is None:    # All meta parameters
            parsKeys = [key for key in self.parsSpecDict.keys() if len(key)==2]
            return sum([self.getPrior(key, customPriors).evalPrior(arg=evalMetaF[key]) if len(key)==2 else 0 for key in set(parsKeys) ])

    def evaluate(self, evalParsH=None, evalMetaF=None, parsKeys=None, autoKeys=None, frqBlkIds=None, funcType=None, evaluatePriors=False, customPriors=None, robust=None, returnSignals=False, evaluateAll=True):
        """Evaluates the objective function (sum of logLikelihoods for each datum + sum of logPriors).
           Inputs:
           evalParsH - a list of hierarchical dictionaries one for each Datum
           evalMetaF - flat dictionary of meta-parameters' values"""
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
            _res, meta = DDD.evaluate(evalParsH[i], parsKeysDatum, autoKeysDatum, frqBlkIds, funcType, evaluatePriors, customPriors, robust=robust, returnSignals=returnSignals)
            result += _res
            m_ampl[..., i] = meta['ampl'][0].ravel()
            S_ampl[...,i] = meta['ampl'][1]
            theta[i] = meta['theta']
            a_sigma2[i], b_sigma2[i] = meta['sigma2']

        if evaluatePriors:
            result += self._fnc_prior(evalParsH, evalMetaF, parsKeys, customPriors=customPriors)      # Add prior on the series level

        return result, {"ampl":(m_ampl, S_ampl), "theta":theta, "sigma2":(a_sigma2, b_sigma2)}

    def optimize(self, parsKeys, autoKeys=None, frqBlkIds=None, funcType=None, evaluatePriors=False, nhop=None, verbose=True):
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
                return -self.evaluate(evalParsH, evalMetaF, parsKeys, autoKeys, frqBlkIds, funcType, evaluatePriors, customPriors, robust=False, evaluateAll=evaluateAll)[0]

            res = self._optimize(costFuncOpti, bounds, initVals, nhop=nhop, verbose=verbose)

            # Update the structure of all parameters
            for k, v in zip(parsKeys, res.x):
                if len(k) == 4:
                    self.data[k[0]].crntParsH[k[1]][k[2]][k[3]] = v
                elif len(k) == 2:
                    evalMetaF[k] = v
            # Evaluate the function skipping the datasets that are not present in parsKeys
            return -self.evaluate(evalParsH, evalMetaF, parsKeys, autoKeys, frqBlkIds, funcType, evaluatePriors, customPriors, robust=False, evaluateAll=evaluateAll)[0]

        res = self._optimize(costFuncOpti, bounds, initVals, nhop=nhop, verbose=verbose)

        # Update the structure of all parameters
        for k, v in zip(parsKeys, res.x):
            if len(k) == 4:
                self.data[k[0]].setCrntVal(k[1:3], v) #   crntParsH[k[1]][k[2]][k[3]] = v
            elif len(k) == 2:
                self.crntMetaF[k] = v

        # Re-evaluatethe posterior
        result, meta = self.evaluate(None, None, parsKeys, autoKeys, frqBlkIds, funcType, evaluatePriors, returnSignals=True)

        if verbose:
            print("Optimization finished. Posterior={:.4g}".format(result))
            print('Found values:')
            for key in parsKeys:
                print("     {} = {:.5g}".format(str(key), self.getCrntVal(key)))

        return result, meta

    def sample(self, parsKeys, autoKeys=None, frqBlkIds=None, funcType=None, evaluatePriors=False, nwalkers=None, nsteps=None):
        """Samples the posterior distribution using the MCMC algorithm."""

        parsKeys, autoKeys = self._prepareKeys(parsKeys, autoKeys, verbose=False)
        evaluateAll = any([key[-2]=='meta' for key in parsKeys])     # If optimizing over any meta parameters, will need to evaluate all Datums, else can skip some
        result = {}
        self.smplDistF.clear()             # Clear the characteristics of marginal distributions

        # If there are no parameters to sample
        # First evaluate the cost function with current parameters. If there is nothing to sample, this will be output as the result (at least for some datasets).
        value, meta = self.evaluate(autoKeys=autoKeys, frqBlkIds=frqBlkIds, funcType=funcType, evaluatePriors=evaluatePriors, returnSignals=True)
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
            return self.evaluate(evalParsH, evalMetaF, parsKeys, autoKeys, frqBlkIds, funcType, evaluatePriors, customPriors, evaluateAll=evaluateAll)

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

    def _prepareKeys(self, parsKeys, autoKeys=None, verbose=True):
        """Convert parameter Keys from Datum to Series representations and make sure there are no repetitions."""
        parsKeys = set([]) if parsKeys is None else set(parsKeys)
        if autoKeys is not None: autoKeys = set(autoKeys)
        for key in list(parsKeys):
            if len(key) == 3:
                parsKeys.remove(key)
                parsKeys.update([(i, *key) for i in range(len(self.data))])        # Repeat the same key for all Datums
        parsKeys = sorted(list(parsKeys))

        # Determine which parameters can be marginalized and remove them from the list of sampled values
        #parsKeys = [key for key in parsKeys if not ( len(key)==4 and ((key[2]=='ampl' and self.data[key[0]].getPrior(key[-3:]).distr=='Gaussian') \
        #            or (key[2]=='sigma2' and self.data[key[0]].getPrior(key[-3:]).distr=='Inverse-Gamma') \
        #            or (key[2]=='theta' and self.data[key[0]].getPrior(key[-3:]).distr=='Uniform' \
        #                                    and self.data[key[0]].getPrior(key[-3:]).min==-np.pi \
        #                                    and self.data[key[0]].getPrior(key[-3:]).max==np.pi)) )]

        if verbose:
            print("Optimizing over {} parameters:".format(len(parsKeys)))
            for key in parsKeys:
                print("     {}".format(str(key)))

        return parsKeys, autoKeys

    def remove(self):
        """Removes itself from the Workspace"""
        self.parent.series.remove(self)

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

    def __init__(self, yT, parent, name='', arrVal=None, crntParsH=None, priors=None):
        self.name = name
        self.parent = parent               # A series object that will contain this Datum
        self.yT = yT    # The acquired signal in time domain (FID) without any preprocessing
        self.arrVal = arrVal if arrVal is not None else len(self.parent.data)+1     # Value of the arrayed parameter in the serial experiment (e.g., extent of reaction)
        self.parsSpecDict = {}
        self.crntParsH = None
        self.smplDistF = dict()          # A flat dictionary of sampled (or marginalized) parameters
        self.mdldPeaks = {}
        self.pckdPeaks = []
        self.refChshKey = None           # A key of the chemical shift that will be used as a reference (will be set to its default value and the rest of the spectrum shifted accordingly)
        self.fullReset(crntParsH, priors)

    def __getattr__(self, attr):
        """Called with the dot notation for attributes not found in the class (e.g. parameters shared between many spectra in the series, c0, f0, etc.)."""
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
            nf = len(self.f)

            # Compute the spectrum of the input signal if necessary
            if self.yF is None:
                self.yF = np.fft.fftshift(np.fft.fft(self.yT * self.wT, nf, axis=0), axes=0) / np.sqrt(nf)

            # Compute (possibly new) phasing terms
            ph = np.exp(-1j*2*np.pi * self.crntParsH["."]["tau"][0] * (self.f*self.c0-self.f0) - 1j*self.crntParsH["."]["theta"][0] ).reshape((-1,1))

            # Get the data spectrum
            self.yFph = self.yF * ph
            if np.mean(self.yFph.ravel().real) < np.median(self.yFph.ravel().real):   # If the distribution is skewed to the left; i.e. only a few points are less than the most of them #sum(yFph.ravel().real) < 0:
                self.yFph = -1 * self.yFph

            # Still need the model spectrum if the evaluation was in time domain (xT is known but xFph is not)
            if self.xFph is None and self.xT is not None:
                xF = np.fft.fftshift(np.fft.fft(self.xT * self.wT, nf, axis=0), axes=0) / np.sqrt(nf)
                self.xFph = self.xF * ph
                if sum(self.xFph.ravel().real) < 0:
                    self.xFph = -1 * self.xFph

    def resetSignals(self):
        self.yF = np.fft.fftshift(np.fft.fft(self.yT * self.wT, len(self.f), axis=0), axes=0) / np.sqrt(len(self.f))
        self.zF = None        # A matrix of component signals
        self.bF = None        # A baseline
        self.sF, self.sT = None, None        # A lineshape kernel
        self.Gz = None

    def resetCrntPars(self, crntParsH=None, priors=None):
        """Resets ALL current parameters."""
        self.parsSpecDict.clear()
        if priors is not None:
            self.parsSpecDict.update(priors)
        self.crntParsH = self.getDfltParsH()
        if crntParsH is not None:
            for key, val in crntParsH.items():
                self.crntParsH[key].update(val)
        self.smplDistF.clear()          # A flat dictionary of sampled (or marginalized) parameters
        self.mdldPeaks.clear()
        self.pckdPeaks.clear()

    def fullReset(self, crntParsH=None, priors=None):
        self.resetSignals()
        self.resetCrntPars(crntParsH, priors)

    def getCrntVal(self, key):
        """Returns the relative or absolute value of the parameter key."""
        # TODO: Will be deprecated.
        return self.crntParsH[key[0]][key[1]][key[2]]

    def setCrntVal(self, key, val):
        """Updates the current value of the parameter key."""
        if key == self.refChshKey:
            self.setGlobalChshVal(self.getGlobalChshVal() + (float(val) - self.getPrior(key).dflt()) )
            val = self.getPrior(key).dflt()
        self.crntParsH[key[0]][key[1]][key[2]] = float(val)
        self.smplDistF.clear()

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

    def isAutofittable(self, key, customPriors=None):
        """Checks if a parameter can be fitted algebraically/marginalized based on the definition of its prior distribution."""
        prior = self.getPrior(key, customPriors)
        if (key[1]=='sigma2' and prior.distr=='Inverse-Gamma') \
            or (key[1]=='theta' and prior.distr=='Uniform' and prior.min==-np.pi and prior.max==np.pi) \
            or (key[1]=='ampl' and key[0] in self.repRootNames and prior.distr=='Gaussian') \
            or (key[1]=='gamma' and prior.distr != 'Constant'):
            return True
        else: return False

    #@profile
    def _fnc_lklhd(self, evalParsH, frqBlkIds=None, autoKeys=None, funcType=None, wnd=None, customPriors=None, returnSignals=False, robust=None, useComplex=True):
        """Computes the value of the likelihood function. If evaluatePriors == True, will also add values of prior distributions for amplitudes, theta, and sigma2, if those parameters can not be integrated out."""
        #funcType = 'TLS'
        useComplex = False

        # 1. Update the settings
        if frqBlkIds is None:
            frqBlkIds = self.steps[-1].frqBlkIds
        if robust is None:
            robust = config.SAMPL_robustLS
        if funcType is None:
            funcType=config.SAMPL_funcType
        if funcType is 'TLS':
            useComplex = False

        # 2. Compute a matrix of model signals Z, either in time or frequency domain
        inTimeDomain = (len(frqBlkIds) == 0)
        if inTimeDomain:
            # -------------------------- TIME ----------------------------
            # 1. Compute model signals in time domain
            zT, repRootNames = getFID(self.T, self.t, self.c0, self.f0, evalParsH)            # 1. Compute the model signals

            # 1. Apply custom lineshape correction if defined
            if self.sT is not None:
                zT *= self.sT

            # 2. Apply window in the time domain if needed
            yTw, zTw = (self.yT * self.wT * wnd, zT * wnd) if wnd is not None else (self.yT * self.wT, zT)

            ## Define modelled and measured signals
            Z, y = zTw[0:,:], yTw[0:, :]
            useComplex = True
        else:
            # --------------------- FREQUENCY ----------------------------
            nw = len(self.sF) if self.sF is not None else 0         # Length of the adaptive lineshape window (in frequency domain)
            nw2 = int(nw/2)
            indxInRange = np.concatenate(tuple(self.freqBlocks[i].indxFreq for i in frqBlkIds))
            indxPadding = np.concatenate(tuple(np.concatenate([np.arange(self.freqBlocks[i].indxFreq[0]-nw2, self.freqBlocks[i].indxFreq[0]),
                                                               np.arange(self.freqBlocks[i].indxFreq[-1]+1, self.freqBlocks[i].indxFreq[-1]+nw2+1)%len(self.f)] ) \
                                        for i in frqBlkIds)) if nw2>0 else np.array([], dtype='int')   # Extra indices used for padding when convolving the signals with lineshape kernel in frequency domain

            if ( 'lshapeR' in evalParsH['.'].keys() and (any(evalParsH['.']['lshapeR']) or any(evalParsH['.']['lshapeI'])) ) or wnd is not None:
                zT, repRootNames = getFID(self.T, self.t, self.c0, self.f0, evalParsH, tau=0.0)            # 1. Compute the model signals

                # 1. Apply custom lineshape correction if defined
                if self.sT is not None:
                    zT *= self.sT

                # 2. Apply window in the time domain if needed
                yTw, zTw = (self.yT * self.wT * wnd, zT * wnd) if wnd is not None else (self.yT * self.wT, zT)

                # 3. Compute the spectra
                zF = np.fft.fftshift(np.fft.fft(zTw, len(self.f), axis=0), axes=0) / np.sqrt(len(self.f))
                zFinRange = zF[indxInRange, :]
                zFPadding = zF[indxPadding, :]
                yF = np.fft.fftshift(np.fft.fft(yTw, len(self.f), axis=0), axes=0) / np.sqrt(len(self.f))
                yFinRange = yF[indxInRange, :]
            else:
                zFall, repRootNames = evalTreeF(self.T, self.f.take(np.concatenate([indxInRange, indxPadding])), self.t[1]-self.t[0], self.c0, self.f0, evalParsH)
                zFinRange, zFPadding = np.split(zFall, [len(indxInRange)] )
                yFinRange = self.yF[indxInRange, :]

                # Apply custom lineshape correction
                if self.sF is not None:
                    indxSplit = np.cumsum([self.freqBlocks[i].indxFreq.size for i in frqBlkIds])[:-1]     # Indices showing how to split the concatenated arrays xF, yF, zF, etc.
                    zFPadded = [np.vstack([y[:nw2, :], x, y[-nw2:, :]]) for x, y in\
                                        zip(np.split(zFinRange, indxSplit, axis=0),
                                            np.split(zFPadding, len(frqBlkIds), axis=0) )]
                    zFinRange = np.vstack([scipy.signal.fftconvolve(z, self.sF, 'valid') for z in zFPadded]) / np.sqrt(len(self.f))

            # Possibly update the phased signal if the first-order phasing parameter has changed
            phFinRange = np.exp(-1j*2*np.pi * evalParsH["."]["tau"][0] * (self.f.take(indxInRange)*self.c0-self.f0) - 1j*0 ).reshape((-1,1))   # The phasing term
            yFinRange *= phFinRange

            # Choose only components that are in the optimization range
            """indxFreq = np.flatnonzero((self.f<=self.freqBlocks[i].max)*(self.f>=self.freqBlocks[i].min))     # Indices of frequency points in the range
            nf = indxFreq.size
            bF = lambda nf : np.hstack(( np.ones((nf, 1)), np.linspace(-1,1, nf).reshape(-1,1), np.linspace(-1,1,nf).reshape(-1,1)**2, 1j*np.ones((nf, 1)), 1j*np.linspace(-1,1, nf).reshape(-1,1), 1j*np.linspace(-1,1,nf).reshape(-1,1)**2 ))         # Define baseline in the frequency domain
            """

            # Include the baseline
            bslnPoly = block_diag(*[self.freqBlocks[i].bF for i in frqBlkIds if self.freqBlocks[i].bF is not None])     # All baseline models padded with zeros; use only real-valued baselines if the model is real-valued
            if not useComplex:
                bslnPoly = bslnPoly[:, np.isreal(bslnPoly).all(axis=0)]
            #else: bslnPoly *= phFinRange
            nb = bslnPoly.shape[1]     # Total number of baseline terms

            # Define modelled and measured signals
            Z, y = np.hstack((zFinRange, bslnPoly)), yFinRange
        ns, nz = Z.shape     # Number of samples and (model signals + baselines)
        na = len(repRootNames)    # Number of model signals, and hence the resulting amplitudes
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
            key = (self.repRootNames[i], 'ampl', 0)
            if self.isAutofittable(key, customPriors=customPriors) and (autoKeys is None or key in autoKeys):
                # Set a Gaussian prior with supplied mean and variance
                spec = self.getPrior(key, customPriors=customPriors)
                m0[i], S0[i,i] = spec.p1, spec.p2
            else:
                ampl[i] = evalParsH[self.repRootNames[i]]['ampl'][0]
                if useComplex: ampl[i] *= np.exp(1j*evalParsH[self.repRootNames[i]]['phase'][0])    # Set possibly different phases for each amplitude
        iS0 = np.linalg.inv(S0)

        # 3.2. Global phase shift
        key = ('.', 'theta', 0)
        if self.isAutofittable(key, customPriors=customPriors) and (autoKeys is None or key in autoKeys):
            # Estimate theta using the closed form expression
            Zy = Z.conj().T.dot(y)
            ZZ = Z.conj().T.dot(Z)
            if ZZ.size > 0 and np.linalg.matrix_rank(ZZ.real) < ZZ.shape[0]:
                ZZ += (1e-09)*np.identity(ZZ.shape[0])          # Make sure ZZ is invertible if it is low rank
            Sc = np.linalg.inv(ZZ.real)
            theta = np.asscalar( 0.5*np.angle(Zy.T.dot(np.dot(Sc, Zy))) )
        else: theta = evalParsH['.']['theta'][0]
        if not useComplex:
            Z, y = Z.real, (y*np.exp(-1j*theta)).real
        else:
            Z, y = np.vstack([Z.real, Z.imag]), np.vstack([(y*np.exp(-1j*theta)).real, (y*np.exp(-1j*theta)).imag])

        # 3.3. Variance of noise
        key = ('.', 'sigma2', 0)
        if self.isAutofittable(key, customPriors=customPriors) and (autoKeys is None or key in autoKeys):
            spec = self.getPrior(key, customPriors=customPriors)
            a_sigma2, b_sigma2 = spec.p1, spec.p2
            sigma2 = None
        else:
            a_sigma2, b_sigma2 = None, None
            sigma2 = evalParsH['.']['sigma2'][0]

        # 3.4. TLS ratio, gamma
        key = ('.', 'gamma', 0)
        if self.isAutofittable(key, customPriors=customPriors) and (autoKeys is None or key in autoKeys):
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
        #self.ZZZ, self.yyy = Z, y
        result, ampl, sigma2, meta = log_likelihood(Z, y, ampl=ampl, sigma2=sigma2, \
            Gz=Gz, Gy=None, gamma=gamma, m0=m0, iS0=iS0, a_sigma2=a_sigma2, b_sigma2=b_sigma2, \
            funcType=funcType, robust=robust)
        diff_theta = np.asscalar( 1/2*np.angle(ampl[:na].T.dot(ampl[:na])) )   # Global phase
        theta = (theta + diff_theta + np.pi) % (2 * np.pi) - np.pi
        m_ampl = ampl*np.exp(-1j*diff_theta)
        m_ampl[:na] = m_ampl[:na].real
        if m_ampl[:na].sum() < 0:      # Make sure that all amplitudes are positive
            m_ampl = - m_ampl
            theta = (theta + +np.pi + np.pi) % (2 * np.pi) - np.pi
            if not useComplex: evalParsH['.']['theta'][0] = (evalParsH['.']['theta'][0] + np.pi + np.pi) % (2 * np.pi) - np.pi  # Always update the phase if it needs to be flipped
        gamma = meta['gamma']
        S_ampl = meta['ampl'][1]
        a_sigma2, b_sigma2 = meta['sigma2']

        mult = 1   # sum(m_ampl)     # Multiplier (can be used to output normalized amplitudes)
        for lbl, val in zip(self.repRootNames, np.abs(m_ampl[:na])):
            evalParsH[lbl]['ampl'][0] = np.asscalar(val) / mult
        evalParsH['.']['mult'][0] = mult
        evalParsH['.']['theta'][0] = theta            # Update the phase
        evalParsH['.']['sigma2'][0] = sigma2
        if gamma is not None: evalParsH['.']['gamma'][0] = gamma

        # Save and output the resulting signals zF and bF
        self.zF, self.bF = None, None
        if returnSignals:
            # Save the estimated signals
            if inTimeDomain:
                self.zF = np.fft.fftshift(np.fft.fft(zT, len(self.f), axis=0), axes=0) / np.sqrt(len(self.f))
            else:
                try:
                    # If there is zT variable computed already
                    zF = np.fft.fftshift(np.fft.fft(zT, len(self.f), axis=0), axes=0) / np.sqrt(len(self.f))
                    zFinRange = zF[indxInRange, :]
                except UnboundLocalError: pass        # If there is no zT variable. Don't do anything; computation has been performed in the frequency domain anyway
                self.zF = np.zeros((len(self.f),na), dtype=complex)
                self.zF[indxInRange,:] = zFinRange ### / Znrm[:, 0:na]
                if nz-na > 0:
                    self.bF = np.zeros((len(self.f),1), dtype=complex)
                    self.bF[indxInRange] = np.dot(bslnPoly, m_ampl[-(nz-na):])
            # Save the characteristics of the marginalized distributions
            for i in range(na):
                key=(self.repRootNames[i], 'ampl', 0)
                if self.isAutofittable(key, customPriors=customPriors) and (autoKeys is None or key in autoKeys):
                    self.smplDistF[key] = smplSpec_Gaussian(np.asscalar(np.abs(m_ampl[i])), np.asscalar(np.abs(S_ampl[i,i])))

            key=('.', 'sigma2', 0)
            if self.isAutofittable(key, customPriors=customPriors) and (autoKeys is None or key in autoKeys):
                self.smplDistF[key] = smplSpec_invGamma( a_sigma2, b_sigma2 )

        # Always save the baseline
        if not inTimeDomain and nz-na>0:
            self.bF = np.zeros((len(self.f),1), dtype=complex)
            self.bF[indxInRange] = np.dot(bslnPoly, m_ampl[-(nz-na):])

        meta['sigma2'] = (2.0, sigma2)
        meta['theta'] = theta       # distr = {"ampl":(m_ampl, S_ampl), "theta":theta, "sigma2":(a_sigma2, b_sigma2)}
        meta['ampl'] = (np.abs(m_ampl[:na]), S_ampl[:na, :na].real)
        return result, meta         # Output the log value and parameters of the marginalized distributions

    def _fnc_prior(self, evalParsH, parsKeys=None, customPriors=None):
        """Computes the prior functions. Priors will be computed only for parameters in the list parsKeys (if parsKeys is None -- all parameters will be included). Assumes that parameters that can be marginalized are not included in parsKeys."""
        if parsKeys == []:
            return 0
        else:
            if parsKeys is None:    # All parameters
                parsKeys = flatten(self.crntParsH).keys()
            return sum([self.getPrior(key, customPriors).evalPrior(arg=evalParsH[key[0]][key[1]][key[2]]) for key in set(parsKeys) if key[1] not in ['ampl', 'theta', 'sigma2']])

    def measure_noise(self, lims, lmda=5.0):
        """Measures the standard deviation of noise in the spectrum within the limits lims in ppm."""

        indxFreq = np.flatnonzero((self.f<=max(lims))*(self.f>=min(lims)))     # Indices of frequency points in the range

        yF = self.yF[indxFreq].ravel()
        yFbsln = whitsm(yF, lmda)
        yFnoise = yF - yFbsln
        #yFbsln[indxFreq] = self.yF[indxFreq] - whitsm(self.yF[indxFreq], 7.0)
        sigma2_est = np.sum(np.abs(yFnoise)**2) * (self.f.size/indxFreq.size) / self.t.size
        print("sigma2_est = {:.6f}".format(np.asscalar(sigma2_est)))
        return yF, yFbsln

    def set_shape(self, frqBlkIds=None, wnd=None):
        """Sets the custom lineshape sF and sT."""

        nt, nf = len(self.t), len(self.f)
        nw = config.MODEL_ShapeKernelSize         # Length of the adaptive lineshape window (in frequency domain)
        nw2 = int(nw/2)

        evalParsH = self.crntParsH

        if frqBlkIds is None:
            frqBlkIds = self.steps[-1].frqBlkIds

        # Compute a matrix of model signals Z, either in time or frequency domain
        if len(frqBlkIds) == 0:
            # -------------------------- TIME ----------------------------
            # 1. Compute model signals in time domain
            zT, repRootNames = getFID(self.T, self.t, self.c0, self.f0, evalParsH)            # 1. Compute the model signals

            # 2. Apply window in the time domain if needed
            yTw, zTw = (self.yT * self.wT * wnd, zT * wnd) if wnd is not None else (self.yT * self.wT, zT)

            # Define modelled and measured signals
            Z, y = zTw, yTw
        else:
            # --------------------- FREQUENCY ----------------------------
            indxInRange = np.concatenate(tuple(self.freqBlocks[i].indxFreq for i in frqBlkIds))
            indxPadding = np.concatenate(tuple(np.concatenate([np.arange(self.freqBlocks[i].indxFreq[0]-nw2, self.freqBlocks[i].indxFreq[0]),
                                                               np.arange(self.freqBlocks[i].indxFreq[-1]+1, self.freqBlocks[i].indxFreq[-1]+nw2+1)% nf] ) \
                                        for i in frqBlkIds)) if nw2>0 else np.array([], dtype='int')   # Extra indices used for padding when convolving the signals with lineshape kernel in frequency domain

            if ( 'lshapeR' in evalParsH['.'].keys() and (any(evalParsH['.']['lshapeR']) or any(evalParsH['.']['lshapeI'])) ) or wnd is not None:
                zT, repRootNames = getFID(self.T, self.t, self.c0, self.f0, evalParsH, tau=0.0)            # 1. Compute the model signals

                # 2. Apply window in the time domain if needed
                yTw, zTw = (self.yT * self.wT * wnd, zT * wnd) if wnd is not None else (self.yT * self.wT, zT)

                # 3. Compute the spectra
                zF = np.fft.fftshift(np.fft.fft(zTw, len(self.f), axis=0), axes=0) / np.sqrt(nf)
                zFinRange = zF[indxInRange, :]
                zFPadding = zF[indxPadding, :]
                yF = np.fft.fftshift(np.fft.fft(yTw, len(self.f), axis=0), axes=0) / np.sqrt(nf)
                yFinRange = yF[indxInRange, :]
            else:
                zFall, repRootNames = evalTreeF(self.T, self.f.take(np.concatenate([indxInRange, indxPadding])), self.t[1]-self.t[0], self.c0, self.f0, evalParsH)
                zFinRange, zFPadding = np.split(zFall, [len(indxInRange)] )
                yFinRange = self.yF[indxInRange, :]

            # Possibly update the phased signal if the first-order phasing parameter has changed
            yFinRange *= np.exp(-1j*2*np.pi * evalParsH["."]["tau"][0] * (self.f.take(indxInRange)*self.c0-self.f0) - 1j*0 ).reshape((-1,1))   # A shorter vector of yF restricted to the optimization range only

            # Choose only samples that are in the optimization range
            """indxFreq = np.flatnonzero((self.f<=self.freqBlocks[i].max)*(self.f>=self.freqBlocks[i].min))     # Indices of frequency points in the range
            nf = indxFreq.size
            bF = lambda nf : np.hstack(( np.ones((nf, 1)), np.linspace(-1,1, nf).reshape(-1,1), np.linspace(-1,1,nf).reshape(-1,1)**2, 1j*np.ones((nf, 1)), 1j*np.linspace(-1,1, nf).reshape(-1,1), 1j*np.linspace(-1,1,nf).reshape(-1,1)**2 ))         # Define baseline in the frequency domain
            """

            # Include the baseline
            bslnPoly = block_diag(*[self.freqBlocks[i].bF for i in frqBlkIds if self.freqBlocks[i].bF is not None])     # All baseline models padded with zeros
            nb = bslnPoly.shape[1]     # Total number of baseline terms

            # Define modelled and measured signals
            Z, y = np.hstack((zFinRange, bslnPoly)), yFinRange
        ns, nz = Z.shape     # Number of samples and (model signals + baselines)
        na = len(repRootNames)    # Number of model signals, and hence the resulting amplitudes

        mc = np.array([evalParsH[name]['ampl'][0] for name in self.repRootNames])           # First na results correspond to the actual amplitudes of components, the rest, if any, correspond to the baselines
        theta = evalParsH['.']['theta'][0]
        m_ampl = mc[:na]

        # 3.a Apply adaptive lineshape correction and reevaluate the amplitudes
        if len(frqBlkIds) == 0:
            # TODO: Estimate lineshape in the time domain as the ratio between the mesured and fitted FIDs
            pass
        elif nw2 > 0:
            # Padd the signal arrays on both ends, separately for each frequency range
            indxSplit = np.cumsum([self.freqBlocks[i].indxFreq.size for i in frqBlkIds])[:-1]     # Indices showing how to split the concatenated arrays xF, yF, zF, etc.
            zFPadded = [np.vstack([y[:nw2, :], x, y[-nw2:, :]]) for x, y in\
                                zip(np.split(zFinRange, indxSplit, axis=0),
                                    np.split(zFPadding, len(frqBlkIds), axis=0) )]
            xFPadded = [np.dot(z, mc[:na]*np.exp(1j*theta)).reshape(-1,1) for z in zFPadded]

            # Form the Toeplitz matrix of shifted arrays
            S = np.vstack([np.hstack([x[i:i+len(x)-2*nw2] for i in np.arange(2*nw2, -1, -1, dtype='int')]) for x in xFPadded])

            # Subtract the baseline from the measured data
            bFinRange = self.bF[indxInRange, :]   # np.dot(bslnPoly, mc[-(nz-na):]) if nz-na>0 else 0

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

    def set_bline(self):
        pass

    def evaluate(self, evalParsH=None, parsKeys=None, autoKeys=None, frqBlkIds=None, funcType=None, evaluatePriors=False, customPriors=None, robust=None, returnSignals=False):
        """Evaluates the objective function (logLikelihood + sum of logPriors).
           Inputs:
           evalParsH - hierarchical dictionary of parameters (node name -> parameter name -> list of parameters); use crntParsH by default
           parsKeys - list of parameter tuples (node name, parameter name, parameter index)
           frqBlkIds - list of indices of frequency blocks over which to evaluate the function; evaluate in time domain by default, []
           evaluatePriors - if True, will add values of priors to the likelihood function to compute the posterior. Only those priors specified by parsKeys will be evaluated. """

        #wnd = scipy.signal.cosine(self.t.size).reshape(-1,1)   # np.ones(self.yT.shape) #
        wnd = None

        if evalParsH is None:
            evalParsH = self.crntParsH

        result, meta = self._fnc_lklhd(evalParsH, frqBlkIds, autoKeys, funcType, customPriors=customPriors, returnSignals=returnSignals, robust=robust, wnd=wnd)

        if evaluatePriors:
            result += self._fnc_prior(evalParsH, parsKeys, customPriors=customPriors)

        return result, meta

    def optimize(self, parsKeys, autoKeys=None, frqBlkIds=None, funcType=None, evaluatePriors=False, nhop=None, verbose=True):
        """Optimization over the tree parameters selected in the parsKeys (list of tuples)."""

        parsKeys, autoKeys = self._prepareKeys(parsKeys, autoKeys, verbose=verbose)

        if len(parsKeys) > 0:
            # Define the objective function using a copy of the parameters dictionary
            evalParsH = copy.deepcopy(self.crntParsH)
            bounds = tuple((self.getPrior(key).min, self.getPrior(key).max) for key in parsKeys)
            initVals = [evalParsH[k[0]][k[1]][k[2]] for k in parsKeys]
            costFuncOpti = lambda x : -self.evaluate(updateFromFlat(evalParsH, parsKeys, x), parsKeys, autoKeys, frqBlkIds, funcType, evaluatePriors, robust=False)[0]

            # Call the optimization routine
            res = self._optimize(costFuncOpti, bounds, initVals, nhop=nhop, verbose=verbose)


        # Update the stored parameters
        updateFromFlat(self.crntParsH, parsKeys, res.x)    # Updated structure of all parameters
        if self.refChshKey in parsKeys: self.setCrntVal(key = self.refChshKey, val = res.x[parsKeys.index(self.refChshKey)])
        self.smplDistF.clear()

        # Re-evaluate the posterior
        result, meta = self.evaluate(None, parsKeys, autoKeys, frqBlkIds, funcType, evaluatePriors, returnSignals=True)

        if verbose:
            if len(parsKeys) > 0:
                print("Optimization finished. Posterior={:.4g}".format(result))
                print('Found values:')
                for key in parsKeys:
                    print("     {} = {:.5g}".format(str(key), self.getCrntVal(key)))

        return result, meta

    def sample(self, parsKeys=None, autoKeys=None, frqBlkIds=None, funcType=None, evaluatePriors=False, nwalkers=None, nsteps=None):
        """Samples the posterior distribution using the MCMC algorithm."""

        parsKeys, autoKeys = self._prepareKeys(parsKeys, autoKeys)
        result = {}
        self.smplDistF.clear()             # Clear the characteristics of marginal distributions

        # If no parameters are set for sampling, just evaluate the marginal posterior
        if len(parsKeys) == 0:
            # Nothing to sample; just evaluate the function
            value, meta = self.evaluate(autoKeys=autoKeys, frqBlkIds=frqBlkIds, funcType=funcType, evaluatePriors=evaluatePriors, returnSignals=True)

            m_ampl = np.array(meta['ampl'][0]).reshape(-1, 1)
            S_ampl = meta['ampl'][1]
            nrep = len(m_ampl)*250     # Number of repeats for each case to sample the amplitudes from the Gaussian distributions
            indx = [i for i, name in enumerate(self.repRootNames) if self.getPrior(key=(name, 'ampl', 0)).distr == 'Gaussian' and (name, 'ampl', 0) not in parsKeys]       # Indices of amplitudes that were not sampled
            smpl = np.linalg.cholesky(S_ampl[np.ix_(indx, indx)]).dot(np.random.randn(len(indx), nrep)) + m_ampl[indx].reshape(-1,1)                  # Multivariate Gaussian random samples
            for i, id in enumerate(indx):
                key = (self.repRootNames[id], 'ampl', 0)
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
        costFuncSmpl = lambda x : self.evaluate(updateFromFlat(evalParsH, parsKeys, x), parsKeys, autoKeys, frqBlkIds, funcType, evaluatePriors)

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
        indx = [i for i, name in enumerate(self.repRootNames) if self.getPrior(key=(name, 'ampl', 0)).distr == 'Gaussian' and (name, 'ampl', 0) not in parsKeys]       # Indices of amplitudes that were not sampled
        if len(indx) > 0:
            smpl = np.hstack([np.linalg.cholesky(np.squeeze(S_ampl[np.ix_(indx, indx, [i])])).dot(np.random.randn(len(indx), nrep)) \
                             + m_ampl[indx,i].reshape(-1,1) for i in range(S_ampl.shape[-1])])              # Multivariate Gaussian random samples
            for i, id in enumerate(indx):
                key = (self.repRootNames[id], 'ampl', 0)
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

    def correct_residual(self, frqBlkIds=None, marginalize=True, customPriors=None):
        # Corrects the model signals and the associated amplitudes by taking into account the residual in the Re channel (with denoising).
        savedSignalsExist = True
        if savedSignalsExist:
            # Need to (re-)evaluate the model if it has not been done before
            pass

        if frqBlkIds is None:
            frqBlkIds = self.steps[-1].frqBlkIds

        # Phase the input signal
        indxInRange = np.concatenate(tuple(self.freqBlocks[i].indxFreq for i in frqBlkIds))
        ph = np.exp(-1j*2*np.pi * self.crntParsH["."]["tau"][0] * (self.f*self.c0-self.f0) - 1j*self.crntParsH["."]["theta"][0] ).reshape((-1,1))
        yFphR = (self.yF * ph).real

        # 1. Compute and smooth the residual
        ampl = np.array([self.crntParsH[name]['ampl'][0] for name in self.repRootNames]).reshape(-1,1)
        xFR = (self.zF.dot(ampl).reshape(-1,1) + self.bF).real
        rFR = yFphR - xFR      # The residual in the fitting range
        sF = scipy.signal.wiener(rFR.real.ravel())[indxInRange].reshape(-1,1)   # + 1j*scipy.signal.wiener(rF.imag.ravel()).reshape(-1,1)

        # 3. Define the corrected spectra
        zT0, repRootNames = getFID(self.T, [0.0], self.c0, self.f0, self.crntParsH, tau=0.0)            # Values of FID at t=0.0 (total integrals)
        na = len(repRootNames)    # Number of model signals, and hence the resulting amplitudes
        bslnConst = zT0.real/(2*np.sqrt(len(self.f)))
        zFnull = self.zF[indxInRange].real - bslnConst           # Remove the constant baselines
        Zcorr = zFnull + (sF / ampl.T) * np.abs(zFnull**2)/(np.abs(zFnull**2)).sum(axis=1, keepdims=True) # Correction terms

        # Scale to keep the peaks' areas
        s = np.cumsum(zFnull.real, axis=0) / (bslnConst * len(self.f))    # Normalize by the total sum of the spectrum, zFnull.sum(axis = 0)
        for i in range(na):
            indx = np.where((0.01 < s[:, i]) & (s[:, i] < 0.99))[0]
            if len(indx)>0:
                Zcorr[:, i] *= zFnull[indx, i].real.sum() / np.abs(Zcorr[indx, i].real).sum()

        # Include the baseline
        bslnPoly = block_diag(*[self.freqBlocks[i].bF for i in frqBlkIds if self.freqBlocks[i].bF is not None])     # All baseline models padded with zeros; use only real-valued baselines if the model is real-valued
        bslnPoly = bslnPoly[:, np.isreal(bslnPoly).all(axis=0)]    # Keep only real-valued baseline terms
        nb = bslnPoly.shape[1]     # Total number of baseline terms
        nz = na + nb     # Number of samples and (model signals + baselines)

        # Determine, which amplitudes can be integrated out; select all aplitudes that are not Gaussianly distributed.
        m0 = np.zeros((nz, 1))         # Prior amplitudes
        S0 = np.where(np.identity(nz)>0, np.inf, 0)           # Prior covariance matrix of amplitudes (vague priors)
        for i in range(na):
            spec = self.getPrior(key=(self.repRootNames[i], 'ampl', 0), customPriors=customPriors)
            if spec.distr != 'Gaussian' or not marginalize:
                # Set a Gaussian (pseudo-)prior with zero variance
                m0[i] = evalParsH[self.repRootNames[i]]['ampl'][0]
                S0[i,i] = 0.0
            else:
                # Set a Gaussian prior with supplied mean and variance
                m0[i] = spec.p1
                S0[i,i] = spec.p2

        # Find new amplitudes
        mc, Sc, Sr, Q = leastSquares(np.hstack((Zcorr, bslnPoly)), yFphR[indxInRange], m0, S0, Gy=None, indxPositive=list(range(na)))

        # Save the corrected values of amplitudes and the signals
        self.zF[indxInRange, :] = Zcorr + bslnConst
        self.bF[indxInRange] = bslnPoly.dot(mc[na:]) - bslnConst.dot(mc[:na])
        for lbl, val in zip(self.repRootNames, np.abs(mc[:na])):
            self.setCrntVal((lbl, 'ampl', 0), val)       # .crntParsH[lbl]['ampl'][0] = np.asscalar(val)

        return np.abs(mc[:na])

    def sweep(self, key, lims=None, npts=50, reoptimize=False, frqBlkIds=None, evaluatePriors=False):
        # Evaluates the posterior and computes the amplitudes while sweeping the parameter parKey in the range lims
        par = self.getPrior(key)     # Settings for the prior distribution of this parameter key
        if lims is None: lims = (par.min, par.max)
        x_arr = np.linspace(min(lims), max(lims), npts)
        lpst_arr = np.zeros(x_arr.shape)
        lpri_arr = np.zeros(x_arr.shape)
        evalParsH = copy.deepcopy(self.crntParsH)      # Make a copy of the parameter dictionary that will be used for evaluation
        for i, x in enumerate(x_arr):
            evalParsH[key[0]][key[1]][key[2]] = x
            lpst_arr[i], meta = self.evaluate(evalParsH=evalParsH, frqBlkIds=frqBlkIds, funcType=None, evaluatePriors=True, parsKeys=[key], autoKeys=None, robust=False)
            lpri_arr[i] = par.evalPrior(arg=x)

        llkl_arr = lpst_arr - lpri_arr

        crntVal = self.crntParsH[key[0]][key[1]][key[2]]

        return x_arr, llkl_arr, lpri_arr, lpst_arr, crntVal

    def _prepareKeys(self, parsKeys, autoKeys=None, verbose=True, print_parameters=False):
        """Make sure that all parameter keys are relevant for the current Datum (e.g. no 4-tuple keys)."""
        parsKeys = set([]) if parsKeys is None else set(parsKeys)
        #autoKeys = set([]) if autoKeys is None else set(autoKeys)
        if autoKeys is not None: autoKeys = set(autoKeys)
        for key in list(parsKeys):
            if key[-2] == 'meta':
                parsKeys.remove(key)
                continue
            if len(key) == 4:
                parsKeys.remove(key)
                if key[0] in [self.parent.data.index(self), '.', None]:
                    parsKeys.update([key[1:]])
        parsKeys = sorted(list(parsKeys))

        # Detremine which parameters can be marginalized and remove them from the list of sampled values
        #parsKeys = [key for key in parsKeys if not self.isAutofittable(key)]

        if verbose:
            npar_auto = len([key for key in autoKeys if self.isAutofittable(key)]) if autoKeys is not None else 0
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
                if autoKeys is not None:
                    for key in autoKeys:
                        if self.isAutofittable(key):
                            print("     {}".format(str(key)))

        return parsKeys, autoKeys

    def modelled_signal(self, phased=True, baseline=False):
        # TODO: will be removed
        nt, nf = len(self.t), len(self.f)
        evalParsH = self.crntParsH
        tau, theta = 0.0, 0.0
        ampl = np.array([evalParsH[name]['ampl'][0] for name in self.repRootNames])
        if not phased:
            tau = evalParsH['.']['tau'][0]
            theta = evalParsH['.']['theta'][0]

        # Generate the signal
        zT, repRootNames = getFID(self.T, self.t, self.c0, self.f0, evalParsH, tau)
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
        """Plots the dataset."""

        # Phase the data
        ph = np.exp(-1j*2*np.pi * self.crntParsH["."]["tau"][0] * (self.f*self.c0-self.f0) - 1j*self.crntParsH["."]["theta"][0] ).reshape((-1,1))
        yFph = self.yF * ph
        if self.zF is not None:
            zF = self.zF * np.array([self.crntParsH[name]['ampl'][0] for name in self.repRootNames]).reshape(1, -1)
            xF = zF.sum(1).reshape(-1,1)
            if self.bF is not None:
                bF = self.bF
                xF += bF

        inRange, outRange = splitFreq([self.freqBlocks[blk] for blk in self.steps[0].frqBlkIds], f=self.f)
        rmsResidual = 0.0
        if outRange:
            supsRatio = ceil(yFph.size / (2**13))   # Subsampling ratio; take no more than 2^13 points
            allIndx = [np.append(r.indxFreq[:-1:supsRatio], r.indxFreq[-1]) for r in outRange]   # Make sure that the first and the last indices of each group are included
            gapsPos = np.cumsum([r.size for r in allIndx])        # Positions of gaps
            allIndx = np.concatenate(allIndx)
            f_outR = np.insert(self.f[allIndx], gapsPos, None)

            # Plot measured data
            yF_outR = np.insert(yFph[allIndx], gapsPos, None)
            ax_main.plot(f_outR, yF_outR.real if real else yF_outR.imag, '-', color=(0,0.58,0.86), linewidth=1.5, label='Measured data')

            # Plot the fitted model
            if xF is not None:
                xF_outR = np.insert(xF[allIndx], gapsPos, None)
                ax_main.plot(f_outR, xF_outR.real if real else xF_outR.imag, '-', color='r', label='Fitted model')

                # Plot the residuals
                ax_residual.plot(f_outR, (yF_outR - xF_outR).real if real else (yF_outR - xF_outR).imag, '-', color='darkkhaki')

        if inRange:
            allIndx = np.concatenate([r.indxFreq for r in inRange])
            gapsPos = np.cumsum([r.indxFreq.size for r in inRange])
            f_inR = np.insert(self.f[allIndx], gapsPos, None)

            # Plot measured data
            yF_inR = np.insert(yFph[allIndx], gapsPos, np.nan)     #  - 1*step.bFph[allIndx]
            ax_main.plot(f_inR, yF_inR.real if real else yF_inR.imag, '-', color=(0,0.58,0.86), linewidth=1.5, label='')

            # Plot the model components
            if showComponents and zF is not None:
                zF = (zF + 1*bF)
                zF_inR = np.insert(zF[allIndx, :], gapsPos, None, axis=0)
                for i, node in enumerate(self.repRootNames):
                    ax_main.plot(f_inR, zF_inR[:, i].real if real else zF_inR[:, i].imag, '-', linewidth=0.5, color=config.colrseq[i], label=node)

            # Plot the fitted model
            if xF is not None:
                xF_inR = np.insert(xF[allIndx], gapsPos, None)       #  - 1*step.bFph[allIndx]
                ax_main.plot(f_inR, xF_inR.real if real else xF_inR.imag, '-', color='r', label='')

                # Plot the residuals
                rF_inR = (yF_inR - xF_inR).real if real else (yF_inR - xF_inR).imag
                ax_residual.plot(f_inR, rF_inR, '-', color='darkkhaki')
                rmsResidual += np.sqrt(np.nanmean(np.abs(rF_inR)**2))

        # Show the residuals plot below the graph
        # Set ticks and labels
        #plt.setp(ax_main.get_xticklabels(), visible=False)
        ax_main.set_xlabel('')
        ax_main.ticklabel_format(scilimits=(-3,3))
        ax_residual.ticklabel_format(scilimits=(-3,3))
        ax_residual.set_xlabel('Chemical shift, ppm', horizontalalignment='right', x=1.0)
        # Show RMS of the residual
        ax_residual.text(0.01, 0.92, "RMS = {:.4g}".format(rmsResidual), fontsize=10,
                        horizontalalignment='left', verticalalignment='top', transform = ax_residual.transAxes)

        # Plot optimization limits
        if showRanges:
            for i, blk in enumerate(self.freqBlocks):
                if showRanges == 'all':
                    ax_main.axvspan(blk.min, blk.max, alpha=0.2 if i in self.steps[0].frqBlkIds else 0.05, facecolor='yellow')
                elif showRanges == 'active' and i in self.steps[0].frqBlkIds:
                    ax_main.axvspan(blk.min, blk.max, alpha=0.2, facecolor='yellow')

        if showLegend: ax_main.legend(loc=0)

        # Set the updated limits
        ax_main.relim()    # recompute the ax.dataLim
        ax_main.margins(0, 0.05)    # x and y margins in percentages
        ax_main.autoscale()    # update ax.viewLim using the new dataLim
        if ax_main.get_xlim()[1] > ax_main.get_xlim()[0]: ax_main.invert_xaxis()

        if returnSignals: return self.f, yFph, xF

    def evalForPlot(self, key, frqBlkIds=None, lims=None, npts=75):
        """Returns an array of argument values and the values of log likelihood, prior, and posterior."""
        result = {}
        par = self.getPrior(key)     # Settings for the prior distribution of this parameter key
        if lims is None: lims = (par.min, par.max)
        x_arr = np.linspace(min(lims), max(lims), npts)
        lpst_arr = np.zeros(x_arr.shape)
        lpri_arr = np.zeros(x_arr.shape)
        m_ampl_arr = np.zeros((len(self.repRootNames), npts))
        S_ampl_arr = np.zeros((len(self.repRootNames), len(self.repRootNames), npts))
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
            lpri = pars[0].evalPrior(arg=x) + pars[1].evalPrior(arg=y)
            return lpst, lpri

        lpst_arr, lpri_arr = np.vectorize(func)(*np.meshgrid(x_arr, y_arr, sparse=True))     # Return a table of f(x, y)
        llkl_arr = lpst_arr - lpri_arr
        crntVal = (self.crntParsH[keys[0][0]][keys[0][1]][keys[0][2]],
                   self.crntParsH[keys[1][0]][keys[1][1]][keys[1][2]])

        return (x_arr, y_arr), llkl_arr, lpri_arr, lpst_arr, crntVal

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

def cut_roi(xT, c0, f0, dt, band, recenter=True, subsample=True):
    """Applies a bandpass filter to the time-domain signal xT specified but cutoff frquencies defined in the tuple band."""
    # Center the signal to the middle of the new range in ppm
    old_center = f0 / c0       # Center of the initial ppm scale (in ppm)
    new_center = (max(band) + min(band)) / 2      # Center of the new ppm scale
    f0_new = new_center * c0
    nt = len(xT)
    t = np.linspace(0, (nt-1)*dt, nt)
    eT = np.exp(-1j*2*np.pi*(f0_new-f0)*t).ravel()
    yT = xT.ravel() * eT    # Shift the signal

    # Create a lowpass filter
    nyq = 0.5/dt
    b, a = scipy.signal.cheby1(8, 0.01, (max(band)-min(band))*c0*dt)
    #b, a = scipy.signal.iirdesign(wp=(max(band)-min(band))*c0*dt, ws=1.1*(max(band)-min(band))*c0*dt, gpass=0.1, gstop=30)
    yT = scipy.signal.filtfilt(b, a, yT).reshape(-1,1)
    return yT, c0, f0_new, dt

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
