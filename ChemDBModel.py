
## Class to work with the chemical database ##

class ChemDBModel():

    def __init__(self, chemDict = {}):
        self._entries = [v for k, v in chemDict.items()]     # A list of entries (of type chemSpec)

    def __len__(self):
        """Returns the number of elements in the database."""
        return len(self._entries)

    def __getitem__(self, indx):
        """Returns an item from the database either based on its name or its id."""

    def addEntry(self, newEntry):
        self._entries.append(newEntry)




import json
import numpy as np
import scipy.sparse as sps
from scipy.linalg import block_diag
from collections import namedtuple
import pickle as pickle
from itertools import chain
import copy
import emcee
from chemTree import *

# Functions for generating FIDs and optimization
from scipy import optimize
from scipy.optimize import minimize
import scipy as sp
import scipy.sparse
import scipy.linalg
from scipy.sparse.linalg import cg
from collections import OrderedDict, MutableMapping

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

    saveTree('Sugars', S, pars)
    S, pars = loadTree('Sugars')

    return S, pars

def setupSugarsMA():
    X = chemNode("Mixture")
    X.addChild(chemNodeDB("Water", alph = [parsSpec(0, 50)]))

    # Load the sugars subtree
    S, sugarsPars = loadTree('Sugars')
    X.addChild(S)

    # Create the acids subtree
    #A = chemNode("Acids")
    #X.addChild(chemNodeDB("Maleic acid"))
    #A.addChild(chemNodeDB("Citric acid"))
    #A.addChild(chemNodeDB("Malic acid"))

    # add QD nodes to the tree
    for node in X.items():
        if isinstance(node, chemNodeDB): node.dendrolize()

    # Set the reported flags (must be done in the end when the leaves are added to the tree)
    X.setReported(False)
    S.setReported(False)

    # Set up the model parameters
    pars = defaultTreePars(X)
    pars.update(sugarsPars)     # use the predefined QD parameters for sugars
    pars["Mixture"]["alph_rel"] = [-0.85]

    # Convert from relative values to absolute ones
    pars = rel2abs(X, pars)

    saveTree('SugarsMA', X, pars)
    X, pars = loadTree('SugarsMA')

    return X, pars

def setupSucroseMA():
    X = chemNode("Mixture")
    #X.addChild(chemNodeDB("Water", alph = [parsSpec(0, 50)]))

    Su = chemNode("Sucrose", chsh = [parsSpec(-0.05, 0.05)])
    Su.addChild(chemNodeDB("Sucrose-F", intn=1))
    Su.addChild(chemNodeDB("Sucrose-G", intn=1))
    X.addChild(Su)

    # Create the acids subtree
    X.addChild(chemNodeDB("Maleic acid"))

    # add QD nodes to the tree
    for node in X.items():
        if isinstance(node, chemNodeDB): node.dendrolize()
    X.setReported(False)

    # Assign the fitted parameters
    pars = defaultTreePars(X)
    pars["Sucrose-F-SPSY1"]["chshQD_rel"] = [0.1279, 0.1279]
    pars["Sucrose-F-SPSY1"]["jcplQD_rel"] = [0]
    pars["Sucrose-F-SPSY2"]["chshQD_rel"] = [0.0192,0.0337,0.0894,0.2890,-0.0665]
    pars["Sucrose-F-SPSY2"]["jcplQD_rel"] = [-0.0994,-0.2246,-0.4012,0.3173,-0.0479]
    pars["Sucrose-G-SPSY1"]["chshQD_rel"] = [0.1845,0.3602,0.1281,0.1974,-0.2918,0.6373,0.7598]
    pars["Sucrose-G-SPSY1"]["jcplQD_rel"] = [-0.0283,0.1973,-0.2828,0.1321,-0.8838,0.2803,-0.3675]
    pars["Mixture"]["alph_rel"] = [-0.85]

    # Convert from relative values to absolute ones
    pars = rel2abs(X, pars)

    saveTree('SucroseMA', X, pars)
    X, pars = loadTree('SucroseMA')

    return X, pars

def setupFructoseMA():
    X = chemNode("Mixture")
    #X.addChild(chemNodeDB("Water", alph = [parsSpec(0, 50)]))

    Fr = chemNode("Fructose", chsh = [parsSpec(-0.05, 0.05)])
    Fr.addChild(chemNodeDB("alpha-D-Fructofuranose", intn=0.0517))
    Fr.addChild(chemNodeDB("beta-D-Fructofuranose", intn=0.2246))
    Fr.addChild(chemNodeDB("alpha-D-Fructopyranose", intn=0.0149))
    Fr.addChild(chemNodeDB("beta-D-Fructopyranose", intn=0.7088))
    X.addChild(Fr)

    # Create the acids subtree
    X.addChild(chemNodeDB("Maleic acid"))

    # add QD nodes to the tree
    for node in X.items():
        if isinstance(node, chemNodeDB): node.dendrolize()
    X.setReported(False)

    # Assign the fitted parameters
    pars = defaultTreePars(X)
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
    pars["Mixture"]["alph_rel"] = [-0.85]

    # Convert from relative values to absolute ones
    pars = rel2abs(X, pars)

    saveTree('FructoseMA', X, pars)
    X, pars = loadTree('FructoseMA')

    return X, pars

def setupGlucose():
    Gl = chemNode("Glucose", chsh = [parsSpec(-0.05, 0.05)])
    aGl = chemNodeDB("alpha-D-Glucopyranose", intn=0.3750)
    bGl = chemNodeDB("beta-D-Glucopyranose", intn=0.6250)
    Gl.addChild(aGl)
    Gl.addChild(bGl)

    # add QD nodes to the tree
    for node in Gl.items():
        if isinstance(node, chemNodeDB): node.dendrolize()
    Gl.setReported(False)

    # Assign the fitted parameters
    pars = defaultTreePars(Gl)
    pars["alpha-D-Glucopyranose-SPSY1"]["chshQD_rel"] = [0.1231,0.1128,0.0962,0.1172,0.0520,0.1289,0.1466]
    pars["alpha-D-Glucopyranose-SPSY1"]["jcplQD_rel"] = [0.0169,-0.1084,0.1663,0.3671,-0.4402,-0.0515,0.2412]
    pars["beta-D-Glucopyranose-SPSY1"]["chshQD_rel"] = [0.1074,0.1005,0.1071,0.1194,0.0897,0.1309,0.1375]
    pars["beta-D-Glucopyranose-SPSY1"]["jcplQD_rel"] = [0.3562,-0.3018,0.2189,-0.3157,-0.2695,0.0124,-0.1188]
    pars["Glucose"]["alph_rel"] = [-0.85]

    # Convert from relative values to absolute ones
    pars = rel2abs(Gl, pars)

    # Reset parameter ranges to center them around the default values
    for node in chain(aGl.children(), bGl.children()):
        for i, par in enumerate(node.chshQD):
            val_abs = pars[node.name]['chshQD'][i]
            node.chshQD[i] = parsSpec(val_abs-0.2, val_abs+0.2, par.label, par.distr, par.mode, par.stdv)
        for i, par in enumerate(node.jcplQD):
            val_abs = pars[node.name]['jcplQD'][i]
            node.jcplQD[i] = parsSpec(val_abs-1, val_abs+1, par.label, par.distr, par.mode, par.stdv)
    pars = defaultTreePars(Gl)

    saveTree('Glucose', Gl, pars)
    Gl, pars = loadTree('Glucose')

    return Gl, pars

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

    saveTree('ScionExp2', X, pars)
    X, pars = loadTree('ScionExp2')

    return X, pars

def setupMixture():
    X = chemNode("Mixture")
    X.addChild(chemNodeDB("Water", alph = [parsSpec(0, 75)]))
    X.addChild(chemNodeDB("Ethanol (aq.)"))
    X.addChild(chemNodeDB("Citric acid"))

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

class Workspace():

    def __init__(self):
        self.series = []
        self.setTree(chemNode("Mixture"))
        self.settings = dict()       # A dictionary of settings for processing

    def getDfltVal(self, key):
        """Returns the DEFAULT value of the parameter key."""
        return self.dfltParsH[key[0]][key[1]][key[2]]

    def setDfltVal(self, key, val):
        """Updates the DEFAULT value of the parameter key."""
        self.dfltParsH[key[0]][key[1]][key[2]] = val

    def setTree(self, T, priors = None, dfltParsH = None):
        self.T = T
        self.parsSpecDict = {('.', 'mult', 0): parsSpec()}
        self.parsSpecDict.update({('.', 'ampl', i):parsSpec() for i, p in enumerate(self.T.repRoots())})
        self.dfltParsH = dfltParsH if dfltParsH is not None else defaultTreePars(self.T)

    def addSeries(self, name = ''):
        self.series.append(Series(parent=self, name=name))

    def optimize(self):
        pass

class Series():
    """Class for the data series (e.g. in reaction monitoring)."""

    def __init__(self, parent, name = ''):
        self.parent = parent     # The workspace that contains the tree
        self.name = name
        self.c0 = None
        self.f0 = None
        self.t = []
        self.f = []
        self.data = []               # A list of Datum structures
        self.steps = []              # Fitting steps; each entry is a set of parsKeys tuples
        self.freqBlocks = []      # a list of optimization frequency ranges
        self.parsSpecDict = {('.', 'tau', 0): parsSpec(-1e-04, 1e-04), \
                                ('.', 'theta', 0): parsSpec(-np.pi, np.pi)}    # Dictionary of parameter specifications that overwrites the parameters in the tree

    def __getattr__(self, attr):
        """Called with the dot notation for attributes not found in the class (e.g. parameters shared between many spectra in the series, c0, f0, etc.)."""
        return getattr(self.parent, attr)

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
            self.reset()
            # TODO: Check if new c0/f0 are the same as the old ones when loading the rest of the data

        nt = len(self.t)
        yT = (np.array(data[nt+6:2*nt+6]) + 1j*np.array(data[-nt:])).reshape(-1,1)
        self.data.append(Datum(yT, parent=self, name = path[path.rfind('\\')+1:path.rfind('.')], arrVal = len(self.data)+1 ))

    def reset(self, apod=0, nf=None):
        """Resets the computed spectra."""
        if nf is None:
            nf = len(self.t)     # Determine the number of samples in the full signal spectrum (possibly including zero-filling)
        self.f = (np.fft.fftshift(np.fft.fftfreq(nf, self.t[1]-self.t[0]))+self.f0)/self.c0
        self.addFreqBlock()   # Add the "all frequencies" block

        # Compute a window in the time domain
        self.wT = np.exp(-apod*self.t) if apod > 0 else 1

        for D in self.data:
            D.reset()

    def addFreqBlock(self, lims=None, indx=-1):
        """Adds a frequency block for optimization at certain position indx in the self.freqBlocks arrays."""
        if lims is None:
            self.freqBlocks = []         # Reset the frequency blocks and add the entire signal
            lims = (-1*float('inf'), float('inf'))
        # Choose only samples that are in the optimization range
        indxFreq = np.flatnonzero((self.f<=max(lims))*(self.f>=min(lims)))     # Indices of frequency points in the range
        nf = indxFreq.size
        bF = np.hstack(( np.ones((nf, 1)), np.linspace(-1,1, nf).reshape(-1,1), np.linspace(-1,1,nf).reshape(-1,1)**2, 1j*np.ones((nf, 1)), 1j*np.linspace(-1,1, nf).reshape(-1,1), 1j*np.linspace(-1,1,nf).reshape(-1,1)**2 ))         # Define baseline in the frequency domain
        bF = np.hstack(( np.ones((nf, 1)), np.linspace(-1,1, nf).reshape(-1,1), 1j*np.ones((nf, 1)), 1j*np.linspace(-1,1, nf).reshape(-1,1) ))         # Define baseline in the frequency domain
        #bF = np.hstack(( np.ones((nf, 1)), 1j*np.ones((nf, 1)) ))         # Define baseline in the frequency domain

        # Overwrite existing frequency block if requested
        if indx == -1 or indx > len(self.freqBlocks)-1:
            self.freqBlocks.append( freqSpec(min(lims), max(lims), indxFreq, bF) )
        else:
            self.freqBlocks[indx] = freqSpec(min(lims), max(lims), indxFreq, bF)

    def _fnc_prior(self, evalParsH=None, parsDblKeys=None):
        """Custom prior probability function. Can be used to describe dependencies among parameters in different planes. Use parsKeys to determine if the prior needs to be computed for the specific keys."""
        if evalParsH is None:
            evalParsH=[None]*len(self.data)

        return 0

    def evaluate(self, evalParsH=None, parsDblKeys=None, frqBlkIds=[]):
        """Evaluates the objective function (sum of logLikelihoods for each datum + sum of logPriors).
           evalParsH is a list of hierarchical dictionaries."""
        if evalParsH is None:
            evalParsH=[None]*len(self.data)

        result = 0

        # Evaluate the posteriors for all data planes
        for i, D in enumerate(self.data):
            result += D.evaluate(evalParsH[i], None, frqBlkIds)

        result += self._fnc_prior(evalParsH, parsDblKeys)

        return result

    def optimize(self, parsDblKeys, frqBlkIds=[]):
        """Optimization over the tree parameters selected in the parsDblKeys (list of tuples of the form: (datum_id, (node_name, parameter_name, parameter_id)), e.g. (2, ('Sucrose-F', 'chshQD', 5)) )."""

        def costFuncOpti(x):
            # Update crntParsH in all datum structures
            for k, v in zip(parsDblKeys, x):
                self.data[k[0]].crntParsH[k[1][0]][k[1][1]][k[1][2]] = v
            # Evaluate the function
            return self.evaluate(parsDblKeys=parsDblKeys, frqBlkIds=frqBlkIds)

        bounds = tuple((self.data[key[0]].getParSpec(key[1]).min, self.data[key[0]].getParSpec(key[1]).max) for key in parsDblKeys)
        initVals = [self.data[k[0]].crntParsH[k[1][0]][k[1][1]][k[1][2]] for k in parsDblKeys]
        res = optimize.basinhopping(costFuncOpti, initVals, \
              niter = 10, niter_success = 5, T = 10, disp = True, \
              minimizer_kwargs=dict(method="L-BFGS-B", bounds=bounds, tol=1e-6), \
              take_step=MyTakeStep())

        # Update the structure of all parameters
        for k, v in zip(parsDblKeys, res.x):
            self.data[k[0]].crntParsH[k[1][0]][k[1][1]][k[1][2]] = v

    def sample(self):
        """Samples the posterior distribution using the MCMC algorithm."""
        pass

class Datum():
    """A single data instance. Contains signals of a single NMR experiment."""

    def __init__(self, yT, parent, name = '', arrVal = 0):
        self.name = name
        self.parent = parent               # A series object that will contain this Datum
        self.yT = yT    # The acquired signal in time domain (FID) without any preprocessing
        self.yF = np.fft.fftshift(np.fft.fft(self.yT * self.wT, len(self.f), axis=0)) / np.sqrt(len(self.f))

        self.crntParsH = copy.deepcopy(self.dfltParsH)
        self.parsSpecDict = {}
        self.arrVal = arrVal      # Value of the arrayed parameter in the serial experiment (e.g., extent of reaction)
        self.yFph = None          # Phased spectrum
        self.xT = None            # Modelled signal
        self.xF = None
        self.xFph = None
        self.mdldPeaks = {}   # modelled peaks
        self.pckdPeaks = []   # Picked peaks

    def __getattr__(self, attr):
        """Called with the dot notation for attributes not found in the class (e.g. parameters shared between many spectra in the series, c0, f0, etc.)."""
        return getattr(self.parent, attr)

    def resetSignals(self):
        stepClass.f, stepClass.xT, stepClass.yF, stepClass.xF, stepClass.yFph, stepClass.xFph = None, None, None, None, None, None

        # Reset the tree
        for item in self.T.items():
            item.reset()

    def update(self):
            """Computes the spectral representation of the signal yT and updates the class parameters."""
            nf = len(self.f)

            # Compute the spectrum of the input signal if necessary
            if self.yF is None:
                stepClass.yF = np.fft.fftshift(np.fft.fft(self.yT * self.wT, nf, axis=0)) / np.sqrt(nf)

            # Phase the spectra
            """if self.xT is None and self.xFph is None:
                self.evaluate()     # Compute the model signal if nothing is known about it
            """

            # Compute (possibly new) phasing terms
            ph = np.exp(-1j*2*np.pi * self.crntParsH["."]["tau"][0] * (self.f*self.c0-self.f0) - 1j*self.crntParsH["."]["theta"][0] ).reshape((-1,1))

            # Get the data spectrum
            self.yFph = self.yF * ph
            if np.mean(self.yFph.ravel().real) < np.median(self.yFph.ravel().real):   # If the distribution is skewed to the left; i.e. only a few points are less than the most of them #sum(yFph.ravel().real) < 0:
                self.yFph = -1 * self.yFph

            # Still need the model spectrum if the evaluation was in time domain (xT is known but xFph is not)
            if self.xFph is None and self.xT is not None:
                xF = np.fft.fftshift(np.fft.fft(self.xT * self.wT, nf, axis=0)) / np.sqrt(nf)
                self.xFph = self.xF * ph
                if sum(self.xFph.ravel().real) < 0:
                    self.xFph = -1 * self.xFph

    def reset(self):
        self.yF = np.fft.fftshift(np.fft.fft(self.yT * self.wT, len(self.f), axis=0)) / np.sqrt(len(self.f))
        self.yFph = None
        self.xT = None
        self.xF = None
        self.xFph = None
        self.mdldPeaks = {}
        self.pckdPeaks = []

    def _fnc_lklhd(self, evalParsH=None, frqBlkIds=[], wnd=None, lockedPhase=True):
        """Computes the value of the likelihood function."""
        if evalParsH is None:
            evalParsH = self.crntParsH
        zT, xT, zF, xF = (None, None, [], [])      # Default results

        # ------------ Work either in time of frequency domain -----------
        if frqBlkIds == []:
            # -------------------------- TIME ----------------------------
            zT, repRootNames = getFID(self.T, self.t, self.c0, self.f0, evalParsH)            # 1. Compute the model signals

            # 2. Apply window in the time domain if needed
            yTw, zTw = (self.yT * wnd, zT * wnd) if wnd is not None else (self.yT, zT)

            # 3. Solve the least-squares problem to find the values of amplitudes
            ampl, theta = leastSquares(zTw, yTw, lockedPhase)
            xTw = np.dot(zTw, ampl*np.exp(1j*theta))
            self.xT = np.dot(zT, ampl*np.exp(1j*theta)) if wnd is not None else xTw         # Found signal (without windowing)
            self.zT = zT * ampl.reshape(1, -1)
            self.xF, self.xFph = None, None      #Reset the spectra, so they will be updated next time if needed to be displayed
            self.bFph, self.zFph = None, None

            # 4. Find the value of the cost function
            res = np.linalg.norm(yTw - xTw, 2)   # Only the squared difference

            # Update the evaluation parameters
            evalParsH["."]["theta"][0] = theta
            evalParsH["."]["ampl"] = ampl.ravel().tolist()

        else:
            # --------------------- FREQUENCY ----------------------------
            indxFreqAll = np.concatenate(tuple(self.freqBlocks[i].indxFreq for i in frqBlkIds))
            # Possibly update the phased signal if the first-order phasing parameter has changed
            ph = np.exp(-1j*2*np.pi * evalParsH["."]["tau"][0] * (self.f*self.c0-self.f0) - 1j*0 ).reshape((-1,1))
            yFph = self.yF * ph
            #if sum(self.yFph.ravel().real) < 0:
            #    self.yFph = -1 * self.yFph
            nf = self.yF.size

            yFinRange = yFph.take(indxFreqAll, axis=0)     # A shorter vector of yF restricted to the optimization range only
            indxSplit = np.cumsum([self.freqBlocks[i].indxFreq.size for i in frqBlkIds])[:-1]     # Indices showing how to split the concatenated arrays xF, yF, zF, etc.

            # Compute model signals in the frequency domain
            zFinRange, repRootNames = evalTreeF(self.T, self.f.take(indxFreqAll), self.t[1]-self.t[0], self.c0, self.f0, evalParsH)

            # Choose only samples that are in the optimization range
            """indxFreq = np.flatnonzero((self.f<=self.freqBlocks[i].max)*(self.f>=self.freqBlocks[i].min))     # Indices of frequency points in the range
            nf = indxFreq.size
            bF = lambda nf : np.hstack(( np.ones((nf, 1)), np.linspace(-1,1, nf).reshape(-1,1), np.linspace(-1,1,nf).reshape(-1,1)**2, 1j*np.ones((nf, 1)), 1j*np.linspace(-1,1, nf).reshape(-1,1), 1j*np.linspace(-1,1,nf).reshape(-1,1)**2 ))         # Define baseline in the frequency domain
            """

            # Include the baseline
            bslnPoly = block_diag(*[self.freqBlocks[i].bF for i in frqBlkIds])     # All baseline models padded with zeros
            nb = bslnPoly.shape[1]     # Total number of baseline terms

            # 3. Solve the least-squares problem to find the values of amplitudes
            mc, theta = leastSquares( np.hstack((zFinRange, bslnPoly)), yFinRange, lockedPhase, indxNonNeg=range(zFinRange.shape[1]))
            bFinRange = np.dot(bslnPoly, mc[-nb:])
            bF = np.zeros((nf,1), dtype=complex)
            bF[indxFreqAll] = bFinRange                     # Found baseline
            xFinRange = bFinRange + np.dot(zFinRange, mc[:-nb])    #  zFinRange = zFinRange * mc[:-nb].reshape(1, -1)         np.sum(zFinRange, axis=1).reshape(-1, 1)                   # Found signal
            xF = np.zeros((nf,1), dtype=complex)
            xF[indxFreqAll] = xFinRange
            zF = np.zeros((nf,zFinRange.shape[1]), dtype=complex)
            zF[indxFreqAll,:] = zFinRange * mc[:-nb].reshape(1, -1)

            ampl = mc[:-nb]

            evalParsH["."]["theta"][0] = theta
            evalParsH["."]["ampl"] = ampl.ravel().tolist()
            self.yFph = yFph * np.exp(-1j*theta)
            self.xFph = xF
            self.bFph = bF
            self.zFph = zF
            self.bT, self.zT = None, None

            # 3. Add the custom baseline component (possibly do this iteratively)
            """#bslnCust = whitsm(yFinRange - xF[indxFreqAll])
            bslnCust = yFinRange - xF[indxFreqAll]
            mc, theta = leastSquares( np.hstack((zFinRange[:,cmpnInRange], bslnPoly, bslnCust)), yFinRange, lockedPhase, indxNonNeg=range(zF.shape[1]))
            bF = np.zeros((nf,1), dtype=complex)
            bF[indxFreqAll] = np.dot(np.hstack((bslnPoly, bslnCust)), mc[-nb-1:]*np.exp(1j*theta))                         # Found baseline
            xF = bF + np.dot(zF[:,cmpnInRange], mc[:-nb-1]*np.exp(1j*theta))                    # Found signal
            """

            # 4. Find the value of the cost function
            res = np.linalg.norm((yFinRange - xFinRange*np.exp(1j*theta)), 2)   # Only the squared difference   #  .real

        return np.log(res).ravel()         # Output the log value

    def _fnc_prior(self, evalParsH=None, parsKeys=None):
        """Computes the prior functions. Priors will be computed only for parameters in the list parsKeys (if parsKeys is None -- all parameters will be included)."""
        if parsKeys == []:
            return 0
        else:
            if evalParsH is None:
                evalParsH = self.crntParsH
            if parsKeys is None:
                parsKeys = flatten(self.crntParsH).keys()
            return sum([self.getParSpec(key).evalPrior(arg=evalParsH[key[0]][key[1]][key[2]]) for key in parsKeys])

    def getCrntVal(self, key):
        """Returns the relative or absolute value of the parameter key."""
        return self.crntParsH[key[0]][key[1]][key[2]]

    def setCrntVal(self, key, val):
        """Updates the value of the parameter key."""
        self.crntParsH[key[0]][key[1]][key[2]] = val

    def getParSpec(self, key):
        """Returns the specification of a parameter in the current tree or tau."""
        # TODO: Make it nicer...
        try:
            return self.parsSpecDict[key]
        except KeyError:
            try:
                return self.parent.parsSpecDict[key]
            except KeyError:
                try:
                    return self.parent.parent.parsSpecDict[key]
                except KeyError:
                    return getattr(self.T[key[0]], key[1])[key[2]]

    def setParSpec(self, key, val):
        """Sets the specification of a parameter in the current tree or tau; val is of parsSpec type."""
        self.parsSpecDict[key] = val

    def evaluate(self, evalParsH=None, parsKeys=None, frqBlkIds=[]):
        """Evaluates the objective function (logLikelihood + sum of logPriors)."""
        return self._fnc_lklhd(evalParsH, frqBlkIds) + self._fnc_prior(evalParsH, parsKeys)

    def optimize(self, parsKeys, frqBlkIds=[]):
        """Optimization over the tree parameters selected in the parsKeys (list of tuples)."""
        costFuncOpti = lambda x : self._fnc_lklhd(updateFromFlat(self.crntParsH, parsKeys, x), frqBlkIds) + \
                        self._fnc_prior(updateFromFlat(self.crntParsH, parsKeys, x), parsKeys)
        bounds = tuple((self.getParSpec(key).min, self.getParSpec(key).max) for key in parsKeys)
        initVals = [self.crntParsH[k[0]][k[1]][k[2]] for k in parsKeys]
        res = optimize.basinhopping(costFuncOpti, initVals, \
              niter = 10, niter_success = 5, T = 10, disp = True, \
              minimizer_kwargs=dict(method="L-BFGS-B", bounds=bounds, tol=1e-6), \
              take_step=MyTakeStep())
        updateFromFlat(self.crntParsH, parsKeys, res.x)    # Updated structure of all parameters

    def sample(self):
        """Samples the posterior distribution using the MCMC algorithm."""
        pass

def whitsm(y, lmda=10):
    """Whittaker smoother. See: https://gist.github.com/zmeri/3c43d3b98a00c02f81c2ab1aaacc3a49"""
    m = len(y)
    E = sp.sparse.identity(m)
    d1 = -1 * np.ones((m),dtype='d')
    d2 = 3 * np.ones((m),dtype='d')
    d3 = -3 * np.ones((m),dtype='d')
    d4 = np.ones((m),dtype='d')
    D = sp.sparse.diags([d1,d2,d3,d4],[0,1,2,3], shape=(m-3, m), format="csr")
    z = sp.sparse.linalg.cg(E + lmda * (D.transpose()).dot(D), y)

    return z[0]

def leastSquares(Z, y, lockedPhase=True, indxNonNeg=None):
    """Solves a phased-constrained complex-valued least-squares problem, y=Zx for x; nb - number of baseline terms (columns in the end of Z)."""
    nz = Z.shape[1]      # Number of dimensions
    if lockedPhase and indxNonNeg is None:
        indxNonNeg = range(nz)
    ampl = np.zeros((nz,1))

    # x. Set up the priors
    S0 = np.identity(nz)
    m0 = np.zeros((nz, 1))
    delta = 1e+42
    iS0 = np.linalg.inv(S0) / (delta**2)

    ZZ = np.dot(Z.conj().T, Z)
    Zy = np.dot(Z.conj().T, y)
    if np.linalg.matrix_rank(ZZ) < nz:
       ZZ = ZZ + 0.000001*np.identity(nz)

    if lockedPhase:                 # Locked phase - real amplitudes
        Sc = np.linalg.inv((iS0 + ZZ).real)
        theta = 0.5*np.angle(np.dot(Zy.T, np.dot(Sc, Zy)))
        mc = np.dot(Sc, (Zy*np.exp(-1j*theta) + np.dot(iS0,m0)).real)
        mc[indxNonNeg] = np.maximum(mc[indxNonNeg], 0) if mc[indxNonNeg].sum() > 0 else np.minimum(mc[indxNonNeg], 0)     # Discard negative values in mc but keep the sign for baseline components
        if mc[indxNonNeg].sum() < 0:    # Make sure that amplitudes are positive
            mc = - mc
            theta = theta + np.pi
    else:
        theta = 0
        Sc = np.linalg.inv(iS0 + ZZ)
        mc = np.dot(Sc, (Zy+np.dot(iS0, m0)))

    theta = np.asscalar( (theta + np.pi) % (2 * np.pi) - np.pi )
    return mc, theta

def getParsTree(T, myOrder = ['chsh', 'alph', 'jcpl']):
    """Returns the tree of parameters P for a chemNode tree T. The variable myOrder defines the order in which the parameters will be sorted."""
    P = viewNode(T.name)
    if type(T) is not chemNodeDB:
        P.addChild(viewNode(name = tuple([T.name] + ['chsh'] + [0]) ))
        P.addChild(viewNode(name = tuple([T.name] + ['alph'] + [0]) ))
        for node in T.children():
            P.addChild(getParsTree(node))
    elif T.childCount() == 1 and T.child(0).childCount() == 1:      # A singlet (one QD node with one T node as a child)
        P.addChild(viewNode(name = tuple([T.child(0).name] + ['chshQD'] + [0]) ))
        P.addChild(viewNode(name = tuple([T.child(0).name] + ['alphQD'] + [0]) ))
    else:         # Several QD systems
        P.addChild(viewNode(name = tuple([T.name] + ['chsh'] + [0]) ))
        P.addChild(viewNode(name = tuple([T.name] + ['alph'] + [0]) ))

        # Fix faulty labels
        for node in T.children():
            for par, val in node.default_pars().items():
                for i in range(len(val)):
                    if not isinstance(getattr(node, par)[i].label, str):
                        old = getattr(node, par)
                        old[i] = parsSpec(old[i].min, old[i].max, label='', distr=old[i].distr, mode=old[i].mode, stdv=old[i].stdv)
                        setattr(node, par, old)

        newParsNodes = [tuple([node.name] + [par] + [i] + [getattr(node, par)[i].label]) for node in T.children() for par, val in node.default_pars().items() for i in range(len(val)) if "QD" in par]      # All new parameter tuples that will be added as children here; keep the label in the fourth element of the tuple
        for pars in sorted(newParsNodes, key = lambda par : [i for i, x in enumerate(myOrder) if x in par[1]][-1] ):
            P.addChild(viewNode(name = pars[0:3], alias = pars[3]))    #         + [node.aliasQD[i]]
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

def updateFromFlatDiff(hierDict, flatKeys, flatValsDiff):
    """Updates a hierarchical parameter dictionary with values in its flattened representation."""
    hierDict = copy.deepcopy(hierDict)
    for k, v in zip(flatKeys, flatValsDiff):
        hierDict[k[0]][k[1]][k[2]] += v
    return hierDict
