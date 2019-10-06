from collections import namedtuple
import numpy as np
import scipy.sparse as sps
import json
import pickle as pickle
from linear_sum_assignment import linear_sum_assignment
import weakref
import scipy
import config
import math
from itertools import product
from numba import njit
import numexpr as ne
import copy

# Ordered set class to store children of a node
import collections

import warnings
warnings.filterwarnings("ignore")

class OrderedSet(collections.MutableSet):

    def __init__(self, iterable=None):
        self.end = end = []
        end += [None, end, end]         # sentinel node for doubly linked list
        self.map = {}                   # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def discard(self, key):
        if key in self.map:
            key, prev, next = self.map.pop(key)
            prev[2] = next
            next[1] = prev

    def __iter__(self):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self, last=True):
        if not self:
            raise KeyError('set is empty')
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)

# --- Loading the database in JSON format ---
class chemSpec:
    """Class for database entires."""

    def __init__(self, name='', chshH=None, chshC=None, nSpinH=None, jcplHH=None, pairHH=None, multH=None, multC=None, chshLabileH=None, jcplHC=None, **kwargs):
        self.name = name
        self.chshH = chshH if chshH is not None else []       # List of chshH parsSpec's
        self.chshC = chshC if chshC is not None else []
        self.nSpinH = nSpinH if nSpinH is not None else [1]*len(self.chshH)    # List of int 1..3 indicating the number of spins for each chshH
        self.jcplHH = jcplHH if jcplHH is not None else []    # List of jcplHH parsSpec's
        self.pairHH = pairHH if pairHH is not None else [None]*len(self.jcplHH)     # List of tuples; each tuple contains indices of coupled protons

        # Find multiplicities for different spin systems
        self._spsyAsgnH = []   # list of size 1 x nSpinH; each entry is the index of spin system to which this proton is assigned
        self.assignSpsy()      # Compute assignment of spins to spin systems
        self.multH = multH if multH is not None or [] else [1]*self._nSpsyH       # Multiplicities of different spin systems (uncoupled, but with the same chemical shifts)
        self.multC = multC if multC is not None or [] else [1]*len(self.chshC)

    def assignSpsy(self):
        """Determines spin systems based on coupling between chshH."""
        if len(self.chshH) == 0:
            # No protons
            self._nSpsyH = 0
            self._spsyAsgnH.clear()
        else:
            ind = np.array([pair[:2] for pair in self.pairHH if pair[0] is not None and pair[1] is not None])     # Indices of used J-couplings, if any
            if len(ind) > 0:
                # There is some coupling
                M = np.zeros((len(self.chshH),len(self.chshH)))    # Connectivity matrix; 1 if two spins are coupled
                M[ind[:,0], ind[:,1]] = 1
                self._spsyAsgnH.clear()
                (self._nSpsyH, newAsgnH) = sps.csgraph.connected_components(M, directed=True)    # Number of spin systems and assignment of spins to spin systems
                self._spsyAsgnH.extend(newAsgnH)
            else:
                # There are only uncoupled protons
                self._nSpsyH = len(self.chshH)
                self._spsyAsgnH.clear()
                self._spsyAsgnH.extend([*range(len(self.chshH))])

    def add_chshH(self, nSpin=1, mult=1, **kwargs):
        self.chshH.append(parsSpec(**kwargs))
        self.nSpinH.append(nSpin)
        self.multH.append(mult)
        # Add this new chemical shift as its own spin system initially (uncoupled)
        self._nSpsyH += 1
        self._spsyAsgnH.append(len(self.chshH)-1)

    def del_chshH(self, indx):
        self.chshH.pop(indx)
        self.nSpinH.pop(indx)
        # Reduce all indices that larger than indx by 1
        ##nSpsy_old = self._nSpsyH
        ##spsyAsgn_old = self._spsyAsgnH[indx]
        multBySpin = [self.multH[i] for i in self._spsyAsgnH if i != indx]     # Spin system multiplicities for each spin
        for i, j in product(range(len(self.pairHH)), [0, 1]):
            if self.pairHH[i][j] == indx or self.pairHH[i][j] is None:
                self.pairHH[i][j] = None
            elif self.pairHH[i][j] > indx:
                self.pairHH[i][j] -= 1
        self.assignSpsy()
        ##if self._nSpsyH < nSpsy_old: self._multH.pop(spsyAsgn_old)    # Remove the multiplicity if there is no longer this spin system
        # Update the multiplicities
        self.multH.clear()
        self.multH.extend([max([mult for mult, j in zip(multBySpin, self._spsyAsgnH) if j==i]) for i in range(self._nSpsyH)])        # Find multiplicities for each spin system

    def add_jcplHH(self, nSpin=1, mult=1, **kwargs):
        self.jcplHH.append(parsSpec(**kwargs))
        self.pairHH.append([None, None])

    def del_jcplHH(self, indx):
        self.jcplHH.pop(indx)
        self.pairHH.pop(indx)
        # Reduce all indices that larger than indx by 1
        multBySpin = [self.multH[i] for i in self._spsyAsgnH]     # Spin system multiplicities for each spin
        self.assignSpsy()
        # Update the multiplicities
        self.multH.clear()
        self.multH.extend([max([mult for mult, j in zip(multBySpin, self._spsyAsgnH) if j==i]) for i in range(self._nSpsyH)])        # Find multiplicities for each spin system

    def set_jcplHH(self, indx, pair, label=None):
        """Sets a jcpl constant index to the pair of protons; pair is a 2-list of proton indices."""
        if len(pair)!=2 or pair[0] == pair[1]: return False
        for p in self.pairHH:
            if set(pair) == set(p):
                print("This coupling is already defined.")
                return False

        multBySpin = [self.multH[i] for i in self._spsyAsgnH]     # Spin system multiplicities for each spin
        self.pairHH[indx] = pair

        self.assignSpsy()
        # Update the multiplicities
        self.multH.clear()
        self.multH.extend([max([mult for mult, j in zip(multBySpin, self._spsyAsgnH) if j==i]) for i in range(self._nSpsyH)])        # Find multiplicities for each spin system
        # Update the label for this j-coupling constant
        if label is None:
            label = self.chshH[pair[0]].label if pair[0] is not None else ''
            label += '-'
            label += self.chshH[pair[1]].label if pair[1] is not None else ''
        self.jcplHH[indx] = self.jcplHH[indx]._replace(label = label)

    def add_chshC(self, mult=1, **kwargs):
        self.chshC.append(parsSpec(**kwargs))
        self.multC.append(mult)

    def del_chshC(self, indx):
        self.chshC.pop(indx)
        self.multC.pop(indx)

    def getSpSy(self, mode='1H'):
        """Returns a list of spin systems."""
        if mode == '1H':
            chsh = self.chshH
            jcpl = self.jcplHH
            pair = self.pairHH
            nSpin = self.nSpinH         # Number of spins
            mult = self.multH           # Number of spin systems
        elif mode == '13C':
            chsh = self.chshC
            jcpl = []
            pair = []
            nSpin = [1]*len(chsh)
            mult = self.multC

        # Create an assignment matrix for chsh
        chshAsgn = []
        for i, n in enumerate(nSpin):
            chshAsgn.extend([i+1]*n)

        # Create an assignment matrix for jcpl
        if mode == '1H':
            jcplAsgn = np.zeros((len(chshAsgn), len(chshAsgn)), 'int32')
            for row in range(len(jcpl)):
                if pair[row][1] < pair[row][0]: pair[row].reverse()      # Make sure the order of the chemical shifts is right
                for i, m in enumerate(chshAsgn):
                    for j, n in enumerate(chshAsgn):
                        if pair[row][0] == m-1 and pair[row][1] == n-1 and j > i:
                            if (pair[row][-1] == 'skip_odd' and (i+j)%2==0) or (pair[row][-1] == 'skip_even' and (i+j)%2==1): continue      # Skip the para-hydrogens
                            jcplAsgn[i][j] = row+1

        else:
            jcplAsgn = None

        spsyBig = spsySpec(chsh, jcpl, chshAsgn, jcplAsgn)
        spsyAll = splitSpSy(spsyBig)
        for i, m in enumerate(mult):
            spsyAll[i] = spsyAll[i]._replace(mult=spsyAll[i].mult * m)

        return spsyAll

    def asdict(self):
        """Returns a dictionary representation of the chemSpec class object."""
        return {'name' : self.name,
                'chshH': self.chshH,
                'chshC': self.chshC,
                'nSpinH': self.nSpinH,
                'multH': self.multH,
                'multC': self.multC,
                'jcplHH': self.jcplHH,
                'pairHH': self.pairHH}

def readChemDB(fname='chemDB'):
    """Reads a chemDB in JSON format and convers it to dictionary of chemSpec class objects."""
    with open(fname+'.json', 'r') as fp:
        chemDB = json.load(fp)
    for k, v in chemDB.items():                   # Convert 2D arays of ranges to the namedtuple representation
        for kk in ['chshH', 'chshC', 'jcplHH']:
            v[kk] = array2parsSpec(v[kk])
        for kk in ['nSpinH', 'multH', 'multC']:      # Make sure that all single numbers are stored within arrays
            if v[kk].__class__ is int:
                v[kk] = [v[kk]]
    chemDB = {k:chemSpec(**v) for k,v in chemDB.items()}    # Conver orderedDict to chemSpec namedtuple
    return chemDB

def writeChemDB(chemDB, fname='result'):
    """Writes the chemDB in JSON format and stores it file name"""
    chemDB = {k:v.asdict() for k,v in chemDB.items()}   # Convert namedtuples to dictionaries
    for k, v in chemDB.items():                   # Convert 2D arays of ranges to the namedtuple representation
        for kk in ['chshH', 'chshC', 'jcplHH']:
            v[kk] = array2parsSpec(v[kk])
        for kk in ['nSpinH', 'multH', 'multC']:      # Make sure that all single numbers are stored within arrays
            if isinstance(v[kk], int):
                v[kk] = [v[kk]]
    with open(fname+'.json', 'w') as fp:
        json.dump(chemDB, fp)

parsSpec = namedtuple('parsSpec', 'min, max, label, distr, p1, p2, dval')
parsSpec.__new__.__defaults__ = (-np.inf, np.inf, '', 'Uniform', None, None, None)     # 'mode' specifies the location of the distribution maximum value
parsSpec.rel = lambda self, arg : abs2rel(self, arg)
parsSpec.abs = lambda self, arg : rel2abs(self, arg)
parsSpec.evalPrior = lambda self, arg : priorProb(self, arg)
parsSpec.dflt = lambda self : (self.min + self.max) / 2 if self.dval is None else self.dval

smplSpec = namedtuple('smplSpec', 'min, max, mean, median, var, q1, q3, p5, p95, hpd5')     # Specification of MCMC samples
smplSpec.__new__.__defaults__ = (-np.inf, np.inf, None, None, None, None, None, None, None, None)

peakSpec = namedtuple('peakSpec', 'chsh, intn, fwhm')
peakSpec.__new__.__defaults__ = (0, 1, None)     #

spsySpec = namedtuple('spsySpec', 'chsh, jcpl, chshAsgn, jcplAsgn, mult')
spsySpec.__new__.__defaults__ = (None, None, None, None, None, 1)

freqSpec = namedtuple('freqSpec', 'min, max, indxFreq, bslnOrder, bF')
freqSpec.__new__.__defaults__ = (-float('inf'), float('inf'), np.array([]), (None, None), np.array([]))
freqSpec.__str__ = lambda self : '{:.2f} ... {:.2f}'.format(self.min, self.max) if not (self.min == -float('inf') and self.max == float('inf')) else 'Entire range'
#freqSpec.indxFreq = lambda self, f : np.flatnonzero((f<=self.max)*(f>=self.min))    # Indices of the frequency vector f that fall into the current range
#freqSpec.bF()

def splitFreq(inRange, f=None):
    """Given a list of freqSpec tuples, divides the frequency range -inf to +inf into lists of disjoint intervals: inRange and outRange by merging overlapping optimization ranges."""

    if inRange:
        inRange = mergeFreq(inRange, f)      # Merged and sorted list of freqRanges
        outRange = []
        if not np.isinf(inRange[0].min):
            lwr = -np.inf
            upr = inRange[0].min
            outRange.append(freqSpec(min=lwr, max=upr, \
                            indxFreq = np.concatenate([ np.flatnonzero((f<=upr)*(f>=lwr)), [inRange[0].indxFreq[0]] ] ) if f is not None else np.array([], dtype = int) ) )
        for i in range(len(inRange)-1):
            lwr = inRange[i].max
            upr = inRange[i+1].min
            outRange.append(freqSpec(min=lwr, max=upr, \
                            indxFreq = np.concatenate([ [inRange[i].indxFreq[-1]], np.flatnonzero((f<=upr)*(f>=lwr)), [inRange[i+1].indxFreq[0]] ] ) if f is not None else np.array([], dtype = int) ) )
        if not np.isinf(inRange[-1].max):
            lwr = inRange[-1].max
            upr = np.inf
            outRange.append(freqSpec(min=lwr, max=upr, \
                            indxFreq = np.concatenate([ [inRange[-1].indxFreq[-1]], np.flatnonzero((f<=upr)*(f>=lwr)) ] ) if f is not None else np.array([], dtype = int) ) )
    else:
        outRange = [freqSpec(min=-np.inf, max=np.inf, \
                        indxFreq = np.arange(len(f)) if f is not None else np.array([], dtype = int) )]  # Infinite interval

    return inRange, outRange

def mergeFreq(intervals, f=None):
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
                merged[-1] = freqSpec(min=lower.min, max=upper_bound, indxFreq = np.flatnonzero((f<=upper_bound)*(f>=lower.min)) if f is not None else np.array([], dtype = int) )  # replace by merged interval
            else:
                merged.append(higher)
    return merged

def rel2abs(parsSpec, rel=0):
    """Converts between relative and absolute values of a parameter given its range in parsSpec."""
    return (parsSpec.max-parsSpec.min)*(rel+1)/2 + parsSpec.min

def abs2rel(parsSpec, arg=0):
    """Converts from absolute to relative values of a parameter given its range in parsSpec."""
    return (arg-parsSpec.min)/(parsSpec.max-parsSpec.min)*2 - 1

def priorProb(parsSpec, arg=0):
    """Computes the values of the (log) prior distribution at the relative argument arg."""
    if arg > parsSpec.max or arg < parsSpec.min:
        #print('Out of bounds')
        return -1e+5   # -np.inf     # -1e+100
    else:
        if parsSpec.distr == 'Uniform':
            return 0
        elif parsSpec.distr == 'Constant':
            if arg == parsSpec.p1:
                return 0
            else:
                return -np.inf
        elif parsSpec.distr == 'Gaussian':
            p2 = parsSpec.p2
            return -(arg-parsSpec.p1)**2 / (2*p2**2) - 1/2*np.log(2*np.pi*(p2**2))
        elif parsSpec.distr == 'Exponential':
            p1 = parsSpec.p1
            if arg < 0:
                return 0
            else:
                return np.log(p1) - p1*arg
        elif parsSpec.distr == 'Log-Normal':
            if arg <= 0:
                return -np.inf
            else:
                return -(np.log(arg)-parsSpec.p1)**2 / (2*parsSpec.p2**2) - np.log(arg*parsSpec.p2*np.sqrt(2*np.pi))
        elif parsSpec.distr == 'Inverse-Gamma':
            if arg <= 0:
                return -np.inF
            else:
                alpha = parsSpec.p1
                beta = parsSpec.p2
                return alpha*np.log(beta) - np.log(scipy.special.gamma(alpha)) - (alpha+1)*np.log(arg) - beta/arg

def array2parsSpec(arr):
    """Converts 2D arrays of min and max values to the array of parsRange namedtuples."""
    ans = arr
    if arr is not None:
        ans = [parsSpec(min=v[0], max=v[1], label=v[2] if len(v)>2 else '') for v in arr]
    else: ans = []
    return ans

def parsSpec2array(par):
    """Converts arrays of parsSpec namedtuples to 2D arrays of min and max values."""
    return [(v.min, v.max) for v in par]

def smplSpec_from_data(data):
    """Returns the statistics of the 1D np.array of samples, data, in the form of smplSpec."""
    data = data.ravel()
    p5, q1, median, q3, p95 = np.percentile(data, [5, 25, 50, 75, 95])
    return smplSpec(min = np.min(data),
                    max = np.max(data),
                    mean = np.mean(data),
                    median = median,
                    q1 = q1, q3 = q3,
                    p5 = p5, p95 = p95,
                    hpd5 = hpd(data, alpha=0.05),
                    var = np.var(data))

def smplSpec_Gaussian(mean, var):
    """Returns the statistics in the form of smplSpec from the parameters of a univariate Gaussian distribution."""
    stdv = np.sqrt(var)
    return smplSpec(mean = mean,
                    median = mean,
                    #q1 = q1, q3 = q3,
                    #p5 = p5, p95 = p95,
                    hpd5 = (mean-1.959963984540*stdv, mean+1.959963984540*stdv),
                    var = var)

def smplSpec_invGamma(a, b):
    """Returns the statistics in the form of smplSpec derived from the parameters of the inverse-Gamma distribution."""
    return smplSpec(min = 0,
                    mean = b/(a-1) if a > 1 else None,
                    #median = median,
                    #q1 = q1, q3 = q3,
                    #p5 = p5, p95 = p95,
                    #hpd5 = hpd(data, alpha=0.05),
                    var = b**2/((a-1)**2 * (a-2))) if a > 2 else None

def calc_min_interval(x, alpha):
    """
    This code was taken form the PyMC library https://github.com/pymc-devs/pymc
    Internal method to determine the minimum interval of a given width
    Assumes that x is sorted numpy array.
    """

    n = len(x)
    cred_mass = 1.0-alpha

    interval_idx_inc = int(np.floor(cred_mass*n))
    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    if len(interval_width) == 0:
        raise ValueError('Too few elements for interval calculation')

    min_idx = np.argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx+interval_idx_inc]
    return hdi_min, hdi_max

def hpd(x, alpha=0.05):
    """
    This code was taken form the PyMC library https://github.com/pymc-devs/pymc
    Calculate highest posterior density (HPD) of array for given alpha.
    The HPD is the minimum width Bayesian credible interval (BCI).
    :Arguments:
        x : Numpy array
        An array containing MCMC samples
        alpha : float
        Desired probability of type I error (defaults to 0.05)
    """

    # Make a copy of trace
    x = x.copy()
    # For multivariate node
    if x.ndim > 1:
        # Transpose first, then sort
        tx = np.transpose(x, list(range(x.ndim))[1:]+[0])
        dims = np.shape(tx)
        # Container list for intervals
        intervals = np.resize(0.0, dims[:-1]+(2,))

        for index in make_indices(dims[:-1]):
            try:
                index = tuple(index)
            except TypeError:
                pass

            # Sort trace
            sx = np.sort(tx[index])
            # Append to list
            intervals[index] = calc_min_interval(sx, alpha)
        # Transpose back before returning
        return np.array(intervals)
    else:
        # Sort univariate node
        sx = np.sort(x)
        return np.array(calc_min_interval(sx, alpha))


"""def readChemDB(name):
    '''Reads a chemDB in JSON format and convers it to dictionary of namedtuples.'''
    with open('chemDB.json', 'r') as fp:
        chemDB = json.load(fp)
    for k, v in chemDB.items():                   # Convert 2D arays of ranges to the namedtuple representation
        for kk in ['chshH', 'chshC', 'jcplHH', 'jcplHC']:
            v[kk] = array2parsSpec(v[kk])
        for kk in ['chshAsgnH', 'chshAsgnC']:      # Make sure that all single numbers are stored within arrays
            if v[kk].__class__ is int:
                v[kk] = [v[kk]]
    chemDB = {k:chemSpec(name=k,**v) for k,v in chemDB.items()}    # Conver orderedDict to chemSpec namedtuple
    return chemDB

def writeChemDB(chemDB, fname='result'):
    '''Writes the chemDB in JSON format and stores it file name'''
    chemDB = {k:v._asdict() for k,v in chemDB.items()}   # Convert namedtuples to dictionaries
    for k, v in chemDB.items():                   # Convert 2D arays of ranges to the namedtuple representation
        for kk in ['chshH', 'chshC', 'jcplHH', 'jcplHC']:
            v[kk] = array2parsSpec(v[kk])
        for kk in ['chshAsgnH', 'chshAsgnC']:      # Make sure that all single numbers are stored within arrays
            if v[kk].__class__ is int:
                v[kk] = [v[kk]]
    with open(fname+'.json', 'w') as fp:
        json.dump(chemDB, fp)"""

def printChemDB():
    """Prints chemDB."""
    for k, v in chemDB.items():
        print(k,v)

chemDB = readChemDB()     # Load the chemical database

# QD simulations
def transition_indices(n_spin, k=0):
    """Returns indices of singlestate transitons for the kth spin of total n_spin spins in an nxn matrix of intensities or frequencies."""
    k = n_spin-k-1
    diag_indx = [(i, i+2**k) for i in range(2**n_spin-2**k)]    # Indices of the 2**k off diagonal
    rows, cols = zip(*[diag_indx[i] for j in range(0, 2**n_spin, 2**(k+1)) for i in range(j, j+2**k)])
    return rows, cols

def group_peaks(omega, intn, maxWidth=0.1, isSplit = False):     # maxWidth = 0.1
    """Groups peaks located at frequencies omega and returns a reduced-sized arrays of aggregate peaks."""
    if hasattr(omega, "__len__") and len(omega) > 0:    # if omega is a non-empty array
        if maxWidth > 0:
            if max(omega) - min(omega) <= maxWidth:
                return [np.mean(omega)], [np.sum(intn)]     # The lists will be unpacked when insreted into omega[i:i+1]
            elif not isSplit:      # If the arrays have not been yet sorted and split at the largest gaps
                p = omega.argsort()
                omega, intn = omega[p], intn[p]
                indx = np.where(np.diff(omega) > maxWidth)[0]+1
            else:    # If arrays have been sorted and split along the largest gaps but the resulting groups are too large
                # TODO!!!!: Do something better...
                # Split along the largest gap
                indx = [np.diff(omega).argmax() + 1]

            omega = np.split(omega, indx)
            intn = np.split(intn, indx)
            i = 0
            while i < len(omega):
                omega[i:i+1], intn[i:i+1] = group_peaks(omega[i], intn[i], maxWidth, isSplit = True)    # Replace the i-th elements
                i += 1
        return omega, intn
    else: return [omega], [intn]

def spinop(n_spin):
    # Construct Carrtesian spin operators; will be used to build the Hamiltonian
    # 1. Define the Pauli matrices (for proton, a spin-1/2 particle)
    sig_x = sps.csr_matrix([[0, 1/2], [1/2, 0]], dtype='float64')
    sig_y = sps.csr_matrix([[0, -1j/2], [1j/2, 0]], dtype='complex128')
    sig_z = sps.csr_matrix([[1/2, 0], [0, -1/2]], dtype='float64')
    unit = sps.identity(2, dtype='float64', format='csr')
    # 2. Build Cartesian spin operators for each spin in the system and the transition probability matrix
    T = 0
    Lx = [None]*n_spin
    Ly = [None]*n_spin
    Lz = [None]*n_spin
    for i in range(n_spin):
        Lx[i] = sps.csr_matrix([1])
        Ly[i] = sps.csr_matrix([1])
        Lz[i] = sps.csr_matrix([1])
        T = sps.kron(sps.identity(2, 'uint', 'csr'), T, 'csr') + sps.kron([[0, 1], [1, 0]], sps.identity(pow(2,i), 'uint', 'csr'), 'csr')
        for j in range(n_spin):
            if i == j:
                Lx[i] = sps.kron(Lx[i], sig_x, format='csr')
                Ly[i] = sps.kron(Ly[i], sig_y, format='csr')
                Lz[i] = sps.kron(Lz[i], sig_z, format='csr')
            else:
                Lx[i] = sps.kron(Lx[i], unit, format='csr')
                Ly[i] = sps.kron(Ly[i], unit, format='csr')
                Lz[i] = sps.kron(Lz[i], unit, format='csr')

    return Lx, Ly, Lz, T

#@profile
def QDsimsGrpd2(H, T):
    """Simulates a QD system based on the spin frequencies and j couplings in Hz. See, e.g., http://www.users.csbsju.edu/~frioux/nmr/Speclab4.htm"""
    n_spin = int(math.log2(T.shape[0]))

    # 5. Compute the eigenvalues/eigenvectors of the Hamiltonian
    vH, uH = np.linalg.eigh(np.asarray(H))      # Need to make sure that the Hamiltonian is passed as an array, not a matrix
    #vH, uH = scipy.linalg.eigh(H)              # Possibly faster in some cases???

    # 6. Find which quantum states each eigenvector corresponds to and rearrange the columns of uH / values of vH. Use the largest entry in the eigenvectors to indicate this
    # TODO! NEEDS REVISION!!!
    #_, lbls = linear_sum_assignment(10000000 - abs(uH))
    #uH = uH[:, lbls]     # rearrange the columns of uH
    #vH = vH[lbls]

    # 7. Find the intensities and transition frequencies
    intn = np.dot(uH.T, T.dot(uH))**2      # Elementwise power!
    omega = abs(vH.reshape(-1,1) - vH)
    intn = np.triu(intn).flatten('F')
    omega = np.triu(omega).flatten('F')
    #return omega, intn

    p = np.flipud(intn.argsort())        # Sort the peaks from highest to lowest intensity
    #intn2 = np.cumsum(intn[p]**2)        # Cumulative sum of sorted squared intensities
    #p = p[0:max( n_spin*(2**(n_spin-1)), np.argmax(intn2/intn2[-1] > 0.99999) )]      # argmax will return the index of first occurence of element that evaluates to True
    #p = range(max( n_spin*(2**(n_spin-1)), np.argmax(intn2/intn2[-1] > 0.99999) ))
    #p = p[range(max( 0*n_spin*(2**(n_spin-1)), np.argmax(intn2/intn2[-1] > 0.99999) ))]

    p = p[0:n_spin*(2**(n_spin-1))]    # Keep only peaks corresponding to single transitions (assuming they are the largest)
    p = p[(intn[p] > 0.00000001)]                # Keep only the largest peaks
    # print(n_spin, len(p))

    # p_max = np.argmin( np.diff(np.log(intn[p]))[:2*n_spin*(2**(n_spin-1))] ) + 1     # All coefficients before the sharpest drop in their intensity but at most 2*n_spin*(2**(n_spin-1))
    # #p_max=1000
    # p = p[:p_max]
    # # print(p_max)

    omega = omega[p]
    intn = intn[p]
    intn = n_spin * intn / intn.sum()

    return omega, intn

def QDsims(freqArr, jcplMtx):
    """Simulates a QD system based on the spin frequencies and j couplings in Hz. See, e.g., http://www.users.csbsju.edu/~frioux/nmr/Speclab4.htm"""
    n_spin = len(freqArr)
    if jcplMtx is not None:
        # 1. Define the Pauli matrices (for proton, a spin-1/2 particle)
        sig_x = np.array([[0, 1/2], [1/2, 0]])
        sig_y = np.array([[0, -1j/2], [1j/2, 0]])
        sig_z = np.array([[1/2, 0], [0, -1/2]])
        unit = np.identity(2)
        # 2. Build Cartesian spin operators for each spin in the system and the transition probability matrix
        T = 0
        Lx = [None]*n_spin
        Ly = [None]*n_spin
        Lz = [None]*n_spin
        for i in range(n_spin):
            Lx[i] = 1
            Ly[i] = 1
            Lz[i] = 1
            T = np.kron(np.identity(2), T) + np.kron([[0, 1], [1, 0]], np.identity(pow(2,i)))
            for j in range(n_spin):
                if i == j:
                    Lx[i] = np.kron(Lx[i], sig_x)
                    Ly[i] = np.kron(Ly[i], sig_y)
                    Lz[i] = np.kron(Lz[i], sig_z)
                else:
                    Lx[i] = np.kron(Lx[i], unit)
                    Ly[i] = np.kron(Ly[i], unit)
                    Lz[i] = np.kron(Lz[i], unit)

        # 4. Build the Hamiltonian
        H = sps.lil_matrix((2**n_spin, 2**n_spin))
        for i in range(n_spin):
            H = H - freqArr[i] * Lz[i];
            for j in range(n_spin):
                if jcplMtx[i][j] != 0:
                    H = H + jcplMtx[i][j] * (np.dot(Lx[i],Lx[j]) + np.dot(Ly[i],Ly[j]) + np.dot(Lz[i],Lz[j]))

        # 5. Compute the eigenvalues/eigenvectors of the Hamiltonian
        vH, uH = np.linalg.eig(H)

        # 6. Find which quantum states each eigenvector corresponds to and rearrange the columns of uH / values of vH. Use the largest entry in the eigenvectors to indicate this
        # TODO! NEEDS REVISION!!!
        _, lbls = linear_sum_assignment(10000000 - abs(uH))
        uH = uH[:, lbls]     # rearrange the columns of uH
        vH = vH[lbls]

        # 7. Find the intensities and transition frequencies
        intn = abs(np.power(np.dot(uH.T, T.dot(uH)), 2))
        omega = abs(vH.reshape(-1,1) - vH)
        intn = np.triu(intn).flatten('F')
        omega = np.triu(omega).flatten('F')

        p = np.flipud(intn.argsort())        # Sort the peaks from highest to lowest intensity
        p = p[0:n_spin*(2**(n_spin-1))]    # Keep only peaks corresponding to single transitions (assuming they are the largest)
        ######    p = p[(p > 0.00000001)]                # Keep only the largest peaks

        omega = omega[p]
        intn = intn[p]
        intn = n_spin * intn / sum(intn)

        omega_Q1, intn_Q1 = [omega], [intn]

    else:
        omega_Q1 = [freqArr]
        intn_Q1 = [np.array([n_spin])]

    return omega_Q1, intn_Q1

def splitSpSy(big):
    """Splits a large spin system in the form of spsySpec namedtuple into a list of smaller disjoint spin systems. The result is a zip object containing separate disjoint spin systems and their corresponding intensities."""
    if big.jcplAsgn is None or big.jcpl is []:
        # Only singlets
        spsyAll = [spsySpec([chsh], [], [1], None, mult=big.chshAsgn.count(i+1)) for i, chsh in enumerate(big.chsh)]     # All chemical shifts
        #spsyAll = [spsySpec([i], [], [1], None) for i in big.chsh]     # All chemical shifts
        #intnAll = [big.chshAsgn.count(i+1) for i in range(len(big.chsh))]     # Multiplicities (intensities) for each chemical shift
    else:
        # Create a pseudo-assignment matrix -- connect spins with the same chemical shifts; will be used only to assign them to spin systems
        M = np.array(big.jcplAsgn)
        for s in set(big.chshAsgn):
            indx = [i for i, ss in enumerate(big.chshAsgn) if ss==s]
            M[np.array(indx), np.array(indx).reshape(-1, 1)] = 1
        (n_spsy, spsyAsgn) = sps.csgraph.connected_components(M, directed=True)    # Number of spin systems and assignment of spins to spin systems
        # Assign chemical shifts and j couplings to different spins (returns big vector/matrix of actual values instead of indices)
        chshSpinAll = [big.chsh[i-1] for i in big.chshAsgn]
        jcplSpinAll = [[big.jcpl[i-1] if i>0 else 0 for i in big.jcplAsgn[j]] for j in range(len(big.jcplAsgn))]
        # Collect the spins into the spin systems (select only certain entries in the above array and matrix)
        spsyAll, intnAll = [], []
        for s in range(n_spsy):       # Loop over new spin systems
            spinIndx = [i for i, val in enumerate(spsyAsgn) if val==s]      # Find indices of spins in the current spin system
            chshSpinNew = [chshSpinAll[i] for i in spinIndx]                # Lists of chch and jcpl assigned to the spins
            jcplSpinNew = [[jcplSpinAll[i][j] for j in spinIndx] for i in spinIndx]

            chshNew, jcplNew = [], []     # Empty lists of unique chsh and jcpl
            chshAsgnNew = [0] * len(spinIndx)
            jcplAsgnNew = [[0]*len(spinIndx) for _ in range(len(spinIndx))]

            # Assign chemical shifts
            for i, val in enumerate(chshSpinNew):                     # Loop over all spins in this spin system
                if not(any([chosen == val for chosen in chshNew])):   # If the chem shift of this spin is not chosen, add it to the list
                    chshNew.append(val)
                    chshAsgnNew[i] = len(chshNew)                     # Assign the last index of the chem shifts to the current spin
                else:
                    chshAsgnNew[i] = [chosen == val for chosen in chshNew].index(True)+1      # Find this chem shift in the list and assign its index to the current spin

            # Assign j couplings
            for i1, v1 in enumerate(jcplSpinNew):
                for i2, v2 in enumerate(v1):
                    if v2!=0 and v2 is not None:                       # Consider only valid jcpl values
                        if not(any([chosen == v2 for chosen in jcplNew])):
                            jcplNew.append(v2)
                            jcplAsgnNew[i1][i2] = len(jcplNew)
                        else:
                            jcplAsgnNew[i1][i2] = [chosen == v2 for chosen in jcplNew].index(True)+1
            if jcplNew == []:
                jcplAsgnNew = None

            # Create spin system and add it to the list of spin systems
            spsyNew = spsySpec(chshNew, jcplNew, chshAsgnNew, jcplAsgnNew)
            if not(any([chosen == spsyNew for chosen in spsyAll])):
                spsyAll.append(spsyNew)
                intnAll.append(1)
            else:
                intnAll[[chosen == spsyNew for chosen in spsyAll].index(True)] += 1

        for indx, mult in enumerate(intnAll):
            spsyAll[indx] = spsyAll[indx]._replace(mult=mult)

    return spsyAll

class treeNode:
    """ A generic tree node."""
    def __init__(self, name, alias=''):
        self.name = name
        self.alias = alias
        self._parent = None
        self._children = OrderedSet()
        self._treeBook = weakref.WeakValueDictionary({self.name:self})    # References to other nodes in the tree; the same dictionary is shared by all nodes

    def __str__(self):
        if self.alias is None or self.alias == '':
            return self.name
        else: return self.alias

    def __getitem__(self, key):
        return self._treeBook[key]

    def __getstate__(self):
        """This method is called when pickling called and the returned object is pickled as the contents for the instance, instead of the contents of the instance’s dictionary."""
        state = self.__dict__.copy()
        try:
            state.pop("_treeBook")              # Don't pickle the weak references
        except:
            pass
        return state

    def __setstate__(self, state):
        """Called with the unpickled state. In that case, there is no requirement for the state object to be a dictionary. Otherwise, the pickled state must be a dictionary and its items are assigned to the new instance’s dictionary."""
        self.__dict__ = state.copy()
        #self.__dict__["_treeBook"] = weakref.WeakValueDictionary({self.name:self})

    def setTreeBook(self):
        """Updates the map of weak references. Can be used after unpickling or for grafting."""
        newBook = weakref.WeakValueDictionary()
        for node in self.items():
            newBook.update({node.name:node})
            node._treeBook = newBook

    def rename(self, newName):
        """Checks for potential name conflicts before renaming the tree node."""
        if newName in [node.name for node in self.items()]:
            raise RuntimeError('The tree already contains a node called: {}'.format(newName))
        else:
            self._treeBook.pop(self.name)
            self.name = newName
            self._treeBook.update({self.name:self})

    def childCount(self):
        return len(self._children)

    def parent(self):
        return self._parent

    def child(self, pos):
        return list(self._children)[pos]

    def children(self):
        yield from self._children

    def siblID(self):
        """Position of the node among its siblings; 0 if there are no siblings."""
        if self._parent is not None:
            return list(self._parent._children).index(self)
        else: return 0

    def addChild(self, child):
        if child._parent is None:
            # Check for potential name conflicts
            commonNames = set([node.name for node in child.items()]).intersection([node.name for node in self.items()])
            if not commonNames:
                child._parent = self
                self._children.add(child)
                # Share the references to all other nodes in the tree
                self._treeBook.update(child._treeBook)
                child._treeBook = self._treeBook
                for node in child.descendants():      # Update all new descendants
                    node._treeBook = self._treeBook
            else:
                raise RuntimeError('The tree already contains node(s) called: {}'.format(commonNames))
        else:
            raise NotImplementedError('The parent of the child needs to be set to None first.')

    def insertChild(self, child, pos):
        """Inserts a new child at position pos."""
        """if pos < 0 or pos > len(self._children):
            return False

        self._children.insert(position, child)
        child._parent = self
        return True"""

    def clearChildren(self):
        for child in self.children():
            self.removeChild(child)

    def cut(self):
        """Removes a brach starting with self from the tree."""
        if not self.isRoot():
            self._parent.removeChild(self)
        return self

    def removeChild(self, child):
        """Removes a child."""
        self._children.discard(child)
        self.setTreeBook()     # Update the tree book
        child._parent = None
        child.setTreeBook()
        return child

    def findRoot(self):
        """The root of the tree."""
        if self.isRoot():
            return self
        else: return self._parent.findRoot()

    def makeRoot(self):
        """Sets teh current node to be the roor of the tree."""
        self._parent = None

    def replace(self, newTree):
        """Replaces the current node with the newTree."""
        newTree = newTree.findRoot()
        if self.isRoot():
            self.__dict__ = newTree.__dict__
        else:
            prnt = self._parent
            prnt.removeChild(self)
            prnt.addChild(newTree)

    def iterDepth(self, method="in-order", exclude=set()):
        """Depth-first iterator (a sequence of nodes that includes self)."""
        if method == "in-order":
            yield self      # Output the key and the node
            for c in self._children: yield from c.iterDepth("in-order")
        elif method == "post-order":
            for c in self._children: yield from c.iterDepth("post-order")
            yield self

    def descendants(self):
        """All descendants of the node (excluding itself)."""
        for child in self._children:
            yield child
            yield from child.descendants()

    def ancestors(self):
        """All ancestors of the node (excluding itself), including the root."""
        if self._parent is not None:
            yield self._parent
            yield from self._parent.ancestors()

    def siblings(self):
        """All children of the same parent (including self)."""
        if not self.isRoot():
            for sibl in self._parent.children:
                yield sibl

    def lineage(self):
        """All nodes linearly connected to itself, ancestors and descendants (including itself)."""
        yield self
        yield from self.descendants()
        yield from self.ancestors()

    def items(self):
        """All nodes in the tree of which itself is a node."""
        root = self.findRoot()
        yield self.findRoot()
        yield from root.descendants()

    def leaves(self):
        """A generator that returns all leaves of a (sub-) tree."""
        if self.isLeaf():
            yield self
        else:
            for node in self.descendants():
                if node.isLeaf():
                    yield node

    def isLeaf(self):
        "Returns TRUE if the node has no children."
        return self.childCount() == 0

    def isRoot(self):
        "Returns TRUE if the node has no parent."
        return (self._parent == None)

    def display(self, depth=0):
        "Displays the tree in text format."
        if depth == 0:
            print("{0}".format(self.name))
        else:
            print("\t"*depth, "{0}".format(self.name))
        depth += 1
        for child in self._children:
            child.display(depth)  # recursive call

    def log(self, depth = 0):
        """Another function to display the tree."""
        output = ""
        if depth > 0:
            output += "\t"*depth + "|------"

        output += str(self.name)
        if self.alias != '' : output += ' (' + self.alias + ')'
        output += '\n'

        depth += 1
        for child in self._children:
            output += child.log(depth)

        return output

    def __repr__(self):
        return self.log()

class viewNode(treeNode):
    """A class for nodes in the tree view model."""
    def __init__(self, name, alias='', nodeType=None, hidden=False, meta=None):
        super().__init__(name, alias)
        self.nodeType = nodeType
        self.hidden = hidden
        self.meta = meta             # Any metadata

class chemNode(treeNode):
    "Main class to store the chemical parameters in the tree"
    def __init__(self, name, chsh = None, alph = None, ampl = None, phase = None, intn = 1., alias=''):
        super().__init__(name, alias)
        self._reported = True
        self.chsh = chsh if chsh is not None else [parsSpec(min=-0.5, max=0.5)]
        self.alph = alph if alph is not None else [parsSpec(min=-5., max=25., dval=0.0)]
        self.ampl = ampl if ampl is not None else [parsSpec(min=0., max=np.inf, distr='Gaussian', p1=0.0, p2=np.inf, dval=1.0)]
        self.phase = phase if phase is not None else [parsSpec(distr='Uniform', min=-np.pi, max=np.pi, dval=0.0)]
        self.intn = intn         # Global intensity
        self.sT = []              # self-response
        self.uF = []             # frequency-domain response
        self.uT = []             # response that includes children/parents along the tree
        self.sPole = 0.          # self-pole determined by alph and chsh
        self.uPoles = 0.         # Poles that includes the effect of all parents
        self.oldTime = []

    def set_intn(self, intn):
        """Sets a new intensity value for the tree node and updates its signals."""
        # TODO! Scale the signals accordingly if possible
        self.intn = intn
        self.uT = []
        self.sT = []
        self.uF = []

    def default_pars(self):
        """Returns a dictionary of default parameters for the node."""
        return {"chsh":[self.chsh[0].dflt()], "alph":[self.alph[0].dflt()], 'ampl':[self.ampl[0].dflt()], 'phase':[self.phase[0].dflt()]}        # "alph":[self.alph.dflt]

    def priors(self):
        """Returns a dictionary of prior parameter specifications for the node."""
        return {"chsh":self.chsh, "alph":self.alph}

    def setReported(self, flag=True):
        """Self the _reported flag of the node."""
        if not self.isLeaf():     # Leafs can only have _reported set to True
            if flag:
                for chld in self._children:
                    chld.setReported(flag)
            elif not self.isRoot() and self._parent.isReported() != flag:
                self._parent.setReported(flag)
            self._reported = flag

    def isReported(self):
        """Returns the _reported flag of the node."""
        return self._reported

    def toggleReported(self):
        self.setReported(flag = not self._reported)

    def repRoots(self):
        """Returns root nodes of all reported subtrees."""
        root = self.findRoot()
        if not root.isReported():
            for v in root.iterDepth('in-order'):
                if v.isReported() and not v.parent().isReported():
                    yield v
        else:
            yield root
        # repRoots = [v for v in root.iterDepth('in-order') if v.isReported() and not v.parent().isReported()] if not root.isReported() else [root]

    def addChild(self, child):
        treeNode.addChild(self, child)
        self.setReported(self._reported)    # Update the reported flags (e.g. if self._reported was True, but child._reported is false, need to set child._reported to True)

    def insertChild(self, child, pos):
        treeNode.insertChild(self, child, pos)
        self.setReported(self._reported)    # Update the reported flags (e.g. if self._reported was True, but child._reported is false, need to set child._reported to True)
        """Inserts a new child at position pos."""
        """if pos < 0 or pos > len(self._children):
            return False

        self._children.insert(position, child)
        child._parent = self
        return True"""

    def removeChild(self, child):
        """Removes a child from position pos."""
        treeNode.removeChild(self, child)
        if self.isLeaf(): self.setReported(True)    # If the node becomes a leaf, it's reported flag must be set to True

    def reset(self):
        """Resets the oldPars, s(t), and u(t) to their default (empty) values. Everything will be recomputed at the next evaluation."""
        self.uT = []
        self.sT = []
        self.oldTime = []

    def getPoles(self, c0, chsh=[], alph=[], **kwargs):
        """Computes the poles and returns 1 if they have changed, 0 otehrwise"""
        newPole = - alph[0] + 1j*2*np.pi*c0*chsh[0]
        # If something has changed
        if self.sPole != newPole:
            self.sPole = newPole
            self.reset()     # Reset the time-domain signals, so they will be recomputed

    def propPoles(self, uPolePrnt = None):
        """Propagates offset poles to all children."""
        if uPolePrnt is None:
            uPolePrnt = 0. if self.isRoot() else self._parent.uPoles     # Set the offset pole to the uPole of the parent
        self.uPoles = self.sPole + uPolePrnt
        for chld in self.children():
            chld.propPoles(self.uPoles)

    #@profile
    def evalTime(self, t, c0, chsh=[], alph=[], **kwargs):
        "Computes the node's response u=sPoleIntn*exp(-alph*t+i*omega*t)"
        self.getPoles(c0, chsh, alph)
        if self.sT == [] or not np.array_equal(self.oldTime, t):
            self.sT = self.intn if self.sPole == 0 else self.intn * np.exp(np.outer(t, self.sPole)).ravel()
            self.oldTime = t

    def evalFreq(self, f, dt, c0, f0=0, tau=0):
        "Computes the node's response in the frequency domain assuming that all nodes have updated uPoles."
        # # Check if the signal needs to be reevaluated
        # if self.uF == []:
        self.uF = np.zeros((len(f), 1), dtype='complex128').ravel()
        for chld in self.children():
            chld.evalFreq(f, dt, c0, f0, tau)
            self.uF += chld.uF
        self.uF *= self.intn

class chemNodeQD(chemNode):
    "Class for QD-computed node, inherited from chemNode"

    #@profile
    def __init__(self, name, spsy, chsh = None, alph = None, alphQD = None, ampl = None, phase = None, intn = 1., alias=''):
        chemNode.__init__(self, name, chsh, alph, ampl, phase, intn, alias)
        self.chshQD = spsy.chsh
        self.jcplQD = spsy.jcpl
        self.intn = spsy.mult
        self.alphQD = alphQD if alphQD is not None else [parsSpec(min=-5.0, max=25.0, label=c.label, dval=0) for c in self.chshQD]
        self.chshAsgn = spsy.chshAsgn
        self.jcplAsgn = spsy.jcplAsgn
        self.oldParsQD = {"chsh":None, "jcpl":None}
        self.qPoles = [None]*len(self.chshQD)                # QD poles excluding the effects of line-broadedning although including any linebroadening due to peak aggregation
        self.qPolesIntn = [None]*len(self.chshQD)

        # Create spin operators
        """# Assign chemical shifts and j coupling values to spins in the system
        freqSpin = [chshQD[i-1] for i in self.chshAsgn]       # Frequencies of each spin after assignment
        jcplSpin = [[jcplQD[i-1] if i>0 else 0 for i in self.jcplAsgn[j]] for j in range(len(self.jcplAsgn))] if len(jcplQD)>0 else None"""

        n_spin = len(self.chshAsgn)
        self.spinopsL = [0.0]*len(self.chshQD)
        self.spinopsJ = [0.0]*len(self.jcplQD)
        Lx, Ly, Lz, self.TM = spinop(n_spin)       # Cartesian spin operators used to construct the Hamiltonian and the Transition matrix
        for i in range(n_spin):
            self.spinopsL[self.chshAsgn[i]-1] += Lz[i]
            if self.jcplAsgn is not None:
                for j in range(n_spin):
                    if self.jcplAsgn[i][j] != 0:
                        self.spinopsJ[self.jcplAsgn[i][j]-1] += (Lx[i].dot(Lx[j]) + Ly[i].dot(Ly[j]) + Lz[i].dot(Lz[j])).real
        #self.spinopsL = [sps.csr_matrix(m) for m in self.spinopsL]
        #self.spinopsJ = [sps.csr_matrix(m) for m in self.spinopsJ]

    def addChild(self, child):
        """Add a terminal node and keep the value of its chemical shift."""
        if type(child) is not chemNodeT:
            raise RuntimeError("Only terminal nodes can be added to a chemQD node.")
        else:
            chemNode.addChild(self, child)
            child.reset()

    def insertChild(self, child, pos):
        chemNode.insertChild(self, child, pos)
        child.reset()

    def removeChild(self, child):
        """Removes a child from position pos."""
        pos = child.siblID()
        self.chshUsed.pop(pos)
        return chemNode.removeChild(self, child)

    def default_pars(self):
        """Returns a dictionary of default parameters for the node."""
        pars = chemNode.default_pars(self)
        pars["chshQD"] = [par.dflt() for par in self.chshQD]
        pars["alphQD"] = [par.dflt() for par in self.alphQD]    # [0]*len(self.alphQD)
        if len(self.jcplQD) > 0: pars["jcplQD"] = [par.dflt() for par in self.jcplQD]
        return pars

    def priors(self):
        """Returns a dictionary of prior parameter specifications for the node."""
        result = chemNode.priors(self)
        result["chshQD"] = [par for par in self.chshQD]
        result["alphQD"] = [par for par in self.alphQD]
        if len(self.jcplQD) > 0: result["jcplQD"] = [par for par in self.jcplQD]
        return result

    def reset(self):
        """Resets the saved old parameters in the node. Evrything will be recomputed on the next step."""
        chemNode.reset(self)
        self.oldParsQD["chsh"] = None
        self.oldParsQD["jcpl"] = None

    #@profile
    def getPoles(self, c0, chsh=[], alph=[], chshQD=[], alphQD=[], jcplQD=[], **kwargs):
        """Computes the poles for all peaks including QD simulations if needed."""
        chemNode.getPoles(self, c0, chsh, alph)     # Compute sPole

        ## Find absolute values of the QD parameters
        chshQD = c0*np.array(chshQD)          # List of absolute values of chemical shifts (in Hz)
        alphQD = np.array(alphQD)
        jcplQD = np.array(jcplQD)

        # Run the QD simulations only if the parameters have changed (assume that chsh, alph, and t have also changed)
        mind_chshQD = np.concatenate([[abs(cs2 - cs1) for cs2 in chshQD[i+1:]] for i, cs1 in enumerate(chshQD)] + [[np.inf]]).min()     # Minimum distance between any two chemical shifts in this spin system; inf if theer is only one chemical shift
        if self.oldParsQD["chsh"] is None or self.oldParsQD["jcpl"] is None or any(self.oldParsQD["jcpl"] != jcplQD) \
                                          or ( any( abs(self.oldParsQD["chsh"] - chshQD) > min(config.QD_RerunQDchshThreshold*c0, 0.5*mind_chshQD) ) \
                                               and self.jcplQD != []):
            # QD simulations
            n_spin = len(self.chshAsgn)
            if len(jcplQD) > 0:
                #print("Running QD simulations.")
                # 4. Build the Hamiltonian
                H = np.zeros((2**n_spin, 2**n_spin), dtype='float64')
                try:
                    for chsh, spinop in zip(chshQD, self.spinopsL):
                        H = H - chsh * spinop
                    for jcpl, spinop in zip(jcplQD, self.spinopsJ):
                        H = H + jcpl * spinop
                except AttributeError:
                    # TO BE REMOVED IN LATER VERSIONS. LEFT FOR COMPATIBILITY
                    print('Using the old version of QM model.')

                    def spinop(n_spin):
                        # Construct Carrtesian spin operators; will be used to build the Hamiltonian
                        # 1. Define the Pauli matrices (for proton, a spin-1/2 particle)
                        sig_x = sps.csr_matrix([[0, 1/2], [1/2, 0]])
                        sig_y = sps.csr_matrix([[0, -1j/2], [1j/2, 0]])
                        sig_z = sps.csr_matrix([[1/2, 0], [0, -1/2]])
                        unit = sps.identity(2, format='csr')
                        # 2. Build Cartesian spin operators for each spin in the system and the transition probability matrix
                        T = 0
                        Lx = [None]*n_spin
                        Ly = [None]*n_spin
                        Lz = [None]*n_spin
                        for i in range(n_spin):
                            Lx[i] = sps.csr_matrix([1])
                            Ly[i] = sps.csr_matrix([1])
                            Lz[i] = sps.csr_matrix([1])
                            T = sps.kron(sps.identity(2, 'uint', 'csr'), T, 'csr') + sps.kron([[0, 1], [1, 0]], sps.identity(pow(2,i), 'uint', 'csr'), 'csr')
                            for j in range(n_spin):
                                if i == j:
                                    Lx[i] = sps.kron(Lx[i], sig_x, format='csr')
                                    Ly[i] = sps.kron(Ly[i], sig_y, format='csr')
                                    Lz[i] = sps.kron(Lz[i], sig_z, format='csr')
                                else:
                                    Lx[i] = sps.kron(Lx[i], unit, format='csr')
                                    Ly[i] = sps.kron(Ly[i], unit, format='csr')
                                    Lz[i] = sps.kron(Lz[i], unit, format='csr')

                        return Lx, Ly, Lz, T

                    self.spinopsL = [0]*len(self.chshQD)
                    self.spinopsJ = [0]*len(self.jcplQD)
                    Lx, Ly, Lz, self.TM = spinop(n_spin)       # Cartesian spin operators used to construct the Hamiltonian and the Transition matrix
                    for i in range(n_spin):
                        self.spinopsL[self.chshAsgn[i]-1] += Lz[i]
                        if self.jcplAsgn is not None:
                            for j in range(n_spin):
                                if self.jcplAsgn[i][j] != 0:
                                    self.spinopsJ[self.jcplAsgn[i][j]-1] += (Lx[i].dot(Lx[j]) + Ly[i].dot(Ly[j]) + Lz[i].dot(Lz[j]))
                    self.spinopsL = [sps.dia_matrix(m) for m in self.spinopsL]
                    self.spinopsJ = [sps.dia_matrix(m) for m in self.spinopsJ]

                    for chsh, spinop in zip(chshQD, self.spinopsL):
                        H -= chsh * spinop
                    for jcpl, spinop in zip(jcplQD, self.spinopsJ):
                        H += jcpl * spinop

                #print("H", np.linalg.matrix_rank(H), H.shape)
                # print(self.chshAsgn)
                # print(self.jcplAsgn)
                omega, intn = QDsimsGrpd2(H, self.TM)
                #print(H.shape)
                #return omega, intn

                # 9. Add the transitions to the arrays of their closest resonances
                freqQPeaks, intnQPeaks = [None]*len(chshQD), [None]*len(chshQD)     # Lists to hold arrays of frequencies and intensities for each spin separately
                indMin = np.argmin(abs(omega.reshape(-1,1) - chshQD.reshape(1,-1)), axis=1)    # Indices of the closest chem shift in freqArr for each transition
                for i in range(len(chshQD)):
                    indx = np.where(indMin == i)
                    if len(indx) == 0:
                        print('No peaks in this group.')
                    freqQPeaks[i] = omega[indx]
                    intnQPeaks[i] = intn[indx]
            else:
                freqQPeaks, intnQPeaks = [[chshQD[0]]], [np.array([n_spin])]

            # Aggregate poles and assign them to different chemical shifts and update the corresponding child node
            for i, chld in enumerate(self.children()):
                qPoles, qPolesIntn = group_peaks(freqQPeaks[i], intnQPeaks[i], maxWidth = config.QD_AggregatePeaksThreshold)                    # relative values of peak positions in ppm

                chld.qPoles = 1j*2*np.pi*np.array(qPoles)
                chld.qPolesIntn = np.array(qPolesIntn) / chld.intn    # Scale all qPoles for a given T node by the number of nuclei with the same chemical shift (i.e. the intensity of the node)

                # TODO: Simplify peaks / aggregate several peaks

                # Include the effect of line broadening
                chld.sT, chld.qT, chld.uT, chld.uF = [], [], [], []        # Remove previous sT
                chld.sPole = 1j*0 - alphQD[i]
                chld.propPoles(self.uPoles)      # Propagate the poles

            # store the parameters
            self.oldParsQD.update({"chsh":chshQD, "jcpl":jcplQD})
        else:
            # Check maybe only some alphas and/or chem shifts have changed
            diffPoles = 1j*2*np.pi*(chshQD - self.oldParsQD["chsh"]) - alphQD
            for i, chld in enumerate(self.children()):
                if chld.sPole != diffPoles[i]:
                    chld.sT, chld.uT, chld.uF = [], [], []
                    chld.sPole = diffPoles[i]
                    chld.uPoles = chld.qPoles + chld.sPole

    #@profile
    def evalTime(self, t, c0, chsh=[], alph=[], chshQD=[], alphQD=[], jcplQD=[], **kwargs):
        "Computes the node's response sT and also updates the children if any QD parameters have changed."
        self.getPoles(c0, chsh, alph, chshQD, alphQD, jcplQD)
        if self.sT == [] or not np.array_equal(self.oldTime, t):
            self.sT = self.intn if self.sPole == 0 else self.intn * np.exp(np.outer(t, self.sPole)).ravel()

        # Evaluate children as well
        for i, chld in enumerate(self.children()):
            if chld.qT == [] or not np.array_equal(self.oldTime, t):
                chld.qT = np.inner( np.exp(np.outer(t, chld.qPoles)), chld.qPolesIntn ).ravel()
            if chld.sT == [] or not np.array_equal(self.oldTime, t):
                chld.sT = chld.intn * chld.qT * np.exp(np.outer(t, chld.sPole)).ravel()

        self.oldTime = t

class chemNodeT(chemNode):
    "Terminal nodes that emit signals. Can only be included as leaves."
    def __init__(self, name, chsh = None, alph = None, ampl = None, phase = None, intn = 1., alias=''):
        chemNode.__init__(self, name, chsh, alph, ampl, phase, intn, alias)
        self.qPoles = np.array([0.])                # QD poles from the parent node that determine peak splitting
        self.qPolesIntn = 1.
        self.uPoles = 0.                # poles computed including the effects of all ancestors
        self.uF = []

    def default_pars(self):
        """Returns a dictionary of default parameters for the node."""
        return {'ampl':[self.ampl[0].dflt()], 'phase':[self.phase[0].dflt()]}

    #@profile
    def evalTime(self, c0, chsh=[], alph=[], **kwargs):
        "Computes the node's response sT"
        pass

    def getPoles(self, c0, chsh=[], alph=[], **kwargs):
        "Computes the poles"
        pass

    def addChild(self, child):
        """Terminal nodes can not have children."""
        raise RuntimeError("Children can not be added to terminal nodes.")

    def insertChild(self, child, pos):
        """Terminal nodes can not have children."""
        raise RuntimeError("Children can not be added to terminal nodes.")

    def propPoles(self, uPolePrnt = None):
        """Propagates offset poles to all children."""
        if uPolePrnt is None:
            uPolePrnt = 0. if self.isRoot() else self._parent.uPoles     # Set the offset pole to the uPole of the parent
        newPoles = self.sPole + uPolePrnt + self.qPoles
        if not np.array_equal(self.uPoles, newPoles):
            self.uPoles = newPoles
            self.uF = []    # Reset the output
            return 1
        else: return 0

    # @njit
    # @profile
    def evalFreq(self, f, dt, c0, f0=0, tau=0):
        "Computes the node's response in the frequency domain assuming that all ancestors have updated uPoles."
        # Check if the signal needs to be reevaluated
        if self.uF == [] or self.uF.size != f.size:
            # print(self.name)
            # print(self.uPoles.shape)
            self.uF = np.exp(1j*tau*(self.uPoles.imag - 2*np.pi*f0)).reshape((1,-1))
            x1 = 1j*2*np.pi*(c0*f-f0).reshape((-1,1))
            x2 = np.conj(self.uPoles - 1j*2*np.pi*f0).reshape((1,-1))
            # self.uF = self.uF / -np.expm1((x1+x2)*dt)
            self.uF = ne.evaluate( 'x / -expm1( (x1 + x2)*dt )', local_dict={'x':self.uF, 'x1':x1, 'x2':x2, 'dt':dt})       # Compute exp(x)-1 in one go
            self.uF = ne.evaluate('sum(conj( x ) * y, axis=1)', local_dict={'x':self.uF, 'y':self.qPolesIntn}).ravel()
            # self.uF = np.inner(np.conj(self.uF), self.qPolesIntn).ravel()
            self.uF *= self.intn * np.sqrt((f[1]-f[0])*c0*dt)
            # print(self.uF.shape)

class chemNodeDB(chemNode):
    """Class for a node describing a chemical from the database, inherited from chemNode. The node can be specified either by passing a name of a species in the database or the QDpars structure (an instance of chemSpec class.)"""
    def __init__(self, name, chsh = None, alph = None, ampl = None, phase = None, intn = 1., alias='', nameDB=None, QDpars=None):
        chemNode.__init__(self, name, chsh, alph, ampl, phase, intn, alias)
        if name in chemDB or nameDB in chemDB:
            self.QDpars = copy.deepcopy(chemDB[self.name if nameDB is None else nameDB])    # Parameters from the database
        elif QDpars is not None:
            self.QDpars = QDpars
        else:
            raise RuntimeError("The chemical \'" + self.name + '\' is not in the database and no QD parameters are supplied.')
        self.HCmode = None            # Mode of experiment if the node is dendrolized

    def setReported(self, flag=True):
        """Self the _reported flag of the node. If the DB node itself is not reported, its terminal leaves, not spin systems become reported."""
        if not self.isLeaf():     # Leafs can only have _reported set to True
            if flag:
                for chld in self._children:
                    chld.setReported(flag)
            elif not self.isRoot() and self._parent.isReported() != flag:
                self._parent.setReported(flag)
            self._reported = flag
            for chld in self._children:
                chld.setReported(flag)

    def setDefaultQD(self, key, dval, min=None, max=None):
        """Sets (updates) the default distributions of QD parameters. key is a 2-tuple of the form ('chshH', i), ('jcplHH', i), or ('chshC', i), where i is the number of the parameter in the zero-order, e.g. ('chshH', 2) for the third chemical shift."""

        # Check if the entire list of parameters need to be updated (e.g. all chshH or all jcplHH, etc.)
        if not isinstance(key, tuple):
            if len(getattr(self.QDpars, key)) == len(dval):
                for i, val in enumerate(dval):
                    # Call the function recursively
                    self.setDefaultQD((key, i), val)
            else:
                raise RuntimeError("The number of supplied values does not match the size of the parameter array.")

        else:
            # Set up the range for the parameter
            if min is None or max is None:
                if 'chsh' in key[0]:
                    min, max = np.round(dval, decimals=1) + np.array([-0.05, 0.05])
                elif 'jcpl' in key[0]:
                    min, max = np.round(dval) + np.array([-1, 1])

            # Update the specification
            parsArray = getattr(self.QDpars, key[0])          # An entire array of the parameters, one of which needs to be updated
            parsArray[key[1]] = parsArray[key[1]]._replace(dval=dval, min=min, max=max)

    def dendrolize(self, experiment="1H"):
        "Creates chemTrees based on the QD parameters of the node"
        # self.QDpars = chemDB[self.nameDB]    # Update the parameters from the database
        self.HCmode = experiment
        # 1. Define big spin systems based on the type of experiment
        #if experiment == "1H":
        #    spsyBig = spsySpec(self.QDpars.chshH, self.QDpars.jcplHH, self.QDpars.chshAsgnH, self.QDpars.jcplAsgnHH)
        #elif experiment == "13C":
        #    spsyBig = spsySpec(self.QDpars.chshC, None, self.QDpars.chshAsgnC, None)

        # 2. Separate spin systems
        spsySmall = self.QDpars.getSpSy(mode=experiment)

        # 3. Add nodes to the tree. Each new node is a QD node for a particular spin system.
        for chld in self.children():             # Loop backwards to avoid missing children when the index increases but the number of children decreases
            self.removeChild(chld)
        for i, spsy in enumerate(spsySmall):
            # Add a QD node with their own terminal nodes
            SPSY = chemNodeQD(self.name + '-SPSY' + str(i+1), spsy)  # New spin system node (QD)
            for j in range(len(spsy.chsh)):
                SPSY.addChild(chemNodeT(self.name + '-' + str(i+1) + '.' + str(j+1), intn=spsy.chshAsgn.count(j+1),
                                        alias=self.name + ' ' + spsy.chsh[j].label if spsy.chsh[j].label!='' else ''))      # , intn=spsy.mult
            self.addChild(SPSY)

# ------------------------- Functions for working with trees -------------------------------

def defaultTreePars(tree, tau=0.0, theta=0.0, sigma2=0.0, lshapeOrder=2, gamma=0.0):
    """Returns a nested array of default tree parameters."""
    pars = {node.name : node.default_pars() for node in tree.items()}
    pars["."] = {"tau" : [tau], "theta" : [theta],  # "ampl" : [1.0]*len([i for i in tree.repRoots()]),
                 "mult" : [1.0], "sigma2" : [sigma2], 'gamma':[gamma],
                 "lshapeR" : [0.0]*lshapeOrder, "lshapeI" : [0.0]*lshapeOrder}
    return pars

#@profile
def evalTreeT(tree, t, c0, pars=None):
    """Evaluate the entire tree of chemNodes. Returns the time-domain response for the specified (reported) nodes in the tree. tree is a chemNode object -- any node in the tree; pars - a nested dictionary of parameters, where the first level is indexed by the names of the nodes, and the second level conatins the names of parameters"""
    # 1. Evaluate all nodes (computes self-responses s(t))
    for node in tree.items():
        node.evalTime(t, c0, **pars[node.name])
        #print(node.sT)

    # 2. Determine the root reported nodes (determine the reported subtrees)
    repRoots = [v for v in tree.repRoots()]

    # 3. Collect the childrens' responses, starting from the bottom, and multiply them with your own
    for rep in repRoots:
        for node in rep.iterDepth(method="post-order"):     # All nodes BELOW the reported nodes
            if node.childCount() > 0:
                node.uT = np.zeros((len(t), ), float) + 1j*np.zeros((len(t), ), float)
                for chld in node._children:
                    node.uT += chld.uT
                node.uT *= node.sT
            else:
                node.uT = node.sT

    # 4. Collect the parents' responses
    for node in tree.findRoot().iterDepth(method="in-order"):     # All nodes ABOVE the reported nodes
        if node not in repRoots:
            if node.isRoot():
                node.uT = node.sT
            else:
                node.uT = node.sT* node.parent().uT

    # 5. Put all responses together
    Z = np.ones((len(t), len(repRoots)), float) + 1j*np.zeros((len(t), len(repRoots)), float)
    for i, rep in enumerate(repRoots):
        Z[:,i] = 1/rep.intn * rep.uT * rep.parent().uT if rep.parent() is not None else 1/rep.intn * rep.uT

    return Z, [i.name for i in repRoots]

# @profile
def evalTreeF(tree, f, dt, c0, f0=0, pars=None):
    """Evaluates the entire tree of chemNodes and returns a model spectrum directly in the frequency domain. Tree is a chemNode object -- any node in the tree; pars - a nested dictionary of parameters, where the first level is indexed by the names of the nodes, and the second level conatins the names of parameters"""
    tau = 0     #    or use
    #tau = -pars['.']['tau'][0]

    # 1. Update all poles of each node in the tree and propagate them to find uPoles of the leaves
    for node in tree.items():
        node.getPoles(c0, **pars[node.name])
    tree.findRoot().propPoles()     # Propagate all poles

    # 2. Determine the root reported nodes (determine the reported subtrees)
    repRoots = [v for v in tree.repRoots()]

    # 3. Collect the childrens' responses, starting from the bottom
    for rep in repRoots:
        rep.evalFreq(f, dt, c0, f0, tau)

    # 4. Put all responses together
    Z = np.ones((len(f), len(repRoots)), float) + 1j*np.zeros((len(f), len(repRoots)), float)
    for i, rep in enumerate(repRoots):
        Z[:,i] = 1/rep.intn * rep.uF

    return Z, [i.name for i in repRoots]

def collectPeaks(tree, c0, pars=None):
    """Picks model peaks from all leaf nodes. returns a dicitionary with peaks groupped according to reported nodes."""
    # 1. Update all poles of each node in the tree and propagate them to find uPoles of the leaves
    if pars is not None:
        for node in tree.items():
            node.getPoles(c0, **pars[node.name])
    tree.findRoot().propPoles()     # Propagate all poles

    # 2. Determine the root reported nodes (determine the reported subtrees)
    repRoots = [v for v in tree.repRoots()]

    # 3. Collect the poles
    allPeaks = {}
    for rep in repRoots:
        allPeaks[rep.name] = {leaf.name : [peakSpec(chsh=pole.imag/(c0*np.pi*2), intn=leaf.qPolesIntn[i]*leaf.intn, fwhm=-pole.real/np.pi) for i, pole in enumerate(leaf.uPoles)] \
                              for leaf in rep.leaves() if leaf.uPoles.size > 0}
        #allPeaks[rep.name] = [peakSpec(chsh=pole.imag/(c0*np.pi*2), intn=leaf.qPolesIntn[i]*leaf.intn, fwhm=-pole.real/np.pi) for leaf in rep.leaves() if leaf.uPoles.size > 0 for i, pole in enumerate(leaf.uPoles)]

    return allPeaks

def peakName2parKey(name):
    """Return a parameter key for the chemical shift that affects position of the peak given by its name and indx (in the multiplet)."""
    name = name.rsplit('-', 1)
    indx = name[1].split('.')
    return (name[0]+'-SPSY'+indx[0], 'chshQD', int(indx[1])-1), (name[0]+'-SPSY'+indx[0], 'alphQD', int(indx[1])-1)

#@profile
def getFID(T, t, c0, f0=0, pars=None, tau=None):
    """Returns modeled signals in the time domain in the form of FID."""
    if pars is None:
        pars = defaultTreePars(T)
    if tau is None:
        tau = pars["."]["tau"][0]
    Z, repRootNames = evalTreeT(T, np.array(t)+tau, c0, pars=pars)
    # Shift the signal by f0
    Z = Z * np.exp(-1j*2*np.pi*f0*(np.array(t)+tau)).reshape(-1,1)
    # Apply lineshape correction
    try:
        ksi = np.array(pars['.']['lshapeR']) + 1j * np.array(pars['.']['lshapeI'])
        Z = Z * np.exp(np.inner(np.array(t).reshape(-1,1) ** np.arange(2, 2+ksi.size, 1).reshape(1,-1), ksi)).reshape(-1,1)
    except KeyError: pass
    #Z [0,:] /= 2
    return Z, repRootNames

# Functions for saving and loading predefined trees and their default parameters
def saveTree(fname, tree, pars=None):
    """Saves a chemTree datastructure along with its defauld parameters within a pickled format."""
    if pars is None:
        pars = defaultTreePars(tree)
    if fname[-4:] != '.ctr':
        fname += '.ctr'
    with open(fname, 'wb') as fp:
        pickle.dump({"tree":tree, "pars":pars}, fp)

def loadTree(fname):
    """Loads a tree and its parameters from the file fname."""
    with open(fname+'.ctr', 'rb') as fp:
        data = pickle.load(fp)
        data["tree"].setTreeBook()
    return data["tree"], data["pars"]
