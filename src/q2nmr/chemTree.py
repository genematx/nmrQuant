from collections import namedtuple
import numpy as np
import scipy.sparse as sps
import json
import os
import dill
import weakref
import scipy
import math
import itertools
import numexpr as ne
import copy
from operator import itemgetter
from molparser.convertmol import parse_sdf_file
import xlsxwriter

import q2nmr.config as config
from .utils.linear_sum_assignment import linear_sum_assignment

import warnings
warnings.filterwarnings("ignore")

# Named tuple to store pairs of min and max values
minmaxTuple = namedtuple('minmaxTuple', 'min, max')
minmaxTuple.__new__.__defaults__ = (-np.inf, np.inf)
minmaxTuple.imin = lambda self, f : np.searchsorted(f.ravel(), self.min)
minmaxTuple.imax = lambda self, f : np.searchsorted(f.ravel(), self.max)

# Specifications of prior distributions of parameters
parsSpec = namedtuple('parsSpec', 'min, max, label, distr, p1, p2, dval')
parsSpec.__new__.__defaults__ = (-np.inf, np.inf, '', 'Uniform', None, None, None)     # 'mode' specifies the location of the distribution maximum value
parsSpec.evalPrior = lambda self, arg : priorProb(self, arg)
parsSpec.dflt = lambda self : (self.min + self.max) / 2 if self.dval is None else self.dval

# Specifications of sets of samples
smplSpec = namedtuple('smplSpec', 'min, max, mean, median, var, q1, q3, p5, p95, hpd5')     # Specification of MCMC samples
smplSpec.__new__.__defaults__ = (-np.inf, np.inf, None, None, None, None, None, None, None, None)

# Specifications of individual peaks
peakSpec = namedtuple('peakSpec', 'freq, intn, fwhm')
peakSpec.__new__.__defaults__ = (0, 1, None)     #

# Specifications of spin systems
spsySpec = namedtuple('spsySpec', 'chsh, jcpl, chshAsgn, jcplAsgn, mult')
spsySpec.__new__.__defaults__ = (None, None, None, None, None, 1)

# Specifications of frequency blocks used for optimization
# NOTE: unused, left for compatibility. Use the workspace.freqSpec class instead
freqSpec = namedtuple('freqSpec', 'min, max, indxFreq, bslnOrder, bF')
freqSpec.__new__.__defaults__ = (-float('inf'), float('inf'), np.array([]), (None, None), None)
freqSpec.__str__ = lambda self : '{:.2f} ... {:.2f}'.format(self.min, self.max) if not (self.min == -float('inf') and self.max == float('inf')) else 'Entire range'

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

# Graph of magnetically inequivalent spins
spinVert = namedtuple('spinVert', 'indxChsh, nspin')             # indxChsh - position in the chshQD array corresponding to the given vertex; nspin - number of magnetically equivalent spins with the same chemical shift
spinEdge = namedtuple('spinEdge', 'indxJcpl, indxVert')          # indxJcpl - position in the jcplH array corresponding to the given link; indxVert is a tuple (or a list) of two MEq spin indices (indices of vertices)
spsyComb = namedtuple('spsyComb', 'name, intn, indxSpsy')        # indxSpsy - list of indices of spin systems assigned to the group

def asgn2meqv(chshAsgn, jcplAsgn=None):
    """Converts the chshAsgn format to the array of magnetically inequivalent spins."""

    if jcplAsgn is None:
        spins = [spinVert(indxChsh = key, nspin = chshAsgn.count(key)) for key in sorted(set(chshAsgn))]
        links = []

    else:
        # Find magnetically equivalent groups (same chemical shifts and same coupling patterns)
        jcplAsgn_full = np.array(jcplAsgn).T + jcplAsgn         # Complete double-sided J-coupling assignment matrix
        allAsgn = np.vstack([chshAsgn, jcplAsgn_full])
        allAsgn = [tuple(x.ravel()) for x in np.hsplit(allAsgn, allAsgn.shape[1])]       # List of tuples, s.t. each tuple contains the number of chsh and assignemts of j-couplings

        uniqAsgn = sorted(set(allAsgn))

        spins = [spinVert(indxChsh = key[0]-1, nspin = allAsgn.count(key)) for key in uniqAsgn]

        indx_meq = [allAsgn.index(key) for key in uniqAsgn]              # List of representative indices of equivalence classes of spins

        links = [spinEdge(jcplAsgn[i][j]-1, sorted([ii, jj])) for jj, j in enumerate(indx_meq) for ii, i in enumerate(indx_meq) if jcplAsgn[i][j] != 0]

    return spins, links

def meqv2asgn(spins, links=[]):
    """Converts the spin system representation as a list of magnetically non-equivalent spins to the chshAsgn representation."""
    chshAsgn = [v.indxChsh+1 for v in spins for _ in range(v.nspin)]
    n_spin = len(chshAsgn)

    jcplAsgn = [[0]*n_spin for _ in range(n_spin)]
    indxAsgn = [i for i, v in enumerate(spins) for _ in range(v.nspin)]      # Indices of MEq group, to which each spin is assigned (zero-indexed)

    # Set the values in the jcplAsgn matrix
    for i in range(n_spin):
        for j in range(i+1, n_spin):
            indxVert = set([indxAsgn[i], indxAsgn[j]])
            if len(indxVert) > 1:       # Consider only the case when indxAsgn[i] != indxAsgn[j]
                for v in links:
                    if indxVert == set(v.indxVert):
                        jcplAsgn[i][j] = v.indxJcpl+1

    return chshAsgn, jcplAsgn

# Sampling
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
                    hpd5 = (mean-1.959963984540*stdv, mean+1.959963984540*stdv),
                    var = var)

def smplSpec_invGamma(a, b):
    """Returns the statistics in the form of smplSpec derived from the parameters of the inverse-Gamma distribution."""
    return smplSpec(min = 0,
                    mean = b/(a-1) if a > 1 else None,
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

    Arguments:
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

class spopDict(dict):
    """A dictionary of precomputed spin operators."""
    def __init__(self):
        super().__init__()

    @staticmethod
    def hashAsgn(chshAsgn, jcplAsgn):
        """A function to compute teh hash value for chsh and jcpl Asgn arrays."""
        arr = tuple(np.concatenate([np.array(chshAsgn).ravel(), np.array(jcplAsgn).ravel()]))
        return arr

    @staticmethod
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

    def __getitem__(self, key):
        chshAsgn, jcplAsgn = key
        key = self.hashAsgn(chshAsgn, jcplAsgn)     # Convert the key to a hashable type

        try:
            return super().__getitem__(key)

        except KeyError:
            # 0. Compute the spin operators if they are not supplied
            n_spin = len(chshAsgn)        # Number of spins in the system
            spinopsL, spinopsJ = [0]*np.max(chshAsgn), [0]*np.max(jcplAsgn)

            Lx, Ly, Lz, TM = self.spinop(n_spin)       # Cartesian spin operators used to construct the Hamiltonian and the Transition matrix
            for i in range(n_spin):
                spinopsL[chshAsgn[i]-1] += Lz[i]
                if jcplAsgn is not None:
                    for j in range(n_spin):
                        if jcplAsgn[i][j] != 0:
                            spinopsJ[jcplAsgn[i][j]-1] += (Lx[i].dot(Lx[j]) + Ly[i].dot(Ly[j]) + Lz[i].dot(Lz[j])).real

            self[key] = (spinopsL, spinopsJ, TM)

            return super().__getitem__(key)

global spinops
spinops = spopDict()

class spinGroup():
    """Represents a group of spins: either an entire spin system, or a part of it."""

    def __init__(self, meqSpins, meqLinks, mult=1):
        self.meqSpins = meqSpins
        self.meqLinks = meqLinks
        self.mult = mult           # Multiplicity of the spin system (i.e. if two or more _identical_ spin systems are present in the molecule); used to scale peak intensities

        self._oldParsQD = {"freq":None, "jcpl":None}
        self._oldResult = {'freq':None, 'intn':None}

    def n_chsh(self):
        """Number of different chemical shift parameters (the length of the chshQD array that should be applied to the system)."""
        return len(set([spin.indxChsh for spin in self.meqSpins]))

    def n_spin(self):
        """Number of spins assigned to each chemical shift."""
        return [sum([spin.nspin for spin in self.meqSpins if spin.indxChsh == i]) for i in range(self.n_chsh())]

    def n_jcpl(self):
        """Number of different J-coupling parameters (the length of the jcplQD array that should be upplied to the system)."""
        return len(set([spin.indxJcpl for spin in self.meqLinks]))

    def get_transitions(self, freqQD, jcplQD, indx=None):
        """Computes transition peaks given arrays of chemical shifts and J-coupling values. The calculations can be restricted to certain spins with indices indx."""

        if indx is not None:
            # Restrict the computation
            pass
        else:
            # Run the QD simulations only if the parameters have changed (assume that chsh, alph, and t have also changed)
            mind_freqQD = np.concatenate([[abs(freq2 - freq1) for freq2 in freqQD[i+1:]] for i, freq1 in enumerate(freqQD)] + [[np.inf]]).min()     # Minimum distance between any two chemical shifts in this spin system; inf if there is only one chemical shift
            freq_diff = freqQD - self._oldParsQD["freq"] if self._oldParsQD["freq"] is not None else np.array([-np.inf, np.inf])           # Difference with previous array of inputs
            if self._oldParsQD["freq"] is None or self._oldParsQD["jcpl"] is None or any(self._oldParsQD["jcpl"] != jcplQD) \
                                              or ( max( abs(freq_diff)) > min(config.QD_RerunQDchshThreshold, 0.5*mind_freqQD ) ):

                # Check maybe the difference in all chemical shifts is the same, then all frequency peaks can be just shifted without running the simulations again
                if len(freqQD)>1 and np.isclose(max(freq_diff), min(freq_diff)) and all(np.isclose(self._oldParsQD["jcpl"], jcplQD)):
                    freqQPeaks, intnQPeaks = self._oldResult['freq'], self._oldResult['intn']
                else:
                    freqQPeaks, intnQPeaks = compute_transitions(freqQD, jcplQD, self.meqSpins, self.meqLinks)
                    freqQPeaks = [f_arr - f0 for f_arr, f0 in zip(freqQPeaks, freqQD)]       # Subtract the cetral frequency from each freqQPeak for centering
                    try: intnQPeaks = [x_arr * self.mult for x_arr in intnQPeaks]
                    except AttributeError: pass

                self._oldParsQD['freq'] = freqQD
                self._oldParsQD['jcpl'] = jcplQD
                self._oldResult['freq'] = freqQPeaks
                self._oldResult['intn'] = intnQPeaks
                flag_updated = True

            else:
                freqQPeaks, intnQPeaks = self._oldResult['freq'], self._oldResult['intn']
                flag_updated = False

        return freqQPeaks, intnQPeaks, flag_updated

    def reset(self):
        """Resetd the stored QD parameters used to run the previous simulation."""
        self._oldParsQD.update({'freq':None, 'intn':None})
        self._oldResult.update({'freq':None, 'intn':None})

def get_hamiltonian(chshQD, jcplQD, chshAsgn, jcplAsgn):
    """Computes teh Hamiltonian and the transition matrix."""
    global spinops

    # 0. Compute the spin operators
    spinopsL, spinopsJ, TM = spinops[(chshAsgn, jcplAsgn)]

    # 1. Build the Hamiltonian
    n_spin = len(chshAsgn)        # Number of spins in the system
    H = np.zeros((2**n_spin, 2**n_spin), dtype='float64')

    for chsh, spinop in zip(chshQD, spinopsL):
        H = H - chsh * spinop

    for jcpl, spinop in zip(jcplQD, spinopsJ):
        H = H + jcpl * spinop

    return H, TM

def transition_indices(n_spin, k=0):
    """Returns indices of singlestate transitons for the kth spin of total n_spin spins in an n_x_n matrix of intensities or frequencies."""
    k = n_spin-k-1
    diag_indx = [(i, i+2**k) for i in range(2**n_spin-2**k)]    # Indices of the 2**k off diagonal
    rows, cols = zip(*[diag_indx[i] for j in range(0, 2**n_spin, 2**(k+1)) for i in range(j, j+2**k)])
    return rows, cols

def splitSpSy(big, mult=None):
    """Splits a large spin system in the form of spsySpec namedtuple into a list of smaller disjoint spin systems. The result is a zip object containing separate disjoint spin systems and their corresponding intensities."""
    if big.jcplAsgn is None or big.jcpl is []:
        # Only singlets
        spsyAll = [spsySpec([chsh], [], [1], None, mult=big.chshAsgn.count(i+1)) for i, chsh in enumerate(big.chsh)]     # All chemical shifts
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
            indxSpin = [i for i, val in enumerate(spsyAsgn) if val==s]      # Find indices of spins in the current spin system
            chshSpinNew = [chshSpinAll[i] for i in indxSpin]                # Lists of chch and jcpl assigned to the spins
            jcplSpinNew = [[jcplSpinAll[i][j] for j in indxSpin] for i in indxSpin]

            chshNew, jcplNew = [], []     # Empty lists of unique chsh and jcpl
            chshAsgnNew = [0] * len(indxSpin)
            jcplAsgnNew = [[0]*len(indxSpin) for _ in range(len(indxSpin))]

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

        for indx, intn in enumerate(intnAll):
            spsyAll[indx] = spsyAll[indx]._replace(mult=intn)

    if mult is not None:
        for i, m in enumerate(mult):
            spsyAll[i] = spsyAll[i]._replace(mult=spsyAll[i].mult * m)

    return spsyAll

def tobin(x,n):
    """Converts an integer x into its binary representation in form of a list with n bits."""
    return [(x>>k)&1 for k in range(n-1,-1,-1)]     # Use range(0, n) for MSB first

def QDsims(H, T, tol=0.0001):
    """Simulates a QD system based on the spin frequencies and j couplings in Hz. See, e.g., http://www.users.csbsju.edu/~frioux/nmr/Speclab4.htm"""
    n_spin = int(math.log2(T.shape[0]))

    # Compute the eigenvalues/eigenvectors of the Hamiltonian
    # vH, uH = np.linalg.eigh(np.asarray(H))      # Need to make sure that the Hamiltonian is passed as an array, not a matrix      # vH, uH = scipy.linalg.eigh(H)              # Possibly faster in some cases???
    vH, uH = scipy.linalg.eigh(np.asarray(H))

    # Find the intensities and transition frequencies
    intn = np.dot(uH.T, T.dot(uH))**2 / (2**(n_spin-1))     # Elementwise power!
    omega = abs(vH.reshape(-1,1) - vH)
    intn = np.triu(intn).flatten('F').real
    omega = np.triu(omega).flatten('F')

    # Drop small transition peaks
    if tol < n_spin:
        p = np.argsort(intn)                    # Sort the peaks from highest to lowest intensity
        p = p[-n_spin*(2**(n_spin-1)):]          # Keep only peaks corresponding to single transitions (assuming they are the largest)
        p = p[np.cumsum(intn[p]) > tol]
        omega, intn = omega[p], intn[p]

    # Renormalize the intensities
    intn = n_spin * intn / intn.sum()

    return omega, intn

def QTransFull(chshQD, jcplQD, meqSpins, meqLinks, tol=0.0001):
    """Use general QD simulations to compute the transition peak frequencies and intensities."""

    def split_arrays(omega, intn, chsh):
        """Splits arrays of peak frequencies and intensities according to the values of corresponding chemical shifts."""
        n_spin = len(chsh)

        #  TODO: Check if there are equal chemical shifts in the array
        if len(chsh) != len(set(chsh)):
            chsh += np.linspace(0, 0.1, len(chsh))

        # Sort the values of chemical shifts
        indx_chsh = np.argsort(chsh)
        chsh = chsh[indx_chsh]

        # Sort the transitions in the increasing order of their frequencies
        indx = np.argsort(omega)
        omega, intn = omega[indx], intn[indx]
        csintn = np.cumsum(intn)

        # Find the indices for splits (indicated by integer values of intensities in the ordered cumsum array)
        indx_split = [np.searchsorted(csintn, i) for i in range(1, n_spin)]       #     Faster than indx_split = np.searchsorted(csintn, [range(1, n_spin)])[0]
        indx_split[0] = max(indx_split[0], 1)    # If the first entry csintn[0]>1 then the first split would occur at the index 0 and create an empty array

        # Loop over all splits and move the boundary forward if it's closer to the left (lower) chemical shift, or backward, if the previous transition is closer to the right chemical shift. The boundaries are defined from the left (i.e. the boundary is the lowest frequency in the next group of peaks).
        for i, ind in enumerate(indx_split):
            if i > 0 and ind == indx_split[i-1]:
                ind += 1     # Prevent repeating splits (and resulting empty arrays)

            if not np.isclose(chsh[i], chsh[i+1]):
                try:
                    while True:
                        if omega[ind]-chsh[i] < chsh[i+1]-omega[ind]:
                            ind += 1
                        elif omega[ind-1]-chsh[i] > chsh[i+1]-omega[ind-1]:
                            ind -= 1
                        else: break
                except IndexError: break

            indx_split[i] = ind

        # Split the arrays and order them according the original oreder of chemical shifts.
        omega = np.split(omega, indx_split)
        intn = np.split(intn, indx_split)

        # Reorder the groups according the order of chemical shifts. Add another empty array at the end (this corresponds to unassigned combination transitions)
        sorter = itemgetter(*ind2pos(indx_chsh))        # The indices to sort the chemical shifts in increasing order
        omega = list(sorter(omega)) + [np.empty(0)]
        intn = list(sorter(intn)) + [np.empty(0)]

        return omega, intn

    chshAsgn, jcplAsgn = meqv2asgn(meqSpins, meqLinks)
    n_spin = len(chshAsgn)        # Number of spins in the system

    # Obtain the Hamiltonian
    H, TM = get_hamiltonian(chshQD, jcplQD, chshAsgn, jcplAsgn)

    # Diagonalize the Hamiltonian
    freq, intn = QDsims(H, TM)

    # Split the transitions according to their closest chemical shifts (return n_spin arrays)
    freq, intn = split_arrays(freq, intn, chsh=chshQD[np.array(chshAsgn)-1])

    # Combine the peaks into arrays corresponding to each chemical shift
    freqQPeaks, intnQPeaks = [None]*len(chshQD), [None]*len(chshQD)     # Lists to hold arrays of frequencies and intensities for each spin separately
    for i in range(len(chshQD)):
        indx_simple = np.where(np.array(chshAsgn)==i+1)[0]
        freqQPeaks[i] = np.concatenate( [freq[j] for j in indx_simple ] )
        intnQPeaks[i] = np.concatenate( [intn[j] for j in indx_simple ] )

    return freqQPeaks, intnQPeaks

def QDsimsGrpd(H, T, states, tol=0.0001):
    """
    Simulates a QD system based on the spin frequencies and j couplings in Hz. See, e.g., http://www.users.csbsju.edu/~frioux/nmr/Speclab4.htm
       Inputs:
           H - The Hamiltonian matrix of the spin system
           T - The matrix of transition probabilities
           states - labels of the combination states encoded by the columns of the Hamiltonian
           tol - tolerance for keeping the transitions. The sum of all kept transitions will differ from the rtue value by no more than tol
        Outputs:
           freq - a list of arrays, with each array containing the frequencies (in ppm) of resonances corresponding to the i-th spin. The last array in the list contains all combination lines (if any).
           intn - a list of arrays of the corresponding intensities
           lbls - a list of lists of corresponding transition labels
    """
    def label_transitions(states_from, states_to):
        """Finds which spins flip during the transition between the states."""
        n_spin = states_from.shape[1]
        TT = np.abs( states_from - states_to[:, None, :] )     # 3D array of the size n_to x n_from x n_spin
        indx_flip = TT.dot(np.arange(n_spin)+1) - 1
        indx_flip = np.where(indx_flip < n_spin, indx_flip, n_spin).T
        return indx_flip

    def arr2mat(arr, filler=0):
        """Converts (rearranges) a list of arrays of flips, omegas, or intns into a single matrix representation.
           Rows of the matrix correspond to transitions involving even states (M=0, 2, 4, ...);
           columns -- odd states (M=1, 3, 5, ...). Filler defines a number in the cells corresponding to transitions between distant states with |M1-M2| > 1."""
        n_spin = len(arr)
        mat = filler*np.ones((2**(n_spin-1), 2**(n_spin-1)), dtype=arr[0].dtype)
        i1, i2 = 0, 0
        for i in range(n_spin):
            if i % 2 == 0:
                n1, n2 = arr[i].shape
                mat[i1:i1+n1, i2:i2+n2] = arr[i]
                i1 += n1
            else:
                n1, n2 = arr[i].T.shape
                mat[i1:i1+n1, i2:i2+n2] = arr[i].T
                i2 += n2
        return mat

    def mat2arr(mat):
        n_spin = int(np.log2(mat.shape[0]))+1
        ss = [scipy.special.comb(n_spin, k, exact=True) for k in range(n_spin+1)]    # Find the sizes of subarrays

        arr = [None]*n_spin      # Initialize the result
        i1, i2 = 0, 0
        n1, n2 = 1, 1
        for k in range(n_spin):
            if k % 2 == 0:
                n2 = scipy.special.comb(n_spin, k+1, exact=True)
                arr[k] = mat[i1:i1+n1, i2:i2+n2]
                i1 += n1
            else:
                n1 = scipy.special.comb(n_spin, k+1, exact=True)
                arr[k] = mat[i1:i1+n1, i2:i2+n2].T
                i2 += n2
        return arr

    def state2pos(M):
        """
        For each state in the list of M values returns its position
        in the combined matrix (its index and whether
        it is a row or a column). M - array of M-values for each state
        """
        n_spin = int(np.log2(len(M)))
        pos = np.zeros_like(M)
        rc = ['r' if m%2 == 0 else 'c' for m in M]     # Row or column (even states - rows, odd states - columns)
        i1, i2 = 0, 0
        for k in range(n_spin+1):
            n = scipy.special.comb(n_spin, k, exact=True)         # Number of states on this level
            if k % 2 == 0:
                pos[M == k] = np.arange(i1, i1+n)
                i1 += n
            else:
                pos[M == k] = np.arange(i2, i2+n)
                i2 += n

        return pos, rc

#     print('Entering the QDSims function\n')
    n_spin = int(math.log2(T.shape[0]))

    # Compute the eigenvalues/eigenvectors of the Hamiltonian
    vH, uH = np.linalg.eigh(np.asarray(H))      # Need to make sure that the Hamiltonian is passed as an array, not a matrix

    # Find quantum numbers (levels, M) for each eigenvector
    M = states.sum(axis=1)        # Quantum number for each state (from 0 to n_spin+1)
    S = np.zeros( (2**n_spin, n_spin+1) )            #     S = sps.csr_matrix(([1]*(2**n_spin), (range(2**n_spin), M)))
    S[range(2**n_spin), M] = 1
    M_uH = np.argmax(S.T.dot(np.abs(uH)), axis=0)

    # Sort the eigenvalues/vectors within each quantum level in increasing order. Then put them into a new array according to the order of the chemical shifts.
    vHs, uHs = np.empty_like(vH), np.empty_like(uH)
    for m in range(n_spin+1):
        indx_put = np.where(M == m)[0]
        indx_take = np.where(M_uH == m)[0]
        indx_take = indx_take[np.argsort(vH[indx_take])]      # Sort the indices according to the values in vH

        # Sort the eigenvalues/eigenvectors
        vHs[indx_put] = vH[indx_take]
        uHs[:, indx_put] = uH[:, indx_take]

    if n_spin > 5:
        # Compute the transitions. Consider only single-order transitions (both simple and combination)
        s1, s2 = np.where(M.reshape(-1,1) - M.reshape(1,-1) == -1)   # Indices of interacting coherences

        # Complete arrays
        omega = vHs[s2] - vHs[s1]
        intn = np.dot(uHs.T, T.dot(uHs))**2 / (2**(n_spin-1))      # Elementwise power!
        intn = intn[s2, s1]                                        # intn = np.sum(T.dot(uHs)[:, s2] * uHs[:, s1], axis=0)**2 / (2**(n_spin-1))

        # Drop small transition peaks
        if tol < n_spin:
            p = np.argsort(intn)                    # Sort the peaks from highest to lowest intensity
            p = p[-n_spin*(2**(n_spin-1)):]          # Keep only peaks corresponding to single transitions (assuming they are the largest)
            p = p[np.cumsum(intn[p]) > tol]
            omega, intn = omega[p], intn[p]

        # Sort the transitions according to their frequencies
        indx = np.argsort(omega)
        omega, intn = omega[indx], intn[indx]
        csintn = np.cumsum(intn)
        indx_split = [np.where(csintn > i)[0][0]+1 for i in range(1, n_spin)]
        omega = np.split(omega, indx_split)
        intn = np.split(intn, indx_split)
        trans = [['?'*i + '*' + '?'*(n_spin-i-1)]*len(gr) for i, gr in enumerate(omega)]

        # Reorder the groups according the order of chemical shifts
        sorter = itemgetter(*np.argmax(states[M==1, :], axis=1).tolist())        # The indices to sort the chemical shifts in increasing order
        omega = list(sorter(omega)) + [np.empty(0)]
        intn = list(sorter(intn)) + [np.empty(0)]
        trans = list(sorter(trans)) + [[]]

    elif n_spin < 4:
        # In this case, the states are ordered correctly, so we can compute and label all transitions rightaway

        # Compute the transitions. Consider only single-order transitions (both simple and combination)
        s1, s2 = np.where(M.reshape(-1,1) - M.reshape(1,-1) == -1)   # Indices of interacting coherences
        trans = states[s1, :] - states[s2, :]       # Encoded transitions
        flipped_indx = np.where(np.sum(np.abs(trans), axis=1) == 1, np.argmax(np.abs(trans), axis=1), n_spin)     # Indices of spins that flipped in each transition (indx = n_spin for combination transitions)

        # Complete arrays
        omega = vHs[s2] - vHs[s1]
        intn = np.sum(T.dot(uHs)[:, s2] * uHs[:, s1], axis=0)**2 / (2**(n_spin-1))

        omega = [omega[flipped_indx == i] for i in range(n_spin + 1)]
        intn = [intn[flipped_indx == i] for i in range(n_spin + 1)]
        trans = [ [''.join([{0:'o', -1:'-', 1:'+'}[t] if t != 0 else {0:'a', 1:'b'}[s] for t, s in zip(ttt, sss)])
                   for ttt, sss in zip(trans[flipped_indx == i, :].tolist(),
                                       states[s1[flipped_indx == i], :].tolist() )]
                   for i in range(n_spin + 1)]

    elif n_spin == 4:
        # Some states may flip; need to label each allowed (-1) transistion by the number of spins that flip (use n_spin for combination transitions)

        # Compute the transitions between each pair of consecutive quantum levels
        trans_by_level = [None]*n_spin
        for t in range(n_spin):
            # Find the indices of involved states (s1 end state, s2 start state)
            s1, s2 = np.where(M == t)[0], np.where(M == t+1)[0]

            # Detrmine which spin flips for each transition (flip = n_spin for combination transitions)
            flip = np.abs( states[s2, :] - states[s1, None, :] ).dot(np.arange(n_spin)+1) - 1
            flip = np.where(flip < n_spin, flip, n_spin)

            # Compute the transition frequencies and intensities
            omega = vHs[None,s2] - vHs[s1,None]
            intn = ( (T.dot(uHs[:, s1])).T.dot(uHs[:, s2]) )**2

            trans_by_level[t] = [flip, omega, intn]

        mat_flip = arr2mat([t[0] for t in trans_by_level], filler=-np.inf)
        mat_intn = arr2mat([t[2] for t in trans_by_level])
        c = np.where(mat_flip < n_spin, 1, -1)
        CI = c*mat_intn
        cost_matrix = np.zeros((2,2))
        for i in range(2):
            for j in range(2):
                cost_matrix[i, j] = sum(c[3+i, :] * mat_intn[3+j, :])
        _, lbls = linear_sum_assignment(-cost_matrix)

        indx0 = (np.where(M==2)[0])[2:4]
        indx = indx0[lbls]
        vHs[indx0] = vHs[indx]
        uHs[:, indx0] = uHs[:, indx]

        # Compute the transitions between each pair of consecutive quantum levels
        trans_by_level = [None]*n_spin
        for t in range(n_spin):
            # Find the indices of involved states (s1 end state, s2 start state)
            s1, s2 = np.where(M == t)[0], np.where(M == t+1)[0]

            # Detrmine which spin flips for each transition (flip = n_spin for combination transitions)
            flip = np.abs( states[s2, :] - states[s1, None, :] ).dot(np.arange(n_spin)+1) - 1
            flip = np.where(flip < n_spin, flip, n_spin)

            # Compute the transition frequencies and intensities
            omega = vHs[None,s2] - vHs[s1,None]
            intn = ( (T.dot(uHs[:, s1])).T.dot(uHs[:, s2]) )**2

            trans_by_level[t] = [flip, omega, intn]

    #     # Compute the transitions. Consider only single-order transitions (both simple and combination)
        s1, s2 = np.where(M.reshape(-1,1) - M.reshape(1,-1) == -1)   # Indices of interacting coherences
        trans = states[s1, :] - states[s2, :]       # Encoded transitions
        flipped_indx = np.where(np.sum(np.abs(trans), axis=1) == 1, np.argmax(np.abs(trans), axis=1), n_spin)     # Indices of spins that flipped in each transition (indx = n_spin for combination transitions)

        # Split the arrays
        omega = np.concatenate([t[1].ravel() for t in trans_by_level])
        intn = np.concatenate([t[2].ravel() for t in trans_by_level])
        flip = np.concatenate([t[0].ravel() for t in trans_by_level])
        omega = [omega[flip == i] for i in range(n_spin + 1)]
        intn = [intn[flip == i] for i in range(n_spin + 1)]
        trans = [ [''.join([{0:'o', -1:'-', 1:'+'}[t] if t != 0 else {0:'a', 1:'b'}[s] for t, s in zip(ttt, sss)])
                   for ttt, sss in zip(trans[flipped_indx == i, :].tolist(),
                                       states[s1[flipped_indx == i], :].tolist() )]
                   for i in range(n_spin + 1)]
        states = [''.join([{0:'a', 1:'b'}[s] for s in sss]) for sss in states.tolist()]

    elif n_spin == 5:
        # Some states may flip; need to label each allowed (-1) transistion by the number
        # of spins that flip (use n_spin for combination transitions)

        # Compute the transitions between each pair of consecutive quantum levels
        trans_by_level = [None]*n_spin
        for t in range(n_spin):
            # Find the indices of involved states (s1 end state, s2 start state)
            s1, s2 = np.where(M == t)[0], np.where(M == t+1)[0]

            # Detrmine which spin flips for each transition (flip = n_spin for combination transitions)
            flip = np.abs( states[s2, :] - states[s1, None, :] ).dot(np.arange(n_spin)+1) - 1
            flip = np.where(flip < n_spin, flip, n_spin)

            # Compute the transition frequencies and intensities
            omega = vHs[None,s2] - vHs[s1,None]
            intn = ( (T.dot(uHs[:, s1])).T.dot(uHs[:, s2]) )**2

            trans_by_level[t] = [flip, omega, intn]

        mat_flip = arr2mat([t[0] for t in trans_by_level], filler=-np.inf)
        mat_omega = arr2mat([t[1] for t in trans_by_level])
        mat_intn = arr2mat([t[2] for t in trans_by_level])
        c = np.where(mat_flip < n_spin, 1, -1)
        CI = c*mat_intn

        # Look at all possible combinations of states and determine which ones give higher intensities of simple transitions
        combs = [[0,1,2,3,4,5], [0,1,2,4,3,5], [0,1,2,4,5,3], [0,1,4,2,3,5], [0,1,4,2,5,3], [1,4,0,2,3,5], \
                 [1,0,2,3,4,5], [1,0,2,4,3,5], [1,0,2,4,5,3], [1,0,4,2,3,5], [1,0,4,2,5,3], [1,4,0,2,5,3]]    # All possible permutations of state indices for M=2 or M=3
#         combs = [list(p) for p in itertools.permutations([0,1,2,3,4,5])]

        # Indices of the permutable states in the state array
        indxM2, indxM3 = [6, 9, 10, 12, 17, 18], [25, 22, 21, 19, 14, 13]

        # Indices of permuatable rows and columns in the aggregated transition matrices
        pos_in_mat, rc_in_mat = state2pos(M)
        rowsM2 = pos_in_mat[indxM2]
        colsM3 = pos_in_mat[indxM3]

        # Find the best rearrangement of rows and columns to maximize the intensities of simple transition lines
        best_cost = -np.inf
        for cr, cc in itertools.product(combs, repeat=2):
            # Swap rows and columns in a copy of the matrix
            mat = mat_intn.copy()
            mat[:, colsM3[cc]] = mat[:, colsM3]
            mat[rowsM2[cr], :] = mat[rowsM2, :]    # Not like this: #             mat[:, colsM3] = mat[:, colsM3[cc]] #             mat[rowsM2, :] = mat[rowsM2[cr], :]

            cost = (c * mat)[:11, 5:].sum()         # Or possibly cost = (c * mat).sum()
            if cost > best_cost:
                best_cr, best_cc, best_cost = cr, cc, cost

        mat_intn[:, colsM3[best_cc]] = mat_intn[:, colsM3]
        mat_intn[rowsM2[best_cr], :] = mat_intn[rowsM2, :]
        mat_omega[:, colsM3[best_cc]] = mat_omega[:, colsM3]
        mat_omega[rowsM2[best_cr], :] = mat_omega[rowsM2, :]

        # Reorder the eigenvectors and eigenvalues
        indx_old = np.concatenate([indxM2, indxM3])
        indx_new = np.concatenate([[indxM2[i] for i in best_cr], [indxM3[i] for i in best_cc]])
        vHs[indx_new] = vHs[indx_old]
        uHs[:, indx_new] = uHs[:, indx_old]

        # Compute the transitions. Consider only single-order transitions (both simple and combination)
        s1, s2 = np.where(M.reshape(-1,1) - M.reshape(1,-1) == -1)   # Indices of interacting coherences
        trans = states[s1, :] - states[s2, :]       # Encoded transitions
        flipped_indx = np.where(np.sum(np.abs(trans), axis=1) == 1, np.argmax(np.abs(trans), axis=1), n_spin)     # Indices of spins that flipped in each transition (indx = n_spin for combination transitions)

        # Complete arrays
        omega, intn = [], []
        for i in range(n_spin + 1):
            indx = np.where(mat_flip == i)
            omega.append(mat_omega[indx])
            intn.append(mat_intn[indx])
        trans = [ [''.join([{0:'o', -1:'-', 1:'+'}[t] if t != 0 else {0:'a', 1:'b'}[s] for t, s in zip(ttt, sss)])
                   for ttt, sss in zip(trans[flipped_indx == i, :].tolist(),
                                       states[s1[flipped_indx == i], :].tolist() )]
                   for i in range(n_spin + 1)]

#     print('Exiting the QDSims function\n')
    return omega, intn, trans, uHs, vHs

def QTransAB(chshQD, jcplQD, n_spin=(1,1)):
    """Simulate an approximate response for an AmBn system."""

    def expand_multiplet(freq, intn, order=2):
        """Expands doublet of peaks into multiplets while preserving the intensities ratio. order is the number of coupled spins, e.g. 2 to get a triplet."""
        if order == 1:
            # Don't do anything
            return freq, intn

        d = np.abs(freq[1]-freq[0])  # Step between the peaks
        m = np.mean(freq)   # Center of the multiplet (doesn't have to be the actual chemical shift)
        coef = np.array([scipy.special.binom(order, i) for i in range(order+1)])      # Binomoal coefficients

        freq_out = m + d * (np.arange(order+1)-order/2)
        intn_out = coef * (intn[0]**np.arange(order, -1, -1)) * (intn[1]**np.arange(order+1))

        return freq_out, intn_out

    if len(chshQD) != 2:
        raise RuntimeError('Only two-spin systems are supported.')

    # Find the transitions for an AB system first. See Keeler, page 2-15
    J = np.abs(jcplQD).item()          # Works only with abs(J)
    D = np.sqrt( (chshQD[1]-chshQD[0])**2 + J**2 ).item()
    S = -np.array(chshQD[0] + chshQD[1]).item()      # Positive in the original source
    sin2t = J/D

    freqQPeaks = [np.array([-D-S-J, -D-S+J])/2, np.array([D-S-J, D-S+J])/2]
    intnQPeaks = [np.array([1-sin2t, 1+sin2t])/2, np.array([1+sin2t, 1-sin2t])/2]

    # Flip the order of the results if the chemcial shifts are in the reverse order
    if chshQD[1] < chshQD[0]:
        freqQPeaks.reverse()
        intnQPeaks.reverse()

    # Treat multiple-spin cases
    freqQPeaks[0], intnQPeaks[0] = expand_multiplet(freqQPeaks[0], intnQPeaks[0], n_spin[1])
    freqQPeaks[1], intnQPeaks[1] = expand_multiplet(freqQPeaks[1], intnQPeaks[1], n_spin[0])

    # Scale intensities by the number of spins
    intnQPeaks[0] *= n_spin[0]
    intnQPeaks[1] *= n_spin[1]
    return freqQPeaks, intnQPeaks

def QTrans3X(chshQD, jcplQD, chshAsgn, jcplAsgn):
    """Simulates a response for a spin system with 3 independent coupled spins."""

    n_spin = len(chshAsgn)        # Number of spins in the system

    if n_spin != 3:
        raise RuntimeError('Only three-spin systems are supported.')

    # Obtain the Hamiltonian
    H, TM = get_hamiltonian(chshQD, jcplQD, chshAsgn, jcplAsgn)

    # 2. Compute the matrix of states.
    # Determine the quantum number for each state (0-alpha, 1-beta)
    states = np.array([state for state in itertools.product([0, 1], repeat=n_spin)])
    # Sort the states according to the order of ALL chemical shifts (with repeats)
    indx = ind2pos(np.argsort( chshQD[np.array(chshAsgn)-1] )[::-1])           # chshAll = chshQD[np.array(self.chshAsgn)-1]
    states = states[states[:, indx].dot(2**np.arange(0, n_spin)[::-1]), :]

    # 3. Diagonalize the Hamiltonian
    omega, intn, trans, _, _ = QDsimsGrpd(H, TM, states)   # assign_by_dist=(len(np.unique(chshAsgn))<len(chshAsgn))

    # Combine the peaks into arrays corresponding to each chemical shift. Add the combination transitions to the arrays of their closest resonances
    freqQPeaks, intnQPeaks = [None]*len(chshQD), [None]*len(chshQD)     # Lists to hold arrays of frequencies and intensities for each spin separately
    indMin = np.argmin(abs(omega[-1].reshape(-1,1) - chshQD.reshape(1,-1)), axis=1)    # Indices of the closest chem shift in freqArr for each transition
    for i in range(len(chshQD)):
        indx_combin = np.where(indMin == i)[0]
        indx_simple = np.where(np.array(chshAsgn)==i+1)[0]
        freqQPeaks[i] = np.concatenate( [omega[j] for j in indx_simple ] + [omega[-1][indx_combin]] )
        intnQPeaks[i] = np.concatenate( [ intn[j] for j in indx_simple ] + [ intn[-1][indx_combin]] )

    return freqQPeaks, intnQPeaks

def QTransPairs(chshQD, jcplQD, meqSpins, meqLinks):
    """Simulates large spin system by splitting them in pairs of coupled spins"""

    # Find lists of lists of np arrays of freq/intn for each meq spin.
    # The second level of lists corresponds to the meq spin being involved in different subgraphs within the spin system. After all subgraphs are computed, their peaks will be convolved with each other.
    freq_meq, intn_meq = [[] for _ in range(len(meqSpins))], [[] for _ in range(len(meqSpins))]

    # Simulate each pair as an AmBn system
    for edge in meqLinks:
        p, q = edge.indxVert           # Indices fo coupled spins
        freq, intn = QTransAB( (chshQD[meqSpins[p].indxChsh], chshQD[meqSpins[q].indxChsh]),
                                jcplQD[edge.indxJcpl], n_spin=(meqSpins[p].nspin, meqSpins[q].nspin) )
        freq_meq[p].append(freq[0])
        freq_meq[q].append(freq[1])
        intn_meq[p].append(intn[0])
        intn_meq[q].append(intn[1])

    # Convolve the multiplets for each equivalent spin and save them in the subarray corresponding to a specific chshQD
    freqQPeaks, intnQPeaks = [[] for _ in chshQD], [[] for _ in chshQD]
    for i, vert in enumerate(meqSpins):
        freq_meq[i], intn_meq[i] = convolve_multiplets(freq_meq[i], intn_meq[i], chshQD[vert.indxChsh], vert.nspin)
        freqQPeaks[vert.indxChsh].extend(freq_meq[i])
        intnQPeaks[vert.indxChsh].extend(intn_meq[i])

    # Convert to np arrays
    freqQPeaks, intnQPeaks = [np.array(x) for x in freqQPeaks], [np.array(x) for x in intnQPeaks]

    return freqQPeaks, intnQPeaks

def QTransClusters(chshQD, jcplQD, meqSpins, meqLinks):
    """Simulates large spin system by splitting them in overlapping clusters of coupled spins"""

    # Find lists of lists of np arrays of freq/intn for each meq spin.
    # The second level of lists corresponds to the meq spin being involved in different subgraphs within the spin system. After all subgraphs are computed, their peaks will be convolved with each other.
    freq_meq, intn_meq = [[] for _ in range(len(meqSpins))], [[] for _ in range(len(meqSpins))]

    # Convolve the multiplets for each equivalent spin and save them in the subarray corresponding to a specific chshQD
    freqQPeaks, intnQPeaks = [[] for _ in chshQD], [[] for _ in chshQD]
    for i, vert in enumerate(meqSpins):
        freq_meq[i], intn_meq[i] = convolve_multiplets(freq_meq[i], intn_meq[i], chshQD[vert.indxChsh], vert.nspin)
        freqQPeaks[vert.indxChsh].extend(freq_meq[i])
        intnQPeaks[vert.indxChsh].extend(intn_meq[i])

    # Convert to np arrays
    freqQPeaks, intnQPeaks = [np.array(x) for x in freqQPeaks], [np.array(x) for x in intnQPeaks]

    return freqQPeaks, intnQPeaks

def convolve_multiplets(freqs, intns, freq_offs, n_spin=None):
    """Convolves peaks of a meq group of spins. freqs and intns are lists of np
       arrays computd if the spins are included in different subgraphs in the
       spin system.
    """
    if n_spin is None:
        n_spin = np.mean([np.sum(x) for x in intns])

    freq_out, intn_out = freqs.pop().ravel(), intns.pop().ravel()         # Intialize the results

    for freq in freqs:
        freq_out = (freq_out[None, :] + freq[:, None] - freq_offs).ravel()

    for intn in intns:
        intn_out = (intn_out[None, :] * intn[:, None]).ravel()

    # Scale the intensitie according to the number of spins
    intn_out /= intn_out.sum()/n_spin

    return freq_out, intn_out

# @ profile
def compute_transitions(chshQD, jcplQD, meqSpins, meqLinks):
    """Computes the transition lines for a spin system. Outputs the lists of np arrays corresponding to the freq and intn of the computed transition peaks. the lengths of the arrays equal the number of chshQD."""

    n_spin = sum([spin.nspin for spin in meqSpins])        # Number of spins in the system

    if len(meqLinks) == 0:
        # Case 1. All spins have the same chemical shift
        freqQPeaks, intnQPeaks = [[chshQD[0]]], [np.array([n_spin])]

    elif n_spin == 2:
        # Case 2. AB system
        freqQPeaks, intnQPeaks = QTransAB(chshQD, jcplQD)

    elif n_spin < 12:     # False: #
        # Case 2. Small spin system
        freqQPeaks, intnQPeaks = QTransFull(chshQD, jcplQD, meqSpins, meqLinks)

    else:
        # Case 3. Combined spin system
        freqQPeaks, intnQPeaks = QTransPairs(chshQD, jcplQD, meqSpins, meqLinks)

    return freqQPeaks, intnQPeaks

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

class chemSpec:
    """Database entiry of a chemical species.

    Attributes:

        name : str
            Name of the chemical
        chshH : list of parsSpec
            Definitions of distinct 1H spin chemical shifts.
        chshC : list of parsSpec
            Definitions of distinct 13C spin chemical shifts.
        jcplH : list of parsSpec
            Definitions of 1H J-couplings.
        spsyCombH, spsyCombC : list of spsyComb
            Combinations sof 1H and 13C spin systems.
        meqSpins : list of spinVert
            Each entry corresponds to a group of magnetically inequivalent 1H spins.
        meqLinks : list of spinEdge
            Each entry corresponds to cpoupling between groups of magnetically
            inequivalent 1H spins.
        multH, multC : list of int
            Multiplicities of different spin systems.
        Mw : float
            Molecular weight, g/mol.
        nh_labile : int
            Number of labile protons (quickly exchanging with water, e.g. OH, NHx, etc.).
        _spsyAsgnSpins : list of int
            Assignemnt of different spins (ordered according to their indices in
            meqSpins) to separate spin systems. Each entry is a spin system indexs
            to which a particular meqSpin is assigned.
        _spsyAsgnLinks : list of int
            Assignment of coupling links to sepearte spin systems.

        TODO : Check the functions for adding/removing of chem shifts and jcpl.
    """

    def __init__(self, name='', chshH=None, chshC=None, jcplH=None, multH=None,
                 multC=None, chshLabileH=None, jcplHC=None, meqSpins=None,
                 meqLinks=None, Mw=None, nH_labile=0, spsyCombH=None, spsyCombC=None, **kwargs):
        self.name = name
        self.chshH = chshH if chshH is not None else []       # List of chshH parsSpec's
        self.chshC = chshC if chshC is not None else []
        self.jcplH = jcplH if jcplH is not None else []    # List of jcplH parsSpec's
        self.spsyCombH = spsyCombH if spsyCombH is not None else []    # Combinations of spin systems, if any
        self.spsyCombC = spsyCombC if spsyCombC is not None else []

        # New format
        self.meqSpins = meqSpins if meqSpins is not None else []
        self.meqLinks = meqLinks if meqLinks is not None else []

        # # Old format
        self._spsyAsgnSpins, self._spsyAsgnLinks = [], []                         # list of spin system indices to which each meqSpin is assigned
        self.assignSpsy()                                                         # Determines the assignment indices and the number of spin systems, _nSpsyH
        self.multH = multH if multH is not None or [] else [1]*self._nSpsyH       # Multiplicities of different spin systems
        self.multC = multC if multC is not None or [] else [1]*len(self.chshC)

        self.Mw = Mw                                                            # Molar weight
        self.nH_labile = nH_labile                                              # Nunmber of labile protons

    def assignSpsy(self):
        """Determines spin systems based on coupling between meqSpins."""
        self._spsyAsgnSpins.clear()
        self._spsyAsgnLinks.clear()
        if len(self.meqSpins) == 0:
            # No protons
            self._nSpsyH = 0
        elif len(self.meqLinks) != 0:
            M = np.zeros((len(self.meqSpins), len(self.meqSpins)))    # Connectivity matrix; 1 if two spins are coupled
            ind = np.array([pair for indx, pair in self.meqLinks])     # Indices of used J-couplings, if any
            M[ind[:,0], ind[:,1]] = 1
            (self._nSpsyH, newAsgnH) = sps.csgraph.connected_components(M, directed=True)    # Number of spin systems and assignment of spins to spin systems
            self._spsyAsgnSpins.extend(newAsgnH)

            # Assign links to the spin systems
            self._spsyAsgnLinks.extend([self._spsyAsgnSpins[pair[0]] for _, pair in self.meqLinks])
        else:
            # There are only uncoupled protons; each meqSpin belongs to its own spin system
            self._nSpsyH = len(self.meqSpins)
            self._spsyAsgnSpins.extend([*range(len(self.meqSpins))])

    def add_chshH(self, nSpin=1, mult=1, **kwargs):
        """Add a new spin chemical shift to the specification."""
        # TODO : Unfinished!
        self.chshH.append(parsSpec(**kwargs))
        self.nSpinH.append(nSpin)
        self.multH.append(mult)
        # Add this new chemical shift as its own spin system initially (uncoupled)
        self._nSpsyH += 1
        # self._spsyAsgnSpins.append(len(self.chshH)-1)
        pass

    def del_chshH(self, indx):
        # TODO : Unfinished!
        self.chshH.pop(indx)
        self.nSpinH.pop(indx)
        # Reduce all indices that larger than indx by 1
        ##nSpsy_old = self._nSpsyH
        ##spsyAsgn_old = self._spsyAsgnSpins[indx]
        multBySpin = [self.multH[i] for i in self._spsyAsgnSpins if i != indx]     # Spin system multiplicities for each spin
        for i, j in itertools.product(range(len(self.pairHH)), [0, 1]):
            if self.pairHH[i][j] == indx or self.pairHH[i][j] is None:
                self.pairHH[i][j] = None
            elif self.pairHH[i][j] > indx:
                self.pairHH[i][j] -= 1
        self.assignSpsy()
        ##if self._nSpsyH < nSpsy_old: self._multH.pop(spsyAsgn_old)    # Remove the multiplicity if there is no longer this spin system
        # Update the multiplicities
        self.multH.clear()
        self.multH.extend([max([mult for mult, j in zip(multBySpin, self._spsyAsgnSpins) if j==i]) for i in range(self._nSpsyH)])        # Find multiplicities for each spin system

    def add_jcplH(self, nSpin=1, mult=1, **kwargs):
        # TODO : Unfinished!
        self.jcplH.append(parsSpec(**kwargs))
        self.pairHH.append([None, None])

    def del_jcplH(self, indx):
        # TODO : Unfinished!
        self.jcplH.pop(indx)
        self.pairHH.pop(indx)
        # Reduce all indices that larger than indx by 1
        multBySpin = [self.multH[i] for i in self._spsyAsgnSpins]     # Spin system multiplicities for each spin
        self.assignSpsy()
        # Update the multiplicities
        self.multH.clear()
        self.multH.extend([max([mult for mult, j in zip(multBySpin, self._spsyAsgnSpins) if j==i]) for i in range(self._nSpsyH)])        # Find multiplicities for each spin system

    def set_jcplH(self, indx, pair, label=None):
        """Sets a jcpl constant index to the pair of protons; pair is a 2-list of
        proton indices."""
        # TODO : Unfinished!
        if len(pair)!=2 or pair[0] == pair[1]: return False
        for p in self.pairHH:
            if set(pair) == set(p):
                print("This coupling is already defined.")
                return False

        multBySpin = [self.multH[i] for i in self._spsyAsgnSpins]     # Spin system multiplicities for each spin
        self.pairHH[indx] = pair

        self.assignSpsy()
        # Update the multiplicities
        self.multH.clear()
        self.multH.extend([max([mult for mult, j in zip(multBySpin, self._spsyAsgnSpins) if j==i]) for i in range(self._nSpsyH)])        # Find multiplicities for each spin system
        # Update the label for this j-coupling constant
        if label is None:
            label = self.chshH[pair[0]].label if pair[0] is not None else ''
            label += '-'
            label += self.chshH[pair[1]].label if pair[1] is not None else ''
        self.jcplH[indx] = self.jcplH[indx]._replace(label = label)

    def add_chshC(self, mult=1, **kwargs):
        # TODO : Unfinished!
        self.chshC.append(parsSpec(**kwargs))
        self.multC.append(mult)

    def del_chshC(self, indx):
        # TODO : Unfinished!
        self.chshC.pop(indx)
        self.multC.pop(indx)

    def getSpSy(self, mode='1H'):
        """Returns a list of spin systems."""
        # NOTE: Left for compatibility with old chemNodeDB.
        if mode == '13C':
            spsyBig = spsySpec(chsh=self.chshC, jcpl=[], chshAsgn=list(range(1,len(self.chshC)+1)), jcplAsgn = None)
            mult = self.multC
        elif mode == '1H':
            # print(self.meqSpins, self.meqLinks)
            spsyBig = spsySpec(self.chshH, self.jcplH, *meqv2asgn(self.meqSpins, self.meqLinks))
            mult = self.multH

        # Split the big spin system
        spsyAll = splitSpSy(spsyBig, mult)

        return spsyAll

    def asdict(self):
        """Returns a dictionary representation of the chemSpec class object."""
        return {'name' : self.name,
                'chshH': self.chshH,
                'chshC': self.chshC,
                'multH': self.multH,
                'multC': self.multC,
                'jcplH': self.jcplH,
                'meqSpins': self.meqSpins,
                'meqLinks': self.meqLinks,
                'spsyCombH': self.spsyCombH,
                'spsyCombC': self.spsyCombC,
                'Mw': self.Mw}

class treeNode:
    """ A class used to represent a generic node in a hierarchical structure.

        Defines the logic of the tree hierarchy. Will be subclassed to create
        more specialised chemical tree nodes.

    Attributes:
        name: str
            The name of the node; must be unique within the tree.
        alias: str
            An alternative name used for display purposes; can be duplicated.

    """

    def __init__(self, name, alias='', **kwargs):

        self.name = name
        self.alias = alias
        self._parent = None
        self._children = []       # List of treeNode instances

        # Set a dictionary of references to other nodes in the tree in the form:
        # 'nodeNane: nodeInstance'. The same dictionary is shared by all nodes.
        self._treeBook = weakref.WeakValueDictionary({self.name:self})

    def __str__(self):
        if self.alias is None or self.alias == '':
            return str(self.name)
        else: return self.alias

    def __getitem__(self, key):
        # for item in self.items():
        #     if item.name == key:
        #         return item
        return self._treeBook[key]

    def __getstate__(self):
        """Called when pickling called and the returned object is pickled as the contents for the instance, instead of the contents of the instance’s dictionary."""
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
        """Update the map of weak references. Can be used after unpickling or for grafting."""
        newBook = weakref.WeakValueDictionary()
        for node in self.items():
            newBook.update({node.name:node})
            node._treeBook = newBook

    def keys(self):
        return self._treeBook.keys()

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
        return self._children[pos]

    def children(self):
        yield from self._children

    def siblID(self):
        """Position of the node among its siblings; 0 if there are no siblings."""
        if self._parent is not None:
            return self._parent._children.index(self)
        else: return 0

    def addChild(self, child, pos=-1):
        if child._parent is None:
            # Check for potential name conflicts
            commonNames = set([node.name for node in child.items()]).intersection([node.name for node in self.items()])
            if not commonNames:
                child._parent = self
                if pos >= 0 and pos < len(self._children):
                    self._children.insert(pos, child)
                else:
                    self._children.append(child)
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

        self._children.insert(pos, child)
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
        self._children.remove(child)
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
        self.setTreeBook()

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
            # Starting from the top
            yield self      # Output the key and the node
            for c in self._children: yield from c.iterDepth("in-order")
        elif method == "post-order":
            # Starting from the bottom
            for c in self._children: yield from c.iterDepth("post-order")
            yield self

    def iterBreadth(self):
        """Breadth-first iterator (a sequence of nodes that includes self)."""
        # queue = []
        # yield self
        # for c in self._children:
        #     queue.append(c)
        #     yield from c.iterBreadth()
        pass

    def descendants(self, include_self=False):
        """All descendants of the node (excluding itself by default)."""
        if include_self: yield self
        for child in self._children:
            yield child
            yield from child.descendants()

    def ancestors(self, include_self=False):
        """All ancestors of the node (excluding itself by default), including the root."""
        if include_self: yield self
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
    """Node in the treeView model in the GUI."""

    def __init__(self, name, alias='', nodeType=None, hidden=False, meta=None):
        super().__init__(name, alias)
        self.nodeType = nodeType
        self.hidden = hidden
        self.meta = meta

class parsNode(treeNode):
    """Node in the treeView model in the GUI to represent a parameter (e.g. chsh)."""

    def __init__(self, name, alias='', crnt=0.0, ancs=0.0):
        super().__init__(name, alias)
        self.crnt = crnt
        self.lims = np.zeros(2)
        self.ancs = ancs      # The value of the parameter pulled from the ancestors (e.g. sum of all chemical shifts above)

        name = self.name[0]
        indx = self.name[2]
        sfx = self.name[1][4:]    # the 'QD' suffix

        # A list of keys related to this chemical shift (chsh, alph, ampl, etc.)    (name, 'chsh'+sfx, indx)
        self.keys = [(name, 'alph'+sfx, indx),
                     (name.replace('SPSY', '')+'.'+str(indx+1) if sfx else name, 'ampl', 0)]

    def propLims(self, limsPrnt=None):
        """Propagates the limits of the parameter through the tree."""

        # Initialize the lims to the ancestral value
        if limsPrnt is None:
            limsPrnt = self.ancs*np.ones(2) if self.isRoot() else self._parent.lims

        self.lims = self.crnt + limsPrnt
        for chld in self.children():
            chld.propLims(self.lims)

    def isInRange(self, range, offset=0.0):
        """Checks whether the self.lims fall into any of the subintervals in the range. Range is a list of tuples, e.g. minmaxTuple."""
        try:
            minl = min(self.lims) + min(offset)
            maxl = max(self.lims) + max(offset)
        except TypeError:
            minl, maxl = min(self.lims)+offset, max(self.lims)+offset
        return any([minl > min(r) and maxl < max(r) for r in range])

    def propCrnt(self, crntPrnt=None):
        """Propagates the current values of the parameter through the tree."""

        # Initialize the lims to the ancestral value
        if crntPrnt is None:
            crntPrnt = self.ancs if self.isRoot() else self._parent.crnt

        self.crnt = self.crnt + crntPrnt
        for chld in self.children():
            chld.propLims(self.crnt)

class chemNode(treeNode):
    """Main chemical species node.

    Attributes:
        chsh : 1-list of parsSpec
            Prior specification of the global chemical shift of the node
        alph : 1-list of parsSpec
            Prior specification of the global relaxation rate of the node
        ampl : 1-list of parsSpec
            Prior specification of the global model amplitude of the node
        phase : 1-list of parsSpec
            Prior specification of the global phase of the node
        intn : float
            Constant intensity paraemeter used to group several nodes (may
            correspond, e.g. to relative concentrations of different isomers,
            if they are to remain unchanged).
    """

    def __init__(self, name, chsh = None, alph = None, ampl = None, phase = None, intn = 1., alias='', **kwargs):
        super().__init__(name, alias, **kwargs)
        self._reported = True
        self.chsh = chsh if chsh is not None else [parsSpec(min=-0.5, max=0.5)]
        self.alph = alph if alph is not None else [parsSpec(min=-1., max=5., dval=0.0)]
        self.ampl = ampl if ampl is not None else [parsSpec(min=0., max=np.inf, distr='Gaussian', p1=0.0, p2=np.inf, dval=0.0)]
        self.phase = phase if phase is not None else [parsSpec(distr='Uniform', min=-np.pi, max=np.pi, dval=0.0)]
        self.intn = intn         # Global intensity
        self.sT = []             # self-response
        self.uF = []             # frequency-domain response
        self.uT = []             # response that includes children/parents along the tree
        self.sPole = 0.          # self-pole determined by alph and chsh
        self.uPoles = 0.         # Poles that includes the effect of all parents
        self._oldHash = None
        self._oldLeafPoles = None    # Poles of all leaf nodes
        self._t_shift = None         # Time samples array for shifting

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

    def setPrior(self, key, par):
        """Sets the prior distribution in the key (possibly in a different node) to parsSpec par."""
        if not isinstance(par, parsSpec):
            raise RuntimeError('The distribution must be of the type parsSpec.')
        if len(key) != 3:
            raise RuntimeError('The key must be a 3-tuple.')

        # Replace the corresponding distribution
        getattr(self[key[0]], key[1])[key[2]] = par

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
        # repRoots = [v for v in root.iterDepth('in-order') if v.isReported() and not v.parent().isReported()] if not root.isReported() else [root]
        if not root.isReported():
            for v in root.iterDepth('in-order'):
                if v.isReported() and not v.parent().isReported():
                    yield v
        else:
            yield root

    def addChild(self, child, pos=-1):
        treeNode.addChild(self, child, pos)
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
        self.uF = []
        self._oldHash = None
        self._oldLeafPoles = None

    def getPoles(self, c0, chsh=None, alph=None, **kwargs):
        """Computes the poles and returns 1 if they have changed, 0 otehrwise"""
        if chsh is not None and alph is not None:
            newPole = 1j*2*np.pi*c0*chsh[0] - alph[0]
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

    def getChshTree(self, sfx='', indx=0):
        """Creates a tree by arranging chemical shift for all descendants of the node."""
        P = parsNode(name = (self.name, 'chsh', 0) )                 # e.g. ('Mixture', 'chsh', 0)
        for chld in self.children():
            P.addChild(chld.getChshTree())
        return P

    #@profile
    def evalTime(self, t, **kwargs):
        "Computes the node's response u=sPoleIntn*exp(-alph*t+i*omega*t)"
        newHash = arrhash(t)
        if self.sT == [] or self._oldHash != newHash:
            self.sT = self.intn if self.sPole == 0 else self.intn * np.exp(np.outer(t, self.sPole)).ravel()

            self._oldHash = newHash

    # @profile
    def evalFreq(self, f, dt, df, c0, f0=0, tau=0, allowShift=True):
        "Computes the node's response in the frequency domain assuming that all nodes have updated uPoles."
        # # Check if the signal needs to be completely reevaluated
        newHash = arrhash(f)
        newLeafPoles = np.concatenate([np.array(leaf.uPoles).ravel() for leaf in self.leaves()])#.ravel()      # New poles for all leaves

        if allowShift and newHash == self._oldHash:
            try:
                diffLeafPoles = self._oldLeafPoles - newLeafPoles
                # If all new poles are just shifted old poles, shift the resulting response
                if np.isclose(max(diffLeafPoles.real), min(diffLeafPoles.real)) and np.isclose(max(diffLeafPoles.imag), min(diffLeafPoles.imag)):
                    if self._t_shift is None or self.uT is None:
                         nf_all = int(np.round(1/(dt*df*c0)))            # Length of the full (initial) spectrum
                         nf_new = len(f)
                         dt_new = dt / nf_new * nf_all                   # The equivalent sampling time for the reduced frequency range

                         self._t_shift = np.fft.fftshift(np.linspace(-(nf_new+1)*dt_new/2, (nf_new-1)*dt_new/2, nf_new), axes=0).ravel()
                         self.uT = np.fft.ifft(np.fft.ifftshift(self.uF, axes=0), axis=0)

                    shift = -diffLeafPoles[0]
                    sT = np.exp( 1j*shift.imag*self._t_shift + shift.real*np.abs(self._t_shift) )
                    self.uT = self.uT * sT
                    self.uF = np.fft.fftshift(np.fft.fft(self.uT, len(self.uT), axis=0), axes=0) # / np.sqrt(len(yTs))

                    self._oldLeafPoles = newLeafPoles
                    return None
            except ValueError: pass

        # Otherwise -- Full computations
        self.uF = np.zeros((len(f), 1), dtype='complex128').ravel()
        for chld in self.children():
            chld.evalFreq(f, dt, df, c0, f0, tau, allowShift=allowShift)
            self.uF += chld.uF
        self.uF *= self.intn
        self.uT, self._t_shift = None, None              # Reset the time-domain signal

        # Update saved hash and poles
        self._oldHash = newHash
        self._oldLeafPoles = newLeafPoles

class chemNodeQM(chemNode, chemSpec):
    """Node in chemTree describing a chemical from the database.

    The node can be specified either by passing a name of a species in the database
    or the QDpars structure (an instance of chemSpec class.)
    """

    def __init__(self, name, chsh = None, alph = None, alphQD = None, ampl = None, phase = None, intn = 1., alias='', HCmode='1H', QDpars=None, nameDB=None):
        if QDpars is None:
            # Load from the database if QD parameters are not supplied
            chemDB = {key:val for _, db in chemLib.items() for key, val in db.items()}
            if name in chemDB or nameDB in chemDB:
                QDpars = copy.deepcopy(chemDB[name if nameDB is None else nameDB])    # Parameters from the database
            else:
                # TODO: Add an empty spin system
                raise RuntimeError("The chemical \'" + self.name + '\' is not in the database and no QD parameters are supplied.')

        QDpars.name = name
        # TODO: Fix multiple inheritance (worked in Python 3.5...)
        super().__init__(chsh=chsh, alph=alph, ampl=ampl, phase=phase, intn=intn, alias=alias, **QDpars.asdict())
        chemSpec.__init__(self, chsh=chsh, alph=alph, ampl=ampl, phase=phase, intn=intn, alias=alias, **QDpars.asdict())

        # Reset the topology of the spin systems
        self.HCmode = HCmode
        self.spinTopo = []          # Array of not connected spin Groups, each representing a specific spin system
        self._indxChsh_by_spsy, self._indxJcpl_by_spsy = [], []          # Arrays of indices of chsh/jcpl from the large chshH/jcplH arrays when assigned to separate spin systems
        self._setSpinTopo()
        # self.dendrolize()

        # Add peak width parameters
        self.alphQD = alphQD if alphQD is not None else [parsSpec(min=-5.0, max=25.0, label=c.label, dval=0) for c in (self.chshQD)]

    def __getattr__(self, attr):
        """Called only when an attribute is not found."""

        if attr == 'chshQD':
            return self.chshH if self.HCmode=='1H' else self.chshC
        elif attr == 'spsyComb':
            return self.spsyCombH if self.HCmode=='1H' else self.spsyCombC
        elif attr == 'jcplQD':
            return self.jcplH if self.HCmode=='1H' else []

    def rename(self, newName):
        for chld in self.descendants():
            chld.rename(newName=chld.name.replace(self.name, newName))
        super().rename(newName)

    def default_pars(self):
        """Returns a dictionary of default parameters for the node."""
        pars = chemNode.default_pars(self)
        pars["chshQD"] = [par.dflt() for par in (self.chshQD)]
        pars["alphQD"] = [par.dflt() for par in self.alphQD]
        if self.HCmode == '1H':
            pars["jcplQD"] = [par.dflt() for par in self.jcplH]
        return pars

    def priors(self):
        """Returns a dictionary of prior parameter specifications for the node."""
        result = chemNode.priors(self)
        pars["chshQD"] = [par for par in (self.chshQD)]
        pars["alphQD"] = [par for par in self.alphQD]
        if self.HCmode == '1H':
            pars["jcplQD"] = [par for par in self.jcplH]
        return result

    def setDefaultQD(self, key, dval, min=None, max=None):
        """Sets (updates) the default distributions of QD parameters. key is a 2-tuple of the form ('chshH', i), ('jcplH', i), or ('chshC', i), where i is the number of the parameter in the zero-order, e.g. ('chshH', 2) for the third chemical shift."""

        # Check if the entire list of parameters need to be updated (e.g. all chshH or all jcplH, etc.)
        if not isinstance(key, tuple):
            if len(getattr(self, key)) == len(dval):
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
            parsArray = getattr(self, key[0])          # An entire array of the parameters, one of which needs to be updated
            parsArray[key[1]] = parsArray[key[1]]._replace(dval=dval, min=min, max=max)

    def _setSpinTopo(self):
        """Returns a list of separate not connected spin systems for a specific mode, 1H/13C. The indices of chsh and jcpl parameters are rebased in each spin system s.t. they start from 0."""
        self.spinTopo.clear()
        self._indxChsh_by_spsy.clear()
        self._indxJcpl_by_spsy.clear()

        if self.HCmode == '1H':
            for indxSpsy in range(self._nSpsyH):
                # Collect only spins and links that belong to the current spin system and renumber them
                indxSpin_old = [i for i, indxSpsy_asgn in enumerate(self._spsyAsgnSpins) if indxSpsy_asgn == indxSpsy]           # Indices of those spins that belong to the current spin system in the big array of all spins, self.meqSpins
                meqSpins_sub = [self.meqSpins[i] for i in indxSpin_old]
                meqLinks_sub = [link._replace(indxVert=[indxSpin_old.index(i) for i in link.indxVert])
                                for link, indxSpsy_asgn in zip(self.meqLinks, self._spsyAsgnLinks) if indxSpsy == indxSpsy_asgn]

                # Find indices for chsh and jcpl in the old arrays
                indxChsh_old = sorted(set([spin.indxChsh for spin in meqSpins_sub]))
                indxJcpl_old = sorted(set([link.indxJcpl for link in meqLinks_sub]))
                self._indxChsh_by_spsy.append(indxChsh_old)
                self._indxJcpl_by_spsy.append(indxJcpl_old)

                # Rebase the assignemt of chsh and jcpl parameters s.t. they start from 0 in each sub-spin system
                meqSpins_sub = [spin._replace(indxChsh=indxChsh_old.index(spin.indxChsh)) for spin in meqSpins_sub]
                meqLinks_sub = [link._replace(indxJcpl=indxJcpl_old.index(link.indxJcpl)) for link in meqLinks_sub]

                self.spinTopo.append(spinGroup(meqSpins_sub, meqLinks_sub, mult=self.multH[indxSpsy]))

        elif self.HCmode == '13C':
            self.spinTopo.extend( [spinGroup(meqSpins=[spinVert(0, 1)], meqLinks=[], mult=self.multC[i]) for i in range(len(self.chshC))] )
            self._indxChsh_by_spsy.extend( [[i] for i in range(len(self.chshC))] )
            self._indxJcpl_by_spsy.extend( [[] for _ in range(len(self.chshC))] )

        # 3. -------------- Dendrolize the node -----------------
        # Creates chemTrees based on the QD parameters of the node. Each new
        # child node corresponds to a chshQD parameter, not meqSpins

        # Add nodes to the tree
        for chld in self.children():             # Loop backwards to avoid missing children when the index increases but the number of children decreases
            self.removeChild(chld)

        # Add nodes for spin combinations, if any
        for i, comb in enumerate(self.spsyComb):
            self.addChild(chemNode(self.name + '-COMB' + str(i+1), intn=comb.intn, alias=comb.name))

        # Add the terminal nodes
        for i, spsy in enumerate(self.spinTopo):
            # Add a QD node with their own terminal nodes
            prntNode = self           # The parent node to which the spin system will be attached; by default, directly to chemNodeQM

            for j, comb in enumerate(self.spsyComb):
                if i in comb.indxSpsy:
                    prntNode = self[self.name + '-COMB' + str(j+1)]
                    break

            for j in range(spsy.n_chsh()):
                label = self.chshQD[self._indxChsh_by_spsy[i][j]].label# if self.HCmode == '1H'\
                                    #else self.chshC[self._indxChsh_by_spsy[i][j]].label
                nodeT = chemNodeQT(self.name + '-' + str(i+1) + '.' + str(j+1), intn = spsy.mult * spsy.n_spin()[j],
                                        alias = self.name + ' ' + label if label != '' else  '')
                prntNode.addChild(nodeT)

    def reset(self):
        """Reset the saved old parameters in the node. Evrything will be recomputed on the next step."""
        super().reset()

    # @profile
    def getPoles(self, c0, chsh=[], alph=[], chshQD=[], alphQD=[], jcplQD=[], **kwargs):
        """Computes the poles for all peaks and all children, running QD simulations if needed."""
        super().getPoles(c0, chsh, alph)     # Compute sPole

        ## Values of the QD parameters
        freqQD = c0*np.array(chshQD)          # Array of absolute values of chemical shifts (in Hz)
        alphQD = np.array(alphQD)
        jcplQD = np.array(jcplQD)

        # Loop over the spin systems and update the corresponding children nodes
        for i_spsy, spsy in enumerate(self.spinTopo):

            freqQD_spsy = freqQD[self._indxChsh_by_spsy[i_spsy]]          # Frequencies restricted to a specific spin system
            alphQD_spsy = alphQD[self._indxChsh_by_spsy[i_spsy]]
            jcplQD_spsy = jcplQD[self._indxJcpl_by_spsy[i_spsy]]

            freqQPeaks_all, intnQPeaks_all, flag_updated = spsy.get_transitions(freqQD_spsy, jcplQD_spsy)

            for i_chsh, (freqQPeaks, intnQPeaks, alphQPeaks) in enumerate(zip(freqQPeaks_all, intnQPeaks_all, alphQD_spsy)):
                chld = self[self.name+'-'+str(i_spsy+1)+'.'+str(i_chsh+1)]    # Terminal node corresponding to the specific chemical shift
                chld.getPoles(c0, [freqQD_spsy[i_chsh]/c0], [alphQPeaks])                   # Sets the sPole of the child and resets the node if it has changed
                if flag_updated:
                    # If qPoles have been updated
                    # TODO: Simplify peaks / aggregate several peaks (possibly introduce line widths when combining QPeaks)
                    chld.reset()
                    qPoles, qPolesIntn = group_peaks(freqQPeaks, intnQPeaks, maxWidth = config.QD_AggregatePeaksThreshold)                    # relative values of peak positions in ppm
                    chld.qPoles = 1j*2*np.pi*np.array(qPoles)
                    chld.qPolesIntn = np.array(qPolesIntn) / chld.intn    # Scale all qPoles for a given T node by the number of nuclei with the same chemical shift (i.e. the intensity of the node)

    def getChshTree(self, sfx='', indx=0):
        """Create a parameters tree. Takes into account the parsKind parameter of
        itself and also all QD parameters, but omits any attached chemNodeQT
        children. sfx = '' or 'QD'.

        """

        P = parsNode( name = (self.name, 'chsh'+sfx, indx) )       # Works for QD parameters as well

        if not sfx:
            for i in range( len( self.chshQD ) ):
                P.addChild( parsNode( name = (self.name, 'chshQD', i) ) )

        return P

class chemNodeQT(chemNode):
    "Terminal nodes that emit signals. Can only be used as leaves."
    def __init__(self, name, chsh = None, alph = None, ampl = None, phase = None, intn = 1., alias=''):
        chemNode.__init__(self, name, chsh, alph, ampl, phase, intn, alias)
        self.qPoles = np.array([0.])                # QD poles from the parent node that determine peak splitting
        self.qPolesIntn = 1.
        self.uPoles = 0.                # Poles computed including the effects of all ancestors
        self.uF = []
        self.qT = []

    def default_pars(self):
        """Returns a dictionary of default parameters for the node."""
        return {'ampl':[self.ampl[0].dflt()], 'phase':[self.phase[0].dflt()]}

    #@profile
    def evalTime(self, t, **kwargs):
        "Computes the node's response sT"
        newHash = arrhash(t)

        if self.qT == [] or self._oldHash != newHash:
            self.qT = np.inner( np.exp(np.outer(t, self.qPoles)), self.qPolesIntn ).ravel()
        if self.sT == [] or self._oldHash != newHash:
            self.sT = self.intn * self.qT * np.exp(np.outer(t, self.sPole)).ravel()

        self._oldHash = newHash

    def addChild(self, child, pos=-1):
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
            self.uPoles = 1j*newPoles.imag + np.minimum(newPoles.real, 0.0)
            self.uF = []    # Reset the output in the frequency domain

    def reset(self):
        super().reset()
        self.qT = []

    # @njit
    # @profile
    def evalFreq(self, f, dt, df, c0, f0=0, tau=0, allowShift=True):
        "Computes the node's response in the frequency domain assuming that all ancestors have updated uPoles."
        # Check if the signal needs to be reevaluated
        newHash = arrhash(f)

        if self.uF == [] or self._oldHash != newHash:
            self.uF = np.exp(1j*tau*(self.uPoles.imag - 2*np.pi*f0)).reshape((1,-1))
            x1 = 1j*2*np.pi*(c0*f-f0).reshape((-1,1))
            x2 = np.conj(self.uPoles - 1j*2*np.pi*f0).reshape((1,-1))
            # self.uF = self.uF / -np.expm1((x1+x2)*dt)
            self.uF = ne.evaluate( 'x / -expm1( (x1 + x2)*dt )', local_dict={'x':self.uF, 'x1':x1, 'x2':x2, 'dt':dt})       # Compute exp(x)-1 in one go
            self.uF = ne.evaluate('sum(conj( x ) * y, axis=1)', local_dict={'x':self.uF, 'y':self.qPolesIntn}).ravel()
            # self.uF = np.inner(np.conj(self.uF), self.qPolesIntn).ravel()
            self.uF *= self.intn * np.sqrt(df*c0*dt)

            self._oldHash = newHash

    def get_parKey(self, pars='chshQD'):
        """Returns the key of the parameter associated with the current terminal node."""
        name = self.name.rsplit('-', 1)
        i, j = name[1].split('.')
        indx = self[name[0]]._indxChsh_by_spsy[int(i)-1][int(j)-1]      # Parent chemNodeQM node that generated this terminal node
        return (name[0], pars, indx)


# ------------------------- Functions for working with trees -------------------------------

def defaultTreePars(tree, tau=0.0, theta=0.0, sigma2=0.0, gamma=0.0, startFromRoot=True):
    """Returns a nested dictionary of default tree parameters."""
    if startFromRoot:
        tree = tree.findRoot()
    pars = {node.name : node.default_pars() for node in tree.descendants(include_self=True)}
    pars["."] = {"tau" : [tau], "theta" : [theta],
                 "mult" : [1.0], "sigma2" : [sigma2], 'gamma':[gamma],
                 "lshapeR" : [0.0]*config.MODEL_LineShapeOrder,
                 "lshapeI" : [0.0]*config.MODEL_LineShapeOrder}
    return pars

def evalTreeT(tree, t, c0, pars=None, xclRootNames=None):
    """Evaluate the entire tree of chemNodes. Returns the time-domain response
        for the specified (reported) nodes in the tree. tree is a chemNode object --
        any node in the tree; pars - a nested dictionary of parameters, where the
        first level is indexed by the names of the nodes, and the second level
        conatins the names of parameters.

    """
    if pars is None:
        pars = defaultTreePars(tree)

    root = tree.findRoot()

    # A set of excluded RootNames
    if xclRootNames is None:
        xclRootNames = set([])

    # 1. Evaluate all nodes (computes self-responses s(t))
    for node in root.items():
        node.getPoles(c0, **pars[node.name])
        node.evalTime(t)
        #print(node.sT)

    # 2. Determine the root reported nodes (determine the reported subtrees)
    repRoots = [v for v in tree.repRoots() if v.name not in xclRootNames]

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
    for node in root.iterDepth(method="in-order"):     # All nodes ABOVE the reported nodes
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

def evalTreeF(tree, f, dt, df, c0, f0=0, pars=None, xclRootNames=None, allowShift=False):
    """Evaluates the entire tree of chemNodes and returns a model spectrum
        directly in the frequency domain. Tree is a chemNode object -- any node
        in the tree; pars - a nested dictionary of parameters, where the first
        level is indexed by the names of the nodes, and the second level conatins
        the names of parameters
    """

    if pars is None:
        pars = defaultTreePars(tree)

    tau = 0     #    or use
    #tau = -pars['.']['tau'][0]
    if xclRootNames is None:
        xclRootNames = set([])

    # 1. Update all poles of each node in the tree and propagate them to find uPoles of the leaves
    for node in tree.items():
        node.getPoles(c0, **pars[node.name])
    tree.findRoot().propPoles()     # Propagate all poles

    # 2. Determine the root reported nodes (determine the reported subtrees)
    repRoots = [v for v in tree.repRoots() if v.name not in xclRootNames]

    # 3. Collect the childrens' responses, starting from the bottom
    for rep in repRoots:
        rep.evalFreq(f, dt, df, c0, f0, tau, allowShift=allowShift)

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
        allPeaks[rep.name] = {leaf.name : [peakSpec(freq=pole.imag/(c0*np.pi*2), intn=leaf.qPolesIntn[i]*leaf.intn, fwhm=-pole.real/np.pi) for i, pole in enumerate(leaf.uPoles)] \
                              for leaf in rep.leaves() if leaf.uPoles.size > 0}
        #allPeaks[rep.name] = [peakSpec(freq=pole.imag/(c0*np.pi*2), intn=leaf.qPolesIntn[i]*leaf.intn, fwhm=-pole.real/np.pi) for leaf in rep.leaves() if leaf.uPoles.size > 0 for i, pole in enumerate(leaf.uPoles)]

    return allPeaks

def getFID(T, t, c0, f0=0, pars=None, tau=None, xclRootNames=None):
    """Returns modeled signals in the time domain in the form of FID."""
    if pars is None:
        pars = defaultTreePars(T)
    if tau is None:
        tau = pars["."]["tau"][0]
    Z, repRootNames = evalTreeT(T, np.array(t)+tau, c0, pars=pars, xclRootNames=xclRootNames)
    # Shift the signal by f0
    Z = Z * np.exp(-1j*2*np.pi*f0*(np.array(t)+tau)).reshape(-1,1)
    # Apply lineshape correction
    try:
        ksi = np.array(pars['.']['lshapeR']) + 1j * np.array(pars['.']['lshapeI'])
        Z = Z * np.exp(np.inner(np.array(t).reshape(-1,1) ** np.arange(2, 2+ksi.size, 1).reshape(1,-1), ksi)).reshape(-1,1)
    except KeyError: pass
    #Z [0,:] /= 2
    return Z, repRootNames

# Functions for saving and loading predefined trees
def saveTree(fname, tree):
    """Saves a chemTree datastructure along with its default parameters within a pickled format."""
    tree = copy.deepcopy(tree)
    for node in tree.items():
        node.reset()

    if os.path.splitext(fname)[1] != '.ctr':
        fname += '.ctr'
    with open(fname, 'wb') as fp:
        dill.dump(tree, fp)

def loadTree(fname):
    """Loads a tree from the file fname."""
    if os.path.splitext(fname)[1] != '.ctr':
        fname += '.ctr'

    with open(fname, 'rb') as fp:
        data = dill.load(fp)

        # Check if the data is a dictionary (kept for compatibility with the prewious format)
        if isinstance(data, dict):
            T = data["tree"]
        else: T = data

        # Compatibility check: Make sure that each node in the tree has an ampl and a phase attributes
        for node in T.items():
            if not hasattr(node, 'ampl'): node.ampl = [parsSpec(min=0., max=np.inf, distr='Gaussian', p1=0.0, p2=np.inf, dval=0.0)]
            if not hasattr(node, '_oldHash'): node._oldHash = None
            if not hasattr(node, '_oldLeafPoles'): node._oldLeafPoles = None
            if not hasattr(node, '_t_shift'): node._t_shift = None
            if not hasattr(node, 'phase'): node.phase = [parsSpec(distr='Uniform', min=-np.pi, max=np.pi, dval=0.0)]
            # if isinstance(node, chemNodeQD) and not hasattr(node, 'spinTopo'): node.spinTopo = spinGroup(*asgn2meqv(node.chshAsgn, node.jcplAsgn))
            # if isinstance(node, chemNodeQM): node._setSpinTopo()
            node._children = list(node._children)

        T.setTreeBook()

    return T

# ----------------------------- Utility functions ------------------------------
def arrhash(x):
    """Very quick and dirty function to hash np arrays."""
    return x[0] + x[-1] + len(x)

def ind2pos(ind):
    """Converts an n-array of unique integers from 0 to n-1 (e.g. indices) into
       the array (positions) whose j-th entry corresponds to the index the
       number j in the original array."""
    pos = np.zeros_like(ind)
    for i, j in enumerate(ind):
        pos[j] = i
    return pos

# ----------------------------- Chemical Library -------------------------------

def array2parsSpec(arr):
    """Converts 2D arrays of min and max values to the array of parsRange namedtuples."""
    ans = arr
    if arr is not None:
        ans = [parsSpec(*v) for v in arr]
        # ans = [parsSpec(min=v[0], max=v[1], label=v[2] if len(v)>2 else '', ) for v in arr]
    else: ans = []
    return ans

def parsSpec2array(par):
    """Converts arrays of parsSpec namedtuples to 2D arrays of min and max values."""
    return [(v.min, v.max) for v in par]

def readChemDB_SDF(fname):
    """Import a model for chemical species from an .sdf file.

        This function loads a .mol table and chemical shifts/J-coupling
        assignemts from nmrdb.org and relates them to each other. It is
        assumed that the _orders_ of indices of spins in both arrays are
        the same (though actual indices may differ).

        The entries from the .mol (.sdf) file are reffered to as atoms,
        and those from the nmrdb table -- as spins.

        Args:
            fname: str
                A path to the .mol or .sdf file containing nmrdb data.
        Returns:
            chemModel: chemSpec
                Chemical species model in chemSpec format.
    """

    def find_by_attr(arr, key, val):
        """Find the index (order) of an element (atom or bond) in a list of dictionaries.
        Args:
            arr : list
                A list of dictionaries, e.g. atom, bonds, or spins
            key : hashable
                A dictionary key to match, e.g. 'indx'
            val : object
                A value in the dictionary key:val pair to match
        Returns:
            i : int or None
                Index of the dictionary in the list such that dict[key] == val. None if such dictionary is not found.
        """

        for i, x in enumerate(arr):
            if x[key] == val:
                return i
        return None

    def get_bond_matrix(atoms, bonds):
        """Compute the connectivity matrix given a list of atoms and bonds.

            The entries encode the number of bonds between the atoms.
        """

        bond_matrix = np.zeros((len(atoms), len(atoms)), dtype=int)
        for bond in bonds:
            a1, a2 = bond['atom_ids']     # Indices of connected atoms
            i1, i2 = find_by_attr(atoms, 'indx', a1), find_by_attr(atoms, 'indx', a2)
            bond_type_int = {'Single':1, 'Double':2, 'Triple':3}[bond['type']]
            bond_matrix[i1, i2] = bond_type_int
            bond_matrix[i2, i1] = bond_type_int
        return bond_matrix

    def assign_spsy(conn_H_matrix):
        """Determine the assignment of spins to spin systems."""

        n_spin = conn_H_matrix.shape[0]

        # Build the (block) connectivity matrix _within_ spin systems (1 if spins in the same system, 0 - otherwise)
        _old = np.minimum(conn_H_matrix + np.eye(n_spin, dtype=int), 1)
        while True:
            conn_matrix = np.minimum(_old.dot(_old), 1)
            if (_old == conn_matrix).all():
                break
            _old = conn_matrix

        #
        asgn_list = []   # List of assignemts; each element - array of spin indices assigned to a correpondingspin system
        unassigned = list(range(n_spin))   # Indices of spins (atoms) yet to be assigned to spin systems
        while len(unassigned) > 0:
            asgn_list.append( np.argwhere(conn_matrix[unassigned[0],:]).ravel() )
            unassigned = [x for x in unassigned if x not in asgn_list[-1]]

        return asgn_list

    mol = parse_sdf_file(fname)

    # Extract all atoms and bonds; order according to the unique atom indx
    # atom['indx'] start from 1 and can have missing values, e.g. [1, 2, 5, 6, 8]
    # Ideally, indx should be related to IUPAC assignements
    atoms = sorted([x for key, x in mol.items() if 'atom' in key], key=lambda x: x['indx'])
    bonds = [x for key, x in mol.items() if 'bond' in key]

    bond_matrix = get_bond_matrix(atoms, bonds)

    # Add hydrogens, if missing
    valence = np.array([{'C':4, 'O':2, 'N':3, 'H':1}[x['symbol']] for x in atoms],
                       dtype=int)    # Maximum number of connections per atom
    boundHs = valence - bond_matrix.sum(axis=0)    # Number of H's for each atom
    if sum(boundHs) > 0:
        # There are hydrogens to add to the list of atoms
        new_atoms, new_bonds = [], []
        last_indx = atoms[-1]['indx']        # The largest index of the atoms
        for i, nH in enumerate(boundHs):
            base_atom = atoms[i]      # The backbone atom bound to the new H
            labile = (base_atom['symbol'] in ['O', 'N'])
            for _ in range(nH):
                # Add a proton and assign the next index to it
                last_indx += 1     # Indx of the new proton
                new_atoms.append({'indx': last_indx,
                                  'labile': labile,
                                  'symbol': 'H',
                                  'base_indx' : base_atom['indx'],          # The index of the backbone (base atom) to which the H is attached
                                  'label': 'H'+str(base_atom['indx'])})
                new_bonds.append({'atom_ids': [base_atom['indx'], last_indx],
                                  'type': 'Single'})
        atoms.extend(new_atoms)
        bonds.extend(new_bonds)

    # Check that all hydrogens have the 'labile' flag assigned
    for atom in atoms:
        if atom['symbol'] == 'H' and ('base_indx' not in atom.keys() or 'labile' not in atom.keys()):
            for bond in bonds:
                if atom['indx'] in bond['atom_ids']:
                    atom['base_indx'] = set(bond['atom_ids']).difference([atom['indx']]).pop()   # Index of the backbone atom
                    base_atom = atoms[find_by_attr(atoms, 'indx', atom['base_indx'])]   # The backbone atom to which the current H atom is connected
                    atom['labile'] = (base_atom['symbol'] in ['O', 'N'])
                    break

    #Update the bond matrix
    bond_matrix = get_bond_matrix(atoms, bonds)

    # Compute the connectivity matrices of various degrees d (1 - if two atoms are d bonds apart)
    conn_1_matrix = np.where(bond_matrix>0, 1, 0) \
                    * (1-np.eye(bond_matrix.shape[0], dtype=int) )
    conn_2_matrix = np.where(bond_matrix.dot(bond_matrix)>0, 1, 0) \
                    * (1-np.eye(bond_matrix.shape[0], dtype=int) ) * (1-conn_1_matrix)
    conn_3_matrix = np.where(bond_matrix.dot(bond_matrix).dot(bond_matrix)>0, 1, 0) \
                    * (1-np.eye(bond_matrix.shape[0], dtype=int) ) * (1-conn_1_matrix) * (1-conn_2_matrix)
    conn_4_matrix = np.where(bond_matrix.dot(bond_matrix).dot(bond_matrix).dot(bond_matrix)>0, 1, 0) \
                    * (1-np.eye(bond_matrix.shape[0], dtype=int) ) * (1-conn_1_matrix) * (1-conn_2_matrix) * (1-conn_3_matrix)

    # Determine array indices of all non-labile (fixed) protons from mol file (labile protons are not considered by nmrdb.org)
    i_Hfixed = np.array([ i for i, atom in enumerate(atoms) if atom['symbol']=='H' and not atom['labile'] ])
    atoms_H_fixed = [atoms[i] for i in i_Hfixed]
    # Build a connectivity mtrix restricted to fixed protons only indicating the number of bonds between them
    conn_H_matrix = 2*conn_2_matrix[np.ix_(i_Hfixed, i_Hfixed)] + \
                    3*conn_3_matrix[np.ix_(i_Hfixed, i_Hfixed)]# + \
    #                 4*conn_4_matrix[np.ix_(i_Hfixed, i_Hfixed)]

    # ----- Read chemical shifts and J-couplings in the NMRDB format ------
    # Returns a list spins_H_nmrdb whose length and order corresponds to spins in conn_H_matrix
    if 'NMRDB-H' in mol.keys():
        spins_H_nmrdb = []
        chshH = []
        for chsh_asgn, row in enumerate(mol['NMRDB-H'].split('\n')[1:]):
            # Each row corresponds to a distinct chemical shift
            try:
                ids, chsh_val, _, _, jcpl_arr = row.split()
                jcpl_arr = [float(x) for x in jcpl_arr.split(',')]
            except ValueError:
                # If there are no J-couplings
                ids, chsh_val, _, _ = row.split()
                jcpl_arr = []
            chsh_val = float(chsh_val)
            chshH.append(parsSpec(dval=chsh_val, min=chsh_val-0.01, max=chsh_val+0.01))
            for indx in ids.split(','):
                spins_H_nmrdb.append({'indx':int(indx),          # Index in the nmrdb entry
                                      'chsh_asgn':chsh_asgn,   # Which chemical shift parameter is assigned to this spin
                                      'jcpl_arr':jcpl_arr})      # Array of related J-couplings
        # Sort the spins according to their indices from mol file (to match the order in conn_H_matrix)
        spins_H_nmrdb.sort(key=lambda x : x['indx'])
    #     # Set labels to nmrdb spins inferred from the backbone connectivity in .mol file
        for a, s in zip(atoms_H_fixed, spins_H_nmrdb):
    #         print(a, s)
            s['label'] = 'H'+str(a['base_indx'])

    else:
        print('No NMR information is available in the imported .mol file.\n{}'.format(fname))
        return None

    # TODO: Read C NMR assignments
    if 'NMRDB-C' in mol.keys():
        pass

    # Assign fixed H spins (only) to their spin systems.
    # Each entry in the list corresponds to a spin system, with the elements in the arrays --
    # to the positional indices of spins in the global connectivity matrix of fixed H's.
    spsy_asgn_list = assign_spsy(conn_H_matrix)

    # Determine multiplicities of the spin systems
    spsy_list = []
    jcplH, meqSpins, meqLinks = [], [], []     # All J coulings and links
    i_sp = 0         # Index to run over all magnetically inequivalent spins
    for spsy_asgn in spsy_asgn_list: #spsy_asgn_dict.keys():
        # Find positional indices of spins in the global connectivity matrix, conn_H_matrix
        # Sort H spins in each spin system according to their assigned chsh id's (not chsh values as those may not be unique!)
        i_in_CM = np.array(sorted(spsy_asgn, key=lambda x : spins_H_nmrdb[x]['chsh_asgn']), dtype=int)

        # Keep only magnetically inequivalent spins (that have different chsh_asgn within same spin system)
        # TODO: More advanced checks for aromatic rings
        chsh_asgn = [spins_H_nmrdb[i]['chsh_asgn'] for i in i_in_CM]   # Assignment of chemical shifts
        chsh_asgn, _unique_indx, spin_mult = np.unique(chsh_asgn, return_index=True, return_counts=True)
        i_in_CM = i_in_CM[_unique_indx]

        # Connectivity (sub-)matrix in the right order
        conn_matrix = conn_H_matrix[np.ix_(i_in_CM, i_in_CM)]

        # Create a list of arrays of J-couplings
        jcpl_arrs = [spins_H_nmrdb[i]['jcpl_arr'] for i in i_in_CM]

        # Set (unique) labels
        spin_labels = ['H'+str(atoms_H_fixed[i]['base_indx']) for i in i_in_CM]
        counts = {k: v if v>1 else 0 for k, v in zip(*np.unique(spin_labels, return_counts=True))}     # Number of occurences of each label
        for i, lbl in enumerate(spin_labels):
            spin_labels[i] = lbl+{1:'a', 2:'b', 3:'c', 0:''}[counts[lbl]]
            counts[lbl] = max(0, counts[lbl]-1)

        # Create new spin system record
        new_spsy = {'i_in_CM': i_in_CM,
                    'spin_labels': spin_labels,
                    'conn_matrix': conn_matrix,
                    'chsh_asgn': chsh_asgn,
                    'jcpl_arrs': jcpl_arrs,
                    'spin_mult': spin_mult,     # Multiplicities of each spin
                    'spsy_mult': 1           # Multiplicity of the entire spin system
                   }

        # Check if an equivalent system is already in the list
        added = False
        for existing_spsy in spsy_list:
            if np.array_equal(existing_spsy['conn_matrix'], new_spsy['conn_matrix']) \
                and np.array_equal(existing_spsy['chsh_asgn'], new_spsy['chsh_asgn']):
                existing_spsy['spsy_mult'] += 1
                added = True
                break

        # If it is a new system, add it to the list and add spins to meqSpins
        if not added:
            # Assign chemical shift parameters
            spin_indx_glob = []      # Global (reordered) indices of spins in the meqSpins array
            # Loop over spins in the new spin system
            for indxChsh, nspin, label in zip(chsh_asgn, spin_mult, spin_labels):
                meqSpins.append(spinVert(indxChsh=indxChsh, nspin=nspin))
                chshH[indxChsh] = chshH[indxChsh]._replace(label=label)   # Set a label to the chemical shift parameter
                spin_indx_glob.append(i_sp)
                i_sp += 1
            new_spsy['spin_indx_glob'] = spin_indx_glob

            # Assign J-coupling parameters - use common values in arrays for both spins in each pair
            for i, j in zip( *np.nonzero(np.triu(conn_matrix)) ):
                # Loop over unique pairs of coupled spins
                if conn_matrix[i,j] == 2:
                    # The protons are bonded to the same carbon, no info in nmrdb in this case; use default value
                    jcpl_val = -12.0
                elif conn_matrix[i,j] == 3:
                    # Use a common value between the spins
                    _arr1, _arr2 = jcpl_arrs[i], jcpl_arrs[j]
                    common = set(_arr1).intersection(_arr2)
                    if len(common) > 0:
                        jcpl_val = max(common)
                    else: jcpl_val = 7.0   # If no common - use default


                jcplH.append(parsSpec(dval=jcpl_val, min=jcpl_val-0.5, max=jcpl_val+0.5,
                                      label='-'.join(sorted( [spin_labels[i], spin_labels[j]] )) ))
                i_jj = len(jcplH) - 1     # Index of the new jcpl
                meqPair = sorted([spin_indx_glob[i], spin_indx_glob[j]])   # Indices of coupled spins
                meqLinks.append( spinEdge(indxJcpl=i_jj, indxVert=meqPair) )

            # Add spin system to the list
            spsy_list.append(new_spsy)

    chemModel = { mol['name']:chemSpec(name=mol['name'], chshH=chshH, jcplH=jcplH, multH=None,
                         meqSpins=meqSpins, meqLinks=meqLinks, Mw=mol['MW'], nH_labile=0) }
    return chemModel

def readChemDB_JSON(fname='chemDB.json'):
    """Read a chemDB in JSON or .cdb format and convers it to a dictionary of chemSpec class objects.
    """
    root, ext = os.path.splitext(fname)

    if ext == '.json':

        try:
            # Try JSON first
            with open(fname, 'r') as fp:
                chemDB = json.load(fp)
        except FileNotFoundError:
            return dict()

        for k, v in chemDB.items():                   # Convert 2D arays of ranges to the namedtuple representation
            if 'meqSpins' in v.keys():
                for i in range(len(v['meqSpins'])):
                    v['meqSpins'][i] = spinVert(v['meqSpins'][i][0], v['meqSpins'][i][1])
            if 'meqLinks' in v.keys():
                for i in range(len(v['meqLinks'])):
                    v['meqLinks'][i] = spinEdge(v['meqLinks'][i][0], sorted(v['meqLinks'][i][1]))
            for kk in ['spsyCombH', 'spsyCombC']:
                if kk in v.keys():
                    for i in range(len(v[kk])):
                        v[kk][i] = spsyComb(*v[kk][i])
            for kk in ['chshH', 'chshC', 'jcplH']:
                try:
                    v[kk] = array2parsSpec(v[kk])
                except KeyError: pass
            for kk in ['nSpinH', 'multH', 'multC']:      # Make sure that all single numbers are stored within arrays
                try:
                    if v[kk].__class__ is int: v[kk] = [v[kk]]
                except KeyError: pass
        chemDB = {k:chemSpec(**v) for k,v in chemDB.items()}    # Conver orderedDict to chemSpec namedtuple

    elif ext == '.cdb':
        with open(fname, 'rb') as fp:
            chemDB = dill.load(fp)

        # Check that what has been loaded is a correct database
        if isinstance(chemDB, dict):
            for key, val in chemDB.items():
                if not isinstance(val, chemSpec):
                    return dict()
        else: return dict()

    return chemDB

def readChemDB_XLSX(ws):
    """Read a chemDB from Excel worksheet
        Args:
            ws - openpyxl worksheet
        Returns:
            chemDB - dict of chemSpecs
    """

    def read_chem(ws, firstrow=1):
        """Read an entire chemSpec entry from an openpyxl worksheet starting at firstrow."""

        def read_row(ws, col1, col2, row):
            """Read a (sub-)row in an Excel worksheet.
                Args:
                    ws - openpyxl worksheet
                    col1, col2 - column literals
                    row - row index (numerical)
                Retruns:
                    A list of values in the columns. For example, if col1=B, col2=D, row=5,
                    will return the values in the range B5:D5.
            """
            cells = ws['{}{}'.format(col1, row):'{}{}'.format(col2, row)][0]
            cells = [cell.value for cell in cells]

            return cells

        # Read the general information
        name = ws.cell(firstrow, 1).value
        mW = ws.cell(firstrow+3, 2).value
        nH_labile = ws.cell(firstrow+4, 2).value

        # Read the 1H chemical shift parameters
        row, chshH = firstrow+3, []
        while True:
            cells = read_row(ws, 'E', 'L', row)
            if any(cells):
                _, label, dval, minval, maxval, distr, p1, p2 = cells
                if dval is not None:
                    if minval is None: minval=dval-0.01
                    if maxval is None: maxval=dval+0.01
                if label is None: label=''
                chshH.append(parsSpec(label=label, dval=dval, min=minval, max=maxval, distr=distr, p1=p1, p2=p2))
                row +=1
            else: break

        # Read the J coupling parameters
        row, jcplH = firstrow+3, []
        while True:
            cells = read_row(ws, 'N', 'U', row)
            if any(cells):
                _, label, dval, minval, maxval, distr, p1, p2 = cells
                if dval is not None:
                    if minval is None: minval=dval-0.5
                    if maxval is None: maxval=dval+0.5
                if label is None: label=''
                jcplH.append(parsSpec(label=label, dval=dval, min=minval, max=maxval, distr=distr, p1=p1, p2=p2))
                row +=1
            else: break

        # Read 13C chemical shifts parameters
        row, chshC, multC = firstrow+3, [], []
        while True:
            cells = read_row(ws, 'W', 'AE', row)
            if any(cells):
                _, label, nspin, dval, minval, maxval, distr, p1, p2 = cells
                if dval is not None:
                    if minval is None: minval=dval-1.0
                    if maxval is None: maxval=dval+1.0
                if label is None: label=''
                chshC.append(parsSpec(label=label, dval=dval, min=minval, max=maxval, distr=distr, p1=p1, p2=p2))
                multC.append(nspin)
                row +=1
            else: break

        # Read the 1H spins
        row, meqSpins, meqLinks = firstrow+3, [], []
        while True:
            cells = read_row(ws, 'AI', 'AJ', row)
            if any(cells):
                indxChsh, nspin = cells
                meqSpins.append(spinVert(indxChsh=indxChsh-1, nspin=nspin))
                row +=1
            else: break

        # Read the connectivity matrix (only the upper part)
        for i in range(len(meqSpins)):
            for j in range(i+1, len(meqSpins)):
                indxJcpl = ws.cell(row=firstrow+3+i, column=38+j).value
                if indxJcpl:
                    meqLinks.append( spinEdge(indxJcpl=indxJcpl-1, indxVert=[i,j]) )

        # Read the information about spin systems multiplicities
        multH = []
        for i in range(len(meqSpins)):
            cells = read_row(ws, 'AG', 'AH', firstrow+3+i)
            if all(cells): multH.append(cells[1])

        # Read the spin system groups
        spsyCombH = []
        if any(read_row(ws, 'A', 'C', row=firstrow+5)):
            row = firstrow+6
            while True:
                row += 1
                cells = read_row(ws, 'A', 'C', row)
                if any(cells):
                    indxSpsy = [int(i)-1 for i in cells[2][1:-1].split(',')]
                    spsyCombH.append(spsyComb(name=cells[0], intn=cells[1], indxSpsy=indxSpsy))
                else:break

        return chemSpec(name=name, mW=mW, nH_labile=nH_labile, chshH=chshH, chshC=chshC, \
                        jcplH=jcplH, multC=multC, multH=multH, spsyCombH=spsyCombH, meqSpins=meqSpins, meqLinks=meqLinks)

    # Detrermine initial rows for each chemical
    firstrows, in_entry = [], False
    for i, row in enumerate(ws.rows):
        valid_row = any([c.value for c in row])
        if valid_row and (not in_entry):
            firstrows.append(i+1)
            in_entry=True
        elif (not valid_row) and (i>firstrows[-1]+2):
            in_entry = False

    chemDB = {}
    for row in firstrows:
        new_chem = read_chem(ws, row)
        chemDB[new_chem.name] = new_chem

    return chemDB

def writeChemDB_JSON(chemDB, fname='chemDB_saved.json'):
    """Writes the chemDB in JSON format and stores it file."""
    print('Saving the chemDB database into file {:s}'.format(fname))
    root, ext = os.path.splitext(fname)

    if ext == '.json':
        # Save in JSON format
        chemDB = {k:v.asdict() for k,v in chemDB.items()}   # Convert namedtuples to dictionaries
        for k, v in chemDB.items():                   # Convert 2D arays of ranges to the namedtuple representation
            for kk in ['chshH', 'chshC', 'jcplH']:
                v[kk] = array2parsSpec(v[kk])
            for kk in ['multH', 'multC']:      # Make sure that all single numbers are stored within arrays
                if isinstance(v[kk], int):
                    v[kk] = [v[kk]]
                if v[kk] == [1]: v[kk] = []
            # Remove empty records
            chemDB[k] = {kk:vv for kk, vv in chemDB[k].items() if
                        (kk != 'Mw' and len(vv) > 0) or (kk == 'Mw' and vv is not None)}
        with open(fname, 'w') as fp:
            json.dump(chemDB, fp)

    elif ext == '.cdb':
        # Save in the dill format (.cdb)
        with open(fname, 'wb') as fp:
            dill.dump(chemDB, fp)

def writeChemLib_XLSX(chemLib, fname='chemDB_saved.xlsx', verbose=False):
    """Export chemLib into xlsx file using xlsxwriter."""

    workbook = xlsxwriter.Workbook(fname)
    fmt_center = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'text_wrap':True})
    fmt_cenrot = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'rotation': 90})

    def write_chem(worksheet, chemToSave, firstrow=0):
        """Write a chemSpec entry to Excel file.

            Args:
                worksheet - xlsxwriter Excel worksheet
                chemToSave - chemSpec object
                firstrow - index of the row to start writing from
        """

        # TODO: Reimplement this function with openpyxl
        firstcol = 0                 # First col for a chemical (zero-indexed)
        lastrow = firstrow           # The last written row for this chemical
        worksheet.merge_range(firstrow, 4, firstrow, 11, '1H chemical shifts', fmt_center)
        for i, (text, col_width) in enumerate(zip(['ID chsh', 'Label', 'Default, ppm', 'Min, ppm', 'Max, ppm', 'Distribution', 'p1', 'p2'],
                                                  [4, 5, 6, 6, 6, 8, 6, 6])):
            col = i+4     # Column index to write to
            worksheet.merge_range(firstrow+1, col, firstrow+2, col, text, fmt_center)
            worksheet.set_column(col, col, col_width)

        # Parameters of chemical shifts
        row, col = firstrow+3, firstcol+4
        for i, chsh in enumerate(chemToSave.chshH):
            worksheet.write_row(row, col, [i+1, chsh.label, chsh.dval, chsh.min, chsh.max, chsh.distr, chsh.p1, chsh.p2])
            row += 1
        lastrow = max(lastrow, row)

        # Parameters of J couplings
        worksheet.merge_range(firstrow, 13, firstrow, 20, '1H J-couplings', fmt_center)
        for i, (text, col_width) in enumerate(zip(['ID jcpl', 'Label', 'Default, Hz', 'Min, Hz', 'Max, Hz', 'Distribution', 'p1', 'p2'],
                                                  [4, 5, 6, 6, 6, 8, 6, 6])):
            col = i+13     # Column index to write to
            worksheet.merge_range(firstrow+1, col, firstrow+2, col, text, fmt_center)
            worksheet.set_column(col, col, col_width)

        row, col = firstrow+3, firstcol+13
        for i, jcpl in enumerate(chemToSave.jcplH):
            worksheet.write_row(row, col, [i+1, jcpl.label, jcpl.dval, jcpl.min, jcpl.max, jcpl.distr, jcpl.p1, jcpl.p2])
            row += 1
        lastrow = max(lastrow, row)

        # Carbon Chemical shifts
        worksheet.merge_range(firstrow, 22, firstrow, 30, '13C spins', fmt_center)
        for i, (text, col_width) in enumerate(zip(['ID', 'Label', 'N spins', 'Default, ppm', 'Min, ppm', 'Max, ppm', 'Distribution', 'p1', 'p2'],
                                                  [3, 5, 3, 6, 6, 6, 8, 6, 6])):
            col = i+22     # Column index to write to
            worksheet.merge_range(firstrow+1, col, firstrow+2, col, text, fmt_center)
            worksheet.set_column(col, col, col_width)

        row, col = firstrow+3, firstcol+22
        for i, (chsh, mult) in enumerate(zip(chemToSave.chshC, chemToSave.multC)):
            worksheet.write_row(row, col, [i+1, chsh.label, mult, chsh.dval, chsh.min, chsh.max, chsh.distr, chsh.p1, chsh.p2])
            row += 1
        lastrow = max(lastrow, row)

        # 1H spins and connections
        n_spinH = len(chemToSave.asdict()['meqSpins'])           # Total umber of spins
        worksheet.merge_range(firstrow, 32, firstrow, 36, '1H spins', fmt_center)
        for i, (text, col_width) in enumerate(zip(['ID spsy', 'Mult. spsy', 'ID chsh', 'N spins', 'ID spin'], [2.5, 2.5, 4, 3, 3])):
            col = i+32     # Column index to write to
            worksheet.merge_range(firstrow+1, col, firstrow+2, col, text, fmt_center)
            worksheet.set_column(col, col, col_width)
        worksheet.merge_range(firstrow, 37, firstrow+1, 36+max(5, n_spinH), 'IDs of J-couplings', fmt_center)
        for i in range(n_spinH):
            col = i+37     # Column index to write to
            worksheet.write(firstrow+2, col, i+1, fmt_center)
            worksheet.set_column(col, col, width=2.5)
        # Write the 1H meq spins
        row, col = firstrow+3, firstcol+32
        for i, spin in enumerate(chemToSave.meqSpins):
            indxSpsy = chemToSave._spsyAsgnSpins[i] # Index of the spin system to which the spin is assigned
            worksheet.write_row(row, col, [indxSpsy+1, chemToSave.multH[indxSpsy], spin.indxChsh+1, spin.nspin, i+1])
            row += 1
        lastrow = max(lastrow, row)
        # Merge rows for each spin system
        row = firstrow+3
        for indxSpsy, mult in enumerate(chemToSave.multH):
            nspin = len([True for i in chemToSave._spsyAsgnSpins if i == indxSpsy])    # Number of spins in the spin system
            worksheet.merge_range(row, firstcol+32, row+nspin-1, firstcol+32, indxSpsy+1, fmt_center)
            worksheet.merge_range(row, firstcol+33, row+nspin-1, firstcol+33, mult, fmt_center)
            row += nspin
        # Write the 1H meq links
        row, col = firstrow+3, firstcol+37    # Top left corner of the connectivity matrix
        for link in chemToSave.meqLinks:
            worksheet.write(row+min(link.indxVert), col+max(link.indxVert), link.indxJcpl+1)

        # Set other column widths (intial columns and separators)
        worksheet.merge_range(firstrow, firstcol, firstrow+2, firstcol+2, chemToSave.name, fmt_center)
        for col, width in zip([0,1,2,3, 12,21,31], [8, 4, 8, 2, 1, 1, 1]):
            worksheet.set_column(col, col, width)
        worksheet.write(firstrow+3, 0, 'Mw, g/mol')
        worksheet.merge_range(firstrow+3, 1, firstrow+3, 2, chemToSave.Mw)
        worksheet.write(firstrow+4, 0, 'Number of labile protons')
        worksheet.merge_range(firstrow+4, 1, firstrow+4, 2, chemToSave.nH_labile)
        lastrow = max(lastrow, firstrow+5)

        # Write spin system combinations
        if len(chemToSave.spsyCombH) > 0:
            worksheet.merge_range(firstrow+5, 0, firstrow+5, 2, '1H spin system groups', fmt_center)
            worksheet.write_row(firstrow+6, 0, ['Name', 'Intensity', 'Spsy IDs'])
            row=firstrow+7
            for comb in chemToSave.spsyCombH:
                worksheet.write_row(row, 0, [comb.name, comb.intn, repr([i+1 for i in comb.indxSpsy])])
                row += 1
            lastrow = max(lastrow, row)

        return lastrow

    for db_name, db in chemLib.items():
        worksheet = workbook.add_worksheet(db_name)
        firstrow = 0
        chemNames = sorted(list(db.keys()))
        for name in chemNames:
            chemToSave = db[name]
            if verbose:
                print('Saving {}/{}'.format(db_name, name))
            lastrow = write_chem(worksheet, chemToSave, firstrow)
            firstrow = lastrow+2

    workbook.close()

def loadChemLibrary(chemLib=None, dirpath=None):
    """Loads the chemical library (a dictionary of chemDB dictionaries)."""

    # Define a DB for common chemicals
    # chemLib is a dictionary of dictionaries; the first level used for grouping
    # and the second level contains chemSpec entries
    if chemLib is None:
        chemLib = {'Built-in models' :
                    {'Water' : chemSpec(name='Water',
                                       chshH=[parsSpec(min=3.75, max=5.75, label='H_water')],
                                       meqSpins=[spinVert(0, 2)],
                                       multH=[1]),
                    'TMS' : chemSpec(name='TMS',
                                       chshH=[parsSpec(min=-0.25, max=0.25)],
                                       chshC=[parsSpec(min=-0.25, max=0.25)],
                                       meqSpins=[spinVert(0, 3)],
                                       multH=[4],
                                       multC=[4]),
                    'TMSP' : chemSpec(name='TMSP',
                                       chshH=[parsSpec(min=-0.25, max=0.25)],
                                       chshC=[parsSpec(min=-0.25, max=0.25)],
                                       meqSpins=[spinVert(0, 3)],
                                       multH=[3],
                                       multC=[3]),
                    'Ethanol' : chemSpec(name='Ethanol',
                                       chshH=[parsSpec(min=0.5, max=1.5, label='H1'), parsSpec(min=3.0, max=4.0, label='H2')],
                                       chshC=[parsSpec(min=14.9, max=15.1, label='C1'), parsSpec(min=57.9, max=58.1, label='C2')],
                                       jcplH=[parsSpec(min=6.0, max=8.0, label='H1-H2',dval=7.0402)],
                                       meqSpins=[spinVert(0, 3), spinVert(1, 2)],
                                       meqLinks=[spinEdge(0, (0,1))],
                                       multH=[1],
                                       multC=[1, 1]),
                    'Chloroform' : chemSpec(name='Chloroform',
                                       chshH=[parsSpec(min=7.2, max=7.3, label='H_Chfm')],
                                       meqSpins=[spinVert(0, 1)],
                                       multH=[1]),
                    'Singlet' : chemSpec(name='Singlet',
                                       chshH=[parsSpec(min=-0.5, max=0.5, label='Singlet')],
                                       meqSpins=[spinVert(0, 1)],
                                       multH=[1]),
                     }
                   }

    return chemLib

chemLib = loadChemLibrary()            # Load the chemical library
