import numpy as np
import dill
import config
from chemTree import minmaxTuple
from chemTree import *
from leastsquares import *
import tabulate
import pywt
#import xlsxwriter
from dataio import read_any_file

# Functions for generating FIDs and optimization
import scipy as sp
#import scipy.sparse
import scipy.linalg
import scipy.signal
from collections import OrderedDict, MutableMapping
from proc_bl import _ps_acme_score, _ps_peak_minima_score, baseline_corrector, med

####                  --------- Utility functions ---------                #####

# Loading and saving

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
                       'autoPick': False, 'ax1Limits': None, 'ax2Limits': None}
    GUIsettings.update( {'_config': config.as_dict()} )
    dataPack = wsp.pack()
    with open(filename, 'wb') as fp:
        dill.dump([dataPack, GUIsettings], fp)

def addDatumFromFile(path, dest):
    """Add new Datum entries specified by the path to the series object.

    Args:
        path: str
            Location of the spectrum file.
        dest: Series, Datum, or Workspace
            A series to which the new spectra should be added. If a Datum is
            passed, the spectrum will be added to its parent Series. If a Workspace
            is passed, its first Series will be used. If the parameters of the
            Series are not consistent with the imported spectrum (e.g. field
            strength c0), will try to find first suitable Series in the Workspace,
            or create a new one.

    Returns:
        dat: Datum
            The newly added Datum. If several spectra were added (e.g. if the
            file contained a series of spectra, all of them will be added but
            only the last one will be returned)

    """

    # Read the data
    t, yT, dic = read_any_file(path)
    c0, f0, dt, name = dic['c0'], dic['f0'], dic['dt'], dic['name']

    # Determine to which Series it should be added
    wsp = dest.wsp      # The Workspace
    ser_ID, _ = dest.selfID()

    if len(wsp.series) == 0:
        wsp.addSeries()

    if ser_ID is None: ser_ID = -1
    ser = wsp.series[ser_ID]

    # TODO: Check if new c0/f0 are the same as the old ones when loading the rest of the data
    # Save the acquisition parameters; these should be the same for all spectra in the series (by convention)
    if ser.c0 is None:
        ser.c0 = c0
        ser.f0 = f0
        ser.t = t
        ser.fullReset()
    elif np.abs(ser.c0 - c0) > 1e-3:
        # Create a new series and put the data into it
        print('The aquisition parameters do not match the current values. Creating a new Series...')
        ser = wsp.addSeries()
        ser.c0 = c0
        ser.f0 = f0
        ser.t = t
        ser.fullReset()

    # Add the data to the current series
    for i in range(yT.shape[1]):
        dat = ser.addDatum(yT[:,i].reshape(-1,1),
                    name = name+str(i+1) if yT.shape[1] > 1 else name, extra=dic)

    return dat


# MCMC samples

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

def flat_window(ntot, npad=0, w_type='gaussian'):
    """Returns a flat-top window with padding on each side. Each end of the window is set equal to 0 if npad > 0."""
    if 2*npad > ntot:
        raise RuntimeError('The length of padding should be no more than a hilf-size of the window.')

    # Define the padding
    if w_type == 'gaussian':
        pT = scipy.signal.gaussian(npad*2, std=npad/5)[:npad].reshape(-1,1)
        pT -= min(pT)
        pT /= max(pT)
    elif w_type == 'zeros':
        pT = np.zeros((npad, 1))
    elif w_type == 'linear':
        pT = np.linspace(0, 1, npad).reshape(-1,1)

    #
    wT = np.vstack([pT, np.ones((ntot-2*npad, 1)), pT[::-1]])
    return wT

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
    """Given a list of freqSpec tuples, divides the frequency range -inf to +inf
        into lists of disjoint intervals: inRange and outRange by merging
        overlapping optimization ranges.
    """
    if inRange:
        inRange = merge_intervals(inRange)      # Merged and sorted list of freqRanges
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

def merge_intervals(intervals, merge_disjoint=False):
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
    # intervals = [minmaxTuple(min(tup), max(tup)) for tup in intervals]
    sorted_by_lower_bound = sorted(intervals, key=lambda tup: tup.min)
    merged = []

    # If return only single big interval
    if merge_disjoint:
        sorted_by_higher_bound = sorted(intervals, key=lambda tup: tup.max)
        merged.append(minmaxTuple(min=sorted_by_lower_bound[0].min, max=sorted_by_higher_bound[-1].max))
        return merged

    # If return several disjoint intervals
    for higher in sorted_by_lower_bound:
        if not merged:
            merged.append(higher)
        else:
            lower = merged[-1]
            # test for intersection between lower and higher:
            # we know via sorting that lower[0] <= higher[0]
            if min(higher) <= lower.max:
                upper_bound = max(lower.max, higher.max)
                merged[-1] = minmaxTuple(min=lower.min, max=upper_bound)  # replace by merged interval
            else:
                merged.append(higher)
    return merged

def whitsm(y, lmda=5.0):
    """Whittaker smoother. See: https://gist.github.com/zmeri/3c43d3b98a00c02f81c2ab1aaacc3a49"""
    m = len(y)

    x = np.arange(1, m+1, 1)
    dx0 = (x[1:-1]-x[:-2]).ravel()
    dx1 = (x[2:]-x[1:-1]).ravel()
    D = sp.sparse.diags([2/dx0/(dx0+dx1), np.append(-2/dx0/dx1, 0.0), 2/dx1/(dx0+dx1)], np.array([0, 1, 2]), shape=(m-2, m)) * np.mean(np.abs(dx0)**2)

    E = sp.sparse.identity(m)
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
    xTc = np.fft.ifft(np.fft.ifftshift(xF_h, axes=0), axis=0)[:int(xF_h.size/2), :]
    xFc = np.fft.fftshift(np.fft.fft(xTc, axis=0), axes=0)
    return xTc, xFc

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

def next_pow_of_2(x):
    # Returns the smallest power of 2 greater than x
    return 0 if x == 0 else 2**(int(x) - 1).bit_length()

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

# Phasing and baseline correction
def flims(dt, nf):
    """Computes the values (in Hz) of the first and last sample in the spectrum
    with nf samples corresponding to a signal with sampling time dt.
    See https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fftfreq.html"""
    if nf%2 == 0:
        return (-nf/(2*dt*nf), (nf/2-1)/(dt*nf))
    else:
        return (-(nf-1)/(2*dt*nf), (nf-1)/(2*nf*dt))

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

def automatic_ps(data, fn, p0=0.0, p1=0.0, fit_Ph1=True):
    """
    Automatic linear phase correction
    COPIED FROM nmrglue

    Parameters
    ----------
    data : ndarray
        Array of NMR data.
    fn : str or function
        Algorithm to use for phase scoring. Built in functions can be
        specified by one of the following strings: "acme", "peak_minima"
    p0 : float
        Initial zero order phase in degrees.
    p1 : float
        Initial first order phase in degrees.

    Returns
    -------
    ndata : ndarray
        Phased NMR data.

    """
    if not callable(fn):
        fn = {
            'peak_minima': _ps_peak_minima_score,
            'acme': _ps_acme_score,
        }[fn]

    if fit_Ph1:
        opt = [p0, p1]
        opt = scipy.optimize.fmin(fn, x0=opt, args=(data, ))

        p0, p1 = opt[0], opt[1]
    else:
        opt = [p0]
        opt = scipy.optimize.fmin(lambda x : fn((x, p1), data), x0=opt)
        p0 = opt[0]

    # phasedspc = ps(data, p0=opt[0], p1=opt[1])

    return p0, p1

def ph_cost(yF, xF, ph0=0.0, ph1=0.0, f=None, mw=2*512, cfun='LS'):
    """Calculate the cost function for phase abd baseline adjustment.

    See "Improving the Accuracy of Model-based Quantitative NMR"
    by Y.Matviychuk et.al. for more detail.

    Args:
        yF : np.array
            Measured (unphased) spectrum
        xF : np.array
            Fitted model
        ph0, ph1 : float
            Zero- and first-order phasing terms
        mw : int
            Size of the median filter used to estimate the baseline.
        f : np.array
            Frequency scale in the fractions of angular frequency
            (i.e. f = [-1/2...(n-2)/2n] for even number of samples n and
            f = [-(n-1)/2n...(n-1)/2n] for odd n). If yF and xF contain non-contiguous
            frequency ranges, f should be specified explicitely.
        cfun : str, 'LS' or 'TV'
            Defines the type of the cost function: least-squares ('LS') or
            total variation ('TV')

    Returns:
        val : float
            Value of the cost function.
        yFph : np.array
            Phase-adjusted data.
        res : np.array
            Residual spectrum.
        bln : np.array
            Estimated baseline after phase-adjustment.

    """

    # Define the frequency scale if it's not given
    if f is None:
        f = np.fft.fftshift(np.fft.fftfreq(len(yF), 1))

    # Compute the phasing term
    #### ph = np.exp(-1j*2*np.pi * tau * f_Hz - 1j*theta ).reshape((-1,1))   # The phasing term
    ph = np.exp(-1j* ph1 * f - 1j*ph0 ).reshape((-1,1))   # The phasing term
    yFph = yF * ph

    # Find and denoise the residual spectrum
    rFdn = wden(yFph.real - xF.real, tptr='sqtwolog', scal='mln', wsize=25)         # Denoised residual

    # Remove the baseline with median filter
    res = med(rFdn.ravel(), mw).reshape(-1,1)               # Deoised residual with baseline removed
    bln = (rFdn - res).reshape(-1,1)

    # Compute the cost function
    if cfun == 'LS':
        val = np.linalg.norm(bln - np.mean(bln), 2)
    elif cfun == 'TV':
        val = np.linalg.norm(bln[1:] - bln[:-1], 1)

    #return np.linalg.norm(res - np.mean(res), 2), yFph, res, bln
    return val, yFph, res, bln

def baseline(yF, wd=20):
    """Estimate the baseline with the standard baseline correction algorithm."""

    # Apply standard baseline correction to the spectrum
    yFbl = baseline_corrector(yF.real.ravel(), wd=wd).reshape(-1,1)
    bF = (yF.reshape(-1,1) - yFbl).real

    return bF
