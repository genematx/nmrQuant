import numpy as np
import os
import time
from ssnake.specIO import loadFile, loadJEOLDelta

# Functions fro reading different file formats.
# Each function has standardized inputs and outputs.
# Input: either a directory or a data file
# Output: yT - FID array (possibly 2-dimensioanl for arrayed experiments)
 #        c0 - spectrometer frequency
 #        f0 - frequncy shift
 #        dt - dwell time

def read(dir, bin_file=None, pars_files=None):
    """
    Read Spinsolve files from a directory.

    Parameters
    ----------
    dir : str
        A dictionary (or any file in the directory) to read from.
    bin_file : str, optional
        Filename of binary (.1d) file in directory. None uses standard files.
    pars_file : str, optional
        List of filename(s) of .par parameter files in directory. None uses
        standard files.

    Returns
    -------
    dic : dict
        Dictionary of Spinsolve parameters.
    data : ndarray
        Array of NMR data.

    See Also
    --------
    read_pdata : Read Spinsolve processed files.
    read_lowmem : Low memory reading of Spinsolve files.
    write : Write Spinsolve files.

    """

    # If the dir is actually a file
    if os.path.isfile(dir):
        dir = os.path.dirname(dir)

    # Read parameter file(s) and update the dictionary
    dic = {}
    if pars_files is None: pars_files = ["acqu.par", "protocol.par"]
    for f in pars_files:
        if os.path.isfile(os.path.join(dir, f)):
            dic.update(read_pars(os.path.join(dir, f)))

    # read the binary file
    if bin_file is not None:
        fileName = os.path.join(dir, bin_file)
    else:
        fileName = os.path.join(dir, 'data.1d')
        # TODO: select 1d or 2d based on the data in the acqu and protocol

    if os.path.isfile(fileName):
        data_dic, data = read_binary(fileName)
        dic.update(data_dic)
    else:
        raise FileNotFoundError(None, None, fileName)


    # read the pulse program and add to the dictionary

    # determine shape and complexity for direct dim if needed

    return dic, data

# Spinsolve binary (fid/ser) reading and writing

def read_binary(filename, big=False):
    """
    Read Spinsolve binary data from file and return dic,data pair.


    Parameters
    ----------
    filename : str
        Filename of Spinsolve binary file.
    big : bool
        Endianness of binary file, True for big-endian, False for
        little-endian.

    Returns
    -------
    dic : dict
        Dictionary containing "FILE_SIZE" key and value.
    data : ndarray
        Array of raw NMR data.

    See Also
    --------
    read_binary_lowmem : Read binary file using minimal memory.

    """
    # open the file and get the data
    with open(filename, 'rb') as f:
        head = (f.read(3*4).decode('ascii'), f.read(1*4))                                   # Read 3+1 4-byte blocks
        size = np.frombuffer(f.read(4*4), dtype='int32')     # Read another 4 4-byte blocks; each of them is an integer
        data = np.frombuffer(f.read(), dtype='float32')

    # create dictionary
    dic = {"FILE_SIZE": os.stat(filename).st_size}

    # Reshape the data according to the read sizes of array dimensions
    if data.size == 3*size.prod() and size[1] == 1:
        # The dataset is saved in the format: [time points, real values, imaginary values], else [real values, imaginary values]
        data = data[size.prod():]
    # print(size)
    # print(data.shape, data.size)
    # if size[1] == 1:
    #     t = data[:size.prod()]
    #     print(t.shape)
    #     data = (data[size.prod()::2] - 1j*data[size.prod()+1::2])
    #     print(data.shape)
    #     print(size)
    #     # .reshape(size[:2])
    # else:
    data = (data[::2] - 1j*data[1::2]).reshape(size[:2], order='F')

    return dic, data

def read_pars(filename):
    """
    Read a Spinsolve parameters .par file into a dictionary.


    Parameters
    ----------
    filename : str
        Filename of a Spinsolve .par file.

    Returns
    -------
    dic : dict
        Dictionary of parameters in file.

    See Also
    --------
    write_pars : Write a Spinsolve .par file.

    """
    dic = {}  # create empty dictionary

    with open(filename, 'r') as f:
        while True:     # loop until end of file is found

            line = f.readline().rstrip()    # read a line
            if line == '':      # end of file found
                break

            else:
                line = line.split('=')
                key = line[0].strip()
                val = line[1].strip()
                if val[0] == "\"":
                    val = val.strip("\"")
                else:
                    try:
                        val = float(val)
                    except ValueError: pass

                dic[key] = val

    dic['timeSaved'] = time.asctime(time.gmtime(os.path.getmtime(filename)))

    return dic

def read_spinsolve(path):
    # Make sure the path is a directory
    if os.path.isdir(path):
        dirName = os.path.abspath(path)
        binFile = None
    elif os.path.isfile(path):
        dirName = os.path.dirname(os.path.abspath(path))
        binFile = os.path.basename(path)
    else:
        raise FileNotFoundError(None, None, 'Invalid directory')

    # Look for the data.1d or data.2d file in the directory
    if binFile is not None:
        dic, yT = read(dirName, bin_file=binFile)
    else:
        try:
            # dic, yT = ng.fileio.spinsolve.read(dirName, bin_file='data.1d')
            dic, yT = read(dirName, bin_file='data.1d')
        except FileNotFoundError:
            try:
                # dic, yT = ng.fileio.spinsolve.read(dirName, bin_file='data.2d')
                dic, yT = read(dirName, bin_file='data.2d')
            except FileNotFoundError:
                raise FileNotFoundError(None, None, 'Neither data.1d nor data.2d file is found in the directory')

    dt = dic['dwellTime'] * 1e-6

    try:
        c0 = dic['b1Freq']
    except KeyError:
        c0 = dic['b1Freq'+dic['nucleus']]     # e.g. b1Freq1H


    try:
        fcar = -dic['lowestFrequency']
    except KeyError:
        fcar = 0.0

    swh = 1 / dt
    f0 = swh/2-fcar      # Frequency shift in Hz

    # nt = dic['nrPnts']



    # t = np.linspace(start=0, stop=(nt-1)*dt, num=nt).reshape(-1,1)

    ## Subsample if the frequency range is too large
    #k = max(math.floor(swh/c0 / 12), 1)   # Sampling factor to make the sweep width 12 ppm
    #t = t[::k]
    #yT = yT[::k, :]

    # name = os.path.split(os.path.dirname(path))[1]    # Only the name of the containing directory

    return yT, c0, f0, dt, dic

def read_jeol(path):
    """Import data in Jeol Delta format."""
    spec = loadJEOLDelta(path)   # ssnake Spectrum object

    if spec.ndim() > 1:
        raise RuntimeError('Only 1D JEOL data is currently supported.')

    # Store the dictionary of parameters
    dic = spec.metaData
    dic.update({'name':spec.name})

    if spec.spec[0]:
        # Compute FID if the data is in the frequency domain (spectrum)
        spec.complexFourier(axis=0)
    n_gd = 0    # Group delay
    yT = spec.getData().data.reshape(-1,1)[n_gd:, :]
    c0 = spec.freq[0] * 1e-06
    f0 = (spec.freq[0] - spec.ref[0])
    sw = spec.sw[0]
    # sw = dic['x_sweep']
    dt = 1/sw

    return yT, c0, f0, dt, dic

def read_spinsolve_subfolders(rootPath, pathList=None):
    """Recursively open folders and returns paths to data.1d files, if found.

    Args:
        rootPath: str
            Location of the root directory to start the serach from.
        pathList: list
            List of individual file paths; will be appended.

    Returns:
        pathList

    """
    # TODO! Make this function more robust
    if pathList is None: pathList = []

    if os.path.isdir(rootPath):
        if (( 'PROTON' in os.path.basename(rootPath) or 'PRESAT' in os.path.basename(rootPath) ) and 'data.1d' in os.listdir(rootPath)) or \
           (( 'Enhanced' not in os.path.basename(rootPath) ) and 'data.1d' in os.listdir(rootPath)):
            pathList.append(os.path.join(rootPath, 'data.1d'))
        else:
            for file in os.listdir(rootPath):
                read_spinsolve_subfolders(os.path.join(rootPath, file), pathList)

    return pathList

def read_any_file(path):
    """Read any file format and return an FID dataset and a dictionary of parameters."""

    dic = {}

    if path.endswith('.pyfid'):
        with open(path, 'rb') as fp:
            data = [float(x.strip()) if i != 5 else x.strip() for i, x in enumerate(fp.readlines())]

        c0, f0, nt = data[0], data[1], int(data[4])     # Number of time points

        t = np.array(data[6:nt+6]).reshape(-1,1)
        yT = (np.array(data[nt+6:2*nt+6]) + 1j*np.array(data[-nt:])).reshape(-1,1)
        name = path[path.rfind('\\')+1:path.rfind('.')]
        dt = t[1]-t[0]

    # Read a JCAMP-DX file
    elif path.endswith('.dx') or path.endswith('.jdx'):

        dic, data = ng.jcampdx.read(path)
        #c0 = float(dic['$BF1'][0])
        #fcar = float(dic['$REFERENCEPOINT'][0])
        #swh = float(dic['$SW'][0]) * c0
        #nt = float(dic['$TD'][0])

        udic = ng.jcampdx.guess_udic(dic,data)[0]     # Dictionary of universal parameters
        print(data)

        c0 = float(udic['obs'])
        fcar = float(udic['car'])
        swh = float(udic['sw'])
        nt = float(udic['size'])
        f0 = swh/2-fcar      # Frequency shift in Hz
        dt = 1 / swh;         # Sampling period (dwell time)

        t = np.linspace(start=0, stop=(nt-1)*dt, num=nt).reshape(-1,1)
        yT = (np.array(data[0]) - 1j*np.array(data[1])).reshape(-1,1)

        # Subsample if the frequency range is too large
        k = max(math.floor(swh/c0 / 12), 1)   # Sampling factor to make the sweep width 12 ppm
        t = t[::k]
        yT = yT[::k, :]

        name = os.path.split(os.path.dirname(path))[1]

    # Read a Bruker FID file
    elif path[-3:] == 'fid':

        dic, data = ng.fileio.bruker.read(path[:-3])

        acqus = dic['acqus']
        ntgrp = int(round(acqus['GRPDLY']))    # Number of time samples of the Bruker filter response;
        swh = acqus['SW_h']     # Spectral width in Hz
        f0 = acqus['O1']        # Offset in Hz
        c0 = acqus['SFO1']      # Frequency of the local oscillator in MHz
        dt = 1 / swh         # Sampling period (dwell time)
        tau = acqus['DE'] * (1e-06)   # Ringdown time delay in sec

        yT = data[ntgrp:].reshape(-1, 1)
        # nt = min(16384, len(yT))
        nt = len(yT)
        t = np.linspace(start=0, stop=(nt-1)*dt, num=nt).reshape(-1,1)
        # yT = yT[:nt].reshape(-1, 1)

        name = os.path.split(os.path.dirname(path))[1]

    # Read a JEOL FID file
    elif path.endswith('.jdf'):
        yT, c0, f0, dt, dic = read_jeol(path)

        nt = yT.shape[0]
        t = np.linspace(start=0, stop=(nt-1)*dt, num=nt).reshape(-1,1)
        name = os.path.split(path)[1]    # File name
        # print(path)

    # Read a Spinsolve data.1d file
    elif path.endswith('.1d') or path.endswith('.2d'):

        yT, c0, f0, dt, dic = read_spinsolve(path)

        nt = yT.shape[0]
        t = np.linspace(start=0, stop=(nt-1)*dt, num=nt).reshape(-1,1)
        name = os.path.split(os.path.dirname(path))[1]    # Only the name of the containing directory

    # Read an Mnova corrected FID file
    elif path.endswith('.txt'):
        with open(path, 'rb') as fp:
            # Read the file header line by line
            for line in fp:
                pair = line.decode().strip().split('=')
                if 'DataPoints' in pair[0]:
                    break           # The next line will be the first data point -- stop reading the header
                elif 'Size' in pair[0]:
                    nt = int(pair[1])
                elif 'SpectrometerFrequency' in pair[0]:
                    c0 = float(pair[1])
                elif 'Hz' in pair[0]:
                    f0 = -float(pair[1])
                elif 'SpectralWidth' in pair[0]:
                    dt = 1 / float(pair[1])

            # Read the remainder of the file into a np array
            data = np.fromfile(fp, sep='\t')

        # Form the arrays
        t = np.linspace(0, dt*(nt-1), nt).reshape(-1,1)
        yT = (data[::2] - 1j*data[1::2]).reshape(-1,1)

        name = path[path.rfind('\\')+1:path.rfind('.')]

    dic.update({'name':name, 'c0':c0, 'f0':f0, 'dt':dt})

    if 'startTime' in dic.keys():
        # TODO: Try dateutil to automatically parse dates in different formats
        timeParsed = time.strptime(dic['startTime'].split('.')[0], '%Y-%m-%dT%H:%M:%S')         # Convert the time format from e.g. "2021-02-14T13:21:33.653" to '%Y%m%d-%H%M%S'
        timeString = time.strftime('%Y%m%d-%H%M%S', timeParsed)
        dic.update({'acqu_time' : timeString})

    return t, yT, dic

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
