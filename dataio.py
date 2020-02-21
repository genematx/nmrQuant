import numpy as np
import os
import time

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
    if size[1] == 1:
        t = data[:size.prod()]
        data = (data[size.prod()::2] - 1j*data[size.prod()+1::2]).reshape(size[:2])
    else:
        data =  (data[::2] - 1j*data[1::2]).reshape(size[:2], order='F')

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
    elif os.path.isfile(path):
        dirName = os.path.dirname(os.path.abspath(path))
    else:
        raise FileNotFoundError(None, None, 'Invalid directory')

    # Look for the data.1d or data.2d file in the directory
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
