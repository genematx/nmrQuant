
QD_AggregatePeaksThreshold = 0.0            # Maximum distance in Hz between two peaks for them to be aggregated in the QD simulations
QD_RerunQDchshThreshold = 0.0               # Minimum difference in chemical shift (in Hz) w.r.t. to the previously used values to rerun QD simulations (all peaks will be simply moved othervise)

OPTIM_maxBasinhoppingSteps = 5             # Maximum number of basinhopping steps of the optimization algorithm
OPTIM_niterSuccess = 5
OPTIM_nvalLinearSearch = 25                # Number of values in the linear search
OPTIM_convergenceEps = 1e-06               # Convergence accuracy
OPTIM_method = 'L-BFGS-B'
OPTIM_startFrom = 'current'                # Initial values to start optimization in each file {'current', 'previous', 'default'}
OPTIM_copyFromPRESAT = False

#SAMPL_varEstimator = 'robust'              # Defines the method for estimation of the variances of intensities ['usual', 'robust', 'liberal', 'tight']
SAMPL_robustLS = False                      # Compute the robust LS estimator for variances
SAMPL_funcType = 'LS'                       # Defines the type of the cost function.
                                            # 'LS' - least Squares
                                            # 'TLS' - total Least Squares
                                            # 'L1' - L1 norm minimization
                                            # 'TV' - total variation
SAMPL_numberField = 'Re'                    # 'Re', 'ReIm', or 'Cx'

MODEL_ShapeKernelSize = 13                  # Size of the custom lineshape correction kernel (must be odd)

DISPL_ShiftToReference = True               # Display the chemical shift scale on the spectrum shifted accordingly to the global chsh parameter of the entire mixture

colrseq = [(0.1216,    0.4706,    0.7059),\
    (0.8902,    0.1020,    0.1098),\
    (0.2000,    0.6275,    0.1725),\
    (1.0000,    0.4980,         0),\
    (0.4157,    0.2392,    0.6039),\
    (0.6941,    0.3490,    0.1569),\
    (0.6510,    0.8078,    0.8902),\
    (0.6980,    0.8745,    0.5412),\
    (0.9843,    0.6039,    0.6000),\
    (0.9922,    0.7490,    0.4353),\
    (0.7922,    0.6980,    0.8392),\
    (1.0000,    0.0000,    0.6000)]*3      # Sequence of colors to plot the results

colr_freqBlocks = [255, 245, 175, 25]
colr_freqBlocks_inactive = [255, 245, 175, 3]

def as_dict():
    return {'QD_AggregatePeaksThreshold' : QD_AggregatePeaksThreshold,
            'QD_RerunQDchshThreshold' : QD_RerunQDchshThreshold,

            'OPTIM_maxBasinhoppingSteps' : OPTIM_maxBasinhoppingSteps,
            'OPTIM_niterSuccess' : OPTIM_niterSuccess,
            'OPTIM_nvalLinearSearch' : OPTIM_nvalLinearSearch,
            'OPTIM_convergenceEps' : OPTIM_convergenceEps,
            'OPTIM_method' : OPTIM_method,
            'OPTIM_startFrom' : OPTIM_startFrom,
            'OPTIM_copyFromPRESAT' : OPTIM_copyFromPRESAT,

            'SAMPL_robustLS' : SAMPL_robustLS,
            'SAMPL_funcType' : SAMPL_funcType,
            'SAMPL_numberField' : SAMPL_numberField,

            'MODEL_ShapeKernelSize' : MODEL_ShapeKernelSize,

            'DISPL_ShiftToReference' : DISPL_ShiftToReference}

def from_dict(self, D):
    for key, val in D.items():
        if key == 'QD_AggregatePeaksThreshold':
            self.QD_AggregatePeaksThreshold = val
        elif key == 'QD_RerunQDchshThreshold':
            self.QD_RerunQDchshThreshold = val
        elif key == 'OPTIM_maxBasinhoppingSteps':
            self.OPTIM_maxBasinhoppingSteps = val
        elif key == 'OPTIM_niterSuccess':
            self.OPTIM_niterSuccess = val
        elif key == 'OPTIM_nvalLinearSearch':
            self.OPTIM_nvalLinearSearch = val
        elif key == 'OPTIM_convergenceEps':
            self.OPTIM_convergenceEps = val
        elif key == 'OPTIM_method':
            self.OPTIM_method = val
        elif key == 'OPTIM_startFrom':
            self.OPTIM_startFrom = val
        elif key == 'OPTIM_copyFromPRESAT':
            self.OPTIM_copyFromPRESAT = val
        elif key == 'SAMPL_robustLS':
            self.SAMPL_robustLS = val
        elif key == 'SAMPL_funcType':
            self.SAMPL_funcType = val
        elif key == 'SAMPL_numberField':
            self.SAMPL_numberField = val
        elif key == 'MODEL_ShapeKernelSize':
            self.MODEL_ShapeKernelSize = val
        elif key == 'DISPL_ShiftToReference':
            self.DISPL_ShiftToReference = val


def reset():
    pass
