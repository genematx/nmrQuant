import numpy as np

def ls(Z, y, m0=None, S0=None, Gy=None, lockedPhase=False, indxPositive=None, robust=True):
    """Solves a phased-constrained complex-valued least-squares problem, y=Zx for x; nb - number of baseline terms (columns in the end of Z).
    If theta=None - the phase will be determined from the data. Gy - covariance matrix of noise (or the diagonal vector of that matrix)."""
    n, k = Z.shape      # Number of dimensions
    if lockedPhase and indxPositive is None:
        indxPositive = list(range(k))
    ampl = np.zeros((k,1))

    # 2. Set up the (Gaussian) priors
    m0 = m0 if m0 is not None else np.zeros((k, 1))
    S0 = S0 if S0 is not None else 1e+42 * np.identity(k)

    # Find which dimensions have priors with infinite or zero variance and invert the covariance matrix
    indx_inf = np.where(np.diag(S0) == np.inf)[0]        # Indices of components with infinite-variance (non-informative) intensities
    indx_fixed = np.where(np.diag(S0) == 0)[0]           # Indices of components with fixed (zero-variance) intensities
    indx_variable = np.where(np.diag(S0) != 0)[0]        # Indices of components with variable intensities
    indx_zero = []                                       # Indices of components with zero intensities
    iS0 = np.zeros((k, k))
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
    if np.linalg.matrix_rank(ZZ) < k:
        ZZ += 0.000001*np.identity(k)

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
        if np.linalg.matrix_rank(ZrZ) < k:
            ZrZ = ZrZ + 0.000001*np.identity(k)
        Sr = Sc.dot( (iS0 + ZrZ).real ).dot(Sc)
    else: Sr = None

    Q = yG.dot(y) + np.dot(np.dot(m0[indx_variable].conj().T, iS0[indx_variable[:,None], indx_variable]), m0[indx_variable]) - np.dot(np.dot(mc[indx_variable].conj().T, iSc[indx_variable[:,None], indx_variable]), mc[indx_variable])
    Q = max(np.asscalar(Q.real), 0.0)
    return mc, Sc, Sr, Q

def tls0(Z, y):
    # Classic total least squares algorithm based on SVD
    n, k = Z.shape
    d = y.shape[1]
    C = np.hstack([Z, y])
    u, s, vh = np.linalg.svd(C, full_matrices=False)
    Chat = (u[:, :k]*s[:k]).dot(vh[:k,:])
    v = vh.conj().T
    if not np.isclose(v[-1,-1], 0.0):
        b = (-v[:k, -d:]/v[-d:,-d:]).reshape(-1,1)

    ey = y - Z.dot(b)
    logdetS0, Q = 0.0, np.linalg.norm(ey)**2

    return b, Chat, logdetS0, Q

def itls(Z, y, niter=25):
    # Iterative TLS. See Schaffrin B, Lee IP, Felus Y, Choi YS (2006) "Total least-squares for geodetic straight-line and plane adjustment"
    nyu = 0
    ZZ = Z.conj().T.dot(Z)
    Zy = Z.conj().T.dot(y.reshape(-1,1))
    b = np.linalg.inv(ZZ).dot(Zy)
    for _ in range(niter):
        d = y - Z.dot(b)
        nyu = d.T.dot(d) / (1 + b.T.dot(b))
        b = np.linalg.inv(ZZ - nyu*np.eye(len(b))).dot(Zy)
    return b

def gtls(Z, y, Wl=None, Wr=None):
    # Generalized total least squares algorithm
    # Wl - left weighting matrix; Wr - right weighting matrix
    n, k = Z.shape
    d = y.shape[1]
    C = np.hstack([Z, y])
    if Wl is not None:
        C = scipy.linalg.sqrtm(Wl).dot(C)
    if Wr is not None:
        C = C.dot(scipy.linalg.sqrtm(Wr))
    u, s, vh = np.linalg.svd(C, full_matrices=False)
    Cm_tls = (u[:, :k]*s[:k]).dot(vh[:k, :])
    C_gtls = np.linalg.inv(scipy.linalg.sqrtm(Wl)).dot(Cm_tls).dot(np.linalg.inv(scipy.linalg.sqrtm(Wr)))
    u, s, vh = np.linalg.svd(C_gtls, full_matrices=False)
    v1, v2 = np.split(vh.conj().T, [k], axis=1)
    #v1 = np.linalg.inv(scipy.linalg.sqrtm(Wr)).dot(v1)
    #v2 = scipy.linalg.sqrtm(Wr).dot(v2)
    Chat = (u[:, :k]*s[:k]).dot(vh[:k,:])
    if not np.linalg.matrix_rank(vh[-d:,-d:]) < d:
        b = -v2[:k, :].dot(np.linalg.inv(v2[-d:,:]))
        #b2 = (v1[-d:, :].dot(np.linalg.inv(v1[:n, :]))).conj().T
    return b.reshape(-1,1), Chat

def ktls(Z, y, Q0=None, Qx=None, Qy=None, lamb=0.5, niter=15):
    # Weighted TLS problem with Kroneker product. "On weighted total least-squares adjustment for linear regression", Burkhard Schaffrin, Andreas Wieser
    # TODO: Make it faster by checking if the covariance matrices are diagonal and multiplying accordingly
    nyu = 0.0
    n, k = Z.shape
    y = y.reshape(-1,1)
    if Q0 is None:
        Q0 = np.eye(k)
    elif Q0.size == k:
        Q0 = np.diag(Q0.ravel())
    if Qx is None:
        Qx = np.eye(n)
    elif Qx.size == n:
        Qx = np.diag(Qx.ravel())
    if Qy is None:
        Qy = np.eye(n)
    elif Qy.size == n:
        Qy = np.diag(Qy.ravel())
    Q0, Qy = lamb*Q0, (1-lamb)*Qy

    ZZ = Z.conj().T.dot(Z)
    Zy = Z.conj().T.dot(y.reshape(-1,1))
    #theta = np.asscalar( 0.5*np.angle(np.dot(Zy.T, np.dot(np.linalg.inv(ZZ.real), Zy))) )
    b = np.linalg.inv(ZZ).dot(Zy)

    M = np.linalg.inv(Qy)
    b = np.linalg.inv(Z.T.dot(M.dot(Z)) - nyu*Q0).dot( Z.T.dot(M.dot(y)) )

    M = np.linalg.inv(Qy + b.T.dot(Q0.dot(b)) * Qx)
    b = np.linalg.inv(Z.T.dot(M.dot(Z)) - nyu*Q0).dot( Z.T.dot(M.dot(y)) )

    for i in range(niter):
        M = np.linalg.inv(Qy + b.T.dot(Q0.dot(b)) * Qx)
        L = M.dot(y-Z.dot(b))
        nyu = L.T.dot(Qx.dot(L))
        b = np.linalg.inv(Z.T.dot(M.dot(Z)) - nyu*Q0).dot( Z.T.dot(M.dot(y)) )
        #mc[:na] = np.maximum(mc[:na], 0.0) if mc[:na].sum() > 0 else np.minimum(mc[:na], 0.0)     # Discard negative values in mc but keep the sign for baseline components

    return b

def wtls(Z, y, Gz=None, Gy=None, gamma=0.5, niter=15, tol=1e-06):
    # Weighted TLS problem. "Weighted total least squares formulated by standard least squares theory", A. Amiri-Simkooei, S. Jazaeri
    # TODO: Full matrix case does not work properly!!!
    # TODO: Write a separate function to compute logdetS0 and Q along with inverses of matrices Gz and Gy
    n, k = Z.shape
    y = y.reshape(-1,1)
    if Gz is None:
        Gz = np.ones((n, k))        #Gz = np.eye(n*k)
    if Gy is None:
        Gy = np.ones((n, 1))        #Gy = np.eye(n)
    Gz, Gy = gamma*Gz, (1-gamma)*Gy
    C_hat = np.hstack([Z, y])     # Initialize C_hat

    # Invert the covariance matrix Gz
    if Gz.shape == (n*k, n*k):
        # Full matrix Qz
        iGz = np.linalg.inv(Gz)
    elif Gz.shape == (n, n, k):
        # Block-diagonal matrix Gz
        iGz = np.dstack([np.linalg.inv(Gz[..., i]) for i in range(k)])
    elif Gz.size == n*k:
        # Diagonal matrix Qz, possibly needs to be reshaped
        Gz = Gz.reshape(n, k)
        iGz = 1 / Gz

    # Invert the covariance matrice Qy and intialize b
    if Gy.shape == (n, n):
        # Full matrix Gy
        iGy = np.linalg.inv(Gy)
        b = np.linalg.inv(Z.conj().T.dot(iGy).dot(Z)).dot(Z.conj().T.dot(iGy).dot(y))
    elif Gy.size == n:
        # Diagonal matrix Qy
        iGy = 1 / Gy
        b = np.linalg.inv((Z*Gy).conj().T.dot(Z)).dot((Z*Gy).conj().T.dot(y))

    i = 0
    while True:
        i += 1
        ey = y - Z.dot(b)

        if Gz.shape == (n*k, n*k):
            # Full matrix Qz
            bGz = Gz.dot(np.kron(np.eye(n), b))
            b2Gz = np.kron(b.conj().T, np.eye(n)).dot(bGz)
        elif Gz.shape == (n, n, k) or Gz.shape == (n, k):
            # Block-diagonal matrix Qz or vectors specifying the diagonals
            bGz = Gz * b.reshape(1,-1)
            b2Gz = Gz.dot(np.abs(b)**2)

        if Gy.shape != b2Gz.shape:
            # TODO: Need to check which one is a diagonal and which one is a full matrix
            raise NotImplementedError
        elif Gy.size == n*n:
            # If both are matrices
            # TODO: Some error here!!! The structured case works with the general-case code below
            Gy_hat = Gy + b2Gz
            iGy_hat = np.linalg.inv( gy_hat )
            logdetS0, Q = np.linalg.slogdet(Gy_hat)[1], ey.conj().T.dot(iGy_hat).dot(ey)
            if i > niter: break

            eZ = -bQz.dot(iGy_hat.dot(ey)).reshape(n, k)
            Z_hat, y_hat = Z - eZ, y - eZ.dot(b)
            Gb = np.linalg.inv(Z_hat.conj().T.dot(iGy_hat).dot(Z_hat))
            b_old = b
            b = Gb.dot(Z_hat.conj().T.dot(iGy_hat.dot(y_hat)))
        elif Gy.size == n:
            # If both are diagonal vectors
            Gy_hat = Gy + b2Gz             # Covariance matrix of the distribution of the measurement vector
            iGy_hat = 1 / Gy_hat
            logdetS0, Q = np.sum(np.log(Gy_hat)), np.sum(iGy_hat*(np.abs(ey)**2))
            if i > niter: break

            # Compute the error terms for the model matrix and apply constraints if any
            eZ = -(bGz * (iGy_hat*ey)).reshape(n, k)
            Z_hat = Z - eZ    # These constitute the matrix C_hat
            Z_hat = Z_hat / Z_hat.sum(axis=0)*Z.sum(axis=0)
            y_hat = y - (Z - Z_hat).dot(b)
            Gb = np.linalg.inv( (Z_hat*iGy_hat).conj().T.dot(Z_hat) )
            b_old = b
            b = Gb.dot((Z_hat*iGy_hat).conj().T.dot(y_hat))

        C_hat = np.hstack([Z_hat, y_hat])

        # Check if the termination condition is satisfied
        #print(i, np.linalg.norm(b_old-b)/np.linalg.norm(b))
        if np.linalg.norm(b_old-b)/np.linalg.norm(b) < tol:
            break

    return b, C_hat, logdetS0, Q

def ll_ls(Z, y, ampl, m0=None, iS0=None, Gy=None, robust=False):
    """Solves the generalized least-squares problem, y=Zx for x; Gy - covariance matrix of noise (or the diagonal vector of that matrix)."""
    n, k = Z.shape      # Number of dimensions
    isReal = np.isreal(Z).all() and np.isreal(y).all() and (np.any(np.equal(ampl, None)) or np.isreal(ampl).all())    # Determine if the problem is real or complex-valued
    Sc = 1e-42 * np.eye(k, dtype=np.complex_ if not isReal else np.float_)         # Initialize the results
    mc = np.zeros((k, 1), dtype=np.complex_ if not isReal else np.float_)

    # 2. Set up the (Gaussian) priors
    if m0 is None: m0 = np.zeros((k, 1))
    if iS0 is None: iS0 = np.zeros((k, k))

    ifxd = np.where(np.not_equal(ampl, None))[0]           # Indices of components with fixed intensities
    ivar = np.where(np.equal(ampl, None))[0]               # Indices of components whose intensities will be integrated out
    gvar = np.ix_(ivar, ivar)                              # Grid of variable indices

    # Select only the relevant columns in the matrix Z and work with them from now on
    if len(ifxd) > 0:
        mc[ifxd] = ampl[ifxd].reshape(-1, 1)
        y = y - Z[:, ifxd].dot(mc[ifxd])
        Z = Z[:, ivar]

    # 3.
    if Gy is None:
        # No weighting matrix (assume identity)
        ZG = Z.conj().T
        yG = y.conj().T
        logdetG = 0.0
    elif Gy.ndim==1 or (Gy.ndim==2 and (Gy.shape[0]==1 or Gy.shape[1]==1)):
        # Weighting matrix G is diagonal and is defined by the vector
        iGy = 1 / Gy.reshape(1, -1)
        ZG = Z.conj().T * iGy
        yG = y.conj().T * iGy
        logdetG = np.log(np.prod(Gy))
    else:
        # G is a full matrix
        iGy = np.linalg.inv(Gy)
        ZG = Z.conj().T.dot(iGy)
        yG = y.conj().T.dot(iGy)
        logdetG = np.linalg.slogdet(Gy)[1]
    #logdetG -= np.log(np.prod(np.diag(iS0[gvar])))
    #print(np.diag(iS0[gvar]))

    ZZ = ZG.dot(Z)
    Zy = ZG.dot(y)
    if ZZ.size > 0 and np.linalg.matrix_rank(ZZ) < ZZ.shape[0]:
        ZZ += (1e-09)*np.identity(ZZ.shape[0])          # Make sure ZZ is invertible if it is low rank

    Q = yG.dot(y)   # Initialize Q

    if len(ivar) > 0:
        # Compute the ML estimates of the unknown amplitudes, mc, if any # LS problem with unconstrained phase (if complex)
        iSc = iS0[gvar] + ZZ
        Sc[gvar] = np.linalg.inv(iSc)
        mc[ivar] = Sc[gvar].dot(Zy+np.dot(iS0[gvar], m0[ivar]))

        Q += - 2*mc[ivar].conj().T.dot(Zy) + mc[ivar].conj().T.dot(ZZ.dot(mc[ivar]))    # Possibly faster would be to compute r'*G*r

    # Include the priors
    Q += (mc-m0).T.dot(iS0.dot(mc-m0))

    Q = max(np.asscalar(Q).real, 0.0)

    # Define a function that computes the covariance matrix
    if robust:
        r = y - Z.dot(mc[ivar])    # the residual
        r2 = np.abs(r.reshape(1,-1))**2
        ZrZ = (ZG * r2).dot(ZG.conj().T)
        if ZrZ.size > 0 and np.linalg.matrix_rank(ZrZ) < ZrZ.shape[0]:
            ZrZ += (1e-09)*np.identity(ZrZ.shape[0])          # Make sure ZZ is invertible if it is low rank
        # iS0 += 1e-42 * np.eye(k)           # Make sure that the result is invertible
        # iS0[gvar] += ZrZ
        Sc[gvar] = Sc[gvar].dot((iS0[gvar] + ZrZ).real).dot(Sc[gvar])
        # Sr = Sc.dot( iS0.real ).dot(Sc)
        fun_Sc = lambda _ : Sc
    else:
        fun_Sc = lambda sigma2 : sigma2/2 * Sc                # Assuming that the amplitudes are always real, need to scale the covariance matrix by 2

    return mc, fun_Sc, Q, logdetG

def ll_ls_withIntegrationOut(Z, y, ampl=None, m0=None, S0=None, Gy=None, lockedPhase=False):
    """Solves the generalized least-squares problem, y=Zx for x; Gy - covariance matrix of noise (or the diagonal vector of that matrix)."""
    n, k = Z.shape      # Number of dimensions
    #if lockedPhase and indxPositive is None:
    #    indxPositive = list(range(k))
    isReal = np.isreal(Z).all() and np.isreal(y).all() and (ampl is None or np.isreal(ampl).all())    # Determine if the problem is real or complex-valued
    if ampl is None or len(ampl) != k :
        ampl = np.array([None]*k)
    ampl = ampl.reshape(-1, 1)
    mc, Sc = np.zeros((k, 1), dtype=np.complex_ if not isReal else np.float_), np.zeros((k, k), dtype=np.complex_ if not isReal else np.float_)         # Initialize the results
    iSc = np.copy(Sc)
    Q, logdetS = 0.0, 0.0

    ifxd = np.where(np.not_equal(ampl, None))[0]           # Indices of components with fixed intensities
    ivar = np.where(np.equal(ampl, None))[0]               # Indices of components whose intensities will be integrated out
    gvar = np.ix_(ivar, ivar)                              # Grid of variable indices

    # Select only the relevant columns in the matrix Z and work with them from now on
    if len(ifxd) > 0:
        y = y - Z[:, ifxd].dot(ampl[ifxd])
        Z = Z[:, ivar]

    # 2. Set up the (Gaussian) priors
    if m0 is None: m0 = np.zeros((k, 1))
    if S0 is None: S0 = 1e+42 * np.identity(k)
    m0[ifxd] -= ampl[ifxd]     # Subtract the known fixed amplitudes from the prior mean

    # Find which dimensions have priors with infinite or zero variance and invert the covariance matrix
    S0 = np.where(S0 == np.inf, 1e+42, S0)        # Indices of components with infinite-variance (non-informative) intensities
    indx_zero = np.where(np.diag(S0) == 0)[0]     # Indices of components with fixed (zero-variance) intensities
    indx_nonz = np.where(np.diag(S0) != 0)[0]     # Indices of components with variable intensities
    iS0 = np.zeros((k, k))
    iS0[indx_nonz[:,None], indx_nonz] = np.linalg.inv(S0[indx_nonz[:, None], indx_nonz])           # Invert the part that that doesn't have zeros on the diagonal as usual
    iS0[indx_zero, indx_zero] = 1e+42   # np.inf                                                             # Substitute the rest of diagonal values with a very large number
    iS0[np.ix_(ivar, ivar)]

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
        logdetS -= 1/2*np.log(np.prod(G)) if isReal else np.log(np.prod(G))
    else:
        # G is a full matrix
        iGy = np.linalg.inv(Gy)
        ZG = Z.conj().T.dot(iGy)
        yG = y.conj().T.dot(iGy)
        logdetS -= 1/2*np.linalg.slogdet(G)[1] if isReal else np.linalg.slogdet(G)[1]
    ZZ = ZG.dot(Z)
    Zy = ZG.dot(y)
    if ZZ.size > 0 and np.linalg.matrix_rank(ZZ) < ZZ.shape[0]:
        ZZ += (1e-09)*np.identity(ZZ.shape[0])

    if len(ivar) > 0:
        if lockedPhase:                 # Locked phase - real amplitudes
            pass
        else:
            # LS problem with unconstrained phase (if complex)
            theta = None
            iSc[gvar] = iS0[gvar] + ZZ
            Sc[gvar] = np.linalg.inv(iSc[gvar])
            mc[ivar] = Sc[gvar].dot(Zy+np.dot(iS0[gvar], m0[ivar]))
        logdetS += 1/2*np.linalg.slogdet(Sc[gvar])[1] if isReal or lockedPhase else np.linalg.slogdet(Sc[gvar])[1]
        Q -= mc[ivar].conj().T.dot(iSc[gvar].dot(mc[ivar]))
    mc[ifxd] = ampl[ifxd]     # Replace values if neecessary (where variance is 0)

    Q += yG.dot(y) + m0.conj().T.dot(iS0.dot(m0))
    Q = max(np.asscalar(Q).real, 0.0)

    return mc, Sc, Q, logdetS

def ll_tls(Z, y, ampl, m0=None, iS0=None, Gz=None, Gy=None, gamma=None, maxiter=15, constr=False):
    from scipy import optimize

    """Finds the ML estimate in the TLS problem; computes the unknown amplitudes and/or gamma if necessary."""
    n, k = Z.shape      # Number of dimensions

    # 2. Set up the (Gaussian) priors
    if m0 is None: m0 = np.zeros((k, 1))
    if iS0 is None: iS0 = np.zeros((k, k))
    Sc = 1e-42 * np.eye(k, dtype=ampl.dtype)         # Initialize the results

    ifxd = np.where(np.not_equal(ampl, None))[0]           # Indices of components with fixed intensities
    ivar = np.where(np.equal(ampl, None))[0]               # Indices of components whose intensities will be integrated out
    gvar = np.ix_(ivar, ivar)                              # Grid of variable indices

    # If at least some amplitudes are unknown, estimate their initial values with the usual LS algorithm
    if len(ivar) > 0 or gamma is None:
        mc, _, _, _ = ll_ls(Z, y, ampl, m0, iS0, Gy)
        bounds = mc[ivar,:] + np.abs(mc[ivar,:])*np.array([-0.5, 0.5])

        # Define functions that need to be optimized
        a_sigma2, b_sigma2 = 2, 10    # 0.1, 0.1    # Need to choose some prior for the noise distribution
        def func_opti_ampl(x):
            """Function that takes a vector of parameters and returns its value, gradient, and Hessian.
             The elemnts of x correspond to the unknown amplitudes whose order is given by ivar."""
            ampl_eval = np.copy(mc)
            ampl_eval[ivar] = x.reshape(-1, 1)
            Q, logdetGc, grad = ll_tls_eval(Z, y, ampl_eval, m0, iS0, Gz, Gy, gamma, jac=True, constr=constr)

            # Compute the value of the entire likelihood function assuming inverse gamma prior for sigma2
            fun = logdetGc/2 + (n/2+a_sigma2)*np.log(Q+b_sigma2)
            jac = grad.dlda/2 + (n/2+a_sigma2)/(Q+b_sigma2)*grad.dQda
            return np.asscalar(fun), jac[ivar].ravel()

        def func_opti_gamma(x):
            """Function that takes a vector of parameters and returns its value, gradient, and Hessian.
             The elemnts of x correspond to the unknown value of gamma."""
            Q, logdetGc, grad = ll_tls_eval(Z, y, ampl=mc, m0=m0, iS0=iS0, Gz=Gz, Gy=Gy, gamma=x, jac=True, constr=constr)

            # Compute the value of the entire likelihood function assuming inverse gamma prior for sigma2
            fun = logdetGc/2 + (n/2+a_sigma2)*np.log(Q+b_sigma2)
            jac = grad.dldg/2 + (n/2+a_sigma2)/(Q+b_sigma2)*grad.dQdg
            return np.asscalar(fun), jac.ravel()

        # Start by optimizing over gamma first
        flag = False
        if gamma is None:
            gamma = np.asscalar( optimize.minimize(func_opti_gamma, jac=True, x0=0.5, bounds=((1e-12, 1-1e-12), ), method='L-BFGS-B').x )
            flag = True

        # Optimize over amplitudes
        if len(ivar) > 0:
            mc[ivar] = optimize.minimize(func_opti_ampl, jac=True, x0=mc[ivar], bounds=bounds, method='L-BFGS-B').x.reshape(-1, 1)
            # If needed, optimize over gamma again and find new amplitudes
            if flag:
                for _ in range(maxiter):
                    gamma = np.asscalar( optimize.minimize(func_opti_gamma, jac=True, x0=0.5, bounds=((1e-12, 1-1e-12), ), method='L-BFGS-B').x )   # Update gamma
                    mc[ivar] = optimize.minimize(func_opti_ampl, jac=True, x0=mc[ivar], bounds=bounds, method='L-BFGS-B').x.reshape(-1, 1)       # UPdate the amplitudes
    else:
        mc = np.copy(ampl)

    # Evaluate the log-likelihood function (compute logdet and Q)
    Q, logdetGc, grad = ll_tls_eval(Z, y, ampl=mc, m0=m0, iS0=iS0, Gz=Gz, Gy=Gy, gamma=gamma, hes=True, constr=constr)
    def fun_Sc(sigma2):
        hes = -grad.d2lda2/2 - grad.d2Qda2/sigma2
        return -np.linalg.inv(hes)

    return mc, fun_Sc, Q, logdetGc, gamma

#@profile
def ll_tls_eval(Z, y, ampl, m0=None, iS0=None, Gz=None, Gy=None, gamma=0.5, jac=False, hes=False, constr=False):
    """Only evaluates the Structured Total Maximum Likelihood with Gaussian priors on the amplitudes. See Beck and Eldar."""
    # NOTE: Possibly there are errors in computing the gradients/hessians for complex-valued arguments (e.g. conj().T vs .T, etc...)
    n, k = Z.shape      # Number of dimensions
    ampl = ampl.reshape(k, 1)
    Q, logdetGc, dQdg, dldg = 0.0, 0.0, 0.0, 0.0
    dQda, dlda = np.zeros((k, 1)), np.zeros((k, 1))
    d2lda2, d2Qda2 = np.zeros((k, k)), np.zeros((k,k))

    if Gz is None: Gz = np.ones((n, k))
    if Gy is None: Gy = np.ones((n, 1))
    Gz, Gy = gamma*Gz, (1-gamma)*Gy

    ey = y - Z.dot(ampl)
    #C_hat = np.hstack([Z, y])     # Initialize C_hat

    # Compute b2Gz
    if Gz.shape == (n*k, n*k):
        # Full matrix Gz
        bGz = Gz.dot(np.kron(np.eye(n), ampl))
        b2Gz = np.kron(ampl.conj().T, np.eye(n)).dot(bGz)
    elif Gz.shape == (n, n, k):
        # Block-diagonal matrix Gz
        b2Gz = Gz.dot(np.abs(ampl)**2).squeeze()
    elif Gz.size == n*k:
        # Diagonal matrix Gz, possibly needs to be reshaped
        Gz = Gz.reshape(n, k)
        b2Gz = Gz.dot(np.abs(ampl)**2)

    # Invert the covariance matrix Gy
    if Gy.shape == (n, n):
        # Full matrix Gy
        iGy = np.linalg.inv(Gy)
    elif Gy.size == n:
        # Diagonal matrix Gy
        iGy = 1 / Gy

    if Gy.size == n*n or b2Gz.size == n*n:
        # If at least one of them is a matrix
        if Gy.size == n: Gy = np.diag(Gy.ravel())
        if b2Gz.size == n: b2Gz = np.diag(b2Gz.ravel())

        Gc = Gy + b2Gz
        iGc = np.linalg.inv( Gc )
        logdetGc, Q = np.linalg.slogdet(Gc)[1], np.asscalar(ey.conj().T.dot(iGc).dot(ey))

        if constr:
            raise NotImplementedError

        # Compute the partial derivatives
        if jac or hes:
            dGc_dg = b2Gz/gamma-Gy/(1-gamma) if gamma != 0 else b2Gz - Gy      # derivative of Gc wrt to gamma
            diGc_dg = -iGc.dot(dGc_dg).dot(iGc)
            dQdg = ey.conj().T.dot(diGc_dg).dot(ey)
            dldg = np.trace(iGc.dot(dGc_dg))
            dGc_da = 2*Gz*ampl.reshape(1, 1, -1)         # gradient vector wrt the amplitudes
            #diGc_da = -np.dstack([iGc.dot(dGc_da[..., i]).dot(iGc) for i in range(k)])
            diGc_da = -( ((iGc.T.dot(dGc_da.reshape(n, -1, order='F'))).reshape(n, n, -1, order='F')).transpose(1,0,2).reshape(n,-1,order='F').T.dot(iGc) ).T.reshape(n,n,-1,order='F').transpose(1,0,2)      # The same as    diGc_da = -np.dstack([iGc.dot(dGc_da[..., i]).dot(iGc) for i in range(k)])
            dQda = ( -2*(ey.conj().T.dot(iGc)).dot(Z).real + \
                     np.array([ey.conj().T.dot(diGc_da[..., i]).dot(ey) for i in range(k)]).reshape(1, -1) ).T
            dlda = np.array([np.trace(iGc.dot(dGc_da[..., i])) for i in range(k)]).reshape(-1, 1)

            if constr:
                raise NotImplementedError

        # Compute the Hessian matrices
        if hes:
            d2lda2 = np.tensordot(diGc_da.T, dGc_da, 2) + 2*np.diag(np.tensordot(iGc, Gz, 2))
            d2iGc_da2 = - 2*iGc[..., None]*(diGc_da*dGc_da + iGc[..., None]*Gz )
            d2Qda2 = 2*(Z.conj().T.dot(iGc.dot(Z)) - 2*np.squeeze(ey.conj().T.dot(diGc_da)).T.dot(Z)).real \
                    + np.diag( np.squeeze(ey.conj().T.dot(d2iGc_da2)).T.dot(ey).ravel() )

            if constr:
                raise NotImplementedError

    elif Gy.size == n:
        # If both are diagonal vectors
        Gc = Gy + b2Gz             # Covariance matrix of the distribution of the measurement vector
        iGc = 1 / Gc
        logdetGc, Q = np.sum(np.log(Gc)), np.sum(iGc*(np.abs(ey)**2))
        """if constr:
            # Compute the correction matrices for Gc and iGc
            print('Constrained')

            g = (1/(2*n)*np.sum(b2Gz) - b2Gz)/n
            C = g + g.T
            luv = np.asscalar(g.T.dot(iGc)) + 1
            luu, lvv = np.asscalar((g**2).T.dot(iGc)) / luv, np.asscalar(iGc.sum()) / luv
            den = luv*(1-luu*lvv)      # Denominator expression
            iC = 1/den * iGc.T*(luu + lvv*g*(g.T) - C)*iGc
            print(den, iC.shape)
            Gc = np.diag(Gc.ravel()) + C
            iGc1 = np.linalg.inv(Gc)
            iGc = np.diag(iGc.ravel()) + iC
            logdetGc, Q = np.linalg.slogdet(Gc)[1], np.asscalar(ey.conj().T.dot(iGc).dot(ey))
            logdetGc, Q = np.log(luv**2*(1 - luu*lvv))+np.sum(np.log(Gy + b2Gz)), np.asscalar(ey.conj().T.dot(iGc).dot(ey))
            """
        if constr:
            # Compute the correction matrices for Gc and iGc
            g = (1/(2*n)*np.sum(b2Gz) - b2Gz)/n   # New matrix Gc can be computed as Gy+b2Gz+ g.dot(np.ones((1,n)))+np.ones((n,1)).dot(g.T)
            luv = np.asscalar(g.T.dot(iGc)) + 1
            luu, lvv = np.asscalar((g**2).T.dot(iGc)) / luv, np.asscalar(iGc.sum()) / luv
            #iC = 1/(luv*(1-luu*lvv)) * iGc.T*( luu + lvv*g*(g.T) - (g+g.T) )*iGc
            #dQ = np.asscalar(ey.conj().T.dot(iC).dot(ey))

            iGc_ey = iGc*ey
            dQ = np.asscalar( lvv*(g.T.dot(iGc_ey))**2 - 2*iGc_ey.sum()*g.T.dot(iGc_ey) + luu*(iGc_ey.sum())**2 )
            dQ *= 1/(luv*(1-luu*lvv))

            logdetGc += np.log(luv**2*(1 - luu*lvv))
            Q += dQ

        # Compute the partial derivatives
        if jac or hes:
            dGc_dg = b2Gz/gamma-Gy/(1-gamma) if gamma != 0 else b2Gz - Gy      # derivative of Gc wrt to gamma
            diGc_dg = -iGc**2 * dGc_dg
            dQdg = (ey * diGc_dg).conj().T.dot(ey)
            dldg = (iGc*dGc_dg).sum()
            dGc_da = 2*Gz*ampl.reshape(1, -1)         # gradient vector wrt the amplitudes
            diGc_da = -iGc**2 * dGc_da
            dQda = ( -2*((ey*iGc).conj().T.dot(Z)).real + (ey**2).T.dot(diGc_da) ).T
            dlda = np.sum(iGc*dGc_da, axis=0).reshape(-1, 1)

            if constr:
                pass
                #raise NotImplementedError

        # Compute the Hessian matrices
        if hes:
            d2lda2 = diGc_da.T.dot(dGc_da) + np.diag(np.sum(2*Gz*iGc, axis=0))
            d2iGc_da2 = - 2*(diGc_da*dGc_da*iGc + Gz*(iGc**2) )
            d2Qda2 = 2*(Z.conj().T.dot(iGc*Z) - 2*(ey.conj()*diGc_da).T.dot(Z)).real + np.diag(np.sum(d2iGc_da2*(np.abs(ey)**2), axis=0))

            if constr:
                pass
                #raise NotImplementedError

    # Include the prior
    if m0 is not None and iS0 is not None:
        Q += (ampl-m0).T.dot(iS0.dot(ampl-m0))
        dQda += iS0.dot(ampl-m0)

    Q = np.asscalar(Q)
    logdetGc = np.asscalar(logdetGc)
    return Q, logdetGc, grad(dQda, dQdg, dlda, dldg, d2lda2, d2Qda2)

def log_likelihood(Z, y, ampl0=None, sigma2_0=None, Gz=None, Gy=None, gamma0=None, m0=None, iS0=None, a_sigma2_0=2.0, b_sigma2_0=10.0, funcType='LS', constr=False, robust=False, nonnegative=True, na=None):
    """
       Computes the value of the Gaussian likelihood function. iG - inverse co-
       variance matrix of the noise.
       ampl0 - array of intial amplitudes, entries which are initialized to None
               will be estimated in closed form.
       If nonnegative=True, first na amplitudes will be forced to have non-negative values.
       """
    # 0. Prepare the inputs
    n, k = Z.shape     # Number of samples and (model signals)
    isReal = np.isreal(Z).all() and np.isreal(y).all() and (ampl0 is None or np.isreal(ampl0).all())    # Determine if the problem is real or complex-valued
    #gamma0 = max(min(gamma0, 1-1e-12), 1e-12)    # Make sure gamma is within allowed bounds
    
    # Initialize the array of amplitudes
    if ampl0 is None:
        ampl0 = np.array([None]*k)
    ampl0 = ampl0.reshape(-1, 1)
    if na is None:
        na = k
    if funcType == 'LS':
        mc, fun_Sc, Q, logdetG = ll_ls(Z, y, ampl0, m0, iS0, Gy, robust=robust)
        gamma = gamma0
    elif funcType == 'TLS':
        mc, fun_Sc, Q, logdetG, gamma = ll_tls(Z, y, ampl0, m0, iS0, Gz, Gy, gamma0, constr=constr)
    result = -logdetG/2 if isReal else -logdetG

    # Sigma2
    if sigma2_0 is not None:
        # Evaluate the posterior (without integrating out sigma2)
        sigma2 = max(sigma2_0, 1e-16)
        a_sigma2, b_sigma2 = a_sigma2_0, b_sigma2_0
        result = result - n/2*np.log(sigma2) if isReal else result - n*np.log(sigma2)
        result -= Q / sigma2
    else:
        a_sigma2 = a_sigma2_0 + n/2 if isReal else a_sigma2_0 + n
        b_sigma2 = b_sigma2_0 + Q           # Parameters of the posterior distribution for sigma2
        sigma2 = b_sigma2/(a_sigma2-1)    # Mean estimator for sigma2
        # Evaluate the posterior (after integrating sigma2 out)
        result += - a_sigma2*np.log(b_sigma2)

    result = result - n/2*np.log(np.pi) if isReal else result - n*np.log(np.pi)
    result = np.asscalar(result.real)

    # Report posterior distributions for amplitudes
    m_ampl = mc.copy()
    S_ampl = fun_Sc(sigma2)

    # # Constrain amplitudes to non-negative values and re-estimate them using fewer components
    theta_0 = np.asscalar( 1/2*np.angle(mc[:na].T.dot(mc[:na])) )   # Global phase estimated from the complex valued amplitudes
    m_ampl[:na] = (mc[:na]*np.exp(-1j*theta_0)).real                                   # #m_ampl[:na] = m_ampl[:na].real
    if m_ampl[:na].sum() < 0:      # Make sure that all amplitudes are positive
        m_ampl *= -1
        mc *= -1

    if np.any(m_ampl[:na] < 0):
        ampl0[:na] = np.where(m_ampl[:na] <= 0.0, 0.0, ampl0[:na])
        result, mc, sigma2, dic = log_likelihood(Z, y, ampl0, sigma2_0, Gz, Gy, gamma0, m0, iS0, a_sigma2_0, b_sigma2_0, funcType, constr, robust, nonnegative, na)
    else:
        dic = {"ampl":(m_ampl, S_ampl), "sigma2":(a_sigma2, b_sigma2), "gamma":gamma}

    return result, mc, sigma2, dic          # Output the log value and parameters of the marginalized distributions







def getRandomSignals(sig2x, sig2y, d=5, n=250):
    Z = np.random.randn(n, d)
    b = np.array([2, 5, 7, 9, 4])[:,None]
    x = Z.dot(b)
    y = (Z + np.sqrt(sig2x/2)*(np.random.randn(n,d)+1j*np.random.randn(n,d))).dot(b) + np.sqrt(sig2y/2)*(np.random.randn(n,1)+1j*np.random.randn(n,1))
    return Z, y, b

def getMeasuredSignals(useComplex=False, DDD=None):
    if DDD is None:
        # Load a saved workspace
        filename = 'Thiamine_PANIC.wsp'    # Thiamine_nonGaussianAmplitudes
        with open(filename, 'rb') as fp:
            dataPack, GUIsettings = pickle.load(fp)
        wsp = Workspace()       # Define a new Workspace object
        wsp.unpack(dataPack)    # Unpack the loaded data into it

        # Update the global settings
        try:
            stngConfig = GUIsettings.pop('_config')
            for key, val in stngConfig.items():
                if key == 'QD_AggregatePeaksThreshold':
                    config.QD_AggregatePeaksThreshold = val
                elif key == 'QD_RerunQDchshThreshold':
                    config.QD_RerunQDchshThreshold = val
                elif key == 'OPTIM_maxBasinhoppingSteps':
                    config.OPTIM_maxBasinhoppingSteps = val
                elif key == 'OPTIM_niterSuccess':
                    config.OPTIM_niterSuccess = val
                elif key == 'OPTIM_method':
                    config.OPTIM_method = val
                elif key == 'SAMPL_varEstimator':
                    config.SAMPL_varEstimator = val
        except: pass

        # Select a Series and a Datum
        DDD = wsp.series[1].data[0]

    ##
    self = DDD
    frqBlkIds = self.steps[-1].frqBlkIds
    evalParsH = self.crntParsH
    wnd = None
    customPriors=None
    returnSignals=False
    varEstimator=None
    marginalize=True
    lockedPhase=True
    nw = len(self.sF) if self.sF is not None else 0         # Length of the adaptive lineshape window (in frequency domain)
    nw2 = int(nw/2)
    indxInRange = np.concatenate(tuple(self.freqBlocks[i].indxFreq for i in frqBlkIds))
    indxPadding = np.concatenate(tuple(np.concatenate([np.arange(self.freqBlocks[i].indxFreq[0]-nw2, self.freqBlocks[i].indxFreq[0]),
                                                       np.arange(self.freqBlocks[i].indxFreq[-1]+1, self.freqBlocks[i].indxFreq[-1]+nw2+1)%len(self.f)] ) \
                                for i in frqBlkIds)) if nw2>0 else np.array([], dtype='int')   # Extra indices used for padding when convolving the signals with lineshape kernel in frequency domain

    if ( 'lshapeR' in evalParsH['.'].keys() and (any(evalParsH['.']['lshapeR']) or any(evalParsH['.']['lshapeI'])) ) or wnd is not None:
        # 1. Compute the model signals
        zT, repRootNames = getFID(self.T, self.t, self.c0, self.f0, evalParsH, tau=0.0)

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
        zFall, repRootNames = evalTreeF(self.T, self.f.take(np.concatenate([indxInRange, indxPadding])), self.t[1]-self.t[0], self.f[1]-self.f[0], self.c0, self.f0, evalParsH)
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
    theta, tau = evalParsH['.']['theta'][0], evalParsH["."]["tau"][0]
    yFinRange *= np.exp(-1j*2*np.pi * tau * (self.f.take(indxInRange)*self.c0-self.f0) - 1j*theta ).reshape((-1,1))   # A shorter vector of yF restricted to the optimization range only

    # Include the baseline
    bslnPoly = block_diag(*[self.freqBlocks[i].bF for i in frqBlkIds if self.freqBlocks[i].bF is not None])     # All baseline models padded with zeros; use only real-valued baselines if the model is real-valued
    if not useComplex:
        bslnPoly = bslnPoly[:, np.isreal(bslnPoly).all(axis=0)]
    nb = bslnPoly.shape[1]     # Total number of baseline terms

    # Define modelled and measured signals
    Z, y = np.hstack((zFinRange, bslnPoly)), yFinRange
    b = np.ones((Z.shape[1], 1))

    return Z, y, b, nb

def getLorentzianSignals(sig2x, sig2y, beta=0.0):
    nt, nf = 1024, 1024
    c0, f0, dt = 1.0, 0.0, 1e-02
    t = np.linspace(0, (nt-1)*dt, nt).reshape(-1,1)
    f = (np.fft.fftshift(np.fft.fftfreq(nf, t[1]-t[0]))+f0).reshape(-1,1)/c0
    alpha = np.array([2.5, 4.0, 2.0, 5.0])
    freq = np.array([10, -33, -35, -13])
    ampl = np.exp(1j*np.pi/3*0) * np.array([13, 5, 7, 16]).reshape(-1,1)

    # Generate model signals
    xT = np.exp(0*1j*2*np.pi/3)*np.exp(-alpha*t+1j*2*np.pi*freq*t)
    xF = np.fft.fftshift(np.fft.fft(xT, nf, axis=0), axes=0) / np.sqrt(nf)

    # Generate noisy signals
    nxT = np.sqrt(sig2x/2)*(np.random.randn(*xT.shape) + 1j*np.random.randn(*xT.shape))
    nyT = np.sqrt(sig2y/2)*(np.random.randn(nt,1) + 1j*np.random.randn(nt,1))
    yT = (xT*np.exp(-beta*t**2) + nxT).dot(ampl).reshape(-1,1) + nyT
    yF = np.fft.fftshift(np.fft.fft(yT, nf, axis=0), axes=0) / np.sqrt(nf)

    return xF, yF, np.abs(ampl)
