# -*- coding: utf-8 -*-
# time: 2023/7/29 15:48
# file: partial trace.py
# author: Felix_Zhang
# email: yuqizhang247@gmail.com

import numpy as np
from scipy.linalg import block_diag

def PartialTrace(X, sys=2, dim=None, mode=-1):
    lX = len(X)
    if dim is None:
        dim = int(np.round(np.sqrt(lX)))

    num_sys = len(dim)

    # allow the user to enter a single number for dim
    if num_sys == 1:
        dim = [dim, lX // dim]
        if abs(dim[1] - round(dim[1])) >= 2 * lX * np.finfo(float).eps:
            raise ValueError('PartialTrace:InvalidDim', 'If DIM is a scalar, DIM must evenly divide length(X).')
        dim[1] = round(dim[1])
        num_sys = 2

    prod_dim = np.prod(dim)
    prod_dim_sys = np.prod(dim[sys - 1])

    # If X has just one row or one column then it is a pure state vector whose
    # partial trace can be computed more quickly.
    if np.min(X.shape) == 1:
        isPureState = True
        X = X.reshape((-1, 1))  # make sure it is a column vector

        # Determine which of two computation methods to use (i.e., guess which
        # method will be faster).
    elif mode == -1:
        mode = isinstance(X, np.ndarray) and X.dtype == np.float64 and \
               sp.issparse(X) and prod_dim_sys ** 2 <= prod_dim

    # If it's a pure state, compute its partial trace a bit more quickly,
    # without computing X*X' (which might take a lot of memory in high
    # dimensions).
    if isPureState:
        perm = [sys - 1] + [i for i in range(num_sys) if i != sys - 1]
        pDimRat = prod_dim / prod_dim_sys

        X = PermuteSystems(X, perm, dim, 1)  # permute the subsystems so that we just have to do the partial trace on the first (potentially larger) subsystem
        Xtmp = X.reshape((pDimRat, prod_dim_sys))
        Xpt = Xtmp @ Xtmp.T

    # If the matrix is sparse and the amount we are tracing over is smaller
    # than the amount remaining, just do the naive thing and manually add up
    # the blocks.
    elif mode:
        sub_sys_vec = np.full((1, prod_dim_sys), prod_dim / prod_dim_sys)

        perm = [sys - 1] + [i for i in range(num_sys) if i != sys - 1]
        X = PermuteSystems(X, perm, dim)
        Xpt = np.zeros((prod_dim_sys, prod_dim_sys), dtype=X.dtype)

        for j in range(prod_dim_sys):
            Xpt += block_diag(*X[j::prod_dim_sys, j::prod_dim_sys])

    # Otherwise, do a clever trick with mat2cell or reshaping, which is almost always faster.
    else:
        sub_prod = prod_dim // prod_dim_sys
        sub_sys_vec = np.full((1, sub_prod), prod_dim / sub_prod)

        perm = [i for i in range(num_sys) if i != sys - 1] + [sys - 1]
        Xpt = PermuteSystems(X, perm, dim)

        if isinstance(X, np.ndarray):  # if the input is a numeric matrix, perform the partial trace operation the fastest way we know how
            Xpt = np.array([np.trace(x) for x in np.array_split(Xpt, sub_prod)])  # partial trace on second subsystem
            if sp.issparse(X):  # if input was sparse, output should be too
                Xpt = sp.csr_matrix(Xpt)
        else:  # if the input is not numeric (such as a variable in a semidefinite program), do a slower method that avoids mat2cell (mat2cell doesn't like non-numeric arrays)
            Xpt = Xpt.reshape((sub_sys_vec[0], sub_prod, sub_sys_vec[0], sub_prod)).transpose(1, 3, 0, 2).reshape(
                (sub_prod, sub_prod, sub_sys_vec[0] ** 2)).sum(axis=2)

    return Xpt
