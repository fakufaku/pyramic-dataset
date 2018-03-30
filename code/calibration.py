"""
Calibration
===========

This module contains routines for the blind calibration of a microphone
array with sources in the far field. The methods are:

    * `joint_calibration_gd`: Straightforward gradient descent method
    * `joint_calibration_sgd`: Straightforward stochastic gradient descent method
    * `structure_from_sound`: The SVD based method from Thrun [1]

[1] Sebastian Thrun, __[Affine Structure from Sound](https://papers.nips.cc/paper/2770-affine-structure-from-sound.pdf)__, NIPS, 2007

Author: 2018 (c) Robin Scheibler
License: MIT License
"""

import numpy as np
import json, os
from scipy.io import wavfile
import pyroomacoustics as pra

def sph2cart(r, colatitude, azimuth):
    """
    spherical to cartesian coordinates
    :param r: radius
    :param colatitude: co-latitude
    :param azimuth: azimuth
    :return:
    """
    r_sin_colatitude = r * np.sin(colatitude)
    x = r_sin_colatitude * np.cos(azimuth)
    y = r_sin_colatitude * np.sin(azimuth)
    z = r * np.cos(colatitude)
    return x, y, z

def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    azimuth = np.arctan2(y, x)
    r_sin_colatitude = np.sqrt(x**2 + y**2)
    colatitude = np.pi / 2 - np.arctan2(z, r_sin_colatitude)

    return r, colatitude, azimuth

def structure_from_sound(delta, gd_step_size=1e-4, gd_n_steps=10000, stop_tol=None, enable_convergence_curve=False, verbose=False):
    '''
    Implementation of "Structure from Sound" calibration algorithm

    Parameters
    ----------
    delta: ndarray
        A matrix of TDOA with M-1 rows and N columns where M is the number of
        microphones and N the number of sound events.
    gd_step_size: float, optional
        The step size for the gradient descent
    gd_n_steps: float, optional
        The number of steps for the gradient descent
    stop_tol: float, optional
        The gradient descent stops when the improvement becomes smaller than this number
    verbose: bool, optional
        Print extra convergence information and plot the convergence curve

    Returns
    -------
    1) An ndarray containing the microphones locations in the columns (relative to reference microphone)
    2) An ndarray containing the directions of the sound events in the columns
    '''

    ### STEP 1 : Perform SVD and low-rank truncation of the delays matrix ###
    U, V, W = np.linalg.svd(delta)

    X1 = np.dot(U[:,:3], np.diag(V[:3]))  # temporary location of sensor matrix
    P1 = W[:3,:]  # temporary direction of acoustic events matrix

    ### STEP 2 : Find the appropriate rotation matrix to make sure X and G satisfy the structure ###
    C = np.eye(3)      # initialize at identity

    err_previous = None

    convergence = []
    interval = gd_n_steps // 30

    for i in range(gd_n_steps):
        # compute gradient
        B = np.dot(C, P1)
        err = np.sum(B**2, axis=0) - np.ones(P1.shape[1])
        gradient = np.sum(err[np.newaxis,np.newaxis,:] * B[:,np.newaxis,:] * P1[np.newaxis,:,:], axis=2)

        e = np.sum(err**2) / P1.shape[1]

        if err_previous is not None:
            improvement = err_previous - e
        else:
            improvement = e

        if verbose and i % interval  == 0:

            if enable_convergence_curve:
                convergence.append(e)
            if verbose:
                print('{} error={} improvement={}'.format(i, e, improvement))

        err_previous = e

        if stop_tol is not None and improvement < stop_tol:
            break

        # step lower
        C -= gd_step_size * gradient

    X = np.dot(X1, np.linalg.inv(C)).T
    P = np.dot(C, P1)

    if enable_convergence_curve:
        return X, P, convergence
    else:
        return X, P


def joint_calibration_gd(delta, mask=None, gd_step_size=0.003, gd_n_steps=3000, 
        X=None, P=None, dim=3, 
        enable_convergence_curve=False, verbose=False):
    '''
    Perform joint calibration of far field sources and microphones locations
    based on TDOA measurements.

    Parameters
    ----------
    delta: ndarray (n_mics, n_sources)
        The TDOA measurements matrix (in meters)
    gd_step_size: float
        The step size for the gradient descent
    gd_n_steps: int
        The number of iterations of the gradient descent
    X: ndarray, optional
        The initial estimate of the microphone locations
    P: ndarray, optional
        The inital estimate of the DOA of the sources
    dim: int, optiona
        The dimension of the Euclidean space (default 3)
    '''

    n_mics, n_sources = delta.shape

    if mask is None:
        mask = np.ones((n_mics, n_sources))

    if X is None:
        X = np.random.randn(dim, n_mics)
        proj_X = False
    else:
        X0 = X
        X = X0.copy()
        proj_X = True

    if P is None:
        P = np.random.randn(dim, n_sources)
        P /= np.linalg.norm(P, axis=0)[None,:]
    else:
        P0 = P
        P = P0.copy()

    if enable_convergence_curve:
        convergence_curve = []
    interval = gd_n_steps // 30

    err_previous = None

    for i in range(gd_n_steps):
        # compute gradient

        err_vec = mask * (np.dot(X.T, P) + delta)
        grad_P = np.dot(X, err_vec)
        grad_X = np.dot(P, err_vec.T)

        # rmse
        err = np.sqrt(np.mean(err_vec**2))

        if err_previous is not None:
            improvement = err_previous - err
        else:
            improvement = err

        if i % interval == 0:
            if enable_convergence_curve:
                convergence_curve.append(err)
            if verbose:
                print('{} error={} improvement={}'.format(i, err, improvement))

        err_previous = err

        # gradient step
        X -= gd_step_size * grad_X
        P -= gd_step_size * grad_P

        # project sources on the unit sphere
        #P /= np.linalg.norm(P, axis=0)[np.newaxis,:]

        # project the microphones to be as close as possible to initial
        # configuration (if it was provided)
        if proj_X:
            u,s,v = np.linalg.svd(np.dot(X0, X.T))
            R = np.dot(u,v)
            X = np.dot(R, X)
            P = np.dot(R, P)

    if enable_convergence_curve:
        return X, P, convergence_curve
    else:
        return X, P

def joint_calibration_sgd(delta, mask=None, gd_step_size=0.003, gd_n_steps=3000, 
        X=None, P=None, dim=3, 
        enable_convergence_curve=False, verbose=False):
    '''
    Perform joint calibration of far field sources and microphones locations
    based on TDOA measurements.

    Parameters
    ----------
    delta: ndarray (n_mics, n_sources)
        The TDOA measurements matrix (in meters)
    gd_step_size: float
        The step size for the gradient descent
    gd_n_steps: int
        The number of iterations of the gradient descent
    X: ndarray, optional
        The initial estimate of the microphone locations
    P: ndarray, optional
        The inital estimate of the DOA of the sources
    dim: int, optiona
        The dimension of the Euclidean space (default 3)
    '''

    n_mics, n_sources = delta.shape

    if mask is None:
        mask = np.ones((n_mics, n_sources))

    if X is None:
        X = np.random.randn(dim, n_mics)
        proj_X = False
    else:
        X0 = X
        X = X0.copy()
        proj_X = True

    if P is None:
        P = np.random.randn(dim, n_sources)
        P /= np.linalg.norm(P, axis=0)[None,:]
    else:
        P0 = P
        P = P0.copy()

    if enable_convergence_curve:
        convergence_curve = []
    interval = gd_n_steps // 30

    err_previous = None

    for i in range(gd_n_steps):

        # run over all microphones
        for m in range(n_mics):
            err_vec = mask[m,:] * (np.dot(P.T, X[:,m]) + delta[m,:])
            grad_X = np.dot(P, err_vec)

            # gradient step
            X[:,m] -= gd_step_size * grad_X

        # project the microphones to be as close as possible to initial
        # configuration (if it was provided)
        if proj_X:
            u,s,v = np.linalg.svd(np.dot(X0, X.T))
            R = np.dot(u,v)
            X = np.dot(R, X)
            P = np.dot(R, P)

        # run over all sources
        for k in range(n_sources):
            err_vec = mask[:,k] * (np.dot(X.T, P[:,k]) + delta[:,k])
            grad_P = np.dot(X, err_vec)

            # gradient step
            P[:,k] -= gd_step_size * grad_P

            # project sources on the unit sphere
            #P[:,k] /= np.linalg.norm(P[:,k])

        # rmse
        err_vec = mask * (np.dot(X.T, P) + delta)
        err = np.sqrt(np.mean(err_vec**2))

        if err_previous is not None:
            improvement = err_previous - err
        else:
            improvement = err

        if i % interval == 0:
            if enable_convergence_curve:
                convergence_curve.append(err)
            if verbose:
                print('{} error={} improvement={}'.format(i, err, improvement))

        err_previous = err


    if enable_convergence_curve:
        return X, P, convergence_curve
    else:
        return X, P
