#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extra tools for Chebyshev polynomials

This provides some extra functions for working with Chebyshev
polynomials beyond those in the Numpy polynomial class. This is of
use to go backwards and forwards between spectral and nodal space 
quite  a lot and is a quicker way to find Chebyshev coefficients
from data, and Chebyshev functions on radial points. It turns out 
the key to this is to pick the radial points with care. If we do 
this the transform becomes a matrix multiplication with vectors 
of coefficents and values, and the matrix only depending on the 
points. The functions below are based on Appendix B.2 of Boyd 
(2014) using numpy's functions to speed things up in places.

References:

Boyd, J. P. (2014) "Solving transcendental equations : the Chebyshev
polynomial and other numerical rootfinders, pertubation series, and 
oracles", Society for Industrial and Applied Mathematics, Philadelphia.
"""
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import numpy.polynomial.polyutils as pu


def cheb_get_nodes(degree, domain=None):
    """Find the nodel points for a Chebyshev of given degree and domain
    
    This is a vectorized version of equation B.4 in Boyd (2014 pp.411)
    Default domain is [-1,1] and points start from 0, consistent with 
    definition
    """
    if domain is None:
        domain = [-1,1]
    points = np.arange(degree+1)
    points = np.cos((points*np.pi)/degree)
    nodes = pu.mapdomain(points, [-1,1], domain)
    return nodes


def cheb_interp_matrix(degree):
    """Return the matrix needed to find coefficents from nodal values
    
    This is a direct implementation of equation B5. of Boyd (2014 p.411).
    The returned matrix L can be reused for each degree, N. The nodal
    points, x_k should first be found with cheb_get_nodes. To find the
    N+1 coefficents, c_k, given N+1 nodal values, f_k, evaluated at x_k do:
    
        c = np.dot(f, L)
        
    and to compute the nodal values from c do:
    
        f = np.dot(np.linalg.inv(L), c)
        
    Note that the returned coefficents are ordered in the same way as the 
    coefficents in the numpy.polynomial.chebyshev module, so that can be
    used to evaluate f at points outside of x_k, perform calculus etc.
    """
    interp = np.empty((degree+1,degree+1))
    for j in range(degree+1):
        for k in range(degree+1):
            if j == 0 or j == degree:
                pj = 2
            else:
                pj = 1
            if k == 0 or k == degree:
                pk = 2
            else:
                pk = 1
            interp[j,k] = (2.0/(pj*pk*degree)) * np.cos((j*np.pi*k)/degree)
    return interp
