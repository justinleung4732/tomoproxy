#!/usr/bin/env python
# coding=utf8

from __future__ import print_function
from __future__ import absolute_import
#from future.utils import native_str

import functools

import numpy as np

import pyshtools as shtools

from . import layered_model as lm
from . import cheb_tools

# Some constants - we should store these somewhere useful
R_EARTH_KM = 6371.0

# numpy type for GLQ points
NodalPoints = np.dtype([
    ('r', np.float_),
    ('lat', np.float_),
    ('lon', np.float_),
])

class SShell(object):
    """A Spherical Shell stored in spectral space

       We represent a scalar field inside a spherical shell on a basis
       formed from spherical harmonics (for theta and phi) and Chebyshev
       polynomials (for r).
    """

    def __init__(self, spherical_degree, radial_degree, r_min, r_max):
        self.sdegree = spherical_degree
        self.rdegree = radial_degree
        self.r_min = r_min
        self.r_max = r_max
        self.cilm_map = self.cilm_compression_map()
        self.cilm_inverse_map = self.cilm_decompression_map()
        self.coef = np.zeros((self.rdegree+1, len(self.cilm_map)))
        self.radial_nodes = cheb_tools.cheb_get_nodes(self.rdegree,
                                                   [r_min, r_max])
        self.cheb_interp = cheb_tools.cheb_interp_matrix(self.rdegree)
        self.cheb_tonodes = np.linalg.inv(self.cheb_interp)
        zeros, w = shtools.shtools.SHGLQ(self.sdegree)
        self.sh_zeros = zeros
        self.sh_weights = w


    @property
    def cheb_radial_nodes(self):
        """compat with PixShell"""
        return self.radial_nodes


    @property
    def data_array(self):
        """A numpy ndarray representing the field"""
        return self.coef


    @data_array.setter
    def data_array(self, value):
        # Should fail if the size is wrong
        self.coef = value


    def fit_coef_from_layeredmodel(self, layer_model):

        """Initialise coefficients from a LayeredModel

           This avoids the need to re-fit the SH coefficients,
           we only need to do a Chebyshev least squared fit. This
           is a fairly expensive operation compared to loading
           nodal points.
        """
        assert isinstance(layer_model, lm.LayeredModel), \
            "Must provide a LayeredModel instance"
        # Smooth the input model to the spherical harmonic degree
        # on all layers... this will raise an exception if any
        # layer is too corse
        smooth_layer_model = lm.copy_low_degree(layer_model, self.sdegree)

        # Work out how many layers are between rmin and rmax, and
        # store cilm's in our compressed format
        radii = []
        coeff_r = []
        for layer in smooth_layer_model.layers:
            r = R_EARTH_KM - layer.depth
            if (r >= self.r_min and r <= self.r_max):
                radii.append(r)
                coeff_r.append(self.compress_cilm(layer.cilm))
        assert len(radii) >= self.rdegree, "More layers needed for this degree"
        radii = np.array(radii)
        coeff_r = np.array(coeff_r)

        # use the Cheby class to fit the coeffs, stuff them into our
        # big array.
        # FIXME: Not sure if we really need the loop
        radii_scaled = np.polynomial.polyutils.mapdomain(radii, [self.r_min, self.r_max], [-1, 1])
        for i in range(len(self.cilm_map)):
            self.coef[:,i] = np.polynomial.chebyshev.chebfit(radii_scaled, coeff_r[:,i], self.rdegree)


    def compress_cilm(self, cilm_full):
        cilm_short = np.empty(len(self.cilm_map))
        for i, m in enumerate(self.cilm_map):
            cilm_short[i] = cilm_full[m[0], m[1], m[2]]
        return cilm_short


    def expand_cilm(self, cilm_short):
        """Return a set of spherical harmonic coefficients with
           shape == (2,lmax+1,lmax+1) given a set of compressed coefficients of
           shape == ((lmax+1)**2,), as obtained from `compress_cilm()`
        """
        cilm=np.zeros((2,self.sdegree+1,self.sdegree+1))
        for i, val in enumerate(cilm_short):
            cilm[self.cilm_map[i][0], self.cilm_map[i][1],
                 self.cilm_map[i][2]] = cilm_short[i]
        return cilm


    def cilm_compression_map(self):
        cilm_map=[]
        for l in range(self.sdegree+1):
            for m in range(l+1):
                cilm_map.append((0, l, m))
                if l != 0 and m != 0:
                    cilm_map.append((1, l, m))
        return cilm_map


    def cilm_decompression_map(self):
        cilm_inverse_map = np.zeros((2, self.sdegree+1, self.sdegree+1),
                                    dtype=np.int8)
        for i, m in enumerate(self.cilm_map):
            cilm_inverse_map[m[0], m[1], m[2]] = i
        return cilm_inverse_map


    def get_sh_coefs_at_r(self, r, spherical_degree=None):
        """Return the SH coefficents at any radius inside the shell.

           The epanded coefficents in a numpy array rady for use in SH tools
           are returned
        """
        assert r >= self.r_min, "Radius inside inner boundary"
        assert r <= self.r_max, "Radius outside outer boundary"
        # This could be vectorised using chebyshev.chebval... however this
        # does not do the domain mapping so we use the Chebyshev class
        cilm_compressed = np.empty((len(self.cilm_map)))
        for i in range(len(self.cilm_map)):
            # Expensive, use an lru cache
            cheby = get_cheby_function(tuple(self.coef[:,i].tolist()), self.r_min, self.r_max)
            cilm_compressed[i] = cheby(r)
        coefs =  self.expand_cilm(cilm_compressed)
        if spherical_degree is not None:
            coefs = coefs[:,:spherical_degree+1, :spherical_degree+1]
        return coefs


    def set_all_sh_coefs(self, cilm):
        """
        Update the whole model from a set of spherical harmonic coefficents

        These must be provided in a 4D array cilm[ir, i, l, m] where,
        where `i` == 0 gives the real and `i` == 1 the imaginary parts of
        the coefficient at radial node index `ir`, radial order `l` and
        angular order `m`.

        No normalisation is perfomed in this routine.
        """
        assert cilm.shape[0] == self.rdegree + 1, "Wrong size (ri)"
        assert cilm.shape[1] == 2, "Wrong size (i)"
        assert cilm.shape[2] == self.sdegree + 1, "Wrong size (l)"
        assert cilm.shape[3] == self.sdegree + 1, "Wrong size (m)"
 
        cilm_at_rnodes = np.empty((len(self.radial_nodes), 
                                   len(self.cilm_map)))

        for ir, r in enumerate(self.radial_nodes):
            cilm_at_rnodes[ir,:] = self.compress_cilm(cilm[ir, :, :, :])

        for j in range(len(self.cilm_map)):
            self.coef[:,j] = np.dot(cilm_at_rnodes[:,j], self.cheb_interp)
    

def zeros_like(template_shell):
    new_shell = SShell(template_shell.sdegree, template_shell.rdegree,
                       template_shell.r_min, template_shell.r_max)
    return new_shell 


@functools.lru_cache(maxsize=1024)
def get_cheby_function(coefs, r_min, r_max):
    cheby = np.polynomial.chebyshev.Chebyshev(coefs, domain=[r_min, r_max])
    return cheby