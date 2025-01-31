#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extra tools for SHTOOLS 

"""
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import pyshtools as shtools

def _get_plmon_max(l,m):
    """
    Approximation to the maximum of normalized associated Legendre function
    
    We simply evaluate PlmON(z) (ON=OrthoNormal) for the chosen l and m for 
    a large number of zs between -1 and 1, and return the maximum value. This 
    version is expected to be accurate to degree 2800. See 
    https://shtools.oca.eu/shtools/www/man/python/pyplmon.html
    https://shtools.oca.eu/shtools/pyplmon.html

    We assume orthonormalized real spherical harmonics without the 
    Condon-Shortley phase.
    """
    plmon = []
    for z in np.linspace(-1.0,1.0,100000):
        plmon.append(shtools.legendre.PlmON(l,
                                 z)[shtools.legendre.PlmIndex(l,m)])
    return np.max(plmon)


def normalized_sh_coefficient(l, m, amplitude=1.0, peak_lon=0):
    """
    Return the normalized coefficents for a simple SH function

    We sometimes want to set a particular SH coefficent (to choose the
    shape of a function) and choose the maximum of the function when 
    expanded in latitude and longitude. For a given value of l and m, 
    an amplitude (which will set the maximum of the function when 
    expanded, defaults to 1.0) and phase (set by peak_lon, defaults to
    0 degrees E of the meridian) this function returns two values, which
    should be set in the SH coefficents to give the chosen pattern. 
    
    We assume orthonormalized real sperical harmonics without the 
    Condon-Shortley phase.
    """
    coeffs = np.zeros((2))

    # We need to calculate the scale factor that we multiply the amplitude
    # by to account for both the normalization, and to make the maximum in 
    # latitude be equal to the amplitude. This amounts to the division by
    # the maximum of the normalised associated Legendre function
    amplitude = amplitude / _get_plmon_max(l,m)
        
    # Now handle the longitude
    if m == 0:
        # Zonal harmonics, don't scale for longitude
        coeffs[0] = amplitude
    else:
        # Sectoral and tesseral harmonics, scale for longitude
        coeffs[0] = amplitude * np.cos(np.radians(l * peak_lon))
        coeffs[1] = amplitude * np.sin(np.radians(l * peak_lon))
        
    return coeffs


def rts_to_sh(rts_coefs):
    """
    Convert between the SH format from the 'sph' file to our internal format

    In '.sph' files used for the SP12RTS tomography model (for example) the 
    spherical harmonic coefficents are complex, fully normalised, with the CS
    phase. For most of our use we use real fully normalised real coefficents 
    without the CS phase. This function does the conversion from 'sph'/'rts' to 
    our normal format.
    """

    inp_shape = np.shape(rts_coefs)
    assert inp_shape[0] == 2, 'rts_coefs must have real and imag parts on 1st dim'
    lmax = inp_shape[1] - 1
    assert inp_shape[2] == lmax + 1, 'must have all ms'

    coefs= shtools.SHCoeffs.from_zeros(lmax, kind='complex', 
                                       normalization='ortho', csphase=-1)

    # We should be able to avoid the loop, but for now...
    for l in range(lmax+1):
        for m in range(l+1):
            coefs.set_coeffs(rts_coefs[0,l,m] - rts_coefs[1,l,m] * 1j, l, m)
            # We don't need to set -m coeffs. minus sign for imaginary part 
            # because real coefficients for sin store negative m degrees,
            # where sin(-m*phi) = -sin(m*phi)

    sh_coefs = coefs.convert(normalization='ortho', csphase=1, kind='real', 
                             check=False).to_array()

     # RTS format multiples non-zero order (m) components by 2
    sh_coefs[:,1:,1:] /= 2

    return sh_coefs


def sh_to_rts(sh_coefs):
    """
    Convert between our SH format and that from from the 'sph' files

    In '.sph' files used for the SP12RTS tomography model (for example) the 
    spherical harmonic coefficents are complex, fully normalised, with the CS
    phase. For moust of our use we use real fully normalised real coefficents 
    without the CS phase. This function does the conversion from our nornal 
    format to the 'sph'/'rts' format.
    """

    inp_shape = np.shape(sh_coefs)
    assert inp_shape[0] == 2, 'sh_coefs must have real and imag parts on 1st dim'
    lmax = inp_shape[1] - 1
    assert inp_shape[2] == lmax + 1, 'must have all ms'

    coefs = shtools.SHCoeffs.from_zeros(lmax, kind='real', 
                                        normalization='ortho', csphase=1)

    # We should be able to avoid the loop, but for now...
    for l in range(lmax+1):
        for m in range(l+1):
            coefs.set_coeffs(sh_coefs[0,l,m], l, m)
            if m != 0:
                # coefs.set_coeffs(sh_coefs[1,l,m], l, -1*m)
                coefs.set_coeffs(-sh_coefs[1,l,m], l, -1*m)
                # Minus sign for imaginary part because real coefficients for 
                # sin store negative m degrees, where sin(-m*phi) = -sin(m*phi)

    complex_coefs = coefs.convert(normalization='ortho', csphase=-1, kind='complex', 
                              check=False).to_array()

    # SHTOOLS stores the two arrays in axis 0 as positive m and negative m shells. We 
    # want to match the RTS format of storing real part in the first array and
    # imaginary aprt in the second array of axis 0
    real = complex_coefs[0].real
    imag = complex_coefs[0].imag

    rts_coefs = np.array([real, imag])

    # RTS format multiples non-zero order (m) components by 2 
    rts_coefs[:,1:,1:] *= 2

    return rts_coefs

    # rts_coefs = coefs.convert(normalization='ortho', csphase=-1, kind='complex', 
    #                           check=False).to_array()
    
    # # rts_coef is complex, but with all imag parts == +/- 0j 
    # return np.real(rts_coefs)