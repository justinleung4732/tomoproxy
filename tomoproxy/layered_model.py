#!/usr/bin/env python
# coding=utf8

from __future__ import print_function
from __future__ import absolute_import

import collections
import io
import six

import numpy as np
import matplotlib.pyplot as plt

import pyshtools as shtools

"""
 LayerData is a 'named tuple' containing a set of information, which can
 be in any form. Elements can be accessed by index or by name.  
 Each layer in a LayeredModel is itself an instance
 of this LayerData type.  So different depths can have different lmax.
"""
LayerData = collections.namedtuple('LayerData', ['depth', 'lmax', 'cilm'])

class LayeredModel(object):

    def __init__(self, init_data, long_format=False, from_string=False):
	# Six is a v2 -> v3 compatibility module. 
        if isinstance(init_data, six.string_types):
            if long_format:
                # Init model from 'long' formatted file
                # (e.g. geoid.ab)
                self.layers = self._read_long_format_file(init_data)
            elif from_string:
                self.layers = self._read_long_format_file(init_data,
                                     from_string=True)
        else:
            self.layers = self._init_zeros(init_data[0], init_data[1])
            self.name = init_data[2]


    def _init_zeros(self, depths, lmaxs):
        """Create empty layers for model"""
        assert (len(depths) == len(lmaxs)), "List lengths must be equal"
        layers = []
        for depth, lmax in zip(depths, lmaxs):
            this_layer = LayerData(depth=depth, lmax=lmax, 
                                   cilm=np.zeros((2,lmax+1,lmax+1)))
            layers.append(this_layer)
        return layers


    def _read_tomography_file(self, fname):
        """Read an HC-formatted (SH) tomography file"""
        layers = []
        with open(fname, 'r') as f:
            num_layers = int(f.readline())
            for i in range(num_layers):
                layer_depth = float(f.readline())
                lmax = int(f.readline())
                # SHTOOLS wants lots of zeros in it's cilm array
                # so keep track of l and m as we step through the 
                # file and bung the data into the right place
                # in a numpy array
                this_layer = LayerData(depth=layer_depth, lmax=lmax, 
                                       cilm=np.zeros((2,lmax+1,lmax+1)))
                for l in range(lmax+1):
                    for m in range(l+1):
                        v1, v2 = f.readline().split()
                        this_layer.cilm[0,l,m] = float(v1)
                        this_layer.cilm[1,l,m] = float(v2)
                layers.append(this_layer)
                    
        return layers


    def _read_long_format_file(self, fname, from_string=False):
        """Read a 'long format' SH expansion.

        This is generally a single layer, but we put it 
        in a layer model anyway.
	See https://github.com/geodynamics/hc/blob/master/README.TXT

	CD: according to sh_power.c inside hc 
	if short format is 0, will expect

	  type lmax shps ilayer nset zlabel ivec
	  A-00 B-00
	  A-10 B-10
	  
	if short format is set, will expect

	  lmax
	  A-00 B-00
	  A-10 B-10
	  ...

	The former is referred to in the README as "long format"; 
	The latter is "short" format

	Note that NEITHER of these are the format of tomography 
	files read in by HC, which have a 3-line header 
	nlayer
	zlabel
	lmax
        """
        layers = []
        if from_string:
            f = io.StringIO(fname)
        else:
            f = open(fname, 'r')

        header = f.readline().split()
        lmax = int(header[0])
        this_layer = int(header[1])
        assert this_layer == 0, "not set up for other layers"
        layer_depth = float(header[2])
        n_layers = int(header[3])
        nrset = int(header[4])
        assert nrset == 1, "Only scalar fields supported"
        sh_type = int(header[5])
        assert sh_type == 0, "Only 'RICKER' supported"

        if n_layers == 1:

            this_layer = LayerData(depth=layer_depth, lmax=lmax,
                                   cilm=np.zeros((2,lmax+1,lmax+1)))
            for l in range(lmax+1):
                for m in range(l+1):
                    v1, v2 = f.readline().split()
                    this_layer.cilm[0,l,m] = float(v1)
                    this_layer.cilm[1,l,m] = float(v2)
            layers.append(this_layer)

        else:
            # Load up the first layer
            this_layer = LayerData(depth=layer_depth, lmax=lmax,
                                   cilm=np.zeros((2,lmax+1,lmax+1)))
            for l in range(lmax+1):
                for m in range(l+1):
                    v1, v2 = f.readline().split()
                    this_layer.cilm[0,l,m] = float(v1)
                    this_layer.cilm[1,l,m] = float(v2)
            layers.append(this_layer)

            for i in range(1, n_layers):
                header = f.readline().split()
                lmax = int(header[0])
                this_layer = int(header[1])
                layer_depth = float(header[2])
                n_layers = int(header[3])
                nrset = int(header[4])
                assert nrset == 1, "Only scalar fields supported"
                sh_type = int(header[5])
                assert sh_type == 0, "Only 'RICKER' supported"
                this_layer = LayerData(depth=layer_depth, lmax=lmax,
                                       cilm=np.zeros((2,lmax+1,lmax+1)))
                for l in range(lmax+1):
                    for m in range(l+1):
                        v1, v2 = f.readline().split()
                        this_layer.cilm[0,l,m] = float(v1)
                        this_layer.cilm[1,l,m] = float(v2)
                layers.append(this_layer)

        f.close()
                
        return layers


    def write_tomography_file(self, filename):
        """Write an HC-formatted (SH) tomography file"""
        with open(filename, 'w') as f:
            f.write("{:d}\n".format(len(self.layers)))
            for layer in self.layers:
                # In the existing files the layer depth looks like an int,
                # but inside HC it is defined as HC_PREC which is a double
                # (e.g. see line 99 of sh_model.c or line 441 of sh_exp.c).
                # We write a float.
                f.write("{:.1f}\n".format(layer.depth))
                f.write("{:d}\n".format(layer.lmax))
                for l in range(layer.lmax+1):
                    for m in range(l+1):
                        f.write(" {:14.7e} {:14.7e}\n".format(
                                                         layer.cilm[0,l,m],
                                                         layer.cilm[1,l,m]))
        

def copy_low_degree(layer_model, new_lmax, min_depth=None):
    """Copy a LayerModel but truncate at a new_lmax

    This copes each layer, but reduces the maximum degree effectivly
    smoothing the model. The smallest lmax in the input layer_model
    must be larger than the new_lmax (which is assigned to each 
    layer in the new model.
    """
    assert isinstance(layer_model, LayeredModel), "Not a LayerModel"
    depths, old_lmaxs = layer_model.get_dimensions()
    assert new_lmax <= old_lmaxs[0], "Must reduce lmax!"
    if min_depth is not None:
        new_depths = []
        new_lmaxs = []
        for i in range(len(depths)):
            if depths[i] > min_depth:
                new_depths.append(depths[i])
                new_lmaxs.append(new_lmax)
    else:
        new_lmaxs = [new_lmax]*len(old_lmaxs) # L max for each depth
        new_depths = depths

    new_name = "Copy of {}".format(layer_model.name)
    new_layered_model =  LayeredModel((new_depths, new_lmaxs, new_name))

    # Copy the needed cilms 
    for i in range(len(new_depths)):
        for l in range(new_lmax+1):
            for m in range(l+1):
                new_layered_model.layers[i].cilm[0,l,m] = \
                    layer_model.layers[i].cilm[0,l,m]
                new_layered_model.layers[i].cilm[1,l,m] = \
                    layer_model.layers[i].cilm[1,l,m]
    return new_layered_model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=
              'Read and plot an "HC" spherical harmonic file')
    parser.add_argument('filename', help='File to read')
    parser.add_argument('-l', '--layer', type=int, default=0,
                        help='layer to plot')
    parser.add_argument('-g', '--geoid', action='store_true',
                        default=False, help='Read a HC geoid filer')
    args = parser.parse_args()

    model = LayeredModel(args.filename, long_format=args.geoid)
