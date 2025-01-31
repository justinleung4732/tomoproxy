#!/usr/bin/env python
# coding=utf8

from __future__ import absolute_import
from __future__ import print_function

from collections import namedtuple

import numpy as np

import burnman

# We are not going to assume either thermodynamic equilibrium or any
# particular rock type. This means we can vary the phase proportions 
# of bridgmanite, periclase, CaSiO3 perovskite and SiO2, the composition
# of bridgmanite (Al, Fe and Mg - i.e. 2 parameters) and periclase (Mg-
# Fe). We turn bridgmanite into post-perovskite and SiO2 stishovite into 
# seifertite if we cross the phase boundaries. This means we have 6 
# "mineralogical" parameters. We may want to fix some of these!

Properties = namedtuple('Properties', ['density','v_p','v_phi','v_s','K_S',
                                       'G'])

class MineralogicalModel(object):

    def __init__(self, thermo_data="SLB_2022", ppv_mode='basic'):
        if thermo_data == "SLB_2011":
            self.pv = burnman.minerals.SLB_2011.mg_fe_perovskite()
            self.ppv = burnman.minerals.SLB_2011.post_perovskite()
            self.mgo = burnman.minerals.SLB_2011.ferropericlase()
            self.capv = burnman.minerals.SLB_2011.ca_perovskite()
            self.stish = burnman.minerals.SLB_2011.stishovite()
            self.seif = burnman.minerals.SLB_2011.seifertite()
            self.cf = burnman.minerals.SLB_2011.ca_ferrite_structured_phase()
        elif thermo_data == "SLB_2022":
            self.pv = burnman.minerals.SLB_2022.bridgmanite()
            self.ppv = burnman.minerals.SLB_2022.post_perovskite()
            self.mgo = burnman.minerals.SLB_2022.ferropericlase()
            self.capv = burnman.minerals.SLB_2022.capv()
            self.stish = burnman.minerals.SLB_2022.st()
            self.cf = burnman.minerals.SLB_2022.calcium_ferrite_structured_phase()
        else:
            raise ValueError("Thermodynamic database error in Mineralogical Model")
        self.thermo_data = thermo_data

        assert (ppv_mode == 'basic' or ppv_mode == 'none' or ppv_mode == 'two_phase'), \
            "PPv mode error in MineralogicalModel"
        self.ppv_mode = ppv_mode

        # We need a list of references to each endmember
        # so we can modify EOS parameters.
        if thermo_data == "SLB_2011":
            self._mineral_list = [self.pv, self.ppv, self.mgo,
                                self.capv, self.stish, self.seif, self.cf]
            self._endmember_count = [3, 3, 2, None, None, None, 3]
        else:
            self._mineral_list = [self.pv, self.ppv, self.mgo,
                              self.capv, self.stish, self.cf]
            self._endmember_count = [3, 3, 3, None, None, 3]

        # Now build lists of the paramers - mean and error are used
        # for update, current is used as a shortcut for writing the
        # state to file
        self.param_error = []
        self.param_mean = []
        self.errnames = []
        self.paramnames = []
        self.mineralnames = []
        self.endmemberids = []
        self.mineral_pointers = []
        
        if thermo_data == "SLB_2011":
            for mineral, emc in zip(self._mineral_list, self._endmember_count):
                if emc is None:
                    paramnames, means, errnames, errors \
                                            = self._get_mineral_params(mineral)
                    self.param_error.extend(errors)
                    self.param_mean.extend(means)
                    self.errnames.extend(errnames)
                    self.paramnames.extend(paramnames)
                    self.mineralnames.extend([mineral.name]*len(errors))
                    self.endmemberids.extend([None]*len(errors))
                    self.mineral_pointers.extend([mineral]*len(errors))
                else:
                    for emi in range(emc):
                        paramnames, means, errnames, errors =\
                        self._get_mineral_params(mineral.endmembers[emi][0])
                        self.param_error.extend(errors)
                        self.param_mean.extend(means)
                        self.errnames.extend(errnames)
                        self.paramnames.extend(paramnames)
                        self.mineralnames.extend(
                                [mineral.endmembers[emi][0].name]*len(errors))
                        self.endmemberids.extend([emi]*len(errors))
                        self.mineral_pointers.extend([mineral]*len(errors))
            # Current state, so we don't need to go too deep
            self.current_parameters = self.param_mean[:]
    

    def evaluate(self, P, T, Xcapv, Xmgo, Xsio, Xcf, Ypv_al, Ypv_fe, Ymgo_fe, Ymgo_na, Ycf_fe, Ycf_na,
                 Xppv=None, Yppv_al=None, Yppv_fe=None):
        """
        Return a Properties named tuple containing the physical properties
        derived from the mineral model with the specified parameters.
        
        NB: P is in **GPa**, not Pa.
        """
        P = P * 1E9

        if Xcapv < -1e-12:
            Xcapv = 0.0
        if Xmgo < -1e-12:
            Xmgo = 0.0
        if Xsio < -1e-12:
            Xsio = 0.0
        if Xcf < -1e-12:
            Xcf = 0.0
        if Ypv_al < -1e-12:
            Ypv_al = 0.0
        if Ypv_fe < -1e-12:
            Ypv_fe = 0.0
        if Ymgo_fe < -1e-12:
            Ymgo_fe = 0.0
        if Ymgo_na < -1e-12:
            Ymgo_na = 0.0
        if Ycf_fe < -1e-12:
            Ycf_fe = 0.0
        if Ycf_na < -1e-12:
            Ycf_na = 0.0
        if self.ppv_mode == 'two_phase':
            if Xppv < -1e-12:
                Xppv = 0.0
            if Yppv_al < -1e-12:
                Ypv_al = 0.0
            if Yppv_fe < -1e-12:
                Ypv_fe = 0.0            

        self.pv.set_composition([1.0-Ypv_fe-Ypv_al, Ypv_fe, Ypv_al])
        if self.ppv_mode == 'two_phase':
            self.ppv.set_composition([1.0-Yppv_fe-Yppv_al, Yppv_fe, Yppv_al])
        else:
            self.ppv.set_composition([1.0-Ypv_fe-Ypv_al, Ypv_fe, Ypv_al])
        if self.thermo_data == "SLB_2011":
            self.mgo.set_composition([1.0-Ymgo_fe, Ymgo_fe])
        else:
            self.mgo.set_composition([1.0-Ymgo_fe-Ymgo_na, Ymgo_fe, Ymgo_na])
        self.cf.set_composition([1.0-Ycf_fe-Ycf_na, Ycf_fe, Ycf_na])

        if Xsio > 0.0 and self.thermo_data == "SLB_2011":
            self.stish.set_state(P, T)
            self.seif.set_state(P, T)
            if self.stish.molar_gibbs < self.seif.molar_gibbs:
                si_phase = self.stish
            else:
                si_phase = self.seif
        else:
            si_phase = self.stish

        if self.ppv_mode == 'none':
            # No ppv!
            Xpv = 1.0-Xcapv-Xmgo-Xsio-Xcf
            Xppv = 0
        elif self.ppv_mode == 'basic':
            self.pv.set_state(P, T)
            self.ppv.set_state(P, T)
            if self.pv.molar_gibbs < self.ppv.molar_gibbs:
                Xpv = 1.0-Xcapv-Xmgo-Xsio-Xcf
                Xppv = 0
            else:
                Xppv = 1.0-Xcapv-Xmgo-Xsio-Xcf
                Xpv = 0
        elif self.ppv_mode == 'two_phase':
            Xpv = 1.0-Xcapv-Xmgo-Xsio-Xcf-Xppv
        else:
            assert False, "Unknown ppv_mode!"

        rock = burnman.Composite([self.pv, self.ppv, self.mgo, self.capv,
                                 si_phase, self.cf], [Xpv,
                                 Xppv, Xmgo, Xcapv, Xsio, Xcf])

        if False:
            print("current properties:")
            rock.debug_print()
        rho, vp, vphi, vs, K, G = rock.evaluate(['density','v_p','v_phi',
                                                 'v_s','K_S','G'],[P],[T])
        return Properties(density=rho[0], v_p=vp[0], v_phi=vphi[0], 
                          v_s=vs[0], K_S=K[0], G=G[0])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=
              'Evaluate seismic properties for the lower mantle')
    parser.add_argument('T', type=float, help='temperature (K)')
    parser.add_argument('P', type=float, help='pressure (GPa)')
    parser.add_argument('-Xmgo', type=float, default=0.2, 
                        help='volume fraction MgFeO')
    parser.add_argument('-Xcapv', type=float, default=0.05, 
                        help='volume fraction CaSiO3 perovskite')
    parser.add_argument('-Xsio', type=float, default=0.0, 
                        help='volume fraction SiO2')
    parser.add_argument('-Xcf', type=float, default=0.0, 
                        help='volume fraction Ca ferrite')
    parser.add_argument('-Ypvfe', type=float, default=0.0, 
                        help='volume fraction Fe in Mg perovskite')
    parser.add_argument('-Ypval', type=float, default=0.0, 
                        help='volume fraction Al in Mg perovskite')
    parser.add_argument('-Ymgofe', type=float, default=0.0, 
                        help='volume fraction Fe in MgO')
    parser.add_argument('-Ymgona', type=float, default=0.0, 
                        help='volume fraction Na in MgO')
    parser.add_argument('-Ycffe', type=float, default=0.0, 
                        help='volume fraction Fe in Ca ferrite')
    parser.add_argument('-Ycfna', type=float, default=0.0, 
                        help='volume fraction Na in Ca ferrite')
    args = parser.parse_args()

    print("Using burnman version: ", burnman.__version__)
    min_model = MineralogicalModel()

    res = min_model.evaluate(args.P, args.T, args.Xcapv, args.Xmgo,args.Xsio, 
                             args.Ypval, args.Ypvfe, args.Ymgofe, args.Ymgona,
                             args.Ycffe, args.Ycfna)

    print("density: ", res.density, 'kg/m^3')
    print("vp: ", res.v_p, 'm/s')
    print("vs: ", res.v_s, 'm/s')
    
