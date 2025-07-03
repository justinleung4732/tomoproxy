import glob
import os

import numpy as np
import pyshtools as shtools
import burnman

from . import layered_model as lm
from . import spherical_shell as sh
from . import mineral_model

# Functions
def _normalise_to_prem(spline, spline_depths, prem_depths, normalise = True):
    """
    Calculates a given spline at PREM_depths.

    Parameters
    -------
    spline: array_like (n)
        The values of the spline evaluated at spline_depths.
    spline_depths: array_like (n)
        The depth values at which the spline values correspond to.
    PREM_depths: array_like (k)
        The list of depths used in PREM. PREM_depths should be an array from 
        outwards towards center of the Earth in radius (not depth)
    normalise: bool
        If normalise = True, the spline will be summed up to a total of 1.
    
    Returns
    -------
    new_spline: array_like (k)
        The values of the new spline evaluated at PREM depths.
    """
    if np.diff(spline_depths)[0] > 1:
        spline_depths = 6371 - spline_depths

    new_spline = np.zeros_like(prem_depths)

    for i, d in enumerate(prem_depths):
        loc = np.argwhere(d >= spline_depths)[0]
        new_spline[i] = (spline[loc] - spline[loc-1]) * (d - spline_depths[loc-1]) / \
                        (spline_depths[loc] - spline_depths[loc-1]) + spline[loc-1]

    if normalise:
        new_spline /= np.sum(new_spline)
    return new_spline


# Variables
# Composition
_COMP_OXIDES = {'pyrolite': {'xSiO2': 38.71, 'xAl2O3': 2.22, 'xCaO': 2.94,
                             'xMgO': 49.85, 'xFeO': 6.17, 'xNa2O': 0.11},
               'BMO': {'xSiO2': 40.15, 'xAl2O3': 1.92, 'xCaO': 2.82,
                       'xMgO': 41.98, 'xFeO': 12.90, 'xNa2O': 0.23},
               'MORB': {'xSiO2': 51.75, 'xAl2O3': 8.16, 'xCaO': 13.88,
                        'xMgO': 14.94, 'xFeO': 7.06, 'xNa2O': 2.18},
               'HC': {'xSiO2': 48.87, 'xAl2O3': 11.28, 'xCaO': 10.59,
                      'xMgO': 20.00, 'xFeO': 12.90, 'xNa2O': 1.50}
               }

# SOLA
_SOLA_PATH = "/Users/univ4732/code/lema/data/SOLA_model/"

# Depths
_SOLA_DEPTHS = np.loadtxt(_SOLA_PATH + 'PREM_layers_depths', usecols = (1,2))
_SOLA_DEPTHS = (_SOLA_DEPTHS[:,0] + _SOLA_DEPTHS[:,1])/2
_SOLA_DEPTHS = _SOLA_DEPTHS[::-1] # Invert so depth array goes towards the core

# Spline
_SOLA_SPLINE_VP = np.loadtxt(_SOLA_PATH + 'kernel_vp.csv', delimiter=',',
                             usecols = (0,2), skiprows = 1)
_SOLA_SPLINE_VS = np.loadtxt(_SOLA_PATH + 'kernel_vs.csv', delimiter=',',
                             usecols = (0,2), skiprows = 1)
_SOLA_SPLINE_VP = _normalise_to_prem(_SOLA_SPLINE_VP[:,0], _SOLA_SPLINE_VP[:,1], _SOLA_DEPTHS)
_SOLA_SPLINE_VS = _normalise_to_prem(_SOLA_SPLINE_VS[:,0], _SOLA_SPLINE_VS[:,1], _SOLA_DEPTHS)

# PREM
_PREM_500_DEPTHS = np.loadtxt('/Users/univ4732/code/lema/data/PREM500.csv', delimiter=',',
                              usecols = (0), skiprows = 1)[::-1] / 1000
_GAMMA = np.loadtxt('/Users/univ4732/code/lema/data/PREM500.csv', delimiter=',', usecols = (-1),
                    skiprows = 1)[::-1]
_GAMMA = _normalise_to_prem(_GAMMA, _PREM_500_DEPTHS, _SOLA_DEPTHS, normalise = False)


# Classes
class BdgPPvTwoPhaseRegion():
    """
    A class object that contains the range of pressures that mark the boundaries 
    of the bdg-pPv two phase region, at a given list of temperatures, for a given 
    LLVP composition and pPv stability scenario. This object is needed for
    determining the most effectively phase assemblage in the calculation of 
    equilibrium phases.
    """

    def __init__(self, comp, temperatures=np.arange(1000., 4500., 50.),
                 min_model="SLB_2022", assemblage_type='depleted', save=False,
                 outdir='', verbose=False, imported=False, lowp=None, highp=None):
        """
        Creates an instance of the BdgPPvTwoPhaseRegion object. The instance can
        be created either by importing from a previously calculated two phase region,
        or be created fresh, which will automatically conduct the two phase region 
        calculation.

        Parameters
        -------
        comp: str or dict
            The name of the LLVP composition. Should be either "pyrolite", "BMO",
            "MORB" or "HC". If comp is a dictionary, the composition will be
            prescribed by the list of oxides given in the dictionary.
        temperatures: array_like (n)
            The list of temperatures at which the two phase region is evaluated at.
        min_model: str
            The mineralogical model used. Should either be "SLB_2022" or "SLB_2011".
        assemblage_type: str
            The mineral assemblage type used to calculate the two phase region.
            Should either be "depleted" or "enriched".
        save: bool
            If save = True, the two phase region will be saved as a .txt file.
        outdir: str
            The directory in which to output the two phase region.
        verbose: bool
            Whether to print output updating the user on the status of calculation.
        imported: bool
            Whether or not the two phase region will be imported. If False, the
            two phase region will be automatically calculated.
        """
        assert comp in ['pyrolite', 'BMO', 'MORB', 'HC'], "Not a valid type of composition"
        assert min_model in ['SLB_2011', 'SLB_2022'], "Mineralogical model must either be \
                                                       2022 (SLB 2022) or 2011 (SLB 2011)"

        self.min_model = min_model
        self.temperatures = temperatures
        self.comp = comp
        if comp in ['pyrolite', 'BMO', 'MORB', 'HC']:
            composition = _COMP_OXIDES[comp]
            if comp in ['pyrolite', 'BMO']:
                self.assemblage_type = 'depleted'
            else:
                self.assemblage_type = 'enriched'
        elif isinstance(comp, dict):
            assert assemblage_type in ['depleted', 'enriched'], "Assemblage type must either" \
                                                                 "be depleted or enriched"
            self.assemblage_type = assemblage_type
            composition = comp
            comp = 'Custom'
        else:
            raise TypeError("comp must be either a dictionary containing oxides or a str equal to"\
                            "'pyrolite', 'BMO', 'MORB', 'HC'")

        if imported:
            assert lowp is not None, "Lower phase boundary needed for import"
            assert highp is not None, "Higher phase boundary needed for import"
            self.lowp = lowp
            self.highp = highp
        else:
            self._calculate(composition, save = save, outdir = outdir, verbose = verbose)


    def _calculate(self, composition, save=False, outdir='', verbose=False):
        """
        Calculates the two phase region.

        Parameters
        -------
        comp: dict
            A dictionary containing the list of oxides for a given composition.
        save: bool
            If save = True, the two phase region will be saved as a .txt file.
        outdir: str
            The directory in which to output the two phase region.
        verbose: bool
            Whether to print output updating the user on the status of calculation.
        """
        if self.min_model == "SLB_2011":
            pv = burnman.minerals.SLB_2011.mg_fe_perovskite()
            ppv = burnman.minerals.SLB_2011.post_perovskite()
            fper = burnman.minerals.SLB_2011.ferropericlase()
            cf = burnman.minerals.SLB_2011.ca_ferrite_structured_phase()
            capv = burnman.minerals.SLB_2011.ca_perovskite()
            stish = burnman.minerals.SLB_2011.stishovite()
        elif self.min_model == "SLB_2022":
            pv = burnman.minerals.SLB_2022.bridgmanite()
            ppv = burnman.minerals.SLB_2022.post_perovskite()
            fper = burnman.minerals.SLB_2022.ferropericlase()
            cf = burnman.minerals.SLB_2022.calcium_ferrite_structured_phase()
            capv = burnman.minerals.SLB_2022.capv()
            stish = burnman.minerals.SLB_2022.st()
        else:
            raise ValueError('Mineralogical model must either be SLB2022 or SLB2011')

        composition = burnman.Composition(composition)
        composition.renormalize(unit_type="atomic",
                    normalization_component='total',
                    normalization_amount=100.)

        pressures_pv = np.zeros_like(self.temperatures)
        pressures_ppv = np.zeros_like(self.temperatures)

        for i, t in enumerate(self.temperatures):
            k = 1
            p = 140.e9 # Otherwise need a reference P array that varies with temperature
            while k != 0:
                pv.set_composition([0.88, 0.06, 0.06])
                ppv.set_composition([0.86, 0.12, 0.02])
                cf.set_composition([0.9,0.05,0.05])
                if self.min_model == "SLB_2011":
                    fper.set_composition([0.9, 0.1])
                elif self.min_model == "SLB_2022":
                    fper.set_composition([0.8, 0.1, 0.1])

                if self.assemblage_type == "depleted":
                    if self.min_model == "SLB_2011":
                        assemblage = burnman.Composite([pv, ppv, fper, capv, cf])
                    elif self.min_model == "SLB_2022":
                        assemblage = burnman.Composite([pv, ppv, fper, capv])
                    else:
                        raise ValueError('Mineralogical model must either be SLB2022 or SLB2011')
                elif self.assemblage_type == "enriched":
                    assemblage = burnman.Composite([pv, ppv, stish, capv, cf])
                else:
                    raise ValueError("Assemblage type must either be depleted or enriched")

                assemblage.set_state(p, t)

                equality_constraints = [('T', t), ('phase_fraction', (ppv, 0.0))]
                try:
                    sol,_ = burnman.equilibrate(composition, assemblage, equality_constraints,
                                                store_iterates=False, store_assemblage=True)
                    k = sol.code
                    if verbose:
                        print('pv', t, sol.assemblage.pressure/1e9, sol.assemblage.molar_fractions,
                              sol.assemblage.phases[0].molar_fractions)
                except:
                    k = 1
                    if verbose:
                        print(f'Solver cannot solve with starting pressure {p}, trying next pressure point')
                p -= 10e9

            pressures_pv[i] = sol.assemblage.pressure

            k = 1
            p = 140.e9
            while k > 1e-30:
                pv.set_composition([0.88, 0.06, 0.06])
                ppv.set_composition([0.86, 0.12, 0.02])
                cf.set_composition([0.9,0.05,0.05])
                if self.min_model == "SLB_2011":
                    fper.set_composition([0.9, 0.1])
                elif self.min_model == "SLB_2022":
                    fper.set_composition([0.8, 0.1, 0.1])

                if self.assemblage_type == "depleted":
                    if self.min_model == "SLB_2011":
                        assemblage = burnman.Composite([pv, ppv, fper, capv, cf])
                    elif self.min_model == "SLB_2022":
                        assemblage = burnman.Composite([pv, ppv, fper, capv])
                elif self.assemblage_type == "enriched":
                    assemblage = burnman.Composite([pv, ppv, stish, capv, cf])

                assemblage.set_state(p, t)

                equality_constraints = [('T', t), ('phase_fraction', (pv, 0.0))]
                try:
                    sol,_ = burnman.equilibrate(composition, assemblage, equality_constraints,
                                                store_iterates=False, store_assemblage=True)
                    k = sol.code
                    if verbose:
                        print('ppv', t, sol.assemblage.pressure/1e9, sol.assemblage.molar_fractions,
                              sol.assemblage.phases[1].molar_fractions)
                except:
                    k = 1
                    if verbose:
                        print(f'Solver cannot solve with starting pressure {p}, trying next pressure point')
                p -= 10e9
                if p < 0:
                    print('Solver reached negative pressure, continue')
                    break
            pressures_ppv[i] = sol.assemblage.pressure

        self.lowp = pressures_pv
        self.highp = pressures_ppv

        if save:
            np.savetxt(os.path.join(outdir,
                                    f'ppv_two_phase_boundary_{self.comp}_{self.min_model[-2:]}'),
                       np.array([self.temperatures, self.lowp, self.highp]).T,
                       header='T, lowp, highp')


    @classmethod
    def from_txt(cls, txtfile):
        """
        Imports a previously calculated two phase region, and stores it in a 
        BdgPPvTwoPhaseRegion class object.

        Parameters
        -------
        txtfile: str
            The name of the txtfile that the two phase region calculation is
            stored in.
        """
        txtfile_split = os.path.basename(txtfile).split('_')
        assert txtfile_split[-6:-2] == ['ppv', 'two', 'phase', 'boundary'], \
            'Filename must be in "ppv_two_phase_boundary_[COMP]_[MIN_MODEL]"'
        comp = txtfile_split[-2]
        min_model = f'SLB_20{txtfile_split[-1]}'

        f = open(txtfile, 'r')
        header = f.readline()
        assert header == '# T, lowp, highp\n', "File must have headers T, lowp, highp"

        data = np.loadtxt(txtfile)

        phaseregion = cls(comp, min_model=min_model,
                          imported=True, temperatures=data[:,0],
                          lowp=data[:,1], highp=data[:,2])

        return phaseregion


class PhaseGrid():
    """
    A class object that contains the equilibrium phase assemblage for a given 
    LLVP composition and post-perovskite stability scenario, evaluated with a given
    mineralogical model. The phase assemblaetes stored in this object can also be
    used to evaluate its elastic parameters.
    """

    def __init__(self, phases, t_grid, depth, lon, lat, comp,
                 min_model='SLB_2022', assemblage_type='depleted'):
        """
        Creates an instance of the PhaseGrid object. This object is created after 
        calculating equilibrium phase assemblage from the oxide_to_phase function.
        It can also be created by importing calculated equilibrium assemblages from
        a file or dictionary.

        Parameters
        -------
        phases: str or dict
            If phases is in a str format, it should be the filename of a file to
            where phase data should be imported from. If phases is a dictionary,
            the object will store this dictionary as values for the different phases.
        t_grid: array_like (n, k)
            A 2-D grid of temperatures evaluated at n depths and k lat/lon points.
            The equilibrium phases are evaluated at the same points as the temperature
            grid.
        depth: array_like (n)
            The list of depths of the t_grid model.
        lon: array_like (k)
            The list of longitude points of the t_grid model.
        lat: array_like (k)
            The list of latitude points of the t_grid model.
        comp: str
            The name of the LLVP composition. Should be either "pyrolite", "pyrolite_TC",
            "BMO", "MORB" or "HC".
        min_model: str
            The mineralogical model used. Should either be "SLB_2022" or "SLB_2011".
        assemblage_type: str
            The mineral assemblage type used to calculate the two phase region.
            Should either be "depleted" or "enriched".
        """
        assert min_model in ['SLB_2011', 'SLB_2022'], "Mineralogical model must either be" \
                                                       "2022 (SLB 2022) or 2011 (SLB 2011)"
        assert len(depth) == t_grid.shape[0], "Depth not matching number of rows in" \
                                               "temperature grid"
        assert len(lon) == len(lat), "List of latitudes must be the same length as list" \
                                      "of longitudes"
        assert len(lon) == t_grid.shape[1], "Lon/Lat not matching number of columns in" \
                                             "temperature grid"

        # Properties of class
        self.comp = comp
        self.min_model = min_model

        self.lon = lon
        self.lat = lat
        self.depth = depth
        self.t_grid = t_grid

        # Storage of minerals
        self.phases = {
            'Xcapv_grid': None,
            'Xmgo_grid': None,
            'Xsio_grid': None,
            'Xcf_grid': None,
            'Xppv_grid': None,
            'Ypv_al_grid': None,
            'Ypv_fe_grid': None,
            'Ymgo_fe_grid': None,
            'Ymgo_na_grid': None,
            'Ycf_fe_grid': None,
            'Ycf_na_grid': None,
            'Yppv_al_grid': None,
            'Yppv_fe_grid': None
        }
        self.phase_keys = ['Xcapv_grid', 'Xmgo_grid', 'Xsio_grid', 'Xcf_grid', 'Ypv_al_grid',
                           'Ypv_fe_grid', 'Ymgo_fe_grid', 'Ymgo_na_grid', 'Ycf_fe_grid', 
                           'Ycf_na_grid', 'Xppv_grid', 'Yppv_al_grid', 'Yppv_fe_grid']

        if isinstance(phases, str):
            ftype = phases.split('.')[-1]

            if ftype == 'npz':
                self._from_npz(phases)
            else:
                self._from_txt(phases, assemblage_type = assemblage_type)
        elif isinstance(phases, dict):
            for name in phases:
                assert name + '_grid' in self.phases, "Variable phases contains a phase that \
                                                       is not included in the class PhaseGrid."
            for name in self.phases:
                self.phases[name] = phases[name.replace('_grid', '')]


    def _from_txt(self, txtfile, assemblage_type='depleted'):
        """
        Imports phase data from a .txt file into a PhaseGrid object.

        Parameters
        -------
        phases: str
            The filename of the txtfile to import the phase data from.
        assemblage_type: str
            The mineral assemblage type used to calculate the two phase region.
            Should either be "depleted" or "enriched".
        """
        assert txtfile == f'phases_{self.comp}_{self.min_model[-2:]}', \
            "File must have the name 'phases_COMP_MINMODEL'"

        phases = np.loadtxt(txtfile, skiprows=1)
        dim = (len(self.depth), len(self.lon))
        assert phases.shape[0] == dim[0] * dim[1] ,\
            "Depth or coordinate array length does not match the shape of the phase grid in file."
        assert assemblage_type in ['depleted', 'enriched'], "Assemblage type must either be \
                                                             depleted or enriched"

        if assemblage_type == "depleted":
            self.phases['Xcapv_grid'] = np.reshape(phases[:,2], dim)
            self.phases['Xmgo_grid'] = np.reshape(phases[:,3], dim)
            self.phases['Ypv_al_grid'] = np.reshape(phases[:,4], dim)
            self.phases['Ypv_fe_grid'] = np.reshape(phases[:,5], dim)
            self.phases['Ymgo_fe_grid'] = np.reshape(phases[:,6], dim)
            self.phases['Ymgo_na_grid'] = np.reshape(phases[:,7], dim)
        elif assemblage_type == "enriched":
            self.phases['Xcapv_grid'] = np.reshape(phases[:,2], dim)
            self.phases['Xcf_grid'] = np.reshape(phases[:,3], dim)
            self.phases['Xsio_grid'] = np.reshape(phases[:,4], dim)
            self.phases['Ypv_al_grid'] = np.reshape(phases[:,5], dim)
            self.phases['Ypv_fe_grid'] = np.reshape(phases[:,6], dim)
            self.phases['Ycf_fe_grid'] = np.reshape(phases[:,7], dim)
            self.phases['Ycf_na_grid'] = np.reshape(phases[:,8], dim)


    def _from_npz(self, npz_file):
        """
        Imports phase data from a .npz file into a PhaseGrid object.

        Parameters
        -------
        phases: str
            The filename of the npzfile to import the phase data from.
        """
        assert f'phases_{self.comp}_{self.min_model[-2:]}.npz' in npz_file, \
            "File must have the name 'phases_COMP_MINMODEL.npz'"

        file = np.load(npz_file)

        assert all(phase.shape == self.t_grid.shape for phase in file.values()) ,\
        "Depth or coordinate array length does not match the shape of the phase grid in file."

        for name in self.phases:
            self.phases[name] = file[name.replace('_grid', '')]


    def calculate_ppv_frac(self, py_phases='', comp_grid=None,
                           exclude_llvp=False, threshold=0.6):
        """
        A function that imports equilibrium phase assemblage files and calculates the 
        pPv fraction at each Terra grid point 

        Parameters
        -------
        py_phases: str or dict
            If phases is in a str format, it should be the filename of a file of the
            equilibrium phases of the reference pyrolite composition (of same pPv stability
            scenario and mineralogical model). If phases is a dictionary, it should contain
            the equilibrium phases of the reference pyrolite composition.
        comp_grid: array_like
            The density grid from the TERRA geodynamic model, which should contain values
            describing the fraction of dense material at each grid point (from 0 to 1).
        exclude_llvp: bool
            Used for the 'partppv' pPv stability scenario. If True, the pPv frac in within the
            LLVP
        thershold: float
            The value used to evaluate the location of the LLVPs, which are defined at points
            where X >= threshold.

        Returns
        -------
        ppv_frac: array_like
            An array with values of pPv fraction at each Terra grid point
        """
        if 'pyrolite' not in self.comp:
            assert isinstance(py_phases, PhaseGrid) or \
                    f'phases_pyroliteTC_{self.min_model[-2:]}' in py_phases, \
                    "py_phase needed for compositional non-heterogeneous part of the mantle. Must"\
                    "be either a file with the name phases_pyroliteTC_MINMODEL file or a PhaseGrid"\
                    "object"
            if isinstance(py_phases, PhaseGrid):
                assert py_phases.comp == 'pyroliteTC', "pyroliteTC composition for py_phase needed"
            assert comp_grid is not None, "comp_grid needed for thermochemical compositions"
            assert comp_grid.shape == self.t_grid.shape, "Shape of X must be same as that of" \
                                                          "temperature field."

            if isinstance(py_phases, str):
                py_phases = PhaseGrid(py_phases, self.t_grid, self.depth, self.lon, self.lat,
                                      'pyroliteTC', self.min_model)

            pv = 1 - comp_grid * (self.phases['Xcapv_grid'] + self.phases['Xmgo_grid'] +\
                                  self.phases['Xsio_grid'] + self.phases['Xcf_grid'] +\
                                  self.phases['Xppv_grid']) +\
                (comp_grid-1) * (py_phases.phases['Xcapv_grid'] + py_phases.phases['Xmgo_grid'] +\
                                 py_phases.phases['Xsio_grid'] + py_phases.phases['Xcf_grid'] +\
                                 py_phases.phases['Xppv_grid'])
            ppv = comp_grid * self.phases['Xppv_grid'] +\
                  (1-comp_grid) * py_phases.phases['Xppv_grid']

            ppv_frac = ppv / (pv + ppv)

        else:
            pv = 1 - self.phases['Xcapv_grid'] - self.phases['Xmgo_grid'] -\
                 self.phases['Xsio_grid'] - self.phases['Xcf_grid'] - self.phases['Xppv_grid']
            ppv_frac = self.phases['Xppv_grid'] / (pv + self.phases['Xppv_grid'])

        if exclude_llvp:
            assert comp_grid is not None, "comp_grid needed for partppv scenario pPv fraction" \
                                           "calclulation"
            llvp_not = np.argwhere(comp_grid > threshold)
            ppv_frac[llvp_not[:,0], llvp_not[:,1]] = 0

        return ppv_frac


    def evaluate_elastic(self, ppv_mode, comp_grid=None, py_model=None,
                         save=False, outdir=''):
        """
        Calculating elastic parameters (rho, vp, vphi, vs, K, G,) at each Terra Grid point.

        Parameters
        -------
        ppv_mode: str
            The name of the pPv stability scenario. Should be either "noppv", "ppv" or "partppv".
        comp_grid: array_like
            The density grid from the TERRA geodynamic model, which should contain values
            describing the fraction of dense material at each grid point (from 0 to 1).
        py_model: ElasticGrid object
            Elastic parameters from the reference pyrolite composition model (of same pPv
            stability scenario and mineralogical model).
        save: bool
            If save = True, the elastic parameters will be saved as a .npz file.
        outdir: str
            The directory in which to output the elastic parameters.

        Returns
        -------
        : ElasticGrid object
            The calculated elastic parameters for this PhaseGrid object.
        """

        if "pyrolite" not in self.comp:
            assert isinstance(py_model, ElasticGrid), "Reference thermal (pyrolite) model needed" \
                                                       "at points outside LLVPs"
            assert py_model.comp == "pyroliteTC", "TC pyrolite elastic model needs to be used"
            assert comp_grid is not None, "A composition grid must be specified for" \
                                           "thermochemical models"

        for name, phase in self.phases.items():
            if phase is None:
                self.phases[name] = np.zeros_like(self.t_grid)
            print(f"Phase {name.replace('grid', '')} is empty, replacing with array of 0s")

        minmod = mineral_model.MineralogicalModel(thermo_data = self.min_model, ppv_mode=ppv_mode)

        pressures = burnman.seismic.PREM().pressure(self.depth * 1000.0) / 1E9

        rho_grid = np.zeros_like(self.t_grid)
        vp_grid = np.zeros_like(self.t_grid)
        vphi_grid = np.zeros_like(self.t_grid)
        vs_grid = np.zeros_like(self.t_grid)
        k_grid = np.zeros_like(self.t_grid)
        g_grid = np.zeros_like(self.t_grid)

        for i, p in enumerate(pressures):
            print(f'Analysing depth {self.depth[i]} km')

            if isinstance(comp_grid, np.ndarray):
                nz_id = np.nonzero(comp_grid[i])[0]
                # find unique pairings of (X,T)
                grid_unique, unique_id, indices = np.unique(np.vstack([self.t_grid[i, nz_id],
                                                                       comp_grid[i, nz_id]]).T,
                                                            axis=0,
                                                            return_index=True,
                                                            return_inverse=True)
                grid_unique = grid_unique[:,0] # only keep the T values
            else:
                nz_id = np.arange(0, len(self.t_grid[i]))
                grid_unique, unique_id, indices = np.unique(self.t_grid[i], return_index=True,
                                                            return_inverse=True)

            print(f'{100*len(grid_unique)/len(self.t_grid[i])}% points to evaluate')
            rho = np.zeros_like(grid_unique)
            vp = np.zeros_like(grid_unique)
            vphi = np.zeros_like(grid_unique)
            vs = np.zeros_like(grid_unique)
            k = np.zeros_like(grid_unique)
            g = np.zeros_like(grid_unique)

            for j, temp in enumerate(grid_unique):
                phase_list = []
                for key in self.phase_keys:
                    phase_list.append(self.phases[key][i,nz_id[unique_id[j]]])
                rho[j], vp[j], vphi[j], vs[j], k[j], g[j] = \
                    minmod.evaluate(p, temp, *phase_list)

            rho_grid[i, nz_id] = rho[indices]
            vp_grid[i, nz_id] = vp[indices]
            vphi_grid[i, nz_id] = vphi[indices]
            vs_grid[i, nz_id] = vs[indices]
            k_grid[i, nz_id] = k[indices]
            g_grid[i, nz_id] = g[indices]

        if "pyrolite" not in self.comp:
            nz_id = np.nonzero(comp_grid)
            rho_grid = (1-comp_grid)[nz_id] * py_model.rho_grid[nz_id] +\
                        comp_grid[nz_id] * rho_grid[nz_id]
            vp_grid = (1-comp_grid)[nz_id] * py_model.vp_grid[nz_id] +\
                        comp_grid[nz_id] * vp_grid[nz_id]
            vphi_grid = (1-comp_grid)[nz_id] * py_model.vphi_grid[nz_id] +\
                        comp_grid[nz_id] * vphi_grid[nz_id]
            vs_grid = (1-comp_grid)[nz_id] * py_model.vs_grid[nz_id] +\
                        comp_grid[nz_id] * vs_grid[nz_id]
            k_grid = (1-comp_grid)[nz_id] * py_model.k_grid[nz_id] +\
                        comp_grid[nz_id] * k_grid[nz_id]
            g_grid = (1-comp_grid)[nz_id] * py_model.g_grid[nz_id] +\
                        comp_grid[nz_id] * g_grid[nz_id]

        if save:
            np.savez(outdir + f"elastic_{self.comp}_{self.min_model[-2:]}_two_phase",
                    rho = rho_grid,
                    vp = vp_grid,
                    vphi = vphi_grid,
                    vs = vs_grid,
                    k = k_grid,
                    g = g_grid)

        return ElasticGrid(self.depth, self.lon, self.lat, rho_grid, vp_grid,
                           vphi_grid, vs_grid, k_grid, g_grid)


class ElasticGrid():
    """
    A class object that stores the elastic parameters calculated from a equilibrium
    phase assemblage.
    """

    def __init__(self, depth, lon, lat, rho_grid=None, vp_grid=None,
                 vphi_grid=None, vs_grid=None,
                 k_grid=None, g_grid=None):
        """
        Creates an instance of the ElasticGrid object. This object is created from
        evaluating the elastic parameters of a PhaseGrid object, or imported from
        a file.

        Parameters
        -------
        depth: array_like (n)
            The list of depths of the t_grid model.
        lon: array_like (k)
            The list of longitude points of the t_grid model.
        lat: array_like (k)
            The list of latitude points of the t_grid model.
        rho_grid: array_like (n,k)
            Density values evaluated at each depth and lat/lon point.
        vp_grid: array_like (n,k)
            Compressional-wave velocities evaluated at each depth and lat/lon point.
        vphi_grid: array_like (n,k)
            Bulk-sound velocities values evaluated at each depth and lat/lon point.
        vs_grid: array_like (n,k)
            Shear-wave velocities values evaluated at each depth and lat/lon point.
        k_grid: array_like (n,k)
            Bulk moduli values evaluated at each depth and lat/lon point.
        g_grid: array_like (n,k)
            Shear moduli values evaluated at each depth and lat/lon point.
        """
        assert len(lon) == len(lat), "List of latitudes must be the same length as list of \
                                      longitudes"
        self.lon = lon
        self.lat = lat
        self.depth = depth

        self.rho_grid = rho_grid
        self.vp_grid = vp_grid
        self.vphi_grid = vphi_grid
        self.vs_grid = vs_grid
        self.k_grid = k_grid
        self.g_grid = g_grid


    @classmethod
    def from_file(cls, fileloc, comp, ppv_model_type, depth, lon, lat,
                  min_model='SLB_2022', comp_grid=None, threshold=0.6):
        """
        Imports elastic parameters from a file and creates an ElasticGrid instance.

        Parameters
        -------
        fileloc: str
            The filename of the npzfile to import the phase data from.
        comp: str
            The name of the LLVP composition. Should be either "pyrolite",
            "pyrolite_TC", "BMO", "MORB" or "HC".
        ppv_model_type: str
            The name of the pPv stability scenario. Should be either "noppv", "ppv" or "partppv".
        depth: array_like (n)
            The list of depths of the t_grid model.
        lon: array_like (k)
            The list of longitude points of the t_grid model.
        lat: array_like (k)
            The list of latitude points of the t_grid model.
        min_model: str
            The mineralogical model used. Should either be "SLB_2022" or "SLB_2011".
        comp_grid: array_like (n,k)
            The density grid from the TERRA geodynamic model, which should contain values
            describing the fraction of dense material at each grid point (from 0 to 1).
        thershold: float
            The value used to evaluate the location of the LLVPs, which are defined at points
            where comp_grid >= threshold.
        """
        assert comp in ['pyrolite', 'pyroliteTC', 'BMO', 'MORB', 'HC'], \
            "Not a valid type of composition"
        assert len(lon) == len(lat), \
            "List of latitudes must be the same length as list of longitudes"
        assert min_model in ['SLB_2011', 'SLB_2022'], \
            "Mineralogical model must either be 2022 (SLB 2022) or 2011 (SLB 2011)"
        if ppv_model_type == 'partppv':
            assert isinstance(comp_grid, np.ndarray),\
            "Composition grid must be defined before importing partial ppv type"

        if ppv_model_type == 'ppv' or ppv_model_type == 'partppv':
            file = f'elastic_{comp}_{min_model[-2:]}_two_phase.npz'
        elif ppv_model_type == 'noppv':
            file = f'elastic_{comp}_{min_model[-2:]}_none.npz'
        else:
            raise ValueError('Not a valid type of ppv model')

        elastic = np.load(fileloc+file)

        rho_grid = elastic['rho']
        vp_grid = elastic['vp']
        vphi_grid = elastic['vphi']
        vs_grid = elastic['vs']
        k_grid = elastic['k']
        g_grid = elastic['g']

        if ppv_model_type == 'partppv':
            llvp = np.argwhere(comp_grid >= threshold)
            file_llvp = f'elastic_{comp}_{min_model[-2:]}_none.npz'
            elastic_llvp = np.load(fileloc+file_llvp)

            rho_grid[llvp[:,0], llvp[:,1]] = elastic_llvp['rho'][llvp[:,0], llvp[:,1]]
            vp_grid[llvp[:,0], llvp[:,1]] = elastic_llvp['vp'][llvp[:,0], llvp[:,1]]
            vphi_grid[llvp[:,0], llvp[:,1]] = elastic_llvp['vphi'][llvp[:,0], llvp[:,1]]
            vs_grid[llvp[:,0], llvp[:,1]] = elastic_llvp['vs'][llvp[:,0], llvp[:,1]]
            k_grid[llvp[:,0], llvp[:,1]] = elastic_llvp['k'][llvp[:,0], llvp[:,1]]
            g_grid[llvp[:,0], llvp[:,1]] = elastic_llvp['g'][llvp[:,0], llvp[:,1]]

        return cls(depth, lon, lat, rho_grid, vp_grid,
                   vphi_grid, vs_grid, k_grid, g_grid)


    def to_continuous_param(self, r_deg=20, sph_deg=8, save=False,
                            outdir='', filename=''):
        """
        Applies a chebyshev spline to the differnet depth layers, and reparameterises lat/lon
        points into spherical harmonics. Returns the parameterisation as a RawSeismicModel
        object.

        Parameters
        -------
        rdeg: int
            The maximum radial degree used to paramerise the chebyshev spline.
        sph_deg: int
            The maximum spherical harmonic degree to parameterise the lateral space.
        save: bool
            If save = True, the continuous parameterisation will be saved as 
            HC-formatted (SH) tomography file
        outdir: str
            The directory in which to output the continuous parameterisation.
        filename: str
            The filename in which to store the continuous parameterisation.

        Returns
        -------
        : RawSeismicModel object
            Seismic velcoities in chebyshev splines for depth and spherical harmonics
            laterally.
        """
        print(f"Converting model {filename}")
        input_data = [self.depth, sph_deg * np.ones(len(self.depth), dtype='int'), 'V']
        vp_layer = lm.LayeredModel(input_data)
        vs_layer = lm.LayeredModel(input_data)
        vphi_layer = lm.LayeredModel(input_data)

        for i, _ in enumerate(self.depth):
            print(f"Converting layer {i}")
            cilm_vp, _ = shtools.expand.SHExpandLSQ(self.vp_grid[i], self.lat, self.lon,
                                                    lmax=sph_deg, norm=4, csphase=1)
            cilm_vs, _ = shtools.expand.SHExpandLSQ(self.vs_grid[i], self.lat, self.lon,
                                                    lmax=sph_deg, norm=4, csphase=1)
            cilm_vphi, _ = shtools.expand.SHExpandLSQ(self.vphi_grid[i], self.lat, self.lon,
                                                    lmax=sph_deg, norm=4, csphase=1)
            vp_layer.layers[i].cilm[:,:,:]= cilm_vp
            vs_layer.layers[i].cilm[:,:,:]= cilm_vs
            vphi_layer.layers[i].cilm[:,:,:]= cilm_vphi

        if save:
            vp_layer.write_tomography_file(os.path.join(outdir, 'SH_' + filename + '_Vp'))
            vs_layer.write_tomography_file(os.path.join(outdir, 'SH_' + filename + '_Vs'))
            vphi_layer.write_tomography_file(os.path.join(outdir, 'SH_' + filename + '_Vc'))

        return RawSeismicModel(vp_layer, vs_layer, vphi_layer, r_deg)


class RawSeismicModel():
    """
    An object that stores raw (unfiltered) seismic velocities in chebyshev splines 
    for depth and spherical harmonics laterally.
    """
    def __init__(self, vp, vs, vphi, r_deg):
        """
        Creates an instance of the RawSeismicModel object. This object is created 
        from calculating the continuous parameterisation from a ElasticGrid object,
        or imported from a file.

        Parameters
        -------
        Vp: LayeredModel class object
            The spherical harmonic coefficients of a Vp model at specific layer depths.
        Vs: LayeredModel class object
            The spherical harmonic coefficients of a Vs model at specific layer depths.
        Vc: LayeredModel class object
            The spherical harmonic coefficients of a Vc model at specific layer depths.
        rdeg: int
            The maximum radial degree used to paramerise the chebyshev spline.
        """
        assert isinstance(vp, lm.LayeredModel), "vp needs to be a LayeredModel instance"
        assert isinstance(vs, lm.LayeredModel), "vs needs to be a LayeredModel instance"
        assert isinstance(vphi, lm.LayeredModel), "vphi needs to be a LayeredModel instance"

        self.lmax = vp.layers[0].lmax
        self.rdeg = r_deg

        self._to_sshell(vp, vs, vphi)


    @classmethod
    def from_file(cls, r_deg, fileloc, comp, ppv_model_type, min_model='', seismic_model=''):
        """
        Creates an instance of the RawSeismicModel object. This object is created 
        from calculating the continuous parameterisation from a ElasticGrid object,
        or imported from a file.

        Parameters
        -------
        rdeg: int
            The maximum radial degree used to paramerise the chebyshev spline.
        fileloc: str
            A directory of where the LayeredModels of the thermal and thermochemical models 
            are stored.
        comp: str
            The name of the LLVP composition. Should be either "pyrolite",
            "pyrolite_TC", "BMO", "MORB" or "HC".
        ppv_model_type: str
            The name of the pPv stability scenario. Should be either "noppv", "ppv" or "partppv".
        min_model: str
            The mineralogical model used. Should either be "SLB_2022" or "SLB_2011".
        seismic_model: str
            The name of the seismic model used in the filename.
        """
        assert comp in ['pyrolite', 'pyroliteTC', 'BMO', 'MORB', 'HC'], \
            "Not a valid type of composition"
        assert ppv_model_type in ['noppv', 'ppv', 'partppv'], \
            "Not a valid type of ppv_model"
        assert min_model in ['SLB_2011', 'SLB_2022'], \
            "Mineralogical model must either be 2022 (SLB 2022) or 2011 (SLB 2011)"

        vp_layer = lm.LayeredModel(f'{fileloc}SH_{seismic_model}_{comp}_'\
                                   f'{ppv_model_type}_{min_model[-2:]}_Vp')
        vs_layer = lm.LayeredModel(f'{fileloc}SH_{seismic_model}_{comp}_'\
                                   f'{ppv_model_type}_{min_model[-2:]}_Vs')
        vphi_layer = lm.LayeredModel(f'{fileloc}SH_{seismic_model}_{comp}_'\
                                     f'{ppv_model_type}_{min_model[-2:]}_Vc')

        return cls(vp_layer, vs_layer, vphi_layer, r_deg)


    def to_sola(self):
        """
        Stores the coefficients in a format in that of Restelli et al. (2023) before 
        tomographic filtering (spherical degree 8, PREM layer depths). Returns the
        filtered tomography model as a SOLAShell object.
        """
        assert self.lmax >= 8, "Spherical degree is not high enough to create SOLAShell"
        vp = np.zeros((len(_SOLA_DEPTHS), 2, 9, 9))
        vs = np.zeros_like(vp)
        vphi = np.zeros_like(vs)

        for i, d in enumerate(_SOLA_DEPTHS):
            if d < self.vp.r_min or d > self.vp.r_max:
                continue
            vp[i] = self._abs_to_rel_velocity(self.vp.get_sh_coefs_at_r(d)[:, :9, :9])
            vs[i] = self._abs_to_rel_velocity(self.vs.get_sh_coefs_at_r(d)[:, :9, :9])
            vphi[i] = self._abs_to_rel_velocity(self.vphi.get_sh_coefs_at_r(d)[:, :9, :9])

            # Set odd degress to 0
            vp[i,:,1::2] = 0
            vs[i,:,1::2] = 0
            vphi[i,:,1::2] = 0

        return SOLAShell(vp, vs, vphi)


    def _to_sshell(self, vp, vs, vphi):
        """
        Applies a chebyshev spline for the different seismic velocities.
        Parameters
        -------
        vp: LayeredModel class object
            The spherical harmonic coefficients of a Vp model at specific layer depths.
        vs: LayeredModel class object
            The spherical harmonic coefficients of a Vs model at specific layer depths.
        vphi: LayeredModel class object
            The spherical harmonic coefficients of a Vc model at specific layer depths.
        """
        assert isinstance(vp, lm.LayeredModel), "vp needs to be a LayeredModel instance"
        assert isinstance(vs, lm.LayeredModel), "vs needs to be a LayeredModel instance"
        assert isinstance(vphi, lm.LayeredModel), "vphi needs to be a LayeredModel instance"

        self.vp = sh.SShell(spherical_degree=self.lmax, radial_degree=self.rdeg,
                            r_min=6371.0-vp.layers[-1].depth, r_max=6371.0-vp.layers[0].depth)
        self.vs = sh.zeros_like(self.vp)
        self.vphi = sh.zeros_like(self.vp)

        # Read layered model into spherical shells
        self.vp.fit_coef_from_layeredmodel(vp)
        self.vs.fit_coef_from_layeredmodel(vs)
        self.vphi.fit_coef_from_layeredmodel(vphi)


    @staticmethod
    def _abs_to_rel_velocity(coefs):
        """
        Calculates the relative velocities as percentages from the 1-D average.
        Parameters
        -------
        coefs: array_like (2,n,n)
            Spherical harmonic coefficients of absolute velocities.
        """
        coefs /= coefs[0,0,0] / (2.0 * np.sqrt(np.pi))
        coefs *= 100
        coefs[0,0,0] = 0
        return coefs


class SOLAShell():
    """
    Stores the coefficients in a format in that of Restelli et al. (2023) before 
    tomographic filtering (spherical degree 8, PREM layer depths). Returns the
    filtered tomography model as a SOLAShell object. The coefficients can be filtered
    with the resolving kernel, which can then be quantitatively compared with 
    the model of Restelli et al. (2023).
    """
    def __init__(self, vp=None, vs=None, vphi=None,
                 vp_err=None, vs_err=None, vphi_err=None):
        """
        Creates an instance of the SOLAShell object. If Vphi is not given,
        it is calculated from Vs and Vp coefficients using the gamma scaling
        profile given from PREM.

        Parameters
        -------
        vp: array_like (96, 2, 9, 9)
            Compressional-wave velocities in spherical harmonics, evaluated at PREM
            depths up to degree 8.
        vs: array_like (96, 2, 9, 9)
            Shear-wave velocities in spherical harmonics, evaluated at PREM depths
            up to degree 8.
        vphi: array_like (96, 2, 9, 9)
            Bulk-sound velocities in spherical harmonics, evaluated at PREM depths
            up to degree 8.
        vp_err: array_like (96, 2, 9, 9)
            Compressional-wave velocities uncertainties in spherical harmonics, 
            evaluated at PREM depths up to degree 8.
        vs_err: array_like (96, 2, 9, 9)
            Shear-wave velocities uncertainties in spherical harmonics, 
            evaluated at PREM depths up to degree 8.
        vphi_err: array_like (96, 2, 9, 9)
            Bulk-sound-wave velocities uncertainties in spherical harmonics, 
            evaluated at PREM depths up to degree 8.
        """
        self.depths = _SOLA_DEPTHS
        self.lmax = 8
        self.filtered = {'vp': False,
                         'vs': False,
                         'vphi': False}

        self.vp = None
        self.vs = None
        self.vphi = None

        self.vp_err = None
        self.vs_err = None
        self.vphi_err = None

        if vp is not None:
            self.update_velocities('vp', vp)

        if vs is not None:
            self.update_velocities('vs', vs)

        if vphi is not None:
            self.update_velocities('vphi', vphi)

        if vp_err is not None:
            self.update_velocity_errors('vp', vp_err)

        if vs_err is not None:
            self.update_velocity_errors('vs', vs_err)

        if vphi_err is not None:
            self.update_velocity_errors('vphi', vphi_err)

        if vp is not None and vs is not None and vphi is None:
            self._calculate_vphi()


    def update_velocities(self, v_type, velocity):
        """
        Updates the velocity array, specified by v_type.

        Parameters
        -------
        v_type: str
            The type of velocity considered. Must either be 'vp', 'vs' or 'vphi'.
        velocity: array_like (96, 2, 9, 9)
            Velocities in spherical harmonics, evaluated at PREM depths up to degree
            8.
        """
        assert self.filtered[v_type] is False, "Cannot update velocities after filtering"
        assert velocity.shape == ((len(self.depths), 2, self.lmax+1, self.lmax+1)), \
            "Wrong shape for velocity array"
        assert v_type in ['vp', 'vs', 'vphi'], "v_type must be 'vp', 'vs' or 'vphi'"

        if v_type == 'vp':
            self.vp = velocity
        elif v_type == 'vs':
            self.vs = velocity
        elif v_type == 'vphi':
            self.vphi = velocity


    def update_velocity_errors(self, err_type, error):
        """
        Updates the velocity uncertainty array, specified by err_type.

        Parameters
        -------
        err_type: str
            The type of velocity uncertainty considered. Must either be 'vp', 'vs'
            or 'vphi'.
        error: array_like (96, 2, 9, 9)
            Velocity errors in spherical harmonics, evaluated at PREM depths up to 
            degree 8.
        """
        assert self.filtered[err_type] is False, "Cannot update velocities after filtering"
        assert error.shape == ((len(self.depths), 2, self.lmax+1, self.lmax+1)), \
            "Wrong shape for velocity array"
        assert err_type in ['vp', 'vs', 'vphi'], "err_type must be 'vp', 'vs' or 'vphi'"

        if err_type == 'vp':
            self.vp_err = error
        elif err_type == 'vs':
            self.vs_err = error
        elif err_type == 'vphi':
            self.vphi_err = error


    @classmethod
    def from_directory(cls, directory):
        """
        Upload the tomography model from Restelli et al. (2023), which
        stores the coefficients in different files and folders based on
        spherical harmonic degree and order.

        Parameters
        -------
        directory: str
            The directory where the coefficients are stored.
        """
        data_files = glob.glob(directory + '/**/mk**.txt', recursive = True)

        # Storage for raw coefficients at PREM depth layers
        vp_raw = np.zeros((len(_SOLA_DEPTHS), 2, 9, 9))
        vs_raw = np.zeros_like(vp_raw)
        vp_err_raw = np.zeros_like(vp_raw)
        vs_err_raw = np.zeros_like(vp_raw)

        for _, deg_file in enumerate(data_files):
            data = np.loadtxt(deg_file)

            # For 0r, 1r, 1i, 2r, 2i...
            info = deg_file.split('/')[-3].split('_')
            v_type = info[0]
            row = int(info[1][1:])
            col = int(np.floor(int(info[2][1:])/2))
            re_im = int(info[2][1:])%2
            #adjust for 0
            if int(info[2][1:]) == 1:
                re_im = 0

            if v_type == 'vp':
                vp_raw[:, re_im, row, col] = data[:,1]
                vp_err_raw[:,re_im, row, col] = data[:,2]

            elif v_type == 'vs':
                vs_raw[:, re_im, row, col] = data[:,1]
                vs_err_raw[:,re_im, row, col] = data[:,2]

        return cls(vp = vp_raw, vs = vs_raw, vp_err = vp_err_raw, vs_err = vs_err_raw)


    def apply_kernel(self):
        """
        Applies the resolution kernel to all the velocities and uncertainties 
        to obtain a filtered version of the tomography model.
        """
        if getattr(self, 'vphi') is not None:
            self._apply_individual_kernel('vphi')
        if getattr(self, 'vp') is not None:
            self._apply_individual_kernel('vp')
        if getattr(self, 'vs') is not None:
            self._apply_individual_kernel('vs')


    def _apply_individual_kernel(self, velocity):
        """
        Inner function that applies the resolution kernel to a specific velocity
        and its uncertainties.
        """
        assert velocity in ['vp', 'vs', 'vphi'], "Velocity must be 'vp', 'vs' or 'vphi'"
        v_err = velocity + '_err'
        assert getattr(self, velocity) is not None, \
        "Kernel cannot be applied without velocities"


        if velocity == 'vp':
            spline = _SOLA_SPLINE_VP
        elif velocity == 'vs':
            spline = _SOLA_SPLINE_VS
        elif velocity == 'vphi':
            spline = (_SOLA_SPLINE_VP + _SOLA_SPLINE_VS) / 2
        else:
            raise ValueError("Velocity must be 'vp', 'vs' or 'vphi'")

        setattr(self, velocity, np.average(getattr(self, velocity), axis = 0, weights = spline))
        if getattr(self, v_err) is not None:
            setattr(self, v_err, np.average(getattr(self, v_err), axis = 0, weights = spline))
        setattr(self, f'spline_{velocity}', spline)
        self.filtered[velocity] = True


    def _calculate_vphi(self):
        """
        Calculates Vphi profile based on Vp, Vs and the PREM gamma profile.
        """
        assert self.vp is not None, 'Vp is empty'
        assert self.vs is not None, 'Vs is empty'
        assert self.vphi is None, 'Vphi should be empty'

        # Creating v_phi
        self.vphi = np.zeros_like(self.vp)
        if self.vp_err is not None and self.vs_err is not None and self.vphi_err is None:
            include_err = True
            self.vphi_err = np.zeros_like(self.vp)
        else:
            include_err = False
            self.vphi_err = np.zeros_like(self.vp)

        self.vphi = (self.vp - _GAMMA[:, None, None, None] * self.vs) / \
            (1 - _GAMMA[:, None, None, None])
        if include_err:
            self.vphi_err = (self.vp_err - _GAMMA[:, None, None, None] * self.vs_err) / \
                 (1 - _GAMMA[:, None, None, None])


# Functions
def oxide_to_phase(t_grid, depth, lon, lat, comp, phase_boundary_reference, comp_grid = 0,
                   min_model = 'SLB_2022', assemblage_type = 'depleted',
                   save = False, outdir = '', verbose = False):
    """
    Calculates the equilibrium phase assemblage at depth and temperature points
    in a geodynamic model, for a given composition using a given mineralogical
    model. The function calculates the phase assemblages by pressure (or depth).
    At each pressure, we take the temperatures associated with the boundaries
    of the two phase region, and we separate the list of points at each pressure
    to a bdg-only, pPv-only, or bdg+pPv region. This speeds up the calculation,
    as the solver is more stable when one of bdg or pPv is included in the 
    assemblage and this method minimises the number of points we evaluate
    assemblages with both bdg and pPv.

    Parameters
    -------
    t_grid: array_like (n, k)
        A 2-D grid of temperatures evaluated at n depths and k lat/lon points.
        The equilibrium phases are evaluated at the same points as the temperature
        grid.
    depth: array_like (n)
        The list of depths of the t_grid model.
    lon: array_like (k)
        The list of longitude points of the t_grid model.
    lat: array_like (k)
        The list of latitude points of the t_grid model.
    comp: str
        The name of the LLVP composition. Should be either "pyrolite", "pyrolite_TC",
        "BMO", "MORB" or "HC".
    phase_boundary_reference: BdgPPVTwoPhaseRegion
        The two phase region reference used to determine which assemblage to
        optimise for at a specific pressure-temperature point.
    comp_grid: array_like
        The density grid from the TERRA geodynamic model, which should contain values
        describing the fraction of dense material at each grid point (from 0 to 1).
    min_model: str
        The mineralogical model used. Should either be "SLB_2022" or "SLB_2011".
    assemblage_type: str
        The mineral assemblage type used to calculate the two phase region.
        Should either be "depleted" or "enriched".
    save: bool
        If save = True, the elastic parameters will be saved as a .npz file.
    outdir: str
        The directory in which to output the equilibrium phase calculation.
    verbose: bool
        Whether to print output updating the user on the status of calculation.
    
    Returns
    -------
    : PhaseGrid object
        Equilibrium phase assemblages evaluated at the pressure and temperature
        points of the geodynamic model.
    """
    assert len(depth) == t_grid.shape[0], "Depth not matching number of rows in temperature grid"
    assert len(lon) == len(lat), "List of latitudes must be the same length as list of longitudes"
    assert len(lon) == t_grid.shape[1], "Lon/Lat not matching number of columns in temperature grid"
    assert min_model in ['SLB_2011', 'SLB_2022'], \
        "Mineralogical model must either be 2022 (SLB 2022) or 2011 (SLB 2011)"
    assert isinstance(phase_boundary_reference, BdgPPvTwoPhaseRegion), \
        "Phase boundary reference must be BdgPPvTwoPhaseRegion Object."

    pressures = burnman.seismic.PREM().pressure(depth * 1000.0)

    if comp in ['pyrolite', 'BMO', 'MORB', 'HC']:
        composition = _COMP_OXIDES[comp]
        if comp in ['pyrolite', 'BMO']:
            assemblage_type = 'depleted'
        else:
            assemblage_type = 'enriched'
    elif comp == 'pyroliteTC':
        composition = _COMP_OXIDES['pyrolite']
        assemblage_type = 'depleted'
    elif isinstance(comp, dict):
        assert assemblage_type in ['depleted', 'enriched'], \
            "Assemblage type must either be depleted or enriched"
        composition = comp
        comp = 'Custom'
    else:
        raise TypeError("comp must be either a dictionary containing oxides or a str equal to" \
                        "'pyrolite', 'BMO', 'MORB', 'HC'")

    assert phase_boundary_reference.comp == comp, \
        "Phase boundary reference composition must match input composition"

    composition = burnman.Composition(composition)
    composition.renormalize(unit_type="atomic",
                  normalization_component='total',
                  normalization_amount=100.)

    phases = {
        'Xcapv_grid': np.empty_like(t_grid),
        'Xmgo_grid': np.empty_like(t_grid),
        'Xsio_grid': np.empty_like(t_grid),
        'Xcf_grid': np.empty_like(t_grid),
        'Xppv_grid': np.empty_like(t_grid),
        'Ypv_al_grid': np.empty_like(t_grid),
        'Ypv_fe_grid': np.empty_like(t_grid),
        'Ymgo_fe_grid': np.empty_like(t_grid),
        'Ymgo_na_grid': np.empty_like(t_grid),
        'Ycf_fe_grid': np.empty_like(t_grid),
        'Ycf_na_grid': np.empty_like(t_grid),
        'Yppv_al_grid': np.empty_like(t_grid),
        'Yppv_fe_grid': np.empty_like(t_grid)
    }

    for i, p in enumerate(pressures):
        print(f'Analysing depth {depth[i]} km')

        if isinstance(comp_grid, np.ndarray):
            nz_id = np.nonzero(comp_grid[i])
        else:
            nz_id = np.arange(0, len(t_grid[i]))

        t_grid_unique, indices = np.unique(t_grid[i, nz_id], return_inverse=True)
        print(f'{100*len(t_grid_unique)/len(t_grid[i])}% points to evaluate')
        Xcapv = np.zeros_like(t_grid_unique)
        Xmgo = np.zeros_like(t_grid_unique)
        Xcf = np.zeros_like(t_grid_unique)
        Xsio = np.zeros_like(t_grid_unique)
        Xppv = np.zeros_like(t_grid_unique)
        Ypv_al = np.zeros_like(t_grid_unique)
        Ypv_fe = np.zeros_like(t_grid_unique)
        Yppv_al = np.zeros_like(t_grid_unique)
        Yppv_fe = np.zeros_like(t_grid_unique)
        Ymgo_fe = np.zeros_like(t_grid_unique)
        Ymgo_na = np.zeros_like(t_grid_unique)
        Ycf_fe = np.zeros_like(t_grid_unique)
        Ycf_na = np.zeros_like(t_grid_unique)

        # DETERMINING TEMPERATURE BOUNDARIES FOR PV AND PPV FOR GIVEN PRESSURE
        if p <= min(phase_boundary_reference.lowp):
            t_pv = [0, 5000]
            t_ppv = [5000, 5000]
        elif p>= max(phase_boundary_reference.highp):
            t_pv = [5000,5000]
            t_ppv = [5000,5000]
        else:
            t_pv = [0,5000]
            t_ppv = [0,5000]
            try:
                assemblage = set_assemblage(100e9 + m * 10e9, 1000 , 0, assemblage_type,
                                            min_model, [1500,1501], [0, 5500])
                equality_constraints_pv = [('P', p), ('phase_fraction', (assemblage.phases[1], 0.))]
                sol_pv, _ = burnman.equilibrate(composition, assemblage, equality_constraints_pv,
                                                store_iterates=False, store_assemblage=True)
                t_pv[0] = sol_pv.assemblage.temperature
            except:
                pass
            try:
                assemblage = set_assemblage(100e9 + m * 10e9, 5000 , 0, assemblage_type,
                                            min_model, [1500, 1501], [0, 5500])
                equality_constraints_pv = [('P', p), ('phase_fraction', (assemblage.phases[1], 0.))]
                sol_pv, _ = burnman.equilibrate(composition, assemblage, equality_constraints_pv,
                                                store_iterates=False, store_assemblage=True)
                t_pv[1] = sol_pv.assemblage.temperature
            except:
                pass

            try:
                assemblage = set_assemblage(100e9 + m * 10e9, 1000 , 0, assemblage_type,
                                            min_model, [1500, 1501], [0, 5500])
                equality_constraints_ppv = [('P', p), ('phase_fraction',(assemblage.phases[0], 0.))]
                sol_ppv, _ = burnman.equilibrate(composition, assemblage, equality_constraints_ppv,
                                                 store_iterates=False, store_assemblage=True)
                t_ppv[0] = sol_ppv.assemblage.temperature
            except:
                pass
            try:
                assemblage = set_assemblage(100e9 + m * 10e9, 5000 , 0, assemblage_type,
                                            min_model, [1500, 1501], [0, 5500])
                equality_constraints_ppv = [('P', p), ('phase_fraction',(assemblage.phases[0], 0.))]
                sol_ppv, _ = burnman.equilibrate(composition, assemblage, equality_constraints_ppv,
                                                 store_iterates=False, store_assemblage=True)
                t_ppv[1] = sol_ppv.assemblage.temperature
            except:
                pass

        for j, temp in enumerate(t_grid_unique):

            # Calculating phase proportion for given temperature and pressure
            k, m = 1, 0
            while k != 0 and m < 100:
                assemblage = set_assemblage(p, temp, m, assemblage_type, min_model, t_pv, t_ppv)
                equality_constraints = [('T', temp), ('P', p)]

                sol,_ = burnman.equilibrate(composition, assemblage, equality_constraints,
                                            store_iterates=False, store_assemblage=True)
                k = sol.code
                m += 1
                if verbose:
                    print(p, temp, k, sol.assemblage.molar_fractions,
                          sol.assemblage.phases[0].molar_fractions)

            # Minerals that are always present
            Xcapv[j] = sol.assemblage.molar_fractions[-1]

            # Minerals depending on assemblage type
            if assemblage_type == 'depleted':
                Xmgo[j] = sol.assemblage.molar_fractions[-2]
                Ymgo_fe[j] = sol.assemblage.phases[-2].molar_fractions[1]
                if min_model == "SLB_2022":
                    Ymgo_na[j] = sol.assemblage.phases[-2].molar_fractions[2]
                else:
                    Xcf[j] = sol.assemblage.molar_fractions[-3]
                    Ycf_fe[j] = sol.assemblage.phases[-3].molar_fractions[1]
                    Ycf_na[j] = sol.assemblage.phases[-3].molar_fractions[2]
            else:
                Xcf[j] = sol.assemblage.molar_fractions[-3]
                Xsio[j] = sol.assemblage.molar_fractions[-2]
                Ycf_fe[j] = sol.assemblage.phases[-3].molar_fractions[1]
                Ycf_na[j] = sol.assemblage.phases[-3].molar_fractions[2]
            # PPV
            ppv_only = sol.assemblage.phases[0].name == 'post_perovskite' or \
                       sol.assemblage.phases[0].name == 'post-perovskite/bridgmanite'
            pv_ppv_both = sol.assemblage.phases[1].name == 'post_perovskite' or \
                          sol.assemblage.phases[1].name == 'post-perovskite/bridgmanite'
            if ppv_only:
                Xppv[j] = sol.assemblage.molar_fractions[0]
                Yppv_al[j] = sol.assemblage.phases[0].molar_fractions[2]
                Yppv_fe[j] = sol.assemblage.phases[0].molar_fractions[1]
            elif pv_ppv_both:
                Ypv_al[j] = sol.assemblage.phases[0].molar_fractions[2]
                Ypv_fe[j] = sol.assemblage.phases[0].molar_fractions[1]

                Xppv[j] = sol.assemblage.molar_fractions[1]
                Yppv_al[j] = sol.assemblage.phases[1].molar_fractions[2]
                Yppv_fe[j] = sol.assemblage.phases[1].molar_fractions[1]
            else:
                Ypv_al[j] = sol.assemblage.phases[0].molar_fractions[2]
                Ypv_fe[j] = sol.assemblage.phases[0].molar_fractions[1]

        phases['Xcapv_grid'][i, nz_id] = Xcapv[indices]
        phases['Xmgo_grid'][i, nz_id] = Xmgo[indices]
        phases['Xcf_grid'][i, nz_id] = Xcf[indices]
        phases['Xsio_grid'][i, nz_id] = Xsio[indices]
        phases['Xppv_grid'][i, nz_id] = Xppv[indices]
        phases['Ypv_al_grid'][i, nz_id] = Ypv_al[indices]
        phases['Ypv_fe_grid'][i, nz_id] = Ypv_fe[indices]
        phases['Ymgo_fe_grid'][i, nz_id] = Ymgo_fe[indices]
        phases['Ymgo_na_grid'][i, nz_id] = Ymgo_na[indices]
        phases['Ycf_fe_grid'][i, nz_id] = Ycf_fe[indices]
        phases['Ycf_na_grid'][i, nz_id] = Ycf_na[indices]
        phases['Yppv_al_grid'][i, nz_id] = Yppv_al[indices]
        phases['Yppv_fe_grid'][i, nz_id] = Yppv_fe[indices]

        del t_pv, t_ppv

    if save:
        np.savez(os.path.join(outdir, f'phases_{comp}_{min_model[-2:]}'),
                Xcapv = phases['Xcapv_grid'],
                Xmgo = phases['Xmgo_grid'],
                Xcf = phases['Xcf_grid'],
                Xsio = phases['Xsio_grid'],
                Xppv = phases['Xppv_grid'],
                Ypv_al = phases['Ypv_al_grid'],
                Ypv_fe = phases['Ypv_fe_grid'],
                Ymgo_fe = phases['Ymgo_fe_grid'],
                Ymgo_na = phases['Ymgo_na_grid'],
                Ycf_fe = phases['Ycf_fe_grid'],
                Ycf_na = phases['Ycf_na_grid'],
                Yppv_al = phases['Yppv_al_grid'],
                Yppv_fe = phases['Yppv_fe_grid'])

    return PhaseGrid(phases, t_grid, depth, lon, lat, comp, min_model, assemblage_type)


def set_assemblage(p, t, iteration, assemblage_type, min_model, t_pv, t_ppv):
    """
    Return the list of phases for an assemblage type at a specified pressure
    p and temperature t.

    Parameters
    -------
    p: float
        Pressure (Pa)
    t: float
        Temperature (K)
    iteration: int
        The iteration number, which is tracked to try different starting 
        composition guesses if the previous iteration does not converge.
    assemblage_type: str
        The mineral assemblage type used to calculate the two phase region.
        Should either be "depleted" or "enriched".
    min_model: str
        The mineralogical model used. Should either be "SLB_2022" or "SLB_2011".
    t_pv: float
        The temperature at which pPv starts becoming stable at pressure p.
    t_ppv: float
        The temperature at which bdg stops becoming stable at pressure p.

    Returns
    -------
    assemblage: burnman.Composite object
        The list of phases considered for a (p,t) point.
    """
    if min_model == 'SLB_2022':
        pv = burnman.minerals.SLB_2022.bridgmanite()
        ppv = burnman.minerals.SLB_2022.post_perovskite()
        fper = burnman.minerals.SLB_2022.ferropericlase()
        capv = burnman.minerals.SLB_2022.ca_perovskite()
        cf = burnman.minerals.SLB_2022.calcium_ferrite_structured_phase()
        stish = burnman.minerals.SLB_2022.stishovite()
    elif min_model == 'SLB_2011':
        pv = burnman.minerals.SLB_2011.mg_fe_perovskite()
        ppv = burnman.minerals.SLB_2011.post_perovskite()
        fper = burnman.minerals.SLB_2011.ferropericlase()
        cf = burnman.minerals.SLB_2011.ca_ferrite_structured_phase()
        capv = burnman.minerals.SLB_2011.ca_perovskite()
        stish = burnman.minerals.SLB_2011.stishovite()
    else:
        raise ValueError('Mineralogical model must either be SLB2022 or SLB2011')

    if assemblage_type == 'depleted':
        if min_model == "SLB_2022":
            if t > t_pv[0] and t < t_pv[1]:
                assemblage = burnman.Composite([pv, fper, capv])
            elif t < t_ppv[0] or t > t_ppv[1]:
                assemblage = burnman.Composite([ppv, fper, capv])
            else:
                assemblage = burnman.Composite([pv, ppv, fper, capv])
        else:
            if t > t_pv[0] and t < t_pv[1]:
                assemblage = burnman.Composite([pv, cf, fper, capv])
            elif t < t_ppv[0] or t > t_ppv[1]:
                assemblage = burnman.Composite([ppv, cf, fper, capv])
            else:
                assemblage = burnman.Composite([pv, ppv, cf, fper, capv])
    elif assemblage_type == 'enriched':
        if t > t_pv[0] and t < t_pv[1]:
            assemblage = burnman.Composite([pv, cf, stish, capv])
        elif t < t_ppv[0] or t > t_ppv[1]:
            assemblage = burnman.Composite([ppv, cf, stish, capv])
        else:
            assemblage = burnman.Composite([pv, ppv, cf, stish, capv])
    else:
        raise ValueError('Assemblage Type not valid. Must be depleted or enriched.')

    assemblage.set_state(p, t)
    ppv.set_composition([0.86, 0.12, 0.02])
    cf.set_composition([0.9, 0.1, 0.0])
    if min_model == "SLB_2022":
        pv.set_composition([0.88 - 0.01 * iteration, 0.06 + 0.005 * iteration,
                            0.06 + 0.005 * iteration])
        fper.set_composition([0.8, 0.1, 0.1])
    else:
        if assemblage_type == 'depleted':
            pv.set_composition([0.88 - 0.01 * iteration, 0.10 + 0.005 * iteration,
                                0.02 + 0.005 * iteration])
            fper.set_composition([0.8, 0.2])
        else:
            pv.set_composition([0.65 - 0.01 * iteration, 0.25 + 0.005 * iteration,
                                0.10 + 0.005 * iteration])

    return assemblage


def calculate_mean_ppv(ppv_array, depth, method = 'average_lateral_variations', min_depth = 2250):
    """
    Calculates the mean pPv fraction as a metric of pPv coverage for filtered
    synthetic velocity models.

    Parameters
    -------
    ppv_array: str
        pPv fraction evaluated at each Terra grid point 
    depth: array_like
        The list of depths used in PREM. PREM_depths should be an array from 
        outwards towards center of the Earth in depth (not radius)
    method: str
        The method used to calculate the mean pPv fraction. If method = 'average_
        lateral_variations', the fraction is calculated by taking the range of pPv
        at differnet depths, and calculating the mean of the values of ranges.
        If method = 'transition_depth_thickness', the fraction is given by the
        depth thickness over which pPv fraction goes from 0 to 1.
    min_depth: float
        The shallowest depth to incorporate into the mean pPv fraction calculation.
    
    Returns
    -------
    ppv_means: float
        The mean pPv fraction.
    """
    assert method in ['transition_depth_thickness', 'average_lateral_variations']

    if method == 'average_lateral_variations':
        # Calculate average of lateral variations in ppv fraction
        min_depth_index = np.argwhere(depth >= min_depth)[0,0]
        ppv_lateral = np.ptp(ppv_array[min_depth_index:], axis = 1)
        ppv_means = np.mean(ppv_lateral)
    else:
        # Calculate depth thickness over which ppv fraction goes from 0 to 1
        ppv_diff = np.argwhere(np.diff(ppv_array, axis = 0).T > 1e-10)
        _, loc = np.unique(ppv_diff[:,0], return_index=True)
        thickness = np.concatenate([np.diff(loc) - 1, [len(ppv_diff) - loc[-1]-1]])
        shallow_bound = ppv_diff[loc,1]
        deeper_bound = ppv_diff[loc + thickness, 1]

        ppv_means = np.mean(depth[1 + deeper_bound] - depth[shallow_bound])

    return ppv_means
