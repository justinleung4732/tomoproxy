import glob
import os

import numpy as np
import pyshtools as shtools
import burnman

from . import layered_model as lm
from . import spherical_shell as sh
from . import mineral_model

# Functions
def _normalise_to_PREM(spline, spline_depths, PREM_depths, normalise = True):
    """
    PREM_depths should be an array from outwards towards center of the Earth in radius (not depth)
    """
    if np.diff(spline_depths)[0] > 1:
        spline_depths = 6371 - spline_depths

    new_spline = np.zeros_like(PREM_depths)
    
    for i, d in enumerate(PREM_depths):
        loc = np.argwhere(d >= spline_depths)[0]
        new_spline[i] = (spline[loc] - spline[loc-1]) * (d - spline_depths[loc-1]) / (spline_depths[loc] - spline_depths[loc-1]) + spline[loc-1]

    if normalise:
        new_spline /= np.sum(new_spline)
    return new_spline


# Variables
# Composition
_comp_oxides = {'pyrolite': {'xSiO2': 38.71, 'xAl2O3': 2.22, 'xCaO': 2.94,
                             'xMgO': 49.85, 'xFeO': 6.17, 'xNa2O': 0.11},
               'BMO': {'xSiO2': 40.15, 'xAl2O3': 1.92, 'xCaO': 2.82,
                       'xMgO': 41.98, 'xFeO': 12.90, 'xNa2O': 0.23},
               'MORB': {'xSiO2': 51.75, 'xAl2O3': 8.16, 'xCaO': 13.88,
                        'xMgO': 14.94, 'xFeO': 7.06, 'xNa2O': 2.18},
               'HC': {'xSiO2': 48.87, 'xAl2O3': 11.28, 'xCaO': 10.59,
                      'xMgO': 20.00, 'xFeO': 12.90, 'xNa2O': 1.50}
               }

# SOLA
_SOLA_path = "/Users/univ4732/code/lema/data/SOLA_model/"

# Depths
_SOLA_depths = np.loadtxt(_SOLA_path + 'PREM_layers_depths', usecols = (1,2))
_SOLA_depths = (_SOLA_depths[:,0] + _SOLA_depths[:,1])/2
_SOLA_depths = _SOLA_depths[::-1] # Invert so depth array goes towards the core

# Spline
_SOLA_spline_vp = np.loadtxt(_SOLA_path + 'kernel_vp.csv', delimiter=',', usecols = (0,2), skiprows = 1)
_SOLA_spline_vs = np.loadtxt(_SOLA_path + 'kernel_vs.csv', delimiter=',', usecols = (0,2), skiprows = 1)
_SOLA_spline_vp = _normalise_to_PREM(_SOLA_spline_vp[:,0], _SOLA_spline_vp[:,1], _SOLA_depths)
_SOLA_spline_vs = _normalise_to_PREM(_SOLA_spline_vs[:,0], _SOLA_spline_vs[:,1], _SOLA_depths)

# PREM
_PREM_500_depths = np.loadtxt('/Users/univ4732/code/lema/data/PREM500.csv', delimiter=',', usecols = (0), skiprows = 1)[::-1] / 1000
_gamma = np.loadtxt('/Users/univ4732/code/lema/data/PREM500.csv', delimiter=',', usecols = (-1), skiprows = 1)[::-1]
_gamma = _normalise_to_PREM(_gamma, _PREM_500_depths, _SOLA_depths, normalise = False)


# Classes
class BdgPPvTwoPhaseRegion():

    def __init__(self, comp, temperatures = np.arange(1000., 4500., 50.),
                 min_model = "SLB_2022", assemblage_type = 'depleted', save = False, 
                 outdir = '', verbose = False, imported = False):
        
        assert comp in ['pyrolite', 'BMO', 'MORB', 'HC'], "Not a valid type of composition"
        assert min_model in ['SLB_2011', 'SLB_2022'], "Database not recognised. Must either be 2022 (SLB 2022) or 2011 (SLB 2011)"

        self.min_model = min_model
        self.temperatures = temperatures
        self.comp = comp
        if comp in ['pyrolite', 'BMO', 'MORB', 'HC']:
            composition = _comp_oxides[comp]
            if comp in ['pyrolite', 'BMO']:
                self.assemblage_type = 'depleted'
            else:
                self.assemblage_type = 'enriched'
        elif isinstance(comp, dict):
            assert assemblage_type in ['depleted', 'enriched'], "Assemblage type must either be depleted or enriched"
            self.assemblage_type = assemblage_type
            composition = comp
            comp = 'Custom'
        else:
            raise TypeError("comp must be either a dictionary containing oxides or a str equal to \
                            'pyrolite', 'BMO', 'MORB', 'HC'")
        
        if not imported:
            self._calculate(composition, save = save, outdir = outdir, verbose = verbose)


    def _calculate(self, composition, save = False, outdir = '', verbose = False):

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
                elif self.assemblage_type == "enriched":
                    assemblage = burnman.Composite([pv, ppv, stish, capv, cf])

                assemblage.set_state(p, t)

                equality_constraints = [('T', t), ('phase_fraction', (ppv, 0.0))]
                try:
                    sol,_ = burnman.equilibrate(composition, assemblage, equality_constraints, store_iterates=False, store_assemblage=True)
                    k = sol.code
                    if verbose:
                        print('pv', t, sol.assemblage.pressure/1e9, sol.assemblage.molar_fractions, sol.assemblage.phases[0].molar_fractions)
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
                    sol,_ = burnman.equilibrate(composition, assemblage, equality_constraints, store_iterates=False, store_assemblage=True)
                    k = sol.code
                    if verbose:
                        print('ppv', t, sol.assemblage.pressure/1e9, sol.assemblage.molar_fractions, sol.assemblage.phases[1].molar_fractions)
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
            np.savetxt(os.path.join(outdir, f'ppv_two_phase_boundary_{self.comp}_{self.min_model[-2:]}'),
                       np.array([self.temperatures, self.lowp, self.highp].T),
                       header='T, lowp, highp')
    

    @classmethod
    def from_txt(cls, txtfile):
        txtfile_split = os.path.basename(txtfile).split('_')
        assert txtfile_split[-6:-2] == ['ppv', 'two', 'phase', 'boundary'], \
            'Filename must be in "ppv_two_phase_boundary_[COMP]_[MIN_MODEL]"'
        comp = txtfile_split[-2]
        min_model = f'SLB_20{txtfile_split[-1]}'

        f = open(txtfile)
        header = f.readline()
        assert header == '# T, lowp, highp\n', "File must have headers T, lowp, highp"

        data = np.loadtxt(txtfile)

        PhaseRegion = cls(comp, min_model = min_model, 
                          imported = True)       
        PhaseRegion.temperatures = data[:,0]
        PhaseRegion.lowp = data[:,1]
        PhaseRegion.highp = data[:,2]

        return PhaseRegion


class PhaseGrid():

    def __init__(self, phases, t_grid, depth, lon, lat, comp, min_model = 'SLB_2022', assemblage_type = 'depleted'):

        assert min_model in ['SLB_2011', 'SLB_2022'], "Database not recognised. Must either be 2022 (SLB 2022) or 2011 (SLB 2011)"
        assert len(depth) == t_grid.shape[0], "Depth not matching number of rows in temperature grid"
        assert len(lon) == len(lat), "List of latitudes must be the same length as list of longitudes"
        assert len(lon) == t_grid.shape[1], "Lon/Lat not matching number of columns in temperature grid"

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
            for name in phases.keys():
                assert name + '_grid' in self.phases.keys(), "Variable phases contains a phase that is not included in the class PhaseGrid."
            for name in self.phases.keys():
                self.phases[name] = phases[name.replace('_grid', '')]




    def _from_txt(self, txtfile, assemblage_type = 'depleted'):

        assert txtfile == f'phases_{self.comp}_{self.min_model[-2:]}', \
            "File must have the name 'phases_COMP_MINMODEL'"
        
        phases = np.loadtxt(txtfile, skiprows=1)
        dim = (len(self.depth), len(self.lon))
        assert phases.shape[0] == dim[0] * dim[1] ,\
            "Depth or coordinate array length does not match the shape of the phase grid in file."
        assert assemblage_type in ['depleted', 'enriched'], "Assemblage type must either be depleted or enriched"

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

        assert f'phases_{self.comp}_{self.min_model[-2:]}.npz' in npz_file, \
            "File must have the name 'phases_COMP_MINMODEL.npz'"

        file = np.load(npz_file)

        assert all(phase.shape == self.t_grid.shape for phase in file.values()) ,\
        "Depth or coordinate array length does not match the shape of the phase grid in file."

        for name in self.phases.keys():
            self.phases[name] = file[name.replace('_grid', '')]


    def calculate_ppv_frac(self, py_phases = '', X = None, exclude_LLVP = False, threshold = 0.6):
        """A function that imports equilibrium phase assemblage files and calculates the 
        bdg-ppv fraction at each Terra grid point"""

        if 'pyrolite' not in self.comp:
            assert isinstance(py_phases, PhaseGrid) or f'phases_pyroliteTC_{self.min_model[-2:]}' in py_phases, \
                "py_phase needed for compositional non-heterogeneous part of the mantle. Must be either a file \
                    with the name phases_pyroliteTC_MINMODEL file or a PhaseGrid object"
            if isinstance(py_phases, PhaseGrid):
                assert py_phases.comp == 'pyroliteTC', "pyroliteTC composition for py_phase needed"
            assert X is not None, "comp_grid needed for thermochemical compositions"
            assert X.shape == self.t_grid.shape, "Shape of X must be same as that of temperature field."

            if isinstance(py_phases, str):
                py_phases = PhaseGrid(py_phases, self.t_grid, self.depth, self.lon, self.lat, 
                                      'pyroliteTC', self.min_model)

            pv = 1 - X * (self.phases['Xcapv_grid'] + self.phases['Xmgo_grid'] + 
                          self.phases['Xsio_grid'] + self.phases['Xcf_grid'] + self.phases['Xppv_grid']) +\
                (X-1) * (py_phases.phases['Xcapv_grid'] + py_phases.phases['Xmgo_grid'] +
                         py_phases.phases['Xsio_grid'] + py_phases.phases['Xcf_grid'] + py_phases.phases['Xppv_grid'])
            ppv = X * self.phases['Xppv_grid'] + (1-X) * py_phases.phases['Xppv_grid']

            ppv_frac = ppv / (pv + ppv)

        else:
            pv = 1 - self.phases['Xcapv_grid'] - self.phases['Xmgo_grid'] - self.phases['Xsio_grid'] -\
                    self.phases['Xcf_grid'] - self.phases['Xppv_grid']
            ppv_frac = self.phases['Xppv_grid'] / (pv + self.phases['Xppv_grid'])
        
        if exclude_LLVP:
            assert X is not None, "comp_grid needed for partppv scenario pPv fraction calclulation"
            LLVP_not = np.argwhere(X > threshold)
            ppv_frac[LLVP_not[:,0], LLVP_not[:,1]] = 0

        return ppv_frac
        

    def evaluate_elastic(self, ppv_mode, X = None, py_model = None, save = False, outdir = ''):
    # Apply above calculation to Earth Model (Creating elastic Earth Model)

    # Note that this step takes quite a lot of time (especially if the resolution of em is high).
    # Nonetheless, this step can run in parallel over the points, but not in a jupyter notebook.
    # Calculate elastic velocities for a range of P and T

        if "pyrolite" not in self.comp:
            assert isinstance(py_model, ElasticGrid), "Reference thermal (pyrolite) model needed at points outside LLVPs"
            assert py_model.comp == "pyroliteTC", "TC pyrolite elastic model needs to be used"
            assert X is not None, "A composition grid must be specified for thermochemical models"

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

            if isinstance(X, np.ndarray):
                nz_id = np.nonzero(X[i])[0]
                grid_unique, unique_id, indices = np.unique(np.vstack([self.t_grid[i, nz_id], X[i, nz_id]]).T, axis = 0, return_index=True, return_inverse=True) # find unique pairings of (X,T)
                grid_unique = grid_unique[:,0] # only keep the T values
            else:
                nz_id = np.arange(0, len(self.t_grid[i]))
                grid_unique, unique_id, indices = np.unique(self.t_grid[i], return_index=True, return_inverse=True)

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
            nz_id = np.nonzero(X)
            rho_grid = (1-X)[nz_id] * py_model.rho_grid[nz_id] + X[nz_id] * rho_grid[nz_id]
            vp_grid = (1-X)[nz_id] * py_model.vp_grid[nz_id] + X[nz_id] * vp_grid[nz_id]
            vphi_grid = (1-X)[nz_id] * py_model.vphi_grid[nz_id] + X[nz_id] * vphi_grid[nz_id]
            vs_grid = (1-X)[nz_id] * py_model.vs_grid[nz_id] + X[nz_id] * vs_grid[nz_id]
            k_grid = (1-X)[nz_id] * py_model.k_grid[nz_id] + X[nz_id] * k_grid[nz_id]
            g_grid = (1-X)[nz_id] * py_model.g_grid[nz_id] + X[nz_id] * g_grid[nz_id]

        if save:
            np.savez(outdir + f"elastic_{self.comp}_{self.min_model[-2:]}_two_phase", 
                    rho = rho_grid,
                    vp = vp_grid,
                    vphi = vphi_grid,
                    vs = vs_grid,
                    k = k_grid,
                    g = g_grid)        

        return ElasticGrid(self.depth, self.lon, self.lat, rho_grid, vp_grid, vphi_grid, vs_grid, k_grid, g_grid)


class ElasticGrid():
    
    def __init__(self, depth, lon, lat, rho_grid = None, vp_grid = None, 
                 vphi_grid = None, vs_grid = None, 
                 k_grid = None, g_grid = None, comp = None):
        
        assert len(lon) == len(lat), "List of latitudes must be the same length as list of longitudes"
        self.lon = lon
        self.lat = lat
        self.depth = depth
        
        self.rho_grid = rho_grid
        self.vp_grid = vp_grid
        self.vphi_grid = vphi_grid
        self.vs_grid = vs_grid
        self.k_grid = k_grid
        self.g_grid = g_grid
    
        self.comp = comp


    @classmethod
    def from_file(cls, fileloc, comp, ppv_model_type, depth, lon, lat, min_model = 'SLB_2022', comp_grid = None, threshold = 0.6):

        assert comp in ['pyrolite', 'pyroliteTC', 'BMO', 'MORB', 'HC'], "Not a valid type of composition" 
        assert len(lon) == len(lat), "List of latitudes must be the same length as list of longitudes"
        assert min_model in ['SLB_2011', 'SLB_2022'], "Database not recognised. Must either be 2022 (SLB 2022) or 2011 (SLB 2011)"
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
            LLVP = np.argwhere(comp_grid >= threshold)
            file_LLVP = f'elastic_{comp}_{min_model[-2:]}_none.npz'
            elastic_LLVP = np.load(fileloc+file_LLVP)

            rho_grid[LLVP[:,0], LLVP[:,1]] = elastic_LLVP['rho'][LLVP[:,0], LLVP[:,1]]
            vp_grid[LLVP[:,0], LLVP[:,1]] = elastic_LLVP['vp'][LLVP[:,0], LLVP[:,1]]
            vphi_grid[LLVP[:,0], LLVP[:,1]] = elastic_LLVP['vphi'][LLVP[:,0], LLVP[:,1]]
            vs_grid[LLVP[:,0], LLVP[:,1]] = elastic_LLVP['vs'][LLVP[:,0], LLVP[:,1]]
            k_grid[LLVP[:,0], LLVP[:,1]] = elastic_LLVP['k'][LLVP[:,0], LLVP[:,1]]
            g_grid[LLVP[:,0], LLVP[:,1]] = elastic_LLVP['g'][LLVP[:,0], LLVP[:,1]]
        
        return cls(depth, lon, lat, rho_grid, vp_grid, 
                   vphi_grid, vs_grid, k_grid, g_grid)
    

    def to_continuous_param(self, r_deg = 20, sph_deg = 8, save = False, outdir = '', filename = ''):

        print(f"Converting model {filename}")
        input_data = [self.depth, sph_deg * np.ones(len(self.depth), dtype='int'), 'V']
        Vp_layer = lm.LayeredModel(input_data)
        Vs_layer = lm.LayeredModel(input_data)
        Vphi_layer = lm.LayeredModel(input_data)

        for i, _ in enumerate(self.depth):
            print(f"Converting layer {i}")
            cilm_Vp, _ = shtools.expand.SHExpandLSQ(self.vp_grid[i], self.lat, self.lon, lmax = sph_deg, norm=4, csphase=1)
            cilm_Vs, _ = shtools.expand.SHExpandLSQ(self.vs_grid[i], self.lat, self.lon, lmax = sph_deg, norm=4, csphase=1)
            cilm_Vphi, _ = shtools.expand.SHExpandLSQ(self.vphi_grid[i], self.lat, self.lon, lmax = sph_deg, norm=4, csphase=1)
            
            Vp_layer.layers[i].cilm[:,:,:]= cilm_Vp
            Vs_layer.layers[i].cilm[:,:,:]= cilm_Vs
            Vphi_layer.layers[i].cilm[:,:,:]= cilm_Vphi

        if save:
            Vp_layer.write_tomography_file(os.path.join(outdir, 'SH_' + filename + '_Vp'))
            Vs_layer.write_tomography_file(os.path.join(outdir, 'SH_' + filename + '_Vs'))
            Vphi_layer.write_tomography_file(os.path.join(outdir, 'SH_' + filename + '_Vc'))

        return RawSeismicModel(Vp_layer, Vs_layer, Vphi_layer, r_deg)


class RawSeismicModel():

    def __init__(self, Vp, Vs, Vphi, r_deg):

        assert isinstance(Vp, lm.LayeredModel), "Vp needs to be a LayeredModel instance"
        assert isinstance(Vs, lm.LayeredModel), "Vs needs to be a LayeredModel instance"
        assert isinstance(Vphi, lm.LayeredModel), "Vphi needs to be a LayeredModel instance"

        self.lmax = Vp.layers[0].lmax
        self.rdeg = r_deg

        self._to_sshell(Vp, Vs, Vphi)


    @classmethod
    def from_file(cls, r_deg, fileloc, comp, ppv_model_type, min_model = '', seismic_model = ''):

        assert comp in ['pyrolite', 'pyroliteTC', 'BMO', 'MORB', 'HC'], "Not a valid type of composition" 
        assert ppv_model_type in ['noppv', 'ppv', 'partppv'], "Not a valid type of ppv_model"
        assert min_model in ['SLB_2011', 'SLB_2022'], "Database not recognised. Must either be 2022 (SLB 2022) or 2011 (SLB 2011)"

        Vp_layer = lm.LayeredModel(f'{fileloc}SH_{seismic_model}_{comp}_{ppv_model_type}_{min_model[-2:]}_Vp')
        Vs_layer = lm.LayeredModel(f'{fileloc}SH_{seismic_model}_{comp}_{ppv_model_type}_{min_model[-2:]}_Vs')
        Vphi_layer = lm.LayeredModel(f'{fileloc}SH_{seismic_model}_{comp}_{ppv_model_type}_{min_model[-2:]}_Vc')

        return cls(Vp_layer, Vs_layer, Vphi_layer, r_deg)


    def to_SOLA(self):

        assert self.lmax >= 8, "Spherical degree is not high enough to create SOLAShell"
        Vp = np.zeros((len(_SOLA_depths), 2, 9, 9))
        Vs = np.zeros_like(Vp)
        Vphi = np.zeros_like(Vs)

        for i, d in enumerate(_SOLA_depths):
            if d < self.vp.r_min or d > self.vp.r_max:
                continue
            Vp[i] = self._abs_to_rel_velocity(self.vp.get_sh_coefs_at_r(d)[:, :9, :9])
            Vs[i] = self._abs_to_rel_velocity(self.vs.get_sh_coefs_at_r(d)[:, :9, :9])
            Vphi[i] = self._abs_to_rel_velocity(self.vphi.get_sh_coefs_at_r(d)[:, :9, :9])

            # Set odd degress to 0
            Vp[i,:,1::2] = 0
            Vs[i,:,1::2] = 0
            Vphi[i,:,1::2] = 0

        return SOLAShell(Vp, Vs, Vphi)


    def _to_sshell(self, Vp, Vs, Vphi):

        self.vp = sh.SShell(spherical_degree = self.lmax, radial_degree = self.rdeg, r_min = 6371.0 - Vp.layers[-1].depth, r_max = 6371.0 - Vp.layers[0].depth)
        self.vs = sh.zeros_like(self.vp)
        self.vphi = sh.zeros_like(self.vp)
        
        # Read layered model into spherical shells
        self.vp.fit_coef_from_layeredmodel(Vp)
        self.vs.fit_coef_from_layeredmodel(Vs)
        self.vphi.fit_coef_from_layeredmodel(Vphi)


    @staticmethod
    def _abs_to_rel_velocity(coefs):
        coefs /= coefs[0,0,0] / (2.0 * np.sqrt(np.pi))
        coefs *= 100
        coefs[0,0,0] = 0
        return coefs
    

class SOLAShell():
    
    def __init__(self, Vp = None, Vs = None, Vphi = None, 
                 Vp_err = None, Vs_err = None, Vphi_err = None):
        
        self.depths = _SOLA_depths
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

        if Vp is not None:
            self.update_velocities('vp', Vp)

        if Vs is not None:
            self.update_velocities('vs', Vs)

        if Vphi is not None:
            self.update_velocities('vphi', Vphi)

        if Vp_err is not None:
            self.update_velocity_errors('vp', Vp_err)

        if Vs_err is not None:
            self.update_velocity_errors('vs', Vs_err)

        if Vphi_err is not None:
            self.update_velocity_errors('vphi', Vphi_err)
        
        if Vp is not None and Vs is not None and Vphi is None:
            self._calculate_vphi()


    def update_velocities(self, v_type, velocity):

        assert self.filtered[v_type] == False, "Cannot update velocities after filtering"
        assert velocity.shape == ((len(self.depths), 2, self.lmax+1, self.lmax+1)), "Wrong shape for velocity array"
        assert v_type in ['vp', 'vs', 'vphi'], "v_type must be 'vp', 'vs' or 'vphi'"

        if v_type == 'vp':
            self.vp = velocity
        elif v_type == 'vs':
            self.vs = velocity
        elif v_type == 'vphi':
            self.vphi = velocity

    
    def update_velocity_errors(self, err_type, error):

        assert self.filtered[err_type] == False, "Cannot update velocities after filtering"
        assert error.shape == ((len(self.depths), 2, self.lmax+1, self.lmax+1)), "Wrong shape for velocity array"
        assert err_type in ['vp', 'vs', 'vphi'], "err_type must be 'vp', 'vs' or 'vphi'"

        if err_type == 'vp':
            self.vp_err = error
        elif err_type == 'vs':
            self.vs_err = error
        elif err_type == 'vphi':
            self.vphi_err = error


    @classmethod
    def from_directory(cls, directory):

        data_files = glob.glob(directory + '/**/mk**.txt', recursive = True)

        # Storage for raw coefficients at PREM depth layers
        vp_raw = np.zeros((len(_SOLA_depths), 2, 9, 9))
        vs_raw = np.zeros_like(vp_raw)
        vp_err_raw = np.zeros_like(vp_raw)
        vs_err_raw = np.zeros_like(vp_raw)

        for i, deg_file in enumerate(data_files):
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
        
        return cls(Vp = vp_raw, Vs = vs_raw, Vp_err = vp_err_raw, Vs_err = vs_err_raw)


    def apply_kernel(self):

        self._apply_individual_kernel('vphi') if getattr(self, 'vphi') is not None else None
        self._apply_individual_kernel('vp') if getattr(self, 'vp') is not None else None
        self._apply_individual_kernel('vs') if getattr(self, 'vs') is not None else None

    
    def _apply_individual_kernel(self, velocity):

        assert velocity in ['vp', 'vs', 'vphi'], "Velocity must be 'vp', 'vs' or 'vphi'"
        v_err = velocity + '_err'
        assert getattr(self, velocity) is not None, \
        "Kernel cannot be applied without velocities"


        if velocity == 'vp':
            spline = _SOLA_spline_vp
        elif velocity == 'vs':
            spline = _SOLA_spline_vs
        elif velocity == 'vphi':
            spline = (_SOLA_spline_vp + _SOLA_spline_vs) / 2

        setattr(self, velocity, np.average(getattr(self, velocity), axis = 0, weights = spline))
        setattr(self, v_err, np.average(getattr(self, v_err), axis = 0, weights = spline)) if getattr(self, v_err) is not None else None
        setattr(self, f'spline_{velocity}', spline)
        self.filtered[velocity] = True


    def _calculate_vphi(self):

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

        self.vphi = (self.vp - _gamma[:, None, None, None] * self.vs) / (1 - _gamma[:, None, None, None])
        if include_err:
            self.vphi_err = (self.vp_err - _gamma[:, None, None, None] * self.vs_err) / (1 - _gamma[:, None, None, None])


# Functions
def oxide_to_phase(t_grid, depth, lon, lat, comp, phase_boundary_reference, X = 0,
                   min_model = 'SLB_2022', assemblage_type = 'depleted', save = False, outdir = '', verbose = False):
    
    assert len(depth) == t_grid.shape[0], "Depth not matching number of rows in temperature grid"
    assert len(lon) == len(lat), "List of latitudes must be the same length as list of longitudes"
    assert len(lon) == t_grid.shape[1], "Lon/Lat not matching number of columns in temperature grid"
    assert min_model in ['SLB_2011', 'SLB_2022'], "Database not recognised. Must either be 2022 (SLB 2022) or 2011 (SLB 2011)"
    assert isinstance(phase_boundary_reference, BdgPPvTwoPhaseRegion), "Phase boundary reference must be BdgPPvTwoPhaseRegion Object."

    pressures = burnman.seismic.PREM().pressure(depth * 1000.0)

    if comp in ['pyrolite', 'BMO', 'MORB', 'HC']:
        composition = _comp_oxides[comp]
        if comp in ['pyrolite', 'BMO']:
            assemblage_type = 'depleted'
        else:
            assemblage_type = 'enriched'
    elif comp == 'pyroliteTC':
        composition = _comp_oxides['pyrolite']
        assemblage_type = 'depleted'
    elif isinstance(comp, dict):
        assert assemblage_type in ['depleted', 'enriched'], "Assemblage type must either be depleted or enriched"
        composition = comp
        comp = 'Custom'
    else:
        raise TypeError("comp must be either a dictionary containing oxides or a str equal to \
                        'pyrolite', 'BMO', 'MORB', 'HC'")
    
    assert phase_boundary_reference.comp == comp, "Phase boundary reference composition must match input composition"

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

        if isinstance(X, np.ndarray):
            nz_id = np.nonzero(X[i])
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
                assemblage = set_assemblage(100e9 + m * 10e9, 1000 , 0, assemblage_type, min_model, [1500,1501], [0, 5500])
                equality_constraints_pv = [('P', p), ('phase_fraction', (assemblage.phases[1], 0.))]
                sol_pv, _ = burnman.equilibrate(composition, assemblage, equality_constraints_pv, store_iterates=False, store_assemblage=True)
                t_pv[0] = sol_pv.assemblage.temperature
            except:
                pass
            try:
                assemblage = set_assemblage(100e9 + m * 10e9, 5000 , 0, assemblage_type, min_model, [1500, 1501], [0, 5500])
                equality_constraints_pv = [('P', p), ('phase_fraction', (assemblage.phases[1], 0.))]
                sol_pv, _ = burnman.equilibrate(composition, assemblage, equality_constraints_pv, store_iterates=False, store_assemblage=True)
                t_pv[1] = sol_pv.assemblage.temperature
            except:
                pass
        
            try:
                assemblage = set_assemblage(100e9 + m * 10e9, 1000 , 0, assemblage_type, min_model, [1500, 1501], [0, 5500])
                equality_constraints_ppv = [('P', p), ('phase_fraction', (assemblage.phases[0], 0.))]
                sol_ppv, _ = burnman.equilibrate(composition, assemblage, equality_constraints_ppv, store_iterates=False, store_assemblage=True)
                t_ppv[0] = sol_ppv.assemblage.temperature
            except:
                pass
            try:
                assemblage = set_assemblage(100e9 + m * 10e9, 5000 , 0, assemblage_type, min_model, [1500, 1501], [0, 5500])
                equality_constraints_ppv = [('P', p), ('phase_fraction', (assemblage.phases[0], 0.))]
                sol_ppv, _ = burnman.equilibrate(composition, assemblage, equality_constraints_ppv, store_iterates=False, store_assemblage=True)
                t_ppv[1] = sol_ppv.assemblage.temperature
            except:
                pass

        for j, temp in enumerate(t_grid_unique):

            # Calculating phase proportion for given temperature and pressure
            k, m = 1, 0
            while k != 0 and m < 100:
                assemblage = set_assemblage(p, temp, m, assemblage_type, min_model, t_pv, t_ppv)
                equality_constraints = [('T', temp), ('P', p)]
                
                sol,_ = burnman.equilibrate(composition, assemblage, equality_constraints, store_iterates=False, store_assemblage=True)
                k = sol.code
                m += 1
                if verbose:
                    print(p, temp, k, sol.assemblage.molar_fractions, sol.assemblage.phases[0].molar_fractions)

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
            ppv_only = sol.assemblage.phases[0].name == 'post_perovskite' or sol.assemblage.phases[0].name == 'post-perovskite/bridgmanite'
            pv_ppv_both = sol.assemblage.phases[1].name == 'post_perovskite' or sol.assemblage.phases[1].name == 'post-perovskite/bridgmanite'
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
                
    return PhaseGrid(phases, t_grid, depth, lon, lat, comp, assemblage_type, min_model)
        

def set_assemblage(p, t, iteration, assemblage_type, min_model, t_pv, t_ppv):
    """
    Return the list of phases for an assemblage type at a specified pressure p and temperature t.
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
        raise ValueError('Database not recognised. Must either be 2022 (SLB 2022) or 2011 (SLB 2011)')

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
        pv.set_composition([0.88 - 0.01 * iteration, 0.06 + 0.005 * iteration, 0.06 + 0.005 * iteration])
        fper.set_composition([0.8, 0.1, 0.1])
    else:
        if assemblage_type == 'depleted':
            pv.set_composition([0.88 - 0.01 * iteration, 0.10 + 0.005 * iteration, 0.02 + 0.005 * iteration])
            fper.set_composition([0.8, 0.2])
        else:
            pv.set_composition([0.65 - 0.01 * iteration, 0.25 + 0.005 * iteration, 0.10 + 0.005 * iteration])

    return assemblage


def calculate_mean_ppv(ppv_array, depth, method = 'average_lateral_variations', min_depth = 2250):
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