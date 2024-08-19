"""
Script containing functions for exceedance probability estimation for a set of
plume height samples.

Author: Shannon Williams

Date: 25/09/2023
"""

import os
import sys
import iris
import numpy as np
import pandas as pd
import iris.analysis.maths as maths

from iris.cube import CubeList
from iris.coords import AuxCoord, DimCoord
from multiprocessing import cpu_count
from subprocess import check_output, Popen, PIPE, CalledProcessError
from itertools import compress
from scipy.stats import truncnorm, rv_continuous
from time import time

from pvauncertainty.csv_funs import *
from pvauncertainty.cubes import *
from pvauncertainty.utils import *
from pvauncertainty.quadrature import quad_vec

class VolcanicNAME(object):
    name: str

    ensemble: bool
    member_dirs: list
    n_members: int
    
    esp_csv: str
    esp_data: pd.DataFrame
    
    output_dir: str
    maininput_file: str
    
    run_finished: str

    def __init__(self,
                 name: str,
                 ensemble: bool,
                 output_dir: str,
                 esp_csv: str,
                 maininput_file: str = "maininput.txt"):
        """Initialise VolcanicNAME object.

        Args:
            name (str): Descriptive name for the object.
            ensemble (bool): Whether ensemble or deterministic met is used.
            output_dir (str): Directory where NAME outputs are stored.
            esp_csv (str): csv file containing eruption source parameters.
            maininput_file (str, optional): File within output_dir containing main NAME inputs. Defaults to "maininput.txt".
        """
        self.name = name
        self.ensemble = ensemble
        self.output_dir = (output_dir + "/" if not output_dir[-1] == "/" 
                           else output_dir)
        self.esp_csv = esp_csv
        self.esp_data = pd.read_csv(esp_csv)
        self.maininput_file = maininput_file
        self.run_finished = False
        
        if self.ensemble:
            self.member_dirs = sort_member_dirs(self.output_dir)
            self.n_members = len(self.member_dirs)
            
            
    def check_run_finished(self):
        """Check whether NAME run(s) are finished so that post-processing can be done.

        Returns:
            bool: Indication of whether NAME run(s) have finished.
        """
        # Check log file to see if NAME run/s has/have finished
        maininput_log = self.maininput_file.split(".")[0] + "Log.txt"

        # Check all ensemble members have finished running
        if self.ensemble:
            fins = [False for _ in range(self.n_members)]
            
            for m, dir in enumerate(self.member_dirs):
                try:
                    output = check_output(["tail", "-1", dir + maininput_log])
                    line = output.decode(sys.stdout.encoding)
                    
                    if line.startswith(" Run completed"):
                        fins[m] = True
                    else: # Sometimes this message is on the penultimate line 
                        output = check_output(["tail", "-2", dir + maininput_log])
                        line = output.decode(sys.stdout.encoding)
                        if line.startswith(" Run completed"):
                            fins[m] = True
                except CalledProcessError:
                    continue
                        
            self.run_finished = all(fins)

        else: # Deterministic met
            output = check_output(["tail", "-1", 
                                   self.output_dir + maininput_log])
            line = output.decode(sys.stdout.encoding)
            self.run_finished = (True if line.startswith(" Run completed") 
                                 else False)
            
        return self.run_finished

    def get_ash_cube_list(self, member: int = None, **kwargs):
        """Construct a CubeList where each cube is the ash concentration output for a different source, as specified in the esp_csv.

        Args:
            member (int, optional): If ensemble met, which member to obtain ash cubes for. If None, obtains list for all ensemble members. Defaults 
            to None.

        Returns:
            CubeList: List of ash concentration cubes.
        """
        if not self.run_finished:
            if not self.check_run_finished():
                print("NAME run not yet finished. Re-run once finished.")
                return None
        
        if self.ensemble:
            # Get list for specified member
            if member is not None:
                dir = self.member_dirs[member]
                ash_cube_list = self._get_cube_list(dir, **kwargs)

            # Default - get list of lists for all members
            else:
                ash_cube_list = []
                for dir in self.member_dirs:
                    this_cube_list = self._get_cube_list(dir, **kwargs)
                    ash_cube_list.append(this_cube_list)

        # Get list for deterministic data 
        else:
            ash_cube_list = self._get_cube_list(self.output_dir, **kwargs)

        return ash_cube_list

    def load_ash_cube_list(self, filename: str, member: int = None,  **kwargs):
        """Load ash cube list from file. If file does not already exist, 
        constructs cube list and saves to file.

        Args:
            filename (str): File to load cube list from, or otherwise save to.
            member (int, optional): If ensemble met, which member to obtain ash cubes for. If None, obtains list for all ensemble members. Defaults to None.

        Returns:
            CubeList: List of ash concentration cubes.
        """
        # Load cube list and sort according to height
        callback = kwargs.pop("callback", None)
        if os.path.exists(filename):
            cube_list = sort_cube_list(iris.load(filename), 
                                       sort_by_height = True)
            # Constrain cube list according to specifications
            cube_list = CubeList(
                [constrain_cube(cube, **kwargs) for cube in cube_list])
                
        else:
            cube_list = self.get_ash_cube_list(member, callback = callback,
                                               **kwargs)
            if cube_list is None:
                raise RuntimeError("No cube list returned.")
            iris.save(cube_list, filename)
            print(f"Saved cube as {filename}.")
            
        return cube_list

    def _get_cube_list(self, dir: str, file_root: str = "Fields_grid99", 
                       **kwargs):
        # Initialise cubelist
        ash_cube_list = CubeList([])

        # Get cube for each chunk/sample/section of the plume
        for i in range(len(self.esp_data)):
            this_cube = get_cube(dir, file_root + f"_S{i}_*", **kwargs)
            this_cube = update_cube_attrs_from_data(this_cube, 
                                                    self.esp_data, i)
            ash_cube_list.append(this_cube)

        return ash_cube_list
                    
def _rescale_cube_list(cube_list: CubeList, 
                       new_mer_gs_ht: float = None,
                       height_diff: float = 0.0, 
                       new_title: str = None):
    """Rescale all cubes in a list by the same scaling factor, proportional to their lengths, except the topmost (final) one if correct_top_chunk then apply a linear correction.
    """
    if not isinstance(cube_list, list):
        cube_list = [cube_list]
    
    # Initialise cumulative MER
    cum_mer = 0.0
    all_min_h = cube_list[0].attributes["Release height min"]

    for i, cube in enumerate(cube_list):
        
        # Rescale according to the MER
        if new_mer_gs_ht is not None:
            cur_mer = float(cube.attributes["Release rate"].split(" ")[0])
            min_h = float(cube.attributes["Release height min"].split(" ")[0])
            max_h = float(cube.attributes["Release height max"].split(" ")[0])
            dZ = max_h - min_h
            
            # Rescale sections by new MER and length
            if (np.isclose(height_diff, 0.0) or 
                (not np.isclose(height_diff, 0.0) and i < len(cube_list)-1)):
                new_mer = new_mer_gs_ht * dZ
                cube = new_mer * cube / cur_mer
            else: # Apply linear correction to top section
                new_mer = new_mer_gs_ht * height_diff
                cube = new_mer * cube * height_diff / (dZ * cur_mer)
            
            cum_mer += new_mer
        
        # Add the new cubes together
        if i == 0:
            rescaled_cube = cube.copy()
        else:
            rescaled_cube = maths.add(rescaled_cube, cube)
            
    # Update cube attributes
    new_max = min_h + height_diff if height_diff > 0.0 else max_h
    new_attrs = {"Release rate": f"{cum_mer} g/s",
                 "Release height min": all_min_h,
                 "Release height max": f"{new_max} m"}
    if new_title is None:
        try:
            new_attrs.update({"Title": rescaled_cube.attributes["Title"]})
        except KeyError:
            pass

    rescaled_cube = update_cube_attrs_from_dict(rescaled_cube, new_attrs)
    
    return rescaled_cube
    
def reconstruct_cube_from_list(cube_list: CubeList, 
                               plume_height: float, 
                               new_mer_gs_ht: float = None,
                               new_title: str = None):
    """Given a list of ash concentration cubes for chunks between the vent and a maximum plume height, construct a cube approximating the ash concentration cube given a plume height below the maximum, and corresponding MER.

    Args:
        cube_list (CubeList): Cube list of separate sources, e.g. output from
        VolcanicNAME.get_ash_cube_list().
        plume_height (float): New plume height in m a.s.l. whose resultant ash concentration cube is to be constructed. Must be less than the maximum release height in cube_list.
        new_mer_gs_ht (float, optional): Mass released in grams per second per 
        unit height. If None, rescaling is not applied. Defaults to None.
        new_title (str, optional) . Defaults to None

    Returns:
        Cube: Ash concentration cube approximating the "true" cube.
    """
    # Get each chunk height in m a.s.l.
    chunk_heights_asl = [float(cube.attributes["Release height max"].split(" ")[0]) 
                     for cube in cube_list]

    if len(chunk_heights_asl) > 1:
        # Get the relevant chunks based on height
        these_chunks = [h <= plume_height for h in chunk_heights_asl]
        
        # Check whether this new height is included in the topmost chunk
        try:
            false_index = these_chunks.index(False)
            correct_top_chunk = (
                False if chunk_heights_asl[false_index-1] == plume_height else True)
        except ValueError:
            correct_top_chunk = False

        # If so add the next chunk to the list to be rescaled
        if correct_top_chunk:
            these_chunks[false_index] = True
            height_diff = plume_height - chunk_heights_asl[false_index-1]
        else:
            height_diff = 0.0

        # Get cube list and rescale according to MER
        cube_list = CubeList(compress(cube_list, these_chunks))
    else:
        height_diff = 0.0
    
    return _rescale_cube_list(cube_list, new_mer_gs_ht, height_diff,
                              new_title)

class SourceChunks(object):
    name: str

    volcano_height: np.float64
    max_plume_height: np.float64
    min_plume_height: np.float64
    height_range: np.float64
    
    chunk_csv: str
    chunk_data: pd.DataFrame
    
    name_output: VolcanicNAME
    ash_cube_list: list
    
    def __init__(self, 
                 volcano_height: float,
                 esp_csv: str, 
                 max_plume_height: float = None,
                 min_plume_height: float = None,
                 n_chunks: int = None,
                 chunk_lengths: float = None,
                 chunk_list: list = None,
                 particles_hr: int = 15000,
                 unit_mer: bool = True,
                 name: str = None, ):
        """Initialise SourceChunks object for running NAME with segmented 
        sources between vent and maximum plume height for later post-processing. 
        
        If esp_csv does not already exist or one of n_chunks, chunk_lenths or chunk_list is given, generate data frame of source parameters for segmented plume between vent and maximum possible plume height, where each source "chunk" has unit MER and releases particles proportionally to its length (corresponding to dZ in the NAME input file sources block). 
        
        If esp_csv already exists, loads this data and the corresponding NAME outputs.

        Args:
            volcano_height (float): Vent height, in m a.s.l.
            esp_csv (str): csv file to load source parameters from, or otherwise save them to.
            max_plume_height (float): Maximum plume height, in m a.s.l.
            min_plume_height (float, optional): Minimum plume height, in m a.s.l. If None, is same as volcano_height. Defaults to None.
            n_chunks (int, optional): Number of source chunks between vent and maximum plume height. Determines uniform length of chunks. Defaults to None.
            chunk_lengths (float, optional): Uniform length of chunks. Determines number of chunks. Defaults to None.
            chunk_list (list, optional): List of chunk heights in m a.s.l. Defaults to None.
            particles_hr (int, optional): Number of particles released by entire plume per hour. Defaults to 15000.
            name (str, optional): Name of SourceChunks object. Defaults to None.
        """
        self.volcano_height = volcano_height
        self.name = name

        if (os.path.exists(esp_csv) 
            and all(x is None for x in (n_chunks, chunk_lengths, chunk_list))):
            self.chunk_csv = esp_csv
            self.chunk_data = pd.read_csv(esp_csv)
            max_heights = self.chunk_data["max (m)"]
            self.min_plume_height = np.min(max_heights)
            self.max_plume_height = np.max(max_heights)

        else:
            self.min_plume_height = (min_plume_height if min_plume_height 
                                     is not None else self.volcano_height)
            if max_plume_height is None:
                if chunk_list is None:
                    raise ValueError(
                        "Provide max_plume_height to generate chunk data.")
            else:
                self.max_plume_height = max_plume_height            
            self._generate_chunk_data(esp_csv, n_chunks, chunk_lengths,
                                      chunk_list, particles_hr, unit_mer)
            self.chunk_data = pd.read_csv(esp_csv)
        
    def _generate_chunk_data(self, 
                             esp_csv: str,
                             n_chunks: int = None,
                             chunk_lengths: float = None,
                             chunk_list: list = None,
                             particles_hr: int = 15000, 
                             unit_mer: bool = True):
        """Functio for generating chunk data."""
        if chunk_list is not None:
            self.min_plume_height = chunk_list[0]
            self.max_plume_height = chunk_list[-1]
            chunk_list = chunk_list
            n_chunks = len(chunk_list) - 1

        elif n_chunks is not None or chunk_lengths is not None:
            if n_chunks is not None:
                if n_chunks > 0:
                    chunk_lengths = ((self.max_plume_height - self.min_plume_height) 
                                    / n_chunks)
                else: 
                    chunk_lengths = 0
                
            else:    
                n_chunks = int(
                    (self.max_plume_height - self.min_plume_height) 
                    / chunk_lengths)
                self.max_plume_height = (self.min_plume_height 
                                         + chunk_lengths * n_chunks)

            chunk_list = [self.min_plume_height + chunk_lengths * i 
                               for i in range(n_chunks+1)]
                
        else:
            raise ValueError("Specify n_chunks, chunk_lengths or chunk_list.")
        
        list_dicts = []    

        # Particles per hour per unit height
        particles_hr_ht = (particles_hr / 
                           (self.max_plume_height - self.volcano_height))
    
        if n_chunks > 0:
            for i in range(n_chunks+1):
                # Set first chunk to be from the vent height to the lowest 
                # possible plume height, since this will be included in every 
                # simulation
                if i == 0:
                    min_z = self.volcano_height
                    max_z = self.min_plume_height
                    label = "Vent"

                else: 
                    min_z = chunk_list[i-1]
                    max_z = chunk_list[i]
                    label = "Chunk " + str(i)

                # Generate source parameters
                this_dict = source_params_dict(
                    label, min_z, max_z, particles_hr_ht, mer_gs = 1)
                list_dicts.append(this_dict)
        else: # Single source with MER given from IVESPA
            if unit_mer:
                mer_gs = 1
            else:
                ivespa = set_ivespa_obs(self.max_plume_height, 
                                        self.volcano_height)
                mer_gs = 10 ** ivespa.mu[0] * 1000
            this_dict = source_params_dict(
                "Source", self.volcano_height, self.max_plume_height,
                particles_hr_ht, mer_gs = mer_gs)
            list_dicts.append(this_dict)

        # Write to csv
        list_dicts_to_csv(list_dicts, esp_csv, overwrite = True)
        self.chunk_csv = esp_csv
        print("Saved chunk_data as " + self.chunk_csv)
        
    def run_chunks_in_name(self,
                           output_name: str,
                           input_dir: str,
                           output_dir: str,
                           ensemble: bool,
                           sub_dir: str = None,
                           rerun: bool = False,
                           maininput_file: str = "maininput.txt",
                           **kwargs):
        """Run NAME with chunks as separate sources in the same run. If output directory already exists and NAME runs have been completed, does not re-run. 

        Args:
            output_name (str): Name for VolcanicNAME object.
            input_dir (str): Directory containing NAME input files.
            output_dir (str): Directory to place NAME output files.
            ensemble (bool): Whether ensemble met is used. If True, runs NAME once per ensemble member. If False, runs NAME once with deterministic met.
            sub_dir (str): Subdirectory within output_dir. Defaults to None.
            rerun (bool): If NAME output directory already exists and runs have finished, whether to re-run NAME. Defaults to False.
            maininput_file (str): Name of maininput file.

        Returns:
            VolcanicNAME: Object containing information on NAME run(s).
        """
        if not rerun:
            full_output_dir = (
                output_dir  + "/" + sub_dir if sub_dir is not None 
                else output_dir)
            if os.path.exists(full_output_dir):
                self.name_output = VolcanicNAME(
                    output_name, 
                    ensemble = ensemble,
                    output_dir = full_output_dir, 
                    esp_csv = self.chunk_csv,
                    maininput_file = maininput_file)
                try:
                    if self.name_output.check_run_finished():
                        print(f"NAME runs finished in {full_output_dir}. " 
                            + "Set rerun=True to set up new runs.")
                        return self.name_output
                    else:
                        print(f"NAME runs not finished in {full_output_dir}.")
                except OSError:
                    pass
            
        self.name_output = run_name_from_csv(
            self.chunk_csv, 
            output_name, 
            input_dir, 
            output_dir, 
            ensemble, 
            sub_dir, 
            return_obj = True,
            maininput_file = maininput_file,
            **kwargs)
        
        return self.name_output
        
class PHQuadrature(object):
    name: str
    
    chunks: SourceChunks
    
    lower_km: float
    upper_km: float
    
    truncnorm: rv_continuous

    mu: np.ndarray
    Sigma: np.ndarray
    
    exc_probs: iris.cube.Cube

    def __init__(self, 
                 chunks: SourceChunks,
                 lower_km: float = None,
                 upper_km: float = None,
                 name: str = None,
                 **kwargs):
        """Initialise PHQuadrature object for fast evaluation of ash concentration exceedance probabilities via Gauss-Kronod quadrature. Unless sample_heights are specified, n_samples number of plume height samples are drawn from a truncated normal distribution. The default parameters of the distribution are determined by the min and max plume height in SourceChunks.

        Args:
            chunks (SourceChunks): SourceChunks object with child VolcanicNAME containing NAME outputs information.
            sample_heights (list, optional): List of plume height samples. If None, draws n_samples plume height samples from a truncated normal distribution. Defaults to None. 
            n_samples (int, optional): Number of plume height samples to be drawn from truncated normal distribution. 
            name (str, optional): Name of PHSamples object, to be assigned to Title attribute of cubes. Defaults to None.

        Raises:
            ValueError: SourceChunks is not an ensemble.
        """
        self.name = name

        if chunks.name_output.ensemble:
            self.chunks = chunks
        else:
            raise ValueError("chunks NAME output should be an ensemble.")
        
        # Get parameters of truncated normal distribution - if not specified, defaults to a distribution with mean given by the average of the min and max plume height, endpoints the min and max, and scale 1
        chunks_min = chunks.min_plume_height / 1000
        chunks_max = chunks.max_plume_height / 1000
        if lower_km is None:
            self.lower_km = chunks_min
        else:
            self.lower_km = lower_km if lower_km > chunks_min else chunks_min
        if upper_km is None:
            self.upper_km = chunks_max
        else:
            self.upper_km = upper_km if upper_km < chunks_max else chunks_max
        loc = kwargs.pop("loc", (self.lower_km + self.upper_km) / 2)
        scale = kwargs.pop("scale", 1)
        a = (self.lower_km - loc) / scale
        b = (self.upper_km - loc) / scale
        self.truncnorm = truncnorm(a, b, loc = loc, scale = scale)

    
    def _eval_cdf(self, 
                  height_asl: float,
                  thresholds: tuple = (2E-4, 2E-3, 5E-3, 1E-2)):
        mean_cdf_cubes = eval_t_cdf(height_asl, self.chunks, thresholds)
        return mean_cdf_cubes.data * self.truncnorm.pdf(height_asl)
    
           
    def quad_exc_prob(self, 
                      cubes_dir: str, 
                      n_points: int = None,
                      points: list = None, 
                      workers: int = -1, 
                      limit: int = 24,
                      file_name: str = None,
                      z_level = None, 
                      time_inds = None, 
                      lat_minmax: tuple = (30.0, 80.0), 
                      lon_minmax: tuple = (-40.0, 40.0), 
                      thresholds: tuple = (2E-4, 2E-3, 5E-3, 1E-2),
                      **kwargs):
        """Use quadrature to evaluate ash concentration exceedance probabilities for a set of thresholds. Provides a deterministic approximation to the exceedance probabilties given the model, rather than a stochastic estimate.

        Args:
            cubes_dir (str): Directory to save/load ash concentration cubes to/from and save result to.
            n_points (int, optional): Number of additional breakpoints for intial intervals. If None, will match the number of workers. Defaults to None.
            points (list, optional): List of initial interval breakpoints. If None, will be n_points number of equally-spaced points. Defaults to None.
            workers (int, optional): Number of cores to use for parallel computation. Defaults to -1 to utilise all available cores.
            limit (int, optional): Limit on number of intervals. Defaults to 24.
            file_name (str, optional): Name of NetCDF file to save output to. Defaults to None.
            time_inds: (list, optional): Time constraint, specified as a range 
            between 0 and the dimension of time. Defaults to None.
            lat_minmax: (tuple, optional): Latitude constraint. Defaults to 
            (30.0, 80.0).
            lon_minmax: (tuple, optional): Longitude constraint. Defaults to 
            (-40.0, 40.0).
            thresholds (tuple, optional): Ash concentration thresholds whose exceedance probabilities are to be estimated. Defaults to (2E-4, 2E-3, 5E-3, 1E-2).
        """
        cubes_dir += "/" if not cubes_dir[-1] == "/" else ""
        
        # Construct list of ash concentration cubes for each ensemble member
        self.chunks.list_member_chunks = []
        for member in range(self.chunks.name_output.n_members):
            ash_chunks_filename = cubes_dir + f"ash_chunks_member_{member}.nc"
            # Load ash cube list and constrain
            self.chunks.list_member_chunks.append(
                self.chunks.name_output.load_ash_cube_list(
                    ash_chunks_filename, member, z_level = z_level,
                    time_inds = time_inds, lat_minmax = lat_minmax,
                    lon_minmax = lon_minmax, **kwargs))
            
        if points is None:
            if n_points is not None:
                points = np.linspace(self.lower_km, self.upper_km, 
                                     num = n_points + 1)
            elif workers == -1: 
                points = np.linspace(self.lower_km, self.upper_km, 
                                     num = cpu_count() + 1)
            elif workers > 1:
                points = np.linspace(self.lower_km, self.upper_km,
                                     num = workers + 1)
            
        min_intervals = 1 if n_points == 1 else 2
        start = time()
        res, err, info = quad_vec(
            self._eval_cdf, a = self.lower_km, b = self.upper_km,
            points = points, workers = workers, limit = limit, 
            quadrature = "gk21", full_output = True, 
            min_intervals = min_intervals, args = (thresholds, ))
        end = time()
        elapsed = end - start

        attrs_dict = {"Quantity": "Exceedance Probability",
                      "Release height min": f"{self.lower_km * 1000} km",
                      "Release height max": f"{self.upper_km * 1000} km",
                      "Convergence": str(info.__dict__["success"])}
        if self.name is not None:
            attrs_dict.update({"Title": self.name})

        file_name = (
            cubes_dir + "quad_exc_prob_cube.nc" if file_name is None 
            else cubes_dir + file_name)        
        
        # Construct cube of results
        self.exc_prob_cube = construct_new_cube(
            data = 1 - res, # convert to exc probs
            cube = self.chunks.list_member_chunks[0][0].copy(),
            attrs = attrs_dict,
            new_dim_coords = [DimCoord(thresholds, long_name = "threshold",
                                        units = "g/m3")],
            new_aux_coords = [AuxCoord(err, long_name = "error", units = ""),
                              AuxCoord(info.__dict__["neval"], 
                                       long_name = "num_eval", units = ""),
                              AuxCoord(elapsed, long_name = "eval_time", units = "s")],
            save_file = file_name,
            long_name = "Volcanic Ash Air Concentration Exceedance Probability",
            units = "")
        
    
           
class PHSamples(object):
    name: str
    
    chunks: SourceChunks
    
    sample_heights: list
    n_samples: int
    
    cubes_dir: str
    cubes_title: str
    
    exc_probs: iris.cube.Cube
    var_ests: iris.cube.Cube

    def __init__(self, 
                 chunks: SourceChunks,
                 sample_heights: list = None,
                 n_samples: int = None,
                 name: str = None,
                 **kwargs):
        """Initialise PHSamples object for fast estimation of ash concentration exceedance probabilities. Unless sample_heights are specified, n_samples number of plume height samples are drawn from a truncated normal distribution. The default parameters of the distribution are determined by the min and max plume height in SourceChunks.

        Args:
            chunks (SourceChunks): SourceChunks object with child VolcanicNAME containing NAME outputs information.
            sample_heights (list, optional): List of plume height samples. If None, draws n_samples plume height samples from a truncated normal distribution. Defaults to None. 
            n_samples (int, optional): Number of plume height samples to be drawn from truncated normal distribution. 
            name (str, optional): Name of PHSamples object, to be assigned to Title attribute of cubes. Defaults to None.

        Raises:
            ValueError: SourceChunks is not an ensemble.
        """
        self.name = name

        if chunks.name_output.ensemble:
            self.chunks = chunks
        else:
            raise ValueError("chunks NAME output should be an ensemble.")
            
        if sample_heights is None:
            # Get parameters of truncated normal distribution - if not specified, defaults to a distribution with mean given by the average of the min and max plume height, endpoints the min and max, and scale 1
            loc = kwargs.pop("loc", ((chunks.min_plume_height +
                                     chunks.max_plume_height) / 2) / 1000)
            scale = kwargs.pop("scale", 1)
            a = kwargs.pop("a", (chunks.min_plume_height / 1000 - loc) / scale)
            b = kwargs.pop("b", (chunks.max_plume_height / 1000 - loc) / scale)
            print(f"Drawing {n_samples} plume height samples (km) from " +
                  f"truncated normal distribution with parameters loc={loc}, " +
                  f"a={a}, b={b} and scale={scale}.")

            # Set seed
            seed = kwargs.pop("seed", None)
            np.random.seed(seed)
            # Sample plume heights and convert back to m
            sample_heights = 1000 * truncnorm.rvs(a = a, b = b, loc = loc, 
                                                  size = n_samples, 
                                                  scale = scale)
        elif n_samples is None:
            raise ValueError("One of sample_heights or n_samples must be " +
                             "provided.")
            
        sample_heights.sort()
        self.sample_heights = sample_heights
        self.n_samples = len(self.sample_heights)
        self.ivespa = set_ivespa_obs(self.sample_heights, 
                                             self.chunks.volcano_height)
        
    def get_member_probs(
        self, 
        member: int,
        thresholds: list = [2E-4, 2E-3, 5E-3, 1E-3],
        **kwargs):
        """For a given ensemble member, evaluate the exceedance probabilities of given ash concentration thresholds for each plume height sample and take the average of these to estimate the exceedance probability given this met instance. 

        Args:
            member (int): Index of ensemble member.
            thresholds (list, optional): Ash concentration thresholds whose exceedance probabilities are to be evaluated.. Defaults to [2E-4, 2E-3, 5E-3, 1E-3].

        Returns:
            tuple: Cube estimating ash concentration exceedance probability for this met member, and CubeList of separate exceedance probabilities given met for each sample.
        """

        # Load data if already exists, otherwise get from NAME output and save
        ash_chunks_filename = self.cubes_dir + f"ash_chunks_member_{member}.nc"
        chunks_member_cube_list = self.chunks.name_output.load_ash_cube_list(
            ash_chunks_filename, member, **kwargs)

        # Initialise cumulative sums of probabilities
        sum_prob_cube = None
        sample_probs_list = CubeList([])
        
        # Evaluate conditional exceedance probability given this member for 
        # each sample in this stratum
        for i in range(self.n_samples):
            # Get plume height and MER
            height_asl = self.sample_heights[i]
            mer_gs = 10 ** self.mu[i] * 1000
            mer_gs_ht = mer_gs / (height_asl - self.chunks.volcano_height)
            
            rescaled_cube = reconstruct_cube_from_list(
                cube_list = chunks_member_cube_list, 
                plume_height = height_asl,
                new_mer_gs_ht = mer_gs_ht,
                new_title = f"Sample {i}")
                                
            # Get log-ash concentration cube
            log_cube = rescaled_cube.copy()
            log_cube.data = np.log10(rescaled_cube.data) 

            # Deal with masked cube for NaN values
            if isinstance(log_cube.data, np.ma.core.MaskedArray):
                log_cube.data = log_cube.data.filled(-np.inf)
                                    
            # Evaluate exceedance probability cube
            prob_cube = member_prob_cube(
                log_cube, 
                df = df, 
                scale = self.Sigma[i],
                h_km = height_asl / 1000, 
                thresholds = thresholds)

            if i == 0:
                sum_prob_cube = prob_cube.copy()
            else:
                sum_prob_cube = maths.add(sum_prob_cube, prob_cube)

            print(f"Member {member} sample {i} done.", 
                    flush = True)

            sample_probs_list.append(prob_cube)

        # Average to obtain conditional exceedance probability estimates for 
        # this met member
        member_probs = sum_prob_cube / self.n_samples

        return member_probs, sample_probs_list

    def get_exc_probs(
        self, 
        cubes_dir: str,
        thresholds: tuple = (2E-4, 2E-3, 5E-3, 1E-2),
        sample_var: bool = True,
        **kwargs):
        """Estimate ash concentration exceedance probabilities and the sample variance of estimtaes.

        Args:
            cubes_dir (str): Directory to save ash concentration cubes to.
            thresholds (tuple, optional): Ash concentration thresholds whose exceedance probabilities are to be estimated. Defaults to (2E-4, 2E-3, 5E-3, 1E-2).
            sample_var (bool, optional): Whether to also evaluate sample variances. Defaults to True.
        """
        # Initialise directory for outputs
        cubes_dir += "/" if not cubes_dir[-1] == "/" else ""
        self.cubes_dir = cubes_dir

        if not os.path.exists(self.cubes_dir):
            os.mkdir(self.cubes_dir)

        # Initialise lists of cumulative sums
        if sample_var:
            sum_sample_probs = CubeList([])
        
        n_members = self.chunks.name_output.n_members
        for m in range(n_members):
            member_prob_cube, sample_member_probs_list = self.get_member_probs(
                m, thresholds, **kwargs)

            # Iteratively evaluate exceedance probability estimate
            if m == 0:
                sum_probs = member_prob_cube.copy()
            else:
                sum_probs = maths.add(sum_probs, member_prob_cube)

            if sample_var:
                for i in range(self.n_samples):
                    sample_prob_cube = sample_member_probs_list[i].copy()
                    # Iteratively evaluate exceedance probability for each sample
                    if m == 0:
                        sum_sample_probs.append(sample_prob_cube)
                    else:
                        sum_sample_probs[i] = maths.add(
                            sum_sample_probs[i], sample_prob_cube)
                    
        # Save to netCDF
        sum_probs.attributes["Quantity"] = "Exceedance Probability Estimate"
        if self.name is not None:
            sum_probs.attributes["Title"] = self.name
        self.exc_probs = sum_probs / n_members
        exc_probs_filename = self.cubes_dir + "exc_probs.nc"
        iris.save(self.exc_probs, exc_probs_filename)
        print("Saved exceedance probability estimates to " +
              f"{exc_probs_filename}.")

        if sample_var:
            # Evaluate sum of squares in sample variance
            for i in range(self.n_samples):
                mean_sample_cube = sum_sample_probs[i] / n_members
                diff_cube = maths.add(mean_sample_cube, -self.exc_probs)
                if isinstance(diff_cube.data, np.ma.core.MaskedArray):
                    diff_cube.data = diff_cube.data.filled(0)
                
                diff_sqr = maths.multiply(diff_cube, diff_cube)
                if i == 0:
                    sum_sqr_diff = diff_sqr.copy()
                else:
                    sum_sqr_diff = maths.add(sum_sqr_diff, diff_sqr)

            # Normalise by N-1
            self.var_ests = sum_sqr_diff / (self.n_samples - 1)
            self.var_ests.attributes["Quantity"] = "Sample Variance"
            if self.name is not None:
                self.var_ests.attributes["Title"] = self.name
            sample_vars_filename = self.cubes_dir + "sample_vars.nc"
            iris.save(self.var_ests, sample_vars_filename)
            print("Saved variances of exceedance probability " + 
                f"estimates to {sample_vars_filename}.")

    def get_ci_cube(self, log: bool = True, alpha: float = 0.05):
        """Approximate 100(1-alpha)% confidence intervals for ash concentration exceedance probabilities.

        Args:
            log (bool, optional): Whether to evaluate CI for log-exceedance probabilities. Defaults to True.
            alpha (float, optional): Confidence level. Defaults to 0.05.
        """
        self.ci_cube = get_ci_cube(self.exc_probs, self.var_ests, 
                                   n_samples = self.n_samples, log = log,
                                   alpha = alpha)
        ci_filename = (self.cubes_dir + 
                       f"{'log_' if log else ''}" + 
                       f"ci_{int(100 * (1-alpha))}.nc")
        iris.save(self.ci_cube, ci_filename)
        print(f"Saved {100 * (1-alpha)}% CI cube to {ci_filename}.")

        
def run_name_from_csv(csv_path: str,
                      output_name: str,
                      input_dir: str,
                      output_dir: str,
                      ensemble: bool,
                      source_line_no: int,
                      out_line_no: int,
                      sub_dir: str = None,
                      script_dir: str = None,
                      scr_file: str = None,
                      maininput_file: str = "maininput.txt",
                      source_file: str = "sources.txt",
                      bash_file: str = "volcanic_ash.sh", 
                      print_out: bool = False,
                      return_obj: bool = False,
                      **kwargs):
    """From a csv file of eruption source parameters, construct and run NAMEwith separate sources specified by the file. Runs as deterministic (one NAME run) or an ensemble (number of runs determined by the met data specified in maininput_file).

    Args:
        csv_path (str): Full path of csv file containing eruption source parameters, e.g. output of posterior_to_csv.
        output_name (str): Name of VolcanicNAME object returned.
        input_dir (str): Directory containing NAME input files. Must contain a main input file and a source file.
        output_dir (str): Directory to store NAME outputs. Will be created if not pre-existing.
        ensemble (bool): Whether ensemble meteorological data is being used. If True, runs NAME once per ensemble member (where met is specified in maininput_file). If False, runs NAME only once with deterministic met.
        source_line_no (int): Line number in maininput_file where source parameters begin.
        out_line_no (int): Line number in maininput_file where output parameters begin.
        sub_dir (str, optional): Subdirectory of output_dir. Will be created if not pre-existing.
        script_dir (str, optional): Directory containing .scr file for setting up NAME. If None, defauilts to "../NAME_scripts". Defaults to None.
        scr_file (str, optional): Script (.scr) file within script_dir for setting up and running NAME. If None, defaults to the relevant file in NAME_scripts. Defaults to None.
        maininput_file (str, optional): Name of main NAME input file in script_dir. See "maininput.txt" for example setup. Defaults to "maininput.txt".
        source_file (str, optional): Name of NAME source input file in script_dir. See "scripts/sources.txt" for example setup. Defaults to "sources.txt".
        bash_file (str, optional): Name of bash file for running NAME in script_dir. Defaults to "volcanic_ash.sh".
        print_out (bool, optional): Whether to print to console. Defaults to False.
        return_obj (bool, optional): Whether to return VolcanicNAME object. Defaults to False.

    Raises:
        OSError: File script_dir/scr_file does not exist.

    Returns:
        VolcanicNAME: Object containing information on NAME run(s) and their outputs.
    """
    
    if script_dir is None:
        script_dir = ("/").join(
            os.path.realpath(os.path.dirname(__file__)).split("/")[:-1]
        ) + "/NAME_scripts"
    if scr_file is None:
        scr_file = ("chunk_ensemble_from_csv.scr" if ensemble 
                    else "chunk_deterministic_from_csv.scr")
    if not os.path.exists(f"{script_dir}/{scr_file}"):
        raise OSError(f"{scr_file} does not exist.")

    output_dir += "/" if not output_dir[-1] == "/" else ""
        
    # Set of commands to be sent to terminal
    commands_list = [
        f"export scriptDIR={script_dir};", 
        f"export inputDIR={input_dir};",
        f"export outDIR={output_dir};",
        f"export maininputfile={maininput_file};", 
        f"export sourcefile={source_file};",
        f"export bashfile={bash_file};", 
        f"export csv={csv_path};",
        f"export sourcelineno={source_line_no};",
        f"export outlineno={out_line_no};"
    ]
    
    if sub_dir is not None:
        commands_list.append(f"export subdirname={sub_dir};")
        name_dir = output_dir + sub_dir
    else:
        name_dir = output_dir
    
    commands_list.append(f"cd {script_dir};")
    commands_list.append(f"./{scr_file}")
    
    p = Popen((" ").join(commands_list), shell = True, stdout = PIPE, 
              **kwargs)
    out, _ = p.communicate()
    if print_out:
        print(out)

    if return_obj:
        name_output = VolcanicNAME(output_name, ensemble, name_dir, 
                                   csv_path, maininput_file)
        return name_output

def eval_t_cdf(height_asl: float, chunks: SourceChunks,
               thresholds: tuple = (0.2E-3, 2E-3, 5E-3, 10E-3)):
    # Heights in km asl to avl for getting params of t-dist
    if chunks.volcano_height > 0.0:         
        height_avl = height_asl - (chunks.volcano_height / 1000)
    else:
        height_avl = height_asl
        
    # Get params from MERPH model
    ivespa = set_ivespa_obs(height_avl * 1000, chunks.volcano_height)

    # Evaluate MER for this height
    mer_gs = 10 ** ivespa.mu[0] * 1000
    mer_gs_ht = mer_gs / (height_avl * 1000)
    
    n_members = chunks.name_output.n_members
    for member in range(n_members):
        # Get CubeList
        chunks_member_cube_list = chunks.list_member_chunks[member]
        
        # Get rescaled cube
        rescaled_cube = reconstruct_cube_from_list(
            cube_list = chunks_member_cube_list, 
            plume_height = height_asl * 1000,
            new_mer_gs_ht = mer_gs_ht)
        
        # Get log-ash concentration cube
        log_cube = rescaled_cube.copy()
        log_cube.data = np.log10(rescaled_cube.data) 

        # Deal with masked cube for NaN values
        if isinstance(log_cube.data, np.ma.core.MaskedArray):
            log_cube.data = log_cube.data.filled(-np.inf)
                                
        # Evaluate exceedance probability cube
        cdf_cube = member_prob_cube(
            log_cube, 
            df = ivespa.df, 
            scale = ivespa.Sigma[0],
            h_km = height_asl, 
            exceed = False,
            thresholds = thresholds)

        # Initialise or add to cumulative sum
        if member == 0:
            sum_cdf_cubes = cdf_cube.copy()
        else:
            sum_cdf_cubes = maths.add(sum_cdf_cubes, cdf_cube)
            
    return sum_cdf_cubes / n_members
