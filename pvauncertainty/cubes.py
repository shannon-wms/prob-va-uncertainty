"""
Script containing functions for dealing with iris cubes for ash concentration 
exceedance probability evaluation and estimation.

Author: Shannon Williams

Date: 25/09/2023
"""

import os
import iris
import iris.analysis.maths as maths
import numpy as np
import pandas as pd
from scipy.stats import rv_continuous, t, norm
from iris.cube import Cube, CubeList
from pvauncertainty.utils import bisection, mixture_dist

# Avoid "divide by zero encountered in log10" warnings
import warnings
warnings.filterwarnings("ignore", category = RuntimeWarning)


def get_cube(dir: str, 
             file_root: str = "Fields_grid99*", 
             field_options: dict = {"Species": "VOLCANIC_ASH",
                                    "Quantity": "Air Concentration"},
             z_level: list = None, 
             time_inds: list = None,
             lat_minmax: tuple = (30.0, 80.0), 
             lon_minmax: tuple = (-40.0, 40.0),
             callback: callable = None):
    """Read selected fields from NAME file according to flight level.

    Args:
        dir (str): The directory in which NAME output files are located.
        file_root (str): A text string common to all files to be loaded. Defaults to "Fields_grid99*".
        field_options (dict, optional): Dictionary containing information about columns to be loaded. Defaults to None.
        z_level (list, optional): List of flight levels to constrain by. Defaults to None.
        time_inds: (list, optional): Time constraint, specified as a range between 0 and the dimension of time. Defaults to None.
        lat_minmax: (tuple, optional): Latitude constraint. Defaults to (30.0, 80.0).
        lon_minmax: (tuple, optional): Longitude constraint. Defaults to (-40.0, 40.0).
        callback (callable, optional): A function to add/remove metadata from the cube in iris.load_cube(). Must have signature (cube, field, filename).

    Returns:
        iris.cube.Cube: Corresponding cube.
    """

    att_spec = ({} if field_options == None else field_options)
    att_constraint = iris.AttributeConstraint(**att_spec)

    if z_level is None:
        try:
            cube = iris.load_cube(dir + "/" + file_root, att_constraint, 
                                  callback)
        except OSError:
            print(f"No files matched in {dir}.")
    else: # Constrain by flight level
        if not isinstance(z_level, list):
            z_level = [z_level]
        level_constraint = iris.Constraint(
            flight_level = lambda cell: cell.point in z_level)
        try:
            cube = iris.load_cube(dir + "/" + file_root, 
                                  att_constraint & level_constraint, callback)
        except OSError:
            print(f"No files matched in {dir}.")

    cube = constrain_cube(cube, 
                          time_inds = time_inds, 
                          lat_minmax = lat_minmax,
                          lon_minmax = lon_minmax)
    
    return cube

def constrain_cube(cube: Cube, 
                   z_level: list = None, 
                   time_inds: list = None,
                   lat_minmax: tuple = None, 
                   lon_minmax: tuple = None):
    """Constrain a cube according to flight level, time, latitude, and longitude.

    Args:
        cube (Cube): Iris cube.
        z_level (list, optional): List of flight levels to constrain by. Defaults to None.
        time_inds: (list, optional): Time constraint, specified as a range between 0 and the dimension of time. Defaults to None.
        lat_minmax: (tuple, optional): Latitude constraint. Defaults to (30.0, 80.0).
        lon_minmax: (tuple, optional): Longitude constraint. Defaults to (-40.0, 40.0).

    Returns:
        Cube: Constrained cube.
    """
    # Subset the cube by latitude and longitude
    if lat_minmax is not None:
        cube = cube.intersection(latitude = lat_minmax)
    if lon_minmax is not None:
        cube = cube.intersection(longitude = lon_minmax)

    # Subset by flight level
    if z_level is not None:
        if not isinstance(z_level, list):
            z_level = [z_level]
        cube = cube.extract(iris.Constraint(
            flight_level = lambda cell: cell.point in z_level))

    # Subset by time if specified
    if time_inds is not None:
        if not isinstance(time_inds, list):
            time_inds = [time_inds]
        t_coord = cube.coord("time")
        times = [t_coord.units.num2date(t_coord.points[i]) for i in time_inds]
        cube =  cube.extract(iris.Constraint(time = lambda t: t.point in times))

    return cube

def get_log_cube(cube: Cube):
    """Get log-ash concentration cube from directory.

    Args:
        cube (Cube): Ash concentration cube.

    Returns:
        Cube: Log-ash concentration cube.
    """
    
    # Take the log-concentration
    log_cube = cube.copy()
    log_cube.data = np.log10(cube.data)
    if isinstance(log_cube.data, np.ma.core.MaskedArray):
        log_cube.data = log_cube.data.filled(-np.inf)

    # Update attributes
    log_cube.units = ""
    log_cube.attributes["Quantity"] = "Log-ash concentration"

    return log_cube

def member_prob_cube(log_cube: Cube, 
                     h_km: float = None, 
                     exceed: bool = True,
                     distribution: rv_continuous = t,
                     thresholds: tuple = (0.2E-3, 2E-3, 5E-3, 10E-3),
                     **kwargs):
    """Evaluate exceedance probability given (log-)ash concentration cube for a single met and plume height instance. 

    Args:
        log_cube (Cube): Logarithm of ash concentration cube.
        h_km (float, optional): Plume height in km a.s.l. Defaults to None.
        exceed (bool, optional): Whether to evaluate exceedance (1 - cdf) of thresholds or evaluate just the cdf. Defaults to True.
        distribution (rv_continuous, optional): The distribution to evaluate the exceedance probability. Defaults to t.
        thresholds (tuple, optional): Ash concentration thresholds whose exceedance probabilities are to be evaluated.. Defaults to (0.2E-3, 2E-3, 5E-3, 10E-3).
    Returns:
        CubeList: Exceedance probability cube, 
    """
    # Initialise cube lists
    prob_cube_list = CubeList([])
        
    # Compute exceedance probability for each threshold
    for threshold in thresholds:
        # Evaluate exceedance probabilities using posterior predictive c.d.f
        prob_cube =  log_cube.copy()
        prob_cube.data = distribution.cdf(np.log10(threshold), loc = log_cube.data, **kwargs)
        if exceed:
            prob_cube = 1 - prob_cube

        # Additional coordinate for threshold
        new_coord = iris.coords.AuxCoord(threshold, long_name = "threshold", 
                                         units = "g/m3")
        prob_cube.add_aux_coord(new_coord)
        
        # Update attributes
        prob_cube.units = ""
        prob_cube.attributes["Quantity"] = "Exceedance Probability"

        if h_km is not None:
            prob_cube.attributes["Plume Height"] = str(h_km) + " km"
        
        prob_cube_list.append(prob_cube)
            
    # Merge list to single cube with threshold coordinate
    return prob_cube_list.merge_cube()

def get_prob_cube(dir: str, 
                  h_km: float = None, 
                  thresholds: tuple = (2E-4, 2E-3, 5E-3, 1E-2),
                  percentiles: bool = False, 
                  p_points: tuple = (0.05, 0.5, 0.95),
                  ppf_interval: tuple = (-6.0, 1.0), 
                  return_cubes: bool = True,
                  save_excprob_members: bool = False, 
                  save_ppf_members: bool = False, 
                  excprob_file: str = None, 
                  ppf_file: str = None, 
                  verbose: bool = True, 
                  distribution: rv_continuous = t,
                  **kwargs):
    """Evaluate exceedance probabilities of an ensemble of NAME runs, using a t-distribution for the log-ash concentration, and also evaluate percentiles.

    Args:
        dir (str): Directory of NAME ensemble member outputs, where each subdirectory begins with "member".
        df (int): Degrees of freedom of t-distribution for computing exceedance probabilities.
        h_km (float, optional): Plume height in km a.s.l. Defaults to None.
        thresholds (tuple, optional): List of ash concentration thresholds whose exceedance probabilities are to be evaluated. Defaults to (2E-4, 2E-3, 5E-3, 1E-2).
        percentiles (bool, optional): Whether to compute percentile cube. Defaults to False.
        p_points (tuple, optional): Percentile points to be evaluated, if percentiles is True. Defaults to (0.05, 0.5, 0.95).
        ppf_interval (tuple, optional): Initial estimate for the range of the percentiles of the log-concentration distribution. Assumes percentile belongs to one of the endpoints if initial guess is not inside of this interval. Defaults to (-6.0, 1.0).
        return_cubes (bool, optional): Whether to return cubes as a CubeList. Defaults to True.
        save_excprob_members (bool, optional): Whether to save exceedance probability cubes as netCDF. Defaults to False.
        save_ppf_members (bool, optional): Whether to save percentile cubes as netCDF, if calculated. Defaults to False.
        excprob_file (str, optional): Name of file for exceedance probability cubes to be saved to. Defaults to None.
        ppf_file (str, optional): Name of file for percentile cubes to be saved to. Defaults to None.
        verbose (bool, optional): Whether to print to console. Defaults to True.

    Returns:
        list: CubeList containing exceedance probabilities, and CubeList of percentiles if also evaluated.
    """
    member_dirs = tuple(f.path for f in os.scandir(dir) if f.is_dir() 
                        and f.name.startswith("member"))
        
    if percentiles: # Initialise list of percentile cubes also
        log_cube_list = CubeList([])
        
    n_members = len(member_dirs)
    sum_prob_cube = None

    for member in range(len(member_dirs)):
        this_subdir = member_dirs[member]
        cube = get_cube(this_subdir, **kwargs)
        log_cube = get_log_cube(cube)

        # Evaluate exceedance probability for this member
        prob_cube = member_prob_cube(log_cube, h_km, thresholds, distribution,
                                     **kwargs)
        # Add to cumulative sum of probabilities
        if member == 0:
            sum_prob_cube = prob_cube.copy()
        else:
            sum_prob_cube = maths.add(sum_prob_cube, prob_cube)

        # Save to netcdf
        if save_excprob_members:
            member_excprob_file = this_subdir + "/exc_prob_cube.nc"
            iris.save(prob_cube, member_excprob_file)
                
        # Save log-concentration cube to compute mixture cdf later
        if percentiles:             
            # Additional coordinate for ensemble member
            new_coord = iris.coords.AuxCoord(member, long_name = "member", 
                                             units = "")
            log_cube.add_aux_coord(new_coord)

            # Update attributes
            log_cube.units = ""
            log_cube.attributes["Quantity"] = "Log-ash Concentration"

            if h_km is not None:
                log_cube.attributes["Plume Height"] = str(h_km) + " km"

            log_cube_list.append(log_cube)

            # If we want to compute percentile points for each member, we only 
            # need the ppf of the t-distribution
            if save_ppf_members:
                for ppf in p_points:
                    ppf_cube = log_cube.copy()
                    # Evaluate percentile
                    ppf_cube.data = 10 ** distribution.ppf(
                        ppf, loc = log_cube.data, **kwargs)

                    # Additional coordinate for percentile
                    new_coord = iris.coords.AuxCoord(ppf, 
                                                     long_name = "percentile", 
                                                     units = "")
                    ppf_cube.add_aux_coord(new_coord)
                    
                    # Update attributes
                    ppf_cube.units = ""
                    ppf_cube.attributes["Quantity"] = "Percentile Point"

                    if h_km is not None:
                        ppf_cube.attributes["Plume Height"] = str(h_km) + " km"

                    member_ppf_file = this_subdir + "/ppf_cube.nc"
                    iris.save(ppf_cube, member_ppf_file)

    # Average the probabilities across ensemble members for each threshold
    mean_prob_cube = sum_prob_cube.merge_cube() / n_members

    # Save as netcdf
    if excprob_file is not None:
        iris.save(mean_prob_cube, excprob_file)
        if verbose:
            print("Saved exceedance probability cube as " + excprob_file)

    # Percentile points of the mixture distribution
    if percentiles:
        iris.util.equalise_attributes(log_cube_list)
        log_cube = log_cube_list.merge_cube()
        ppf_cube = get_ppf_cube(log_cube, distribution, interval = ppf_interval, **kwargs)

        # Save as netcdf
        if ppf_file is not None:
            iris.save(ppf_cube, ppf_file)
            if verbose:
                print("Saved percentiles cube as " + ppf_file)
        out = (mean_prob_cube, ppf_cube)
    else:
        out = mean_prob_cube

    if return_cubes:
        return out

def avg_prob_cubes(work_dir: str, 
                   esp_csv: str = None,
                   df: int = None, 
                   scale: tuple = None,
                   h_km: tuple = None, 
                   sample_var: bool = False, 
                   save_cubes: bool = False, 
                   percentiles = False, 
                   verbose = True, 
                   **kwargs):
    """For a set of plume height samples, average the exceedance probabilities and evaluate the sample variance of these probability estimates.

    Args:
        work_dir (str): Output directory containing NAME outputs. Subdirectories must begin with "sample".
        esp_csv (str, optional): Path of csv file from which to obtain eruption source parameters and t-distribution parameters. If not specified, these
        must be provided separately. Defaults to None.
        df (int, optional): Degrees of freedom of t-distribution for computing exceedance probabilities. Defaults to None.
        scale (tuple, optional): Scale of t-distribution.. Defaults to None.
        h_km (tuple, optional): List of plume height samples, in km a.s.l. Defaults to None.
        sample_var (bool, optional): Whether to compute sample variance. Defaults to False.
        save_cubes (bool, optional): Whether to save cubes to netCDF. Defaults to False.
        percentiles (bool, optional): Whether to compute percentile cube. Defaults to True.
        verbose (bool, optional): Whether to print to console. Defaults to True.

    Raises:
        ValueError: t-distribution not specified properly.

    Returns:
        CubeList: Exceedance probability estimates and associated sample variances.
    """
    # List subdirectories to obtain samples
    sample_dirs = tuple(f.path for f in os.scandir(work_dir) if f.is_dir() 
                        and f.name.startswith("sample"))
    n_samples = len(sample_dirs)
    if verbose:
        print("Number of samples: " + str(n_samples))

    # Get t-distribution data
    if esp_csv is not None:
        data = pd.read_csv(esp_csv)
        df = tuple(data["df"])
        h_km = tuple(data["H (km asl)"])
        scale = tuple(data["sigma"])
    elif df is None or h_km is None or scale is None:
        raise ValueError("Either esp_csv or all of df, h_km and scale must "
                         + "be specified.")

    if not isinstance(scale, tuple):
        scale = tuple(scale for _ in range(len(h_km)))

    for i in range(n_samples):
        if verbose:
            print("Sample {i} of {n}".format(i = i+1, n = n_samples))

        # File names for saving
        this_dir = sample_dirs[i]
        if save_cubes:
            excprob_file = this_dir + "/exc_prob_cube.nc"
            ppf_file = this_dir + "/ppf_cube.nc"
        else:
            excprob_file = None
            ppf_file = None

        # Evaluate conditional exceedance probability for this member        
        output = get_prob_cube(dir = this_dir, 
                               df = df[i], 
                               scale = scale[i], 
                               h_km = h_km[i], 
                               percentiles = percentiles, 
                               excprob_file = excprob_file, 
                               ppf_file = ppf_file,
                               verbose = verbose,
                               **kwargs)
        excprob_cube = output[0] if percentiles else output

        # Initialise cumulative sums
        if i == 0: 
            # Cumulative sum of probs
            sum_prob_cube = excprob_cube.copy()
            if sample_var: # Cumulative sum of squared probs
                sum_prob_sqr = maths.multiply(excprob_cube, excprob_cube)
        else: # Add to sums
            sum_prob_cube = maths.add(sum_prob_cube, excprob_cube)
            if sample_var: 
                prob_sqr = maths.multiply(excprob_cube, excprob_cube)
                sum_prob_sqr = maths.add(sum_prob_sqr, prob_sqr)
    
    # Average exceedance probs
    mean_prob_cube = sum_prob_cube / n_samples
    
    # Compute sample variance
    if sample_var: 
        mean_sqr = (maths.multiply(sum_prob_cube, sum_prob_cube) 
                    / n_samples)
        
        sample_var_cube = maths.add(sum_prob_sqr / (n_samples - 1), 
                                    - mean_sqr) 
        # sample_var_cube /= (n_samples - 1)

        sample_var_cube.units = ""
        sample_var_cube.attributes["Quantity"] = "Sample Variance"

    # Save cubes to netCDF
    if save_cubes:
        mean_file = work_dir + "/mean_exc_prob_cubes.nc"
        iris.save(mean_prob_cube, mean_file)
        if verbose:
            print("Saved mean cube as " + mean_file)
            
        if sample_var:
            var_file = work_dir + "/svar_exc_prob_cubes.nc"
            iris.save(sample_var_cube, var_file)
            if verbose:
                print("Saved variance cube as " + var_file)
                
    if sample_var:
        return mean_prob_cube, sample_var_cube
    else:
        return mean_prob_cube

def get_rel_var(prob_cube: Cube, svar_cube: Cube):
    """Compute relative variance (sample variance / square of probability) given exceedance probability estimate cube and sample variance cube.

    Args:
        prob_cube (Cube): Cube containing exceedance probability estimates.
        svar_cube (Cube): Associated sample variance cube.

    Returns:
        Cube: Relative variance cube.
    """
    prob_sqr = maths.multiply(prob_cube, prob_cube)
    relvar_cube = maths.divide(svar_cube, prob_sqr)
    
    # Update attributes
    relvar_cube.units = ""
    relvar_cube.attributes["Quantity"] = "Relative Variance"

    return relvar_cube

def get_ppf_cube(log_cube: Cube, 
                 scale: float, 
                 percentiles: tuple = (0.05, 0.5, 0.95), 
                 interval: tuple = (-6, 1), 
                 n_iter: int = 10, 
                 tol: float = 1E-2,
                 distribution: rv_continuous = t,
                 **kwargs):
    """Evaluate percentiles of log-ash concentration distribution via bisection (root-finding) method.

    Args:
        log_cube (Cube): Log of ash concentration cube.
        scale (float): Scale parameter of location-scale distribution.
        percentiles (list, optional): Percentile points to evaluate. Defaults to (0.05, 0.5, 0.95).
        interval (tuple, optional): Estimate of interval in which the percentileof log-ash concentration belongs. Defaults to (-6, 1).
        n_iter (int, optional): Maximum number of iterations of bisection algorithm for finding the root. Defaults to 10.
        tol (float, optional): Tolerance parameter for acceptance. Defaults to 1E-2.
        distribution (rv_continuous, optional): Distribution to evaluate. Defaults to t.

    Returns:
        Cube: Cube of log-ash concentration percentiles.
    """
    # Initialise list of cubes
    ppf_cube_list = CubeList([])

    for p in range(len(percentiles)):
        # Initialise each cube
        ppf_cube_list.append(log_cube.extract(
            iris.Constraint(member = lambda cell: cell == 0)).copy())

        # Additional coordinate for percentile
        new_coord = iris.coords.AuxCoord(percentiles[p], 
                                         long_name = "percentile", units = "")
        ppf_cube_list[p].add_aux_coord(new_coord)
        
        # Update attributes
        ppf_cube_list[p].units = ""
        ppf_cube_list[p].attributes["Quantity"] = "Percentile Point"

    # Set equal weights
    n_members = len(log_cube.coord("member").points)
    weights = [1 / n_members] * n_members

    # Change behaviour when only one time coord
    t_points = log_cube.coord("time").points
    one_time = True if len(t_points) == 1 else False

    # Change behaviour when only one FL coord
    fl_points = log_cube.coord("flight_level").points
    one_fl = True if len(fl_points) == 1 else False
        
    for i in range(len(t_points)):
        for j in range(len(fl_points)):
            for k in range(len(log_cube.coord("latitude").points)):
                for l in range(len(log_cube.coord("longitude").points)):
                    if one_time and one_fl:
                        mus = log_cube.data[:, k, l]
                    elif one_time:
                        mus = log_cube.data[:, j, k, l]
                    elif one_fl:
                        mus = log_cube.data[:, i, k, l]
                    else:
                        mus = log_cube.data[:, i, j, k, l]

                    for p in range(len(percentiles)):
                        # Ash concentration is 0 in all cases
                        if all(mu == -np.inf for mu in mus):
                            if one_time and one_fl:
                                ppf_cube_list[p].data[k, l] = -np.inf
                            elif one_time:
                                ppf_cube_list[p].data[j, k, l] = -np.inf
                            elif one_fl:
                                ppf_cube_list[p].data[i, k, l] = -np.inf
                            else:
                                ppf_cube_list[p].data[i, j, k, l] = -np.inf
                        else:
                            # Compute pecentile
                            x = bisection(interval, mixture_dist, 
                                          c = percentiles[p], 
                                          n_iter = n_iter, tol = tol, 
                                          locs = mus, 
                                          scales = scale, 
                                          distribution = distribution,
                                          weights = weights,
                                          **kwargs)
                            # Add to cube
                            if one_time and one_fl:
                                ppf_cube_list[p].data[k, l] = x
                            elif one_time:
                                ppf_cube_list[p].data[j, k, l] = x
                            elif one_fl:
                                ppf_cube_list[p].data[i, k, l] = x
                            else:
                                ppf_cube_list[p].data[i, j, k, l] = x

    # Return merged cube
    return ppf_cube_list.merge_cube()

def get_ci_cube(prob_cube: Cube, svar_cube: Cube, n_samples: int, 
                alpha: float = 0.05, log: bool = False):
    """Evaluate 100(1-alpha)% confidence intervals for ash concentration exceedance probabilities.

    Args:
        prob_cube (Cube): Cube containing estimated ash concentration exceedance probabilities for ash concentration threshold(s).
        svar_cube (Cube): Sample variance cube corresponding to prob_cube with the same dimension.
        n_samples (int): Number of samples used to obtain exceedance probability estimates.
        alpha (float, optional): Confidence level. Defaults to 0.05.
        log (bool, optional): Whether to evaluate CI for log-exceedance probability. Defaults to False.

    Returns:
        Cube: Iris cube with 3-point confidence_level coordinate corresponding to the lower limit of the CI, the estimate (exceedance probability of log-probability) and upper limit of the CI.
    """
    #  CI for log-exceedance probability
    if log:
        log_prob_cube = prob_cube.copy()
        log_prob_cube.data = np.log10(log_prob_cube.data)

        if isinstance(log_prob_cube.data, np.ma.core.MaskedArray):
            log_prob_cube.data = log_prob_cube.data.filled(-np.inf)

        log_prob_cube.attributes["Quantity"] = "Log-Exceedance Probability"
        
    z_cube = svar_cube.copy() / n_samples
    
    z_alpha = norm.ppf(1 - alpha / 2)
    z_cube.data = z_alpha * np.sqrt(z_cube.data)
    if log:
        z_cube = maths.multiply(z_cube, 1 / prob_cube)

    # Construct CI cube
    x_cube = log_prob_cube if log else prob_cube
    ci_cube_list = CubeList([])
    levels = (alpha / 2, 0.5, 1 - alpha / 2)
    for i, x in enumerate((-1, 0, 1)):
        ci_cube_list.append(maths.add(x_cube, x * z_cube))
        # Additional coordinate for CI
        new_coord = iris.coords.AuxCoord(
            levels[i], long_name = "confidence_level", units = "")
        ci_cube_list[i].add_aux_coord(new_coord)

    # Return merged cube
    ci_cube = ci_cube_list.merge_cube()    
    ci_cube.attributes["Confidence Level"] = f"{int(100 * (1 - alpha))}%"
    ci_cube.attributes["Quantity"] = "Log-Exceedance Probability"

    return ci_cube
        
def sort_cube_list(cube_list: CubeList, 
                   sort_by_height: bool = False, 
                   sort_by_title: bool = False, 
                   sort_by_named: str = None,
                   split: bool = False,
                   str_split: str = " ",
                   split_ind: int = 0):
    """Sort list of cubes, either according to the height ("Release height max") attribute,  title or another named attribute. 

    Args:
        cube_list (CubeList): List of cubes to be sorted.
        sort_by_height (bool, optional): Whether to sort by height attribute. 
        Defaults to False.
        sort_by_title (bool, optional): Whether to sort by the suffix of the title attribute. Defaults to False.
        sort_by_named (str, optional): Named attribute to sort by. Defaults to None.
        split (bool, optional): Whether to split attribute by str_split. Defaults to False.
        str_split (str, optional): String to split attribute by. Defaults to " ".
        split_ind (int, optional): Index of split attribute to sort by. Defaults to 0.

    Returns:
        CubeList: Sorted list of cubes.
    """

    if sort_by_named is not None:
        if split:
            sort_by = [float(cube.attributes[sort_by_named].split(str_split)[split_ind]) 
                       for cube in cube_list]
        else:
            sort_by = [float(cube.attributes[sort_by_named]) for cube in cube_list]
    elif sort_by_height: # Sort according to release height
        sort_by = [float(
            cube.attributes["Release height max"].split(str_split)[0]) 
                   for cube in cube_list]
    elif sort_by_title: # Sort according to index in title
        sort_by = [int(cube.attributes["Title"].split(str_split)[-1]) 
                   for cube in cube_list]
    else: # Return unsorted list
        print("Returning unsorted list. Set either sort_by_height or "
              + "sort_by_title to True")
        return cube_list 
            
    # Return sorted list
    return CubeList([cube_list[i] for i in list(np.argsort(sort_by))])
    
def update_cube_attrs_from_dict(cube: Cube, attrs: dict):
    """Update attributes of a cube according to dictionary.

    Args:
        cube (Cube): Cube to update.
        attrs (dict): Dictionary where key is attribute name and value is the
        updated value of the attribute.

    Returns:
        Cube: Updated cube.
    """
    for key, value in attrs.items():
        cube.attributes[key] = value
    return cube

def update_cube_attrs_from_data(cube: Cube, esp_data: pd.DataFrame, i: int):
    """Update title, MER and release height attributes of a cube according to dataframe of eruption source parameters.

    Args:
        cube (Cube): Cube to update.
        esp_data (pd.DataFrame): Dataframe of eruption source parameters.
        i (int): Index of dataframe to update by.

    Returns:
        Cube: Updated cube.
    """
    cube.attributes["Title"] += " " + esp_data["label"][i]
    cube.attributes["Release rate"] = str(esp_data["Q (g s)"][i]) + " g/s"
    cube.attributes["Release height min"] = str(esp_data["min (m)"][i]) + " m"
    cube.attributes["Release height max"] = str(esp_data["max (m)"][i]) + " m"
    return cube
    
def construct_new_cube(data: np.ndarray, 
                       cube: Cube,
                       attrs: dict = None,
                       new_dim_coords: list = None,
                       new_aux_coords: list = None,
                       save_file: str = None,
                       **kwargs):
    """Construct new cube with given data, taking dimensions and attributes from cube and additional/replacement dimensions and attributes as specified in function call. 

    Args:
        data (np.ndarray): Data to be added to cube.
        cube (Cube): Cube to take dimensions and attributes from.
        attrs (dict, optional): Dictionary of attributes to be added/replaced. Defaults to None.
        new_dim_coords (list, optional): List of DimCoord objects to be added to the cube. Defaults to None.
        new_aux_coords (list, optional): List of AuxCoord objects to be added to the cube (scalar coordinates). Defaults to None.
        save_file (str, optional): Name of NetCDF file to save cube to. Defaults to None.

    Returns:
        Cube: New cube.
    """
    standard_name = kwargs.pop("standard_name", cube.standard_name)
    long_name = kwargs.pop("long_name", cube.long_name)
    units = kwargs.pop("units", cube.units)
    cell_methods = kwargs.pop("cell_methods", cube.cell_methods)
    
    if attrs is not None:
        cube = update_cube_attrs_from_dict(cube, attrs)
    
    # Construct list of coords with dim mappings
    list_dim_coords = []
    n_dim_coords = 0
    if new_dim_coords is not None:
        if not isinstance(new_dim_coords, list):
            new_dim_coords = [new_dim_coords]
        for i, coord in enumerate(new_dim_coords):
            list_dim_coords.append((coord, i))
            n_dim_coords += 1
    
    # Construct list of coords with scalar dim mappings
    list_aux_coords = []
    n_aux_coords = 0
    if new_aux_coords is not None:
        if not isinstance(new_aux_coords, list):
            new_aux_coords = [new_aux_coords]
        for i, coord in enumerate(new_aux_coords):
            list_aux_coords.append((coord, None))
            n_aux_coords += 1

    coord_names = [
        (x.standard_name if x.standard_name is not None 
         else x.long_name for x, _ in list_dim_coords + list_aux_coords)]
    
    # Add any other coords from template cube
    for i, coord in enumerate(cube.coords(dim_coords = False)):
        # Don't add if coord has been specified in function call
        if coord.standard_name in coord_names or coord.long_name in coord_names:
            continue
        list_aux_coords.append((coord, None))
        n_aux_coords += 1

    for i, coord in enumerate(cube.coords(dim_coords = True)):
        # Don't add if coord has been specified in function call
        if coord.standard_name in coord_names or coord.long_name in coord_names:
            continue
        list_dim_coords.append((coord, n_dim_coords))
        n_dim_coords += 1

    new_cube = Cube(data, 
                    standard_name = standard_name,
                    long_name = long_name, 
                    units = units, 
                    attributes = cube.attributes, 
                    cell_methods = cell_methods, 
                    dim_coords_and_dims = list_dim_coords, 
                    aux_coords_and_dims = list_aux_coords)
    
    if save_file is not None:
        iris.save(new_cube, save_file)
        print(f"Saved cube to {save_file}.")
    
    return new_cube
