import os
import sys
import iris
import iris.plot as iplt
from iris.cube import CubeList
import iris.analysis.maths as maths
import numpy as np
import datetime as dt
from scipy.stats import t
import pandas as pd

from merphuncertainty.ph_sampling import *
from merphuncertainty.plotting import _constrain_cube_for_plot
from merphuncertainty.cubes import member_prob_cube, get_ppf_cube

# cubes_dir = "/user/work/hg20831/postproc-share/quadrature/"
cubes_dir = "../data/name_out_cubes/"
csv_file = "../data/chunks_10_14km.csv"
fig_dir = "figures/"

volcano_height = 1725
height_asl = 12
height_avl = height_asl - (volcano_height / 1000)

thresholds = [0.2E-3, 2E-3, 5E-3, 10E-3]
percentiles = [0.05, 0.5, 0.95]

ivespa = set_ivespa_obs(height_avl * 1000, volcano_height)
df = ivespa.df
mu = ivespa.mu[0]
sigma = ivespa.Sigma[0]

# Evaluate MER for this height
mer_gs = 10 ** mu * 1000
mer_gs_ht = mer_gs / (height_avl * 1000)

######## Ensemble exceedance probabilities
chunks = SourceChunks(volcano_height, csv_file)
chunks.name_output = VolcanicNAME("ex3", True, "/", csv_file)
chunks.name_output.n_members = 18
chunks.list_member_chunks = []

for member in range(18):
    ash_chunks_filename = cubes_dir + f"ash_chunks_member_{member}.nc"
    # Load ash cube list and constrain
    chunks.list_member_chunks.append(
        chunks.name_output.load_ash_cube_list(ash_chunks_filename, member))

member_prob_list = CubeList([])
log_cube_list = CubeList([])
n_members = len(chunks.list_member_chunks)

for i in range(n_members):
    exc_cube_list = CubeList([])
    chunks_member_cube_list = chunks.list_member_chunks[i]
    # Get rescaled cube
    rescaled_cube = reconstruct_cube_from_list(
        cube_list = chunks_member_cube_list, 
        plume_height = height_asl * 1000,
        new_mer_gs_ht = mer_gs_ht)
    log_cube = rescaled_cube.copy()
    log_cube.data = np.log10(rescaled_cube.data)
    # Additional coordinate for ensemble member
    new_coord = iris.coords.AuxCoord(i, long_name = "member", units = "")
    log_cube.add_aux_coord(new_coord)
    # Update attributes
    log_cube.units = ""
    log_cube.attributes["Quantity"] = "Log-ash Concentration"
    if isinstance(log_cube.data, np.ma.core.MaskedArray):
        log_cube.data = log_cube.data.filled(-np.inf)
    log_cube_list.append(log_cube.copy())
    # Evaluate exceedance probability cube given MER t-distribution
    prob_cube = member_prob_cube(
        log_cube, 
        df = df, 
        sigma = sigma,
        h_km = height_asl, 
        exceed = True,
        thresholds = thresholds)
    prob_cube.attributes["Title"] = f"Member {i}"
    member_prob_list.append(prob_cube.copy())
    sqr_cube = maths.multiply(prob_cube, prob_cube)
    # Evaluate exceedance given constant MER value
    for j, threshold in enumerate(thresholds):
       exc_cube = rescaled_cube.copy()
       new_coord = iris.coords.AuxCoord(threshold, long_name = "threshold", 
                                       units = "g/m3")
       exc_cube.add_aux_coord(new_coord)
       exc_cube.data = (exc_cube.data >= threshold).astype(int)
       exc_cube_list.append(exc_cube)
    member_exc_cube = exc_cube_list.merge_cube()
    sqr_exc_cube = maths.multiply(member_exc_cube, member_exc_cube)
    if i == 0:
       # Save control member
       iris.save(member_prob_list, "../data/control_prob_cube_1.nc")
       # Add to cumulative sum
       sum_prob_cube = prob_cube.copy()
       sum_sqr_cube = sqr_cube.copy()
       sum_exc_cube = member_exc_cube.copy()
       sum_sqr_exc_cube = sqr_exc_cube.copy()
       # Evaluate percentiles for control met
       ppf_cube_list = CubeList([])
       for ppf in percentiles:
           ppf_cube = log_cube.copy()
           ppf_cube.data = t.ppf(ppf, df = df, loc = log_cube.data, scale = sigma)
           ppf_cube.data[ppf_cube.data == -np.inf] = np.nan
           ppf_cube.data = np.power(10, ppf_cube.data) * 1E+6 # convert to ug/m^3
           new_coord = iris.coords.AuxCoord(ppf, long_name = "percentile", 
                                            units = "")
           ppf_cube.add_aux_coord(new_coord)
           ppf_cube.attributes["Quantity"] = "Percentile Point"
           ppf_cube_list.append(ppf_cube.copy())
       ppf_cube = ppf_cube_list.merge_cube()
       iris.save(ppf_cube, "../data/control_ppf_1.nc")	
       print("Saved ppf cube.")
    else:
        sum_prob_cube = maths.add(sum_prob_cube, prob_cube)
        sum_sqr_cube = maths.add(sum_sqr_cube, sqr_cube)
        sum_exc_cube = maths.add(sum_exc_cube, member_exc_cube)
        sum_sqr_exc_cube = maths.add(sum_sqr_exc_cube, sqr_exc_cube)
# 
iris.save(member_prob_list, "../data/ensemble_member_exc_probs.nc")

mean_exc_cube = sum_exc_cube / n_members
iris.save(mean_exc_cube, "../data/ensemble_exc_prob.nc")

mean_prob_cube = sum_prob_cube / n_members
iris.save(mean_prob_cube, "../data/ensemble_merph_exc_prob.nc")

sqr_mean_cube = maths.multiply(mean_prob_cube, mean_prob_cube)
pop_var_cube = sum_sqr_cube / n_members - sqr_mean_cube
pop_sd_cube = pop_var_cube.copy()
pop_sd_cube.data = np.sqrt(pop_var_cube.data)
iris.save(pop_sd_cube, "../data/ensemble_merph_pop_sd.nc")

sqr_exc_cube = maths.multiply(mean_exc_cube, mean_exc_cube)
pop_var_exc_cube = sum_sqr_exc_cube / n_members - sqr_exc_cube
pop_sd_exc_cube = pop_var_exc_cube.copy()
pop_sd_exc_cube.data = np.sqrt(pop_var_exc_cube.data)
iris.save(pop_sd_exc_cube, "../data/ensemble_merph_pop_sd_exc.nc")

iris.util.equalise_attributes(log_cube_list)
log_cube = log_cube_list.merge_cube()
log_cube_const, _ = _constrain_cube_for_plot(log_cube, time_index = 6, fl_index = 0)

ppf_cube = get_ppf_cube(log_cube_const, df, sigma, interval = (-6, 1))
iris.save(ppf_cube, "../data/ensemble_ppf.nc")

def get_subcubes_from_cubes(cube_a, cube_b, grid):
    # Assign values based on data type
    if grid == 'grid88' or grid == 'grid99':
        subcubes_a = []
        subcubes_b = []

        for i, _ in enumerate(cube_a.coord('flight_level')):
            subcube_a = cube_a[i, ..., ...]
            subcube_b = cube_b[i, ..., ...]

            # subcube_a.convert_units('ug/m3')
            # subcube_b.convert_units('ug/m3')

            subcubes_a.append(subcube_a)
            subcubes_b.append(subcube_b)

    elif grid == 'grid23':
        subcubes_a = [cube_a]
        subcubes_b = [cube_b]

    else:
        raise KeyError('Invalid grid reference.')

    return subcubes_a, subcubes_b

def get_cube_info(cube_a, grid = "grid99"):
    # Obtain cube information
    quantity = cube_a.attributes['Quantity']
    t_coord = cube_a.coord('time')

    fn_time = t_coord.units.num2date(t_coord.points[0]).strftime('%Y%m%d%H%M')
    fl_bound = cube_a.coord("flight_level").bounds[0]
    fl = f"FL{int(fl_bound[0]):03d}-{int(fl_bound[1]):03d}"

    if grid == 'grid88' or grid == 'grid99':
        release_time = pd.to_datetime(cube_a.attributes['Start of release'], 
                                      format='%H%MUTC %d/%m/%Y')
    elif grid == 'grid23':
        release_time = pd.to_datetime(cube_a.attributes['Start of release'], 
                                      format='%d/%m/%Y %H:%M UTC')

    return quantity, fn_time, fl, release_time


def l2_log(ini_dict):

    csv_path= ini_dict.get('csv_path', None)

    lat_coord_name = ini_dict.get('lat_coord_name', 'latitude')
    lon_coord_name = ini_dict.get("lon_coord_name", "longitude")

    grid = ini_dict['grid']
    cubes_a = ini_dict['cubes_a']
    cubes_b = ini_dict['cubes_b']
    timepoints = len(cubes_a.coord("time").points)
    timeseries = pd.DataFrame(index=range(timepoints))
    

    for i in range(timepoints):
        cube_a = cubes_a[i, :, :, :]
        cube_b = cubes_b[i, :, :, :]
        subcubes_a, subcubes_b = get_subcubes_from_cubes(cube_a, cube_b, grid)

        for c, cube_a in enumerate(subcubes_a):
            cube_b = subcubes_b[c]
            
            # Calculate overlap percentage
            cube_a.coord(lat_coord_name).long_name = 'source_lat'  # Must be renamed for area weights to work
            cube_a.coord(lon_coord_name).long_name = 'source_lon'  # Must be renamed for area weights to work

            l2_cube = maths.add(cube_a, -cube_b)
            l2_cube.data = np.abs(l2_cube.data)
            l2 = float(l2_cube.collapsed(["longitude", "latitude"], iris.analysis.MEAN).data)

            # Obtain cube information
            _, fn_time, fl, release_time = get_cube_info(cube_a)

            # Add time and overlap values to dataframe for timeseries plot
            if 'time' not in timeseries:
                timeseries['time'] = np.nan
            timeseries.iloc[i, 0] = pd.to_datetime(fn_time)

            if 'time_diff' not in timeseries:
                timeseries['time_diff'] = np.nan
            timeseries.iloc[i, 1] = timeseries.iloc[i, 0] - release_time.tz_localize(None)

            if fl not in timeseries:
                timeseries[fl] = np.nan

            timeseries.iloc[i, c+2] = l2

    timeseries.sort_values(by='time_diff', inplace=True)
    timeseries.set_index('time_diff', inplace=True)
    timeseries.index = [int(t) for t in timeseries.index.astype('timedelta64[h]')]

            
    if csv_path is not None:
        if os.path.exists(csv_path):
            mode = "a" # append
            header = False
        else:
            mode = "w" # create/write
            header = True
        timeseries.to_csv(csv_path, mode = mode, header = header)
        print("Saved dataframe as " + csv_path)

    return timeseries

sd = 1
n_intervals = [1, 2, 4, 6, 12]
control_cube = iris.load(cubes_dir + f"quad_exc_prob_24_sd{str(sd).replace('.', '_')}.nc")[0]
full_df = pd.DataFrame()

for j, n_int in enumerate(n_intervals):
    cube = iris.load(cubes_dir + f"quad_exc_prob_{n_int}_sd{str(sd).replace('.', '_')}.nc")[0]
    for threshold in thresholds:
        print(f"sd={sd}, n={n_int}, threshold={threshold}")
        control_th_cube = control_cube.extract(iris.Constraint(threshold = lambda cell: cell == threshold))
        th_cube = cube.extract(iris.Constraint(threshold = lambda cell: cell == threshold))

        ini_dict = {
            "plot": False,
            "threshold": 1E-10,
            "cubes_a": control_th_cube,
            "cubes_b": th_cube,
            "grid": "grid99",
            "lat_coord_name": "latitude",
            "lon_coord_name": "longitude",
            "extent_list":[40, -40,30,80]
        }

        df = l2_log(ini_dict)
        # add two columns to beginning of data frame for sd and n_int
        df.insert(0, 'threshold', threshold)
        df.insert(1, 'n_int', n_int)
        # append to full_df
        full_df = full_df.append(df)

# create column "time_since" from index
full_df['time_since'] = full_df.index
full_df['time_since'] = full_df['time_since'].astype(int)
full_df.reset_index(drop=True, inplace=True)
full_df.to_csv("../data/quad_diff.csv", mode = "w", header = True)