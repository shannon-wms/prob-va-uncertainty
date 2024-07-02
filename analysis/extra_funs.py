import os
import pandas as pd
import numpy as np
import iris.analysis.maths as maths

def get_subcubes_from_cubes(cube_a, cube_b, grid):
    # Assign values based on data type
    if grid == 'grid88' or grid == 'grid99':
        subcubes_a = []
        subcubes_b = []

        for i, _ in enumerate(cube_a.coord('flight_level')):
            subcube_a = cube_a[i, ..., ...]
            subcube_b = cube_b[i, ..., ...]

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