"""
Script containing functions for manipulating .csv files.

Author: Shannon Williams

Date: 25/09/2023
"""

import os
import csv
import numpy as np
from merph.stats import QHStats

def list_dicts_to_csv(dicts_list: str, csv_path: str, overwrite: bool = False):
    """Given a list of dictionaries, saves the data as .csv in tidy format. 

    Args:
        dicts_list (list): List of dictionaries. 
        csv_path (str): Path of csv file to be written to.
        overwrite (bool, optional): If csv_path already exists, whether to 
        overwrite the file or append to it. Defaults to False.
    """
    
    if os.path.exists(csv_path) and not overwrite:
        mode = "a" # append
        header = True
    else:
        mode = "w" # create/write
        header = False

    with open(csv_path, mode) as csv_file:
        # Initialise csv file from header
        if not isinstance(dicts_list, list):
            dicts_list = [dicts_list]
        l = len(dicts_list)
        fieldnames = list(dicts_list[0].keys())

        writer = csv.DictWriter(csv_file, fieldnames = fieldnames)
        if not header:
            writer.writeheader()

        for i in range(l):
            this_dict = dicts_list[i]
            is_list = False
            j = 0
            while True:
                data = {}
                for key in this_dict:
                    try: 
                        if isinstance(this_dict[key], (list, np.ndarray)):
                            data[key] = this_dict[key][j]
                            is_list = True
                        else:
                            data[key] = this_dict[key]
                    except:
                        is_list = False
                        break
                else: # If loop not broken, write to csv
                    writer.writerow(data)
                    j += 1
                    if is_list:
                        continue
                break

def posterior_to_csv(qhstats: QHStats, plume_height: tuple, 
                     csv_path: str = None, volcano_height: float = 0.0, 
                     uniform_release: bool = True, labels: tuple = None):
    """Use QHStats object from MERPH to generate eruption source parameters for 
    each plume height sample, including information on the release height, MER, 
    and parameters of the t-distribution of log-MER.

    Args:
        qhstats (merph.stats.QHStats): QHStats object initialised in merph.
        plume_height (tuple): Plume height samples, in km above vent 
        level (m a.s.l.).
        csv_path (str, optional): Path of csv file for data to be saved to. If 
        None, data is not saved as csv and returned as a list of dictionaries. 
        Defaults to None.
        volcano_height (float, optional): Vent height, in m a.s.l. Defaults to 
        0.0.
        uniform_release (bool, optional): Whether volcanic ash is released 
        uniformly between the vent and top of plume. If False, ash is released 
        only from a point at the top of the plume. Defaults to True.
        labels (tuple, optional): List of labels for each sample. Defaults to 
        None.

    Returns:
        list: List of dictionaries containing eruption source parameters.
    """
    if not isinstance(plume_height, (tuple, list)):
        plume_height = [plume_height]
                     
    dicts = [{"label": "sample_{}".format(i) if labels is None else labels[i],
              "H (km asl)": plume_height[i] / 1000,
              "mu": qhstats.mu[i],
              "sigma": qhstats.Sigma[i],
              "df": qhstats.df,
              "Q (g s)": (10 ** qhstats.mu[i] * 1000),
              "Q (g hr)": (10 ** qhstats.mu[i] * 1000 * (60 ** 2))} 
             for i in range(len(plume_height))]
    
    if uniform_release:
        dz = [h - volcano_height for h in plume_height]
        source_height = [(h + volcano_height) / 2 for h in plume_height]
    else:
        dz = [0] * len(plume_height)
        source_height = plume_height
    
    for i in range(len(dicts)):
        dicts[i].update({"Z (m)": source_height[i], "dZ (m)": dz[i]})

    # Save data to csv
    if csv_path is not None:
        list_dicts_to_csv(dicts, csv_path, overwrite = True)

    return dicts

def source_params_dict(label: str, min_z: float, max_z: float, 
                       particles_hr_ht: float, mer_gs: float = None, 
                       mer_gs_ht: float = None): 
    """Generate dictionary of source parameters.

    Args:
        label (str): Name of source.
        min_z (float): Minimum height of release, in m a.s.l.
        max_z (float): Maximum height of release, in m a.s.l.
        particles_hr_ht (float): Number of particles released in NAME per hour, 
        per unit height (m).
        mer_gs (float, optional): Mass released from source in grams per second.
        mer_gs_ht (float, optional): Mass released from source in grams per 
        second per unit height (m).

    Raises:
        ValueError: Neither mer_gs nor mer_gs_ht provided.

    Returns:
        dict: Dictionary of eruption source parameters.
    """
    
    dz = max_z - min_z
    z = min_z + (dz / 2)
        
    particles_hr = int(particles_hr_ht * dz)

    if mer_gs is None:
        if mer_gs_ht is not None:
            mer_gs = mer_gs_ht * dz
        else:
            raise ValueError("Provide one of mer_gs or mer_gs_ht.")

    dict = {"label": label,
            "Q (g s)": mer_gs,
            "Q (g hr)": mer_gs * (60 ** 2),
            "min (m)": min_z,
            "max (m)": max_z,
            "Z (m)": z,
            "dZ (m)": dz,
            "particles (p hr)": particles_hr}
    
    return dict