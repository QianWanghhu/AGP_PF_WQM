"""
Help functions used to process and prepare the template files.
"""
import pandas as pd
import numpy as np
import os


def pre_ies_obs_template(obsval, obsnme, obgnme, weight):
    # Prepare the observation into the format for .pst file
    obs_pst = observation_pst(obsval, obsnme, obgnme, weight)
    # obs_pst.weight.format(f'%:.2E')
    obs_pst.weight = obs_pst.weight.map('{:,.2E}'.format)
    obs_pst.obsval = obs_pst.obsval.map('{:,.3f}'.format)
    return obs_pst
# End pre_ies_obs_template()

def generate_noise(size, dist = 'normal', pars={'mean': 0, 'std': 1}):
    """
    Help funtion to add noise to the model outputs.
    Parameters:
    -----------
    size : int, the number of noise to generate
    dist : str, distribution of the noise, default is normal
    pars : other arguments used to specify the dist

    Return:
    -------
    noise : np.narray, noise
    """
    if dist == 'normal':
        noise = np.random.normal(pars['mean'], pars['std'], size = size)
    else:
        raise Exception("Sorry, the type of distribution for noise is not included")

    return noise
# End add_noise()


def noise_stats(data, noise_level):
    """
    Help funtion to calculate the mean and standard deviation.
    Parameters
    -----------

    Return
    -------
    """
    if not isinstance(data, np.ndarray):
        raise Exception("data is not an array")

    if not isinstance(noise_level, float):
        raise Exception("noise_level is not an array")

    data_mean = data.mean()
    std_level = data_mean * noise_level

    return std_level
# End noise_stats()

def observation_pst(observation, obsnme, obgnme, weight):
    """
    Format observartions into the .pst file for PEST++.
    Parameters:
    -----------

    Returns:
    ---------
    """ 
    obs_pst = pd.DataFrame(data = observation, columns = ['obsval'])
    obs_pst.index = obsnme
    # obs_pst.loc[:, 'weight'] = 1 / obs_pst.shape[0]
    obs_pst.loc[:, 'weight'] = weight
    obs_pst.loc[:, 'obgnme'] = obgnme
    obs_pst.index.name = 'obsnme'

    return obs_pst
# observation_pst()

def generate_obsnme(x, obgnme, time_format="%d_%m_%Y"):
    if isinstance(x, pd.core.indexes.datetimes.DatetimeIndex):
        obsnme = pd.to_datetime(x).strftime(time_format)
    else:
        obsnme = x[:]
    obsnme = [f'{obgnme}_{i}' for i in obsnme]  

    return obsnme
# End generate_obsnme()
    

def add_noise_obs(load, noise_level):
    """
    Add noise to observations.
    Parameters:
    ===========
    load: pd.DataFrame without noise
    noise_level: float, the level of noise to add.

    Returns:
    ===========
    noised_obs: pd.DataFrame with noise
    """
    column_names = load.columns[0]
    # Add noise to the model outputs
    std_level = noise_stats(load.values, noise_level)
    noise = generate_noise(load.shape[0], dist='normal', pars={'mean':0, 'std': std_level})
    load_min = load.loc[:, column_names].min()
    noised_obs = load.loc[:, column_names].add(noise)
    noised_obs = np.where(noised_obs >= 0, noised_obs, load_min)
    load.loc[:, 'noised'] = noised_obs
    return load    
# End add_noise_obs()


def format_obs_ensemble(obs_ensemble, obgnme):    
    """
    Reformat observation ensemble.
    Parameters:
    ===========
    obs_ensemble: pd.DataFrame, the observation ensemble

    Returns:
    ===========
    obs_ensemble: pd.DataFrame, the observation ensemble after new format.
    """
    obs_ensemble.index = generate_obsnme(list(obs_ensemble.index), obgnme)
    obs_ensemble = obs_ensemble.T
    obs_ensemble.reset_index(drop=True, inplace=True)
    obs_ensemble.index.name = 'real_name'
    return obs_ensemble
# End format_obs_ensemble()

def stat_output(df, summary_scale='annual'):
    """
    Calculate the statistics / summarized results according to the user defined criteria. 
    Parameters:
    ===========
    df: pd.DataFrame, timeseries to summarize.
    summary_scale: str, define the temporal scale of summary

    Returns:
    ===========
    summarized_df: pd.DataFrame of the new time step.
    """
    assert isinstance(df, pd.DataFrame), 'df should be a dataframe.'
    assert (summary_scale in (['annual', 'monthly'])), 'summary_scale is not supported.'
    if summary_scale == 'monthly':
        summarized_df = df.resample('M').sum()
    else:
        summarized_df = df.resample('Y').sum()
    return summarized_df
# End stat_output()