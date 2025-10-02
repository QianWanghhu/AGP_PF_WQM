#!/usr/bin/env ffexplore
"""This script is used different data"""
import numpy as np
import pandas as pd
import json
import pyapprox as pya
from pyapprox.variables import IndependentMarginalsVariable

from scipy.stats import uniform, beta

def file_settings():
    model_dir = '../output/'
    input_dir = '../data/'
    model_ts_full = f'{input_dir}2000_2014_ave_annual.csv'
    model_ts_reduced = f'{model_dir}samples_adjust.csv'
    
    param_full = f'{input_dir}Parameters.csv'
    param_reduced = f'{input_dir}Parameters-PCE.csv'
    return [model_dir, input_dir, model_ts_full, model_ts_reduced, param_full, \
        param_reduced]
# END file_settings()

def read_model_ts(filename):
    """Read the model outputs used for building surrogates.
    Parameters:
    ===========
    filename: str, filename of the model output to read.

    Returns:
    samples: np.ndarray, of two dimension N * D 
        where N is the number of samples and D is the number of parameters
    values: np.ndarray, the Quantity of interest to simulate.
    """
    data = np.loadtxt(filename, delimiter=",", skiprows=1)[:,1:]
    samples = data[:, :-1].T
    values = data[:, -1:]
    return samples, values
# END read_model_ts()

def read_parameters(filename, product_uniform):
    variable = variables_prep(filename, product_uniform=product_uniform)
    param_all = pd.read_csv(filename).loc[:, 'Veneer_name'].values
    return variable, param_all
# END read_parameters()

def read_ranks(filename):
    with open(f'{filename}', 'r') as fp:
        partial_order = json.load(fp)
    return partial_order
# END read_ranks()

def read_specify(data_type, param_type, product_uniform, num_vars=22):
    filenames = file_settings()
    assert (param_type in ['full', 'reduced']), 'param_type is not'
    if data_type == 'model':
        if param_type == 'full':
            return read_model_ts(filenames[2], num_vars)
        elif param_type == 'reduced':
            return read_model_ts(filenames[3], num_vars)
    elif data_type == 'parameter':
        if param_type == 'full':
            assert (product_uniform is False), 'product_uniform should be None when using full model.'
            assert (num_vars == 22), 'num_vars should be 22 when using full model.'
            return read_parameters(filenames[4], product_uniform)
        elif param_type == 'reduced':
            assert (num_vars == 11), 'num_vars should be 11 when using reduced model.'
            return read_parameters(filenames[5], product_uniform)
    else:
        rank_name = f'{filenames[0]}partial_reduce_{product_uniform}_552.json'
        return read_ranks(rank_name)
# END read_specify()          

def variables_prep(filename, product_uniform=False, dummy=False):
    """
    Help function for preparing the data training data to fit PCE.
    Parameters:
    ===========
    filename : str
    product_uniform : False do not colapse product into one variable
                      'uniform' uniform distributions are used for product; 
                      'beta', beta distributions are used for variables which 
                      are adapted considering the correlations
                      'exact' the true PDF of the product is used

    """
    # import parameter inputs and generate the dataframe of analytical ratios between sensitivity indices
    if (product_uniform is False) or (product_uniform == 'uniform'):    
        ranges = np.loadtxt(
            filename,delimiter=",",usecols=[3,4],skiprows=1).flatten()
        univariate_variables = [uniform(ranges[2*ii],ranges[2*ii+1]-ranges[2*ii]) for ii in range(0, ranges.shape[0]//2)]
        # breakpoint()    
    else:
        param_adjust = pd.read_csv(filename)
        beta_index = param_adjust[param_adjust['distribution']== 'beta'].index.to_list()
        ranges = np.array(param_adjust.loc[:, ['min','max']])
        ranges[:, 1] = ranges[:, 1] - ranges[:, 0]
        # param_names = param_adjust.loc[[0, 2, 8], 'Veneer_name'].values
        univariate_variables = []
        for ii in range(param_adjust.shape[0]):
            if ii in beta_index:
                shape_ab = param_adjust.loc[ii, ['a','b']].values.astype('float')
                univariate_variables.append(beta(shape_ab[0], shape_ab[1], 
                                            loc=ranges[ii][0], scale=ranges[ii][1]))
            else:
                # uniform_args = ranges[ii]
                univariate_variables.append(uniform(ranges[ii][0], ranges[ii][1]))
            # End if
        # End for()

    if dummy == True: univariate_variables.append(uniform(0, 1))

    variable = IndependentMarginalsVariable(univariate_variables)
    return univariate_variables, variable
    
#END variables_prep()