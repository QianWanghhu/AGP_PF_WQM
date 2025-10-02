# import packages
import os
import numpy as np
import pandas as pd
import pyapprox as pya
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from toposort import toposort

from .read_data import file_settings

def return_sa(parameters, pce):
    """
    The function is used to calculate the sensitivity indices by PCE.
    Parameters:
    ===========
    year: int, determines which PCE to load.
    parameters: np.ndarray or list, of parameter names.

    Returns:
    sa: pd.DataFrame, contains both the main and total effects
    """

    # import pces
    sobol_analysis = pya.sensitivity_analysis.analyze_sensitivity_polynomial_chaos
    res = sobol_analysis(pce, 2)
    total_effects = res.total_effects
    main_effects = res.main_effects
    # export the sensitivity of parameters in terms of SSE    
    sa = pd.DataFrame(index = parameters, columns=['main_effects', 'total_effects'])
    sa['main_effects'] = main_effects
    sa['total_effects'] = total_effects
    return sa


def dotty_plot(x_samples, y_vals, x_opt, y_opt, param_names, y_lab='SSE', orig_x_opt=None, orig_y_opt=None):
    """
    Create dotty plots for the model inputs and outputs.
    Parameteres:
    ============
    x_samples: np.ndarray, input sample set of the shape D * N where D is the number of parameters and N is the sample size;
    y_vals: np.ndarray, outputs corresponding to the x_samples and of the shape N * 1
    x_opt: np.ndarray, parameter data points resulting in the selected optima
    y_opt: np.ndarray, output values of the selected optima corresponding to x_opt
    param_names: list, parameter names
    orig_x_opt: np.ndarray, parameter data points resulting in the selected optima 
                and the selection is based on outputs without factor fixing.
    orig_y_opt: np.ndarray, output values of the selected optima corresponding to x_opt
    
    Returns:
    ========
    fig
    """
    fig, axes = plt.subplots(4, 4, figsize = (20, 20), sharey=True)
    for ii in range(x_samples.shape[0]): 
        if orig_x_opt is not None:
            ax = sns.scatterplot(x=orig_x_opt[ii, :], y=orig_y_opt.flatten(), ax=axes[ii // 4, ii % 4], color='g', s=20, alpha=0.8)
            
        ax = sns.scatterplot(x=x_samples[ii, :], y=y_vals.flatten(), ax=axes[ii // 4, ii % 4], s=20, alpha=0.5)
        ax = sns.scatterplot(x=x_opt[ii, :], y=y_opt.flatten(), ax=axes[ii // 4, ii % 4], color='orange', s=20, alpha=0.5)
                   
        ax.set_xlabel(param_names[ii])
        # ax.set_ylim(-2.5, 4)
        if ii % 4 == 0: ax.set_ylabel(y_lab)

    return fig

def define_constants(x, num_vars, stats = np.median):
    """Return default values of the parameters to fix
    Parameters:
    ===========
    x: np.ndarray, data points used to calculate the constant value to fix a parameter at.
        It is of the shape (D, N) where D is the number of parameters and N is the size of x.
    stats: str, used to define whether it is the mean or median to return. 
            The default value is 'median'.
    Return:
    x_default: np.ndarray, of the shape (D, 1)
    """
    assert x.shape[0] == num_vars, "Samples should be of two dimensions N*D where D is the number of parameters"
    x_default = stats(x, axis=1)
    return x_default
# End define_constants

def fix_sample_set(index_fix, dot_samples, x_default):
    import copy
    """ 
    Return the samples by fixing certain parameters.
    Parameters:
    ===========
    index_fix: np.ndarray, of the index of parameters to fix.
    dot_samples: np.ndarray, random samples of which the rows representing each parameter are to fix according to the index_fix.
                It is of the shape (D, N), where D is the number of parameters and N is the size of sampling.
    x_default: np.ndarray, of the shape (D, 1).  The default values of each parameter to fix at.
    
    Returns:
    ========
    samples_fix: np.ndarray, random samples of which the rows representing each parameter are fixed according to the index_fix.
                It is of the shape (D, N), where D is the number of parameters and N is the size of sampling.
    """
    samples_fix = copy.deepcopy(dot_samples)
    for ii in range(index_fix.shape[0]):
        samples_fix[index_fix[ii], :] = x_default[index_fix[ii]]    
    return samples_fix
# END fix_sample_set()

def surrogate_sample(sample_file, variable, nsamples, seed_state=None):
    """
    Random sampling and produce the surrogate results.
    Parameters:
    ===========
    samples_file: str, the  file name containing the random samples for evaluation.
    variable: variable
    nsamples: int, the number of samples to generate.
    seed_state: int, the random state to set so as to ensure the results are reproducible.
    
    returns:
    ========
    dot_samples: np.ndarray, random samples.
                It is of the shape (D, N), where D is the number of parameters and N is the size of sampling.
    """
    if seed_state is not None: np.random.RandomState(seed=seed_state)
        
    if os.path.exists(sample_file):
        dot_samples = np.loadtxt(sample_file)
    else:
        dot_samples = pya.generate_independent_random_samples(variable, nsamples)
        np.savetxt(sample_file, dot_samples)
        
    return dot_samples
# END surrogate_sample()

def timeseries_sum(df, temp_scale = 'annual'):
    """
    Obtain the sum of timeseries of different temporal scale.
    Parameters:
    ===========
    df: pd.Dataframe.
    temp_scale: str, default is 'Y', monthly using 'M'
    """
    assert temp_scale in ['monthly', 'annual'], 'The temporal scale given is not supported.'
    if temp_scale == 'monthly':
        sum_126001A = df.resample('M').sum()
    else:
        month_126001A = df.resample('M').sum()
        sum_126001A = pd.DataFrame(index = np.arange(df.index[0].year, df.index[-1].year), 
            columns=df.columns)
        for i in range(sum_126001A.shape[0]):
            sum_126001A.iloc[i, :] = month_126001A.iloc[i*12: (i+1)*12, :].sum()

    return sum_126001A
# End timeseries_sum()


def update_bounds(f_name, bounds):
    """
    Save the bounds to a file and update during the adaptive process.
    Parameters:
    ===========
    f_name: str, filename that contains the uncertainty bounds during each iteration.
    bounds: list or np.ndarray, of the lower and upper bounds
    
    returns:
    ========
    bounds_dict: dict, of the updated file.
    """
    if not os.path.exists(f_name):
        with open(f_name, 'wb') as f:
            bounds_dict = {'iteration_0': bounds}
            pickle.dump(bounds_dict, f)
    else:
        with open(f_name, 'rb') as f:
            bounds_dict = pickle.load(f)
            bounds_dict[f'iteration_{len(bounds_dict)}'] = bounds
        with open(f_name, 'wb') as f:
            pickle.dump(bounds_dict, f)
    return bounds_dict
# End update_bounds

# Define helper function for partial sorting
def partial_rank(sa, len_params, conf_level=0.95):
    """
    Help function for partial ranking.
    Parameters:
    ===========
    sa: ndarray, matrix of sensitivity indices, 
        of shape (N * P) where N is the number of bootstraps, P is the number of parameters
    len_params: int, the number of parameters
    conf_level: float, the confidence level used for the calculation of confidence intervals of rankings

    Returns:
    ========
    partial_rankings: dict, dictionary of partial rankings of parameters
                    e.g. {0:['2'], 1:['0', '1']} 
                        which means the parameter of index 2 is in group 0 which is more sensitive than the group 1 
    """
    num_boot = sa.shape[0]
    # print(num_boot)
    rankings = np.zeros([num_boot, len_params])
    ranking_ci = np.zeros([2, len_params])
    for resample in range(num_boot):
        rankings[resample,:] = np.argsort(sa[resample,:]).argsort()
    
    ## Generate confidence intervals of rankings based on quantiles (if the num_resample is big enough)
    ranking_ci = np.quantile(rankings,[(1-conf_level)/2, 0.5 + conf_level/2], axis=0)
    # rci = norm.ppf(0.5 + conf_level / 2) * rankings.std(ddof=1, axis=0)
    # ranking_ci = [rankings.mean(axis=0) - rci, rankings.mean(axis=0) + rci]
    # ranking_ci = np.rint(ranking_ci)
    # ranking_ci[ranking_ci<0] = 0        
    conf_low = ranking_ci[0]
    conf_up = ranking_ci[1]
    rank_conf = {j:None for j in range(len_params)}
    abs_sort= {j:None for j in range(len_params)}

    for j in range(len_params):
        rank_conf[j] = [conf_low[j], conf_up[j]]  
    # End for

    for m in range(len_params):
        list_temp = np.where(conf_low >= conf_up[m])
        set_temp = set()
        if len(list_temp) > 0:
            for ele in list_temp[0]:
                set_temp.add(ele)
        abs_sort[m] = set_temp
        # End if
    # End for

    order_temp = list(toposort(abs_sort))
    partial_rankings = {j: list(order_temp[j]) for j in range(len(order_temp))}
    return partial_rankings