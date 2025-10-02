#! user/bin/env python
import numpy as np
import pandas as pd
import pickle
import pyapprox as pya
from funcs.read_data import file_settings, variables_prep


fp = '../output/gp_run_1117/sampling-sa/old/'

fn = 'rankings.json.npz'
sa = np.load(fp+fn, allow_pickle = True)['arr_0'].reshape(1)[0]
order = 2
param_file = file_settings()[-1]
ind_vars, variables_full = variables_prep(param_file, product_uniform='uniform', dummy=False)
interaction_terms = pya.compute_hyperbolic_indices(len(ind_vars), order)
interaction_terms = interaction_terms[:, np.where(
    interaction_terms.max(axis=0) == 1)[0]]
param_names = pd.read_csv(param_file, usecols=[2]).values.flatten()


Si_mean = sa['sobol_indices']['mean']           
Si_mean = pd.DataFrame(data = Si_mean, index = np.arange(Si_mean.shape[0]), columns=['Si'])
Si_mean['std'] = sa['sobol_indices']['std']
for ii in range(Si_mean.shape[0]):
    val_pars = param_names[interaction_terms[:, ii] == 1]
    if len(val_pars) == 2:
        Si_mean.loc[ii, ['par1', 'par2']] = val_pars
    else:
        Si_mean.loc[ii, ['par1', 'par2']] = [val_pars[0], val_pars[0]]

Si_mean.to_csv(f'{fp}/SI.csv')