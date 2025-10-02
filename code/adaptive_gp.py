#!/usr/bin/env python
from multiprocessing import Pool
import numpy as np
import os
import matplotlib.pyplot as plt
from functools import partial
import time
import copy
import pandas as pd
import pickle
import pyapprox as pya
from scipy import stats
# from scipy.optimize import root
from scipy.optimize import bisect
from pyapprox.surrogates.gaussianprocess.kernels import RBF
from sklearn.gaussian_process.kernels import Matern
    
from pyapprox.variables.density import tensor_product_pdf
from pyapprox.surrogates.gaussianprocess.gaussian_process import CholeskySampler, \
    AdaptiveGaussianProcess, generate_gp_candidate_samples
from pyapprox.expdesign.low_discrepancy_sequences import transformed_halton_sequence
from pyapprox.variables.risk import compute_f_divergence
from pyapprox.util.utilities import get_tensor_product_quadrature_rule
from pyapprox.variables.sampling import generate_independent_random_samples
from pyapprox.util.visualization import get_meshgrid_function_data
from pyapprox.variables import IndependentMarginalsVariable
from pyapprox.variables import AffineTransform
from pyapprox.variables.sampling import rejection_sampling

import matplotlib as mpl
from matplotlib import rc
from funcs.read_data import variables_prep, file_settings
from funcs.modeling_funcs import vs_settings, \
        modeling_settings, paralell_vs, obtain_initials, change_param_values
import spotpy as sp
from joblib import Parallel, delayed
from sklearn.metrics import r2_score, root_mean_squared_error

mpl.rcParams['font.size'] = 16
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['text.usetex'] = False  # use latex for all text handling
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.format'] = 'pdf'  # gives best resolution plots
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['legend.fontsize'] = 16
# print mpl.rcParams.keys()
mpl.rcParams['text.latex.preamble'] = \
    r'\usepackage{siunitx}\usepackage{amsmath}\usepackage{amssymb}'

vs_list = []
def run_source_lsq(vars, vs_list=vs_list):
    """
    Script used to run_source and return the output file.
    The function is called by AdaptiveLejaPCE.
    """
    from funcs.modeling_funcs import modeling_settings, generate_observation_ensemble
    import spotpy as sp
    print('Read Parameters')
    parameters = pd.read_csv('../data/Parameters-PCE.csv', index_col='Index')

    # Use annual or monthly loads
    def timeseries_sum(df, temp_scale = 'annual'):
        """
        Obtain the sum of timeseries of different temporal scale.
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

    def viney_F(evaluation, simulation):
        pb = sp.objectivefunctions.pbias(evaluation, simulation) / 100
        nse = sp.objectivefunctions.nashsutcliffe(evaluation, simulation)
        F = nse - 5 *( np.abs(np.log(1 + pb)))**2.5
        return F

    # Define functions for the objective functions
    def cal_obj(x_obs, x_mod, obj_type = 'nse'):
        obj_map = {'nse': sp.objectivefunctions.nashsutcliffe,
                    'rmse': sp.objectivefunctions.rmse,
                    'pbias': sp.objectivefunctions.pbias,
                    'viney': viney_F
                }
        obj = []
        for  k in range(x_mod.shape[1]):
            obj.append(obj_map[obj_type](x_obs, x_mod[:, k].reshape(x_mod.shape[0], 1)))
        # if obj[0] == 0: obj[0] = 1e-8
        obj = np.array(obj)
        if obj_type =='pbias':
            obj = obj / 100
        obj = obj.reshape(obj.shape[0], 1)
        print(obj)
        return obj
    # End cal_obj
    
    # rescale samples to the absolute range
    vars_copy = copy.deepcopy(vars)
    # vars_copy[0, :] = vars_copy[0, :] * 100
    # vars_copy[1, :] = vars_copy[1, :] * 100

    # import observation if the output.txt requires the use of obs.
    date_range = pd.to_datetime(['2009/07/01', '2018/06/30'])
    observed_din = pd.read_csv(f'{file_settings()[1]}126001A.csv', index_col='Date')
    observed_din.index = pd.to_datetime(observed_din.index)
    observed_din = observed_din.loc[date_range[0]:date_range[1], :].filter(items=[observed_din.columns[0]]).apply(lambda x: 1000 * x)
    
    # loop over the vars and try to use parallel     
    parameter_df = pd.DataFrame(index=np.arange(vars.shape[1]), columns=parameters.Name_short)
    for i in range(vars.shape[1]):
        parameter_df.iloc[i] = vars[:, i]

    # set the time period of the results
    retrieve_time = [pd.Timestamp('2009-07-01'), pd.Timestamp('2018-06-30')]

    # define the modeling period and the recording variables
    _, _, criteria, start_date, end_date = modeling_settings()
    initial_values = obtain_initials(vs_list[0])
    din = generate_observation_ensemble(vs_list, 
        criteria, start_date, end_date, parameter_df, retrieve_time, initial_values)

    # obtain the sum at a given temporal scale
    din_126001A = timeseries_sum(din, temp_scale = 'annual')
    obs_din = timeseries_sum(observed_din, temp_scale = 'annual')
    din_126001A = pd.DataFrame(din_126001A,dtype='float').values
    obs_din = pd.DataFrame(obs_din,dtype='float').values


    obj = cal_obj(obs_din, din_126001A, obj_type = 'viney')
    print(f'Finish {obj.shape[0]} run')

    # calculate the objective NSE and PBIAS
    obj_nse = cal_obj(obs_din, din_126001A, obj_type = 'nse')
    obj_pbias = cal_obj(obs_din, din_126001A, obj_type = 'pbias')
    train_iteration = np.append(vars, obj_nse.T, axis=0)
    train_iteration = np.append(train_iteration, obj_pbias.T, axis=0)
    # save sampling results of NSE and PBIAS
    train_file = 'training_samples.txt'
    if os.path.exists(train_file):
        train_samples = np.loadtxt(train_file)
        train_samples = np.append(train_samples, train_iteration, axis=1)
        np.savetxt(train_file, train_samples)
    else:
        np.savetxt(train_file, train_iteration) 
    # END if-else

    return obj
# END run_source_lsq()

from scipy.stats import qmc

def _farthest_points(X, k):
    """Greedy max-min selection for diversity (Euclidean)."""
    X = np.asarray(X, float)
    sel = [0]
    dists = np.linalg.norm(X - X[0], axis=1)
    for _ in range(1, min(k, len(X))):
        i = int(np.argmax(dists))
        sel.append(i)
        dists = np.minimum(dists, np.linalg.norm(X - X[i], axis=1))
    return X[sel]

def generate_samples_above_threshold(gp, bounds, *, n=100, threshold=0.38,
                                     batch=50000, expand=2, max_pool=200000,
                                     diversify=True, seed=None):
    """
    Generate n samples in [bounds] with GP mean >= threshold.
    If fewer pass, fill with highest-mean leftovers to reach n.

    bounds: (d,2) array with [lo, hi] per dimension
    returns: (n, d) array
    """
    rng = np.random.default_rng(seed)
    d = bounds.shape[0]
    M = batch
    X_keep = None
    y_keep = None

    while True:
        # QMC batch in [0,1]^d → scale to bounds
        sob = qmc.Sobol(d, scramble=True, seed=rng.integers(1 << 32))
        X01 = sob.random(M)
        X = bounds[:, 0] + X01 * (bounds[:, 1] - bounds[:, 0])  # (M, d)

        y = np.ravel(gp.predict(X))  # GP mean

        mask = y >= threshold
        X_ok = X[mask]
        y_ok = y[mask]

        if X_keep is None:
            X_keep, y_keep = X_ok, y_ok
        else:
            if X_ok.size:
                X_keep = np.vstack([X_keep, X_ok])
                y_keep = np.concatenate([y_keep, y_ok])

        if X_keep is not None and len(X_keep) >= n:
            # enough passing points
            if diversify:
                X_sel = _farthest_points(X_keep, n)
            else:
                # take top-n by mean (descending)
                idx = np.argsort(-y_keep)[:n]
                X_sel = X_keep[idx]
            return X_sel  # (n, d)

        # Not enough yet → expand or stop
        if M >= max_pool:
            # Fill with best of all sampled (even if < threshold) to reach n
            X_all = X_keep if X_keep is not None else np.empty((0, d))
            y_all = y_keep if y_keep is not None else np.empty((0,))
            # add current batch too
            X_all = np.vstack([X_all, X])
            y_all = np.concatenate([y_all, y])
            idx = np.argsort(-y_all)[:n]
            return X_all[idx]
        M *= expand

def update_candidate_fast(
    gp,
    sampler,
    threshold: float = 0.38,
    n_to_replace: int = 100,
    seed: int | None = None,
    replace_strategy: str = "least",  # "least" or "random"
):
    """
    1) Generate n_to_replace new samples with GP mean >= threshold.
    2) Choose n_to_replace non-pivot columns to overwrite (by strategy).
    3) Write the new samples into those columns.

    sampler must have:
      - candidate_samples: (d, m) ndarray
      - init_pivots: 1D indices of pivot columns to keep
    """
    cs = sampler.candidate_samples  # shape (d, m)
    d, m = cs.shape
    all_cols = np.arange(m)
    pivots = np.asarray(sampler.init_pivots, dtype=int)
    non_pivot = np.setdiff1d(all_cols, pivots, assume_unique=False)

    # Build bounds from current candidate pool (or pass your own bounds here)
    lo = cs.min(axis=1)
    hi = cs.max(axis=1)
    bounds = np.c_[lo, hi]  # (d, 2)

    # --- 1) generate new samples (n_to_replace x d)
    X_new = generate_samples_above_threshold(
        gp, bounds, n=n_to_replace, threshold=threshold, seed=seed
    )  # (n, d)
    X_new = X_new.T  # → (d, n) to match column assignment

    # --- 2) pick which non-pivot columns to replace
    if replace_strategy == "least":
        # replace the non-pivot columns with the smallest GP means
        # y_non = np.ravel(gp.predict(cs[:, non_pivot].T))
        # idx_least = np.argsort(y_non)[:n_to_replace]
        # fill_cols = non_pivot[idx_least]
        fill_cols = np.sort(non_pivot)[-n_to_replace:] 
    else:
        # random choice among non-pivots
        rng = np.random.default_rng(seed)
        fill_cols = rng.choice(non_pivot, size=n_to_replace, replace=False)

    # --- 3) assign directly (no chained indexing)
    cs[:, fill_cols] = X_new[:, :len(fill_cols)]
    return cs


def convergence_study(kernel, function, sampler,
                      num_vars, generate_samples, num_new_samples,
                      update_kernel_scale_num_samples,
                      noise_level=0, return_samples=False,
                      norm=np.linalg.norm, callback=None, gp_kernel=None):

    # dirty hack to include two GP kernel types (for IVAR)
    if hasattr(kernel, "__len__"):
        # in this case, kernel is an array and we assume to have received
        # two kernels
        sampler_kernel = kernel[1]
        kernel = kernel[0]
    else:
        sampler_kernel = kernel

    # Instantiate a Gaussian Process model
    if gp_kernel is None:
        gp_kernel = kernel
    gp = AdaptiveGaussianProcess(
        gp_kernel, n_restarts_optimizer=4, alpha=1e-12)
    gp.setup(function, sampler)
    if hasattr(sampler, "set_gaussian_process"):
        sampler.set_gaussian_process(gp)
    
    print('sampler kerneobjl', kernel, 'gp kernel', gp_kernel)

    # Mesh the input space for evaluations of the real function,
    # the prediction and its MSE

    num_samples = np.cumsum(num_new_samples)
    num_steps = num_new_samples.shape[0]
    errors = np.empty(num_steps, dtype=float)
    nsamples = np.empty(num_steps, dtype=int)
    sample_step = 0
    optimizer_step = 0
    error_gp_comp = np.empty(num_steps, dtype=float)
    error_accept = 0
    while sample_step < num_steps:
        if hasattr(gp, 'kernel_'):
            # if using const * rbf + noise kernel
            # kernel.theta = gp.kernel_.k1.k2.theta
            # if using const * rbf
            # kernel.theta = gp.kernel_.k2.theta
            # if using rbf
            kernel.theta = gp.kernel_.theta

        # Fit to data using Maximum Likelihood Estimation of the parameters
        # if True:
        if ((optimizer_step >= update_kernel_scale_num_samples.shape[0]) or
            (sampler.ntraining_samples <
             update_kernel_scale_num_samples[optimizer_step])):
            gp.optimizer = None
        else:
            gp.optimizer = "fmin_l_bfgs_b"
            optimizer_step += 1

        flag = gp.refine(np.sum(num_new_samples[:sample_step+1]))
        
        # allow points to be added to gp more often than gp is evaluated for
        # validation
        vali_samples = np.loadtxt('../output/vali_samples.txt')
        x_vali = vali_samples[0:13, 0:100].T
        y_vali = vali_samples[-1, 0:100]
        if sampler.ntraining_samples >= num_samples[sample_step]:
            if callback is not None:
                callback(gp)
            print(gp.kernel_)
            print('N', sampler.ntraining_samples)
            nsamples[sample_step] = sampler.ntraining_samples
            pickle.dump(gp, open(f'gp_{np.mod(sample_step, 2)}.pkl', "wb"))
            
            # Use the    
            if sample_step >=1:
                # Compute error using validation samples
                pred_values_pre_gp = gp.predict(x_vali).flatten()
                error_gp_comp[sample_step] = root_mean_squared_error(y_vali, pred_values_pre_gp)
                error_gp_com_norm = r2_score(y_vali, pred_values_pre_gp)
                print('-----------error_gp_comp---------')
                print('N', num_samples[sample_step], 'Error', error_gp_comp[sample_step])  
                print('error_gp_com_norm', error_gp_com_norm)    
                if sample_step <= 3:
                    print('Do Not Update Candidate Samples.')
                else:
                # breakpoint()
                # threshold = np.quantile(gp.y_train_.squeeze(), 0.50)
                # if threshold >= 0:
                    # threshold = 0
                # if threshold  >= 0.38:
                #     threshold = 0.38
                    threshold = 0
                    print(f'Update Candidate Samples. Threshold = {threshold}')
                    # new_cand_size = sampler.candidate_samples.shape[1] - sampler.ntraining_samples
                    new_cand_size = 100
                    sampler.candidate_samples = update_candidate_fast(gp, sampler, threshold, new_cand_size)
                if (error_gp_comp[sample_step] <= 0.01) or (error_gp_com_norm > 0.92):
                    error_accept += 1  
            
                errors[sample_step] = error_gp_comp[sample_step]
            # Increase sample_step after checking the error of gp_load
            sample_step += 1   

        if flag > 0:
            errors, nsamples = errors[:sample_step], nsamples[:sample_step]
            print('Terminating study. Points are becoming ill conditioned')
            np.savetxt('error_gp_comp.txt', error_gp_comp)
            break
        if error_accept >= 3:
            print('Terminating study. GP performance is acceptable.')
            np.savetxt('error_gp_comp.txt', error_gp_comp)
            break
    np.savetxt('error_gp_comp.txt', error_gp_comp)        
    if return_samples:
        return errors, nsamples, sampler.training_samples

    return errors, nsamples

def unnormalized_posterior(gp, prior_pdf, samples, temper_param=1):
    prior_vals = prior_pdf(samples).squeeze()
    gp_vals = gp.predict(samples.T).squeeze()
    vals_max = max(gp_vals.max(), 0.1)
    # unnormalized_posterior_vals = prior_vals*(1 / (1 - gp_vals))**temper_param
    unnormalized_posterior_vals = prior_vals*np.exp(-(1 - gp_vals / vals_max))**temper_param
    return unnormalized_posterior_vals


class BayesianInferenceCholeskySampler(CholeskySampler):
    def __init__(self, prior_pdf, num_vars,
                 num_candidate_samples, variables,
                 max_num_samples=None, generate_random_samples=None,
                 temper=True, true_nll=None):
        self.prior_pdf = prior_pdf
        if not temper:
            self.temper_param = 1
        else:
            self.temper_param = 0
        self.true_nll = true_nll
        self.gp = None

        super().__init__(num_vars, num_candidate_samples, variables,
                         None, generate_random_samples)

    def set_gaussian_process(self, gp):
        self.gp = gp

    # Qian: understand the purpose of function increment_temper_param()
    def increment_temper_param(self, num_training_samples):

        # samples = np.random.uniform(0, 1, (self.nvars, 1000))
        samples = generate_independent_random_samples(self.variable, 3000)
        density_vals_prev = self.weight_function(samples)

        def objective(beta):
            new_weight_function = partial(
                unnormalized_posterior, self.gp, self.prior_pdf,
                temper_param=beta)
            density_vals = new_weight_function(samples)
            II = np.where(density_vals_prev > 1e-15)[0]
            JJ = np.where(density_vals_prev < 1e-15)[0]
            assert len(np.where(density_vals[JJ] > 1e-15)[0]) == 0
            ratio = np.zeros(samples.shape[1])
            ratio[II] = density_vals[II]/density_vals_prev[II]
            obj = ratio.std()/ratio.mean()
            return obj
        print('temper parameter', self.temper_param)
        x0 = self.temper_param + 1e-4
        # result = root(lambda b: objective(b)-1, x0)
        # x_opt = result.x
        # x_opt = bisect(lambda b: objective(b)-1, x0, 1)
        # if not optimize temper_param
        x_opt = self.temper_param + 1e-2
        self.temper_param = x_opt

    def __call__(self, num_samples):
        if self.gp is None:
            raise ValueError("must call self.set_gaussian_process()")
        
        if self.ntraining_samples > 0 and self.temper_param < 1:
            self.increment_temper_param(self.training_samples)

        assert self.temper_param <= 1
        if self.ntraining_samples == 0:
            weight_function = self.prior_pdf
        else:
            if self.true_nll is not None:
                def weight_function(x): return self.prior_pdf(x)*np.exp(
                    -self.true_nll(x)[:, 0])**self.temper_param
            else:
                weight_function = partial(
                    unnormalized_posterior, self.gp, self.prior_pdf,
                    temper_param=self.temper_param)

        self.set_weight_function(weight_function)

        samples, flag = super().__call__(num_samples)
        return samples, flag


def get_prior_samples(num_vars, variables, nsamples):
    rosenbrock_samples = generate_independent_random_samples(variables, nsamples)

    return rosenbrock_samples

def get_posterior_samples(num_vars, weight_function, nsamples, variables):
    x, w = get_tensor_product_quadrature_rule(
        200, num_vars, np.polynomial.legendre.leggauss,
        transform_samples=lambda x: (x+1)/2,
        density_function=lambda x: 0.5*np.ones(x.shape[1]))
    vals = weight_function(x)
    C = 1/vals.dot(w)

    def posterior_density(samples):
        return weight_function(samples)*C

    def proposal_density(samples):
        return np.ones(samples.shape[1])

    def generate_proposal_samples(nsamples):
        return generate_independent_random_samples(variables, nsamples)

    envelope_factor = C*vals.max()*1.1

    itrt_post_samples = rejection_sampling(
        posterior_density, proposal_density,
        generate_proposal_samples, envelope_factor,
        num_vars, nsamples, verbose=True,
        batch_size=None)

    return itrt_post_samples

def bayesian_inference_example():
    # read parameter distributions
    datapath = file_settings()[1]

    # define the variables for PCE
    param_file = file_settings()[-1]
    
    # Must set variables if not using uniform prior on [0,1]^D
    # variables = None
    ind_vars, variables = variables_prep(param_file, product_uniform='uniform', dummy=False)
    var_trans = AffineTransform(variables, enforce_bounds=True)
    init_scale = 50 # used to define length_scale for the kernel
    num_vars = variables.nvars
    num_candidate_samples = 10000
    # num_new_samples = np.asarray([20]+[8]*10+[16]*20+[24]*16+[40]*14)
    num_new_samples = np.asarray([20]+[10]*10+[15]*15+[20]*10+[40]*10)

    nvalidation_samples = 1000

    from scipy import stats 
    prior_pdf = partial(tensor_product_pdf, 
        univariate_pdfs=[partial(stats.beta.pdf, a=1, b=1, scale=ind_vars[ii].args[1]) for ii in range(num_vars)])


    # Get validation samples from prior
    rosenbrock_samples = get_prior_samples(num_vars, variables, nvalidation_samples + num_candidate_samples)

    def generate_random_samples(nsamples, idx=0):
        assert idx+nsamples <= rosenbrock_samples.shape[1]
        return rosenbrock_samples[:, idx:idx+nsamples]

    generate_validation_samples = partial(
        generate_random_samples, nvalidation_samples,
        idx=num_candidate_samples)

    def get_filename(method, fixed_scale):
        filename = 'bayes-example-%s-d-%d-n-%d.npz' % (
            method, num_vars, num_candidate_samples)
        if not fixed_scale:
            filename = filename[:-4]+'-opt.npz'
        return filename

    # defining kernel
    length_scale = [init_scale, init_scale, *(3*np.ones(num_vars -2, dtype=float))]
    kernel = RBF(length_scale, [(5e-2, 200), (5e-2, 100), (5e-2, 20), (5e-2, 20),
        (5e-2, 20), (5e-2, 20), (5e-2, 20), (5e-2, 20), (5e-2, 20), 
        (5e-2, 20), (5e-2, 20), (5e-2, 20), (5e-2, 20)])

    # this is the one Qian should use. The others are for comparision only
    adaptive_cholesky_sampler = BayesianInferenceCholeskySampler(
        prior_pdf, num_vars, num_candidate_samples, variables,
        max_num_samples=num_new_samples.sum(),
        generate_random_samples=None)
    adaptive_cholesky_sampler.set_kernel(copy.deepcopy(kernel))

    samplers = [adaptive_cholesky_sampler]
    methods = ['Learning-Weighted-Cholesky-b']
    labels = [r'$\mathrm{Adapted\;Weighted\;Cholesky}$']
    fixed_scales = [False]

    for sampler, method, fixed_scale in zip(samplers, methods, fixed_scales):
        filename = get_filename(method, fixed_scale)
        print(filename)
        if os.path.exists(filename):
            continue

        if fixed_scale:
            update_kernel_scale_num_samples = np.empty(0)
        else:
            update_kernel_scale_num_samples = np.cumsum(num_new_samples)

        cond_nums = []
        temper_params = []

        def callback(gp):
            cond_nums.append(np.linalg.cond(gp.L_.dot(gp.L_.T)))
            if hasattr(sampler, 'temper_param'):
                temper_params.append(sampler.temper_param)
                print(temper_params)

        errors, nsamples, samples = convergence_study(
            kernel, run_source_lsq, sampler, num_vars,
            generate_validation_samples, num_new_samples,
            update_kernel_scale_num_samples, callback=callback,
            return_samples=True)

        np.savez(filename, nsamples=nsamples, errors=errors,
                 cond_nums=np.asarray(cond_nums), samples=samples,
                 temper_params=np.asarray(temper_params))

    fig, axs = plt.subplots(1, 3, figsize=(3*8, 6), sharey=False)
    styles = ['-']
    # styles = ['k-','r-.','b--','g:']
    for method, label, ls, fixed_scale in zip(
            methods, labels, styles, fixed_scales):
        filename = get_filename(method, fixed_scale)
        data = np.load(filename)
        nsamples, errors = data['nsamples'][:-1], data['errors'][:-1]
        temper_params, cond_nums = data['temper_params'][1:-1], data['cond_nums'][:-1]
        axs[0].loglog(nsamples, errors, ls=ls, label=label)
        axs[1].loglog(nsamples, cond_nums, ls=ls, label=label)
        axs[2].semilogy(np.arange(1, nsamples.shape[0]),
                    temper_params, 'k-o')
        axs[2].set_xlabel(r'$\mathrm{Iteration}$ $j$')
        axs[2].set_ylabel(r'$\beta_j$')

    for ii in range(2):
        axs[ii].set_xlabel(r'$m$')
        axs[ii].set_xlim(10, 1000)
    axs[0].set_ylabel(r'$\tilde{\epsilon}_{\omega,2}$', rotation=90)
    ylim0 = axs[0].get_ylim()
    ylim1 = axs[1].get_ylim()
    ylim = [min(ylim0[0], ylim1[0]), max(ylim0[1], ylim1[1])]
    axs[0].set_ylim(ylim)
    axs[1].set_ylim(ylim)
    axs[1].set_ylabel(r'$\kappa$', rotation=90)

    figname = 'bayes_example_comparison_%d.pdf' % num_vars
    axs[0].legend()
    plt.savefig(figname) 

if __name__ == '__main__':
    try:
        import sklearn
    except:
        msg = 'Install sklearn using pip install sklearn'
        raise Exception(msg)
    # Create the copy of models and veneer list
    project_name = 'MW_BASE_RC10.rsproj'
    veneer_name = 'vcmd45\\FlowMatters.Source.VeneerCmd.exe'   
    first_port=15000; num_copies = 8
    _, things_to_record, _, _, _ = modeling_settings()
    processes, ports = paralell_vs(first_port, num_copies, project_name, veneer_name)

    vs_list = vs_settings(ports, things_to_record)
    # obtain the initial values of parameters 
    initial_values = obtain_initials(vs_list[0])

    bayesian_inference_example()

