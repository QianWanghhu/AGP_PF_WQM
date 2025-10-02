#!/usr/bin/env ffexplore
import numpy as np
import pyapprox as pya
from scipy.stats import uniform
from sklearn.model_selection import KFold

from pyapprox.approximate import approximate
from pyapprox.utilities import total_degree_space_dimension
from scipy import stats
from pyapprox.variable_transformations import AffineRandomVariableTransformation
from pyapprox.multivariate_polynomials import PolynomialChaosExpansion
from pyapprox.univariate_quadrature import gauss_jacobi_pts_wts_1D
from pyapprox.variables import get_distribution_info

def identity_fun(x):
    return x

def get_poly_opts(variable, product_uniform):

    if product_uniform != 'exact':
        var_trans = AffineRandomVariableTransformation(
            variable)
        poly_opts = define_poly_options_from_variable_transformation(var_trans)
        return poly_opts, var_trans
    
    from basic.read_data import file_settings
    from basic.read_data import variables_prep
    input_path = file_settings()[1]
    filename = file_settings()[4]
    index_product = np.load(f'{input_path}index_product.npy', allow_pickle=True)
    full_variable = variables_prep(filename, product_uniform=False)
    var_trans = AffineRandomVariableTransformation(
        variable)
    poly = PolynomialChaosExpansion()
    basis_opts = dict()
    identity_map_indices = []
    cnt = 0
    for ii in range(variable.nunique_vars):
        
        rv = variable.unique_variables[ii]
        name, scales, shapes = get_distribution_info(rv)
        if (type(rv.dist) != stats._continuous_distns.beta_gen):
            opts = {'rv_type': name, 'shapes': shapes,
                    'var_nums': variable.unique_variable_indices[ii]}
            basis_opts['basis%d' % ii] = opts
            continue

        identity_map_indices += list(variable.unique_variable_indices[ii])
        
        quad_rules = []
        inds = index_product[cnt]
        nquad_samples_1d = 100

        for jj in inds:
            # breakpoint()
            a, b = full_variable.all_variables()[jj].interval(1)
            x, w = gauss_jacobi_pts_wts_1D(nquad_samples_1d, 0, 0)
            x = (x+1)/2 # map to [0, 1]
            x = (b-a)*x+a # map to [a,b]
            quad_rules.append((x, w))
        funs = [identity_fun]*len(inds)
        basis_opts['basis%d' % ii] = {'poly_type': 'product_indpnt_vars',
                                      'var_nums': variable.unique_variable_indices[ii],
                                      'funs': funs,
                                      'quad_rules': quad_rules}
        cnt += 1
        
    poly_opts = {'var_trans': var_trans}
    poly_opts['poly_types'] = basis_opts
    var_trans.set_identity_maps(identity_map_indices)
    return poly_opts, var_trans


from pyapprox.indexing import compute_hyperbolic_indices
from pyapprox.multivariate_polynomials import \
        define_poly_options_from_variable_transformation
def fun(variable, train_samples, train_values, product_uniform, nboot=10):
    poly_opts, var_trans = get_poly_opts(variable, product_uniform)
   
    nterms = total_degree_space_dimension(train_samples.shape[0], 2)
    # breakpoint()
    # Find best PCE basis
    nfolds = min(nboot, train_samples.shape[1])
    solver_options = {'cv': nfolds}
    options = {'basis_type': 'expanding_basis', 'variable': variable,
               'verbosity': 1, 'poly_opts': poly_opts,
                'options': {'max_num_init_terms': nterms,
                'max_num_expansion_steps_iter': 3,
               'linear_solver_options': solver_options}}
    approx_res = approximate(train_samples, train_values, 'polynomial_chaos', options)

    # Compute PCE on each fold using best PCE basis and least squares
    linear_solver_options = [
        {'alpha':approx_res.reg_params[ii]}
        for ii in range(len(approx_res.reg_params))]
    indices = [approx_res.approx.indices[:, np.where(np.absolute(c)>0)[0]]
               for c in approx_res.approx.coefficients.T]

    # for now just use quadratic basis
    # indices = compute_hyperbolic_indices(variable.num_vars(), 2) # comment out for basis selection
    options = {'basis_type': 'fixed', 'variable': variable,
               'poly_opts': poly_opts,
               'options': {'linear_solver_options': dict(),
                           'indices': indices, 'solver_type': 'lstsq'}}
    from pyapprox.approximate import cross_validate_approximation
    # this does not use fast leave many out cross validation for least squares
    # (which is used by approximate because that function does not return
    # all the approximations on each fold
    
    # error occurred in the following line 
    approx_list, residues_list, cv_score = cross_validate_approximation(
        train_samples, train_values, options, nfolds,
        'polynomial_chaos', random_folds='sklearn')
    # import pdb; pdb.set_trace()
    pce_cv_total_effects = []
    pce_cv_main_effects = []
    for ii in range(nfolds):
        pce_sa_res_ii = pya.analyze_sensitivity_polynomial_chaos(
            approx_list[ii])
        pce_cv_main_effects.append(pce_sa_res_ii.main_effects)
        pce_cv_total_effects.append(pce_sa_res_ii.total_effects)
    return cv_score, pce_cv_main_effects, pce_cv_total_effects, approx_list
