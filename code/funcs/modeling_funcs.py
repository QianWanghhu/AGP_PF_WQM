# RUN SOURCE to generate observation_ensemble.csv
import pandas as pd
import numpy as np
import veneer
from veneer.pest_runtime import *
import time
import os
from veneer.manage import start,kill_all_now
# Import packages
from SALib.sample import latin


def paralell_vs(first_port, num_copies, project_name, veneer_name):
    first_port = first_port
    num_copies = num_copies
    # define catchment_project and veneer_exe
    project_path = 'pest_source/'
    catchment_project= project_path + project_name

    # Setup Veneer
    # define paths to veneer command and the catchment project
    veneer_path = project_path + veneer_name   
    #Now, go ahead and start source
    processes, ports = start(catchment_project,
                            n_instances=num_copies,
                            ports=first_port,
                            debug=True,
                            veneer_exe=veneer_path,
                            remote=False,
                            overwrite_plugins=True)

    return processes, ports

def modeling_settings():
    """
    Set and return user_defined arguments for model configuration, recording and retrieving results.
    Returns:
    NODEs: list, list of elements to record
    things_to_record, list of dict, to confifure the model
    criteria: dict, criteria used to filter results
    start_date, end_date
    """
    NODEs = ['gauge_126001A_SandyCkHomebush']
    things_to_record = [{'NetworkElement':node,'RecordingVariable':'Constituents@N_DIN@Downstream Flow Mass'} for node in NODEs]
    criteria = {'NetworkElement': NODEs[0],'RecordingVariable':'Constituents@N_DIN@Downstream Flow Mass'}
    start_date = '01/07/2007'; end_date='30/06/2018'
    assert isinstance(start_date, str),"start_date has to be time str."
    assert isinstance(end_date, str),"end_date has to be time str."
    assert isinstance(things_to_record, list),"things_to_record has to be a list of dict."
    return NODEs, things_to_record, criteria, start_date, end_date


def generate_observation_default(v, criteria, start_date, end_date, retrieve_time):
    """
    Run the model to obtain observations without noise.
    Parameters:
    ===========
    vs: veneer object
    criteria: dict, criteria used to configure the model
    start_date, end_date: str of dates when the model simulates, e.g., 
        start_date = '01/07/2014'; end_date='01/07/2016'

    Returns:
    ===========
    load: pd.DataFrame, loads to return
    """
    # store the results into a .csv file
    time_start = time.time()
    load = pd.DataFrame()
    assert isinstance(v, veneer.general.Veneer),"vs has to be an veneer object."
    
    assert isinstance(retrieve_time, list),"retrieve_time has to be a list of timestamp."
    
    print('-----------Run the model to produce synthetic observations-----------')
    v.drop_all_runs()
    v.run_model(params={'NoHardCopyResults':True}, start = start_date, end = end_date) 
    column_names = ['DIN_obs_load']
    # import pdb; pdb.set_trace()
    get_din = v.retrieve_multiple_time_series(criteria=criteria)
    get_din.columns = column_names
    din = get_din.loc[retrieve_time[0]:retrieve_time[1]]
    # store the daily results and the index of sampling
    load = pd.concat([load, din], axis=1)
    v.drop_all_runs()
    time_end = time.time()
    print(f'{time_end - time_start} seconds')
    print('----------Finished generate_observation_noised-----------')
    return load
# End generate_observation_noised()

def generate_observation_ensemble(vs_list, criteria, start_date, end_date, parameters, retrieve_time, initial_values):
    """
    Run the model to obtain observations without noise.
    Parameters:
    ===========
    vs: veneer object
    criteria: dict, criteria used to configure the model
    start_date, end_date: str of dates when the model simulates, e.g., 
        start_date = '01/07/2014'; end_date='01/07/2016'

    Returns:
    ===========
    load: pd.DataFrame, loads to return
    """
    assert isinstance(vs_list, list),"vs has to be a list of veneer objects."
    num_copies = len(vs_list)     
    # install the results into a csv
    load = pd.DataFrame()
    time_start = time.time()
    # parameters = parameters.iloc[0:12, :]
    num_runs = parameters.shape[0]
    group_loops = np.floor_divide(num_runs, num_copies) + 1
    total_runs = 0
    for index_group in range(group_loops):
        group_run_responses = []
        if index_group == (group_loops - 1):
            num_copies_loop = (num_runs - index_group * num_copies)
        else:
            num_copies_loop = num_copies

        for j in range(num_copies_loop):
            total_runs += 1
            if (index_group * num_copies + j) >= num_runs: break

            vs= vs_list[j]
            vs.drop_all_runs()
            parameter_dict = parameters.iloc[total_runs-1]
            # Make sure names of parameters are correct!
            vs = change_param_values(vs, parameter_dict, initial_values=initial_values)
            response = vs.run_model(params={'NoHardCopyResults':True}, start=start_date, end=end_date, run_async=True)
            group_run_responses.append(response)

        for j in range(num_copies_loop):
            run_index = index_group * num_copies + j
            if (run_index) >= num_runs: break
                
            vs = vs_list[j]
            r = group_run_responses[j]   
            code = r.getresponse().getcode() # wait until the job finished   
            column_names = ['DIN_' + str(run_index)]
            get_din = vs.retrieve_multiple_time_series(criteria=criteria)
            get_din.columns = column_names
            din = get_din.loc[retrieve_time[0]:retrieve_time[1]]
            # store the daily results and the index of sampling
            load = pd.concat([load, din], axis=1)
        
        print(f'Finish {total_runs} runs')

    # kill_all_now()
    time_end = time.time()
    print(f'{time_end - time_start} seconds')
    print('----------Finished generate_observation_ensemble-----------')
    return load
# End generate_observation_ensemble() 

def generate_parameter_ensemble(nsample, param_ensemble, datapath, seed=None):
    """
    The function is used to generate the parameter and data ensemble.
    The parameters are generated with Sobol' sampling or LHS.
    Parameters:
    ===========
    nsample: int, the number of parameter samples to generate (e.g., 512)
    param_ensemble: the name containing the relative path of parameter ensemble
    seed: int, the random seed to generate samples, default is None

    Returns:
    ===========
    None. Save parameter results to the given path.
    """
    fname = param_ensemble
    if not os.path.exists(fname):      
        parameters = pd.read_csv(datapath + 'Parameters-PCE.csv', index_col = 'Index')

        problem = {
            'num_vars': parameters.shape[0],
            'names': parameters.Name_short.values,
            'bounds': parameters.loc[:, ['Min', 'Max']].values
            }
        parameters_ensemble = latin.sample(problem, nsample, seed=88)
        df = pd.DataFrame(data=parameters_ensemble, index = np.arange(nsample), columns=problem['names'])
        df.index.name = 'real_name'
        df.to_csv(param_ensemble)
    else:
        print(f'The file of parameter ensemble exists under the folder')

def change_param_values(v, pvalue_dict, fromList=False, initial_values=None, subcatment=None):
    assert isinstance(v, veneer.general.Veneer),"vs has to be an veneer object."
    if initial_values is not None:
        dwc_init = np.array(initial_values['DWC'])
        dwc_set = np.where(dwc_init>0, pvalue_dict['DWC'], 0)
        
    if subcatment is None:
        v.model.catchment.generation.set_param_values('DeliveryRatioSurface',pvalue_dict['DRF'],  fus=['Sugarcane'], fromList=fromList)
        v.model.catchment.generation.set_param_values('DeliveryRatioSeepage',pvalue_dict['DRP'],  fus=['Sugarcane'], fromList=fromList)
        v.model.catchment.generation.set_param_values('DWC', dwc_set, fus=['Sugarcane'], fromList=True)
        # v.model.catchment.generation.set_param_values('Load_Conversion_Factor', pvalue_dict['LCF'], fus=['Sugarcane'], fromList=fromList)
        v.model.catchment.generation.set_param_values('dissConst_DWC', pvalue_dict['gfDWC'], fus=['Grazing Forested'], fromList=fromList)
        v.model.catchment.generation.set_param_values('dissConst_EMC', pvalue_dict['gfEMC'], fus=['Grazing Forested'], fromList=fromList)
        v.model.catchment.generation.set_param_values('dissConst_DWC', pvalue_dict['goDWC'], fus=['Grazing Open'], fromList=fromList)
        v.model.catchment.generation.set_param_values('dissConst_EMC', pvalue_dict['goEMC'], fus=['Grazing Open'], fromList=fromList)
        v.model.catchment.generation.set_param_values('dissConst_DWC', pvalue_dict['cDWC'], fus=['Conservation'], fromList=fromList)
        v.model.catchment.generation.set_param_values('dissConst_EMC', pvalue_dict['cEMC'], fus=['Conservation'], fromList=fromList)
        v.model.catchment.generation.set_param_values('dissConst_DWC', pvalue_dict['fDWC'], fus=['Forestry'], fromList=fromList)
        v.model.catchment.generation.set_param_values('dissConst_EMC', pvalue_dict['fEMC'], fus=['Forestry'], fromList=fromList)
        v.model.catchment.generation.set_param_values('dissConst_DWC', pvalue_dict['oDWC'], fus=['Horticulture', 'Urban', 'Water', 'Other', 'Irrigated Cropping'], fromList=fromList)
        v.model.catchment.generation.set_param_values('dissConst_EMC', pvalue_dict['oEMC'], fus=['Horticulture', 'Urban', 'Water', 'Other', 'Irrigated Cropping'], fromList=fromList)
    else:
        for subcat in subcatment:
            v.model.catchment.generation.set_param_values('DeliveryRatioSurface',pvalue_dict[f'DRF{subcat}'],  fus=['Sugarcane'], catchments=f'SC #{subcat}', fromList=fromList)
            v.model.catchment.generation.set_param_values('DeliveryRatioSeepage',pvalue_dict[f'DRP{subcat}'],  fus=['Sugarcane'], catchments=f'SC #{subcat}', fromList=fromList)
            v.model.catchment.generation.set_param_values('DWC', pvalue_dict[f'DWC{subcat}'], fus=['Sugarcane'], catchments=f'SC #{subcat}', fromList=fromList)
            # v.model.catchment.generation.set_param_values('Load_Conversion_Factor', pvalue_dict[f'LCF{subcat}'], fus=['Sugarcane'], catchments=f'SC #{subcat}', fromList=fromList)
            v.model.catchment.generation.set_param_values('dissConst_DWC', pvalue_dict[f'gfDWC{subcat}'], fus=['Grazing Forested'], catchments=f'SC #{subcat}', fromList=fromList)
            v.model.catchment.generation.set_param_values('dissConst_EMC', pvalue_dict[f'gfEMC{subcat}'], fus=['Grazing Forested'], catchments=f'SC #{subcat}', fromList=fromList)
            v.model.catchment.generation.set_param_values('dissConst_DWC', pvalue_dict[f'goDWC{subcat}'], fus=['Grazing Open'], catchments=f'SC #{subcat}', fromList=fromList)
            v.model.catchment.generation.set_param_values('dissConst_EMC', pvalue_dict[f'goEMC{subcat}'], fus=['Grazing Open'], catchments=f'SC #{subcat}', fromList=fromList)
            v.model.catchment.generation.set_param_values('dissConst_DWC', pvalue_dict[f'cDWC{subcat}'], fus=['Conservation'], catchments=f'SC #{subcat}', fromList=fromList)
            v.model.catchment.generation.set_param_values('dissConst_EMC', pvalue_dict[f'cEMC{subcat}'], fus=['Conservation'], catchments=f'SC #{subcat}', fromList=fromList)
            v.model.catchment.generation.set_param_values('dissConst_DWC', pvalue_dict[f'fDWC{subcat}'], fus=['Forestry'], catchments=f'SC #{subcat}', fromList=fromList)
            v.model.catchment.generation.set_param_values('dissConst_EMC', pvalue_dict[f'fEMC{subcat}'], fus=['Forestry'], catchments=f'SC #{subcat}', fromList=fromList)
            v.model.catchment.generation.set_param_values('dissConst_DWC', pvalue_dict[f'oDWC{subcat}'], fus=['Horticulture', 'Urban', 'Water', 'Other', 'Irrigated Cropping'], catchments=f'SC #{subcat}', fromList=fromList)
            v.model.catchment.generation.set_param_values('dissConst_EMC', pvalue_dict[f'oEMC{subcat}'], fus=['Horticulture', 'Urban', 'Water', 'Other', 'Irrigated Cropping'], catchments=f'SC #{subcat}', fromList=fromList)
    return v

# obtain initial values
def obtain_initials(v, subcatchment=None):
    """Obtain the initial values of the model.
    Parameters:
    ===========
    v: veneer objects.

    Returns:
    ========
    initial_values: dict, initla values of parameters
    """
    assert isinstance(v, veneer.general.Veneer),"vs has to be an veneer object."

    initial_values = {}
    if subcatchment is None:
        initial_values['DRF'] = v.model.catchment.generation.get_param_values('DeliveryRatioSurface', fus=['Sugarcane'])
        initial_values['DRP'] = v.model.catchment.generation.get_param_values('DeliveryRatioSeepage',  fus=['Sugarcane'])
        initial_values['DWC'] = v.model.catchment.generation.get_param_values('DWC', fus=['Sugarcane'])
        initial_values['LCF'] = v.model.catchment.generation.get_param_values('Load_Conversion_Factor', fus=['Sugarcane'])
        initial_values['gfDWC'] = v.model.catchment.generation.get_param_values('dissConst_DWC', fus=['Grazing Forested'])
        initial_values['gfEMC'] = v.model.catchment.generation.get_param_values('dissConst_EMC', fus=['Grazing Forested'])
        initial_values['goDWC'] = v.model.catchment.generation.get_param_values('dissConst_DWC', fus=['Grazing Open'])
        initial_values['goEMC'] = v.model.catchment.generation.get_param_values('dissConst_EMC', fus=['Grazing Open'])
        initial_values['cDWC'] = v.model.catchment.generation.get_param_values('dissConst_DWC', fus=['Conservation'])
        initial_values['cEMC'] = v.model.catchment.generation.get_param_values('dissConst_EMC', fus=['Conservation'])
        initial_values['fDWC'] = v.model.catchment.generation.get_param_values('dissConst_DWC', fus=['Forestry'])
        initial_values['fEMC'] = v.model.catchment.generation.get_param_values('dissConst_EMC', fus=['Forestry'])
        initial_values['oDWC'] = v.model.catchment.generation.get_param_values('dissConst_DWC', fus=['Horticulture', 'Urban', 'Water', 'Other', 'Irrigated Cropping'])
        initial_values['oEMC'] = v.model.catchment.generation.get_param_values('dissConst_EMC', fus=['Horticulture', 'Urban', 'Water', 'Other', 'Irrigated Cropping'])
    else:
        for subcat in subcatchment:
            initial_values[f'DRF{subcat}'] = v.model.catchment.generation.get_param_values('DeliveryRatioSurface', catchments=f'SC #{subcat}', fus=['Sugarcane'])
            initial_values[f'DRP{subcat}'] = v.model.catchment.generation.get_param_values('DeliveryRatioSeepage', catchments=f'SC #{subcat}', fus=['Sugarcane'])
            initial_values[f'DWC{subcat}'] = v.model.catchment.generation.get_param_values('DWC', catchments=f'SC #{subcat}', fus=['Sugarcane'])
            initial_values[f'LCF{subcat}'] = v.model.catchment.generation.get_param_values('Load_Conversion_Factor', catchments=f'SC #{subcat}', fus=['Sugarcane'])
            initial_values[f'gfDWC{subcat}'] = v.model.catchment.generation.get_param_values('dissConst_DWC', catchments=f'SC #{subcat}', fus=['Grazing Forested'])
            initial_values[f'gfEMC{subcat}'] = v.model.catchment.generation.get_param_values('dissConst_EMC', catchments=f'SC #{subcat}', fus=['Grazing Forested'])
            initial_values[f'goDWC{subcat}'] = v.model.catchment.generation.get_param_values('dissConst_DWC', catchments=f'SC #{subcat}', fus=['Grazing Open'])
            initial_values[f'goEMC{subcat}'] = v.model.catchment.generation.get_param_values('dissConst_EMC', catchments=f'SC #{subcat}', fus=['Grazing Open'])
            initial_values[f'cDWC{subcat}'] = v.model.catchment.generation.get_param_values('dissConst_DWC', catchments=f'SC #{subcat}', fus=['Conservation'])
            initial_values[f'cEMC{subcat}'] = v.model.catchment.generation.get_param_values('dissConst_EMC', catchments=f'SC #{subcat}', fus=['Conservation'])
            initial_values[f'fDWC{subcat}'] = v.model.catchment.generation.get_param_values('dissConst_DWC', catchments=f'SC #{subcat}', fus=['Forestry'])
            initial_values[f'fEMC{subcat}'] = v.model.catchment.generation.get_param_values('dissConst_EMC', catchments=f'SC #{subcat}', fus=['Forestry'])
            initial_values[f'oDWC{subcat}'] = v.model.catchment.generation.get_param_values('dissConst_DWC', catchments=f'SC #{subcat}', fus=['Horticulture', 'Urban', 'Water', 'Other', 'Irrigated Cropping'])
            initial_values[f'oEMC{subcat}'] = v.model.catchment.generation.get_param_values('dissConst_EMC', catchments=f'SC #{subcat}', fus=['Horticulture', 'Urban', 'Water', 'Other', 'Irrigated Cropping'])

    return initial_values

def vs_settings(ports, things_to_record):
    """
    Set up the vs objects.
    Parameters:
    ===========
    ports: list, list of ports to generate veneer objects
    things_to_record: dict, configurations of vs objects

    Returns:
    ===========
    vs_list: list, list of vs ojeects

    """
    
    vs_list = []
    assert isinstance(ports, list),"vs has to be a list of int."

    for veneer_port in ports:
        # veneer_port = first_port 
        vs_list.append(veneer.Veneer(port=veneer_port))
    
    for vs in vs_list:
        vs.configure_recording(disable=[{}], enable=things_to_record)
    return vs_list
