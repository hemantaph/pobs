import os
import random
import numpy as np
from importlib import resources
import itertools

from scipy.special import comb
import pickle
import json
import h5py
import bilby

class NumpyEncoder(json.JSONEncoder):
    """
    Class for storing a numpy.ndarray or any nested-list composition as JSON file. This is required for dealing np.nan and np.inf.

    Parameters
    ----------
    json.JSONEncoder : `class`
        class for encoding JSON file

    Returns
    ----------
    json.JSONEncoder.default : `function`
        function for encoding JSON file

    Example
    ----------
    >>> import numpy as np
    >>> import json
    >>> from ler import helperroutines as hr
    >>> # create a dictionary
    >>> param = {'a': np.array([1,2,3]), 'b': np.array([4,5,6])}
    >>> # save the dictionary as json file
    >>> with open('param.json', 'w') as f:
    >>>     json.dump(param, f, cls=hr.NumpyEncoder)
    >>> # load the dictionary from json file
    >>> with open('param.json', 'r') as f:
    >>>     param = json.load(f)
    >>> # print the dictionary
    >>> print(param)
    {'a': [1, 2, 3], 'b': [4, 5, 6]}
    """

    def default(self, obj):
        """function for encoding JSON file"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def load_pickle(file_name):
    """Load a pickle file.

    Parameters
    ----------
    file_name : `str`
        pickle file name for storing the parameters.

    Returns
    ----------
    param : `dict`
    """
    with open(file_name, "rb") as handle:
        param = pickle.load(handle)

    return param

def save_pickle(file_name, param):
    """Save a dictionary as a pickle file.

    Parameters
    ----------
    file_name : `str`
        pickle file name for storing the parameters.
    param : `dict`
        dictionary to be saved as a pickle file.
    """
    with open(file_name, "wb") as handle:
        pickle.dump(param, handle, protocol=pickle.HIGHEST_PROTOCOL)

# hdf5
def load_hdf5(file_name):
    """Load a hdf5 file.

    Parameters
    ----------
    file_name : `str`
        hdf5 file name for storing the parameters.

    Returns
    ----------
    param : `dict`
    """

    return h5py.File(file_name, 'r')

def save_hdf5(file_name, param):
    """Save a dictionary as a hdf5 file.

    Parameters
    ----------
    file_name : `str`
        hdf5 file name for storing the parameters.
    param : `dict`
        dictionary to be saved as a hdf5 file.
    """
    with h5py.File(file_name, 'w') as f:
        for key, value in param.items():
            f.create_dataset(key, data=value)

def load_json(file_name):
    """Load a json file.

    Parameters
    ----------
    file_name : `str`
        json file name for storing the parameters.

    Returns
    ----------
    param : `dict`
    """
    with open(file_name, "r", encoding="utf-8") as f:
        param = json.load(f)

    return param

def save_json(file_name, param):
    """Save a dictionary as a json file.

    Parameters
    ----------
    file_name : `str`
        json file name for storing the parameters.
    param : `dict`
        dictionary to be saved as a json file.
    """
    with open(file_name, "w", encoding="utf-8") as write_file:
        json.dump(param, write_file)

def append_json(file_name, new_dictionary, old_dictionary=None, replace=False):
    """
    Append (values with corresponding keys) and update a json file with a dictionary. There are four options:

    1. If old_dictionary is provided, the values of the new dictionary will be appended to the old dictionary and save in the 'file_name' json file.
    2. If replace is True, replace the json file (with the 'file_name') content with the new_dictionary.
    3. If the file does not exist, create a new one with the new_dictionary.
    4. If none of the above, append the new dictionary to the content of the json file.

    Parameters
    ----------
    file_name : `str`
        json file name for storing the parameters. 
    new_dictionary : `dict`
        dictionary to be appended to the json file.
    old_dictionary : `dict`, optional
        If provided the values of the new dictionary will be appended to the old dictionary and save in the 'file_name' json file. 
        Default is None.
    replace : `bool`, optional
        If True, replace the json file with the dictionary. Default is False.

    """

    # check if the file exists
    # time
    # start = datetime.datetime.now()
    if old_dictionary:
        data = old_dictionary
    elif replace:
        data = new_dictionary
    elif not os.path.exists(file_name):
        #print(f" {file_name} file does not exist. Creating a new one...")
        replace = True
        data = new_dictionary
    else:
        #print("getting data from file")
        with open(file_name, "r", encoding="utf-8") as f:
            data = json.load(f)
    # end = datetime.datetime.now()
    # print(f"Time taken to load the json file: {end-start}")

    # start = datetime.datetime.now()
    if not replace:
        data = add_dictionaries_together(data, new_dictionary)
        # data_key = data.keys()
        # for key, value in new_dictionary.items():
        #     if key in data_key:
        #         data[key] = np.concatenate((data[key], value)).tolist()
    # end = datetime.datetime.now()
    # print(f"Time taken to append the dictionary: {end-start}")

    # save the dictionary
    # start = datetime.datetime.now()
    #print(data)
    with open(file_name, "w", encoding="utf-8") as write_file:
        json.dump(data, write_file, indent=4, cls=NumpyEncoder)
    # end = datetime.datetime.now()
    # print(f"Time taken to save the json file: {end-start}")

    return data

# def add_dict_values(dict1, dict2):
#     """Adds the values of two dictionaries together.
    
#     Parameters
#     ----------
#     dict1 : `dict`
#         dictionary to be added.
#     dict2 : `dict`
#         dictionary to be added.

#     Returns
#     ----------
#     dict1 : `dict`
#         dictionary with added values.
#     """
#     data_key = dict1.keys()
#     for key, value in dict2.items():
#         if key in data_key:
#             dict1[key] = np.concatenate((dict1[key], value))

    return dict1

def get_param_from_json(json_file):
    """
    Function to get the parameters from json file.

    Parameters
    ----------
    json_file : `str`
        json file name for storing the parameters.

    Returns
    ----------
    param : `dict`
    """
    with open(json_file, "r", encoding="utf-8") as f:
        param = json.load(f)

    for key, value in param.items():
        param[key] = np.array(value)
    return param


def get_dict_or_file(path):

        if isinstance(path, str):
            # if it is a json file
            if path.endswith('.json'):
                result = load_json(path)
            elif path.endswith('.pkl') or path.endswith('.pickle'):
                result = load_pickle(path)
            elif path.endswith('.h5') or path.endswith('.hdf5'):
                result = load_hdf5(path)
            else:
                raise ValueError("dict should be a json, pickle or h5 file")
        elif isinstance(path, dict):
            result = path
        else:
            raise ValueError("dict should be a string to json path or a dictionary")
        return result

def dir_check(path):

    # check if pobs_directory exists, if not create one
    if not os.path.exists(path):
        os.makedirs(path)
    # check if pobs_directory ends with '/', if not add '/'
    if not path.endswith('/'):
        path += '/'

    return path

def load_data_from_module(package, directory, filename):
    """
    Function to load a specific dataset from a .json .txt .pkl .hdf5 file within the package

    Parameters
    ----------
    package : str
        name of the package
    directory : str
        name of the directory within the package
    filename : str
        name of the .json file

    Returns
    ----------
    data : `dict`
        Dictionary loaded from the .json file
    """


    with resources.path(package + '.' + directory, filename) as path_:
        if filename.endswith('.json'):
            with open(path_, "r", encoding="utf-8") as f:
                return load_json(path_)
        elif filename.endswith('.pkl'):
            with open(path_, "rb") as f:
                return pickle.load(f)
        elif filename.endswith('.hdf5'):
            return load_hdf5(path_)


def data_check_astro_lensed(data_dict):

    lensed_param = {}

    param_list = ['ra', 'dec', 'theta_jn', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'mass_1', 'mass_2', 'effective_luminosity_distance', 'effective_geocent_time', 'optimal_snr_net']
    for key in param_list:
        try:
            lensed_param[key] = np.array(data_dict[key])
        except:
            print(f"data_dict should have the following keys: {param_list}")
            raise ValueError(f"{key} is not present in the data_dict")

    # seperate out image 1, 2
    # this is wrt to time of arrival
    lensed_param_1 = {}
    lensed_param_2 = {}

    for key, value in lensed_param.items():
        if np.shape(np.shape(value))[0]==2:
            lensed_param_1[key] = np.array(value)[:,0]
            lensed_param_2[key] = np.array(value)[:,1]
        else:
            lensed_param_1[key] = np.array(value)
            lensed_param_2[key] = np.array(value)

    # For image 1 and 2 only
    # only keep snr > 8
    idx_snr1 = lensed_param_1['optimal_snr_net'] > 8
    idx_snr2 = lensed_param_2['optimal_snr_net'] > 8
    idx_snr = idx_snr1 & idx_snr2

    # with effective spin
    # Note: chi_eff for image 1 and 2 is the same
    chi_eff = (lensed_param_1['a_1']*np.cos(lensed_param_1['tilt_1']) + lensed_param_1['a_2']*np.cos(lensed_param_1['tilt_2']))/(lensed_param_1['mass_1'] + lensed_param_1['mass_2'])

    # log10 for (time/86400) and luminosity distance
    data_dict = dict(
        mass_1 = lensed_param_1['mass_1'][idx_snr],
        mass_2 = lensed_param_1['mass_2'][idx_snr],
        ra = lensed_param_1['ra'][idx_snr],
        sindec = np.cos(np.pi/2. - lensed_param_1['dec'][idx_snr]),
        costheta_jn = np.cos(lensed_param_1['theta_jn'][idx_snr]),
        chi_eff = chi_eff[idx_snr],
        log10_dl_1 = np.log10(lensed_param_1['effective_luminosity_distance'][idx_snr]),
        log10_dl_2 = np.log10(lensed_param_2['effective_luminosity_distance'][idx_snr]),
        log10_dt_12_days = np.log10(((lensed_param_2['effective_geocent_time'][idx_snr] - lensed_param_1['effective_geocent_time'][idx_snr])/86400.)),
    )

    return data_dict

def data_check_posterior(posterior1, posterior2):

    param_1 = {}
    param_2 = {}

    param_list = ['mass_1','mass_2','ra','dec', 'theta_jn','chi_eff','luminosity_distance', 'geocent_time']
    for key in param_list:
        try:
            param_1[key] = np.array(posterior1['posterior'][key])
        except:
            print(f"posterior1 should have the following keys: {param_list}")
            raise ValueError(f"{key} is not present in the posterior1")
        try:
            param_2[key] = np.array(posterior2['posterior'][key])
        except:
            print(f"posterior2 should have the following keys: {param_list}")
            raise ValueError(f"{key} is not present in the posterior2")


    # log10 for (time/86400) and luminosity distance
    data_dict1 = dict(
        mass_1 = param_1['mass_1'],
        mass_2 = param_1['mass_2'],
        ra = param_1['ra'],
        sindec = np.cos(np.pi/2. - param_1['dec']),
        costheta_jn = np.cos(param_1['theta_jn']),
        chi_eff = param_1['chi_eff'],
        log10_dl = np.log10(param_1['luminosity_distance']),
    )

    data_dict2 = dict(
        mass_1 = param_2['mass_1'],
        mass_2 = param_2['mass_2'],
        ra = param_2['ra'],
        sindec = np.cos(np.pi/2. - param_2['dec']),
        costheta_jn = np.cos(param_2['theta_jn']),
        chi_eff = param_2['chi_eff'],
        log10_dl = np.log10(param_2['luminosity_distance']),
    )

    t1 = np.median(param_1['geocent_time'])
    t2 = np.median(param_2['geocent_time'])
    if t1>t2:
        data_dict1, data_dict2 = data_dict2, data_dict1
    dt_12 = np.abs(np.median(param_2['geocent_time']) - np.median(param_1['geocent_time']))
    log10_dt_12_days = np.log10(dt_12/86400.)

    # for combine posterior
    len_ = len(data_dict1['mass_1'])
    if len(data_dict2['mass_1']) < len_:
        len_ = len(data_dict2['mass_1'])
    len_ = int(len_/2)-1

    data_combine = dict(
        mass_1 = np.concatenate([data_dict1['mass_1'][:len_], data_dict2['mass_1'][:len_]]),
        mass_2 = np.concatenate([data_dict1['mass_2'][:len_], data_dict2['mass_2'][:len_]]),
        ra = np.concatenate([data_dict1['ra'][:len_], data_dict2['ra'][:len_]]),
        sindec = np.concatenate([data_dict1['sindec'][:len_], data_dict2['sindec'][:len_]]),
        costheta_jn = np.concatenate([data_dict1['costheta_jn'][:len_], data_dict2['costheta_jn'][:len_]]),
        chi_eff = np.concatenate([data_dict1['chi_eff'][:len_], data_dict2['chi_eff'][:len_]]),
        log10_dl_1 = data_dict1['log10_dl'][:2*len_],
        log10_dl_2 = data_dict2['log10_dl'][:2*len_],
    )

    return data_dict1, data_dict2, data_combine, log10_dt_12_days

def data_check_astro_unlensed(data_dict, size=100000):

    unlensed_param = {}

    param_list = ['ra', 'dec', 'theta_jn', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'mass_1', 'mass_2', 'luminosity_distance', 'geocent_time']

    len_ = len(data_dict['geocent_time'])
    idx_all = np.arange(0, len_)
    # randomize idx_all
    np.random.shuffle(idx_all)

    for key in param_list:
        try:
            # randomize the data data_dict[key]
            unlensed_param[key] = np.array(data_dict[key])[idx_all]
        except:
            print(f"data_dict should have the following keys: {param_list}")
            raise ValueError(f"{key} is not present in the data_dict")

    #########################################
    # pairing indices for the unlensed data #
    #########################################
    # how many possible combination all
    num_combinations = comb(len_, 2, exact=True)
    # Known number of combinations
    C = size if size < num_combinations else num_combinations
    len_ = (1 + np.sqrt(1 + 8 * C)) / 2
    # Define the index array
    idx_all = np.arange(0, int(len_))
    # # randomize idx_all
    # np.random.shuffle(idx_all)

    # Generate all possible two-element combinations
    combination_array = np.array(list(itertools.combinations(idx_all, 2)))
    idx1 = combination_array[:,0]
    idx2 = combination_array[:,1]

    geocent_time1 = unlensed_param['geocent_time'][idx1]
    geocent_time2 = unlensed_param['geocent_time'][idx2]
    # (geocent_time2 - geocent_time1) > 0, otherwise swap. 1 is earlier than 2
    mask = geocent_time2 <= geocent_time1  
    idx1[mask], idx2[mask] = idx2[mask], idx1[mask]

    # with effective spin
    # Note: chi_eff for image 1 and 2 is the same
    chi_eff = (unlensed_param['a_1']*np.cos(unlensed_param['tilt_1']) + unlensed_param['a_2']*np.cos(unlensed_param['tilt_2']))/(unlensed_param['mass_1'] + unlensed_param['mass_2'])
    unlensed_param['chi_eff'] = chi_eff
    unlensed_param['sindec'] = np.cos(np.pi/2. - unlensed_param['dec'])
    unlensed_param['costheta_jn'] = np.cos(unlensed_param['theta_jn'])
    unlensed_param['log10_dl'] = np.log10(unlensed_param['luminosity_distance'])

    param_list = ['mass_1','mass_2','ra','sindec', 'costheta_jn','chi_eff','log10_dl']

    data_dict1 = {}
    data_dict2 = {}
    for key in param_list:
        data_dict1[key] = unlensed_param[key][idx1]
        data_dict2[key] = unlensed_param[key][idx2]

    data_dict2['log10_dt_12_days'] = np.log10((unlensed_param['geocent_time'][idx2]-unlensed_param['geocent_time'][idx1])/86400.)

    return data_dict1, data_dict2

def initialize_bilby_priors(size=100000):
    """
    Initialize bilby priors for the following parameters:
    mass_1, mass_2, ra, dec, theta_jn, a_1, a_2, tilt_1, tilt_2, luminosity_distance, geocent_time
    """

    mass_1 = bilby.core.prior.base.Constraint(minimum=5, maximum=140)
    mass_2 = bilby.core.prior.base.Constraint(minimum=5, maximum=140)
    chirp_mass = bilby.core.prior.Uniform(name='chirp_mass', minimum=5, maximum=140)
    mass_ratio = bilby.core.prior.Uniform(name='mass_ratio', minimum=0.125, maximum=1)
    a_1 = bilby.core.prior.Uniform(name='a_1', minimum=0, maximum=0.99)
    a_2 = bilby.core.prior.Uniform(name='a_2', minimum=0, maximum=0.99)
    tilt_1 = bilby.core.prior.Sine(name='tilt_1')
    tilt_2 = bilby.core.prior.Sine(name='tilt_2')
    phi_12 = bilby.core.prior.Uniform(name='phi_12', minimum=0, maximum=2*np.pi, boundary='periodic')
    phi_jl = bilby.core.prior.Uniform(name='phi_jl', minimum=0, maximum=2*np.pi, boundary='periodic')
    luminosity_distance = bilby.core.prior.PowerLaw(alpha=2, name='luminosity_distance', minimum=50, maximum=15000, unit='Mpc', latex_label='$d_L$')
    dec = bilby.core.prior.Cosine(name='dec')
    ra = bilby.core.prior.Uniform(name='ra', minimum=0, maximum=2*np.pi, boundary='periodic')
    theta_jn = bilby.core.prior.Sine(name='theta_jn')
    psi = bilby.core.prior.Uniform(name='psi', minimum=0, maximum=np.pi, boundary='periodic')
    phase = bilby.core.prior.Uniform(name='phase', minimum=0, maximum=2*np.pi, boundary='periodic')

    data_dict = dict(
        chirp_mass = chirp_mass.sample(size=size),
        mass_ratio = mass_ratio.sample(size=size),
        a_1 = a_1.sample(size=size),
        a_2 = a_2.sample(size=size),
        tilt_1 = tilt_1.sample(size=size),
        tilt_2 = tilt_2.sample(size=size),
        phi_12 = phi_12.sample(size=size),
        phi_jl = phi_jl.sample(size=size),
        luminosity_distance = luminosity_distance.sample(size=size),
        dec = dec.sample(size=size),
        ra = ra.sample(size=size),
        theta_jn = theta_jn.sample(size=size),
        psi = psi.sample(size=size),
        phase = phase.sample(size=size),
    )

    data_dict['mass_1'] = (data_dict['chirp_mass']*(1+data_dict['mass_ratio'])**(3/5))/data_dict['mass_ratio']**(6/5)
    data_dict['mass_2'] = data_dict['chirp_mass']*(1+data_dict['mass_ratio'])**(2/5)/data_dict['mass_ratio']**(3/5)

    return data_dict

def data_check_pe_prior(data_dict):

    param = {}

    param_list = ['ra', 'dec', 'theta_jn', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'mass_1', 'mass_2', 'luminosity_distance']
    for key in param_list:
        try:
            param[key] = np.array(data_dict[key])
        except:
            print(f"data_dict should have the following keys: {param_list}")
            raise ValueError(f"{key} is not present in the data_dict")

    # with effective spin
    # Note: chi_eff for image 1 and 2 is the same
    chi_eff = (param['a_1']*np.cos(param['tilt_1']) + param['a_2']*np.cos(param['tilt_2']))/(param['mass_1'] + param['mass_2'])

    # log10 for (time/86400) and luminosity distance
    data_dict = dict(
        mass_1 = param['mass_1'],
        mass_2 = param['mass_2'],
        ra = param['ra'],
        sindec = np.cos(np.pi/2. - param['dec']),
        costheta_jn = np.cos(param['theta_jn']),
        chi_eff = chi_eff,
        log10_dl = np.log10(param['luminosity_distance']),
    )

    return data_dict

# prior-dict={{mass_1 = Constraint(name='mass_1', minimum=5, maximum=140), mass_2 = Constraint(name='mass_2', minimum=5, maximum=140), chirp_mass =  Uniform(name='chirp_mass', minimum={mc_min}, maximum={mc_max}, latex_label='$M_{{c}}$'), mass_ratio =  Uniform(name='mass_ratio', minimum=0.125, maximum=1, latex_label='$q$'), a_1 = Uniform(name='a_1', minimum=0, maximum=0.99), a_2 = Uniform(name='a_2', minimum=0, maximum=0.99), tilt_1 = Sine(name='tilt_1'), tilt_2 = Sine(name='tilt_2'), phi_12 = Uniform(name='phi_12', minimum=0, maximum=2 * np.pi, boundary='periodic'), phi_jl = Uniform(name='phi_jl', minimum=0, maximum=2 * np.pi, boundary='periodic'), luminosity_distance = PowerLaw(alpha=2, name='luminosity_distance', minimum=50, maximum=15000, unit='Mpc', latex_label='$d_L$'),  dec = Cosine(name='dec'), ra = Uniform(name='ra', minimum=0, maximum=2 * np.pi, boundary='periodic'), theta_jn = Sine(name='theta_jn'), psi =  Uniform(name='psi', minimum=0, maximum=np.pi, boundary='periodic'), phase =  Uniform(name='phase', minimum=0, maximum=2 * np.pi, boundary='periodic'),}}

