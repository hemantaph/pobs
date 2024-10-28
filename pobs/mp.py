import numpy as np
from .utils import (
    load_json,
    load_pickle,
)


def bayes_factor_multiprocessing(input_arguments):

    sample_size_original = input_arguments[0]  
    sample_size = sample_size_original 
    log_dt_12_days = input_arguments[1]
    astro_lensed = input_arguments[2]
    astro_unlensed1 = input_arguments[3]
    astro_unlensed2 = input_arguments[4]
    pe_prior = input_arguments[5]
    posterior1 = input_arguments[6]
    posterior2 = input_arguments[7]
    posterior_combine = input_arguments[8]

    #############
    # numerator #
    #############
    result_size = 0 
    numerator_array = np.array([])
    while result_size < sample_size_original:
        # print('numerator', result_size)
        # print('sample_size', sample_size)
        # resample
        data_posterior_combine = posterior_combine.resample(sample_size)

        # posterior1
        data_dict1 = data_posterior_combine.copy()
        data_dict1['log10_dl'] = data_posterior_combine['log10_dl_1']
        del data_dict1['log10_dl_1'], data_dict1['log10_dl_2']
        # posterior2
        data_dict2 = data_dict1.copy()
        data_dict2['log10_dl'] = data_posterior_combine['log10_dl_2']
        # atrso_lensed
        data_dict3 = data_posterior_combine.copy()
        data_dict3['log10_dt_12_days'] = log_dt_12_days*np.ones(sample_size)
        # pe_prior
        # data_dict4 = data_dict1.copy()
        # data_dict5 = data_dict2.copy()
        # posterior_combine
        # data_dict6 = data_posterior_combine.copy()

        pdf1 = posterior1.pdf(data_dict1)
        pdf2 = posterior2.pdf(data_dict2)
        pdf3 = astro_lensed.pdf(data_dict3)
        pdf4 = pe_prior.pdf(data_dict1)
        pdf5 = pe_prior.pdf(data_dict2)
        pdf6 = posterior_combine.pdf(data_posterior_combine)
        pdf123 = pdf1 * pdf2 * pdf3
        pdf456 = pdf4 * pdf5 * pdf6

        # ignore the zero values
        # note that buffer_array can have zero values if pdf123<<pdf456
        idx = pdf456!=0
        idx &= pdf1!=0
        idx &= pdf2!=0
        idx &= pdf3!=0
        buffer_array = pdf123[idx] / pdf456[idx]

        # check inf
        idx = buffer_array!=np.inf
        idx &= buffer_array!=-np.inf
        # check for nan
        idx &= np.isnan(buffer_array)==False
        idx &= buffer_array!=0
        buffer_array = buffer_array[idx]
        
        if len(buffer_array) != 0:
            # append
            numerator_array = np.concatenate((numerator_array, buffer_array))
            result_size = len(numerator_array)
            sample_size = sample_size_original - result_size

    ###############
    # denominator #
    ###############
    result_size = 0
    sample_size = sample_size_original
    denominator_array = np.array([])
    while result_size < sample_size_original:
        # print('denominator', result_size)
        # print('sample_size', sample_size)
        # resample
        data_posterior1 = posterior1.resample(sample_size)
        data_posterior2 = posterior2.resample(sample_size)

        # astro_unlensed1
        # data_dict1 = data_posterior1.copy()
        # astro_unlensed2
        data_dict2 = data_posterior2.copy()
        data_dict2['log10_dt_12_days'] = log_dt_12_days*np.ones(sample_size)
        # pe_prior
        # data_dict3 = data_posterior1.copy()
        # data_dict4 = data_posterior2.copy()

        pdf1 = astro_unlensed1.pdf(data_posterior1)
        pdf2 = astro_unlensed2.pdf(data_dict2)
        pdf3 = pe_prior.pdf(data_posterior1)
        pdf4 = pe_prior.pdf(data_posterior2)
        pdf12 = pdf1 * pdf2
        pdf34 = pdf3 * pdf4

        # ignore the zero values
        # note that buffer_array can have zero values if pdf12<<pdf34
        idx = pdf34!=0
        idx &= pdf1!=0
        idx &= pdf2!=0
        buffer_array = pdf12[idx] / pdf34[idx]

        # check inf
        idx = buffer_array!=np.inf
        idx &= buffer_array!=-np.inf
        # check for nan
        idx &= np.isnan(buffer_array)==False
        idx &= buffer_array!=0
        buffer_array = buffer_array[idx]
        
        if len(buffer_array) != 0:
            # append
            denominator_array = np.concatenate((denominator_array, buffer_array))
            result_size = len(denominator_array)
            sample_size = sample_size_original - result_size
    

    return list(numerator_array), list(denominator_array)

# with Pool(processes=npool) as pool:
#             bf = list(pool.map(bayes_factor_multiprocessing, input_arguments))