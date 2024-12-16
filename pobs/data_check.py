import numpy as np


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

    # log10 for (time/86400) and luminosity distance
    data_dict = dict(
        mass_1 = lensed_param_1['mass_1'][idx_snr],
        mass_2 = lensed_param_1['mass_2'][idx_snr],
        ra = lensed_param_1['ra'][idx_snr],
        dec = lensed_param_1['dec'][idx_snr],
        theta_jn = lensed_param_1['theta_jn'][idx_snr],
        log10_dl_1 = np.log10(lensed_param_1['effective_luminosity_distance'][idx_snr]),
        log10_dl_2 = np.log10(lensed_param_2['effective_luminosity_distance'][idx_snr]),
        log10_dt_12_days = np.log10(((lensed_param_2['effective_geocent_time'][idx_snr] - lensed_param_1['effective_geocent_time'][idx_snr])/86400.)),
    )

    return data_dict