import os
import numpy as np
from .utils import get_param_from_json, save_json, load_json
from .scaler import (
    uniform_to_gaussian,
    gaussian_to_uniform,
    sine_to_gaussian,
    gaussian_to_sine,
    cosine_to_gaussian,
    gaussian_to_cosine,
    scale_to_range,
    unscale_to_range,
)
from sklearn.mixture import BayesianGaussianMixture

import h5py
import pandas as pd
import json
import pickle
from sklearn.preprocessing import StandardScaler

from .utils import (
    load_json,
    save_json,
    load_pickle,
    save_pickle,
    get_dict_or_file,
    dir_check,
    load_data_from_module,
    save_min_max,
    data_check_astro_lensed_dpgmm,
    data_check_astro_lensed_sky_dpgmm,
    data_check_astro_unlensed_dpgmm,
    data_check_astro_unlensed_time_dpgmm,
)


class ModelGenerator:
    def __init__(
        self,
        model_name,
        data_dict=None,  # if provided new model will be generated
        pobs_directory="./pobs_data",
        create_new=False,
        **kwargs,
    ):

        self.model_name = model_name
        self.pobs_directory = dir_check(pobs_directory)
        self.kwargs = kwargs
        self.model = None
        self.scaler = None
        self.min_max = None
        self.create_new = create_new

        data_dict = self.init_data_dict(data_dict)
        if data_dict is not None:
            self.scaler = None
            self.min_max = None
            self.create_model(data_dict)

        self.data_dict = data_dict

    def init_data_dict(self, data_dict):

        self.init_meta_dict()

        if data_dict is not None:
            data_dict = get_dict_or_file(data_dict)
            return data_dict
        else:
            model_path = self.meta_dict["model_path"]
            scaler_path = self.meta_dict["scaler_path"]
            min_max_path = self.meta_dict["min_max_path"]
            # check if the paths exists
            if (
                os.path.exists(model_path)
                and os.path.exists(scaler_path)
                and os.path.exists(min_max_path)
                and not self.create_new
            ):
                print(f"Loading model, scaler and min_max from {self.pobs_directory}")
                self.model = load_pickle(model_path)
                self.scaler = load_pickle(scaler_path)
                self.min_max = load_json(min_max_path)
            else:
                if self.model_name == "astro_lensed":
                    print("astro_lensed is None")
                    print("getting default astro_lensed data_dict from pobs module")
                    data_dict = load_data_from_module(
                        "pobs", "data", "n_lensed_detectable_bbh_po_spin.json"
                    )
                    data_dict = data_check_astro_lensed_dpgmm(data_dict)

                elif self.model_name == "astro_lensed_sky":
                    print("astro_lensed_sky is None")
                    print("getting default astro_lensed data_dict from pobs module")
                    data_dict = load_data_from_module(
                        "pobs", "data", "n_lensed_detectable_bbh_po_spin.json"
                    )
                    data_dict = data_check_astro_lensed_sky_dpgmm(data_dict)
                elif self.model_name == "astro_unlensed":
                    print("astro_unlensed is None")
                    print("getting default astro_unlensed data_dict from pobs module")
                    data_dict = load_data_from_module(
                        "pobs", "data", "n_unlensed_detectable_bbh_po_spin.json"
                    )
                    data_dict = data_check_astro_unlensed_dpgmm(data_dict)
                elif self.model_name == "astro_unlensed_time":
                    print("astro_unlensed is None")
                    print("getting default astro_unlensed data_dict from pobs module")
                    data_dict = load_data_from_module(
                        "pobs", "data", "n_unlensed_detectable_bbh_po_spin.json"
                    )
                    data_dict = data_check_astro_unlensed_time_dpgmm(data_dict)


                return data_dict

    def mass_scaler(self, data, min_data, max_data, which_type="forward"):

        if which_type == "forward":
            data_scaled = scale_to_range(data, min_data, max_data)
            data_scaled = sine_to_gaussian(data_scaled)
        elif which_type == "backward":
            data_scaled = gaussian_to_sine(data)
            data_scaled = unscale_to_range(data_scaled, min_data, max_data)
        else:
            raise ValueError("which_type should be either forward or backward")
        return data_scaled

    def inclination_scaler(self, data, which_type='forward', **kwargs):

        if which_type == 'forward':
            data_scaled = sine_to_gaussian(data)
        elif which_type == 'backward':
            data_scaled = gaussian_to_sine(data)
        else:
            raise ValueError("which_type should be either forward or backward")

        return data_scaled

    def ra_scaler(self, data, which_type='forward', **kwargs):

        mu=0
        sigma=1
        upper_bound=6.29
        lower_bound=0

        if which_type == 'forward':
            data_scaled = uniform_to_gaussian(data, mu=mu, sigma=sigma, upper_bound=upper_bound, lower_bound=lower_bound)
        elif which_type == 'backward':
            data_scaled = gaussian_to_uniform(data, mu=mu, sigma=sigma, upper_bound=upper_bound, lower_bound=lower_bound)
        else:
            raise ValueError("which_type should be either forward or backward")

        return data_scaled

    def scaling(self, data_dict=None, data_list=None, which_type="forward"):

        if data_dict is None:
            keys_ = self.meta_dict["scaling_param"].keys()
            # column should correspond to each parameter type
            data_dict = {key: np.array(data_list[:,i]) for i, key in enumerate(keys_)}
        elif data_list is None:
            data_list = np.array([data_dict[key] for key in data_dict.keys()]).T
        else:
            raise ValueError("Either data_dict or data_list should be provided")

        # check scaler
        if self.scaler is None:
            scaler = StandardScaler()
            scaler.fit(data_list)
            self.scaler = scaler
        else:
            scaler = self.scaler

        if self.min_max is None:
            filename = self.meta_dict["min_max_path"]
            min_max = save_min_max(filename, data_dict)
            self.min_max = min_max
        else:
            min_max = self.min_max

        result_dict = {}
        result_list = []
        # this for extra scaling besides standard scaling
        scaling_param = self.meta_dict["scaling_param"]

        if which_type == "backward":
            # standard scaling
            data_list = scaler.inverse_transform(data_list)
            data_dict = {key: data_list[:, i] for i, key in enumerate(data_dict.keys())}

        for key, value in data_dict.items():
            # print(type(scaling_param[key]))
            if scaling_param[key] is not None:
                result_dict[key] = getattr(self, scaling_param[key])(
                    data = value,
                    min_data = min_max[key]["min_data"],
                    max_data = min_max[key]["max_data"],
                    which_type=which_type,
                )
            else:
                result_dict[key] = value
            result_list.append(result_dict[key])

        result_list = np.array(result_list).T
        if which_type == "forward":
            # standard scaling
            result_list = scaler.transform(result_list)
            result_dict = {key: result_list[:, i] for i, key in enumerate(data_dict.keys())}

        return result_dict, result_list

    def create_model(self, data_dict):

        # each column corresponds to each parameter type
        _, scaled_data = self.scaling(data_dict=data_dict, which_type="forward")
        
        self.data_dict_x = data_dict


        # use self.kwargs
        args = dict(
            n_components=20,
            covariance_type="full",
            weight_concentration_prior=1e-2,
            max_iter=1000,
            random_state=0,
        )
        args.update(self.kwargs)

        # self.scaled_data = scaled_data

        self.model = BayesianGaussianMixture(
            n_components=args["n_components"],
            covariance_type=args["covariance_type"],
            weight_concentration_prior=args["weight_concentration_prior"],
            max_iter=args["max_iter"],
            random_state=args["random_state"],
        )

        self.model.fit(scaled_data)

        # save the model, scaler and min_max
        save_pickle(self.meta_dict["model_path"], self.model)
        save_pickle(self.meta_dict["scaler_path"], self.scaler)
        save_json(self.meta_dict["min_max_path"], self.min_max)

        return None

    def resample(self, size=10000):

        model = self.model
        min_max = self.min_max
        label_list = list(self.meta_dict["scaling_param"].keys())

        result_dict = {}
        for key in label_list:
            result_dict[key] = np.array([])

        # loop until we get the desired size
        batch_size = size
        while True:
            # generate random data
            data = model.sample(batch_size)[0]
            # unscale the data
            # each column corresponds to each parameter type
            _, unsclaed_data = self.scaling(data_list=data, which_type="backward")

            # making sure the data is within the range
            j = 0
            idx = np.ones(unsclaed_data.shape[0], dtype=bool)
            # keys of min_max should be the same as label_list
            for key in label_list:
                min_data = min_max[key]["min_data"]
                max_data = min_max[key]["max_data"]
                data_ = unsclaed_data[:, j]
                idx &= (data_ < min_data) | (data_ > max_data)
                j += 1
            
            # fill up the result in each loop
            for i, key in enumerate(label_list):
                result_dict[key] = np.concatenate([result_dict[key], unsclaed_data[:, i][~idx]])

            final_size = result_dict[label_list[0]].shape[0]
            batch_size = (size - final_size) + 1
            if final_size >= size:
                for key, value in result_dict.items():
                    result_dict[key] = value[:size]
                break

        return result_dict

    # dont forget about prior reweighting function

    def init_meta_dict(self):

        model_path = f"{self.pobs_directory}model_path_{self.model_name}.pkl"
        scaler_path = f"{self.pobs_directory}scaler_path_{self.model_name}.pkl"
        min_max_path = f"{self.pobs_directory}min_max_path_{self.model_name}.json"

        meta_dict_all = {
            "astro_lensed": {
                "scaling_param": {
                    "mass_1": "mass_scaler",
                    "mass_2": "mass_scaler",
                    "theta_jn": "inclination_scaler",
                    "log10_dl_1": None,
                    "log10_dl_2": None,
                    "log10_dt_12_days": None,
                },
            },
            "astro_lensed_sky": {
                "scaling_param": {
                    "ra": "ra_scaler",
                    "dec": None,
                },
            },
            "astro_unlensed": {
                "scaling_param": {
                    "mass_1": "mass_scaler",
                    "mass_2": "mass_scaler",
                    "theta_jn": "inclination_scaler",
                    "log10_dl": None,
                },
            },
            "astro_unlensed_time": {
                "scaling_param": {
                    "log10_dt_12_days": None,
                },
            },
        }

        self.meta_dict = meta_dict_all[self.model_name]
        self.meta_dict["model_path"] = model_path
        self.meta_dict["scaler_path"] = scaler_path
        self.meta_dict["min_max_path"] = min_max_path