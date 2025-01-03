import os
import numpy as np
from .utils import save_json, load_json
from .scaler import (
    uniform_to_gaussian,
    gaussian_to_uniform,
    sine_to_gaussian,
    gaussian_to_sine,
    scale_to_range,
    unscale_to_range,
)
from sklearn.mixture import BayesianGaussianMixture
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
    data_check_astro_lensed,
    data_check_astro_lensed_sky,
    data_check_astro_unlensed,
    data_check_astro_unlensed_time,
    data_check_astro_unlensed_sky,
)

from .meta_dict import meta_dict as meta_dict_all


class ModelGenerator:
    def __init__(
        self,
        model_name="posterior1",
        model_type="posterior1",
        data_dict=None,  # if provided new model will be generated
        pobs_directory="./pobs_data",
        # path_dict = None,  # don't allow this for now
        kde_model_type="dpgmm",
        create_new=False,
        num_images=2,
        kde_args=None,
    ):

        self.model_name = model_name
        self.model_type = model_type
        self.kde_model_type = kde_model_type
        self.num_images = num_images
        self.pobs_directory = dir_check(
            pobs_directory
        )  # check if the directory exists, also add '/' at the end
        if kde_args is None:
            self.kde_args = {}
        else:
            self.kde_args = kde_args
        self.model = None
        self.scaler = None
        self.min_max = None
        self.create_new = create_new

        data_dict = self.init_data_dict(data_dict)
        # now I have the data_dict and meta_dict
        # data_dict is None if the model, scaler and min_max exists

        if (
            data_dict is not None
        ):  # this means doesn't model, scaler and min_max exist yet
            self.scaler = None
            self.min_max = None
            self.create_model(data_dict)

        self.data_dict = data_dict

    def init_data_dict(self, data_dict):

        # get all the paths
        self.init_meta_dict()
        # check new data is provided
        # if not, check if the model, scaler and min_max exists
        # if not, get the default data from pobs module
        # the data should contain appropriate keys and values
        if data_dict is not None:
            data_dict = get_dict_or_file(data_dict)
            return data_dict
        else:
            model_path = self.meta_dict["model_path"]
            scaler_path = self.meta_dict["scaler_path"]
            min_max_path = self.meta_dict["min_max_path"]
            renorm_const_path = self.meta_dict["renorm_const_path"]
            # check if the paths exists
            if (
                os.path.exists(model_path)
                and os.path.exists(scaler_path)
                and os.path.exists(min_max_path)
                and not self.create_new
            ):
                print(f"Loading model from {self.meta_dict['model_path']}")
                self.model = load_pickle(model_path)
                print(f"Loading scaler from {self.meta_dict['scaler_path']}")
                self.scaler = load_pickle(scaler_path)
                print(f"Loading min_max from {self.meta_dict['min_max_path']}")
                self.min_max = load_json(min_max_path)
                print(
                    f"Loading renormalization constant from {self.meta_dict['renorm_const_path']}"
                )
                self.renorm_const = load_json(renorm_const_path)
            else:
                if self.model_type == "astro_lensed":
                    print(f"data_dict for {self.model_type} is not provided")
                    print("getting default astro_lensed data_dict from pobs module")
                    data_dict = load_data_from_module(
                        "pobs", "data", "n_lensed_detectable_bbh_po_spin.json"
                    )
                    data_dict = data_check_astro_lensed(data_dict)

                elif self.model_type == "astro_lensed_sky":
                    print(f"data_dict for {self.model_type} is not provided")
                    print("getting default astro_lensed data_dict from pobs module")
                    data_dict = load_data_from_module(
                        "pobs", "data", "n_lensed_detectable_bbh_po_spin.json"
                    )
                    data_dict = data_check_astro_lensed_sky(data_dict)
                elif self.model_type == "astro_unlensed":
                    print(f"data_dict for {self.model_type} is not provided")
                    print("getting default astro_unlensed data_dict from pobs module")
                    data_dict = load_data_from_module(
                        "pobs", "data", "n_unlensed_detectable_bbh_po_spin.json"
                    )
                    data_dict = data_check_astro_unlensed(data_dict)
                elif self.model_type == "astro_unlensed_time":
                    print(f"data_dict for {self.model_type} is not provided")
                    print("getting default astro_unlensed data_dict from pobs module")
                    data_dict = load_data_from_module(
                        "pobs", "data", "n_unlensed_detectable_bbh_po_spin.json"
                    )
                    data_dict = data_check_astro_unlensed_time(data_dict)
                elif self.model_type == "astro_unlensed_sky":
                    print(f"data_dict for {self.model_type} is not provided")
                    print("getting default astro_unlensed data_dict from pobs module")
                    data_dict = load_data_from_module(
                        "pobs", "data", "n_unlensed_detectable_bbh_po_spin.json"
                    )
                    data_dict = data_check_astro_unlensed_sky(data_dict)

                else:
                    raise ValueError(
                        "data_dict and default model type not prvided.\n Model_type should be one of the following: astro_lensed, astro_lensed_sky, astro_unlensed, astro_unlensed_time, astro_unlensed_sky"
                    )

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

    def inclination_scaler(self, data, which_type="forward", **kwargs):

        if which_type == "forward":
            data_scaled = sine_to_gaussian(data)
        elif which_type == "backward":
            data_scaled = gaussian_to_sine(data)
        else:
            raise ValueError("which_type should be either forward or backward")

        return data_scaled

    def distance_scaler(self, data, which_type="forward", **kwargs):

        if which_type == "forward":
            data_scaled = np.log10(data)
        elif which_type == "backward":
            data_scaled = 10**data
        else:
            raise ValueError("which_type should be either forward or backward")

        return data_scaled

    def time_scaler(self, data, which_type="forward", **kwargs):

        if which_type == "forward":
            data_scaled = np.log10(data / 86400)
        elif which_type == "backward":
            data_scaled = 10 ** (data) * 86400
        else:
            raise ValueError("which_type should be either forward or backward")

        return data_scaled

    def ra_scaler(self, data, which_type="forward", **kwargs):

        mu = 0
        sigma = 1
        upper_bound = 6.29
        lower_bound = 0

        if which_type == "forward":
            data_scaled = uniform_to_gaussian(
                data,
                mu=mu,
                sigma=sigma,
                upper_bound=upper_bound,
                lower_bound=lower_bound,
            )
        elif which_type == "backward":
            data_scaled = gaussian_to_uniform(
                data,
                mu=mu,
                sigma=sigma,
                upper_bound=upper_bound,
                lower_bound=lower_bound,
            )
        else:
            raise ValueError("which_type should be either forward or backward")

        return data_scaled

    def scaling(self, data_dict=None, data_list=None, which_type="forward"):

        if data_dict is None:
            keys_ = self.meta_dict["scaling_param"].keys()
            # column should correspond to each parameter type
            data_dict = {key: np.array(data_list[:, i]) for i, key in enumerate(keys_)}
        elif data_list is None:
            data_list = np.array([data_dict[key] for key in data_dict.keys()]).T
        else:
            raise ValueError("Either data_dict or data_list should be provided")

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

        # Backward scaling
        if which_type == "backward":
            # standard scaling
            data_list = self.scaler.inverse_transform(data_list)
            data_dict = {key: data_list[:, i] for i, key in enumerate(data_dict.keys())}

        for key, value in data_dict.items():
            if scaling_param[key] is not None:
                result_dict[key] = getattr(self, scaling_param[key])(
                    data=value,
                    min_data=min_max[key]["min_data"],
                    max_data=min_max[key]["max_data"],
                    which_type=which_type,
                )
            else:
                result_dict[key] = value
            result_list.append(result_dict[key])

        result_list = np.array(result_list).T

        # check scaler
        if self.scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(result_list)

        # Forward scaling
        if which_type == "forward":
            # standard scaling
            result_list = self.scaler.transform(result_list)
            result_dict = {
                key: result_list[:, i] for i, key in enumerate(data_dict.keys())
            }

        return result_dict, result_list

    def create_model(self, data_dict, renormalization=True):

        # each column corresponds to each parameter type
        # data will be scale according to the meta_dict, dictated by the model_name
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
        args.update(self.kde_args)

        self.model = BayesianGaussianMixture(
            n_components=args["n_components"],
            covariance_type=args["covariance_type"],
            weight_concentration_prior=args["weight_concentration_prior"],
            max_iter=args["max_iter"],
            random_state=args["random_state"],
        )

        # fit the model
        print(f"Fitting the model for {self.model_name}. This may take a while...")
        self.model.fit(scaled_data)

        # save the model, scaler and min_max
        print(f"Saving model to {self.meta_dict['model_path']} ")
        save_pickle(self.meta_dict["model_path"], self.model)
        print(f"Saving scaler to {self.meta_dict['scaler_path']}")
        save_pickle(self.meta_dict["scaler_path"], self.scaler)
        print(f"Saving min_max to {self.meta_dict['min_max_path']}")
        save_json(self.meta_dict["min_max_path"], self.min_max)

        # find renormalization constant
        if renormalization:
            self.renormalization()

    def renormalization(self, batch_size=1000000):

        min_max_list = []
        for _, value in self.min_max.items():
            min_max_list.append([value["min_data"], value["max_data"]])

        min_max_list = np.array(min_max_list).T
        _, min_max_list = self.scaling(data_list=min_max_list, which_type="forward")
        num_ = min_max_list.shape[1]

        data = []
        norm_ = 1.0
        for i in range(num_):
            min_data = min_max_list[:, i].min()
            max_data = min_max_list[:, i].max()
            norm_ *= max_data - min_data

            data.append(np.random.uniform(min_data, max_data, batch_size))

        data = np.array(data).T
        pdf = np.exp(self.model.score_samples(data))

        self.renorm_const = np.mean(pdf) * norm_
        print(
            f"Saving renormalization constant to {self.meta_dict['renorm_const_path']}"
        )
        save_json(self.meta_dict["renorm_const_path"], self.renorm_const)

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
                idx = idx & (data_ > min_data) & (data_ < max_data)
                j += 1

            # fill up the result in each loop
            for i, key in enumerate(label_list):
                result_dict[key] = np.concatenate(
                    [result_dict[key], unsclaed_data[:, i][idx]]
                )

            final_size = result_dict[label_list[0]].shape[0]
            batch_size = (size - final_size) + 1
            if final_size >= size:
                for key, value in result_dict.items():
                    result_dict[key] = value[:size]
                break

        return result_dict

    def pdf(self, data_dict, renormalization=True):

        _, scaled_data = self.scaling(data_dict, which_type="forward")

        # pdf
        pdf = np.exp(self.model.score_samples(scaled_data))
        if renormalization:
            pdf = pdf / self.renorm_const

        # # get min and max data
        # for key, value in self.min_max.items():
        #     min_data = value['min_data']
        #     max_data = value['max_data']
        #     idx = (data_dict[key]<min_data) | (data_dict[key]>max_data)
        #     pdf[idx] = 0.0

        return pdf

    def init_meta_dict(self):

        self.meta_dict = meta_dict_all(self.num_images)[self.model_type]
        self.meta_dict["model_path"] = (
            f"{self.pobs_directory}model_{self.model_name}.pkl"
        )
        self.meta_dict["scaler_path"] = (
            f"{self.pobs_directory}scaler_{self.model_name}.pkl"
        )
        self.meta_dict["min_max_path"] = (
            f"{self.pobs_directory}min_max_{self.model_name}.json"
        )
        self.meta_dict["renorm_const_path"] = (
            f"{self.pobs_directory}renorm_const_{self.model_name}.json"
        )
