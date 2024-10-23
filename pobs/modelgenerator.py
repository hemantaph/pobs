import os
import numpy as np
from ler.utils import get_param_from_json, save_json, load_json
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import jax.numpy as jnp
import jax.random as random
from jax.scipy.stats import gaussian_kde as jax_gaussian_kde
import corner
import matplotlib.lines as mlines
import h5py 
import pandas as pd
import json
import pickle
from sklearn.preprocessing import StandardScaler

class ModelGenerator():
    def __init__(self, 
        model_name, 
        data_dict, 
        param_list=None, 
        pobs_directory="./pobs_data",
        model_type="gaussian_kde",
        path_dict = None,
        batch_size=100000,
        create_new=True,
    ):
        self.model_name = model_name
        self.batch_size = batch_size

        if data_dict is None:
            raise ValueError("data_dict cannot be None")
        elif isinstance(data_dict, str):
            try:
                data_dict = get_param_from_json(data_dict)
            except:
                raise ValueError("data_dict is not a valid json file")

        if param_list is not None:
            self.param_list = param_list
            for keys in param_list.keys():
                del data_dict[keys]
        else:
            self.param_list = list(data_dict.keys())
        self.data_dict = data_dict

        self.pobs_directory = pobs_directory
        if not os.path.exists(self.pobs_directory):
            os.makedirs(self.pobs_directory) 

        self.model_path = f"{self.pobs_directory}/model_path_{self.model_name}.pkl"
        self.scaler_path = f"{self.pobs_directory}/scaler_path_{self.model_name}.pkl"
        self.min_max_path = f"{self.pobs_directory}/min_max_path_{self.model_name}.pkl"

        self.model_type = model_type
        if path_dict is None:
            self.path_dict = {
                "model_path": self.model_path,
                "scaler_path": self.scaler_path,
                "min_max_path": self.min_max_path,
                "label_list": list(self.param_list),
                #"bandwidth_factor": self.bandwidth_factor,
            }
        else:
            if isinstance(path_dict, str):
                try:
                    self.path_dict = load_json(path_dict)
                except:
                    raise ValueError("path_dict is not a valid json file")
            elif isinstance(path_dict, dict):
                self.path_dict = path_dict
            else:
                raise ValueError("path_dict should be a string to json path or a dictionary")
    

        # save path dict
        save_json(f"{self.pobs_directory}/path_dict_{self.model_name}.json", self.path_dict)
        
        # save the min and max of the data_dict
        self.save_min_max()

        if create_new:
            self.create_model()


    def save_min_max(self, data_dict=None, filename=None):

        if data_dict is None:
            data_dict = self.data_dict.copy()
        if filename is None:
            filename = self.min_max_path

        min_max = {}
        for key, value in data_dict.items():
            min_max[key] = dict(
                min_data = np.min(value), 
                max_data = np.max(value),
            )
        save_json(filename, min_max)

    def get_model_scaler_minmax(self, model_path=None, scaler_path=None, min_max_path=None):

        if model_path is None:
            model_path = self.model_path
        if scaler_path is None:
            scaler_path = self.scaler_path
        if min_max_path is None:
            min_max_path = self.min_max_path
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        if min_max_path is not None:
            min_max = load_json(min_max_path)

        return model, scaler, min_max

    def feature_scaling(self, data_dict=None, save=True, filename=None, call=False):
        
        if data_dict is None:
            data_dict = self.data_dict.copy()
        if filename is None:
            filename = self.scaler_path

        data = np.array(list(data_dict.values())).T

        if not call:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
        else:
            with open(filename, 'rb') as f:
                scaler = pickle.load(f)
            scaled_data = scaler.transform(data)

        if save:
            with open(filename, 'wb') as f:
                pickle.dump(scaler, f)

        return scaled_data, scaler

    def create_model(self, data_dict=None, save=True, filename=None):

        if data_dict is None:
            data_dict = self.data_dict.copy()
        if filename is None:
            filename = self.model_path

        scaled_data, _ = self.feature_scaling(data_dict, save=save)
        if self.model_type == "gaussian_kde":
            kde = gaussian_kde(scaled_data.T, bw_method='scott')
        elif self.model_type == "jax_gaussian_kde":
            scaled_data = jnp.array(scaled_data)
            kde = jax_gaussian_kde(scaled_data.T, bw_method='scott')
        else:
            raise ValueError("model_type should be either 'gaussian_kde' or 'jax_gaussian_kde'")
        # jax.scipy

        if save:
            with open(filename, 'wb') as f:
                pickle.dump(kde, f)

        #kde.set_bandwidth(bw_method=kde.factor * self.bandwidth_factor)

        return kde

    def resample(self, size=10000, label_list=None, model_path=None, scaler_path=None, min_max_path=None, batch_size=None):

        if model_path is None:
            model_path = self.model_path
        if scaler_path is None:
            scaler_path = self.scaler_path
        if min_max_path is None:
            min_max_path = self.min_max_path
        if label_list is None:
            label_list = self.path_dict['label_list']
        if batch_size is None:
            batch_size = self.batch_size
            if batch_size > size:
                batch_size = size
            

        # get the model, scaler and min_max
        kde, scaler, min_max = self.get_model_scaler_minmax(model_path, scaler_path, min_max_path)

        result_dict = {}
        for i in range(len(label_list)):
            result_dict[label_list[i]] = np.array([])

        while True:

            # generate random data
            # kde.set_bandwidth(bw_method=kde.factor * self.bandwidth_factor)
            if self.model_type == "gaussian_kde":
                data = kde.resample(batch_size).T
            elif self.model_type == "jax_gaussian_kde":
                key = random.PRNGKey(seed=np.random.randint(0, 1000000))
                data = kde.resample(key, (batch_size,)).T
                data = np.array(data)

            # inverse transform the data
            data = scaler.inverse_transform(data)

            j = 0
            idx = np.ones(data.shape[0], dtype=bool)
            for key, value in min_max.items():
                min_data = value['min_data']
                max_data = value['max_data']
                data_ = data[:, j]
                idx &= (data_<min_data) | (data_>max_data)
            
            for i in range(len(label_list)):
                result_dict[label_list[i]] = np.concatenate([result_dict[label_list[i]], data[:, i][~idx]])

            final_size = result_dict[label_list[0]].shape[0]
            batch_size = (size - final_size) + 1
            if final_size >= size:
                for key, value in result_dict.items():
                    result_dict[key] = value[:size]
                break

        return result_dict

    def random(self, size=10000, min_max_dict=None):

        if min_max_dict is None:
            min_max_dict = load_json(self.min_max_path)

        result_dict = {}
        for key, value in min_max_dict.items():
            min_data = value['min_data']
            max_data = value['max_data']
            result_dict[key] = np.random.uniform(min_data, max_data, size)

        return result_dict

    def pdf(self, data_dict, extra_key=False):

        model_path = self.model_path
        scaler_path = self.scaler_path
        min_max_path = self.min_max_path
        
        if extra_key:
            label_list = self.path_dict['label_list']
            key_list = list(data_dict.keys())
            # delete the keys not in label_list
            for key in key_list:
                if key not in label_list:
                    del data_dict[key]

        # data = np.array(list(data_dict.values())).T
        
        scaled_data, _ = self.feature_scaling(data_dict, save=False, call=True, filename=scaler_path)
        
        with open(model_path, 'rb') as f:
            KDE = pickle.load(f)
        # KDE.set_bandwidth(bw_method=KDE.factor * self.bandwidth_factor)

        # pdf
        if self.model_type == "gaussian_kde":
            pdf = KDE.pdf(scaled_data.T)
        elif self.model_type == "jax_gaussian_kde":
            scaled_data = jnp.array(scaled_data)
            pdf = KDE.pdf(scaled_data.T)
            pdf = np.array(pdf)
        else:
            raise ValueError("model_type should be either 'gaussian_kde' or 'jax_gaussian_kde'")

        # get min and max
        min_max = load_json(min_max_path)

        for key, value in min_max.items():
            min_data = value['min_data']
            max_data = value['max_data']
            idx = (data_dict[key]<min_data) | (data_dict[key]>max_data)

            pdf[idx] = 0

        return pdf

    def plot(self, data_dict1, data_dict2=None, labels=None):

        if labels is None:
            labels = self.path_dict['label_list']

        data_1 = []
        for key, value in data_dict1.items():
            data_1.append(value)
        data_1 = np.array(data_1).T

        if data_dict2 is not None:

            data_2 = []
            for key, value in data_dict2.items():
                data_2.append(value)
            data_2 = np.array(data_2).T

            # plot the corner plot
            fig = corner.corner(data_2 ,color = 'C0', density = True, plot_datapoints=False, label='train', hist_kwargs={'density':True})

            corner.corner(data_1, fig=fig,color='C1',density=True, labels=labels, show_titles=True, plot_datapoints=False, quantiles=[0.05, 0.5, 0.95], hist_kwargs={'density':True})

            colors = ['C0', 'C1']
            sample_labels = ['data_2', 'data_1']
            plt.legend(
                handles=[
                    mlines.Line2D([], [], color=colors[i], label=sample_labels[i])
                    for i in range(2)
                ],
                fontsize=20, frameon=False,
                bbox_to_anchor=(1, data_1.shape[1]), loc="upper right"
            )
            plt.gcf().set_size_inches(20, 20)           
        else:
            corner.corner(
                data_1, 
                color='C1',
                density=True, 
                labels=labels, 
                show_titles=True, 
                plot_datapoints=False, 
                quantiles=[0.05, 0.5, 0.95], 
                hist_kwargs={'density':True},
                )





        
