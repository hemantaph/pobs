import os
import numpy as np
import corner
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from .utils import (
    load_json, 
    save_json, 
    load_hdf5, 
    save_hdf5,
)

from .modelgenerator import ModelGenerator
from .utils import (
    get_dict_or_file,
    dir_check,
    data_check_astro_lensed,
    data_check_posterior,
    data_check_astro_unlensed,
    data_check_pe_prior,
    load_data_from_module,
    initialize_bilby_priors,
)

class POBS():
    def __init__(self,
        posterior1,
        posterior2,
        npool=4,
        kde_model_type="gaussian_kde", # or "jax_gaussian_kde"
        astro_lensed=None,
        astro_unlensed=None,
        pe_prior=None,
        path_dict_all="path_dict_all.json", # path_dict_all will become pobs_directory+path_dict_all; dont add pobs_directory path directly
        pobs_directory="./pobs_data",
        create_new=False,
        spin_zero=False,
        **kwargs
    ):

        # dir_check adds '/' at the end of the path if it is not present
        self.npool = npool
        self.kde_model_type = kde_model_type
        self.pobs_directory = pobs_directory
        self.path_dict_all = dir_check(pobs_directory)+path_dict_all
        self.spin_zero = spin_zero
        if create_new:
            # delete the existing path_dict_all.json
            if os.path.exists(self.path_dict_all):
                os.remove(self.path_dict_all)

        self.pobs_dict = self.init_path_dict(
            path_dict_all=self.path_dict_all,
            astro_lensed=astro_lensed,
            astro_unlensed1=astro_unlensed,
            astro_unlensed2=astro_unlensed,
            posterior1=posterior1,
            posterior2=posterior2,
            pe_prior=pe_prior,
            )

        # check for all models and scalers
        self.check_astro_lensed_model(data_dict=astro_lensed)
        self.check_astro_unlensed_model(data_dict=astro_unlensed)
        self.check_pe_prior_model(data_dict=pe_prior)
        self.check_posterior_model(posterior1=posterior1, posterior2=posterior2)

    def check_astro_lensed_model(self, data_dict):
        """
        Check if the astro_lensed model exists, if not create a new one.
        data_dict is consider a dictionary or a json file
        """

        # path_dict_ keys can have the values: model_path, scaler_path, min_max_path, label_list, bandwidth_factor 
        # if one of the keys is not present, create_new=True is already set by init_path_dict fuction
        # if data_dict is not None, create_new=True is already set by init_path_dict fuction
        pobs_dict_ = self.pobs_dict['astro_lensed']

        # if data is not provided get it from the pobs module, a default data_dict
        if data_dict is not None:
            data_dict = get_dict_or_file(data_dict)
        else:
            print("astro_lensed is None")
            print("getting default astro_lensed data_dict from pobs module")
            data_dict = load_data_from_module('pobs', 'data', 'n_lensed_detectable_bbh_po_spin.json')

        data_dict = data_check_astro_lensed(data_dict)
        # check if spin_zero is True, if True delete chi_eff from data_dict
        if self.spin_zero:
            del data_dict['chi_eff']
        
        self.astro_lensed = self.create_or_get_model(
            model_name="astro_lensed",
            data_dict=data_dict,
            #bandwidth_factor=0.25,
            create_new=pobs_dict_['create_new'],
        )

    def check_pe_prior_model(self, data_dict):

        pobs_dict_ = self.pobs_dict['pe_prior']

        # if data is not provided get it from the pobs module, a default data_dict
        if data_dict is not None:
            data_dict = get_dict_or_file(data_dict)
        else:
            print("pe_prior is None")
            print("getting default pe_prior data_dict from pobs module")
            data_dict = initialize_bilby_priors(size=100000)

        data_dict = data_check_pe_prior(data_dict)
        if self.spin_zero:
            del data_dict['chi_eff']

        self.pe_prior = self.create_or_get_model(
            model_name="pe_prior",
            data_dict=data_dict,
            #bandwidth_factor=0.25,
            create_new=pobs_dict_['create_new'],
        )

    def check_posterior_model(self, posterior1, posterior2):
        """
        Check if the posterior model exists, if not create a new one.
        posterior1 and posterior2 are consider a dictionary or a json file
        """

        pobs_dict1 = self.pobs_dict['posterior1']
        pobs_dict2 = self.pobs_dict['posterior2']

        # if data is not provided get it from the pobs module, a default data_dict
        if posterior1 is not None:
            data_dict1 = get_dict_or_file(posterior1)
        else:
            print("posterior1 is None")
            print("getting default posterior1 data_dict from pobs module")
            data_dict1 = load_data_from_module('pobs', 'data', 'image1.hdf5')

        if posterior2 is not None:
            data_dict2 = get_dict_or_file(posterior2)
        else:
            print("posterior2 is None")
            print("getting default posterior2 data_dict from pobs module")
            data_dict2 = load_data_from_module('pobs', 'data', 'image2.hdf5')

        data_dict1, data_dict2, data_combine, self.log_dt_12_days = data_check_posterior(data_dict1, data_dict2)
        # check if spin_zero is True, if True delete chi_eff from data_dict
        if self.spin_zero:
            del data_dict1['chi_eff'], data_dict2['chi_eff'], data_combine['chi_eff']

        self.posterior1 = self.create_or_get_model(
            model_name="posterior1",
            data_dict=data_dict1,
            #bandwidth_factor=0.25,
            create_new=pobs_dict1['create_new'],
        )

        self.posterior2 = self.create_or_get_model(
            model_name="posterior2",
            data_dict=data_dict2,
            #bandwidth_factor=0.25,
            create_new=pobs_dict2['create_new'],
        )

        self.posterior_combine = self.create_or_get_model(
            model_name="posterior_combine",
            data_dict=data_combine,
            #bandwidth_factor=0.25,
            create_new= pobs_dict1['create_new'] or pobs_dict2['create_new'],
        )

    def check_astro_unlensed_model(self, data_dict):
        """
        Check if the astro_unlensed model exists, if not create a new one.
        data_dict is consider a dictionary or a json file
        """

        # path_dict_ keys can have the values: model_path, scaler_path, min_max_path, label_list, bandwidth_factor 
        # if one of the keys is not present, create_new=True is already set by init_path_dict fuction
        # if data_dict is not None, create_new=True is already set by init_path_dict fuction
        pobs_dict1 = self.pobs_dict['astro_unlensed1']
        pobs_dict2 = self.pobs_dict['astro_unlensed2']

        # if data is not provided get it from the pobs module, a default data_dict
        if data_dict is not None:
            data_dict = get_dict_or_file(data_dict)
        else:
            print("astro_unlensed is None")
            print("getting default astro_unlensed data_dict from pobs module")
            data_dict = load_data_from_module('pobs', 'data', 'n_unlensed_detectable_bbh_po_spin.json')

        data_dict1, data_dict2 = data_check_astro_unlensed(data_dict)
        # check if spin_zero is True, if True delete chi_eff from data_dict
        if self.spin_zero:
            del data_dict1['chi_eff'], data_dict2['chi_eff']

        self.astro_unlensed1 = self.create_or_get_model(
            model_name="astro_unlensed1",
            data_dict=data_dict1,
            #bandwidth_factor=0.25,
            create_new=pobs_dict1['create_new'],
        )

        self.astro_unlensed2 = self.create_or_get_model(
            model_name="astro_unlensed2",
            data_dict=data_dict2,
            #bandwidth_factor=0.25,
            create_new=pobs_dict2['create_new'],
        )
        
    def init_path_dict(self, path_dict_all, **kwargs):
        
        # check if ./pobs_data/path_dict_all.json exists
        if os.path.exists(path_dict_all):
            pobs_dict = load_json(path_dict_all)
        else:
            pobs_dict = {}

        name_list1 = ["astro_lensed", "astro_unlensed1", "astro_unlensed2", "pe_prior", "posterior1", "posterior2", "posterior_combine"]
        name_list2 = ['model_path', 'scaler_path', 'min_max_path', 'label_list', 'bandwidth_factor']

        for name in name_list1:
            # check if name is in pobs_dict
            if name not in pobs_dict:
                pobs_dict[name] = {'create_new' : True}
            else:
                # check name_list2 exists in pobs_dict[name]
                if not all([key in pobs_dict[name] for key in name_list2]):
                    pobs_dict[name] = {'create_new' : True}
                else:
                    pobs_dict[name]['create_new'] = False

        # Overwrite the 'create_new' flag based on the presence of new arguments
        for key, value in kwargs.items():
            if value is not None and key in pobs_dict:
                pobs_dict[key]['create_new'] = True

        return pobs_dict


    def create_or_get_model(self, model_name, data_dict, create_new=True):

        if create_new:
            print(f"creating a new {model_name} model...")
            model = ModelGenerator(
                model_name=model_name,
                data_dict=data_dict,
                param_list=None, 
                pobs_directory=self.pobs_directory,
                model_type=self.kde_model_type,
                #bandwidth_factor=bandwidth_factor,
                create_new=True,
            )
            self.pobs_dict[model_name] = model.path_dict
            save_json(self.path_dict_all, self.pobs_dict)
        else:
            print(f"{model_name} already exists at {self.path_dict_all}, will load the existing model...")
            model = ModelGenerator(
                model_name=model_name,
                data_dict=data_dict,
                param_list=None, 
                pobs_directory=self.pobs_directory,
                model_type=self.kde_model_type,
                #bandwidth_factor=bandwidth_factor,
                create_new=False,
            )
        return model

    def po_hemanta_numerator(self, sample_size=100000):

        data_posterior_combine = self.posterior_combine.resample(sample_size)

        # posterior1
        data_dict1 = data_posterior_combine.copy()
        data_dict1['log10_dl'] = data_posterior_combine['log10_dl_1']
        del data_dict1['log10_dl_1'], data_dict1['log10_dl_2']
        # posterior2
        data_dict2 = data_dict1.copy()
        data_dict2['log10_dl'] = data_posterior_combine['log10_dl_2']
        # atrso_lensed
        data_dict3 = data_posterior_combine.copy()
        data_dict3['log10_dt_12_days'] = self.log_dt_12_days*np.ones(sample_size)
        # pe_prior
        # data_dict4 = data_dict1.copy()
        # data_dict5 = data_dict2.copy()
        # posterior_combine
        # data_dict6 = data_posterior_combine.copy()

        pdf1 = self.posterior1.pdf(data_dict1)
        pdf2 = self.posterior2.pdf(data_dict2)
        pdf3 = self.astro_lensed.pdf(data_dict3)
        pdf4 = self.pe_prior.pdf(data_dict1)
        pdf5 = self.pe_prior.pdf(data_dict2)
        pdf6 = self.posterior_combine.pdf(data_posterior_combine)

        blu_numerator = pdf1 * pdf2 * pdf3 / pdf4 / pdf5 / pdf6

        # check inf
        idx = blu_numerator!=np.inf
        idx &= blu_numerator!=-np.inf
        # check for nan
        idx &= np.isnan(blu_numerator)==False
        blu_numerator = blu_numerator[idx]

        return blu_numerator, pdf1, pdf2, pdf3, pdf4, pdf5, pdf6

    def po_hemanta_denominator(self, sample_size=100000):

        data_posterior1 = self.posterior1.resample(sample_size)
        data_posterior2 = self.posterior2.resample(sample_size)

        # astro_unlensed1
        # data_dict1 = data_posterior1.copy()
        # astro_unlensed2
        data_dict2 = data_posterior2.copy()
        data_dict2['log10_dt_12_days'] = self.log_dt_12_days*np.ones(sample_size)
        # pe_prior
        # data_dict3 = data_posterior1.copy()
        # data_dict4 = data_posterior2.copy()

        pdf1 = self.astro_unlensed1.pdf(data_posterior1)
        pdf2 = self.astro_unlensed2.pdf(data_dict2)
        pdf3 = self.pe_prior.pdf(data_posterior1)
        pdf4 = self.pe_prior.pdf(data_posterior2)

        blu_denominator = pdf1 * pdf2 / pdf3 / pdf4

        # check inf
        idx = blu_denominator!=np.inf
        idx &= blu_denominator!=-np.inf
        # check for nan
        idx &= np.isnan(blu_denominator)==False
        blu_denominator = blu_denominator[idx]

        return blu_denominator, pdf1, pdf2, pdf3, pdf4


    def bayes_factor(self, sample_size=100000):
        """
        Calculate the bayes factor
        """

        blu_numerator = self.po_hemanta_numerator(sample_size=sample_size)[0]
        blu_denominator = self.po_hemanta_denominator(sample_size=sample_size)[0]

        avg_numerator = np.average(blu_numerator)
        avg_denominator = np.average(blu_denominator)

        log10_bayes_factor = np.log10(avg_numerator)-np.log10(avg_denominator)
        bayes_factor = avg_numerator/avg_denominator

        return bayes_factor, log10_bayes_factor

    def plot(self, data_dict1, data_dict2=None, labels=None):

        if labels is None:
            labels = list(data_dict1.keys())

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

