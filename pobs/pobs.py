import os
from multiprocessing import Pool
import numpy as np

from .utils import (
    load_json, 
    save_json, 
)

from .mp import (
    bayes_factor_multiprocessing,
)

from .modelgenerator import ModelGenerator
from .utils import (
    get_dict_or_file,
    dir_check,
    data_check_astro_lensed,
    data_check_astro_lensed_sky,
    data_check_astro_unlensed,
    data_check_astro_unlensed_time,
    data_check_astro_unlensed_sky,
    data_check_posterior,
    data_check_posterior_sky,
    data_check_posterior_combined,
    data_check_posterior_combined_sky,
)

class POBS():
    def __init__(self,
        posterior1,
        posterior2,
        npool=4,
        kde_model_type="dpgmm", 
        astro_lensed=None,
        astro_unlensed=None,
        path_dict_all="path_dict_all.json", # path_dict_all will become pobs_directory+path_dict_all; dont add pobs_directory path directly
        pobs_directory="./pobs_data",
        create_new=False,
        num_images=2,
        kde_args=None,
        **kwargs
    ):

        # dir_check adds '/' at the end of the path if it is not present
        self.npool = npool
        self.num_images = num_images
        self.kde_model_type = kde_model_type
        self.pobs_directory = pobs_directory
        self.path_dict_all = dir_check(pobs_directory)+path_dict_all
        self.kde_args = kde_args    
        self.kwargs = kwargs

        # 
        self.pobs_dict = self.init_path_dict(
            path_dict_all=self.path_dict_all,
            create_new=create_new,
            astro_lensed=False if astro_lensed is None else True,
            astro_lensed_sky=False if astro_lensed is None else True,
            astro_unlensed=False if astro_unlensed is None else True,
            astro_unlensed_sky=False if astro_unlensed is None else True,
            astro_unlensed_time=False if astro_unlensed is None else True,
            posterior1=False if posterior1 is None else True,
            posterior1_sky=False if posterior1 is None else True,
            posterior2=False if posterior2 is None else True,
            posterior2_sky=False if posterior2 is None else True,
            posterior_combined=False if (posterior1 is None) or (posterior2 is None) else True,
            posterior_combined_sky=False if (posterior1 is None) or (posterior2 is None) else True,
            )

        # check for all models and scalers
        self.check_astro_lensed_model(data_dict=astro_lensed)
        self.check_astro_unlensed_model(data_dict=astro_unlensed)
        self.check_posterior_model(posterior1=posterior1, posterior2=posterior2)

    def init_path_dict(self, path_dict_all, create_new=False, **kwargs):
        
        # check if ./pobs_data/path_dict_all.json exists
        if os.path.exists(path_dict_all):
            if create_new:
                os.remove(path_dict_all)
                pobs_dict = {}
            else:
                pobs_dict = load_json(path_dict_all)
        else:
            pobs_dict = {}

        name_list1 = ["astro_lensed", "astro_lensed_sky", "astro_unlensed", "astro_unlensed_sky", "astro_unlensed_time", "posterior1", "posterior1_sky", "posterior2", "posterior2_sky", "posterior_combined", "posterior_combined_sky"]
        if self.kde_model_type == "dpgmm":
            name_list2 = ['scaling_param', 'model_path', 'scaler_path', 'min_max_path', 'renorm_const_path']

        for name in name_list1:
            # check if name is in pobs_dict
            if name not in pobs_dict:
                pobs_dict[name] = {'create_new' : True}
            else:  # if name is in pobs_dict
                # check name_list2 exists in pobs_dict[name]
                if not all([key in pobs_dict[name] for key in name_list2]):
                    pobs_dict[name] = {'create_new' : True}
                else:
                    pobs_dict[name]['create_new'] = False

        # Overwrite the 'create_new' flag based on the presence of new arguments
        for key, value in kwargs.items():
            if value is True and key in pobs_dict:
                pobs_dict[key]['create_new'] = True

        return pobs_dict

    def check_astro_lensed_model(self, data_dict):
        """
        Check if the astro_lensed model exists, if not create a new one.
        data_dict is consider a dictionary or a json file
        """

        # if data is not provided get it from the pobs module, a default data_dict
        if data_dict is not None:
            data_dict = get_dict_or_file(data_dict)
            astro_lensed_dict = data_check_astro_lensed(data_dict)
            astro_lensed_sky_dict = data_check_astro_lensed_sky(data_dict)
            del data_dict
        else:
            astro_lensed_dict = None
            astro_lensed_sky_dict = None

        # path_dict_ keys can have the values: model_path, scaler_path, min_max_path, renorm_const_path, label_list
        # if one of the keys is not present, create_new=True is already set by init_path_dict fuction
        # if data_dict is not None, create_new=True is already set by init_path_dict fuction
        model_name = "astro_lensed" 
        pobs_dict_ = self.pobs_dict[model_name]
        self.astro_lensed = ModelGenerator(
            model_name=model_name,
            model_type=model_name,
            data_dict=astro_lensed_dict,
            pobs_directory=self.pobs_directory,
            create_new=pobs_dict_['create_new'],
            num_images=self.num_images,
            kde_args = self.kde_args,
        )
        self.pobs_dict[model_name] = self.astro_lensed.meta_dict
        save_json(self.path_dict_all, self.pobs_dict)
        
        model_name = "astro_lensed_sky"
        pobs_dict_ = self.pobs_dict[model_name]
        self.astro_lensed_sky = ModelGenerator(
            model_name=model_name,
            model_type=model_name,
            data_dict=astro_lensed_sky_dict,
            pobs_directory=self.pobs_directory,
            create_new=pobs_dict_['create_new'],
            num_images=self.num_images,
            kde_args = self.kde_args,  
        )
        self.pobs_dict[model_name] = self.astro_lensed_sky.meta_dict
        save_json(self.path_dict_all, self.pobs_dict)

    def check_posterior_model(self, posterior1, posterior2):
        """
        Check if the posterior model exists, if not create a new one.
        posterior1 and posterior2 are consider a dictionary or a json file
        """
        if (posterior1 is not None) or (posterior2 is not None):
            posterior1 = get_dict_or_file(posterior1, bilby_hdf5_posterior=True)
            posterior2 = get_dict_or_file(posterior2, bilby_hdf5_posterior=True)

            t1 = np.median(posterior1['geocent_time']) 
            t2 = np.median(posterior2['geocent_time'])
            if t1>t2:
                print("posterior1 is for image 2 and posterior2 is for image 1. Swapping them the data in correct order.")
                posterior1, posterior2 = posterior2, posterior1
                self.pobs_dict['posterior1'], self.pobs_dict['posterior2'] = self.pobs_dict['posterior2'], self.pobs_dict['posterior1']
            self.dt_12 = np.abs(t2-t1)
            save_json(f"{self.pobs_directory}dt_12.json", self.dt_12)

            posterior1_dict = data_check_posterior(posterior1)
            posterior1_sky_dict = data_check_posterior_sky(posterior1)

            posterior2_dict = data_check_posterior(posterior2)
            posterior2_sky_dict = data_check_posterior_sky(posterior2)

            posterior_combined_dict = data_check_posterior_combined(posterior1, posterior2)
            posterior_combined_sky_dict = data_check_posterior_combined_sky(posterior1, posterior2)
            del posterior1, posterior2
        else:
            posterior1_dict = None
            posterior1_sky_dict = None
            posterior2_dict = None
            posterior2_sky_dict = None
            posterior_combined_dict = None
            posterior_combined_sky_dict = None
            self.dt_12 = load_json(f"{self.pobs_directory}dt_12.json")

        model_name = "posterior1" 
        pobs_dict_ = self.pobs_dict[model_name]
        self.posterior1 = ModelGenerator(
            model_name=model_name,
            model_type='posterior',
            data_dict=posterior1_dict,
            pobs_directory=self.pobs_directory,
            create_new=pobs_dict_['create_new'],
            num_images=self.num_images,
            kde_args = self.kde_args,  
        )
        self.pobs_dict[model_name] = self.posterior1.meta_dict
        save_json(self.path_dict_all, self.pobs_dict)

        model_name = "posterior1_sky"
        pobs_dict_ = self.pobs_dict[model_name]
        self.posterior1_sky = ModelGenerator(
            model_name=model_name,
            model_type='posterior_sky',
            data_dict=posterior1_sky_dict,
            pobs_directory=self.pobs_directory,
            create_new=pobs_dict_['create_new'],
            num_images=self.num_images,
            kde_args = self.kde_args,  
        )
        self.pobs_dict[model_name] = self.posterior1_sky.meta_dict
        save_json(self.path_dict_all, self.pobs_dict)

        model_name = "posterior2"
        pobs_dict_ = self.pobs_dict[model_name]
        self.posterior2 = ModelGenerator(
            model_name=model_name,
            model_type='posterior',
            data_dict=posterior2_dict,
            pobs_directory=self.pobs_directory,
            create_new=pobs_dict_['create_new'],
            num_images=self.num_images,
            kde_args = self.kde_args,  
        )
        self.pobs_dict[model_name] = self.posterior2.meta_dict
        save_json(self.path_dict_all, self.pobs_dict)

        model_name = "posterior2_sky"
        pobs_dict_ = self.pobs_dict[model_name]
        self.posterior2_sky = ModelGenerator(
            model_name=model_name,
            model_type='posterior_sky',
            data_dict=posterior2_sky_dict,
            pobs_directory=self.pobs_directory,
            create_new=pobs_dict_['create_new'],
            num_images=self.num_images,
            kde_args = self.kde_args,  
        )
        self.pobs_dict[model_name] = self.posterior2_sky.meta_dict
        save_json(self.path_dict_all, self.pobs_dict)

        model_name = "posterior_combined"
        pobs_dict_ = self.pobs_dict[model_name]
        self.posterior_combined = ModelGenerator(
            model_name=model_name,
            model_type=model_name,
            data_dict=posterior_combined_dict,
            pobs_directory=self.pobs_directory,
            create_new=pobs_dict_['create_new'],
            num_images=self.num_images,
            kde_args = self.kde_args,  
        )
        self.pobs_dict[model_name] = self.posterior_combined.meta_dict
        save_json(self.path_dict_all, self.pobs_dict)

        model_name = "posterior_combined_sky"
        pobs_dict_ = self.pobs_dict[model_name]
        self.posterior_combined_sky = ModelGenerator(
            model_name=model_name,
            model_type=model_name,
            data_dict=posterior_combined_sky_dict,
            pobs_directory=self.pobs_directory,
            create_new=pobs_dict_['create_new'],
            num_images=self.num_images,
            kde_args = self.kde_args,  
        )
        self.pobs_dict[model_name] = self.posterior_combined_sky.meta_dict
        save_json(self.path_dict_all, self.pobs_dict)

    def check_astro_unlensed_model(self, data_dict):
        """
        Check if the astro_unlensed model exists, if not create a new one.
        data_dict is consider a dictionary or a json file
        """

        # if data is not provided get it from the pobs module, a default data_dict
        if data_dict is not None:
            data_dict = get_dict_or_file(data_dict)
            astro_unlensed_dict = data_check_astro_unlensed(data_dict)
            astro_unlensed_sky_dict = data_check_astro_unlensed_sky(data_dict)
            astro_unlensed_time_dict = data_check_astro_unlensed_time(data_dict)
            del data_dict
        else:
            astro_unlensed_dict = None
            astro_unlensed_sky_dict = None
            astro_unlensed_time_dict = None

        model_name = "astro_unlensed" 
        pobs_dict_ = self.pobs_dict[model_name]
        self.astro_unlensed = ModelGenerator(
            model_name=model_name,
            model_type=model_name,
            data_dict=astro_unlensed_dict,
            pobs_directory=self.pobs_directory,
            create_new=pobs_dict_['create_new'],
            num_images=self.num_images,
            kde_args = self.kde_args,  
        )
        self.pobs_dict[model_name] = self.astro_unlensed.meta_dict
        save_json(self.path_dict_all, self.pobs_dict)

        model_name = "astro_unlensed_sky"
        pobs_dict_ = self.pobs_dict[model_name]
        self.astro_unlensed_sky = ModelGenerator(
            model_name=model_name,
            model_type=model_name,
            data_dict=astro_unlensed_sky_dict,
            pobs_directory=self.pobs_directory,
            create_new=pobs_dict_['create_new'],
            num_images=self.num_images,
            kde_args = self.kde_args,  
        )
        self.pobs_dict[model_name] = self.astro_unlensed_sky.meta_dict
        save_json(self.path_dict_all, self.pobs_dict)

        model_name = "astro_unlensed_time"
        pobs_dict_ = self.pobs_dict[model_name]
        self.astro_unlensed_time = ModelGenerator(
            model_name=model_name,
            model_type=model_name,
            data_dict=astro_unlensed_time_dict,
            pobs_directory=self.pobs_directory,
            create_new=pobs_dict_['create_new'],
            num_images=self.num_images,
            kde_args = self.kde_args,  
        )
        self.pobs_dict[model_name] = self.astro_unlensed_time.meta_dict
        save_json(self.path_dict_all, self.pobs_dict)

    def po_hemanta_numerator(self, sample_size=100000):

        result_size = 0 
        sample_size_original = sample_size
        blu_numerator = np.array([])
        while result_size < sample_size_original:
            # print('numerator', result_size)
            # print('sample_size', sample_size)
            # resample
            posterior_combined_dict = self.posterior_combined.resample(sample_size)
            posterior_combined_sky_dict = self.posterior_combined_sky.resample(sample_size)

            # posterior1
            posterior1_dict = posterior_combined_dict.copy()
            posterior1_dict['dl'] = posterior_combined_dict['dl_1']
            del posterior1_dict['dl_1'], posterior1_dict['dl_2']

            # posterior2
            posterior2_dict = posterior1_dict.copy()
            posterior2_dict['dl'] = posterior_combined_dict['dl_2']

            # atrso_lensed
            astro_lensed_dict = posterior_combined_dict.copy()
            # dt_12 is a constant
            astro_lensed_dict['dt_12'] = self.dt_12*np.ones(sample_size)

            # pdf calculations
            pdf_posterior1 = self.posterior1.pdf(posterior1_dict)
            pdf_posterior1_sky = self.posterior1_sky.pdf(posterior_combined_sky_dict)
            pdf_posterior2 = self.posterior2.pdf(posterior2_dict)
            pdf_posterior2_sky = self.posterior2_sky.pdf(posterior_combined_sky_dict)
            pdf_astro_lensed = self.astro_lensed.pdf(astro_lensed_dict)
            pdf_astro_lensed_sky = self.astro_lensed_sky.pdf(posterior_combined_sky_dict)
            pdf_posterior_combined = self.posterior_combined.pdf(posterior_combined_dict)
            pdf_posterior_combined_sky = self.posterior_combined_sky.pdf(posterior_combined_sky_dict)

            # ignore the zero values
            # note that buffer_array can have zero values if pdf123<<pdf456
            pdf_nu = pdf_posterior1 * pdf_posterior1_sky * pdf_posterior2 * pdf_posterior2_sky * pdf_astro_lensed * pdf_astro_lensed_sky
            pdf_de = pdf_posterior_combined * pdf_posterior_combined_sky

            # idx = (pdf_de!=0) & (pdf_posterior1!=0) & (pdf_posterior1_sky!=0) & (pdf_posterior2!=0) & (pdf_posterior2_sky!=0) & (pdf_astro_lensed!=0) & (pdf_astro_lensed_sky!=0)
            idx = (pdf_de!=0)
            buffer_array = pdf_nu[idx] / pdf_de[idx]

            # check for inf and nan 
            idx = (buffer_array!=np.inf) & (buffer_array!=-np.inf) & (np.isnan(buffer_array)==False) & (buffer_array!=0)
            buffer_array = buffer_array[idx]
            
            if len(buffer_array) != 0:
                # append
                blu_numerator = np.concatenate((blu_numerator, buffer_array))
                result_size = len(blu_numerator)
                sample_size = sample_size_original - result_size

        return blu_numerator

    def po_hemanta_denominator(self, sample_size=100000):

        result_size = 0
        sample_size_original = sample_size
        blu_denominator = np.array([])
        while result_size < sample_size_original:
            # resample
            posterior1_dict = self.posterior1.resample(sample_size)
            posterior1_sky_dict = self.posterior1_sky.resample(sample_size)
            posterior2_dict = self.posterior2.resample(sample_size)
            posterior2_sky_dict = self.posterior2_sky.resample(sample_size)

            # pdf calculations
            pdf_astro_unlensed1 = self.astro_unlensed.pdf(posterior1_dict)
            pdf_astro_unlensed1_sky = self.astro_unlensed_sky.pdf(posterior1_sky_dict)
            pdf_astro_unlensed2 = self.astro_unlensed.pdf(posterior2_dict)
            pdf_astro_unlensed2_sky = self.astro_unlensed_sky.pdf(posterior2_sky_dict)

            # ignore the zero values, inf and nan
            # note that buffer_array can have zero values if pdf12<<pdf34
            buffer_array = pdf_astro_unlensed1 * pdf_astro_unlensed1_sky * pdf_astro_unlensed2 * pdf_astro_unlensed2_sky
            idx = (buffer_array!=np.inf) & (buffer_array!=-np.inf) & (np.isnan(buffer_array)==False) 
            buffer_array = buffer_array[idx]
            
            if len(buffer_array) != 0:
                # append
                blu_denominator = np.concatenate((blu_denominator, buffer_array))
                result_size = len(blu_denominator)
                sample_size_original = sample_size_original - result_size

        return blu_denominator

    def bayes_factor(self, sample_size=100000):
        """
        Calculate the bayes factor
        """

        blu_numerator = self.po_hemanta_numerator(sample_size=sample_size)
        blu_denominator = self.po_hemanta_denominator(sample_size=sample_size)

        # astro_unlensed_time
        atrso_unlensed_dict = {'dt' : self.dt_12*np.ones(1)}
        pdf_astro_unlensed_time = self.astro_unlensed_time.pdf(atrso_unlensed_dict)[0]

        avg_numerator = np.average(blu_numerator)
        avg_denominator = np.average(blu_denominator)*pdf_astro_unlensed_time

        log10_bayes_factor = np.log10(avg_numerator)-np.log10(avg_denominator)
        bayes_factor = avg_numerator/avg_denominator

        return bayes_factor, log10_bayes_factor

    def bayes_factor_multiprocessing(self, sample_size=10000):

        npool = self.npool
        # divide the sample_size by npool
        size_ = int(sample_size/npool)
        sample_size_list = [size_ for i in range(npool)]
        # take care of the remainder
        sample_size_list[-1] += sample_size - size_*npool

        # return input_arguments
        input_arguments = [
            [sample_size_list[i], # 0
            self.log_dt_12_days, # 1
            self.astro_lensed, # 2
            self.astro_unlensed1, # 3
            self.astro_unlensed2, # 4
            self.pe_prior, # 5
            self.posterior1, # 6
            self.posterior2, # 7
            self.posterior_combined, # 8 
            ] for i in range(npool)]

        numerator_list = []
        denominator_list = []

        with Pool(processes=npool) as pool:
            for numerator_, denominator_ in pool.map(bayes_factor_multiprocessing, input_arguments):
                numerator_list += numerator_
                denominator_list += denominator_

        # # with for loop
        # for input_arguments_ in input_arguments:
        #     numerator_, denominator_ = bayes_factor_multiprocessing(input_arguments_)
        #     numerator_list += numerator_
        #     denominator_list += denominator_

        avg_numerator = np.average(numerator_list)
        avg_denominator = np.average(denominator_list)

        log10_bayes_factor = np.log10(avg_numerator)-np.log10(avg_denominator)
        bayes_factor = avg_numerator/avg_denominator

        return bayes_factor, log10_bayes_factor

