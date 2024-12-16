import os
import numpy as np
import pobs
from scipy.special import comb
import itertools
import pobs_mp
from pobs.utils import append_json
from multiprocessing import Pool

def main():
    # get all hdf5 files in the directory
    dir_ = '/Users/phurailatpamhemantakumar/phd/mypackages/pobs/processed_data/pe_results/unlensed/'
    files = os.listdir(dir_)
    files = [os.path.join(dir_, f) for f in files]

    len_ = len(files)
    size = 1000
    num_combinations = comb(len_, 2, exact=True)
    # Known number of combinations
    C = size if size < num_combinations else num_combinations
    print(C)
    len_ = (1 + np.sqrt(1 + 8 * C)) / 2
    # Define the index array
    idx_all = np.arange(0, int(len_))
    # # randomize idx_all
    # np.random.shuffle(idx_all)

    # Generate all possible two-element combinations
    combination_array = np.array(list(itertools.combinations(idx_all, 2)))
    idx1 = combination_array[:,0]
    idx2 = combination_array[:,1]

    # setting up input_arguments
    input_arguments = []
    size = 100000
    kde_model_type = "gaussian_kde"
    for i in range(len(idx1)):
        input_arguments.append([size, kde_model_type, files[idx1[i]], files[idx2[i]]])

    save_file_name = 'unlensed_pobs_results.json'

    # for i in range(len(input_arguments)):
    #     result = pobs_mp.pobs_mp(input_arguments[i])
    #     dict_ = {
    #         'bayes_factor': [result[0]],
    #         'log10_bayes_factor': [result[1]],
    #     }
    #     append_json(save_file_name, dict_, replace=False)

    test = pobs.POBS(
            posterior1=None,
            posterior2=None,
            create_new=True,
            kde_model_type=kde_model_type,
            spin_zero=True,
            npool=1,
        )
    # use multiprocessing
    
    npool = 4
    with Pool(processes=npool) as pool:
        for result in pool.map(pobs_mp.pobs_mp, input_arguments):
            print(result)

if __name__ == "__main__":
    main();