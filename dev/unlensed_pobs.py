import os
import numpy as np
import pobs
from scipy.special import comb
import itertools
import pobs_mp
from pobs.utils import append_json


# get all hdf5 files in the directory
dir_ = '/home/hemantakumar.phurailatpam/pobs/processed_data/pe_results/unlensed/'
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
kde_model_type = "jax_gaussian_kde"
for i in range(len(idx1)):
    input_arguments.append([size, kde_model_type, files[idx1[i]], files[idx2[i]]])

save_file_name = 'unlensed_pobs_results.json'

for i in range(len(input_arguments)):
    result = pobs_mp.pobs_mp(input_arguments[i])
    dict_ = {
        'bayes_factor': [result[0]],
        'log10_bayes_factor': [result[1]],
    }
    append_json(save_file_name, dict_, replace=False)
    
