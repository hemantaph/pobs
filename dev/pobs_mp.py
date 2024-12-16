import pobs
import contextlib
from pobs.utils import append_json

def pobs_mp(input_arguments):
    sample_size = input_arguments[0]
    kde_model_type = input_arguments[1]
    posterior1 = input_arguments[2]
    posterior2 = input_arguments[3]

    # with contextlib.redirect_stdout(None):
    #     test = pobs.POBS(
    #         posterior1=posterior1,
    #         posterior2=posterior2,
    #         # create_new=True,
    #         kde_model_type=kde_model_type,
    #         spin_zero=True,
    #         npool=1,
    #     )
    test = pobs.POBS(
        posterior1=posterior1,
        posterior2=posterior2,
        # create_new=True,
        kde_model_type=kde_model_type,
        spin_zero=True,
        npool=1,
    )

    result = test.bayes_factor(sample_size=sample_size)
    try:
        dict_ = {
            'bayes_factor': [result[0]],
            'log10_bayes_factor': [result[1]],
        }
        save_file_name = 'unlensed_pobs_results.json'
        append_json(save_file_name, dict_, replace=False)
    except:
        pass

    return result