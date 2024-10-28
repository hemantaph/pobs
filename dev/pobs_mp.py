import pobs
import contextlib

def pobs_mp(input_arguments):
    sample_size = input_arguments[0]
    kde_model_type = input_arguments[1]
    posterior1 = input_arguments[2]
    posterior2 = input_arguments[3]

    with contextlib.redirect_stdout(None):
        test = pobs.POBS(
            posterior1=posterior1,
            posterior2=posterior2,
            # create_new=True,
            kde_model_type=kde_model_type,
            spin_zero=True,
            npool=1,
        )

    return test.bayes_factor(sample_size=sample_size)