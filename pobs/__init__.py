"""
pobs: Posterior Overlap Bayesian Statistics 
"""

import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

__author__ = 'hemanta_ph <hemantaphurailatpam@gmail.com>'

__version__ = "0.1.0"

# add __file__
import os
__file__ = os.path.abspath(__file__)

from .pobs import *
from .njit_functions import *
from .utils import *
from .mp import *
from .scaler import *
from .modelgenerator import *
from .data_check import *
# import data directory
from . import data
from . import meta_dict