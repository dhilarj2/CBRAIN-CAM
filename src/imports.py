"""
Just all the imports for all other scripts and notebooks.
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
import pandas as pd
import pickle
import pdb
import netCDF4 as nc
import xarray as xr
import h5py
from glob import glob
import sys, os
from os import path
from configargparse import ArgParser
import fire
import logging


with open(os.path.join(os.path.dirname(__file__), 'hyai_hybi.pkl'), 'rb') as f:
    hyai, hybi = pickle.load(f)
