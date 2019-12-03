'''
@adapted from rasp.cbrain
'''
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import xarray as xr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *
from normalization import *


class SubsetSampler(Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices


    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

    def on_epoch_end(self):
        self.indices = np.arange(self.n_batches)
        if self.shuffle: np.random.shuffle(self.indices)



class spcamDataset(Dataset):
    def __init__(self, params,phase="train"):
        # Just copy over the attributes
        if phase == "train":
            self.data_fn  = params['data_fn']
            self.batch_size, self.shuffle = params['batch_size'], params['shuffle']
        else :
            self.data_fn =  params['valid_fn']
            self.batch_size, self.shuffle = params['batch_size'] * 10, params['shuffle']
        

        self.norm_fn = params['norm_fn']
        self.input_vars, self.output_vars = params['input_vars'], params['output_vars']
        

        # Open datasets
        self.data_ds = xr.open_dataset(self.data_fn)
        if  params['norm_fn'] is not None: self.norm_ds = xr.open_dataset( params['norm_fn'])

        # Compute number of samples and batches
        self.n_samples = self.data_ds.vars.shape[0]
        self.n_batches = int(np.floor(self.n_samples) / self.batch_size)


        # Get input and output variable indices
        self.input_idxs = return_var_idxs(self.data_ds, params['input_vars'], params['var_cut_off'])
        self.output_idxs = return_var_idxs(self.data_ds, params['output_vars'])
        self.n_inputs, self.n_outputs = len(self.input_idxs), len(self.output_idxs)

        # Initialize input and output normalizers/transformers

        self.input_transform = InputNormalizer(
                self.norm_ds, params['input_vars'], params['input_transform'][0], params['input_transform'][1], params['var_cut_off'])

        self.output_transform = PrectNormalizer(
                self.norm_ds, params['output_vars'], params['input_transform'][0], params['input_transform'][1], params['var_cut_off'],params['model_type'])

        self.indices = np.arange(self.n_batches)
        if self.shuffle: np.random.shuffle(self.indices)
    
    def __len__(self, index):
        return self.n_batches

    def __getitem__(self, index):
        # Compute start and end indices for batch
        start_idx = index * self.batch_size
        end_idx = start_idx + self.batch_size

        # Grab batch from data
        batch = self.data_ds['vars'][start_idx:end_idx]

        # Split into inputs and outputs
        X = batch[:, self.input_idxs]
        Y = batch[:, self.output_idxs]

        # Normalize
        X = self.input_transform.transform(X)
        Y = self.output_transform.transform(Y)
        
        return np.array(X),np.array(Y)



if __name__ == "__main__":
    
    params = {
        'data_fn' : 'data/VerySmallDataset/000_train.nc',
        'input_vars' : ['TBP', 'QBP', 'PS', 'SOLIN', 'SHFLX', 'LHFLX'],
        'output_vars' : ['PRECT'],
        'norm_fn':'data/VerySmallDataset/000_norm.nc',
        'input_transform':("min", "maxrs"),
        'batch_size': 1024,
        'shuffle' : True, 
        'var_cut_off': None,
        'model_type':"classification", ## dataset related
        }
    training_set = spcamDataset(params)
    sd = DataLoader(training_set, sampler = SubsetSampler(training_set.indices))

    for i in sd:
        print(i[0].shape,i[1].shape)
        
        break