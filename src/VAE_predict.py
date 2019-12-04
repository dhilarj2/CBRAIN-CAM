import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from models import CVAE
from sklearn import metrics
import torch.optim as optim
import torch.nn.functional as F
from __dataset import spcamDataset
from __dataset import SubsetSampler
from torch.utils.data import DataLoader
import xarray as xr
from normalization import PrectNormalizer


def predict(params):

    inverter =  PrectNormalizer(
                xr.open_dataset( params['norm_fn']), params['output_vars'],
                params['input_transform'][0], params['input_transform'][1], params['var_cut_off'],params['model_type'])

    model = CVAE(params)
    optimizer = optim.Adam(model.parameters(),lr=0.00001,weight_decay=0.001)

    ### Load model
    checkpoint = torch.load('./runs/VAE_model_DNN_classifier_exp_VAE_Exp04.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    model.cuda()
    model.eval()

    valid_dataset = spcamDataset(params, phase = "validation")
    valid_loader = DataLoader(valid_dataset, sampler = SubsetSampler(valid_dataset.indices))


    result_predicted, result_actual = [], []
    with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):

                target = target.squeeze(0).type(torch.float32).to(params['device'])

                z = torch.randn(data.shape[1], 16)

                z = torch.cat((z, data.squeeze(0)), dim=1).cuda()

                predPrecit = model.decoder(z)

                
                print("Batch MSE {}".format(metrics.mean_squared_error(predPrecit.detach().cpu().numpy(), target.squeeze(0).detach().cpu().numpy())))

                #val_loss = compute_loss(target, sampled_precit, mean, log_var) #.type(torch.FloatTensor).to(params['device']))
                #assert val_loss.requires_grad == False
                 
                result_predicted.extend(predPrecit.cpu().detach().numpy())
                result_actual.extend(target.squeeze(0).cpu().detach().numpy())


    mse = metrics.mean_squared_error(np.array(result_actual), np.array(result_predicted))

    print("MSE {}".format(mse))



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
        'model_type':"regression", ## dataset related

        'inputDim' : 64,
        'hiddenDim' : [32,32], ## Symmetrical
        'outputDim' : 16, #Hidden state #VAE related


        'epochs' : 30,
        'expNumber' : "VAE_Exp04", # to be changed
        'device' : 'cuda',
        'valid_fn' : 'data/VerySmallDataset/000_valid.nc' #Misc
    }
    predict(params)

    pass

