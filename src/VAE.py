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


def compute_loss(x, sampled_precit, mean, log_var):
    RCL = F.binary_cross_entropy(sampled_precit, x)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return RCL + KLD
    

def train(params):

    model = CVAE(params)
    model.to(params['device'])

    # data generator
    train_dataset = spcamDataset(params, phase = "train")
    valid_dataset = spcamDataset(params, phase = "validation")

    train_loader = DataLoader(train_dataset, sampler = SubsetSampler(train_dataset.indices))
    valid_loader = DataLoader(valid_dataset, sampler = SubsetSampler(valid_dataset.indices))

    optimizer = optim.Adam(model.parameters(),lr=0.00001,weight_decay=0.001)
    #criterion = nn.BCELoss().to(params['device'])

    performance = {}

    # training 
    for epoch in range(params['epochs']):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            target = target.squeeze(0).type(torch.float32).to(params['device'])
            sampled_precit, mean, log_var = model(target, data.squeeze(0).type(torch.float32).to(params['device']))
            loss = compute_loss(target, sampled_precit, mean, log_var)
            loss.backward()
            optimizer.step()

            if batch_idx > 1000:  # batch size 1024
                break
        
        result_predicted, result_actual = [], []
        
        #validation
        model.eval()

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                target = target.squeeze(0).type(torch.float32).to(params['device'])
                sampled_precit, mean, log_var = model(target,  data.squeeze(0).type(torch.float32).to(params['device']))
                val_loss = compute_loss(target, sampled_precit, mean, log_var) #.type(torch.FloatTensor).to(params['device']))

                assert val_loss.requires_grad == False
                 
                result_predicted.extend(sampled_precit.cpu().detach().numpy())
                result_actual.extend(target.squeeze(0).cpu().detach().numpy())


                if batch_idx > 100:  # batch size 10240 
                    break

        mse = metrics.mean_squared_error(np.array(result_actual), np.array(result_predicted))
        #auc_ = metrics.auc(fpr, tpr) 

        print('Epoch {} : Train Loss {}: Validation MSE {} : Val Loss {}'.format(epoch+1,loss,mse,val_loss))

        performance[epoch+1] = {
            'Train_loss' : loss.detach().cpu().data,
            'Val_mse_loss' : mse,
            'Val_loss' : val_loss.detach().cpu().data
        }

    torch.save(   {'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss}
            
            ,'runs/VAE_model_DNN_classifier_exp_{}.pt'.format(params['expNumber']))
    
    with open('runs/VAE_model_runs.txt','a') as fobj:
      
        fobj.write("\nmodel_VAE_classifier_exp_{}.pt - Validation MSE {}, Val Loss {}".format(params['expNumber'],mse,val_loss))
        fobj.write("-----*************Params**************-----")
        fobj.write(json.dumps(params))
        fobj.write("-----**************End****************-----")
    pd.DataFrame(performance).to_csv("runs/VAE_model_DNN_classifier_exp_{}.csv".format(params['expNumber']))


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
    train(params)

    pass







