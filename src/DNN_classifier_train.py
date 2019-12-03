import torch.optim as optim
from DNN_classifier import DNNclassifier
from __dataset import spcamDataset
from __dataset import SubsetSampler
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from sklearn import metrics
import numpy as np
import json 

def train(params):

    model = DNNclassifier(params)
    model.to(params['device'])

    # data generator
    train_dataset = spcamDataset(params, phase = "train")
    valid_dataset = spcamDataset(params, phase = "validation")

    train_loader = DataLoader(train_dataset, sampler = SubsetSampler(train_dataset.indices))
    valid_loader = DataLoader(valid_dataset, sampler = SubsetSampler(valid_dataset.indices))

    optimizer = optim.Adam(model.parameters(),lr=0.000001,weight_decay=0.001)
    criterion = nn.BCELoss().to(params['device'])

    # training 
    for epoch in range(params['epochs']):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()

            prediction = model(data)
            loss = criterion(prediction, target.squeeze(0).type(torch.FloatTensor).to(params['device']))
            loss.backward()
            optimizer.step()

            if batch_idx > 5000: # batch size 1024
                break



        result_predicted,result_actual = [], []
        
        
        #validation
        model.eval()

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                outputs = model(data)
                val_loss = criterion(outputs, target.squeeze(0).type(torch.FloatTensor).to(params['device']))

                assert val_loss.requires_grad == False
                 
                result_predicted.extend((outputs.cpu().detach().numpy()>=0.5)*1)
                result_actual.extend(target.squeeze(0).cpu().detach().numpy())


                if batch_idx > 100:  # batch size 10240 
                    break

        fpr, tpr, thresholds = metrics.roc_curve(np.array(result_actual), np.array(result_predicted))
        auc_ = metrics.auc(fpr, tpr) 

        print('Epoch {} : Train Loss {}: Validation AUC {} : Val Loss {}'.format(epoch+1,loss,auc_,val_loss))


    torch.save(model,'runs/model_DNN_classifier_exp_{}.pt'.format(params['expNumber']))
    
    with open('runs/model_runs.txt','a') as fobj:
        fobj.write("model_DNN_classifier_exp_{}.pt - Validation AUC {}, Val Loss {}".format(params['expNumber'],auc_,val_loss))
        fobj.write("-----*************Params**************-----")
        fobj.write(json.dumps(params))
        fobj.write("-----**************End****************-----")


if __name__ == "__main__":

    params = {
        'data_fn' : 'data/VerySmallDataset/000_train.nc',
        'input_vars' : ['TBP', 'QBP', 'PS', 'SOLIN', 'SHFLX', 'LHFLX'],
        'output_vars' : ['PRECT'],
        'norm_fn':'data/VerySmallDataset/000_norm.nc',
        'input_transform':("min", "maxrs"),
        'batch_size': 256,
        'shuffle' : True, 
        'var_cut_off': None,
        'model_type':"classification", ## dataset related

        'inputDim' : 64,
        'hiddenDim' : [128,128,128,64],
        'outputDim' : 1, #DNN related


        'epochs' : 70,
        'expNumber' : "Exp01", # to be changed
        'device' : 'cuda',
        'valid_fn' : 'data/VerySmallDataset/000_valid.nc' #Misc
    }
    train(params)

    pass