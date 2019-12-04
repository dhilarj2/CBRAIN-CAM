import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

if torch.cuda.is_available():
    print("Using GPU")

##### - DNN for classification

class DNNclassifier(nn.Module):
    def __init__(self, params):
        super(DNNclassifier, self).__init__()
        self.layer_stack = OrderedDict()
        self.inputDim = params['inputDim']
        ## hidden layers
        self.hidden = params['hiddenDim']
        self.outputDim = params['outputDim']
        self.device = params['device']

        self.layer_stack['hidden_{}'.format(0)] =  nn.Linear(in_features=self.inputDim, out_features=self.hidden[0])
        self.layer_stack['hidden_bn_{}'.format(0)] = nn.BatchNorm1d(num_features = self.hidden[0])
        self.layer_stack['hidden_activation_{}'.format(0)] = nn.RReLU()

        for idx in range(len(self.hidden[1:])):
            self.layer_stack['hidden_{}'.format(idx+1)] = nn.Linear(in_features=self.hidden[idx], out_features=self.hidden[idx+1])
            self.layer_stack['hidden_bn_{}'.format(idx+1)] = nn.BatchNorm1d(self.hidden[idx+1])
            self.layer_stack['hidden_activation_{}'.format(idx+1)] = nn.RReLU()

        self.layer_stack['linear_output'] = nn.Linear(in_features = self.hidden[idx+1],out_features=self.outputDim)
        self.layer_stack['output'] = nn.Sigmoid()
            

        self.model = nn.Sequential(self.layer_stack)


    def forward(self, input):
        return self.model(input.to(self.device).squeeze(0))


########## - VAE

class Encoder(nn.Module):

    def __init__(self, params):
        super(Encoder,self).__init__()
                
        self.layer_stack = OrderedDict()
        self.inputDim = params['inputDim'] + 1 # +1 for precit
        
        ## hidden layers
        self.hidden = params['hiddenDim']
        self.outputDim = params['outputDim'] # dimensions of the latent space
        self.device = params['device']

        self.layer_stack['hidden_{}'.format(0)] =  nn.Linear(in_features=self.inputDim, out_features=self.hidden[0])
        self.layer_stack['hidden_bn_{}'.format(0)] = nn.BatchNorm1d(num_features = self.hidden[0])
        self.layer_stack['hidden_activation_{}'.format(0)] = nn.RReLU()

        for idx in range(len(self.hidden[1:])):
            self.layer_stack['hidden_{}'.format(idx+1)] = nn.Linear(in_features=self.hidden[idx], out_features=self.hidden[idx+1])
            self.layer_stack['hidden_bn_{}'.format(idx+1)] = nn.BatchNorm1d(self.hidden[idx+1])
            self.layer_stack['hidden_activation_{}'.format(idx+1)] = nn.RReLU()

        #self.layer_stack['linear_output'] = nn.Linear(in_features = self.hidden[idx+1],out_features=self.outputDim)
        #self.layer_stack['output'] = nn.Sigmoid()


        self.model = nn.Sequential(self.layer_stack)


        self.mu = nn.Linear(self.hidden[idx+1], self.outputDim)
        self.var = nn.Linear(self.hidden[idx+1], self.outputDim)
    
    def forward(self, X):

        mu = self.mu(self.model(X))
        var = self.var(self.model(X))

        return mu, var ##log_var


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder,self).__init__()

        self.layer_stack = OrderedDict()
        self.inputDim = params['inputDim'] + 1 # +1 for precit

        self.latentDim = params['outputDim']   #(including Prect) ## Latent space dimension z | X(PRECT) , y(input vars)
        self.hidden= params['hiddenDim']
        self.device = params['device']


        self.layer_stack['hidden_{}'.format(0)] =  nn.Linear(in_features=self.latentDim + (self.inputDim - 1), out_features=self.hidden[0])
        self.layer_stack['hidden_bn_{}'.format(0)] = nn.BatchNorm1d(num_features = self.hidden[0])
        self.layer_stack['hidden_activation_{}'.format(0)] = nn.RReLU()


        for idx in range(len(self.hidden[1:])):
            self.layer_stack['hidden_{}'.format(idx+1)] = nn.Linear(in_features=self.hidden[idx], out_features=self.hidden[idx+1])
            self.layer_stack['hidden_bn_{}'.format(idx+1)] = nn.BatchNorm1d(self.hidden[idx+1])
            self.layer_stack['hidden_activation_{}'.format(idx+1)] = nn.RReLU()

        self.layer_stack['mu'] = nn.Linear(self.hidden[idx+1], 1) 
        self.model = nn.Sequential(self.layer_stack)

    def forward(self, X):
        
        return F.torch.sigmoid(self.model(X))



class CVAE(nn.Module):
    def __init__(self, params):
        super(CVAE,self).__init__()

        ## initialize parameters

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)


    def forward(self, x, y): ##  to manage train prect and conditioned on input_vars
        x = torch.cat((x, y), dim=1)

        # encode 
        z_mu, z_var = self.encoder(x) ## z_var : log_var

        # sample from the distribution having latent parameters z_mu, z_var
        # reparameterize
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)

        z = torch.cat((x_sample, y), dim=1)

        return self.decoder(z), z_mu, z_var  ## z_var : log_var


if __name__ == "__main__":

    params = {
        'inputDim' : 64,
        'hiddenDim' : [64,64,64,64],
        'outputDim' : 16,
        'device' : 'cuda'
    }
    #dnn = DNNclassifier(params)
    
    cvae = CVAE(params)
    print(cvae)
    
    pass