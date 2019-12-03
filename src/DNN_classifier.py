import torch
import torch.nn as nn
from collections import OrderedDict

if torch.cuda.is_available():
    print("Using GPU")


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






if __name__ == "__main__":

    params = {
        'inputDim' : 64,
        'hiddenDim' : [64,64,64,64],
        'outputDim' : 2
    }
    dnn = DNNclassifier(params)
    print(dnn)
    pass