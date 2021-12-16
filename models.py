

import torch.nn as nn
import torch.nn.functional as F
#from torchsummary import summary
#import torch.nn.init as weight_init




class AutoencoderTwoLayers(nn.Module):
    def __init__(self, input_size, hidden_size, more_hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.soft1 = nn.Softplus(hidden_size)
        self.fc2 = nn.Linear(hidden_size, more_hidden_size)
        self.soft2 = nn.Softplus(more_hidden_size)
        self.fc3 = nn.Linear(more_hidden_size, hidden_size)
        self.soft3 = nn.Softplus(hidden_size)
        self.fc4 = nn.Linear(hidden_size, input_size)
        self.soft4 = nn.Softplus(input_size)
        #print("self.fc1.weight in model",self.fc1.weight)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.xavier_normal_(self.fc4.weight)

    def forward(self, x):
        # forward always defines connectivity
        representation = self.soft2(self.fc2(self.soft1(self.fc1(x))))
        output = self.soft4(self.fc4(self.soft3(self.fc3(representation))))
        return representation.detach(), output


class AutoencoderTwoLayers_encoder(nn.Module):
    def __init__(self, input_size, hidden_size, more_hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.soft1 = nn.Softplus(hidden_size)
        self.fc2 = nn.Linear(hidden_size, more_hidden_size)
        self.soft2 = nn.Softplus(more_hidden_size)

    def forward(self, x):
        representation = self.soft2(self.fc2(self.soft1(self.fc1(x))))
        return representation


class AutoencoderTwoLayers_encoder_w_softmax(nn.Module):
    def __init__(self, input_size, hidden_size, more_hidden_size, dropout):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.soft1 = nn.Softplus(hidden_size)
        self.fc2 = nn.Linear(hidden_size, more_hidden_size)
        self.soft2 = nn.Softplus(more_hidden_size)
        self.drop = nn.Dropout(dropout)
        self.fcnew = nn.Linear(more_hidden_size, 1)
        nn.init.xavier_normal_(self.fcnew.weight)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        out = self.sigm(self.fcnew(self.soft2(self.fc2(self.soft1(self.fc1(x))))))
        return out


class TwoLayerMLP_dropout(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        # self.drop1 = nn.Dropout(0.8)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.soft1 = nn.Softplus(hidden_size)
        self.drop2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigm = nn.Sigmoid()

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        out = self.sigm(self.fc2(self.drop2(self.soft1(self.fc1(x)))))
        return out
