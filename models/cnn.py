import torch
import torch.nn as nn
import ipdb

class CNN(nn.Module):

    def __init__(self, out_dim=10, in_channel=1, img_sz=32, hidden_dim=1000):
        super(CNN, self).__init__()
        self.in_dim = in_channel*img_sz*img_sz
        self.in_channel = in_channel
        
        ### CNN part
        self.layer_size = 32
        self.z_dim = img_sz ** 2
        self.layers = {}
        
        # self.layers[0] = nn.Sequential(
            # nn.Conv2d(in_channel, self.layer_size, kernel_size=5, stride=1, padding=2),
            # nn.ReLU())
        
        # self.layers[1] = nn.Sequential(
            # nn.Conv2d(self.layer_size, self.layer_size, kernel_size=5, stride=1, padding=2),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2))
        
        # self.layers[2] = nn.Sequential(
            # nn.Conv2d(self.layer_size, self.layer_size, kernel_size=5, stride=1, padding=2),
            # nn.ReLU())
        
        # self.layers[3] = nn.Sequential(
            # nn.Conv2d(self.layer_size, self.layer_size, kernel_size=5, stride=1, padding=2),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layers[0]= nn.Sequential(
            nn.Conv2d(in_channel, self.layer_size, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layers[1] = nn.Sequential(
            nn.Conv2d(self.layer_size, self.layer_size, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.embed_layer=  nn.Linear(8 * 8 * self.layer_size, self.z_dim) 
         
        ### Original MLP part
        self.linear = nn.Sequential(
            nn.Linear(self.z_dim, hidden_dim),
            #nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            #nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.last = nn.Linear(hidden_dim, out_dim)  # Subject to be replaced dependent on task

    # def features(self, x):
        # x = self.linear(x.view(-1,self.in_dim))
        # return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        for ldx, layer in self.layers.items():
            layer.cuda()
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.embed_layer(x)
        x = self.linear(x)
        x = self.logits(x)
        return x


class CNN01(nn.Module):
    def __init__(self, out_dim=10, in_channel=1, img_sz=32, hidden_dim=1000):
        super(CNN, self).__init__()
        self.in_dim = in_channel*img_sz*img_sz
        self.in_channel = in_channel
        
        ### CNN part
        self.layer_size = 16
        self.z_dim = img_sz ** 2

        # ipdb.set_trace()
        self.layers = {}
        self.layers[0]= nn.Sequential(
            nn.Conv2d(in_channel, self.layer_size, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layers[1] = nn.Sequential(
            nn.Conv2d(self.layer_size, self.layer_size, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.embed_layer=  nn.Linear(8 * 8 * self.layer_size, self.z_dim) 
         
        ### Original MLP part
        self.linear = nn.Sequential(
            nn.Linear(self.z_dim, hidden_dim),
            #nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            #nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.last = nn.Linear(hidden_dim, out_dim)  # Subject to be replaced dependent on task

    # def features(self, x):
        # x = self.linear(x.view(-1,self.in_dim))
        # return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):

        for ldx, layer in enumerate(self.layers):
            x = layer(x)

        x = x.view(x.size(0), -1)
        x = self.embed_layer(x)
        x = self.linear(x)
        x = self.logits(x)
        return x

def CNN1000_img_sz(pretrained_model_type, in_channel, img_sz):
    return CNN(hidden_dim=1000, in_channel=in_channel, img_sz=img_sz)

def CNN1000_MHA():
    return CNN_MHA(hidden_dim=1000)

def CNN100():
    return CNN(hidden_dim=100)


def CNN400():
    return CNN(hidden_dim=400)


def CNN1000():
    return CNN(hidden_dim=1000)


def CNN2000():
    return CNN(hidden_dim=2000)


def CNN5000():
    return CNN(hidden_dim=5000)

