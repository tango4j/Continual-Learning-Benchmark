import torch
import torch.nn as nn
import ipdb
import torchvision

class PRETRAINED(nn.Module):

    def __init__(self, pretrained_model_type=None, frozen=False, out_dim=10, in_channel=1, img_sz=32, hidden_dim=256):
        super(PRETRAINED, self).__init__()
        self.in_dim = in_channel*img_sz*img_sz
        self.in_channel = in_channel
        self.pretrained_model_type = pretrained_model_type 
        self.frozen = frozen
        ### PRETRAINED part
        layer1_N = 16
        layer2_N = 16
        self.z_dim = img_sz ** 2
        
        ### We should use adaptive Ave. Pooling layer for changing the image size.
        self.image_size_change = nn.AdaptiveAvgPool2d((224, 224))
        self.pretrained_model = getattr(torchvision.models, self.pretrained_model_type)(pretrained=True)
        
        imagenet_out_dim = 1000 
        self.embed_layer=  nn.Linear(imagenet_out_dim, self.z_dim) 
         
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
        if self.in_channel == 1:
            x = x.repeat(1, 3, 1, 1)
        # x = self.pretrained_model(x)
        if self.frozen:
            with torch.no_grad():
                x = self.pretrained_model(self.image_size_change(x))
        else:
            x = self.pretrained_model(self.image_size_change(x))
            
        x = self.embed_layer(x)
        x = self.linear(x)
        x = self.logits(x)
        return x

def PRETRAINED1000_img_sz(pretrained_model_type, in_channel, img_sz):
    return PRETRAINED(pretrained_model_type=pretrained_model_type, frozen=False, hidden_dim=1000, in_channel=in_channel, img_sz=img_sz)

def PRETRAINED_FROZEN_1000_img_sz(pretrained_model_type, in_channel, img_sz):
    return PRETRAINED(pretrained_model_type=pretrained_model_type, frozen=True, hidden_dim=1000, in_channel=in_channel, img_sz=img_sz)

def PRETRAINED1000_MHA():
    return PRETRAINED_MHA(hidden_dim=1000)

def PRETRAINED100():
    return PRETRAINED(hidden_dim=100)


def PRETRAINED400():
    return PRETRAINED(hidden_dim=400)


def PRETRAINED1000():
    return PRETRAINED(hidden_dim=1000)


def PRETRAINED2000():
    return PRETRAINED(hidden_dim=2000)


def PRETRAINED5000():
    return PRETRAINED(hidden_dim=5000)


