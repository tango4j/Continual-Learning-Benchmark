''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
import torchvision
import sys
sys.path.append('../.')

import models.cnn as cnn
import models.pretrained as pretrained

__author__ = "Yu-Hsiang Huang"

class MGN(nn.Module):
    def __init__(self, opt, x_dim, h_dim1, h_dim2, z_dim, mgn_model_type="cnn_2layers"):
        super(MGN, self).__init__()
        for key, val in opt.__dict__.items():
            setattr(self, key, val)
        # encoder part
        self.mgn_model_type = mgn_model_type
        self.ch = self.image_shape[0]
        self.imageNet_shape = (224, 224)
        # layer_size = 64
        if self.mgn_model_type=="cnn_2layers":
            self.cnn = cnn.CNN(in_channel=self.image_shape[0], img_sz=self.img_sz)
            # self.layer1 = nn.Sequential(
		# nn.Conv2d(self.ch, layer_size, kernel_size=5, stride=1, padding=2),
		# nn.ReLU(),
		# nn.MaxPool2d(kernel_size=2, stride=2))
            # self.layer2 = nn.Sequential(
		# nn.Conv2d(layer_size, layer_size, kernel_size=5, stride=1, padding=2),
		# nn.ReLU(),
		# nn.MaxPool2d(kernel_size=2, stride=2))
            # self.fc1 =  nn.Linear(8 * 8 * 16, h_dim1) 
            # self.drop_out = nn.Dropout()
            # self.fc2 = nn.Linear(1000, 10) 
            last_cnn_dim = 8*8*self.cnn.layer_size
        
        elif self.mgn_model_type == self.pretrained_model_type:
            img_sz = (3, 32, 32)
            print("Loading pretrained ImageNet model {} ...".format(self.pretrained_model_type))
            self.pretrained = pretrained.PRETRAINED(pretrained_model_type=self.pretrained_model_type,
                                                    in_channel=self.image_shape[0], img_sz=self.img_sz)
            # self.image_size_change = nn.AdaptiveAvgPool2d((224, 224))
            ### This upscales the image
            # self.adap_img = nn.AdaptiveAvgPool2d(self.imageNet_shape)
            last_cnn_dim = 1000

        elif self.mgn_model_type == 'mlp':
            self.net1 = nn.Linear(x_dim, h_dim1)
            last_cnn_dim = h_dim1
        
        else:
            raise ValueError('No such MGN model type such as {}'.format(self.mgn_model_type))

        self.fc31 = nn.Linear(last_cnn_dim, z_dim)
        self.fc32 = nn.Linear(last_cnn_dim, z_dim)
        
        # model_list = [self.net1, self.fc2, self.fc31, self.fc32]
        # model_list = [self.layer1, self.layer2, self.fc31, self.fc32]

        # if opt.orthogonal_init: 
            # print("------====== Orthogonalizing the initial weights")
            # for _model in model_list:
                # torch.nn.init.orthogonal_(_model.weight)

    # def encoder(self, x):
        # with torch.no_grad():
    def encoder(self, x):
        if self.mgn_model_type == 'cnn_2layers':
            x = x.view(-1, *self.image_shape)
                
            for ldx, layer in self.cnn.layers.items():
                layer.cuda()
                x = layer(x)
            h = x.view(x.size(0), -1)
            # out = self.drop_out(out)
            # try:
                # h = self.fc1(out_flat)
            # except:
                # ipdb.set_trace()
        elif self.mgn_model_type == self.pretrained_model_type:
            if self.image_shape[0] == 1:
                x = x.repeat(1, 3, 1, 1)
            x = x.view(-1, 3, *self.image_shape[1:])
            
            ### h is BS x 1000
            h = self.pretrained.pretrained_model(self.pretrained.image_size_change(x))
        else:
            h = F.relu(self.net1(x))
        # h = F.relu(self.fc2(h))
        mu, log_var = self.fc31(h), self.fc32(h) # mu, log_var
        return mu, log_var
    
    def sampling(self, mu, log_var):
        # std = torch.exp(0.5*log_var)
        std = torch.exp(0.5*log_var)
        # ipdb.set_trace()
        if self.scale_std == True:
            std = self.fixed_std * std
        else:
            std = self.fixed_std * torch.ones_like(std)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    # def decoder(self, z):
        # h = F.relu(self.fc4(z))
        # h = F.relu(self.fc5(h))
        # return F.sigmoid(self.fc6(h)) 
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        # return self.decoder(z), mu, log_var
        return z, mu, log_var

class mlp_embeddingExtractor(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, opt):
    # def __init__(self, n_head, d_model, d_k, d_v, compress=False, d_model_embed=1024, dropout=0.1):
        super().__init__()
        for key, val in opt.__dict__.items():
            setattr(self, key, val)

        self.relu01 = nn.ReLU()
        self.net1 = nn.Linear(self.d_model, self.n_head * self.d_k)
        if opt.compress:
            self.fc2 = nn.Linear(self.n_head * self.d_k, self.d_model_embed)
        else:
            self.fc2 = nn.Linear(self.n_head * self.d_k, self.d_model)
        
        if opt.orthogonal_init: 
            print("------====== Orthogonalizing the initial weights")
            for _model in [self.net1, self.fc2]:
                torch.nn.init.orthogonal_(_model.weight)

        self.init_W1 = self.net1.weight
        self.init_W2 = self.fc2.weight
        # ipdb.set_trace()
    def forward(self, _q, k, v,mask=None):
        M = self.net1(_q)
        # M = self.relu01(M)
        if self.mlp_mha == 6:
            with torch.no_grad():
                M = self.fc2(M)
        else:
            M = self.fc2(M)
        return M, None



