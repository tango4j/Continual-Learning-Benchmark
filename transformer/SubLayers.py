''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.Modules import ScaledDotProductAttention
import ipdb

__author__ = "Yu-Hsiang Huang"
class mlp_embeddingExtractor(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, opt):
    # def __init__(self, n_head, d_model, d_k, d_v, compress=False, d_model_embed=1024, dropout=0.1):
        super().__init__()
        for key, val in opt.__dict__.items():
            setattr(self, key, val)

        self.relu01 = nn.ReLU()
        self.fc1 = nn.Linear(self.d_model, self.n_head * self.d_k)
        if opt.compress:
            self.fc2 = nn.Linear(self.n_head * self.d_k, self.d_model_embed)
        else:
            self.fc2 = nn.Linear(self.n_head * self.d_k, self.d_model)
        
        if opt.orthogonal_init: 
            print("------====== Orthogonalizing the initial weights")
            for _model in [self.fc1, self.fc2]:
                torch.nn.init.orthogonal_(_model.weight)
        self.init_W1 = self.fc1.weight
        self.init_W2 = self.fc2.weight
        # ipdb.set_trace()
    def forward(self, _q, k, v,mask=None):
        M = self.fc1(_q)
        # M = self.relu01(M)
        if self.mlp_mha == 6:
            with torch.no_grad():
                M = self.fc2(M)
        else:
            M = self.fc2(M)
        return M, None

class VAE(nn.Module):
    def __init__(self, opt, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()
        for key, val in opt.__dict__.items():
            setattr(self, key, val)
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        # self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim1, z_dim)
        self.fc32 = nn.Linear(h_dim1, z_dim)
        
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
       
        # model_list = [self.fc1, self.fc2, self.fc31, self.fc32]
        model_list = [self.fc1, self.fc31, self.fc32]

        if opt.orthogonal_init: 
            print("------====== Orthogonalizing the initial weights")
            for _model in model_list:
                torch.nn.init.orthogonal_(_model.weight)

    def encoder(self, x):
        with torch.no_grad():
            h = F.relu(self.fc1(x))
            # h = F.relu(self.fc2(h))
            mu, log_var = self.fc31(h), self.fc32(h) # mu, log_var
        return mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        # ipdb.set_trace()
        # std = self.fixed_std * torch.ones_like(std)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h)) 
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        # return self.decoder(z), mu, log_var
        return z, mu, log_var



# class mlp2_MultiHeadAttentionMemory(nn.Module):
    # ''' Multi-Head Attention module '''

    # def __init__(self, n_head, d_model, d_k, d_v, compress=False, d_model_embed=1024, dropout=0.1):
        # super().__init__()
        # self.fc1 = nn.Linear(d_model, n_head * d_k)
        # self.fc2 = nn.Linear(n_head * d_k, d_model)
        # self.fc1_res = nn.Linear(d_model, n_head * d_k)
        # if compress:
            # self.fc2= nn.Linear(n_head * d_k, d_model_embed)
        # else:
            # self.fc2 = nn.Linear(n_head * d_k, d_model)
        # self.relu = nn.ReLU() 
        # # self.n_head = n_head
        # # self.d_k = d_k
        # # self.d_v = d_v
        
        # self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        # self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        # self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        # self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        # self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        # self.dropout = nn.Dropout(dropout)
        # self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    # def forward(self, _q, k, v, mask=None):
        # M = self.fc1(_q)
        # M = self.fc2(M)
        # return M, ''


class MultiHeadAttentionMemory(nn.Module):
    ''' Multi-Head Attention module '''

    # def __init__(self, n_head, d_model, d_k, d_v, compress=False, d_model_embed=1024, mlp_res=False, dropout=0.1):
    def __init__(self, opt):
        super().__init__()
        for key, val in opt.__dict__.items():
            setattr(self, key, val)

        n_head = self.n_head 
        d_k = self.d_k 
        d_v = self.d_v 
        mlp_res = self.mlp_res 
        d_model = self.d_model
        d_model_embed = self.d_model_embed

        self.fc1_res = nn.Linear(d_model, n_head * d_k)
       
        # if mha_mlp == 3:
            # if self.compress:
                # d_model_mlp_out = d_model_embed
            # else:
                # d_model_mlp_out = d_model
        # else:
        if self.compress:
            d_model_mlp_out = d_model_embed
        else:
            d_model_mlp_out = d_model
                
        self.fc2_res = nn.Linear(n_head * d_k, d_model_mlp_out)
        # self.relu1 = nn.ReLU() 
        # self.relu2= nn.ReLU() 
        

        if self.mlp_mha == 3:
            d_model = d_model_mlp_out
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        
        
        if self.compress:
            self.fc = nn.Linear(n_head * d_v, d_model_embed, bias=False)
        else:
            self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        # self.fc = nn.Linear(2* n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        # self.dropout = nn.Dropout(dropout)
        if self.compress and self.mlp_mha <= 2:
            lf.layer_norm = nn.LayerNorm(d_model_embed, eps=1e-6)
        else:
            self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, _q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = _q.size(0), _q.size(1), k.size(1), v.size(1)
        

        if self.mlp_res:
            residual = self.fc1_res(_q)
            residual = self.fc2_res(residual)
        else:
            residual = _q
        
            
        # _q = self.fc2_res(self.fc1_res(_q))
        # k = self.fc2_res(self.fc1_res(k))
        # v = self.fc2_res(self.fc1_res(v))
        # residual = _q


        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        _q = self.w_qs(_q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        
        # qmax = torch.max(torch.abs(_q.view(-1)))
        # print("1 qmax {}".format(qmax))
        
        # Transpose for attention dot product: b x n x lq x dv
        _q, k, v = _q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        # print("After transpose 1,2", _q.shape)
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.
        
        # print("After mask unsqueeze", _q.shape)
        # with torch.no_grad():
        _q, attn = self.attention(_q, k, v, mask=mask)
        # print("After attention", _q.shape)
        
        # qmax = torch.max(torch.abs(_q.view(-1)))
        # print("2 qmax {}".format(qmax))

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        _q = _q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)

        # print("After _q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)", _q.shape)

        # _q = self.dropout(self.fc(_q))
        # mu = 100.0
        # self.mu = 50.0
        # mu = 0.0

        if self.mlp_res:
            if _q.shape != residual.shape:
                _q = _q.unsqueeze(dim=2)
            _q = self.fc(_q)
            qmax = torch.max(torch.abs(_q.view(-1)))
            rmax = torch.max(torch.abs(residual.view(-1)))
            _q = _q/qmax + self.mu * residual/rmax
        else:  
            _q = self.fc(_q)
            if _q.shape != residual.shape:
                _q = _q.unsqueeze(dim=2)
            _q += residual
        
        _q = self.layer_norm(_q)
        return _q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):
        _q = q

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        # ipdb.set_trace()
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x
