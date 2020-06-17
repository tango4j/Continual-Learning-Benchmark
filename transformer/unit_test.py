import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import argparse

from SubLayers import MultiHeadAttention, PositionwiseFeedForward
from SubLayers import MultiHeadAttentionMemory


parser = argparse.ArgumentParser()

parser.add_argument('-epoch', type=int, default=10)
parser.add_argument('-b', '--batch_size', type=int, default=2048)

parser.add_argument('-d_model', type=int, default=28*28)
parser.add_argument('-d_inner_hid', type=int, default=2048)
parser.add_argument('-d_k', type=int, default=64)
parser.add_argument('-d_v', type=int, default=64)

parser.add_argument('-n_head', type=int, default=8)
parser.add_argument('-n_layers', type=int, default=6)
parser.add_argument('-warmup','--n_warmup_steps', type=int, default=4000)
parser.add_argument('-dropout', type=float, default=0.1)


opt = parser.parse_args()
opt.cuda = True

device = torch.device('cuda' if opt.cuda else 'cpu')


slf_attn = MultiHeadAttentionMemory(opt.n_head, opt.d_model, opt.d_k, opt.d_v, dropout=opt.dropout).cuda()

# ebd_input = torch.randn(opt.batch_size, opt.d_mode, opt.n_head, opt.d_kl).cuda()
length_ebd = 1
ebd_input = torch.randn(opt.batch_size, length_ebd, opt.d_model).cuda()

repeat = 10
for _ in range(repeat):
    ebd_input = slf_attn(ebd_input, ebd_input, ebd_input)
# class EncoderLayer(nn.Module):
    # ''' Compose with two layers '''

    # def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        # super(EncoderLayer, self).__init__()
        # self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        # self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    # def forward(self, enc_input, slf_attn_mask=None):
        # enc_output, enc_slf_attn = self.slf_attn(
            # enc_input, enc_input, enc_input, mask=slf_attn_mask)
        # enc_output = self.pos_ffn(enc_output)
        # return enc_output, enc_slf_attn


########## DEBUG: unit test code ##########
# input_size = 44
input_size = 4
seq_length = 7
# batch_size = 32
batch_size = 5
### self.mem_size = self.head_size * self.num_heads
# model = RelationalMemory(mem_slots=10, head_size=20, input_size=input_size, num_tokens=66, num_heads=8, num_blocks=1, forget_bias=1., input_bias=0.)

### memory_size = head_size * input_size 
# model = RelationalMemory(mem_slots=3, head_size=6, input_size=input_size, num_heads=8, num_blocks=8, forget_bias=1., input_bias=0.)
# # model = RelationalMemory(mem_slots=10, head_size=20, input_size=input_size, num_heads=8, num_blocks=1, forget_bias=1., input_bias=0., num_tokens=10)

# model = model.cuda()

# model_memory = model.initial_state(batch_size=batch_size).cuda()

# # random input
# random_input = torch.randn((batch_size, seq_length, input_size))
# # random targets
# random_targets = torch.randn((batch_size, seq_length, input_size))

# random_input, random_targets = random_input.cuda(), random_targets.cuda()

# # take a one step forward
# # logit, next_memory = model(random_input, model_memory, random_targets, treat_input_as_matrix=True)
# # logit, next_memory = model(random_input, model_memory, random_targets) 
# # logit, next_memory = model(random_input, model_memory)
# # print("memory 1 :", next_memory)
# next_memory = model_memory
# for i in range(10):
    # logit, next_memory = model(random_input, next_memory)
    # print("memory:", next_memory[0][0][:8])


