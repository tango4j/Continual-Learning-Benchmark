import torch
import torch.nn.functional as F
import numpy as np
import random
from importlib import import_module
from .default import NormalNN
from .default import *
from .regularization import SI, L2, EWC, MAS
from dataloaders.wrapper import Storage
from collections import Counter
from scipy import stats

import dataloaders
from dataloaders.datasetGen import SplitGen, PermutedGen
import torchvision
import ipdb
from copy import deepcopy
from utils.metric import accuracy, AverageMeter, Timer
from agents.model_gen_nets import mlp_embeddingExtractor, MGN
# from transformer.SubLayers import mlp_embeddingExtractor
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans

from sklearn.metrics.pairwise import euclidean_distances
from collections import Counter

from scipy.stats import entropy
from torch.nn import Softmax


class maxPoolFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        maxpool1d = nn.MaxPool1d(mp, stride=mp)
        self.w_2 = nn.Linear(d_hid//mp, d_hid//mp) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        mu = 0.1
        x += mu * residual
        x = self.layer_norm(x)
        return x

class PriorDist:
    def __init__(self, count_np, seed=0):
        '''
        args:
            count_list: <list> with <int? in it. the count of each embedding [12, 9, 1, 35 ....]
            total_samples: <int> value. Total number of samples.
        '''
        total_sample_count = np.sum(count_np)
        prob_tup = tuple(count_np/total_sample_count)
        # print("probability tuple: ", prob_tup)
        xk = np.arange(len(prob_tup))
        self.prior = stats.rv_discrete(name='prior', values=(xk, prob_tup), seed=seed) 
    
    def getPriorIndex(self, index_size):
        return self.prior.rvs(size=index_size)

class OPT:
    def __init__(self, agent_config, raw_n_image_pixels):
        # self.n_head = 8
        for key, val in agent_config.items():
            setattr(self, key, val)
        
        self.n_head = agent_config['n_head']
        # self.d_model = raw_n_image_pixels
        self.d_model = raw_n_image_pixels
        self.dropout = 0.1
        # self.d_k = 64
        # self.d_v = 64
        self.d_k = agent_config['d_k']
        self.d_v = agent_config['d_v']
        self.max_pool_size = agent_config['max_pool_size']
        self.img_sz = agent_config['img_sz']
        self.dropout = True

        self.mlp_res = False
        if agent_config['compress_data'] == 1:
            ### This controls the embedding size
            self.d_model_embed=agent_config['img_sz']**2
            self.compress=True
        else:
            self.d_model_embed=opt.d_model
            self.compress=False
        
class MHA(nn.Module):
    def __init__(self, agent_config, raw_n_image_pixels): 
        super().__init__()
        opt = OPT(agent_config, raw_n_image_pixels=raw_n_image_pixels)  ### Parameter Class
        self.agent_config = agent_config 
        self.pretrained_model_type = opt.pretrained_model_type        
        
        ### Multi-head attention for Memory 
        if self.agent_config['mlp_mha'] in [7,8,9,10,11]:
            pretrained_model_type='mlp'
            if self.agent_config['mlp_mha'] in [10]:
                pretrained_model_type = 'cnn_2layers'
            elif self.agent_config['mlp_mha'] in [11]:
                pretrained_model_type = self.pretrained_model_type

            self.model_gen_net1 = MGN(opt, x_dim=1024, h_dim1= 320, h_dim2=256, z_dim=self.agent_config['img_sz']**2, mgn_model_type=pretrained_model_type)
            # self.model_gen_net2 = MGN(opt, x_dim=1024, h_dim1= 320, h_dim2=256, z_dim=self.agent_config['img_sz']**2, mgn_model_type=pretrained_model_type)
        mp = opt.max_pool_size
        self.maxpool1d = nn.MaxPool1d(mp, stride=mp)
        
        # self.fc_dim_control= nn.Linear(raw_n_image_pixels, opt.d_model, bias=False).cuda()
    
    def forward(self, M):
        # M,     attn = self.model_gen_net1(M, M, M)
        # M = self.fc_dim_control(M)
        NL = self.agent_config['num_mha_layers']

        if self.agent_config['mlp_mha'] in [7,8,9,10,11]:
            M, mu, log_var = self.model_gen_net1(M)
            attn = (mu, log_var)
        
        else:
            if self.agent_config['mlp_mha'] in [2,3]:
                if self.agent_config['mlp_mha'] == 2:
                    with torch.no_grad():
                        M = self.mlp_embed(M)
                elif self.agent_config['mlp_mha'] == 3:
                    M = self.mlp_embed(M)
                
            if NL == 1:
                M, attn = self.model_gen_net1(M, M, M)
            elif NL == 2:
                M, attn = self.model_gen_net1(M, M, M)
                M, attn = self.model_gen_net2(M, M, M)
            elif NL == 3:
                M, attn = self.model_gen_net1(M, M, M)
                M, attn = self.model_gen_net2(M, M, M)
                M, attn = self.model_gen_net3(M, M, M)
            else:
                raise ValueError('Layer number {} is not implemented.'.format(NL))


        ### After max-pool
        ### 1 x lq x ch x image_vector_size
        # M_out_TR = M_out.transpose(1, 3)
        # M_pooled = self.maxpool1d(M_out_TR.squeeze(dim=0))
        # M_pooled_batch = M_pooled.unsqueeze(dim=0)
        # M_pooled = M_pooled_batch.transpose(1, 3)
        # return M_pooled, attn2

        return M, attn
class Memory(Storage):
    def reduce(self, m):
        self.storage = self.storage[:m]

class Naive_Rehearsal(NormalNN):

    def __init__(self, agent_config):
        super(Naive_Rehearsal, self).__init__(agent_config)
        self.task_count = 0
        self.memory_size = 1000 ### Default Memory Size
        self.task_memory = {}

    def learn_batch(self, train_loader, val_loader=None):
        # 1.Combine training set
        dataset_list = []

        ### self.task_memory is dict()
        for storage in self.task_memory.values():
            ### strorgae -> list and each element is ([1, 32, 32], int, str)
            dataset_list.append(storage)
        dll = len(dataset_list) 
        ### self.memory_size affects training data here.

        ### We assume memory size is full 
        dataset_list *= max(len(train_loader.dataset)//self.memory_size,1)  # Let old data: new data = 1:1 ### Not exactly 1:1
        dataset_list.append(train_loader.dataset)

        ### Wrap mixed dataset (dataset_list).
            # pdb.set_trace()
        dataset = torch.utils.data.ConcatDataset(dataset_list)

        new_train_loader = torch.utils.data.DataLoader(dataset,
                                                       batch_size=train_loader.batch_size,
                                                       shuffle=True,
                                                       num_workers=train_loader.num_workers)

        # 2.Update model as normal
        super(Naive_Rehearsal, self).learn_batch(new_train_loader, val_loader)

        # 3.Randomly decide the images to stay in the memory
        self.task_count += 1
       
        # (a) Decide the number of samples for being saved

        ### Let memory size M = self.memory_size 
        ### the first task, M, the second M/2, the third M/3... Memory Diverges - cf) \Sigma_{1}^{\infinity} 1/n
        num_sample_per_task = self.memory_size // self.task_count
        num_sample_per_task = min(len(train_loader.dataset),num_sample_per_task)
       
       
        # (b) Reduce current exemplar set to reserve the space for the new dataset
        for storage in self.task_memory.values():
            storage.reduce(num_sample_per_task)
        
        # (c) Randomly choose some samples from new task and save them to the memory
        self.task_memory[self.task_count] = Memory()  # Initialize the memory slot
        randind = torch.randperm(len(train_loader.dataset))[:num_sample_per_task]  # randomly sample some data
        for ind in randind:  # save it to the memory
            self.task_memory[self.task_count].append(train_loader.dataset[ind])


class Fed_Memory_Rehearsal(NormalNN):
    def __init__(self, agent_config, method):
        super(Fed_Memory_Rehearsal, self).__init__(agent_config)
        self.method = method 
        self.agent_config = agent_config
        self.task_count = 0
        self.task_memory = {}
        self.memory_size = 0 
        ### +--------------------------------
        ### | Train data specifications
        ### +--------------------------------
        self.ch = self.agent_config['image_shape'][0]
        self.img_sz = self.agent_config['img_sz'] 
        self.feat_length = self.agent_config['img_sz']**2
        self.raw_input_image_shape = self.agent_config['image_shape']
        self.raw_n_image_pixels = self.raw_input_image_shape[1] ** 2
        # self.model_input_image_shape = (self.ch, 32, 32)
        self.model_input_image_shape = (self.ch, self.img_sz, self.img_sz)

        self.task_numbers = 100
        self.n_max_label = 100
        self.ls = 1000

        ### +--------------------------------
        ### | Model Specification
        ### +--------------------------------
        self.dict_task_models = {}
        self.MHA_model = MHA(self.agent_config, self.raw_n_image_pixels).cuda()
        self.MHA_params = self.MHA_model.parameters()
        # self.mha_optimizer = torch.optim.Adam(self.MHA_params, lr=0.001)
        self.log("-------------> self.MHA_model structure:")
        self.log(str(self.MHA_model))
        self.memory_init_optimizer()
        
        # self.mha_optimizer = torch.optim.SGD(MHA_params, lr=0.0001)
        self.neural_task_memory = {str(x+1): { y: {z: [] for z in range(self.ls)} for y in range(self.n_max_label)} for x in range(self.task_numbers)}
        # self.neural_task_memory = {str(x+1): [] for x in range(self.task_numbers)}
        self.noise_test = False
   

    def perform_pretrain_MGN(self, dataset='EMNIST'):
        pre_train_dataset, pre_val_dataset= dataloaders.base.EMNIST('data', emnist_split='digits')
        
        batch_size = self.agent_config['batch_size']
        
        train_dataset_splits, val_dataset_splits, task_output_space = SplitGen(pre_train_dataset, pre_val_dataset,
                                                                          first_split_sz=10,
                                                                          other_split_sz=10,
                                                                          rand_split=False,
                                                                          remap_class=True)
        pre_train_loader = torch.utils.data.DataLoader(train_dataset_splits['1'],
                                                       batch_size=batch_size,
                                                       shuffle=True,
                                                       num_workers=1)
        pre_val_loader = torch.utils.data.DataLoader(val_dataset_splits['1'],
                                                       batch_size=batch_size,
                                                       shuffle=True,
                                                       num_workers=1)
        self.memory_learn_batch(pre_train_loader, pre_val_loader, pretraining=True)

    def p_log(self, *args):
        # print("=-===1=24=124=12=4=124 LOG")
        self.pI.a_print(args)
        # print(*args)

    def learn_batch(self, train_loader, val_loader=None):
        if self.task_count == 0:
            # self.memory_size = self.memory_size * (self.raw_n_image_pixels//(self.img_sz**2))
            if self.method == "Model_Generating_Rehearsal":
                self.image_shape = self.agent_config['image_shape']
                full_image_size = self.image_shape[0] * self.image_shape[1] ** 2
                ### If you want to use the same amount of memory with Naive rehearsal.
                self.memory_size = self.memory_size * (full_image_size // (2 *self.img_sz ** 2))
            else:
                self.memory_size = self.memory_size

        self.log("---------========== MEMORY SIZE : ", self.memory_size)
        self.log("[method] {}".format(self.method))
        # 1.Combine training set
        dataset_list = []

        ### self.task_memory is dict()
        active_memory_size = 0
        for storage in self.task_memory.values():
            ### strorgae -> list and each element is ([1, 32, 32], int, str)
            dataset_list.append(storage)
            active_memory_size += len(storage)
        self.log("=========== Actual active memory size : ", active_memory_size)
        dll = len(dataset_list) 
        
        if 'Model_Generating_Rehearsal' in self.agent_config['agent_name'] and self.train_info[0] > 0:
            self.replay_memory = torch.utils.data.ConcatDataset(dataset_list)

            dataset_list = [train_loader.dataset]
            pass
        ### self.memory_size affects training data here.
        else:
            ### We assume memory size is full 
            dataset_list *= max(len(train_loader.dataset)//self.memory_size,1)  # Let old data: new data = 1:1 ### Not exactly 1:1
            if self.method == "No_Rehearsal":
                dataset_list = []
        
            dataset_list.append(train_loader.dataset)

        ### Wrap mixed dataset (dataset_list).
        dataset = torch.utils.data.ConcatDataset(dataset_list)

        new_train_loader = torch.utils.data.DataLoader(dataset,
                                                       batch_size=train_loader.batch_size,
                                                       shuffle=True,
                                                       num_workers=train_loader.num_workers)

        # 2.Update model as normal
        # super(Fed_Memory_Rehearsal, self).learn_batch(new_train_loader, val_loader)
        self.log("--------======== boost_scale :", self.agent_config['boost_scale'])

        if self.agent_config['pretrain'] in [1] and self.train_info[0] == 0:
            if self.agent_config['dataset'] == 'EMNIST':
                self.perform_pretrain_MGN() 
             

        self.memory_learn_batch(new_train_loader, val_loader, pretraining=False)

        if self.agent_config['do_task1_training'] in [2]:
            # ipdb.set_trace()
            self.dict_task_models[self.train_info[0]] = deepcopy(self.MHA_model)

        # 3.Randomly decide the images to stay in the memory
        self.task_count += 1
       
        # (a) Decide the number of samples for being saved

        ### Let memory size M = self.memory_size 
        ### the first task, M, the second M/2, the third M/3... Memory Diverges - cf) \Sigma_{1}^{\infinity} 1/n
        num_sample_per_task = self.memory_size // self.task_count
        num_sample_per_task = min(len(train_loader.dataset),num_sample_per_task)
        # ipdb.set_trace() 
        torch.cuda.empty_cache() 
        # (b) Reduce current exemplar set to reserve the space for the new dataset
        if self.method == "Model_Generating_Rehearsal" and self.agent_config['clustering_type'] in [1,2]:
            for key in self.task_memory.keys():
                # storage.reduce(num_sample_per_task)
                storage = self.task_memory[key]
                self.task_memory[key] = self.getReducedMemory(storage, num_sample_per_task)
        else:
            for storage in self.task_memory.values():
                storage.reduce(num_sample_per_task)

        self.task_memory[self.task_count] = Memory()  # Initialize the memory slot
        self.log("============================  METHOD : {}".format(self.method))
        if self.method == "Naive_Rehearsal":
            # (c) Randomly choose some samples from new task and save them to the memory
            randind = torch.randperm(len(train_loader.dataset))[:num_sample_per_task]  # randomly sample some data
            for ind in randind:  # save it to the memory
                self.task_memory[self.task_count].append(train_loader.dataset[ind])
        
        elif self.method == "Noise_Rehearsal":
            self.reqGradToggle(False)  ### Freeze the model
            embed_memory_data = self.saveEmbeddingMemory(train_loader, num_sample_per_task, noise_test=True)
            self.reqGradToggle(True)  ### Freeze the model
            randind = torch.randperm(len(embed_memory_data))[:num_sample_per_task]  # randomly sample some data
            for ind in randind:  # save it to the memory
                self.task_memory[self.task_count].append(embed_memory_data[ind])

        elif self.method == "No_Rehearsal":
            pass 

        elif self.method == "Model_Generating_Rehearsal":
            if self.agent_config['clustering_type'] in [0]:
                embed_memory_data = self.saveEmbeddingMemory(train_loader, num_sample_per_task, noise_test=False)
                randind = torch.randperm(len(embed_memory_data))[:num_sample_per_task]  # randomly sample some data
                for ind in randind:  # save it to the memory
                    self.task_memory[self.task_count].append(embed_memory_data[ind])
            
            elif self.agent_config['clustering_type'] in [1,2]:
                # self.reqGradToggle(False)  ### Freeze the model
                fed_memory_data, count_list = self.getCompressedMemory(train_loader, num_sample_per_task, noise_test=False)
                self.reqGradToggle(True)  ### Freeze the model
                # for ind in range(len(fed_memory_data)):  # save it to the memory
                big2small_idx = np.argsort(count_list)[::-1][:num_sample_per_task]
                small2big_idx = np.argsort(count_list)[::-1][:num_sample_per_task]
                # for ind in big2small_idx:
                random_big2small_idx = np.random.permutation(np.argsort(count_list)[::-1])[:num_sample_per_task]
                for ind in np.random.permutation(random_big2small_idx):
                # for ind in small2big_idx:
                    memory_tup = (*fed_memory_data[ind], count_list[ind])
                    self.task_memory[self.task_count].append(memory_tup)
            else:
                raise ValueError('self.clustering_type {} is not implemented.'.format(self.agent_config['clustering_type']) )
            # for ind in randind:  # save it to the memory
        
        elif self.method == "Memory_Embedding_Rehearsal":
            self.reqGradToggle(False)  ### Freeze the model
            embed_memory_data = self.saveEmbeddingMemory(train_loader, num_sample_per_task, noise_test=False)
            self.reqGradToggle(True)  ### Freeze the model
            randind = torch.randperm(len(embed_memory_data))[:num_sample_per_task]  # randomly sample some data
            for ind in randind:  # save it to the memory
                self.task_memory[self.task_count].append(embed_memory_data[ind])
        
        else:
            raise ValueError('method name {} does not exist.'.format(self.method))
        # ipdb.set_trace()

    def reqGradToggle(self, _bool):
        for layer in self.model.parameters():
            layer.requires_grad = _bool
        self.model.train(_bool)
    
    def vae_loss_function(self, recon_x, x_vec,  mu, log_var):
        BCE = F.binary_cross_entropy(recon_x, x_vec, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE, KLD

    def kld_loss_function(self,  mu, log_var, fixed_std):
        # BCE = F.binary_cross_entropy(recon_x, x_vec, reduction='sum')
        # KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        log_fixed_var = np.log(fixed_std ** 2)
        log_fixed_var = log_fixed_var * torch.ones_like(log_var)
        # ipdb.set_trace()
        KLD = -0.5 * torch.sum(1 + log_var - log_fixed_var - mu.pow(2) - log_var.exp()/log_fixed_var.exp())
        return KLD

    
    def aggregatedClassLoss(self, preds, target):
        if isinstance(self.valid_out_dim, int):  
            pred = preds['All'][:,:self.valid_out_dim]
        else:
            if type(preds) == type(dict()) and 'All' in preds.keys(): # Single-headed model
                pred = preds['All']
            else:
                pred = preds
        loss = self.criterion_fn(pred, target)
        return loss 
    
    def saveAsTuple(self, M_imaged, target, task, fed_memory_data):
        M = M_imaged.squeeze(0)

        ### For the conveience of loading the data, zeropad the embedding output.
        if self.agent_config['compress_data'] == 1:
            if self.agent_config['mlp_mha'] in [8,9,10,11]:
                M = self.addZeroPadding(M, isStat=True)
            else:
                M = self.addZeroPadding(M, isStat=False)
        
        for k in range(M.shape[0]):
            if self.noise_test == True:
                # M_tup = ( torch.randn(self.raw_input_image_shape), target[k].detach().cpu().numpy().item(), task[k])
                M_tup = ( torch.randn(self.raw_input_image_shape), target[k], task[k])
            else:
                assert torch.sum(M[0,0] != 0).item() != self.img_sz ** 2, "non-zero values should be 2x  self.img_sz**2 for stats"
                M_memory = M[k].view(1, self.raw_input_image_shape[1], self.raw_input_image_shape[2]).detach().cpu()
                if self.ch != 1:
                    M_memory = M_memory.repeat(self.ch, 1, 1)

                M_memory.requires_grad = False
                M_tup = (M_memory, target[k].detach().cpu().numpy().item(), task[k])
            fed_memory_data.append(M_tup)

        return fed_memory_data
    
    def intraClassPredict(self, input, target, task, isTraining=False, batch_idx=None):
        org_shape = input[0, :].shape

        embed_memory_data = []
        data_count_check = {"old": 0, "new": 0, "org": 0}
        data_count_check['org'] += len(task)
        
        if self.agent_config['compress_data'] == 1:
            if self.agent_config['mlp_mha'] in [8,9,10,11]:
                new_input = torch.zeros(*input.shape[:2], 2*self.img_sz, self.img_sz)
            else:
                new_input = torch.zeros(*input.shape[:2], self.img_sz, self.img_sz)
        else:
            if self.agent_config['mlp_mha'] in [8,9,10,11]:
                new_input = torch.zeros(*input.shape[:2], self.ch, 2*self.raw_input_image_shape[1], self.raw_input_image_shape[2])
            else:
                new_input = torch.zeros(*input.shape[:2], *self.raw_input_image_shape)

        new_target= torch.zeros(*target.shape).long()

        target_np = target.detach().cpu().numpy()
        task_np = np.array(task)
        current_task_str = self.train_info[1]
        # print("------ tasks {} current_task_str {} ".format(set(task), current_task_str) )

        input_list = []
        task_list = []
        target_list = []

        new_task_bool = torch.tensor(np.array(task) == current_task_str)
        old_task_bool = torch.tensor(np.array(task) != current_task_str)

        # new_input_tensor = input[new_task_bool]
        new_target_tensor = target[new_task_bool]
        new_task_list = tuple( task_np[new_task_bool.cpu().numpy()].tolist() )
        
        old_input_tensor = input[old_task_bool]
        old_target_tensor = target[old_task_bool]
        old_task_list = tuple( task_np[old_task_bool.cpu().numpy()].tolist() )

        assert (len(new_task_list) + len(old_task_list)) == len(task), "[ERROR] Task count is somewhat off."
        data_count_check['new'] += len(new_task_list)
        data_count_check['old'] += len(old_task_list)
        
        ### First go through new task samples (Needs to be converted)
        new_target_labels = sorted(set(new_target_tensor.cpu().numpy()))
       
        ### These are real images, not representations.
        for idx, label in enumerate(new_target_labels):
            task_idx = task[idx]
            global_label_bool_np = (target_np == label) * (new_task_bool.numpy())
            global_label_bool_tensor = torch.tensor(global_label_bool_np)
            
            this_new_target = target[global_label_bool_tensor]

            M_before_integ = input[global_label_bool_tensor, :].view(-1, self.ch, self.raw_n_image_pixels)
            # M_in = M_before_integ.unsqueeze(dim=0)  ### Batch --> input_length
            M_in = M_before_integ.cuda()
            
            ### +-------------------------------------------------
            ### | New Images
            ### +-------------------------------------------------
            
            ### Do MHA process + max_pool
            try:
                M, attention = self.MHA_model(M_in)
            except:
                ipdb.set_trace()
            if self.agent_config['mlp_mha'] in [8,9,10,11]:
                mu, log_var = attention
                try:
                    M = torch.cat((mu, log_var), dim=1)
                except:
                    ipdb.set_trace()
                # if self.agent_config['mlp_mha'] in [9]:
                    # M = mu
                # ipdb.set_trace()

            ### Before max_pool
            if self.agent_config['compress_data'] == 1:
                if self.agent_config['mlp_mha'] in [8,9,10,11]:
                    # M_imaged = M.view(-1, self.ch, 2*self.img_sz, self.img_sz)
                    M_imaged = M.view(-1, 1, 2*self.img_sz, self.img_sz)
                else:
                    M_imaged = M.view(-1, self.ch, self.img_sz, self.img_sz)
            else:
                if self.agent_config['mlp_mha'] in [8,9,10,11]:
                    # M_imaged = M.view(-1, self.ch, 2*self.raw_input_image_shape[1], self.raw_input_image_shape[2])
                    M_imaged = M.view(-1, 1, 2*self.raw_input_image_shape[1], self.raw_input_image_shape[2])
                else:
                    M_imaged = M.view(-1, *self.raw_input_image_shape)
          
            input_list.append(M_imaged)
            target_list.append(this_new_target)
           

        old_input = torch.randn(1)
        old_target= torch.randn(1)

        new_input = torch.cat(input_list, dim=0).cuda()
        new_target = torch.cat(target_list, dim=0).cuda()

        if self.agent_config['mlp_mha'] in [8,9,10,11]:
            if not isTraining:  ### That means we are saving the embedding for the future use.
                TS = torch.sum(new_input[0,0] != 0).item()
                assert TS != self.img_sz ** 2, "Should have 2x embedding size not {}".format(TS)
                return new_input, new_target
            
            elif isTraining:  ### This means we are in the midst of training.
                mu = new_input[:, :, :self.img_sz, :].view(-1, 1, self.img_sz**2)
                log_var = new_input[:, :, self.img_sz:, :].view(-1, 1, self.img_sz**2)
                new_input = self.MHA_model.model_gen_net1.sampling(mu, log_var)
                if self.agent_config['mlp_mha'] in [9]:
                    new_input = mu
                regular_new_input = new_input.detach().cpu()
                boost_scale = self.agent_config['boost_scale']
               
                
                ### +-------------------------------------------------
                ### | Boost memory
                ### +-------------------------------------------------
                # if boost_scale  > 0 and self.train_info[0] > 0 and len(old_input.shape) > 1:
                if boost_scale  > 0 and self.train_info[0] > 0:
                    ### Get the same amount of previous memory as the input data.
                    # replay_list = random.choices(self.replay_memory, k=new_input.shape[0])
                    boosted_list = [new_input] ### We add memory data on top of newly obtained inputs
                    boosted_target = [new_target]
    
                    stt, end = self.getStartEnd(batch_idx, mem_length=len(self.replay_memory), new_input_len=new_input.shape[0])
                    try:
                        replay_list = [self.replay_memory[k] for k in range(stt, end)]
                    except:
                        ipdb.set_trace()
                    replay_loader = torch.utils.data.DataLoader(replay_list,
                                                               batch_size=new_input.shape[0],
                                                               shuffle=False)
                    if self.agent_config['clustering_type'] in [0]:
                        pool_input, pool_target, pool_task = [x for x in replay_loader][0]
                    elif self.agent_config['clustering_type'] in [1,2]:
                        pool_input, pool_target, pool_task, pool_count = [x for x in replay_loader][0]
                        # prior = PriorDist(pool_count.cpu().numpy(), seed=0)
                        # idxS = prior.getPriorIndex(new_input.shape[0])
                    
                    
                    for bdx in range(boost_scale):
                        if self.agent_config['clustering_type'] in [0,1,2]:
                            old_input, old_target = pool_input, pool_target
                        # elif self.agent_config['clustering_type'] in [1]:
                            # old_input, old_target = pool_input[idxS, :], pool_target[idxS]
                        old_input, old_target = old_input.cuda(), old_target.cuda()
                        old_input = self.removeZeroPadding(old_input, isStat=True)

                        mu_old = old_input[:, :, :self.img_sz, :].view(-1, 1, self.img_sz**2)
                        log_var_old = old_input[:, :, self.img_sz:, :].view(-1, 1, self.img_sz**2)

                        old_input = self.MHA_model.model_gen_net1.sampling(mu_old, log_var_old)
                        if self.agent_config['mlp_mha'] in [9]:
                            old_input = mu
                        boosted_list.append(old_input)
                        boosted_target.append(old_target)
                    # ipdb.set_trace() 
                    # old_target_repeated = old_target.repeat((boost_scale))
                    new_input = torch.cat(boosted_list, dim=0)
                    new_target = torch.cat(boosted_target, dim=0)
                    # new_target = torch.cat([new_target, old_target_repeated], dim=0)
                    
                ### Make it image shape     
                new_input = new_input.view(-1, 1, self.img_sz, self.img_sz)

                # assert (data_count_check['old'] + data_count_check['new']) == data_count_check['org'], "[ERROR] Sample count is somewhat off."
                # assert torch.all(target.cpu() == new_target).item() == 1, "new_target and target are different. Abort."
                # print("===== Data quantity check passed! :", data_count_check)
                assert torch.sum(new_input[0,0] != 0).item() == self.img_sz ** 2, "This should be 1x embedding size."
                # print("Before output", type(new_target))                
                return new_input, new_target
      
        else:
            ### Before max-pooling 
            assert (data_count_check['old'] + data_count_check['new']) == data_count_check['org'], "[ERROR] Sample count is somewhat off."
            # print("===== Data quantity check passed! :", data_count_check)
            assert torch.all(target.cpu() == new_target).item() == 1, "new_target and target are different. Abort."

            return new_input, new_target

    def getStartEnd(self, batch_idx, mem_length=None, new_input_len=None):
        loop_num = max(mem_length//new_input_len, 1)
        idx_rep = batch_idx % loop_num
        stt = (idx_rep)*new_input_len
        end = (idx_rep+1)*new_input_len
        assert stt >=0 and end <= mem_length, "end length is wrong."
        return stt, end
        

    def removeZeroPadding(self, input_tensor, isStat=False):
        raw_dim = self.raw_input_image_shape
        if isStat:
            non_zero_n = 2 * self.img_sz**2
            input_tensor = input_tensor.view(-1, raw_dim[0], raw_dim[1] ** 2)[:,:,:non_zero_n]
            # input_tensor = input_tensor.view(-1, self.ch, 2*self.img_sz, self.img_sz)
            ### only use the first channel 
            # ipdb.set_trace()
            input_tensor = input_tensor[:,0,:].view(-1, 1, 2*self.img_sz, self.img_sz)
        else: 
            non_zero_n = self.img_sz**2
            input_tensor = input_tensor.view(-1, raw_dim[0], raw_dim[1] ** 2)[:,:,:non_zero_n]
            # input_tensor = input_tensor.view(-1, *self.model_input_image_shape)
            input_tensor = input_tensor[:,0,:].view(-1, 1, self.model_input_image_shape[1:])
        return input_tensor
    
    def addZeroPadding(self, M_imaged, isStat=False):
        if isStat:
            ### First half is mu, another half is log_var
            # M_vector =  M_imaged.view(-1, self.raw_input_image_shape[0], 2*self.img_sz**2)
            M_vector =  M_imaged.view(-1, 1, 2*self.img_sz**2)
        else:
            # M_vector =  M_imaged.view(-1, self.raw_input_image_shape[0], self.img_sz**2)
            M_vector =  M_imaged.view(-1, 1, self.img_sz**2)
        zero_n = self.raw_n_image_pixels - M_vector.shape[-1]
        zero_pad = torch.zeros(*M_vector.shape[:-1], zero_n).cuda()
        M_zeroPadded_vec = torch.cat((M_vector, zero_pad), dim=2)
        M = M_zeroPadded_vec.view((-1, 1, *self.raw_input_image_shape[1:]))
        return M
    
    def getReducedMemory(self, storage, num_sample_per_task):
        n_c = num_sample_per_task // self.agent_config['first_split_size']
        new_storage = []
        count_np = np.array([x[3] for x in storage])
        prior_big2small = np.argsort(count_np)[::-1][:num_sample_per_task]
        prior_small2big = np.argsort(count_np)[:num_sample_per_task]
        # for idx in prior_big2small:
        random_big2small = np.random.permutation(np.argsort(count_np)[::-1])[:num_sample_per_task]
        for idx in np.random.permutation(random_big2small):
        # for idx in prior_small2big:
            new_storage.append(storage[idx])
        print("storage size is reduced from {} to {}".format(len(storage), len(new_storage)) )
        return new_storage
        
        # storage_loader = torch.utils.data.DataLoader(storage, batch_size=len(storage), shuffle=False) 
       
        # new_storage = Memory()
        # for i, (input, target, task) in enumerate(storage_loader):
            # input = input.cuda()
            # target = target.cuda()
            # reduced_input, reduced_target, reduced_task, count_list = self.getClustredStats(input, target, task, n_clusters=n_c)
        
        # new_storage= self.saveAsTuple(reduced_input, reduced_target, reduced_task, new_storage)
        # ipdb.set_trace()

    def getCompressedMemory(self, train_loader, num_sample_per_task, noise_test=False):
        self.noise_test = noise_test
        
        embed_memory_data = []
        max_n_train_mha = 10 if not noise_test else 1
        current_task_str = self.train_info[1]
        
        data_count_check = {"old": 0, "new": 0, "org": 0}
        
        self.log("Clustering and Saving stats for each batch...")
        self.MHA_model.eval()

        total_stats_list, total_target_list, total_task_list, total_count_list = [], [], [], []
        for i, (input, target, task) in enumerate(train_loader):
            if (i+1) % 10 == 0:
                self.log('Batch {}/{} Clustering with n_c={}, size {} and saving...'.format(i+1, \
                len(train_loader), self.agent_config['n_clusters'], input.shape))
            
            input = input.cuda()
            target = target.cuda()
            with torch.no_grad():
                memory_replaced_input, this_target = self.intraClassPredict(input, target, task, isTraining=False, batch_idx=i)
                memory_replaced_input, this_target, this_task, count_list = self.getClustredStats(memory_replaced_input, this_target, task)

            total_stats_list.append(memory_replaced_input) 
            total_target_list.append(this_target)
            total_task_list.extend(this_task)
            total_count_list.extend(count_list)

        total_stats_vec = torch.cat(total_stats_list, dim=0)
        total_this_target = torch.cat(total_target_list, dim=0)
        
        print("Saving clustered data... total_stats_vec.shape", total_stats_vec.shape)
        embed_memory_data = self.saveAsTuple(total_stats_vec, total_this_target, total_task_list, embed_memory_data)
        return embed_memory_data, total_count_list


    def getClustredStats(self, memory_replaced_input, this_target, this_task, n_clusters=None):
        sample_count_list = []

        org_shape = memory_replaced_input.shape
        if n_clusters == None:
            n_clusters = self.agent_config['n_clusters']
        
        target_np = this_target.detach().cpu().numpy()
        new_target_labels = sorted(set(target_np))
        
        clustered_vec_list, target_list, task_list = [], [], []
        this_task_int_np = np.array([int(x) for x in this_task])
        
        for idx, label in enumerate(new_target_labels):
            global_label_bool_np = (target_np == label) 
            global_label_bool_tensor = torch.tensor(global_label_bool_np)
            stats_raw = memory_replaced_input[global_label_bool_tensor, :]
            label_this_target = this_target[global_label_bool_tensor]
            label_this_task = this_task_int_np[global_label_bool_tensor.cpu().numpy()]
            stats_vec = stats_raw.view(-1, 2 * self.img_sz * self.img_sz)
        
            # X_mat = euclidean_distances(stats_vec[:, :(self.img_sz ** 2)].cpu().numpy())
            X_mat_dist= euclidean_distances(stats_vec.cpu().numpy())
            X_mat = np.exp(-1*X_mat_dist ** 2)
            this_n_clusters = min(X_mat.shape[0], n_clusters)

            # spectral_model = SpectralClustering(affinity='precomputed', n_jobs=4, n_clusters=this_n_clusters,eigen_tol=1e-10)
            # Y = spectral_model.fit_predict(stats_vec.cpu().numpy())
            # Y = spectral_model.fit_predict(X_mat)
            kmeans = KMeans(n_clusters=this_n_clusters, random_state=0).fit(stats_vec.cpu().numpy())
            Y = kmeans.labels_
            this_task_str = str(Counter(list(label_this_task)).most_common()[0][0])
            setY= list(set(Y))
            # print("Counter Y:", Counter(Y))
            # ipdb.set_trace()
            for k in setY:
                # mean_vec = np.mean(stats_vec[Y==k, :].cpu().numpy(), axis=0)
                # clustered_vec_list.append(torch.tensor(mean_vec).unsqueeze(0))
                np.argsort(np.sum(X_mat[Y==k], axis=1))
                rep_vec_idx = np.argsort(np.sum(X_mat[Y==k], axis=1))[-1]
                if self.agent_config['clustering_type'] in [0,1]:
                    rep_vec = stats_vec[Y==k, :][rep_vec_idx]
                    clustered_vec_list.append(rep_vec)
                    target_list.append(label)
                    task_list.append(this_task_str)
                    sample_count_list.append(stats_vec[Y==k, :].shape[0])
                elif self.agent_config['clustering_type'] in [2]:
                    for vec_idx in range(stats_vec[Y==k, :].shape[0]):
                        vec = stats_vec[Y==k, :][vec_idx]
                        clustered_vec_list.append(vec)
                        target_list.append(label)
                        task_list.append(this_task_str)
                        sample_count_list.append(stats_vec[Y==k, :].shape[0])
        
        out_clustered_vecs = torch.cat(clustered_vec_list, 0).unsqueeze(1).cuda()
        clustered_this_target = torch.tensor(target_list).cuda()
        this_task = tuple(task_list)
        # ipdb.set_trace()
        return out_clustered_vecs, clustered_this_target, this_task, sample_count_list
        
    def saveEmbeddingMemory(self, train_loader, num_sample_per_task, noise_test=False):
        self.noise_test = noise_test
        
        embed_memory_data = []
        max_n_train_mha = 10 if not noise_test else 1
        current_task_str = self.train_info[1]
        
        data_count_check = {"old": 0, "new": 0, "org": 0}
        
        self.log("Saving memory embedding for each batch...")
        self.MHA_model.eval()
        for i, (input, target, task) in enumerate(train_loader):
            
            input = input.cuda()
            target = target.cuda()
            with torch.no_grad():
                memory_replaced_input, this_target= self.intraClassPredict(input, target, task, isTraining=False)
                embed_memory_data = self.saveAsTuple(memory_replaced_input, this_target, task, embed_memory_data)

        return embed_memory_data

    def predict(self, inputs):
        self.model.eval()
        out = self.forward(inputs)
        for t in out.keys():
            out[t] = out[t].detach()
        return out
    
    def class_wise_predict(self, inputs, target, task):
        n_classes = len(set(target.cpu().numpy()))
        self.model.eval()
        self.MHA_model.eval()
        labels = sorted(list(set(target.cpu().numpy())))
        
        out_list = []
        target_list = []
        
        # M_before_integ = inputs[target == label, :].view(-1, self.ch, self.raw_n_image_pixels)
        org_shape = inputs.shape[1:]
        M_before_integ = inputs.view(-1, self.ch, self.raw_n_image_pixels)
        # M_in = M_before_integ.unsqueeze(dim=0)  ### Batch --> input_length: DON'T USE THIS FOR INFERENCE
        M_in = M_before_integ.unsqueeze(dim=1)  ### Batch --> Batch for inference
        M, attention = self.MHA_model(M_in)
        if self.agent_config['do_task1_training'] in [2] and self.train_info[0] > 0:
            past_Ms = {}
            for task_int in range(self.train_info[0]):
                past_model = self.dict_task_models[task_int]
                past_Ms[task_int] = past_model(M_in)
        
        if self.agent_config['mlp_mha'] in [8,9,10,11]:
            mu, log_var = attention
            M_sampled = self.MHA_model.model_gen_net1.sampling(mu, log_var)
            assert M_sampled.shape == mu.shape, "Sampled M has a different dimension"
            if self.agent_config['no_random_prediction']:
                M = mu 
            else:
                M = M_sampled
                
            # ipdb.set_trace() 
            
            
        # ipdb.set_trace() 
        ### Before max_pool

        if self.agent_config['compress_data'] == 1:
            M_imaged = M.view(-1, self.ch, self.img_sz, self.img_sz).detach()
            if self.agent_config['do_task1_training'] in [2] and self.train_info[0] > 0:
                with torch.no_grad():
                    for task_int in range(self.train_info[0]):
                        M_, attn_ = past_Ms[task_int] 
                        M_imaged_ = M_.view(-1, 1, self.img_sz, self.img_sz).detach()
                        past_Ms[task_int] = M_imaged_
        else:
            M_imaged = M.view(-1, 1, *self.raw_input_image_shape[1:]).detach()
        # M_imaged = M.view(-1, self.ch, self.img_sz, self.img_sz).detach()
        
        if len(M_imaged) == 3:
            M_imaged = M_imaged.unsqueeze(0)
        
        M_imaged_batch = M_imaged
        out = self.forward(M_imaged_batch)
        if self.agent_config['do_task1_training'] in [2] and self.train_info[0] > 0:
            out_past = {}
            with torch.no_grad():
                for task_int in range(self.train_info[0]):
                    out_past[task_int] = self.forward(past_Ms[task_int])['All'].detach().cpu()

        sm = Softmax(dim=1)
        
        for t in out.keys():
            out[t] = out[t].detach()
            out_sm_tensor = []
            if self.agent_config['do_task1_training'] in [2] and self.train_info[0] > 0:
                with torch.no_grad():
                    entropy_vals_np = np.zeros( (out['All'].shape[0], self.train_info[0]+1))
                    out_softmax_tensor = torch.zeros( (*out['All'].shape, self.train_info[0]+1))
                    for ek in range(self.train_info[0]):
                        SM = sm(out_past[ek]).cpu().numpy()
                        entropy_vals_np[:, ek] = entropy( SM, axis=1)
                        out_softmax_tensor[:, :, ek] = out_past[ek].cuda()
                    SM = sm(out['All']).cpu().numpy()
                    entropy_vals_np[:, -1] = entropy( SM, axis=1)
                    out_softmax_tensor[:, :, -1] = out['All']
                    argmin_task_idx = np.argmin(entropy_vals_np, axis=1)
                    out_sm_np = out_softmax_tensor.detach().cpu().numpy()
                    out['All'] = out_softmax_tensor[:, :, argmin_task_idx]
                    for k, adx in enumerate(argmin_task_idx):
                        out_sm_tensor.append(out_softmax_tensor[k, :, adx])
                    out_sm_tensor = torch.cat(out_sm_tensor, dim=0).view(out['All'].shape[0], -1)
                    out['All'] = out_sm_tensor.cuda()
                    # del out_past, past_Ms, entropy_vals_np
                # ipdb.set_trace()
        return out, target
    
    def validation(self, dataloader):
        # This function doesn't distinguish tasks.
        batch_timer = Timer()
        acc = AverageMeter()
        batch_timer.tic()

        orig_mode = self.training
        self.eval()
        for i, (input, target, task) in enumerate(dataloader):

            if self.gpu:
                with torch.no_grad():
                    input = input.cuda()
                    target = target.cuda()
            # output = self.predict(input)
            output, new_target = self.class_wise_predict(input, target, task)
            target = new_target

            # Summarize the performance of all tasks, or 1 task, depends on dataloader.
            # Calculated by total number of data.
            try:
                acc = accumulate_acc(output, target, task, acc)
            except:
                ipdb.set_trace()
            del target, output
            torch.cuda.empty_cache() 

        self.train(orig_mode)

        self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
              .format(acc=acc,time=batch_timer.toc()))
        return acc.avg

    def embed_init_optimizer(self):
        # if self.agent_config['do_task1_training'] in [2]:
            # self.MHA_model = MHA(self.agent_config, self.raw_n_image_pixels).cuda()
            # self.MHA_params = self.MHA_model.parameters()
        optimizer_arg = {'params': list(self.MHA_model.parameters()),
                         'lr': 0.001,
                         'weight_decay':self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD','RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'
        
        self.log("====== embed_init_optimizer ======")
        self.memory_optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.memory_optimizer, 
                                                              # milestones=self.config['schedule'],
                                                              # gamma=0.1)
    
    def memory_init_optimizer(self):
        optimizer_arg = {'params': list(self.model.parameters()) + list(self.MHA_model.parameters()),
                         'lr': 0.001,
                         'weight_decay':self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD','RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'
        
        self.log("====== memory_init_optimizer ======")
        self.memory_optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.memory_optimizer, 
                                                              # milestones=self.config['schedule'],
                                                              # gamma=0.1)
    
    def sampleWeights(self, name):
        if name == 'model':
            MHA_model_param_list = [ x for x in self.model.parameters() ]    
        else: 
            MHA_model_param_list = [ x for x in self.MHA_model.parameters() ]    

        return MHA_model_param_list[0][0]

    def original_update_model(self, inputs, targets, tasks):
        out = self.forward(inputs)
        loss = self.criterion(out, targets, tasks)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach(), out
    
    def memory_update_model(self, input, target, tasks):
        ### Go through memory representation generator first
        labels = sorted(list(set(target.cpu().numpy())))
        input_list = []
        input_raw_image_list = []
        task_list = []
        target_list = []
        tasks_array = np.array(tasks)
        self.MHA_model = self.MHA_model.cuda()
        self.MHA_model.train(True)
        # img_sz = self.agent_config['img_sz'] 
        for idx, label in enumerate(labels):
            task_idx = tasks[idx]
            this_target = target[target == label].detach().cpu()
            input_images = input[target == label].detach().cpu()
            this_task_list = tasks_array[ (target.cpu().numpy() == label) ].tolist()
            
            M_before_integ = input[target == label, :].view(-1, self.ch, self.raw_n_image_pixels)
            M_in = M_before_integ.unsqueeze(dim=0)  ### Batch --> input_length
            # M_in = M_before_integ.unsqueeze(dim=1)  ### Batch --> Batch
            # try: 
            M, attn = self.MHA_model(M_in)
            # except:
                # ipdb.set_trace()

            ###  Before max-pooling 
            M = M.squeeze(0)
            
            if self.agent_config['compress_data'] == 1:
                try:
                    M_imaged = M.view(-1, self.ch, self.img_sz, self.img_sz)
                except:
                    ipdb.set_trace()
            else:
                ish = M_before_integ.shape
                M_imaged = M.view(*ish)
            # M_imaged = M.view(-1, self.ch, self.img_sz, self.img_sz)
           
            ### For max-pooling 
            # M_imaged = M.view(1, -1, ish[1], ish[2])
            
            # if len(M_imaged.shape) == 3:
                # M_imaged = M_imaged.unsqueeze(0)
            
            ### My mistake --- I predicted twice!  
            # M_embed, attn = self.MHA_model(M_imaged)
            # M_imaged_embed = M_embed.view(M_before_integ.shape)
            
            ### Before max_pool
            input_list.append(M_imaged)
            input_raw_image_list.append(input_images)
            target_list.append(this_target)
            task_list.extend(this_task_list)
            
            ### After max_pool
            # pooled_image_N = M_imaged.shape[1]
            # input_list.append(M_imaged)
            # target_list.append(this_target[:pooled_image_N])
            # task_list.extend(this_task_list[:pooled_image_N])
       
        ### Before max-pool
        try:
            aggr_embed_inputs = torch.cat(input_list, dim=0).view(-1, *self.model_input_image_shape).cuda()
        except:
            ipdb.set_trace() 
        ### After max-pooling 
        # aggr_embed_inputs = torch.cat(input_list, dim=1).view(-1, *self.model_input_image_shape).cuda()
        # aggr_embed_inputs = aggr_embed_inputs.squeeze(0)


        aggr_targets = torch.cat(target_list, dim=0).cuda()
        aggr_input_images= torch.cat(input_raw_image_list, dim=0).cuda()
        aggr_targets = aggr_targets.cuda()

        aggr_tasks = tuple(task_list)

        ### ########################
        out = self.forward(aggr_embed_inputs)
        loss = self.criterion(out, aggr_targets, aggr_tasks)

        if self.agent_config['mlp_mha'] == 7:
            # kld = self.vae_loss_function(*attn)
            try:
                org_M = aggr_input_images.view(-1, self.ch, self.img_sz**2) 
                recon_x = self.MHA_model.model_gen_net1.decoder(aggr_embed_inputs.view(-1, self.ch, self.img_sz**2))
                bce, kld = self.vae_loss_function(recon_x, org_M, *attn)
            except:
                ipdb.set_trace()
            loss = loss + bce + 0.1*kld
        
        if self.agent_config['mlp_mha'] == 8:


            # fixed_std = self.agent_config['fixed_std']
            # kld = self.kld_loss_function( attn[0], attn[1], fixed_std)

            # loss = loss + 0.001*kld
            pass
                    
        # self.optimizer.zero_grad()
        # print("Before backward model : ", self.sampleWeights('model') )
        # print("Before backward MHA: ", self.sampleWeights('MHA_model') )

        self.memory_optimizer.zero_grad()
        if self.agent_config['mlp_mha'] == 5:
            W = self.MHA_model.state_dict()['model_gen_net1.fc2.weight']
            dimW0 = W.shape[0]
            dimW1 = W.shape[1]
            wwt = torch.matmul( W, W.transpose(1,0) )
            wtw = torch.matmul( W.transpose(1,0) , W)
            reg_SO = wwt - torch.eye(dimW0).cuda()
            # so_loss = (1/( W.shape[0] *W.shape[1] )) * torch.norm(reg_SO) ** 2
            so_loss =  torch.norm(reg_SO) ** 2
            so_loss.requires_grad = True
            loss = loss + self.agent_config['mu'] * so_loss
            
        
        loss.backward()

        self.memory_optimizer.step()
        # print("AFTER backward model : ", self.sampleWeights('model') )
        # print("AFTER backward MHA: ", self.sampleWeights('MHA_model') )

        ### Weight Check
        return loss.detach(), out, aggr_targets

    def restore_update_model(self, input, target, task, force_train=False, batch_idx=None):
        ### Replace intputs of the current task to memory representations.
        # new_input = self.getMemoryRepresentations(input, target, task, )
        
        # W = self.MHA_model.state_dict()['model_gen_net1.fc1.weight']
        # print("force train : " , force_train, "fc1 W ", W)
        
        if self.gpu:
            # with torch.no_grad():
            input = input.cuda()
            target = target.cuda()

        self.model.train(True)

        if force_train == True:
            self.MHA_model.train(True)
            self.MHA_model = self.MHA_model.cuda()
            memory_replaced_input, target = self.intraClassPredict(input, target, task, isTraining=True, batch_idx=batch_idx)
        else:
            self.MHA_model.eval()
            with torch.no_grad():
                memory_replaced_input, target = self.intraClassPredict(input, target, task, isTraining=True, batch_idx=batch_idx)
        
        if self.gpu:
            new_input  = memory_replaced_input.cuda()
        else:
            new_input = memory_replaced_input 
        
        # if self.gpu:
            # # with torch.no_grad():
            # new_input = new_input.cuda()
            # target = target.cuda()
            
        out = self.forward(new_input)
        try:
            loss = self.criterion(out, target, task)
        except:
            ipdb.set_trace() 

        if force_train == True:
            self.memory_optimizer.zero_grad()
            loss.backward()
            self.memory_optimizer.step()
        else:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss.detach(), out, target
    
    # def getMemoryRepresentations(self, input, target, task):
        # self.MHA_model.eval()
        # memory_replaced_input, target = self.intraClassPredict(input, target, task)
        # if self.gpu:
            # memory_replaced_input = memory_replaced_input.cuda()
        # return memory_replaced_input

    def memory_learn_batch(self, train_loader, val_loader=None, pretraining=False):
        # if self.reset_optimizer:  # Reset optimizer before learning each task
            # # self.log('Optimizer is reset!')
        # if self.train_info[0] == 0: ### If this is the first task
            # self.memory_init_optimizer()
            # self.log('Memory Optimizer is reset!')
            # else:
                # self.init_optimizer()
                # self.log('Regular Optimizer is reset!')

        # if self.agent_config['do_task1_training'] in [2]:
            # self.embed_init_optimizer()
            # self.log('Memory Optimizer is reset!')

        if pretraining:
            n_epoch = 3
        else:
            n_epoch = self.config['schedule'][-1]

        for epoch in range(n_epoch):
            if pretraining:
                self.log("000000000================ Pretraining =================00000000")
            data_timer = Timer()
            batch_timer = Timer()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            acc = AverageMeter()

            # Config the model and optimizer
            self.log('agent_name: {}'.format(self.agent_config['agent_name']))
            self.log('exp_note: {}'.format(self.agent_config['exp_note']))
            
            if pretraining:
                self.log('------> Pretraining Epoch:{} Task: {} '.format(epoch, self.train_info))
            else:
                self.log('------> Epoch:{} Task: {} '.format(epoch, self.train_info))

            self.model.train()
            # self.scheduler.step(epoch)
            for param_group in self.optimizer.param_groups:
                self.log('LR:',param_group['lr'])

            # Learning with mini-batch
            data_timer.tic()
            batch_timer.tic()
            self.log('Itr\t\tTime\t\t  Data\t\t  Loss\t\tAcc')
            for i, (input, target, task) in enumerate(train_loader):
                data_time.update(data_timer.toc())  # measure data loading time

                if self.gpu:
                    input = input.cuda()
                    target = target.cuda()

                if self.agent_config['do_task1_training'] == 0:  ### Use random weights as embedding extractor.
                    if pretraining and self.agent_config['pretrain'] in [1]:
                        loss, output, new_target = self.restore_update_model(input, target, task, force_train=True, batch_idx=i)
                    else:
                        loss, output, new_target = self.restore_update_model(input, target, task, force_train=False, batch_idx=i)

                elif self.agent_config['do_task1_training'] == 1:  ### Embedding extractor original strategy.
                    if self.train_info[0] == 0: ### If this is the first task
                        # loss, output, new_target = self.memory_update_model(input, target, task)
                        loss, output, new_target = self.restore_update_model(input, target, task, force_train=True, batch_idx=i)
                    else: 
                        loss, output, new_target = self.restore_update_model(input, target, task, force_train=False, batch_idx=i)
                elif self.agent_config['do_task1_training'] == 2:  ### Just train
                    if self.train_info[0] % 2 == 0:
                        loss, output, new_target = self.restore_update_model(input, target, task, force_train=True, batch_idx=i)
                    else:
                        loss, output, new_target = self.restore_update_model(input, target, task, force_train=False, batch_idx=i)

                input = input.detach()
                target = new_target
                target = target.detach()

                # measure accuracy and record loss
                acc = accumulate_acc(output, target, task, acc)
                losses.update(loss, input.size(0))

                batch_time.update(batch_timer.toc())  # measure elapsed time
                data_timer.toc()

                if ((self.config['print_freq']>0) and ( (i+1) % self.config['print_freq'] == 0)) or (i+1)==len(train_loader):
                    self.log('[{0}/{1}]\t'
                          '{batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                          '{data_time.val:.4f} ({data_time.avg:.4f})\t'
                          '{loss.val:.3f} ({loss.avg:.3f})\t'
                          '{acc.val:.2f} ({acc.avg:.2f})'.format(
                        i+1, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, acc=acc))

            self.log(' * Train Acc {acc.avg:.3f}'.format(acc=acc))

            # Evaluate the performance of current task
            if val_loader != None:
                self.validation(val_loader)

