import torch
import torch.nn.functional as F
import numpy as np
from importlib import import_module
from .default import NormalNN
from .default import *
from .regularization import SI, L2, EWC, MAS
from dataloaders.wrapper import Storage
import ipdb

from utils.metric import accuracy, AverageMeter, Timer

from transformer.SubLayers import MultiHeadAttentionMemory, mlp_embeddingExtractor, VAE
# from transformer.SubLayers import mlp_embeddingExtractor

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
        
        # ipdb.set_trace()
# class MHA_pool(nn.Module):
    # def __init__(self, raw_n_image_pixels): 
        # super().__init__()
        # opt = OPT(raw_n_image_pixels=raw_n_image_pixels)  ### Parameter Class
        
        # ### Multi-head attention for Memory 
        # self.mul_head_attn1 = MultiHeadAttentionMemory(opt.n_head, opt.d_model, opt.d_k, opt.d_v, dropout=opt.dropout).cuda()
        # self.mul_head_attn2 = MultiHeadAttentionMemory(opt.n_head, opt.d_model, opt.d_k, opt.d_v, dropout=opt.dropout).cuda()
        # self.mul_head_attn3 = MultiHeadAttentionMemory(opt.n_head, opt.d_model, opt.d_k, opt.d_v, dropout=opt.dropout).cuda()
        # self.mha_fc_pool =  maxPoolFeedForward(opt.d_model, opt.d_model, opt.max_pool_size)
    
    # def forward(self, M):
        # M,     attn = self.mul_head_attn1(M, M, M)
        # M,     attn = self.mul_head_attn2(M, M, M)
        # M_out, attn = self.mul_head_attn3(M, M, M)
        # return M_out, attn2

    
class MHA(nn.Module):
    def __init__(self, agent_config, raw_n_image_pixels): 
        super().__init__()
        opt = OPT(agent_config, raw_n_image_pixels=raw_n_image_pixels)  ### Parameter Class
        self.agent_config = agent_config 
        
        ### Multi-head attention for Memory 
        if self.agent_config['mlp_mha'] == 0:
            self.mul_head_attn1 = MultiHeadAttentionMemory(opt).cuda()
            self.mul_head_attn2 = MultiHeadAttentionMemory(opt).cuda()

        elif self.agent_config['mlp_mha'] == 1:
            self.mul_head_attn1 = mlp_embeddingExtractor(opt).cuda()
            self.mul_head_attn2 = mlp_embeddingExtractor(opt).cuda()
            
        elif self.agent_config['mlp_mha'] == 2:
            opt.mlp_res = True 
            self.mul_head_attn1 = MultiHeadAttentionMemory(opt).cuda()
            opt.d_model = opt.d_model_embed
            self.mul_head_attn2 = MultiHeadAttentionMemory(opt).cuda()
            self.mul_head_attn3 = MultiHeadAttentionMemory(opt).cuda()
        
        elif self.agent_config['mlp_mha'] == 3:
            opt.mlp_res = False
            self.mlp_embed = mlp_embeddingExtractor(opt).cuda()
            if opt.compress:
                opt.d_model = opt.d_model_embed
            self.mul_head_attn1 = MultiHeadAttentionMemory(opt).cuda()
            self.mul_head_attn2 = MultiHeadAttentionMemory(opt).cuda()
            self.mul_head_attn3 = MultiHeadAttentionMemory(opt).cuda()
        elif self.agent_config['mlp_mha'] == 4:
            pass
            # opt.mlp_res = False
            # self.mlp_embed = mlp_embeddingExtractor(opt).cuda()
            # if opt.compress:
                # opt.d_model = opt.d_model_embed
            # self.mul_head_attn1 = MultiHeadAttentionMemory(opt).cuda()
            # self.mul_head_attn2 = MultiHeadAttentionMemory(opt).cuda()
            # self.mul_head_attn3 = MultiHeadAttentionMemory(opt).cuda()

        elif self.agent_config['mlp_mha'] in [5,6]:
            self.mul_head_attn1 = mlp_embeddingExtractor(opt).cuda()
            self.mul_head_attn2 = mlp_embeddingExtractor(opt).cuda()
        
        elif self.agent_config['mlp_mha'] in [7, 8, 9]:
            self.mul_head_attn1 = VAE(opt, x_dim=1024, h_dim1= 320, h_dim2=256, z_dim=self.agent_config['img_sz']**2)
            self.mul_head_attn2 = VAE(opt, x_dim=1024, h_dim1= 320, h_dim2=256, z_dim=self.agent_config['img_sz']**2)
        
            # self.mul_head_attn1 = MultiHeadAttentionMemory(opt.n_head, opt.d_model, opt.d_k, opt.d_v, d_model_embed=opt.dm_out, compress=opt.compress,mlp_res=True, dropout=opt.dropout).cuda()
            # self.mul_head_attn2 = MultiHeadAttentionMemory(opt.n_head, opt.d_model, opt.d_k, opt.d_v, d_model_embed=opt.dm_out, compress=opt.compress,mlp_res=True, dropout=opt.dropout).cuda()
        
        mp = opt.max_pool_size
        self.maxpool1d = nn.MaxPool1d(mp, stride=mp)
        
        # self.fc_dim_control= nn.Linear(raw_n_image_pixels, opt.d_model, bias=False).cuda()
    
    def forward(self, M):
        # M,     attn = self.mul_head_attn1(M, M, M)
        # M = self.fc_dim_control(M)
        NL = self.agent_config['num_mha_layers']

        if self.agent_config['mlp_mha'] in [7, 8, 9]:
            M, mu, log_var = self.mul_head_attn1(M)
            attn = (mu, log_var)
        
        else:
            
            if self.agent_config['mlp_mha'] in [2,3]:
                if self.agent_config['mlp_mha'] == 2:
                    with torch.no_grad():
                        M = self.mlp_embed(M)
                elif self.agent_config['mlp_mha'] == 3:
                    M = self.mlp_embed(M)
                
            if NL == 1:
                M, attn = self.mul_head_attn1(M, M, M)
            elif NL == 2:
                M, attn = self.mul_head_attn1(M, M, M)
                M, attn = self.mul_head_attn2(M, M, M)
            elif NL == 3:
                M, attn = self.mul_head_attn1(M, M, M)
                M, attn = self.mul_head_attn2(M, M, M)
                M, attn = self.mul_head_attn3(M, M, M)
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
        self.ch = 1
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
    
    def p_log(self, *args):
        # print("=-===1=24=124=12=4=124 LOG")
        self.pI.a_print(args)
        # print(*args)

    def learn_batch(self, train_loader, val_loader=None):
        if self.task_count == 0:
            # self.memory_size = self.memory_size * (self.raw_n_image_pixels//(self.img_sz**2))
            self.memory_size = self.memory_size
        print("--------------========== MEMORY SIZE : ", self.memory_size)
        # method = "Noise_Rehearsal"
        # method = "No_Rehearsal"
        # method = "Memory_Embedding_Rehearsal" 
        # method = "Compressed_Memory_Embedding_Rehearsal"
        # method = "Fed_Memory_Rehearsal" 
        self.log("[method] {}".format(self.method))
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
        
        if self.method == "No_Rehearsal":
            dataset_list = []
       
        # if self.method == "Memory_Embedding_Rehearsal" and self.train_info[0] > 0:
            # memoryListInstance = self.replaceImage2EmbeddingMemory(train_loader, noise_test=False)
            # dataset_list.append(memoryListInstance.storage)
        # else:
        dataset_list.append(train_loader.dataset)

        ### Wrap mixed dataset (dataset_list).
            # pdb.set_trace()
        dataset = torch.utils.data.ConcatDataset(dataset_list)

        new_train_loader = torch.utils.data.DataLoader(dataset,
                                                       batch_size=train_loader.batch_size,
                                                       shuffle=True,
                                                       num_workers=train_loader.num_workers)

        # 2.Update model as normal
        # super(Fed_Memory_Rehearsal, self).learn_batch(new_train_loader, val_loader)
        
        self.memory_learn_batch(new_train_loader, val_loader)
            

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

        elif self.method == "Fed_Memory_Rehearsal":
            self.reqGradToggle(False)  ### Freeze the model
            fed_memory_data = self.getCompressedMemory(train_loader, num_sample_per_task, noise_test=False)
            self.reqGradToggle(True)  ### Freeze the model
            # self.task_memory[self.task_count].extend(fed_memory_data)
            randind = torch.randperm(len(fed_memory_data))[:num_sample_per_task]  # randomly sample some data
            for ind in randind:  # save it to the memory
                self.task_memory[self.task_count].append(fed_memory_data[ind])
        
        elif self.method == "Memory_Embedding_Rehearsal":
            self.reqGradToggle(False)  ### Freeze the model
            embed_memory_data = self.saveEmbeddingMemory(train_loader, num_sample_per_task, noise_test=False)
            self.reqGradToggle(True)  ### Freeze the model
            randind = torch.randperm(len(embed_memory_data))[:num_sample_per_task]  # randomly sample some data
            for ind in randind:  # save it to the memory
                self.task_memory[self.task_count].append(embed_memory_data[ind])
        
        elif self.method == "Compressed_Memory_Embedding_Rehearsal":
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
        try:
            BCE = F.binary_cross_entropy(recon_x, x_vec, reduction='sum')
        except:
            ipdb.set_trace()
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
    
    # def saveAsMemoryInstance(self, M_imaged, target, task, Mem):
        # M = M_imaged.squeeze(0)
        # for k in range(M.shape[0]):
            # if self.noise_test == True:
                # M_tup = ( torch.randn(self.model_input_image_shape), target[k].detach().cpu().numpy().item(), task[k])
            # else:
                # M_tup = (M[k].view(self.model_input_image_shape), target[k].detach().cpu().numpy().item(), task[k])
            # fed_memory_data.append(M_tup)

        # return fed_memory_data
    
    def saveAsTuple(self, M_imaged, target, task, fed_memory_data):
        M = M_imaged.squeeze(0)

        ### For the conveience of loading the data, zeropad the embedding output.
        if self.agent_config['compress_data'] == 1:
            if self.agent_config['mlp_mha'] in [8,9]:
                M = self.addZeroPadding(M, isStat=True)
            else:
                M = self.addZeroPadding(M, isStat=False)
        
        for k in range(M.shape[0]):
            if self.noise_test == True:
                M_tup = ( torch.randn(self.raw_input_image_shape), target[k].detach().cpu().numpy().item(), task[k])
            else:
                assert torch.sum(M[0,0] != 0).item() != self.img_sz ** 2, "non-zero values should be 2x  self.img_sz**2 for stats"
                try:
                    M_tup = (M[k].view(self.raw_input_image_shape), target[k].detach().cpu().numpy().item(), task[k])
                except:
                    ipdb.set_trace()
            fed_memory_data.append(M_tup)

        return fed_memory_data

    def intraClassPredict(self, input, target, task, isTraining=False):
        org_shape = input[0, :].shape

        embed_memory_data = []
        data_count_check = {"old": 0, "new": 0, "org": 0}
        data_count_check['org'] += len(task)
        
        if self.agent_config['compress_data'] == 1:
            if self.agent_config['mlp_mha'] in [8,9]:
                new_input = torch.zeros(*input.shape[:2], 2*self.img_sz, self.img_sz)
            else:
                new_input = torch.zeros(*input.shape[:2], self.img_sz, self.img_sz)
        else:
            if self.agent_config['mlp_mha'] in [8,9]:
                new_input = torch.zeros(*input.shape[:2], self.ch, 2*self.raw_input_image_shape[1], self.raw_input_image_shape[2])
            else:
                new_input = torch.zeros(*input.shape[:2], *self.raw_input_image_shape)
        # if self.agent_config['compress_data'] == 1:
            # new_input = torch.zeros(*input.shape[:2], self.img_sz, self.img_sz)
        # else:
            # new_input = torch.zeros(*input.shape[:2], *self.raw_input_image_shape)

        new_target= torch.zeros(*target.shape).long()

        if self.gpu:
            with torch.no_grad():
                input = input.cuda()
                target = target.cuda()

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
            
            this_target = target[global_label_bool_tensor]

            M_before_integ = input[global_label_bool_tensor, :].view(-1, self.ch, self.raw_n_image_pixels)
            M_in = M_before_integ.unsqueeze(dim=0)  ### Batch --> input_length
            
            ### Do MHA process + max_pool
            M, attention = self.MHA_model(M_in)
            if self.agent_config['mlp_mha'] in [8,9]:
                mu, log_var = attention
                M = torch.cat((mu, log_var), dim=3)
                # if self.agent_config['mlp_mha'] in [9]:
                    # M = mu
                # ipdb.set_trace()

            ### Before max_pool
            if self.agent_config['compress_data'] == 1:
                if self.agent_config['mlp_mha'] in [8,9]:
                    M_imaged = M.view(-1, self.ch, 2*self.img_sz, self.img_sz)
                else:
                    M_imaged = M.view(-1, self.ch, self.img_sz, self.img_sz)
            else:
                if self.agent_config['mlp_mha'] in [8,9]:
                    M_imaged = M.view(-1, self.ch, 2*self.raw_input_image_shape[1], self.raw_input_image_shape[2])
                else:
                    M_imaged = M.view(-1, *self.raw_input_image_shape)
          
            ### For max-pooling
            # ish = M_before_integ.shape
            # M_imaged = M.view(1, -1, ish[1], ish[2])
            
            if len(M_imaged) == 3:
                M_imaged = M_imaged.unsqueeze(0)
            
            ### Before max_pool
            new_input[global_label_bool_tensor] = M_imaged.detach().cpu()
            new_target[global_label_bool_tensor] = this_target.cpu()
           
            ### For max_pool_version
            # pooled_image_N = M_imaged.shape[1]
            # input_list.append(M_imaged)
            # target_list.append(this_target[:pooled_image_N])
            
        ### Second, go through old samples that are already memory samples
        old_input = torch.randn(1)
        old_target= None 

        if old_input_tensor.shape[0] != 0:
            ### Before max_pool
            if self.agent_config['compress_data'] == 1:
                old_portion = input[old_task_bool, :]
                if self.agent_config['mlp_mha'] in [8,9]:
                    M_zero_removed = self.removeZeroPadding(old_portion.cpu(), isStat=True)
                else:
                    M_zero_removed = self.removeZeroPadding(old_portion.cpu(), isStat=False)
                new_input[old_task_bool, :] = M_zero_removed
                old_input = M_zero_removed

            else:     
                new_input[old_task_bool, :] = input[old_task_bool, :].cpu()
                old_input = input[old_task_bool, :].cpu()
            
            new_target[old_task_bool] = target[old_task_bool].cpu() 
            old_target = target[old_task_bool].cpu() 
          
            ### For max-pooling
            # input_list.append(input[old_task_bool, :].cpu())
            # target_list.append(target[old_task_bool].cpu())

        if self.agent_config['mlp_mha'] in [8,9]:
            if not isTraining:  ### That means we are saving the embedding for the future use.
                assert torch.sum(new_input[0,0] != 0).item() != self.img_sz ** 2, "Should have 2x embedding size not 256"
                return new_input, new_target
            
            elif isTraining:
                # mu = new_input[:, :, :self.img_sz, :].view(-1, self.ch, self.img_sz**2)
                # log_var = new_input[:, :, self.img_sz:, :].view(-1, self.ch, self.img_sz**2)
                mu = new_input[:, :, :self.img_sz, :].view(-1, self.ch, self.img_sz**2)
                log_var = new_input[:, :, self.img_sz:, :].view(-1, self.ch, self.img_sz**2)
                new_input = self.MHA_model.mul_head_attn1.sampling(mu, log_var)
                if self.agent_config['mlp_mha'] in [9]:
                    new_input = mu
                regular_new_input = new_input.detach().cpu()
                boost_scale = self.agent_config['boost_scale']
                
                if boost_scale  > 0 and self.train_info[0] > 0 and len(old_input.shape) > 1:
                    mu_old = old_input[:, :, :self.img_sz, :].view(-1, self.ch, self.img_sz**2)
                    log_var_old = old_input[:, :, self.img_sz:, :].view(-1, self.ch, self.img_sz**2)
                    boosted_list = []
                    boosted_list.append(new_input)
                    for bdx in range(boost_scale):
                        old_input = self.MHA_model.mul_head_attn1.sampling(mu_old, log_var_old)
                        if self.agent_config['mlp_mha'] in [9]:
                            old_input = mu
                        boosted_list.append(old_input)
                    new_input = torch.cat(boosted_list, dim=0)
                    old_target_repeated = old_target.repeat((boost_scale))
                    new_target = torch.cat([new_target, old_target_repeated], dim=0)
                    
                ### Make it image shape     
                new_input = new_input.view(-1, self.ch, self.img_sz, self.img_sz)

                # assert (data_count_check['old'] + data_count_check['new']) == data_count_check['org'], "[ERROR] Sample count is somewhat off."
                # assert torch.all(target.cpu() == new_target).item() == 1, "new_target and target are different. Abort."
                # print("===== Data quantity check passed! :", data_count_check)
                assert torch.sum(new_input[0,0] != 0).item() == self.img_sz ** 2, "This should be 1x embedding size."
                
                return new_input, new_target
      
        else:
            ### Before max-pooling 
            assert (data_count_check['old'] + data_count_check['new']) == data_count_check['org'], "[ERROR] Sample count is somewhat off."
            # print("===== Data quantity check passed! :", data_count_check)
            assert torch.all(target.cpu() == new_target).item() == 1, "new_target and target are different. Abort."

            return new_input, new_target

    def removeZeroPadding(self, input_tensor, isStat=False):
        raw_dim = self.raw_input_image_shape
        if isStat:
            non_zero_n = 2 * self.img_sz**2
            input_tensor = input_tensor.view(-1, raw_dim[0], raw_dim[1] ** 2)[:,:,:non_zero_n]
            input_tensor = input_tensor.view(-1, self.ch, 2*self.img_sz, self.img_sz)
        else: 
            non_zero_n = self.img_sz**2
            input_tensor = input_tensor.view(-1, raw_dim[0], raw_dim[1] ** 2)[:,:,:non_zero_n]
            input_tensor = input_tensor.view(-1, *self.model_input_image_shape)
        return input_tensor
    
    def addZeroPadding(self, M_imaged, isStat=False):
        if isStat:
            ### First half is mu, another half is log_var
            M_vector =  M_imaged.view(-1, self.raw_input_image_shape[0], 2*self.img_sz**2)
        else:
            M_vector =  M_imaged.view(-1, self.raw_input_image_shape[0], self.img_sz**2)
        zero_n = self.raw_n_image_pixels - M_vector.shape[-1]
        zero_pad = torch.zeros(*M_vector.shape[:-1], zero_n)
        M_zeroPadded_vec = torch.cat((M_vector, zero_pad), dim=2)
        M = M_zeroPadded_vec.view((-1, *self.raw_input_image_shape))
        return M

        
    def saveEmbeddingMemory(self, train_loader, num_sample_per_task, noise_test=False):
        self.noise_test = noise_test
        
        embed_memory_data = []
        max_n_train_mha = 10 if not noise_test else 1
        current_task_str = self.train_info[1]
        
        data_count_check = {"old": 0, "new": 0, "org": 0}
        
        self.log("Saving memory embedding for each batch...")
        for i, (input, target, task) in enumerate(train_loader):
            
            # print("Saving memory embedding batch {}/{}".format(i+1, len(train_loader)))
            memory_replaced_input, this_target= self.intraClassPredict(input, target, task, isTraining=False)
            embed_memory_data = self.saveAsTuple(memory_replaced_input, this_target, task, embed_memory_data)

        return embed_memory_data

    # def old_saveEmbeddingMemory(self, train_loader, num_sample_per_task, noise_test=False):
        # self.noise_test = noise_test
        
        # embed_memory_data = []
        # max_n_train_mha = 10 if not noise_test else 1
        # current_task_str = self.train_info[1]
        
        # data_count_check = {"old": 0, "new": 0, "org": 0}
        # for i, (input, target, task) in enumerate(train_loader):
            # self.log("Saving memory embedding batch {}/{}".format(i+1, len(train_loader)))
            # data_count_check['org'] += len(task)

            # # if i == len(train_loader)-1:
                # # break
            # if self.gpu:
                # with torch.no_grad():
                    # input = input.cuda()
                    # target = target.cuda()

            # target_np = target.detach().cpu().numpy()
            # task_np = np.array(task)
            # new_task_bool = torch.tensor(np.array(task) == current_task_str)
            # old_task_bool = torch.tensor(np.array(task) != current_task_str)

            # # new_input_tensor = input[new_task_bool]
            # new_target_tensor = target[new_task_bool]
            # new_task_list = tuple( task_np[new_task_bool.cpu().numpy()].tolist() )
            
            # old_input_tensor = input[old_task_bool]
            # old_target_tensor = target[old_task_bool]
            # old_task_list = tuple( task_np[old_task_bool.cpu().numpy()].tolist() )
            # try:
                # assert (len(new_task_list) + len(old_task_list)) == len(task), "[ERROR] Task count is somewhat off."
            # except:
                # ipdb.set_trace()
            # data_count_check['new'] += len(new_task_list)
            # data_count_check['old'] += len(old_task_list)
            
            # ### First go through new task samples (Needs to be converted)
            # try:
                # new_target_labels = sorted(set(new_target_tensor.cpu().numpy()))
            # except:
                # ipdb.set_trace()

            # for idx, label in enumerate(new_target_labels):
                # task_idx = task[idx]
                
                # label_bool_tensor = torch.tensor(new_target_tensor.cpu().numpy() == label)
                # this_target = new_target_tensor[label_bool_tensor]
                
                # M_before_integ = input[label_bool_tensor, :].view(-1, self.ch, self.raw_n_image_pixels)
                # M_in = M_before_integ.unsqueeze(dim=0)  ### Batch --> input_length
                
                # M, attention = self.MHA_model(M_in)

                # # M_imaged = M.view(M_before_integ.shape)
                # if self.agent_config['compress_data'] == 1:
                    # M_imaged = M.view(-1, self.ch, self.img_sz, self.img_sz)
                # else:
                    # M_imaged = M.view(M_before_integ.shape)
                
                # if len(M_imaged) == 3:
                    # M_imaged = M_imaged.unsqueeze(0)
                
                # # embed_memory_data = self.saveAsTuple(M_imaged.detach().cpu(), 
                                                     # # this_target, 
                                                     # # new_task_list, 
                                                     # # embed_memory_data)
                # embed_memory_data = self.saveAsTuple(M_imaged,
                                                     # this_target, 
                                                     # new_task_list, 
                                                     # embed_memory_data)

                # # print("=== Saving label-wise memory embedding...{}/{}".format(idx, len(new_target_labels)))
                # # print("length embed_memory_data : ", len(embed_memory_data) )
               
            # ### Second, go through old samples that are already memory samples
            # if old_input_tensor.shape[0] != 0:
                # embed_memory_data = self.saveAsTuple(old_input_tensor,
                                                     # old_target_tensor, 
                                                     # old_task_list,
                                                     # embed_memory_data)
                
        # assert (data_count_check['old'] + data_count_check['new']) == data_count_check['org'], "[ERROR] Sample count is somewhat off."
        # # self.log("===== Data quantity check passed! :", data_count_check)
        # # ipdb.set_trace()
        # return embed_memory_data

    # def getCompressedMemory(self, train_loader, num_sample_per_task, noise_test=False):
        # self.noise_test = noise_test
        
        # fed_memory_data = []
        # max_n_train_mha = 10 if not noise_test else 1
        
        # for epoch_idx in range(max_n_train_mha):
            # self.log("*Compressed Mem Epoch: {}/{}".format(epoch_idx+1, max_n_train_mha) )
            # batch_acc = AverageMeter()
            # for i, (input, target, task) in enumerate(train_loader):
                # # print("task : ", task[0], flush=True)
                # if i == len(train_loader)-1:
                    # break
                # if self.gpu:
                    # with torch.no_grad():
                        # input = input.cuda()
                        # target = target.cuda()
                # M = input
                # raw_n_image_pixels = M.shape[2] * M.shape[3]
                # M = M.view(-1, self.ch, raw_n_image_pixels)            

                # # max_n_train_mha = 1000000
                # labels = sorted(list(set(target.cpu().numpy())))

                # self.MHA_model.train()
                # acc = AverageMeter()
                # c_loss = 0
                # out_box, label_box = [], []
                
                # for idx, label in enumerate(labels):
                    # acc_label = AverageMeter()
                    # task_idx = task[idx]
                    # this_target = target[target == label]
                    
                    # M_before_integ = input[target == label, :].view(-1, self.ch, self.raw_n_image_pixels)
                    # M_in = M_before_integ.unsqueeze(dim=0)  ### Batch --> input_length
                    
                    # if epoch_idx == 0:
                        # M_before_integ = input[target == label, :].view(-1, self.ch, self.raw_n_image_pixels)
                        # M_in = M_before_integ.unsqueeze(dim=0)  ### Batch --> input_length
                    # else:
                        # M_in = self.neural_task_memory[task_idx][label][i].cuda()
                        # M_before_integ = M_in

                    # self.mha_optimizer.zero_grad()
                    # M, attention = self.MHA_model(M_in)
                    # # M = M[0,:,:]
                    # # else:
                        # # M = self.neural_task_memory[task_idx][label]
                    # if self.agent_config['compress_data'] == 1:
                        # M_imaged = M.view(-1, self.ch, self.img_sz, self.img_sz)
                    # else:
                        # M_imaged = M.view(M_before_integ.shape)
                    
                    # if len(M_imaged) == 3:
                        # M_imaged = M_imaged.unsqueeze(0)

                    # self.neural_task_memory[task_idx][label][i] = M_imaged.detach().cpu()

                    # MHA_output = self.model.forward(M_imaged)
                    # class_loss = self.aggregatedClassLoss(MHA_output, this_target)
                    
                    # if epoch_idx == max_n_train_mha -1:
                        # fed_memory_data = self.saveAsTuple(M_imaged.detach().cpu(), target, task, fed_memory_data)

                    # class_loss.backward()
                    # self.mha_optimizer.step()
                    
                    # acc = accumulate_acc(MHA_output['All'].detach(), this_target.detach(), task, acc)
                    # batch_acc = accumulate_acc(MHA_output['All'].detach(), this_target.detach(), task, batch_acc)

                    # c_loss += class_loss.detach().cpu().numpy()

            # self.log("Epoch Acc {}/{} ACC: {:2.4f} ".format(epoch_idx+1, max_n_train_mha, batch_acc.avg))
            # batch_acc = accumulate_acc(MHA_output['All'].detach(), this_target, task, batch_acc)
        # return fed_memory_data
    
    
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
        # for idx, label in enumerate(labels):
            
        # task_idx = task[idx]
        # this_target = target[target == label]
        
        # M_before_integ = inputs[target == label, :].view(-1, self.ch, self.raw_n_image_pixels)
        org_shape = inputs.shape[1:]
        M_before_integ = inputs.view(-1, self.ch, self.raw_n_image_pixels)
        # M_in = M_before_integ.unsqueeze(dim=0)  ### Batch --> input_length: DON'T USE THIS FOR INFERENCE
        M_in = M_before_integ.unsqueeze(dim=1)  ### Batch --> Batch for inference
    
        # if self.agent_config['predict_dim_match'] == 1:
            # # n_classes = 10
            # dim_repeat = int(self.agent_config['batch_size']//n_classes)
            # M_in = M_in.repeat(1, dim_repeat, 1, 1) 
            # M_in += 0.05*torch.randn(*M_in.shape).cuda()
            # M, attention = self.MHA_model(M_in)
            # M = torch.mean(M, dim=1, keepdim=True)
        # else:
            # M, attention = self.MHA_model(M_in)
        M, attention = self.MHA_model(M_in)
        
        if self.agent_config['mlp_mha'] in [8,9]:
            mu, log_var = attention
            M_sampled = self.MHA_model.mul_head_attn1.sampling(mu, log_var)
            assert M_sampled.shape == mu.shape, "Sampled M has a different dimension"
            if self.agent_config['no_random_prediction'] in [8,9]:
                M = mu 
            else:
                M = M_sampled
                
            # ipdb.set_trace() 
            
            
        # ipdb.set_trace() 
        ### Before max_pool
        try:
            if self.agent_config['compress_data'] == 1:
                M_imaged = M.view(-1, self.ch, self.img_sz, self.img_sz).detach()
            else:
                M_imaged = M.view(-1, *self.raw_input_image_shape).detach()
            # M_imaged = M.view(-1, self.ch, self.img_sz, self.img_sz).detach()
        except:
            ipdb.set_trace()
        
        ### for max-pooling 
        # ish = M_before_integ.shape
        # M_imaged = M.view(1, -1, ish[1], ish[2])
        
        if len(M_imaged) == 3:
            M_imaged = M_imaged.unsqueeze(0)
        
        ### Before max_pool
        # out_list.append(M_imaged) 
        # target_list.append(this_target) 

            ### For max-pooling
            # pooled_image_N = M_imaged.shape[1]
            # out_list.append(M_imaged)
            # target_list.append(this_target[:pooled_image_N])

        #### Before max-pooling 
        # M_imaged_batch = torch.cat(out_list, dim=0).view(-1, *self.model_input_image_shape)

        ### For max-pooling
        # M_imaged_batch = torch.cat(out_list, dim=1).view(-1, *self.model_input_image_shape).cuda()
        # M_imaged_batch = M_imaged_batch.squeeze(0)
        # ipdb.set_trace()
        # try:
            # M_imaged = self.addZeroPadding(M_imaged)
            # M_imaged_batch = M_imaged.view(-1, *org_shape)

        # except:
            # ipdb.set_trace(
        M_imaged_batch = M_imaged
        out = self.forward(M_imaged_batch)
        # new_target = torch.cat(target_list, dim=0)
        for t in out.keys():
            out[t] = out[t].detach()
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
            # output, new_target = self.intraClassPredict(input, target, task)
            
            target = new_target

            # Summarize the performance of all tasks, or 1 task, depends on dataloader.
            # Calculated by total number of data.
            try:
                acc = accumulate_acc(output, target, task, acc)
            except:
                ipdb.set_trace()

        self.train(orig_mode)

        self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
              .format(acc=acc,time=batch_timer.toc()))
        return acc.avg


    def org_learn_batch(self, train_loader, val_loader=None):
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
                                                       shuffle=False,
                                                       num_workers=train_loader.num_workers)
        # 2.Update model as normal
        # super(Fed_Memory_Rehearsal, self).learn_batch(new_train_loader, val_loader)
        self.memory_learn_batch(new_train_loader, val_loader)
            
        self.reqGradToggle(False)  ### Freeze the model
        self.getCompressedMemory(new_train_loader)
        self.reqGradToggle(True)  ### Freeze the model

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
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.memory_optimizer, 
                                                              milestones=self.config['schedule'],
                                                              gamma=0.1)
    

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
                recon_x = self.MHA_model.mul_head_attn1.decoder(aggr_embed_inputs.view(-1, self.ch, self.img_sz**2))
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
            W = self.MHA_model.state_dict()['mul_head_attn1.fc2.weight']
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

    def restore_update_model(self, input, target, task, force_train=False):
        ### Replace intputs of the current task to memory representations.
        # new_input = self.getMemoryRepresentations(input, target, task, )
        if force_train == True:
            self.MHA_model.train(True)
            memory_replaced_input, target = self.intraClassPredict(input, target, task, isTraining=True)
        else:
            self.MHA_model.eval()
            with torch.no_grad():
                memory_replaced_input, target = self.intraClassPredict(input, target, task, isTraining=True)
        
        if self.gpu:
            new_input  = memory_replaced_input.cuda()
        else:
            new_input = memory_replaced_input 
        
        if self.gpu:
            with torch.no_grad():
                new_input = new_input.cuda()
                target = target.cuda()
        self.model.train(True)

            
        out = self.forward(new_input)
        loss = self.criterion(out, target, task)
        
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

    def memory_learn_batch(self, train_loader, val_loader=None):
        if self.reset_optimizer:  # Reset optimizer before learning each task
            # self.log('Optimizer is reset!')
            if self.train_info[0] == 0: ### If this is the first task
                self.memory_init_optimizer()
                self.log('Memory Optimizer is reset!')
            else:
                self.init_optimizer()
                self.log('Regular Optimizer is reset!')

        for epoch in range(self.config['schedule'][-1]):
            data_timer = Timer()
            batch_timer = Timer()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            acc = AverageMeter()

            # Config the model and optimizer
            self.log('agent_name: {}'.format(self.agent_config['agent_name']))
            self.log('exp_note: {}'.format(self.agent_config['exp_note']))
            self.log('------> Epoch:{} Task: {} '.format(epoch, self.train_info))

            self.model.train()
            self.scheduler.step(epoch)
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
                    # try:
                    loss, output, new_target = self.restore_update_model(input, target, task, force_train=False)
                    # except:
                        # ipdb.set_trace()

                elif self.agent_config['do_task1_training'] == 1:  ### Embedding extractor original strategy.
                    if self.train_info[0] == 0: ### If this is the first task
                        loss, output, new_target = self.memory_update_model(input, target, task)
                    else: 
                        loss, output, new_target = self.restore_update_model(input, target, task, force_train=False)
                elif self.agent_config['do_task1_training'] == 2:  ### Just train
                    loss, output, new_target = self.restore_update_model(input, target, task, force_train=True)

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


class Naive_Rehearsal_SI(Naive_Rehearsal, SI):

    def __init__(self, agent_config):
        super(Naive_Rehearsal_SI, self).__init__(agent_config)


class Naive_Rehearsal_L2(Naive_Rehearsal, L2):

    def __init__(self, agent_config):
        super(Naive_Rehearsal_L2, self).__init__(agent_config)


class Naive_Rehearsal_EWC(Naive_Rehearsal, EWC):

    def __init__(self, agent_config):
        super(Naive_Rehearsal_EWC, self).__init__(agent_config)
        self.online_reg = True  # Online EWC


class Naive_Rehearsal_MAS(Naive_Rehearsal, MAS):

    def __init__(self, agent_config):
        super(Naive_Rehearsal_MAS, self).__init__(agent_config)


class GEM(Naive_Rehearsal):
    """
    @inproceedings{GradientEpisodicMemory,
        title={Gradient Episodic Memory for Continual Learning},
        author={Lopez-Paz, David and Ranzato, Marc'Aurelio},
        booktitle={NIPS},
        year={2017},
        url={https://arxiv.org/abs/1706.08840}
    }
    """

    def __init__(self, agent_config):
        super(GEM, self).__init__(agent_config)
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}  # For convenience
        self.task_grads = {}
        self.quadprog = import_module('quadprog')
        self.task_mem_cache = {}

    def grad_to_vector(self):
        vec = []
        for n,p in self.params.items():
            if p.grad is not None:
                vec.append(p.grad.view(-1))
            else:
                # Part of the network might has no grad, fill zero for those terms
                vec.append(p.data.clone().fill_(0).view(-1))
        return torch.cat(vec)

    def vector_to_grad(self, vec):
        # Overwrite current param.grad by slicing the values in vec (flatten grad)
        pointer = 0
        for n, p in self.params.items():
            # The length of the parameter
            num_param = p.numel()
            if p.grad is not None:
                # Slice the vector, reshape it, and replace the old data of the grad
                p.grad.copy_(vec[pointer:pointer + num_param].view_as(p))
                # Part of the network might has no grad, ignore those terms
            # Increment the pointer
            pointer += num_param

    def project2cone2(self, gradient, memories):
        """
            Solves the GEM dual QP described in the paper given a proposed
            gradient "gradient", and a memory of task gradients "memories".
            Overwrites "gradient" with the final projected update.

            input:  gradient, p-vector
            input:  memories, (t * p)-vector
            output: x, p-vector

            Modified from: https://github.com/facebookresearch/GradientEpisodicMemory/blob/master/model/gem.py#L70
        """
        margin = self.config['reg_coef']
        memories_np = memories.cpu().contiguous().double().numpy()
        gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
        t = memories_np.shape[0]
        #print(memories_np.shape, gradient_np.shape)
        P = np.dot(memories_np, memories_np.transpose())
        P = 0.5 * (P + P.transpose())
        q = np.dot(memories_np, gradient_np) * -1
        G = np.eye(t)
        P = P + G * 0.001
        h = np.zeros(t) + margin
        v = self.quadprog.solve_qp(P, q, G, h)[0]
        x = np.dot(v, memories_np) + gradient_np
        new_grad = torch.Tensor(x).view(-1)
        if self.gpu:
            new_grad = new_grad.cuda()
        return new_grad

    def learn_batch(self, train_loader, val_loader=None):

        # 1.Update model as normal
        super(GEM, self).learn_batch(train_loader, val_loader)

        # 2.Randomly decide the images to stay in the memory
        self.task_count += 1
        # (a) Decide the number of samples for being saved
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
        # (d) Cache the data for faster processing
        for t, mem in self.task_memory.items():
            # Concatenate all data in each task
            mem_loader = torch.utils.data.DataLoader(mem,
                                                     batch_size=len(mem),
                                                     shuffle=False,
                                                     num_workers=2)
            assert len(mem_loader)==1,'The length of mem_loader should be 1'
            for i, (mem_input, mem_target, mem_task) in enumerate(mem_loader):
                if self.gpu:
                    mem_input = mem_input.cuda()
                    mem_target = mem_target.cuda()
            self.task_mem_cache[t] = {'data':mem_input,'target':mem_target,'task':mem_task}

    def update_model(self, inputs, targets, tasks):

        # compute gradient on previous tasks
        if self.task_count > 0:
            for t,mem in self.task_memory.items():
                self.zero_grad()
                # feed the data from memory and collect the gradients
                mem_out = self.forward(self.task_mem_cache[t]['data'])
                mem_loss = self.criterion(mem_out, self.task_mem_cache[t]['target'], self.task_mem_cache[t]['task'])
                mem_loss.backward()
                # Store the grads
                self.task_grads[t] = self.grad_to_vector()

        # now compute the grad on the current minibatch
        out = self.forward(inputs)
        loss = self.criterion(out, targets, tasks)
        self.optimizer.zero_grad()
        loss.backward()

        # check if gradient violates constraints
        if self.task_count > 0:
            current_grad_vec = self.grad_to_vector()
            mem_grad_vec = torch.stack(list(self.task_grads.values()))
            dotp = current_grad_vec * mem_grad_vec
            dotp = dotp.sum(dim=1)
            if (dotp < 0).sum() != 0:
                new_grad = self.project2cone2(current_grad_vec, mem_grad_vec)
                # copy gradients back
                self.vector_to_grad(new_grad)

        self.optimizer.step()
        return loss.detach(), out
