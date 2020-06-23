import torch
import numpy as np
from importlib import import_module
from .default import NormalNN
from .default import *
from .regularization import SI, L2, EWC, MAS
from dataloaders.wrapper import Storage
import ipdb

from utils.metric import accuracy, AverageMeter, Timer

from transformer.SubLayers import MultiHeadAttentionMemory

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
        # x = self.dropout(x)
        mu = 0.1
        x += mu * residual

        x = self.layer_norm(x)

        return x
    
class OPT:
    def __init__(self, n_image_pixels):
        self.n_head = 8
        self.d_model = n_image_pixels
        self.d_k = 64
        self.d_v = 64
        self.max_pool_size = 2
        self.dropout = True

class MHA_pool(nn.Module):
    def __init__(self, n_image_pixels): 
        super().__init__()
        opt = OPT(n_image_pixels=n_image_pixels)  ### Parameter Class
        
        ### Multi-head attention for Memory 
        self.mul_head_attn1 = MultiHeadAttentionMemory(opt.n_head, opt.d_model, opt.d_k, opt.d_v, dropout=opt.dropout).cuda()
        self.mul_head_attn2 = MultiHeadAttentionMemory(opt.n_head, opt.d_model, opt.d_k, opt.d_v, dropout=opt.dropout).cuda()
        self.mha_fc_pool =  maxPoolFeedForward(opt.d_model, opt.d_model, opt.max_pool_size)
    
    def forward(self, M):
        M,     attn1 = self.mul_head_attn1(M, M, M)
        M_out, attn2 = self.mul_head_attn2(M, M, M)
        return M_out, attn2

    
class MHA(nn.Module):
    def __init__(self, n_image_pixels): 
        super().__init__()
        opt = OPT(n_image_pixels=n_image_pixels)  ### Parameter Class
        
        ### Multi-head attention for Memory 
        self.mul_head_attn1 = MultiHeadAttentionMemory(opt.n_head, opt.d_model, opt.d_k, opt.d_v, dropout=opt.dropout).cuda()
        self.mul_head_attn2 = MultiHeadAttentionMemory(opt.n_head, opt.d_model, opt.d_k, opt.d_v, dropout=opt.dropout).cuda()
        # self.mha_fc_pool =  maxPoolFeedForward(opt.d_model, opt.d_model, opt.max_pool_size)
    
    def forward(self, M):
        M,     attn1 = self.mul_head_attn1(M, M, M)
        M_out, attn2 = self.mul_head_attn2(M, M, M)
        return M_out, attn2

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
    def __init__(self, agent_config):
        super(Fed_Memory_Rehearsal, self).__init__(agent_config)
        self.task_count = 0
        self.memory_size = 4000 ### Default Memory Size
        self.task_memory = {}

        ### +--------------------------------
        ### | Train data specifications
        ### +--------------------------------
        self.ch = 1
        self.n_image_pixels = 32*32
        self.image_shape = (self.ch, 32, 32)
        self.task_numbers = 100
        self.n_max_label = 100
        self.ls = 1000


        ### +--------------------------------
        ### | Model Specification
        ### +--------------------------------
        self.MHA_model = MHA(self.n_image_pixels)
        self.MHA_params = self.MHA_model.parameters()
        self.mha_optimizer = torch.optim.Adam(self.MHA_params, lr=0.01)
        self.memory_init_optimizer()

        # self.mha_optimizer = torch.optim.SGD(MHA_params, lr=0.0001)
        self.neural_task_memory = {str(x+1): { y: {z: [] for z in range(self.ls)} for y in range(self.n_max_label)} for x in range(self.task_numbers)}
        # self.neural_task_memory = {str(x+1): [] for x in range(self.task_numbers)}
        self.noise_test = False

    def reqGradToggle(self, _bool):
        for layer in self.model.parameters():
            layer.requires_grad = _bool
        self.model.train(_bool)
    
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
        for k in range(M.shape[0]):
            if self.noise_test == True:
                M_tup = ( torch.randn(self.image_shape), target[k].detach().cpu().numpy().item(), task[k])
            else:
                M_tup = (M[k].view(self.image_shape), target[k].detach().cpu().numpy().item(), task[k])
            fed_memory_data.append(M_tup)

        return fed_memory_data

    def intraClassPredict(self, input, target, task):
        org_shape = input[0, :].shape

        embed_memory_data = []
        data_count_check = {"old": 0, "new": 0, "org": 0}
        data_count_check['org'] += len(task)

        # if i == len(train_loader)-1:
            # break
        new_input = torch.zeros(*input.shape)
        if self.gpu:
            with torch.no_grad():
                input = input.cuda()
                target = target.cuda()

        target_np = target.detach().cpu().numpy()
        task_np = np.array(task)
        current_task_str = self.train_info[1]
        # print("------ tasks {} current_task_str {} ".format(set(task), current_task_str) )

        new_task_bool = torch.tensor(np.array(task) == current_task_str)
        old_task_bool = torch.tensor(np.array(task) != current_task_str)

        new_input_tensor = input[new_task_bool]
        new_target_tensor = target[new_task_bool]
        new_task_list = tuple( task_np[new_task_bool.cpu().numpy()].tolist() )
        
        old_input_tensor = input[old_task_bool]
        old_target_tensor = target[old_task_bool]
        old_task_list = tuple( task_np[old_task_bool.cpu().numpy()].tolist() )

        try:
            assert (len(new_task_list) + len(old_task_list)) == len(task), "[ERROR] Task count is somewhat off."
        except:
            ipdb.set_trace()
        data_count_check['new'] += len(new_task_list)
        data_count_check['old'] += len(old_task_list)
        
        ### First go through new task samples (Needs to be converted)
        try:
            new_target_labels = sorted(set(new_target_tensor.cpu().numpy()))
        except:
            ipdb.set_trace()

        for idx, label in enumerate(new_target_labels):
            task_idx = task[idx]
            global_label_bool_np = (target_np == label) * (new_task_bool.numpy())
            global_label_bool_tensor = torch.tensor(global_label_bool_np)
            
            this_target = target[global_label_bool_tensor]

            M_before_integ = input[global_label_bool_tensor, :].view(-1, self.ch, self.n_image_pixels)
            M_in = M_before_integ.unsqueeze(dim=0)  ### Batch --> input_length
            
            M, attention = self.MHA_model(M_in)
            M_imaged = M.view(-1, *org_shape)
            
            if len(M_imaged) == 3:
                M_imaged = M_imaged.unsqueeze(0)
            
            new_input[global_label_bool_tensor] = M_imaged.detach().cpu()

            # print("=== Saving label-wise memory embedding...{}/{}".format(idx, len(new_target_labels)))
            # print("length embed_memory_data : ", len(embed_memory_data) )
           
        ### Second, go through old samples that are already memory samples
        if old_input_tensor.shape[0] != 0:
            new_input[old_task_bool, :] = input[old_task_bool, :].cpu()
        
        assert (data_count_check['old'] + data_count_check['new']) == data_count_check['org'], "[ERROR] Sample count is somewhat off."
        # print("===== Data quantity check passed! :", data_count_check)

        return new_input

    def saveEmbeddingMemory(self, train_loader, num_sample_per_task, noise_test=False):
        self.noise_test = noise_test
        
        embed_memory_data = []
        max_n_train_mha = 10 if not noise_test else 1
        current_task_str = self.train_info[1]
        
        data_count_check = {"old": 0, "new": 0, "org": 0}
        for i, (input, target, task) in enumerate(train_loader):
            print("Saving memory embedding batch {}/{}".format(i+1, len(train_loader)))
            data_count_check['org'] += len(task)

            # if i == len(train_loader)-1:
                # break
            if self.gpu:
                with torch.no_grad():
                    input = input.cuda()
                    target = target.cuda()

            target_np = target.detach().cpu().numpy()
            task_np = np.array(task)
            new_task_bool = torch.tensor(np.array(task) == current_task_str)
            old_task_bool = torch.tensor(np.array(task) != current_task_str)

            new_input_tensor = input[new_task_bool]
            new_target_tensor = target[new_task_bool]
            new_task_list = tuple( task_np[new_task_bool.cpu().numpy()].tolist() )
            
            old_input_tensor = input[old_task_bool]
            old_target_tensor = target[old_task_bool]
            old_task_list = tuple( task_np[old_task_bool.cpu().numpy()].tolist() )
            try:
                assert (len(new_task_list) + len(old_task_list)) == len(task), "[ERROR] Task count is somewhat off."
            except:
                ipdb.set_trace()
            data_count_check['new'] += len(new_task_list)
            data_count_check['old'] += len(old_task_list)
            
            ### First go through new task samples (Needs to be converted)
            try:
                new_target_labels = sorted(set(new_target_tensor.cpu().numpy()))
            except:
                ipdb.set_trace()

            for idx, label in enumerate(new_target_labels):
                task_idx = task[idx]
                
                label_bool_tensor = torch.tensor(new_target_tensor.cpu().numpy() == label)
                this_target = new_target_tensor[label_bool_tensor]
                
                M_before_integ = input[label_bool_tensor, :].view(-1, self.ch, self.n_image_pixels)
                M_in = M_before_integ.unsqueeze(dim=0)  ### Batch --> input_length
                
                M, attention = self.MHA_model(M_in)
                M_imaged = M.view(M_before_integ.shape)
                
                if len(M_imaged) == 3:
                    M_imaged = M_imaged.unsqueeze(0)
                
                embed_memory_data = self.saveAsTuple(M_imaged.detach().cpu(), 
                                                     this_target, 
                                                     new_task_list, 
                                                     embed_memory_data)

                # print("=== Saving label-wise memory embedding...{}/{}".format(idx, len(new_target_labels)))
                # print("length embed_memory_data : ", len(embed_memory_data) )
               
            ### Second, go through old samples that are already memory samples
            if old_input_tensor.shape[0] != 0:
                embed_memory_data = self.saveAsTuple(old_input_tensor.detach().cpu(), 
                                                     old_target_tensor, 
                                                     old_task_list,
                                                     embed_memory_data)
                
        assert (data_count_check['old'] + data_count_check['new']) == data_count_check['org'], "[ERROR] Sample count is somewhat off."
        print("===== Data quantity check passed! :", data_count_check)
        # ipdb.set_trace()
        return embed_memory_data

    def getCompressedMemory(self, train_loader, num_sample_per_task, noise_test=False):
        self.noise_test = noise_test
        
        fed_memory_data = []
        max_n_train_mha = 10 if not noise_test else 1
        
        for epoch_idx in range(max_n_train_mha):
            print("*Compressed Mem Epoch: {}/{}".format(epoch_idx+1, max_n_train_mha) )
            batch_acc = AverageMeter()
            for i, (input, target, task) in enumerate(train_loader):
                # print("task : ", task[0], flush=True)
                if i == len(train_loader)-1:
                    break
                if self.gpu:
                    with torch.no_grad():
                        input = input.cuda()
                        target = target.cuda()
                M = input
                n_image_pixels = M.shape[2] * M.shape[3]
                M = M.view(-1, self.ch, n_image_pixels)            

                # max_n_train_mha = 1000000
                labels = sorted(list(set(target.cpu().numpy())))

                self.MHA_model.train()
                acc = AverageMeter()
                c_loss = 0
                out_box, label_box = [], []
                
                for idx, label in enumerate(labels):
                    acc_label = AverageMeter()
                    task_idx = task[idx]
                    this_target = target[target == label]
                    
                    M_before_integ = input[target == label, :].view(-1, self.ch, self.n_image_pixels)
                    M_in = M_before_integ.unsqueeze(dim=0)  ### Batch --> input_length
                    
                    if epoch_idx == 0:
                        M_before_integ = input[target == label, :].view(-1, self.ch, self.n_image_pixels)
                        M_in = M_before_integ.unsqueeze(dim=0)  ### Batch --> input_length
                    else:
                        M_in = self.neural_task_memory[task_idx][label][i].cuda()
                        M_before_integ = M_in

                    self.mha_optimizer.zero_grad()
                    M, attention = self.MHA_model(M_in)
                    ipdb.set_trace()
                    # M = M[0,:,:]
                    # else:
                        # M = self.neural_task_memory[task_idx][label]
                    M_imaged = M.view(M_before_integ.shape)
                    
                    if len(M_imaged) == 3:
                        M_imaged = M_imaged.unsqueeze(0)
                    self.neural_task_memory[task_idx][label][i] = M_imaged.detach().cpu()

                    MHA_output = self.model.forward(M_imaged)
                    class_loss = self.aggregatedClassLoss(MHA_output, this_target)
                    
                    if epoch_idx == max_n_train_mha -1:
                        fed_memory_data = self.saveAsTuple(M_imaged.detach().cpu(), target, task, fed_memory_data)

                    class_loss.backward()
                    self.mha_optimizer.step()
                    
                    acc = accumulate_acc(MHA_output.detach(), this_target.detach(), task, acc)
                    batch_acc = accumulate_acc(MHA_output.detach(), this_target.detach(), task, batch_acc)

                    c_loss += class_loss.detach().cpu().numpy()

            print("Epoch Acc {}/{} ACC: {:2.4f} ".format(epoch_idx+1, max_n_train_mha, batch_acc.avg))
            batch_acc = accumulate_acc(MHA_output, this_target, task, batch_acc)
        return fed_memory_data
    
    def learn_batch(self, train_loader, val_loader=None):
        # method = "Fed_Memory_Rehearsal" 
        method = "Memory_Embedding_Rehearsal" 
        # method = "No_Rehearsal" 
        # method = "Naive_Rehearsal"
        # method = "noise_Rehearsal"
        
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
        
        if method == "No_Rehearsal":
            data_list = []
        
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
        print("============================  METHOD : {}".format(method))
        if method == "Naive_Rehearsal":
            # (c) Randomly choose some samples from new task and save them to the memory
            randind = torch.randperm(len(train_loader.dataset))[:num_sample_per_task]  # randomly sample some data
            for ind in randind:  # save it to the memory
                self.task_memory[self.task_count].append(train_loader.dataset[ind])
        
        elif method == "noise_Rehearsal":
            self.reqGradToggle(False)  ### Freeze the model
            fed_memory_data = self.getCompressedMemory(train_loader, num_sample_per_task, noise_test=True)
            self.reqGradToggle(True)  ### Freeze the model
            # self.task_memory[self.task_count].extend(fed_memory_data)
            randind = torch.randperm(len(fed_memory_data))[:num_sample_per_task]  # randomly sample some data
            for ind in randind:  # save it to the memory
                self.task_memory[self.task_count].append(fed_memory_data[ind])

        elif method == "No_Rehearsal":
            pass 

        elif method == "Fed_Memory_Rehearsal":
            self.reqGradToggle(False)  ### Freeze the model
            fed_memory_data = self.getCompressedMemory(train_loader, num_sample_per_task, noise_test=False)
            self.reqGradToggle(True)  ### Freeze the model
            # self.task_memory[self.task_count].extend(fed_memory_data)
            randind = torch.randperm(len(fed_memory_data))[:num_sample_per_task]  # randomly sample some data
            for ind in randind:  # save it to the memory
                self.task_memory[self.task_count].append(fed_memory_data[ind])
        
        elif method == "Memory_Embedding_Rehearsal":
            self.reqGradToggle(False)  ### Freeze the model
            embed_memory_data = self.saveEmbeddingMemory(train_loader, num_sample_per_task, noise_test=False)
            self.reqGradToggle(True)  ### Freeze the model
            randind = torch.randperm(len(embed_memory_data))[:num_sample_per_task]  # randomly sample some data
            for ind in randind:  # save it to the memory
                self.task_memory[self.task_count].append(embed_memory_data[ind])
        else:
            raise ValueError('method name {} does not exist.'.format(method))
        # ipdb.set_trace()
    
    def predict(self, inputs):
        self.model.eval()
        out = self.forward(inputs)
        for t in out.keys():
            out[t] = out[t].detach()
        return out
    
    def class_wise_predict(self, inputs, target, task):
        self.model.eval()
        self.MHA_model.eval()
        labels = sorted(list(set(target.cpu().numpy())))
        
        out_list = []
        target_list = []
        for idx, label in enumerate(labels):
            task_idx = task[idx]
            this_target = target[target == label]
            
            M_before_integ = inputs[target == label, :].view(-1, self.ch, self.n_image_pixels)
            M_in = M_before_integ.unsqueeze(dim=0)  ### Batch --> input_length
            
            M, attention = self.MHA_model(M_in)
            M_imaged = M.view(M_before_integ.shape).detach()
            
            if len(M_imaged) == 3:
                M_imaged = M_imaged.unsqueeze(0)
            out_list.append(M_imaged) 
            target_list.append(this_target) 
        # out = self.forward(inputs)
        M_imaged_batch = torch.cat(out_list, dim=0).view(-1, *self.image_shape)
        out = self.forward(M_imaged_batch)
        new_target = torch.cat(target_list, dim=0)
        try:
            for t in out.keys():
                out[t] = out[t].detach()
        except:
            ipdb.set_trace()
        return out, new_target
    
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
            acc = accumulate_acc(output, target, task, acc)

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
        
        print("====== memory_init_optimizer ======")
        self.memory_optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.memory_optimizer, 
                                                              milestones=self.config['schedule'],
                                                              gamma=0.1)
    
    def memory_update_model(self, input, target, tasks):
        ### Go through memory representation generator first
        labels = sorted(list(set(target.cpu().numpy())))
        input_list = []
        task_list = []
        target_list = []
        tasks_array = np.array(tasks)
        self.MHA_model.train(True)
        
        for idx, label in enumerate(labels):
            task_idx = tasks[idx]
            this_target = target[target == label].detach().cpu()
            
            M_before_integ = input[target == label, :].view(-1, self.ch, self.n_image_pixels)
            M_in = M_before_integ.unsqueeze(dim=0)  ### Batch --> input_length
            
            M, attention = self.MHA_model(M_in)
            M_imaged = M.view(M_before_integ.shape)
            
            if len(M_imaged) == 3:
                M_imaged = M_imaged.unsqueeze(0)
            
            M_embed, attn = self.MHA_model(M_imaged)
            M_imaged_embed = M_embed.view(M_before_integ.shape)
            
            input_list.append(M_imaged_embed)
            target_list.append(this_target)
            task_list.extend(tasks_array[ (target.cpu().numpy() == label) ].tolist() )
            
        aggr_embed_inputs = torch.cat(input_list, dim=0).view(-1, *self.image_shape).cuda()
        aggr_targets = torch.cat(target_list, dim=0).cuda()
        
        aggr_targets = aggr_targets.cuda()

        aggr_tasks = tuple(task_list)

        ### ########################
        out = self.forward(aggr_embed_inputs)
        loss = self.criterion(out, aggr_targets, aggr_tasks)
        # self.optimizer.zero_grad()
        # print("Before backward model : ", self.sampleWeights('model') )
        # print("Before backward MHA: ", self.sampleWeights('MHA_model') )
        self.memory_optimizer.zero_grad()
        loss.backward()
        self.memory_optimizer.step()
        # print("AFTER backward model : ", self.sampleWeights('model') )
        # print("AFTER backward MHA: ", self.sampleWeights('MHA_model') )

        return loss.detach(), out, aggr_targets

    def sampleWeights(self, name):
        if name == 'model':
            MHA_model_param_list = [ x for x in self.model.parameters() ]    
        else: 
            MHA_model_param_list = [ x for x in self.MHA_model.parameters() ]    

        return MHA_model_param_list[0][0]


    def restore_update_model(self, input, target, task):
        ### Replace intputs of the current task to memory representations.
        new_input = self.getMemoryRepresentations(input, target, task)
        
        if self.gpu:
            with torch.no_grad():
                new_input = new_input.cuda()
                target = target.cuda()

        out = self.forward(new_input)
        loss = self.criterion(out, target, task)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach(), out, target
    
    def getMemoryRepresentations(self, input, target, task):
        self.MHA_model.eval()
        memory_replaced_input = self.intraClassPredict(input, target, task)
        assert memory_replaced_input.shape == input.shape
        if self.gpu:
            memory_replaced_input = memory_replaced_input.cuda()
        return memory_replaced_input
        # for i in range(inputs.shape[0]):
            # ipdb.set_trace()
            

    def memory_learn_batch(self, train_loader, val_loader=None):
        # ipdb.set_trace()        
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
            self.log('Epoch:{0}'.format(epoch))
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

                if self.train_info[0] == 0: ### If this is the first task
                    loss, output, new_target = self.memory_update_model(input, target, task)
                else: 
                    loss, output, new_target = self.restore_update_model(input, target, task)

                input = input.detach()
                target = new_target
                target = target.detach()

                # measure accuracy and record loss
                acc = accumulate_acc(output, target, task, acc)
                losses.update(loss, input.size(0))

                batch_time.update(batch_timer.toc())  # measure elapsed time
                data_timer.toc()

                if ((self.config['print_freq']>0) and (i % self.config['print_freq'] == 0)) or (i+1)==len(train_loader):
                    self.log('[{0}/{1}]\t'
                          '{batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                          '{data_time.val:.4f} ({data_time.avg:.4f})\t'
                          '{loss.val:.3f} ({loss.avg:.3f})\t'
                          '{acc.val:.2f} ({acc.avg:.2f})'.format(
                        i, len(train_loader), batch_time=batch_time,
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
