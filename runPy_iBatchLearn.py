import os
import sys
import argparse
import torch
import numpy as np
from random import shuffle
from collections import OrderedDict
import dataloaders.base
from dataloaders.datasetGen import SplitGen, PermutedGen
import agents

import ipdb
import namegenerator
from datetime import datetime

import os

LOG_PATH="/sam/home/inctrl/Dropbox/Papers/_2020_internship/result_0706"


if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)

def getLogTag():
    dt = datetime.now()
    dt_str = dt.strftime("%m%d%Y-%H%M")
    LOG_TAG = namegenerator.gen() + "-" + dt_str
    return LOG_TAG

class Aprint:
    def __init__(self, log_tag, full_path):
        self.full_path = full_path
        self.log_tag = log_tag 
    
    def a_print(self, *args):
        tst = ''
        for x in args:
            tst += ' ' + str(x)
        
        print("[{}] ".format('-'.join(self.log_tag.split('-')[0:3])), tst)
        with open(self.full_path, 'a') as f:
            f.write(tst+'\n')


def writeDictLOG(my_dict, full_path):
    with open(full_path, 'a') as f:
        for key in my_dict.keys():
            vd = my_dict[key]
            for k2, val in vd.items():
                pass
            f.write("%s,%s\n"%(key, val))

def writeDictArgs(my_dict, full_path):
    with open(full_path, 'w') as f:
        for key in my_dict.keys():
            val = my_dict[key] 
            f.write("%s,%s\n"%(key, val))

def getImageShape(dataset):
    if dataset  == 'MNIST':
        image_shape = (1, 32, 32)
    elif dataset  == 'EMNIST':
        image_shape = (1, 32, 32)
    elif dataset  == 'CIFAR10':
        image_shape = (3, 32, 32)
    elif dataset  == 'CIFAR100':
        image_shape = (3, 32, 32)
    else:
        raise NotImplementedError 
    return image_shape 

def run(args):
    pI = args.pI
    args.image_shape = getImageShape(args.dataset)
    if not os.path.exists('outputs'):
        os.mkdir('outputs')

    # Prepare dataloaders
    train_dataset, val_dataset = dataloaders.base.__dict__[args.dataset](args.dataroot, args.train_aug)
    if args.n_permutation>0:
        train_dataset_splits, val_dataset_splits, task_output_space = PermutedGen(train_dataset, val_dataset,
                                                                             args.n_permutation,
                                                                             remap_class=not args.no_class_remap)
    else:
        train_dataset_splits, val_dataset_splits, task_output_space = SplitGen(train_dataset, val_dataset,
                                                                          first_split_sz=args.first_split_size,
                                                                          other_split_sz=args.other_split_size,
                                                                          rand_split=args.rand_split,
                                                                          remap_class=not args.no_class_remap)

    # ipdb.set_trace()
    # Prepare the Agent (model)
    agent_config = {'lr': args.lr, 'momentum': args.momentum, 'weight_decay': args.weight_decay,'schedule': args.schedule,
                    'model_type':args.model_type, 'model_name': args.model_name, 'model_weights':args.model_weights,
                    'out_dim':{'All':args.force_out_dim} if args.force_out_dim>0 else task_output_space,
                    'optimizer':args.optimizer,
                    'print_freq':args.print_freq, 
                    'log_tag': args.log_tag,
                    'gpuid': args.gpuid,
                    'reg_coef':args.reg_coef,
                    'pI':args.pI}
    
    agent_config.update(args.__dict__)
    
    agent = agents.__dict__[args.agent_type].__dict__[args.agent_name](agent_config)
    # agent.pI = args.pI
    pI.a_print(str(agent.model))
    pI.a_print('#parameter of model:',agent.count_parameter())

    # Decide split ordering
    task_names = sorted(list(task_output_space.keys()), key=int)
    pI.a_print('Task order:',task_names)
    if args.rand_split_order:
        shuffle(task_names)
        pI.a_print('Shuffled task order:', task_names)

    acc_table = OrderedDict()
    if args.offline_training:  # Non-incremental learning / offline_training / measure the upper-bound performance
        task_names = ['All']
        train_dataset_all = torch.utils.data.ConcatDataset(train_dataset_splits.values())
        val_dataset_all = torch.utils.data.ConcatDataset(val_dataset_splits.values())
        train_loader = torch.utils.data.DataLoader(train_dataset_all,
                                                   batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        val_loader = torch.utils.data.DataLoader(val_dataset_all,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

        agent.learn_batch(train_loader, val_loader)

        acc_table['All'] = {}
        acc_table['All']['All'] = agent.validation(val_loader)

    else:  # Incremental learning
        # Feed data to agent and evaluate agent's performance
        for i in range(len(task_names)):
            train_name = task_names[i]
            agent.train_info= (i, train_name)
            pI.a_print('======================',train_name,'=======================')
            train_loader = torch.utils.data.DataLoader(train_dataset_splits[train_name],
                                                        batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
            val_loader = torch.utils.data.DataLoader(val_dataset_splits[train_name],
                                                      batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

            if args.incremental_class:
                agent.add_valid_output_dim(task_output_space[train_name])

            # Learn
            agent.learn_batch(train_loader, val_loader)

            # Evaluate
            acc_table[train_name] = OrderedDict()
            for j in range(i+1):
                val_name = task_names[j]
                pI.a_print('validation split name:', val_name)
                val_data = val_dataset_splits[val_name] if not args.eval_on_train_set else train_dataset_splits[val_name]
                val_loader = torch.utils.data.DataLoader(val_data,
                                                         batch_size=args.batch_size, shuffle=False,
                                                         num_workers=args.workers)
                acc_table[val_name][train_name] = agent.validation(val_loader)
    
    # sys.stdout.close()
    return acc_table, task_names

def get_args(argv):
    # This function prepares the variables shared across demo.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                        help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--model_type', type=str, default='mlp', help="The type (mlp|lenet|vgg|resnet) of backbone network")
    parser.add_argument('--model_name', type=str, default='MLP', help="The name of actual model for the backbone")
    parser.add_argument('--force_out_dim', type=int, default=2, help="Set 0 to let the task decide the required output dimension")
    parser.add_argument('--agent_type', type=str, default='default', help="The type (filename) of agent")
    parser.add_argument('--agent_name', type=str, default='NormalNN', help="The class name of agent")
    parser.add_argument('--optimizer', type=str, default='SGD', help="SGD|Adam|RMSprop|amsgrad|Adadelta|Adagrad|Adamax ...")
    parser.add_argument('--dataroot', type=str, default='data', help="The root folder of dataset or downloaded data")
    parser.add_argument('--dataset', type=str, default='MNIST', help="MNIST(default)|EMNIST|CIFAR10|CIFAR100")
    parser.add_argument('--n_permutation', type=int, default=0, help="Enable permuted tests when >0")
    parser.add_argument('--isIncDomain', type=bool, default=True, help="Enable Incremental domain setting")
    parser.add_argument('--first_split_size', type=int, default=2)
    parser.add_argument('--other_split_size', type=int, default=2)
    parser.add_argument('--no_class_remap', dest='no_class_remap', default=False, action='store_true',
                        help="Avoid the dataset with a subset of classes doing the remapping. Ex: [2,5,6 ...] -> [0,1,2 ...]")
    parser.add_argument('--train_aug', dest='train_aug', default=False, action='store_true',
                        help="Allow data augmentation during training")
    parser.add_argument('--rand_split', dest='rand_split', default=False, action='store_true',
                        help="Randomize the classes in splits")
    parser.add_argument('--rand_split_order', dest='rand_split_order', default=False, action='store_true',
                        help="Randomize the order of splits")
    parser.add_argument('--workers', type=int, default=1, help="#Thread for dataloader")
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--schedule', nargs="+", type=int, default=[2],
                        help="The list of epoch numbers to reduce learning rate by factor of 0.1. Last number is the end epoch")
    parser.add_argument('--print_freq', type=float, default=100, help="Print the log at every x iteration")
    parser.add_argument('--model_weights', type=str, default=None,
                        help="The path to the file for the model weights (*.pth).")
    parser.add_argument('--reg_coef', nargs="+", type=float, default=[0.], help="The coefficient for regularization. Larger means less plasilicity. Give a list for hyperparameter search.")
    parser.add_argument('--eval_on_train_set', dest='eval_on_train_set', default=False, action='store_true',
                        help="Force the evaluation on train set")
    parser.add_argument('--offline_training', dest='offline_training', default=False, action='store_true',
                        help="Non-incremental learning by make all data available in one batch. For measuring the upperbound performance.")
    parser.add_argument('--repeat', type=int, default=1, help="Repeat the experiment N times")
    parser.add_argument('--incremental_class', dest='incremental_class', default=False, action='store_true',
                        help="The number of output node in the single-headed model increases along with new categories.")
    
    ### tango4j added 
    parser.add_argument('--log_tag', type=str, default=getLogTag(), help="Set the name of the log file")
    parser.add_argument('--n_head', type=int, default=5, help="Transformer arg n_head")
    parser.add_argument('--d_k', type=int, default=64, help="Transformer arg d_k")
    parser.add_argument('--d_v', type=int, default=64, help="Transformer arg d_v")
    parser.add_argument('--compress_data', type=int, default=1, help="If compress_data is 1, embedding output dim != image size")
    parser.add_argument('--max_pool_size', type=int, default=8, help="For compression, max_pool_size")
    parser.add_argument('--predict_dim_match', type=int, default=0, help="For compression, max_pool_size")
    parser.add_argument('--num_mha_layers', type=int, default=1, help="For compression, max_pool_size")
    # parser.add_argument('--do_task1_training', type=int, default=0, help="Do not train at the first task stage")
    # parser.add_argument('--do_task1_training', type=int, default=1, help="Do not train at the first task stage")
    parser.add_argument('--do_task1_training', type=int, default=2, help="Do not train at the first task stage")
    parser.add_argument('--mlp_mha', type=int, default=11, help="Use simple lp instead of MHA, \
                        -1 for bypass, \
                        0 for MHA, \
                        1 for mlp, \
                        2 for hybrid, \
                        3 for stacked frozen MLP-> plastic MHA,\
                        4 for stacked plastic MLP-> plastic MHA,\
                        5 for orthogonalization \
                        6 for orthogonalization but freeze only last layer \
                        7 for VAE style encoder \
                        8 for std fixed VAE-style encoder \
                        9 for std fixed No stat 8 control group \
                        10 for CNN std-scaled MGN encoder \
                        11 for pretrained std-scaled MGN encoder" )
    parser.add_argument('--pretrained_model_type', type=str, default='resnet18', help="Pretrained network with image net dataset")
    parser.add_argument('--orthogonal_init', type=bool, default=True, help="Initializee weights with orthogonal vectors")
    parser.add_argument('--log_path', type=str, default=LOG_PATH, help="Log path for log csv and txt files")
    # parser.add_argument('--mu', type=float, default=0.0, help="Mixing coeff for ML + MHA. Only valid if mlp_mha = 2")
    parser.add_argument('--mu', type=float, default=0.0, help="regularization coeff")
    # parser.add_argument('--lambda', type=float, default=0.0, help="regularization coeff")
    parser.add_argument('--img_sz', type=int, default=16, help="Image size (img_sz**2=Embedding Size)")
    parser.add_argument('--fixed_std', type=float, default=0.1, help="Image size (img_sz**2=Embedding Size)")
    parser.add_argument('--scale_std', type=bool, default=True, help="Image size (img_sz**2=Embedding Size)")
    # parser.add_argument('--scale_std', type=bool, default=False, help="Image size (img_sz**2=Embedding Size)")
    parser.add_argument('--no_random_prediction', type=bool, default=True, help="Image size (img_sz**2=Embedding Size)")
    parser.add_argument('--boost_scale', type=int, default=1, help="Image size (img_sz**2=Embedding Size)")
    # EXP_NOTE='@ stacked MLP-MHA training but qkv are all the different mlp + big mha model 3 laerys + nh 32'
    # EXP_NOTE='@ stacked MLP-MHA training + MLP frozen + plastic MHA  + random_proj + 16dim + nh2,dk32'
    # EXP_NOTE='@  MLP only + 16 dim  fixed orthogonalization test sanity check mu=0.0'
    EXP_NOTE='@  resnet18 on CIFAR100'
    # EXP_NOTE='@ Debugging'
    
    parser.add_argument('--exp_note', type=str, 
                default=EXP_NOTE, ### @
                   help='Note for the model/experiment')
   
   # parser.add_argument('--pI', default=getLogTag(), help="print funciton")
    args = parser.parse_args(argv)
    return args

def main(argv_list):
    sys.argv = argv_list
    args = get_args(sys.argv[1:])
    reg_coef_list = args.reg_coef
    avg_final_acc = {}

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    # The for loops over hyper-paramerters or repeats
    pI = Aprint(args.log_tag, '{}/'.format(args.log_path)+args.log_tag+"_stdout.txt")
    args.pI = pI
    writeDictArgs(args.__dict__,  '{}/'.format(args.log_path)+args.log_tag+"_args.csv")
    if args.mlp_mha == 0:
        assert args.img_sz == 32, "Original MHA should have the same size input/output"

    for reg_coef in reg_coef_list:
        args.reg_coef = reg_coef
        avg_final_acc[reg_coef] = np.zeros(args.repeat)
        for r in range(args.repeat):
            # Run the experiment
            acc_table, task_names = run(args)
            print(acc_table)
            writeDictLOG(acc_table,  '{}/'.format(args.log_path)+args.log_tag+"_dict.csv")
            # Calculate average performance across tasks
            # Customize this part for a different performance metric
            avg_acc_history = [0] * len(task_names)
            for i in range(len(task_names)):
                train_name = task_names[i]
                cls_acc_sum = 0
                for j in range(i + 1):
                    val_name = task_names[j]
                    cls_acc_sum += acc_table[val_name][train_name]
                avg_acc_history[i] = cls_acc_sum / (i + 1)
                pI.a_print('Task', train_name, 'average acc:', avg_acc_history[i])

            # Gather the final avg accuracy
            avg_final_acc[reg_coef][r] = avg_acc_history[-1]

            # Print the summary so far
            pI.a_print('===Summary of experiment repeats:',r+1,'/',args.repeat,'===')
            pI.a_print('The regularization coefficient:', args.reg_coef)
            pI.a_print('The last avg acc of all repeats:', avg_final_acc[reg_coef])
            pI.a_print('mean:', avg_final_acc[reg_coef].mean(), 'std:', avg_final_acc[reg_coef].std())
    for reg_coef,v in avg_final_acc.items():
        pI.a_print('reg_coef:', reg_coef,'mean:', avg_final_acc[reg_coef].mean(), 'std:', avg_final_acc[reg_coef].std())

if __name__ == '__main__':
    print('Main script is executed.')
    # ipdb.set_trace()

