from runPy_iBatchLearn import main
import sys

gpuid = int(sys.argv[1])
# gpuid=3

### Permuted-MNIST incremental class
offline_training = ''
boost_scale = 0

############## DATASET SELECT ##############

dataset='EMNIST'

### CNN
mlp_mha=10
batch_size=600
print_freq=100

### pretrained 
# mlp_mha=11
# batch_size=256
# print_freq=50

# dataset='CIFAR100'
# batch_size=256


n_permutation=0
# n_permutation=10
first_split_size=10
other_split_size=10
# n_permutation=2
repeat=10 # How many experiments
# schedule=4 # Epoch
# batch_size=128

schedule=1 # Epoch
schedule=2 # Epoch
schedule=3 # Epoch
schedule=4 # Epoch
# schedule=5 # Epoch
# schedule=10 # Epoch
# schedule=20 # Epoch
# schedule=40 # Epoch

# schedule=1 # Epoch
# batch_size=2048


learning_rate=0.001


# agent_name="Fed_Memory_4000"
# arg_input = "iBatchLearn.py --gpuid {gpuid} --repeat {repeat} --incremental_class --optimizer Adam    --n_permutation {n_permutation} --force_out_dim 100 --schedule {schedule} --batch_size {batch_size} --model_name MLP1000 --agent_type customization  --agent_name  {agent_name} --lr 0.0001          | tee ${OUTDIR}/Naive_Rehearsal_4000.log".format(gpuid=gpuid, repeat=repeat, n_permutation=n_permutation, schedule=schedule, batch_size=batch_size, agent_name=agent_name, OUTDIR="outputs/permuted_MNIST_incremental_class")


# ---------Memory Embedding Rehearsal-------
# agent_name = "Noise_Rehearsal_4400"
# agent_name = "No_Rehearsal_4400"
# agent_name = "Memory_Embedding_Rehearsal_1100" 

# agent_name = "Memory_Embedding_Rehearsal_4400" 


# agent_name = "Model_Generating_Rehearsal_8800" 
# boost_scale = 1
agent_name = "Model_Generating_Rehearsal_4400" 
boost_scale = 1
# agent_name = "Model_Generating_Rehearsal_2200" 
# boost_scale = 1
# agent_name = "Model_Generating_Rehearsal_1100" 
# boost_scale = 1

model_type="mlp"
model_name="MLP1000_img_sz"

# ------------Naive_Rehearsal--------------

# agent_name="Naive_Rehearsal_1100"
# agent_name="Naive_Rehearsal_2200"
agent_name="Naive_Rehearsal_4400"
boost_scale = 1
model_type="cnn"
model_name="CNN1000_img_sz"

# model_type="pretrained"
# model_name="PRETRAINED1000_img_sz"
# offline_training = '--offline_training'

arg_input = "iBatchLearn.py --gpuid {gpuid} --print_freq {print_freq} --dataset {dataset} {offline_training} --repeat {repeat} --first_split_size {first_split_size} --boost_scale {boost_scale} --mlp_mha {mlp_mha} --other_split_size {other_split_size} --optimizer Adam --n_permutation {n_permutation}  --force_out_dim 10 --schedule {schedule} --batch_size {batch_size} --model_type {model_type} --model_name {model_name} --agent_type customization  --agent_name  {agent_name} --lr {learning_rate}".format(gpuid=gpuid, print_freq=print_freq, dataset=dataset, offline_training=offline_training, repeat=repeat, first_split_size=first_split_size, boost_scale=boost_scale, mlp_mha=mlp_mha, other_split_size=other_split_size, n_permutation=n_permutation, schedule=schedule, batch_size=batch_size, model_type=model_type, model_name=model_name, agent_name=agent_name, learning_rate=learning_rate)
# python -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --n_permutation 10 --no_class_remap --force_out_dim 10 --schedule $SCHEDULE --batch_size $BS --model_name MLP1000 --agent_type customization  --agent_name Naive_Rehearsal_4000   --lr 0.0001          | tee ${OUTDIR}/Naive_Rehearsal_4000.log

arg_list = arg_input.split("|")[0].split()

main(arg_list)
