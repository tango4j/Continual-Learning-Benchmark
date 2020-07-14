from runPy_iBatchLearn import main
import sys

gpuid = int(sys.argv[1])
# gpuid=3

### Permuted-MNIST incremental class
# agent_name="Naive_Rehearsal_4000"
n_permutation=10
# n_permutation=2
repeat=10 # How many experiments
schedule=4 # Epoch
batch_size=128

# schedule=1 # Epoch
# batch_size=2048


learning_rate=0.001
# model_name="MLP1000"
model_name="MLP1000_img_sz"
# agent_name="Fed_Memory_4000"
# arg_input = "iBatchLearn.py --gpuid {gpuid} --repeat {repeat} --incremental_class --optimizer Adam    --n_permutation {n_permutation} --force_out_dim 100 --schedule {schedule} --batch_size {batch_size} --model_name MLP1000 --agent_type customization  --agent_name  {agent_name} --lr 0.0001          | tee ${OUTDIR}/Naive_Rehearsal_4000.log".format(gpuid=gpuid, repeat=repeat, n_permutation=n_permutation, schedule=schedule, batch_size=batch_size, agent_name=agent_name, OUTDIR="outputs/permuted_MNIST_incremental_class")


# agent_name = "Noise_Rehearsal_4400"
# agent_name = "No_Rehearsal_4400"
agent_name = "Memory_Embedding_Rehearsal_4400" 
# agent_name = "Compressed_Memory_Embedding_Rehearsal_4400"

# agent_name = "Memory_Embedding_Rehearsal_1100" 
# agent_name = "Compressed_Memory_Embedding_Rehearsal_1100"

# agent_name="Naive_Rehearsal_1100"
# agent_name="Naive_Rehearsal_4400"
# model_name="MLP1000"


arg_input = "iBatchLearn.py --gpuid {gpuid} --repeat {repeat} --optimizer Adam --n_permutation {n_permutation} --no_class_remap --force_out_dim 10 --schedule {schedule} --batch_size {batch_size} --model_name {model_name} --agent_type customization  --agent_name  {agent_name} --lr {learning_rate}".format(gpuid=gpuid, repeat=repeat, n_permutation=n_permutation, schedule=schedule, batch_size=batch_size, model_name=model_name, agent_name=agent_name, learning_rate=learning_rate)
# python -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --n_permutation 10 --no_class_remap --force_out_dim 10 --schedule $SCHEDULE --batch_size $BS --model_name MLP1000 --agent_type customization  --agent_name Naive_Rehearsal_4000   --lr 0.0001          | tee ${OUTDIR}/Naive_Rehearsal_4000.log

arg_list = arg_input.split("|")[0].split()

main(arg_list)
