from runPy_iBatchLearn import main

gpuid=3

### Permuted-MNIST incremental class
# agent_name="Naive_Rehearsal_4000"
n_permutation=10
repeat=1
schedule=3 # Iteration
batch_size=2048

# agent_name="Fed_Memory_4000"
# arg_input = "iBatchLearn.py --gpuid {gpuid} --repeat {repeat} --incremental_class --optimizer Adam    --n_permutation {n_permutation} --force_out_dim 100 --schedule {schedule} --batch_size {batch_size} --model_name MLP1000 --agent_type customization  --agent_name  {agent_name} --lr 0.0001          | tee ${OUTDIR}/Naive_Rehearsal_4000.log".format(gpuid=gpuid, repeat=repeat, n_permutation=n_permutation, schedule=schedule, batch_size=batch_size, agent_name=agent_name, OUTDIR="outputs/permuted_MNIST_incremental_class")


agent_name="Fed_Memory_4000"
# agent_name="Naive_Rehearsal_4000"
arg_input = "iBatchLearn.py --gpuid {gpuid} --repeat {repeat} --optimizer Adam --n_permutation {n_permutation} --no_class_remap --force_out_dim 10 --schedule {schedule} --batch_size {batch_size} --model_name MLP1000 --agent_type customization  --agent_name  {agent_name} --lr 0.0001".format(gpuid=gpuid, repeat=repeat, n_permutation=n_permutation, schedule=schedule, batch_size=batch_size, agent_name=agent_name)
# python -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --n_permutation 10 --no_class_remap --force_out_dim 10 --schedule $SCHEDULE --batch_size $BS --model_name MLP1000 --agent_type customization  --agent_name Naive_Rehearsal_4000   --lr 0.0001          | tee ${OUTDIR}/Naive_Rehearsal_4000.log

arg_list = arg_input.split("|")[0].split()

main(arg_list)
