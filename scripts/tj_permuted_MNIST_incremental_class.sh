
#!/bin/bash

GPUID=$1
OUTDIR=outputs/permuted_MNIST_incremental_class
REPEAT=2
SCHEDULE=2
BS=1024
PYTHON=python
mkdir -p $OUTDIR
#$PYTHON -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --n_permutation 10 --force_out_dim 100 --schedule $SCHEDULE --batch_size $BS --model_name MLP1000                                                     --lr 0.0001  --offline_training  | tee ${OUTDIR}/Offline.log
#$PYTHON -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --n_permutation 10 --force_out_dim 100 --schedule $SCHEDULE --batch_size $BS --model_name MLP1000                                                     --lr 0.0001                      | tee ${OUTDIR}/Adam.log
#$PYTHON -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer SGD     --n_permutation 10 --force_out_dim 100 --schedule $SCHEDULE --batch_size $BS --model_name MLP1000                                                     --lr 0.001                       | tee ${OUTDIR}/SGD.log
#$PYTHON -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adagrad --n_permutation 10 --force_out_dim 100 --schedule $SCHEDULE --batch_size $BS --model_name MLP1000                                                     --lr 0.001                       | tee ${OUTDIR}/Adagrad.log
#$PYTHON -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --n_permutation 10 --force_out_dim 100 --schedule $SCHEDULE --batch_size $BS --model_name MLP1000 --agent_type customization  --agent_name EWC_online_mnist --lr 0.0001 --reg_coef 50  | tee ${OUTDIR}/EWC_online.log
#$PYTHON -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --n_permutation 10 --force_out_dim 100 --schedule $SCHEDULE --batch_size $BS --model_name MLP1000 --agent_type customization  --agent_name EWC_mnist        --lr 0.0001 --reg_coef 10  | tee ${OUTDIR}/EWC.log
#$PYTHON -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --n_permutation 10 --force_out_dim 100 --schedule $SCHEDULE --batch_size $BS --model_name MLP1000 --agent_type regularization --agent_name SI         --lr 0.0001 --reg_coef 0.3       | tee ${OUTDIR}/SI.log
#$PYTHON -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --n_permutation 10 --force_out_dim 100 --schedule $SCHEDULE --batch_size $BS --model_name MLP1000 --agent_type regularization --agent_name L2         --lr 0.0001 --reg_coef 0         | tee ${OUTDIR}/L2.log
$PYTHON -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --n_permutation 10 --force_out_dim 100 --schedule $SCHEDULE --batch_size $BS --model_name MLP1000 --agent_type customization  --agent_name Naive_Rehearsal_4000   --lr 0.0001          | tee ${OUTDIR}/Naive_Rehearsal_4000.log
#$PYTHON -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --n_permutation 10 --force_out_dim 100 --schedule $SCHEDULE --batch_size $BS --model_name MLP1000 --agent_type customization  --agent_name Naive_Rehearsal_16000  --lr 0.0001          | tee ${OUTDIR}/Naive_Rehearsal_16000.log
#$PYTHON -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --n_permutation 10 --force_out_dim 100 --schedule $SCHEDULE --batch_size $BS --model_name MLP1000 --agent_type regularization --agent_name MAS        --lr 0.0001 --reg_coef 0.003     | tee ${OUTDIR}/MAS.log
#$PYTHON -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer SGD     --n_permutation 10 --force_out_dim 100 --schedule $SCHEDULE --batch_size $BS --model_name MLP1000 --agent_type customization  --agent_name GEM_4000   --lr 0.1 --reg_coef 0.5          | tee ${OUTDIR}/GEM_4000.log
#$PYTHON -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer SGD     --n_permutation 10 --force_out_dim 100 --schedule $SCHEDULE --batch_size $BS --model_name MLP1000 --agent_type customization  --agent_name GEM_16000  --lr 0.1 --reg_coef 0.5          | tee ${OUTDIR}/GEM_16000.log
