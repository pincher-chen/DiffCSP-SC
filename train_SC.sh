#!/bin/sh

export CUDA_VISIBLE_DEVICES=3
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMP_NUM_THREADS=1

python diffcsp/run.py model=guidance data=property data.subdir=SuperCon data.prop=logtc data.task=regression data.opt_target=1 exptag=SuperCon_logtc expname=guidance

python scripts/optimization_sr.py --model_type diffcsp  --model_path ${PWD}/singlerun/SuperCon_logtc/guidance --uncond_path ${PWD}/singlerun/2023-04-18/pure_pretrain --step_lr 1e-5 --aug 50

#python scripts/compute_sr.py --root_path ${PWD}/singlerun/SuperCon_logtc/guidance --train_csv ${PWD}/data/properties/SuperCon/train.csv --property logtc --eval_model_path $PWD$/singlerun/SuperCon_logtc/prediction
python scripts/eval_optimization.py --dir ${PWD}/singlerun/SuperCon_logtc
