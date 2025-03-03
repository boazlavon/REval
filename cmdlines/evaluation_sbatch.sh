#!/bin/sh -x
#export PARTITION=gpu-ai
export PARTITION=killable
#export PARTITION=gpu-ai
#export PARTITION=gpu-h100-killable
export GPUS_COUNT=1
export CPUS_COUNT=1
export MODEL_NAME=google/gemma-2-2b-it
model_name=$(echo $MODEL_NAME | sed 's_/_\__g')
export RUN="${model_name}_${GPUS_COUNT}_gpus"
export PYTHONPATH=$PYTHONPATH:/a/home/cc/students/cs/boazlavon/code/custom-trepan-xpy/traces_dumper
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export OUTPUT_FILE_PATH="output/evaluation.cov.cot.6.$RUN.$PARTITION.out"
sbatch \
-c $CPUS_COUNT \
-G $GPUS_COUNT \
--partition=$PARTITION \
--output $OUTPUT_FILE_PATH \
--error $OUTPUT_FILE_PATH \
--job-name=$RUN \
--nodelist=n-204 \
slurms/evaluation.slurm
#--nodelist=rack-bgw-dgx1 \
#--nodelist=n-100 \
#--nodelist=n-350 \
