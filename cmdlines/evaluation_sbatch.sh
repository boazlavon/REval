#!/bin/sh -x
export PARTITION=gpu-ai
export GPUS_COUNT=1
export CPUS_COUNT=1
export MODEL_NAME=google/gemma-2-2b-it
model_name=$(echo $MODEL_NAME | sed 's_/_\__g')
export RUN="${model_name}_${GPUS_COUNT}_gpus"
export PYTHONPATH=$PYTHONPATH:/a/home/cc/students/cs/boazlavon/code/custom-trepan-xpy/traces_dumper
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export OUTPUT_FILE_PATH="output/evaluation.state.cot.$RUN.$PARTITION.out"
sbatch \
-c $CPUS_COUNT \
-G $GPUS_COUNT \
--partition=$PARTITION \
--output $OUTPUT_FILE_PATH \
--error $OUTPUT_FILE_PATH \
--job-name=$RUN \
--nodelist=n-351 \
slurms/evaluation.slurm 
