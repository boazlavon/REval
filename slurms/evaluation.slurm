#!/bin/sh
export CUDA_VISIBLE_DEVICES=1
export NVIDIA_GPU_VISABLE=1
checkpoint=
if [ $CHECKPOINT_PATH ]; then
  checkpoint="--checkpoint_path $CHECKPOINT_PATH"
fi

if [ $GPUS_COUNT -gt 1 ]; then
  accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=$GPUS_COUNT evaluation.py run
else
  #accelerate launch --mixed_precision=fp16 --num_processes=$GPUS_COUNT evaluation.py run
  python evaluation.py run
fi
