#! /bin/sh
#export CUDA_VISIBLE_DEVICES="4,5,6,7"
checkpoint=
if [ $CHECKPOINT_PATH ]; then
  checkpoint="--checkpoint_path $CHECKPOINT_PATH"
fi

if [ $GPUS_COUNT -gt 1 ]; then
  accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=$GPUS_COUNT python evaluation.py run
else
  python evaluation.py run
fi
