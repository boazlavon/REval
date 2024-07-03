#!/bin/bash

CONFIG="$1"

if ! [ -f $CONFIG ]; then
    echo "No $CONFIG file found"
    exit 1
fi

# dirty hack for tmux, so that `jq` can be found
PATH=$PATH:$HOME/anaconda3/bin

CUDA_VISIBLE_DEVICES=$(cat $CONFIG | jq -r '.gpu_ordinals | join(",")') \
python -m vllm.entrypoints.openai.api_server \
    --model $(cat $CONFIG | jq -r '.model_path') \
    --dtype auto \
    --port  $(cat $CONFIG | jq -r '.port') \
    --tensor-parallel-size $(cat $CONFIG | jq -r '.num_gpus') \
    --disable-log-requests \
