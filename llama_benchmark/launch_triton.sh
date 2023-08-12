#!/bin/bash

# our work:
# tensor_para_size: 2 (2-GPU)
# pipe_para_size: 1 (1-node)

# input parameters:
# $1: MODEL_NAME: llama or vicuna

# WARN: This script should be run in docker container

# echo $(pwd) # where you run this script

# default parameters
MODEL_NAME="llama"
MAX_BATCH_SIZE=1024
TENSOR_PARA=2
ENABLE_DYNAMIC_BATCH="0"
PREFER_BATCH_SIZE=50

# recv parameters
while getopts m:t:d:pb: option
do
    case "${option}"
    in
        m) MODEL_NAME=${OPTARG};;
        t) TENSOR_PARA=${OPTARG};;
        d) ENABLE_DYNAMIC_BATCH=${OPTARG};;
        pb) PREFER_BATCH_SIZE=${OPTARG};;
    esac
done

# you can run this script like this: 
#
# bash fastertransformer_backend/llama_benchmark/launch_triton.sh -d 1 -pb 4

if [ -d "triton-model-store" ]; then
    echo "triton-model-store already exists."
else
    mkdir triton-model-store
fi

# copy the model directory to triton-model-store
cp fastertransformer_backend/all_models/${MODEL_NAME} triton-model-store/ -r

# prepare the config.pbtxt
echo '
name: "fastertransformer"
backend: "fastertransformer"
default_model_filename: "llama"
max_batch_size: '"${MAX_BATCH_SIZE}"'
' > ./triton-model-store/${MODEL_NAME}/fastertransformer/config.pbtxt

if [ "$ENABLE_DYNAMIC_BATCH" = "true" ] || [ "$ENABLE_DYNAMIC_BATCH" = "1" ]; then
    echo "Dynamic batch enabled."
    echo '
dynamic_batching {
    preferred_batch_size: [ '"${PREFER_BATCH_SIZE}"' ]
    max_queue_delay_microseconds: 110400000
}
    ' >> ./triton-model-store/${MODEL_NAME}/fastertransformer/config.pbtxt
else
    echo "Dynamic batch disabled."
fi

echo '
model_transaction_policy {
  decoupled: False
}

batch_input [
  {
    kind: BATCH_ITEM_SHAPE
    target_name: "input_ids_item_shape"
    data_type: TYPE_INT32
    source_input: "input_ids"
  }
]

input [
  {
    name: "input_ids"
    data_type: TYPE_UINT32
    dims: [ -1 ]
    allow_ragged_batch: true
  },
  {
    name: "start_id"
    data_type: TYPE_UINT32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "end_id"
    data_type: TYPE_UINT32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "input_lengths"
    data_type: TYPE_UINT32
    dims: [ 1 ]
    reshape: { shape: [ ] }
  },
  {
    name: "request_output_len"
    data_type: TYPE_UINT32
    dims: [ -1 ]
  },
  {
    name: "runtime_top_k"
    data_type: TYPE_UINT32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "runtime_top_p"
    data_type: TYPE_FP32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "beam_search_diversity_rate"
    data_type: TYPE_FP32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "temperature"
    data_type: TYPE_FP32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "len_penalty"
    data_type: TYPE_FP32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "repetition_penalty"
    data_type: TYPE_FP32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "random_seed"
    data_type: TYPE_UINT64
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "is_return_log_probs"
    data_type: TYPE_BOOL
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "beam_width"
    data_type: TYPE_UINT32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "bad_words_list"
    data_type: TYPE_INT32
    dims: [ 2, -1 ]
    optional: true
  },
  {
    name: "stop_words_list"
    data_type: TYPE_INT32
    dims: [ 2, -1 ]
    optional: true
  },
  {
    name: "prompt_learning_task_name_ids"
    data_type: TYPE_UINT32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "top_p_decay"
    data_type: TYPE_FP32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "top_p_min"
    data_type: TYPE_FP32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: "top_p_reset_ids"
    data_type: TYPE_UINT32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  }
]
output [
  {
    name: "output_ids"
    data_type: TYPE_UINT32
    dims: [ -1, -1 ]
  },
  {
    name: "sequence_length"
    data_type: TYPE_UINT32
    dims: [ -1 ]
  },
  {
    name: "cum_log_probs"
    data_type: TYPE_FP32
    dims: [ -1 ]
  },
  {
    name: "output_log_probs"
    data_type: TYPE_FP32
    dims: [ -1, -1 ]
  }
]
instance_group [
  {
    count: 1
    kind: KIND_CPU
  }
]
parameters {
  key: "tensor_para_size"
  value: {
    string_value: "'"${TENSOR_PARA}"'"
  }
}
parameters {
  key: "pipeline_para_size"
  value: {
    string_value: "1"
  }
}
parameters {
  key: "data_type"
  value: {
    string_value: "fp16"
  }
}
parameters {
  key: "model_type"
  value: {
    string_value: "Llama"
  }
}
parameters {
  key: "model_checkpoint_path"
  value: {
    string_value: "./models/'"${MODEL_NAME}"'/2-gpu"
  }
}
parameters {
  key: "enable_custom_all_reduce"
  value: {
    string_value: "0"
  }
}
' >> ./triton-model-store/${MODEL_NAME}/fastertransformer/config.pbtxt


/opt/tritonserver/bin/tritonserver --model-repository=./triton-model-store/${MODEL_NAME}/