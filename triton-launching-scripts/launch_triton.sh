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
PREFER_BATCH_SIZE=32


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

DELAY=1000000
# you can run this script like this: 
#
# bash fastertransformer_backend/llama_benchmark/launch_triton.sh -d 1 -pb 4

if [ -d "triton-model-store" ]; then
    echo "triton-model-store already exists."
else
    mkdir triton-model-store
fi

# copy the model directory to triton-model-store
cp dongyazhu/fastertransformer_backend/all_models/${MODEL_NAME} triton-model-store/ -r

# fastertransformer config.pbtxt
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
    max_queue_delay_microseconds: '"${DELAY}"'
}

    ' >> ./triton-model-store/${MODEL_NAME}/fastertransformer/config.pbtxt
else
    echo "Dynamic batch disabled."
fi

# put this into the dynamic batching zone
# parameters: {
#   key: "TRITON_BATCH_STRATEGY_PATH", value: {string_value: "/be_workspace/dongyazhu/fastertransformer_backend/src/batching_strategies/build/install/batching/lora_batching/libtriton_lorabatching.so"}
# }
# parameters { key: "lora_request_batchsize" value: {string_value: "20"}}

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
    name: "lora_type"
    data_type: TYPE_UINT32
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
    string_value: "./models/'"${MODEL_NAME}"'/'"${TENSOR_PARA}"'-gpu"
  }
}
parameters {
  key: "enable_custom_all_reduce"
  value: {
    string_value: "0"
  }
}
' >> ./triton-model-store/${MODEL_NAME}/fastertransformer/config.pbtxt

# ensemble config.pbtxt
echo '
name: "ensemble"
platform: "ensemble"
max_batch_size: 1024

input [
  {
    name: "INPUT_0"
    data_type: TYPE_STRING
    dims: [ -1 ]
  },
  {
    name: "INPUT_1"
    data_type: TYPE_UINT32
    dims: [ -1 ]
  },
  {
   name: "INPUT_2"
   data_type: TYPE_STRING
   dims: [ -1 ]
  },
  {
   name: "INPUT_3"
   data_type: TYPE_STRING
   dims: [ -1 ]
  },
  {
    name: "runtime_top_k"
    data_type: TYPE_UINT32
    dims: [ 1 ]
    optional: true
  },
  {
    name: "runtime_top_p"
    data_type: TYPE_FP32
    dims: [ 1 ]
    optional: true
  },
  {
    name: "beam_search_diversity_rate"
    data_type: TYPE_FP32
    dims: [ 1 ]
    optional: true
  },
  {
    name: "temperature"
    data_type: TYPE_FP32
    dims: [ 1 ]
    optional: true
  },
  {
    name: "len_penalty"
    data_type: TYPE_FP32
    dims: [ 1 ]
    optional: true
  },
  {
    name: "repetition_penalty"
    data_type: TYPE_FP32
    dims: [ 1 ]
    optional: true
  },
  {
    name: "random_seed"
    data_type: TYPE_UINT64
    dims: [ 1 ]
    optional: true
  },
  {
    name: "is_return_log_probs"
    data_type: TYPE_BOOL
    dims: [ 1 ]
    optional: true
  },
  {
    name: "beam_width"
    data_type: TYPE_UINT32
    dims: [ 1 ]
    optional: true
  },
  {
    name: "start_id"
    data_type: TYPE_UINT32
    dims: [ 1 ]
    optional: true
  },
  {
    name: "end_id"
    data_type: TYPE_UINT32
    dims: [ 1 ]
    optional: true
  },
  {
    name: "lora_type"
    data_type: TYPE_UINT32
    dims: [ 1 ]
    optional: true
  },
  {
    name: "prompt_learning_task_name_ids"
    data_type: TYPE_UINT32
    dims: [ 1 ]
    optional: true
  },
  {
    name: "top_p_decay"
    data_type: TYPE_FP32
    dims: [ 1 ]
    optional: true
  },
  {
    name: "top_p_min"
    data_type: TYPE_FP32
    dims: [ 1 ]
    optional: true
  },
  {
    name: "top_p_reset_ids"
    data_type: TYPE_UINT32
    dims: [ 1 ]
    optional: true
  }
]
output [
  {
    name: "OUTPUT_0"
    data_type: TYPE_STRING
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
ensemble_scheduling {
  step [
    {
      model_name: "preprocessing"
      model_version: -1
      input_map {
        key: "QUERY"
        value: "INPUT_0"
      }
      input_map {
        key: "REQUEST_OUTPUT_LEN"
        value: "INPUT_1"
      }
      input_map {
        key: "BAD_WORDS_DICT"
        value: "INPUT_2"
      }
      input_map {
        key: "STOP_WORDS_DICT"
        value: "INPUT_3"
      }
      output_map {
        key: "INPUT_ID"
        value: "_INPUT_ID"
      }
      output_map {
        key: "BAD_WORDS_IDS"
        value: "_BAD_WORDS_IDS"
      }
      output_map {
        key: "STOP_WORDS_IDS"
        value: "_STOP_WORDS_IDS"
      }
      output_map {
        key: "REQUEST_INPUT_LEN"
        value: "_REQUEST_INPUT_LEN"
      }
      output_map {
        key: "REQUEST_OUTPUT_LEN"
        value: "_REQUEST_OUTPUT_LEN"
      }
    },
    {
      model_name: "fastertransformer"
      model_version: -1
      input_map {
        key: "input_ids"
        value: "_INPUT_ID"
      }
      input_map {
        key: "input_lengths"
        value: "_REQUEST_INPUT_LEN"
      }
      input_map {
        key: "request_output_len"
        value: "_REQUEST_OUTPUT_LEN"
      }
      input_map {
        key: "prompt_learning_task_name_ids"
        value: "prompt_learning_task_name_ids"
      }
      input_map {
          key: "runtime_top_k"
          value: "runtime_top_k"
      }
      input_map {
          key: "runtime_top_p"
          value: "runtime_top_p"
      }
      input_map {
          key: "beam_search_diversity_rate"
          value: "beam_search_diversity_rate"
      }
      input_map {
          key: "temperature"
          value: "temperature"
      }
      input_map {
          key: "len_penalty"
          value: "len_penalty"
      }
      input_map {
          key: "repetition_penalty"
          value: "repetition_penalty"
      }
      input_map {
          key: "random_seed"
          value: "random_seed"
      }
      input_map {
          key: "is_return_log_probs"
          value: "is_return_log_probs"
      }
      input_map {
          key: "beam_width"
          value: "beam_width"
      }
      input_map {
          key: "start_id"
          value: "start_id"
      }
      input_map {
          key: "end_id"
          value: "end_id"
      }
      input_map {
          key: "lora_type"
          value: "lora_type"
      }
      input_map {
          key: "stop_words_list"
          value: "_STOP_WORDS_IDS"
      }
      input_map {
          key: "bad_words_list"
          value: "_BAD_WORDS_IDS"
      }
      input_map {
        key: "top_p_decay"
        value: "top_p_decay"
      }
      input_map {
        key: "top_p_min"
        value: "top_p_min"
      }
      input_map {
        key: "top_p_reset_ids"
        value: "top_p_reset_ids"
      }
      output_map {
        key: "output_ids"
        value: "_TOKENS_BATCH"
      }
      output_map {
        key: "sequence_length"
        value: "sequence_length"
      }
      output_map {
        key: "cum_log_probs"
        value: "cum_log_probs"
      }
      output_map {
        key: "output_log_probs"
        value: "output_log_probs"
      }
    },
    {
      model_name: "postprocessing"
      model_version: -1
      input_map {
        key: "TOKENS_BATCH"
        value: "_TOKENS_BATCH"
      }
      input_map {
        key: "sequence_length"
        value: "sequence_length"
      }
      output_map {
        key: "OUTPUT"
        value: "OUTPUT_0"
      }
    }
  ]
}
' > ./triton-model-store/${MODEL_NAME}/ensemble/config.pbtxt



/opt/tritonserver/bin/tritonserver \
--model-repository=./triton-model-store/${MODEL_NAME}/ \
--trace-config triton,file=/be_workspace/dongyazhu/results/trace.json \
    --trace-config triton,log-frequency=50 \
    --trace-config rate=100 \
    --trace-config level=TIMESTAMPS \
    --trace-config count=100 