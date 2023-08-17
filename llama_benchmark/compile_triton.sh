#!/bin/bash

# this script should be executed in /be_workspace

PREFER_BATCH_SIZE=16
DYNAMIC_ENABLED=0

while getopts ":pb:d:" opt; do
  case $opt in
    pb)
      PREFER_BATCH_SIZE="$OPTARG"
      ;;
    d)
      DYNAMIC_ENABLED="$OPTARG"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      ;;
  esac
done

cd /be_workspace/dongyazhu/fastertransformer_backend/build
make -j12
cd /be_workspace
cp dongyazhu/fastertransformer_backend/build/libtriton_fastertransformer.so /opt/tritonserver/backends/fastertransformer
bash ./dongyazhu/fastertransformer_backend/llama_benchmark/launch_triton.sh -d $DYNAMIC_ENABLED -pb $PREFER_BATCH_SIZE
