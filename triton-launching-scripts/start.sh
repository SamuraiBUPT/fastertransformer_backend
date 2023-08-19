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

# TODO finish

# compile triton backend
cd /be_workspace/dongyazhu/fastertransformer_backend/build
cmake \
      -D CMAKE_EXPORT_COMPILE_COMMANDS=1 \
      -D CMAKE_BUILD_TYPE=Release \
      -D ENABLE_FP8=OFF \
      -D CMAKE_INSTALL_PREFIX=/opt/tritonserver \
      -D TRITON_COMMON_REPO_TAG="r23.04" \
      -D TRITON_CORE_REPO_TAG="r23.04" \
      -D TRITON_BACKEND_REPO_TAG="r23.04" \
      ..
make -j12
cd /be_workspace
cp dongyazhu/fastertransformer_backend/build/libtriton_fastertransformer.so /opt/tritonserver/backends/fastertransformer

# compile custom batching strategy
cd dongyazhu/fastertransformer_backend/src/batching_strategies/build
cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install ..
make install

cd /be_workspace
bash ./dongyazhu/fastertransformer_backend/llama_benchmark/launch_triton.sh -d $DYNAMIC_ENABLED -pb $PREFER_BATCH_SIZE