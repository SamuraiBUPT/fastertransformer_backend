#!/usr/bin/python

import argparse
import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
import time
import re
import asyncio
import requests
import random
import os
from tqdm import tqdm
from transformers import AutoTokenizer
from functools import partial

from metric_utils import Trace
from typing import Tuple

tokenizer = AutoTokenizer.from_pretrained("gpt2")

def create_inference_server_client(protocol, url, concurrency, verbose):
    client_util = httpclient if protocol == "http" else grpcclient
    if protocol == "http":
        return client_util.InferenceServerClient(url,
                                                concurrency=concurrency,
                                                verbose=verbose)
    elif protocol == "grpc":
        return client_util.InferenceServerClient(url,
                                                verbose=verbose)


def clean_output(output):
    output = re.sub(r'(<s>)+', '<s>', output)  # 替换重复的<s>
    output = output.replace('<s>', '\n')  # 将<s>替换为换行符
    return output
    
# m_dataset = BuildDataset()  # build dataset here

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w',
                        '--coca_directory',
                        action="store_true",
                        required=False,
                        default="./dataset/sources.txt",
                        help='The input file of COCA dataset')
    parser.add_argument('-c',
                        '--cc_directory',
                        action="store_true",
                        required=False,
                        default="./dataset/warc.paths",
                        help='The input file of COCA dataset')

    parser.add_argument('-beam',
                        '--beam_width',
                        type=int,
                        default=1,
                        help='beam width.')
    parser.add_argument('-topk',
                        '--topk',
                        type=int,
                        default=1,
                        required=False,
                        help='topk for sampling')
    parser.add_argument('-topp',
                        '--topp',
                        type=float,
                        default=0.0,
                        required=False,
                        help='topp for sampling')
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        help='Inference server URL.')
    parser.add_argument(
        '-i',
        '--protocol',
        type=str,
        required=False,
        default='grpc',
        help='Protocol ("http"/"grpc") used to ' +
        'communicate with inference service. Default is "http".')
    parser.add_argument('--return_log_probs',
                        action="store_true",
                        default=False,
                        required=False,
                        help='return the cumulative log probs and output log probs or not')

    args = parser.parse_args()

    if (args.protocol != "http") and (args.protocol != "grpc"):
        print("unexpected protocol \"{}\", expects \"http\" or \"grpc\"".format(args.protocol))
        exit(1)
    if args.url is None:
        args.url = "localhost:8000" if args.protocol == "http" else "localhost:8001"

    request_size = 1
    batch_size = 1
    
    model_name = "ensemble"
    with create_inference_server_client(args.protocol,
                                        args.url,
                                        concurrency=1,
                                        verbose=args.verbose) as client:
        
        trace = Trace(args, time_period=100, text_directory='./dataset/cc_dataset/txt')
        

        def warm_up(_inputs):
            # warm up
            start_time = time.time()
            result = client.infer(model_name, _inputs)
            whole_time = time.time() - start_time
            return whole_time, result
        
        def sync_request_pull_http(_model_name, _inputs):
            pass    # TODO: implement this function
        
        
        def sync_request_pull_grpc(_model_name, 
                                   req_delay_pull_time,
                                   max_request=100) -> float:
            user_data = []
            pushed_request = 0
            req_sequence_idx = 0
            incident = 5
            req_delay_pull_time = 5.5 * 4.1 * 2/3
            print("=================================================")
            print(f"[INFO] max_request of this epoch: {max_request}")
            print("=================================================")
            request_sequence = trace.request_time_stamp
            # print(f"[INFO] request_sequence: {request_sequence}")
            
            # grpc callback
            def callback(user_data, result, error):
                if error:
                    user_data.append(error)
                else:
                    user_data.append(result)
                    
            start_time = time.time()
            while pushed_request <= max_request:
                start_time_pull_request = time.time()
                request_parallelism = request_sequence[req_sequence_idx]    # number of requests to pull
                
                selected_input_list = [] # it will be a 3d list: [ [[input_tensor 1]], [[input_tensor 2]], ... ]
                
                # prepare the list of input tensors (or requests)
                for offset in range(1, request_parallelism+1):
                    selected_text = trace.text_pool[pushed_request + offset]   # fetch text like: ['text A']
                    selected_text_tensor = trace.get_single_request_input(selected_text) # get the input tensor, like: [['text']]
                    client.async_infer(model_name=_model_name, 
                                        inputs=selected_text_tensor, 
                                        callback=partial(callback, user_data))
                    
                print(f"[INFO] {request_parallelism} requests sent")
                    
                pushed_request += request_parallelism
                req_sequence_idx += 1
                
                period_time_pull_request = time.time() - start_time_pull_request
                print(f"[TIME] request pull time: {period_time_pull_request} s")
                
                if req_sequence_idx % incident != 0:
                    # [[4], gap_time_here[9], ...]
                    time.sleep(req_delay_pull_time)    # gap 0.01 s to pull next requests
                else:
                    # [[4, 9, 8, 3, 5], gap_time_here[]]
                    time.sleep(req_delay_pull_time)     # gap 0.1 s to pull next requests
            period_time_waiting_on = 0
            while True:
                if len(user_data)!=pushed_request:
                    continue
                else:
                    period_time_waiting_on = time.time() - start_time
                    print(f"[TIME] inference latency with waiting on: {period_time_waiting_on} s")
                    break
            
            period_time_waiting_off = period_time_waiting_on - req_delay_pull_time*req_sequence_idx
            print(f"[TIME] inference latency with waiting off: {period_time_waiting_off} s")
            return period_time_waiting_on, period_time_waiting_off

        try:
            input_warm = trace.get_single_request_input(trace.text_pool[0]) # get the input tensor, like: [['text']]

            # warm up
            timme_0, result = warm_up(input_warm)
            print("done")
            output0 = result.as_numpy("OUTPUT_0")

            print("============After ensemble============")
            print(output0)
            print(result.as_numpy("sequence_length"))

            cleaned_outputs = [clean_output(str(output, 'utf-8')) for output in output0]

            print("============String context============")
            for i, output in enumerate(cleaned_outputs):
                print(f'Output {i}:')
                print(output)

            print(f"Warm up Inference time: {(timme_0)} s/request")

            
            # latency test on baseline and dynamic batching
            # WARN: max_batch_size = 1024, each request is 8, the max request_parallelism is 128
            
            # request_sequence = trace.request_time_stamp # [4, 9, 8, 1, 2, ...]
            max_request = [100, 150, 200, 250, 300, 350]
            
            waiting_on_collection = []
            waiting_off_collection = []
            for req_epoch in max_request:
                req_delay_pull_time = 1
                time_on, time_off = sync_request_pull_grpc(model_name, req_delay_pull_time, max_request=req_epoch)
                waiting_on_collection.append(time_on)
                waiting_off_collection.append(time_off)
                
            for k, v in enumerate(waiting_on_collection):
                print(f"[INFO] max_request: {max_request[k]}, waiting_on: {v}, waiting_off: {waiting_off_collection[k]}")
                

            # # visualize
            # import matplotlib.pyplot as plt
            # plt.figure(figsize=(10, 5))
            # plt.subplot(1, 2, 1)
            # plt.plot(req_para_range, latency_per_req_collection)
            # plt.xlabel('request_parallelism')
            # plt.ylabel('latency_per_req')
            # plt.subplot(1, 2, 2)    
            # plt.plot(req_para_range, latency_epoch_collection)
            # plt.xlabel('request_parallelism')
            # plt.ylabel('latency_epoch')

            # plt.savefig(f'./plot_result/latency.png')


            # if args.return_log_probs:
            #     print(result.as_numpy("cum_log_probs"))
            #     print(result.as_numpy("output_log_probs"))
                
        except Exception as e:
            print(e)

