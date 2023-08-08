import argparse
import csv
import json
import os
import sys
import subprocess
import time
from threading import Thread


class GPUUtilTracker():
    def __init__(self):
        self.max_gpu_mem_usage = []
        self.stop = False
    
    def get_results(self):
        return self.max_gpu_mem_usage

    def terminate(self):
        self.stop = True

    def run(self):
        cmd = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
        print(cmd)
        while(True):
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)    # gpu info collect
            out, err = proc.communicate()
            print(out, err)
            gpu_mem_usage = [float(n) for n in out.decode('utf-8').strip().split('\n')]
            print(gpu_mem_usage)
            if len(self.max_gpu_mem_usage) == 0:
                self.max_gpu_mem_usage = gpu_mem_usage
            else:
                for i in range(len(self.max_gpu_mem_usage)):
                    self.max_gpu_mem_usage[i] = gpu_mem_usage[i] if gpu_mem_usage[i] > self.max_gpu_mem_usage[i] else self.max_gpu_mem_usage[i]
            if self.stop:
                break
            time.sleep(5)


class Benchmark():
    def __init__(self, model_name, input_len, output_len, num_run, num_decoder_layer, num_header, size_per_header, max_batch_size, vocab_size, tensor_para_size=8):
        self.model_name = model_name
        self.input_len = input_len
        self.output_len = output_len
        self.num_run = num_run
        self.num_decoder_layer = num_decoder_layer
        self.num_header = num_header
        self.size_per_header = size_per_header
        self.gpu_mem_footprint = []
        self.max_batch_size = max_batch_size
        self.vocab_size = vocab_size
        self.tensor_para_size = tensor_para_size
        self.server_log = f"{self.model_name}_inference_server.log"
        self.client_log = f"{self.model_name}_client.log"
        self.data_points = []

    def cal_num_params(self):
        hidden_size = self.num_header * self.size_per_header
        return 12 * self.num_decoder_layer * hidden_size * hidden_size * \
            (1 + 13 / (12 * hidden_size) + (self.vocab_size + 2048) / (12 * self.num_decoder_layer * hidden_size))

    # get client result
    def parse_log(self, batch_size):
        cmd = f"tail -n 1 {self.client_log} | grep -Eo '[+-]?[0-9]+([.][0-9]+)?'"
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate()
        avg_latency = float(out.strip())
        cmd = f"tail -2 {self.client_log} | head -n 1"
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate()
        print(out, err)
        latencies = ["{:.2f}".format(float(n)) for n in out.decode('utf-8').strip('\n').strip(']').strip('[').split(", ")]
        
        # cmd = f"(cat {self.server_log} | grep 'after allocation' | sort | awk '{{print $8}}' )"
        # print(cmd)
        # proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # out, err = proc.communicate()
        # print(out, err)
        # free_gpu_mem = [float(n) for n in out.decode('utf-8').strip().split('\n')]


        # cmd = f"(cat {self.server_log} | grep 'after allocation' | sort | awk '{{print $11}}' )"
        # print(cmd)
        # proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # out, err = proc.communicate()
        # print(out, err)
        # total_gpu_mem = [float(n) for n in out.decode('utf-8').strip().split('\n')]
        # assert len(free_gpu_mem) == len(total_gpu_mem)
        # print(free_gpu_mem)
        # print(total_gpu_mem)
        # gpu_mem_usage = ["{:.2f}".format(b - a) for b, a in zip(total_gpu_mem, free_gpu_mem)]
        return [batch_size] + latencies + [avg_latency]

    def call_once(self, batch_size):
        g_tracker = GPUUtilTracker()
        t = Thread(target=g_tracker.run)
        t.start()
        devices = "CUDA_VISIBLE_DEVICES=" + ",".join([ str(i) for i in range(self.tensor_para_size) ])
        cmd = f"{devices} bash /ft_workspace/fastertransformer_backend/llama_benchmark/launch_triton.sh -m {self.model_name} -b {batch_size} "
        print(cmd)

        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate()

        bs_latency = self.parse_log(batch_size)
        g_tracker.terminate()
        t.join()
        self.data_points.append(bs_latency + g_tracker.get_results())

    def start(self):
        print("Estimated num param: ", "{:,}".format(int(self.cal_num_params())))
        os.getenv('WORKSPACE')
        bs = 1
        while (bs <= self.max_batch_size):
            self.call_once(bs)
            bs = bs * 2

    def to_csv(self):
        import csv

        with open(f"{self.model_name}_perf.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.data_points) 