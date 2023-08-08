#!/usr/bin/python

# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
import time
import re
import asyncio

from tritonclient.utils import np_to_triton_dtype
from typing import Tuple

FLAGS = None

START_LEN = 8
OUTPUT_LEN = 100
BATCH_SIZE = 8

start_id = 0
end_id = 1

def prepare_tensor(name, input, protocol):
    client_util = httpclient if protocol == "http" else grpcclient
    t = client_util.InferInput(
        name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)    # return ids
    return t

def create_inference_server_client(protocol, url, concurrency, verbose):
    client_util = httpclient if protocol == "http" else grpcclient
    if protocol == "http":
        return client_util.InferenceServerClient(url,
                                                concurrency=concurrency,
                                                verbose=verbose)
    elif protocol == "grpc":
        return client_util.InferenceServerClient(url,
                                                verbose=verbose)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        '-i',
        '--protocol',
        type=str,
        required=False,
        default='http',
        help='Protocol ("http"/"grpc") used to ' +
        'communicate with inference service. Default is "http".')
    parser.add_argument('--return_log_probs',
                        action="store_true",
                        default=False,
                        required=False,
                        help='return the cumulative log probs and output log probs or not')

    FLAGS = parser.parse_args()
    if (FLAGS.protocol != "http") and (FLAGS.protocol != "grpc"):
        print("unexpected protocol \"{}\", expects \"http\" or \"grpc\"".format(
            FLAGS.protocol))
        exit(1)

    if FLAGS.url is None:
        FLAGS.url = "localhost:8000" if FLAGS.protocol == "http" else "localhost:8001"

    # Run async requests to make sure backend handles request batches
    # correctly. We use just HTTP for this since we are not testing the
    # protocol anyway.

#    ######################
#    model_name = "preprocessing"
#    with create_inference_server_client(FLAGS.protocol,
#                                        FLAGS.url,
#                                        concurrency=1,
#                                        verbose=FLAGS.verbose) as client:
#        input0 = [
#                ["Blackhawks\n The 2015 Hilltoppers"],
#                ["Data sources you can use to make a decision:"],
#                ["\n if(angle = 0) { if(angle"],
#                ["GMs typically get 78% female enrollment, but the "],
#                ["Previous Chapter | Index | Next Chapter"],
#                ["Michael, an American Jew, called Jews"],
#                ["Blackhawks\n The 2015 Hilltoppers"],
#                ["Data sources you can use to make a comparison:"]
#                ]
#        input0_data = np.array(input0).astype(object)
#        output0_len = np.ones_like(input0).astype(np.uint32) * OUTPUT_LEN
#        bad_words_list = np.array([
#            ["Hawks, Hawks"],
#            [""],
#            [""],
#            [""],
#            [""],
#            [""],
#            [""],
#            [""]], dtype=object)
#        stop_words_list = np.array([
#            [""],
#            [""],
#            [""],
#            [""],
#            [""],
#            [""],
#            [""],
#            ["month, month"]], dtype=object)
#        inputs = [
#            prepare_tensor("QUERY", input0_data, FLAGS.protocol),
#            prepare_tensor("BAD_WORDS_DICT", bad_words_list, FLAGS.protocol),
#            prepare_tensor("STOP_WORDS_DICT", stop_words_list, FLAGS.protocol),
#            prepare_tensor("REQUEST_OUTPUT_LEN", output0_len, FLAGS.protocol),
#        ]
#
#        try:
#            result = client.infer(model_name, inputs)
#            output0 = result.as_numpy("INPUT_ID")
#            output1 = result.as_numpy("REQUEST_INPUT_LEN")
#            output2 = result.as_numpy("REQUEST_OUTPUT_LEN")
#            output3 = result.as_numpy("BAD_WORDS_IDS")
#            output4 = result.as_numpy("STOP_WORDS_IDS")
#            # output0 = output0.reshape([output0.shape[0], 1, output0.shape[1]]) # Add dim for beam width
#            print("============After preprocessing============")
#            print(output0, output1, output2)
#            print("===========================================\n\n\n")
#        except Exception as e:
#            print(e)
#
#    ######################
#    model_name = "fastertransformer"
#    with create_inference_server_client(FLAGS.protocol,
#                                        FLAGS.url,
#                                        concurrency=1,
#                                        verbose=FLAGS.verbose) as client:
#        runtime_top_k = (FLAGS.topk * np.ones([output0.shape[0], 1])).astype(np.uint32)
#        runtime_top_p = FLAGS.topp * np.ones([output0.shape[0], 1]).astype(np.float32)
#        beam_search_diversity_rate = 0.0 * np.ones([output0.shape[0], 1]).astype(np.float32)
#        temperature = 1.0 * np.ones([output0.shape[0], 1]).astype(np.float32)
#        len_penalty = 1.0 * np.ones([output0.shape[0], 1]).astype(np.float32)
#        repetition_penalty = 1.0 * np.ones([output0.shape[0], 1]).astype(np.float32)
#        random_seed = 0 * np.ones([output0.shape[0], 1]).astype(np.uint64)
#        is_return_log_probs = FLAGS.return_log_probs * np.ones([output0.shape[0], 1]).astype(np.bool)
#        beam_width = (FLAGS.beam_width * np.ones([output0.shape[0], 1])).astype(np.uint32)
#        start_ids = start_id * np.ones([output0.shape[0], 1]).astype(np.uint32)
#        end_ids = end_id * np.ones([output0.shape[0], 1]).astype(np.uint32)
#        prompt_learning_task_name_ids = 0 * np.ones([output0.shape[0], 1]).astype(np.uint32)
#        inputs = [
#            prepare_tensor("input_ids", output0, FLAGS.protocol),
#            prepare_tensor("input_lengths", output1, FLAGS.protocol),
#            prepare_tensor("request_output_len", output2, FLAGS.protocol),
#            prepare_tensor("runtime_top_k", runtime_top_k, FLAGS.protocol),
#            prepare_tensor("runtime_top_p", runtime_top_p, FLAGS.protocol),
#            prepare_tensor("beam_search_diversity_rate", beam_search_diversity_rate, FLAGS.protocol),
#            prepare_tensor("temperature", temperature, FLAGS.protocol),
#            prepare_tensor("len_penalty", len_penalty, FLAGS.protocol),
#            prepare_tensor("repetition_penalty", repetition_penalty, FLAGS.protocol),
#            prepare_tensor("random_seed", random_seed, FLAGS.protocol),
#            prepare_tensor("is_return_log_probs", is_return_log_probs, FLAGS.protocol),
#            prepare_tensor("beam_width", beam_width, FLAGS.protocol),
#            prepare_tensor("start_id", start_ids, FLAGS.protocol),
#            prepare_tensor("end_id", end_ids, FLAGS.protocol),
#            prepare_tensor("bad_words_list", output3, FLAGS.protocol),
#            prepare_tensor("stop_words_list", output4, FLAGS.protocol),
#        ]
#
#        try:
#            result = client.infer(model_name, inputs)
#            output0 = result.as_numpy("output_ids")
#            output1 = result.as_numpy("sequence_length")
#            print("============After fastertransformer============")
#            print(output0)
#            print(output1)
#            if FLAGS.return_log_probs:
#                output2 = result.as_numpy("cum_log_probs")
#                output3 = result.as_numpy("output_log_probs")
#                print(output2)
#                print(output3)
#            print("===========================================\n\n\n")
#        except Exception as e:
#            print(e)
#
#    ######################
#    model_name = "postprocessing"
#    with create_inference_server_client(FLAGS.protocol,
#                                        FLAGS.url,
#                                        concurrency=1,
#                                        verbose=FLAGS.verbose) as client:
#        inputs = [
#            prepare_tensor("TOKENS_BATCH", output0, FLAGS.protocol),
#        ]
#        inputs[0].set_data_from_numpy(output0)
#
#        try:
#            result = client.infer(model_name, inputs)
#            output0 = result.as_numpy("OUTPUT")
#            print("============After postprocessing============")
#            print(output0)
#            print("===========================================\n\n\n")
#        except Exception as e:
#            print(e)

    ######################
    def clean_output(output):
        output = re.sub(r'(<s>)+', '<s>', output)  # 替换重复的<s>
        output = output.replace('<s>', '\n')  # 将<s>替换为换行符
        return output


    def choose_input(num: int) -> Tuple[list, int]:
        if num == 1:
            # text generation < 20 words
            return [["answer the following questions as concisely as possible."],
                    ["Data sources you can use to make a decision:"],
                    ["\n if(angle = 0) { if(angle"],
                    ["GMs typically get 78% female enrollment, but the "],
                    ["Previous Chapter | Index | Next Chapter"],
                    ["Michael, an American Jew, called Jews"],
                    ["Blackhawks\n The 2015 Hilltoppers"],
                    ["Data sources you can use to make a comparison:"]], 100   # shape: [8, 1]

        elif num == 2:
            # text generation > 100 words
            return [["In the heart of the bustling city, amidst the towering skyscrapers and busy streets, life unfolds in a vibrant tapestry of diverse cultures, endless possibilities, and captivating stories. The city pulses with energy, its rhythm fueled by the dreams and aspirations of its inhabitants. From the aroma of freshly brewed coffee wafting through the air to the laughter that echoes in lively cafes, every corner tells a tale. Streets lined with historic architecture whisper secrets of the past, while modern marvels "],

                    ["A Los Angeles restaurant has come under fire after a prominent podcaster took issue with an unfamiliar surcharge on his guest check: an extra 4% fee automatically added to the bill to help fund the workers' health insurance. While Alimento, the restaurant in the the Silver Lake neighborhood in Los Angeles, California, was singled out for the move, eateries across the U.S. are increasingly upcharging diners beyond the stated food prices on menus. One in six restaurants said they are adding fees or surcharges to checks to combat higher costs, according "],

                    ["A 25-year-old man has been taken into custody in connection with a shooting over the weekend in downtown Cleveland that left nine people wounded, authorities confirmed. The suspect was identified as Jaylon Jennings, a city of Cleveland spokesperson confirmed to CBS News on Tuesday. Jennings was detained by U.S. Marshals in the Ohio city of Lorain thanks to tips provided to investigators, police later tweeted. Lorain is located about 30 miles west of Cleveland. Just before 2:30 a.m. on Sunday in Cleveland's Warehouse District, a gunman opened "],

                    ["The Justice Department on Tuesday reversed its position that former President Donald Trump was shielded from a 2019 defamation lawsuit filed by the writer E. Jean Carroll. The government had originally argued that Trump was protected from liability by the Westfall Act, because he was acting as a federal employee. Under the act, federal employees are entitled to absolute immunity from personal lawsuits for conduct occurring within the scope of their employment.Principal Deputy Assistant Attorney General Brian Boynton wrote in a letter Tuesday to attorneys for Trump and Carroll that "],

                    ["Flash floods develop when heavy rains hit in a short time. If there's more rain than the ground or sewage can absorb, that extra water flows downhill — a flash flood. Flash flooding can happen anywhere in the country and is most common in low-lying areas with poor drainage. These floods can develop within minutes and can even occur miles away from where a storm hits. Most infrastructure systems across the country are not designed to handle the level of precipitation that has hit the Northeast, Janey Camp, a research progressor and director of Vanderbilt University's Engineering Center for "],

                    ["Professional surfer Mikala Jones died Sunday after a surfing accident in Indonesia, his father told The Associated Press. Friends, family and members of the surfing community took to social media to mourn the loss of Jones, who was known for shooting stunning photos and videos from inside barreling waves. Daughter Isabella Bella Jones posted a touching tribute to her father on Instagram, saying he was doing what he loved the most before he died. The post accompanied a carousel of "],

                    ["Americans have transmitted COVID-19 to wild deer hundreds of times, an analysis of thousands of samples collected from the animals suggests, and people have also caught and spread mutated variants from deer at least three times. The analysis published Monday stems from the first year of a multiyear federal effort to study the virus as it has spread into American wildlife, spearheaded by the U.S. Department of Agriculture's Animal and Plant Health Inspection Service, or APHIS. The agency has been collecting samples from wild deer, elk and moose in 32 states since "],

                    ["An unaccompanied migrant girl from Guatemala with a pre-existing medical condition died in U.S. custody earlier this week after crossing the southern border in May, according to information provided to Congress and obtained by CBS News. The 15-year-old migrant was hospitalized throughout her time in the custody of the Department of Health and Human Services (HHS), which cares for unaccompanied children who lack a legal immigration status. At the time Customs and Border Protection (CBP) transferred the child to HHS custody in May, she was already hospitalized in a "]], 200   # shape: [8, 1]

        elif num == 3:
            # question answering < 20 words
            return [["What are the potential implications of quantum entanglement for secure communication and computing?"],
                    ["How do neural networks with recurrent connections enable the modeling of sequential data in natural language processing?"],
                    ["What are the underlying principles and algorithms used in computer vision systems for object detection and recognition?"],
                    ["How does the interaction between genes and the environment contribute to the development of complex traits and diseases?"],
                    ["What are the challenges and potential applications of blockchain technology beyond cryptocurrencies?"],
                    ["How do astronomers detect and study exoplanets outside our solar system, and what can their discoveries tell us about the possibility of extraterrestrial life?"],
                    ["What are the ethical considerations surrounding the use of artificial intelligence in autonomous vehicles, particularly in terms of decision-making and safety?"],
                    ["How do scientists measure and quantify the impact of climate change on ecosystems and biodiversity, and what are the potential consequences for the planet?"]], 100   # shape: [8, 1]

        elif num == 4:
            # question answering > 100 words
            return [["In the era of rapid technological advancements, particularly in the field of artificial intelligence and automation, profound questions arise regarding the ethical implications and potential societal impacts. As intelligent machines become increasingly integrated into our lives, how do we address the ethical considerations surrounding their decision-making capabilities and potential biases? How can we ensure that these technologies are developed and deployed responsibly, with transparency and accountability? "],

                    ["During developing the triton model, I met some problems. When I use decoupled mode and launch the tritonserver, it got stuck and hang. While I remove the model-transation-policy in the config.pbtxt, it works well. I don't know why. I think it may be the problem of the model-transation-policy. I want to know more about model-transation-policy and wander how to change it to make sure my model can work well."],

                    ["I made create-react-app using npx create-react-app --template typescript command. After installation, I typed command npm run build in order to generate bundle. It creates many files inside of build folder. However I want to generate one javascript bundle file and one css bundle file, named as project-bundle.js and project-bundle.css inside of public/ui directory. How can I generate these two files? Should I build react.js with webpack.config.js file without create-react-app template?"], 

                    ["C++11 introduced a standardized memory model, but what exactly does that mean? And how is it going to affect C++ programming? C++ programmers used to develop multi-threaded applications even before, so how does it matter if it's POSIX threads, or Windows threads, or C++11 threads? What are the benefits? I want to understand the low-level details. Additionally, I am curious about the specific features of the C++11 memory model and how it handles atomic operations, data races, and memory synchronization. How does it ensure the correctness and consistency of shared data in multi-threaded environments?"],

                    ["The hf model performs left padding for inputs of different lengths, tokenizer.pad_token_id = 2. With such a configuration, the result of iterative inference for each sample is consistent with the result of one-time batch inference.But when I reviewed the FT code, I found that it only supports right padding (in invokeRemovePadding), and the result of batch inference with padding is not consistent with hf. How does FT deal with the problem of input with padding batch inference?"],

                    ["This is a Python script sentence: def choose_input(num: int) -> Tuple(list, int): It's not the full code, but enough to check out the fault. Please check the Python sentence and refer to the Python syntax, where do you think the problem may be? Tell me about your inference result. Additionally, could you provide more context or the surrounding code related to the choose_input function? It would be helpful in identifying potential issues or errors within the provided code snippet."],

                    ["When developing a large language model, the model may generate words that are highly repeated. Can you figure out some way to reduce this phenomenon? Or can you tell me which parameters to change to fix this problem? Specifically, I'm curious if changing the repetition_penalty parameter would be effective in mitigating word repetition. Moreover, are there any other techniques or strategies that can be employed to address this issue? It would be helpful to explore potential approaches to encourage more diverse and varied output from the language model while maintaining coherence and fluency. Your insights on this matter would be appreciated."],

                    ["In the realm of climate change, an urgent global challenge, how can we effectively mitigate its impacts and transition towards a sustainable future? What are the most promising strategies for reducing greenhouse gas emissions and achieving carbon neutrality? How can we balance economic development with environmental stewardship? Moreover, how do we address the social and economic disparities that exacerbate vulnerability to climate change, ensuring that climate action is equitable and inclusive?"]], 200

    # TODO: unknown input_length, unkonwn batch_size
    def get_final_input(num: int) -> list:
        input0, OUTPUT_LEN = choose_input(num)
        input0_data = np.array(input0).astype(object)
        bad_words_list = np.array([["" for _ in range(input0_data.shape[1])] for _ in range(input0_data.shape[0])], dtype=object)
        stop_words_list = np.array([["" for _ in range(input0_data.shape[1])] for _ in range(input0_data.shape[0])], dtype=object)

        # convert the datatype
        output0_len = OUTPUT_LEN * np.ones_like(input0).astype(np.uint32)   # [[100], [100], [100], [100], [100], [100], [100], [100]]
        runtime_top_k = (FLAGS.topk * np.ones([input0_data.shape[0], 1])).astype(np.uint32)
        runtime_top_p = FLAGS.topp * np.ones([input0_data.shape[0], 1]).astype(np.float32)
        beam_search_diversity_rate = 0.0 * np.ones([input0_data.shape[0], 1]).astype(np.float32)    # [[0.0], [0.0], ...]
        temperature = 1.0 * np.ones([input0_data.shape[0], 1]).astype(np.float32)   # [[1.0], [1.0], ...]
        len_penalty = 1.0 * np.ones([input0_data.shape[0], 1]).astype(np.float32)
        repetition_penalty = 1.2 * np.ones([input0_data.shape[0], 1]).astype(np.float32)
        random_seed = 0 * np.ones([input0_data.shape[0], 1]).astype(np.uint64)
        is_return_log_probs = True * np.ones([input0_data.shape[0], 1]).astype(bool)
        beam_width = (FLAGS.beam_width * np.ones([input0_data.shape[0], 1])).astype(np.uint32)
        start_ids = start_id * np.ones([input0_data.shape[0], 1]).astype(np.uint32)
        end_ids = end_id * np.ones([input0_data.shape[0], 1]).astype(np.uint32)
        
        inputs = [
            prepare_tensor("INPUT_0", input0_data, FLAGS.protocol),
            prepare_tensor("INPUT_1", output0_len, FLAGS.protocol),
            prepare_tensor("INPUT_2", bad_words_list, FLAGS.protocol),
            prepare_tensor("INPUT_3", stop_words_list, FLAGS.protocol),
            prepare_tensor("runtime_top_k", runtime_top_k, FLAGS.protocol),
            prepare_tensor("runtime_top_p", runtime_top_p, FLAGS.protocol),
            prepare_tensor("beam_search_diversity_rate", beam_search_diversity_rate, FLAGS.protocol),
            prepare_tensor("temperature", temperature, FLAGS.protocol),
            prepare_tensor("len_penalty", len_penalty, FLAGS.protocol),
            prepare_tensor("repetition_penalty", repetition_penalty, FLAGS.protocol),
            prepare_tensor("random_seed", random_seed, FLAGS.protocol), 
            prepare_tensor("is_return_log_probs", is_return_log_probs, FLAGS.protocol),
            prepare_tensor("beam_width", beam_width, FLAGS.protocol),
            prepare_tensor("start_id", start_ids, FLAGS.protocol),
            prepare_tensor("end_id", end_ids, FLAGS.protocol),
        ]
        return inputs

    model_name = "ensemble"
    with create_inference_server_client(FLAGS.protocol,
                                        FLAGS.url,
                                        concurrency=1,
                                        verbose=FLAGS.verbose) as client:

        async def async_infer(_model_name, _inputs):
            result = client.infer(_model_name, _inputs)
            return result

        def warm_up(_inputs):
            # warm up
            start_time = time.time()
            result = client.infer(model_name, _inputs)
            whole_time = time.time() - start_time
            return whole_time, result

        def latency_test(_request_parallelism, _model_name, _inputs):
            req_mean = []
            per_req_mean = []
            for i in range(3):
                # each epoch
                _time = []
                tasks = []
                for i in range(_request_parallelism):
                    tasks.append(asyncio.ensure_future(async_infer(_model_name, _inputs)))
                loop = asyncio.get_event_loop()

                start_time_ = time.time()

                loop.run_until_complete(asyncio.wait(tasks))

                _time.append(time.time() - start_time_)
                total_time_  = (sum(_time) / len(_time)) # request parallelism total time
                req_mean.append(total_time_)
                total_time_per_req = total_time_ / _request_parallelism # each request time
                per_req_mean.append(total_time_per_req)
                time.sleep(5)
            mean_1 = sum(req_mean) / len(req_mean)
            mean_2 = sum(per_req_mean) / len(per_req_mean)
            return mean_1, mean_2

        try:
            """
              type:
                1: text generation < 20 words
                2: text generation > 100 words
                3: question answering < 20 words
                4: question answering > 100 words
            """
            choosed_type = 1

            inputs = get_final_input(choosed_type)

            # warm up
            timme_0, result = warm_up(inputs)
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

            # latency test on different request_parallelism
            # WARN: max_batch_size = 1024, each request is 8, the max request_parallelism is 128
            req_para_range = np.arange(0, 50, 5).tolist()
            req_para_range[0] = 1

            _start_time = time.time()

            latency_per_req_collection = []
            latency_epoch_collection = []
            for request_parallelism in req_para_range:
                lat_epoch, lat_per_req = latency_test(request_parallelism, model_name, inputs)
                print(f"Standard Inference time: request_parallelism: {request_parallelism}, {lat_epoch} s/epoch, {lat_per_req} s/request")
                latency_per_req_collection.append(lat_per_req)
                latency_epoch_collection.append(lat_epoch)

            whole_time__ = time.time() - _start_time
            print(f"Total time: {whole_time__} s")

            # visualize
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.plot(req_para_range, latency_per_req_collection)
            plt.xlabel('request_parallelism')
            plt.ylabel('latency_per_req')
            plt.subplot(1, 2, 2)    
            plt.plot(req_para_range, latency_epoch_collection)
            plt.xlabel('request_parallelism')
            plt.ylabel('latency_epoch')

            plt.savefig(f'./plot_result/latency_type={choosed_type}.png')


            if FLAGS.return_log_probs:
                print(result.as_numpy("cum_log_probs"))
                print(result.as_numpy("output_log_probs"))
                
        except Exception as e:
            print(e)
