from flask import Flask, request
import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
from functools import partial

app = Flask(__name__)
app.secret_key = 'secret!'

"""
global variables
    triton_status: [bool] the status of triton server: True or False
    max_queue_delay: [int] the maximum delay of the request waiting queue
                     if requests > target number, schedule them
                     if requests < target number, push them all as well.
"""

triton_status = None  # global variable
start_id = 0
end_id = 1

def prepare_tensor(name, input, protocol):
    """Prepare a tensor for inference based on the input data."""
    client_util = httpclient if protocol == "http" else grpcclient
    t = client_util.InferInput(
        name, input.shape, np_to_triton_dtype(input.dtype)) # set the input attributes
    t.set_data_from_numpy(input)    # return input data
    return t

def prepare_input_tensor(text_2d, protocal, lora_type, 
                         topk, topp, beam_width,
                         print_input=False):
    input0 = text_2d # 2-dim [['I am text']]
    OUTPUT_LEN = 500

    input0_data = np.array(input0).astype(object)
    bad_words_list = np.array([["" for _ in range(input0_data.shape[1])] for _ in range(input0_data.shape[0])], dtype=object)
    stop_words_list = np.array([["" for _ in range(input0_data.shape[1])] for _ in range(input0_data.shape[0])], dtype=object)

    # convert the datatype
    output0_len = OUTPUT_LEN * np.ones_like(input0).astype(np.uint32)   # [[100], [100], [100], [100], [100], [100], [100], [100]]

    runtime_top_k = (topk * np.ones([input0_data.shape[0], 1])).astype(np.uint32)
    runtime_top_p = topp * np.ones([input0_data.shape[0], 1]).astype(np.float32)
    beam_search_diversity_rate = 0.0 * np.ones([input0_data.shape[0], 1]).astype(np.float32)    # [[0.0], [0.0], ...]
    temperature = 1.0 * np.ones([input0_data.shape[0], 1]).astype(np.float32)   # [[1.0], [1.0], ...]
    len_penalty = 1.0 * np.ones([input0_data.shape[0], 1]).astype(np.float32)
    repetition_penalty = 1.2 * np.ones([input0_data.shape[0], 1]).astype(np.float32)
    random_seed = 0 * np.ones([input0_data.shape[0], 1]).astype(np.uint64)
    is_return_log_probs = True * np.ones([input0_data.shape[0], 1]).astype(bool)
    beam_width_ = (beam_width * np.ones([input0_data.shape[0], 1])).astype(np.uint32)
    start_ids = start_id * np.ones([input0_data.shape[0], 1]).astype(np.uint32)
    end_ids = end_id * np.ones([input0_data.shape[0], 1]).astype(np.uint32)
    
    # prepare lora input type
    lora_type_ = lora_type * np.ones([input0_data.shape[0], 1]).astype(np.uint32)

    inputs = [
        prepare_tensor("INPUT_0", input0_data, protocal),
        prepare_tensor("INPUT_1", output0_len, protocal),
        prepare_tensor("INPUT_2", bad_words_list, protocal),
        prepare_tensor("INPUT_3", stop_words_list, protocal),
        prepare_tensor("runtime_top_k", runtime_top_k, protocal),
        prepare_tensor("runtime_top_p", runtime_top_p, protocal),
        prepare_tensor("beam_search_diversity_rate", beam_search_diversity_rate, protocal),
        prepare_tensor("temperature", temperature, protocal),
        prepare_tensor("len_penalty", len_penalty, protocal),
        prepare_tensor("repetition_penalty", repetition_penalty, protocal),
        prepare_tensor("random_seed", random_seed, protocal), 
        prepare_tensor("is_return_log_probs", is_return_log_probs, protocal),
        prepare_tensor("beam_width", beam_width_, protocal),
        prepare_tensor("start_id", start_ids, protocal),
        prepare_tensor("end_id", end_ids, protocal),
        prepare_tensor("lora_type", lora_type_, protocal),
    ]
    
    client_util = httpclient if protocal == "http" else grpcclient

    if print_input:
        print("============Input Material============")
        print(input0)
        return inputs
    else:
        return inputs
    
    
def create_inference_server_client(protocol, url, concurrency, verbose):
    client_util = httpclient if protocol == "http" else grpcclient
    if protocol == "http":
        return client_util.InferenceServerClient(url,
                                                concurrency=concurrency,
                                                verbose=verbose)
    elif protocol == "grpc":
        return client_util.InferenceServerClient(url,
                                                verbose=verbose)
        

def grpc_request(model_name, input_tensor):
    def callback(user_data, result, error):
            if error:
                user_data.append(error)
            else:
                user_data.append(result)
                
    user_data = []
    with create_inference_server_client('grpc',
                                        'localhost:8001',
                                        concurrency=1,
                                        verbose=False) as client:
        client.async_infer(model_name=model_name, 
                            inputs=input_tensor, 
                            callback=partial(callback, user_data))
        while(True):
            if len(user_data) == 1:
                return user_data
        
def get_output_text(output_raw):
    output = output_raw.as_numpy('OUTPUT_0')
    return output

@app.route('/arrival', methods=['POST'])
async def arrival():
    data = request.json
    # print("===============INPUT=================")
    # print(data['INPUT_0'])
    input_tensor = prepare_input_tensor(text_2d=[[data['INPUT_0']]], 
                                       protocal=data['protocal'], 
                                       lora_type=data['lora_type'],
                                       topk=data['topk'],
                                       topp=data['topp'],
                                       beam_width=data['beam_width'])
    if data['protocal'] == 'http':
        pass
    else:
        print("[INFO] grpc request")
        output0_data = grpc_request(model_name=data['model_name'], 
                                    input_tensor=input_tensor)[0]
        # output0_data = output0_data.as_numpy('OUTPUT0')
        output = get_output_text(output0_data)
        # print("===============OUTPUT=================")
        # print(output)
                                       

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8003, debug=True) # 8000-8002 was used by triton services