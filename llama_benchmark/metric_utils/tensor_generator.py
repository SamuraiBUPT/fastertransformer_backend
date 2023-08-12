import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

start_id = 0
end_id = 1

def prepare_tensor(name, input, protocol):
    client_util = httpclient if protocol == "http" else grpcclient
    t = client_util.InferInput(
        name, input.shape, np_to_triton_dtype(input.dtype)) # set the input attributes
    t.set_data_from_numpy(input)    # return input data
    return t

class TensorGenerator:
    def __init__(self, args, batch_size_req:int):
        self.args = args
        self.batch_size_req = batch_size_req
    
    

    def get_input_tensor(self, text_2d, print_input:bool=False) -> list:
        input0 = text_2d # 2-dim [['I am text']]
        OUTPUT_LEN = 500

        input0_data = np.array(input0).astype(object)
        bad_words_list = np.array([["" for _ in range(input0_data.shape[1])] for _ in range(input0_data.shape[0])], dtype=object)
        stop_words_list = np.array([["" for _ in range(input0_data.shape[1])] for _ in range(input0_data.shape[0])], dtype=object)

        # convert the datatype
        output0_len = OUTPUT_LEN * np.ones_like(input0).astype(np.uint32)   # [[100], [100], [100], [100], [100], [100], [100], [100]]

        runtime_top_k = (self.args.topk * np.ones([input0_data.shape[0], 1])).astype(np.uint32)
        runtime_top_p = self.args.topp * np.ones([input0_data.shape[0], 1]).astype(np.float32)
        beam_search_diversity_rate = 0.0 * np.ones([input0_data.shape[0], 1]).astype(np.float32)    # [[0.0], [0.0], ...]
        temperature = 1.0 * np.ones([input0_data.shape[0], 1]).astype(np.float32)   # [[1.0], [1.0], ...]
        len_penalty = 1.0 * np.ones([input0_data.shape[0], 1]).astype(np.float32)
        repetition_penalty = 1.2 * np.ones([input0_data.shape[0], 1]).astype(np.float32)
        random_seed = 0 * np.ones([input0_data.shape[0], 1]).astype(np.uint64)
        is_return_log_probs = True * np.ones([input0_data.shape[0], 1]).astype(bool)
        beam_width = (self.args.beam_width * np.ones([input0_data.shape[0], 1])).astype(np.uint32)
        start_ids = start_id * np.ones([input0_data.shape[0], 1]).astype(np.uint32)
        end_ids = end_id * np.ones([input0_data.shape[0], 1]).astype(np.uint32)

        inputs = [
            prepare_tensor("INPUT_0", input0_data, self.args.protocol),
            prepare_tensor("INPUT_1", output0_len, self.args.protocol),
            prepare_tensor("INPUT_2", bad_words_list, self.args.protocol),
            prepare_tensor("INPUT_3", stop_words_list, self.args.protocol),
            prepare_tensor("runtime_top_k", runtime_top_k, self.args.protocol),
            prepare_tensor("runtime_top_p", runtime_top_p, self.args.protocol),
            prepare_tensor("beam_search_diversity_rate", beam_search_diversity_rate, self.args.protocol),
            prepare_tensor("temperature", temperature, self.args.protocol),
            prepare_tensor("len_penalty", len_penalty, self.args.protocol),
            prepare_tensor("repetition_penalty", repetition_penalty, self.args.protocol),
            prepare_tensor("random_seed", random_seed, self.args.protocol), 
            prepare_tensor("is_return_log_probs", is_return_log_probs, self.args.protocol),
            prepare_tensor("beam_width", beam_width, self.args.protocol),
            prepare_tensor("start_id", start_ids, self.args.protocol),
            prepare_tensor("end_id", end_ids, self.args.protocol),
        ]

        if print_input:
            print("============Input Material============")
            print(input0)
            return inputs
        else:
            return inputs