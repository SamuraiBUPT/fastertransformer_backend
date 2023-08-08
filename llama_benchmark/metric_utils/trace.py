import random
from .tensor_generator import TensorGenerator
from .build_dataset import BuildDataset

m_dataset = BuildDataset()

class Trace:
    def __init__(self, args, text_directory:str, time_period:int=100):
        self.trace = []
        self.rand_seed = 3
        self.text_pool = m_dataset.prepare_text_pool(text_directory=text_directory)  # prepare text pool, each one is ['text'], the pool is [[''], ['']]
        self.tensor_manager = TensorGenerator(args, batch_size_req=1)
        self.time_period = time_period
        self.request_time_stamp = self.prepare_period_request()
        
    def prepare_period_request(self):
        length = 200
        random.seed(self.rand_seed)
        request_time_stamp = [random.randint(1, 9) for _ in range(length)]    # 495 request in total
        return request_time_stamp
    
    def get_single_request_input(self, text:list):
        # the text_input is a list with only one element, e.g. ['I am text']
        return self.tensor_manager.get_input_tensor([text])
    