import os
import random
import re
from bs4 import BeautifulSoup
from warcio.archiveiterator import ArchiveIterator
from tqdm import tqdm
import numpy as np


class BuildDataset:
    def __init__(self):
        self.coca_directory = './dataset/sources.txt'
        self.cc_directory = './dataset/warc.paths'
        print(f"[INFO] Got COCA dataset from directory {self.coca_directory}")
        print(f"[INFO] Got Common Crawler dataset from directory {self.cc_directory}")
        self.rand_seed_num = 42

    def get_coca_source(self) -> list:
        with open(self.coca_directory, 'r', encoding='latin-1') as f:
            lines = f.readlines()[4:]

        coca_text_raw = [line.split('\t')[5].strip() for line in lines if line.strip()]
        return coca_text_raw

    def download_cc_dataset(self, dataset_limit: int) -> list:
        with open(self.cc_directory, 'r', encoding='latin-1') as f:
            lines = f.readlines()
        base_url = "https://data.commoncrawl.org/"
        url_list = [base_url + line for line in lines[:dataset_limit]]
        
        print("[WARN] If you want to download the dataset, we suggest you to do it manually: ")
        print("[WARN] 1. cd to your/dataset/directory")
        for url in url_list:
            print("[WARN] 2. wget " + url)

    def pre_build_cc_dataset(self, save_directory) -> list:
        # this function is used to extract the text from the html file

        # read file list in './dataset/cc_dataset'
        dataset_path = "./dataset/cc_dataset"
        for item in os.listdir(dataset_path):
            text_list = []
            with open(dataset_path+'/'+item, 'rb') as stream:
                for record in tqdm(ArchiveIterator(stream), desc='Processing records', unit='record'):
                    if record.rec_type == 'response':
                        # 提取 HTTP 响应的内容
                        http_response = record.content_stream().read()
                        
                        # 使用 BeautifulSoup 提取 HTML 内容
                        try:
                            soup = BeautifulSoup(http_response, 'html.parser')
                        except:
                            continue
                        paragraphs = [p.get_text() for p in soup.find_all('p')]
                        # 打印出 HTML 内容
                        long_paragraphs = [p for p in paragraphs if len(p.split()) > 100]   # sometimes empty
                        
                        # if len(long_paragraphs) > 0 and content is english
                        for text_raw in long_paragraphs:
                            if re.match(r'[a-zA-Z0-9]+', text_raw):
                                text_list.append(text_raw)
            with open(save_directory+'/'+item + '.txt', 'w') as f:
                f.write('\n'.join(text_list))
                print(f"[INFO] Saved {len(text_list)} paragraphs in {save_directory+'/'+item + '.txt'}")
                
    def post_build_cc_dataset(self):
        # this function is used to clean the text
        
        def clean_text(text):
            # english only
            cleaned_text = re.sub(r'[^A-Za-z0-9\s\.\,\;\:\-\'\"]', '', text)
            return cleaned_text

        file_directory = './dataset/cc_dataset/txt'
        save_directory = './dataset/cc_dataset/output'
        for item in os.listdir(file_directory):
            # clean all raw text
            print(item)
            with open(file_directory + '/' + item, "r", encoding="utf-8") as f_in, open(save_directory + '/' + "output.txt", "w", encoding="utf-8") as f_out:
                for line in f_in:
                    cleaned_line = clean_text(line)
                    tokens = tokenizer.tokenize(cleaned_line)
                    if len(tokens) > 100 and len(tokens) < 400:
                        f_out.write(cleaned_line)

    def prepare_text_pool(self, text_directory='./dataset/cc_dataset/output') -> list:
        # this function is to load random text from the dataset

        text_file_list = os.listdir(text_directory)

        # choose a file randomly
        length = len(text_file_list)
        random_file_index = np.random.randint(0, length)

        # fetch the text randomly
        with open(text_directory + '/' + text_file_list[random_file_index], 'r', encoding='utf-8') as f:
            # read line index: random_num
            data_index = np.arange(0, 10000, 1).tolist()

            random.seed(self.rand_seed_num)
            sample_size = 400
            random_list = random.sample(data_index, sample_size)    # get 400 random text index
            lines = f.readlines()
            
            # pool: [['text_A'], ['text_b'], ... ]
            rand_text_pool = [[lines[i]] for i in random_list]
            self.random_text_pool = rand_text_pool    # get 400 random lines
            
        return rand_text_pool