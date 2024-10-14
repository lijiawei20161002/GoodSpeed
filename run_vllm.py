import csv
import nltk
from nltk.corpus import brown
import time
import sys
import numpy as np
from typing import List
from huggingface_hub import snapshot_download

#'/data/public_models/huggingface/meta-llama/Llama-2-13b-chat-hf'
#model_name ='/data/huggingface/models--deepseek-ai--deepseek-llm-7b-base/snapshots/7683fea62db869066ddaff6a41d032262c490d4f' 
#'/data/public_models/huggingface/meta-llama/Meta-Llama-3-8B'  
#model_name = '/data/huggingface/models--Qwen--Qwen1.5-14B/snapshots/dce4b190d34470818e5bec2a92cb8233aaa02ca2'
model_name = '/data/huggingface/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/1480bb72e06591eb87b0ebe2c8853127f9697bae'
#'/data/public_models/huggingface/Qwen/Qwen1.5-14B'
#'/data/public_models/Llama-3-70B'
#model_name = '/data/huggingface/models--deepseek-ai--deepseek-llm-67b-chat/snapshots/79648bef7658bb824e4630740f6e1484c1b0620b'
#model_name = '/data/huggingface/models--meta-llama--Llama-2-13b-chat-hf/snapshots/a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8'
sys.path = ['/data/jiawei_li/GoodSpeed/vllm_source'] + sys.path
from vllm import LLM, SamplingParams, RequestOutput
import os
import ray
os.environ["RAY_TEMP_DIR"] = "/data/jiawei_li/ray_temp"
runtime_env = {"env_vars": {"PYTHONPATH": "/data/jiawei_li/GoodSpeed/vllm_source"}}
ray.init(address="auto", runtime_env=runtime_env)

nltk.download('brown')
brown_words = brown.words()
np.random.seed(42)

def read_data_from_csv(file_path: str):
    """
    Reads the CSV file and extracts arrival times, input token lengths, 
    and output token lengths.
    """
    arrivals = []
    input_lens = []
    output_lens = []
    workload_types = []

    with open(file_path, mode='r', newline='') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            arrivals.append(float(row['Timestamp'])*0.0001)
            input_lens.append(int(row['Request tokens']))
            output_lens.append(int(row['Response tokens']))
            workload_types.append(row['Log Type'])

    return arrivals, input_lens, output_lens, workload_types

num_requests = 1000
csv_file_path = "data/gpt4.csv"
arrival_times, input_lens, output_lens, workload_types = read_data_from_csv(csv_file_path)
arrival_times, input_lens, output_lens, workload_types = arrival_times[:num_requests], input_lens[:num_requests], output_lens[:num_requests], workload_types[:num_requests]
brown_text = ' '.join(brown_words)
prompts = []
current_index = 0
for length in input_lens:
    prompt = ' '.join(brown_words[current_index:current_index + length])
    prompts.append(prompt)
    current_index += length
sampling_params = SamplingParams(temperature=0, top_p=0.95)
goodput = 0
current_time = time.time()
arrival_times = [current_time + arrival for arrival in arrival_times]
llm = LLM(model=model_name, trust_remote_code=True, enable_chunked_prefill=False, max_num_seqs=16, tensor_parallel_size=4)
outputs = llm.generate(prompts=prompts, sampling_params=sampling_params, arrivals=arrival_times, output_lens=output_lens, workload_types=workload_types)

metrics_data = []
for output in outputs:
    print(output.metrics.workload_type, output.metrics.arrival_time, output.metrics.finished_time, output.metrics.deadline)
    if output.metrics.finished_time <= output.metrics.deadline:
        goodput += 1
    metrics_data.append({
        'prompt': output.prompt,
        'arrival_time': output.metrics.arrival_time,
        'finish_time': output.metrics.finished_time,
        'deadline': output.metrics.deadline,
        'generated_text': output.outputs[0].text,
        'price': output.metrics.price
    })
print("goodput:", goodput)