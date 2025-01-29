import csv
import nltk
from nltk.corpus import brown
import time
import sys
import numpy as np
from typing import List

#model_name = '/data/public_models/Llama-2-7b-chat-hf'
#'/data/public_models/huggingface/meta-llama/Llama-2-13b-chat-hf'
#model_name ='/data/public_models/huggingface/deepseek-ai/deepseek-llm-7b-base' 
#'/data/public_models/huggingface/meta-llama/Meta-Llama-3-8B'  
#'/data/public_models/huggingface/Qwen/Qwen1.5-14B'
#model_name = '/data/public_models/huggingface/meta-llama/Meta-Llama-3-70B-Instruct'
#'/data/public_models/huggingface/Qwen/Qwen1.5-14B'
#'/data/public_models/Llama-3-70B'
#model_name = '/data/public_models/models--deepseek-ai--deepseek-llm-67b-chat/snapshots/79648bef7658bb824e4630740f6e1484c1b0620b'
#model_name = '/data/public_models/huggingface/deepseek-ai/deepseek-llm-67b-chat'
#model_name = '/data/public_models/huggingface/meta-llama/Llama-2-13b-chat-hf'
model_name = '/data/public_models/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a'
sys.path = ['/data/jiawei_li/GoodSpeed/vllm_source'] + sys.path
from vllm import LLM, SamplingParams, RequestOutput
import os
#import ray
#os.environ["RAY_TEMP_DIR"] = "/data/jiawei_li/ray_temp"
runtime_env = {"env_vars": {"PYTHONPATH": "/data/jiawei_li/GoodSpeed/vllm_source"}}
#ray.init(address="auto", runtime_env=runtime_env)

nltk.download('brown')
brown_words = brown.words()
np.random.seed(42)

def poisson_arrival_times_with_bursts(
    rate: float, 
    num_requests: int, 
    start_time: float, 
    burst_rate: float, 
    burst_duration: float, 
    burst_interval: float
) -> List[float]:
    arrival_times = []
    current_time = start_time
    
    while len(arrival_times) < num_requests:
        # Determine if we are in a burst period
        time_since_start = current_time - start_time
        in_burst = (int(time_since_start *1000) % int(burst_interval*1000))/1000 < burst_duration

        # Choose the rate based on whether we are in a burst period
        current_rate = burst_rate if in_burst else rate
        
        # Generate the next interval
        interval = np.random.exponential(scale=1/current_rate)
        current_time += interval
        
        # Append the current time to the arrival times
        arrival_times.append(current_time)
        
        # Ensure we don't exceed the number of requested arrivals
        if len(arrival_times) >= num_requests:
            break
    
    return arrival_times

# Example usage
rate = 1  # Average rate of 1 requests per second
burst_rate = 100.0  # Average rate of 100 requests per second during bursts
burst_duration = 0.1  # Each burst lasts for 0.1 seconds
burst_interval = 0.4  # Bursts occur every 0.4 seconds

num_requests = 1000
brown_text = ' '.join(brown_words)
prompts = [brown_text.split('.')[i] for i in range(num_requests)]
sampling_params = SamplingParams(temperature=0, top_p=0.95)
goodput = 0
current_time = time.time()
arrival_times = poisson_arrival_times_with_bursts(rate, num_requests, current_time, burst_rate, burst_duration, burst_interval)
llm = LLM(model=model_name, trust_remote_code=True, enable_chunked_prefill=False, max_num_seqs=16) # tensor_parallel_size=4)
outputs = llm.generate(prompts=prompts, sampling_params=sampling_params, arrivals=arrival_times)

metrics_data = []
for output in outputs:
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
'''
csv_file = 'fcfs_output_metrics.csv'
csv_columns = ['prompt', 'arrival_time', 'finish_time', 'deadline', 'generated_text', 'price']
try:
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in metrics_data:
            writer.writerow(data)
except IOError:
    print("I/O error")'''