from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import os
import datetime
import pandas as pd
import nltk
from nltk.corpus import brown
import time
import sys
import numpy as np
from typing import List, Union, Optional

model_name = '/data/public_models/huggingface/meta-llama/Llama-2-13b-chat-hf'
sys.path.append('/data/jiawei_li/GoodSpeed/vllm_source')
from vllm import LLM, SamplingParams, RequestOutput

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

num_requests = 100
brown_text = ' '.join(brown_words)
prompts = [brown_text.split('.')[i] for i in range(num_requests)]
sampling_params = SamplingParams(temperature=0, top_p=0.95)
goodput = 0
current_time = time.time()
arrival_times = poisson_arrival_times_with_bursts(rate, num_requests, current_time, burst_rate, burst_duration, burst_interval)
llm = LLM(model=model_name, trust_remote_code=True, max_num_seqs=16)
outputs = llm.generate(prompts=prompts, sampling_params=sampling_params, arrivals=arrival_times)

for output in outputs:
    if output.metrics.finished_time <= output.metrics.deadline:
        goodput += 1
    else:
        print("prompt:", output.prompt)
        print("arrival time:", output.metrics.arrival_time)
        print("finish time:", output.metrics.finished_time)
        print("deadline:", output.metrics.deadline)
        print("generated text:", output.outputs[0].text)
        print('======================')
print("goodput:", goodput)

# Identify prompts with no outputs
#prompt_ids_with_outputs = {output.request_id for output in outputs}
#print(prompt_ids_with_outputs)
#missing_prompt_ids = [i for i in range(num_requests) if str(i) not in prompt_ids_with_outputs]
#print(f"Unfinished Requests: {missing_prompt_ids}")
