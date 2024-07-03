from modelscope.hub.api import HubApi
from huggingface_hub import HfApi, HfFolder, login
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

access_token = os.getenv('ACCESS_TOKEN')
if access_token is None:
    raise ValueError("ACCESS_TOKEN environment variable not set.")

api = HubApi()
api.login(access_token)

hf_token = os.getenv('hf_token')
if hf_token is None:
    raise ValueError("hf_token environment variable not set.")
login(token=hf_token)
print("Successfully logged in to Hugging Face")

model_name = 'PartAI/Dorna-Llama3-8B-Instruct'
sys.path.append('/data/jiawei_li/GoodSpeed/vllm_source')
from vllm import LLM, SamplingParams, RequestOutput

nltk.download('brown')
brown_words = brown.words()

def poisson_arrival_times(rate: float, num_requests: int, start_time: float) -> List[float]:
    intervals = np.random.exponential(scale=1/rate, size=num_requests)
    arrival_times = np.cumsum(intervals) + start_time
    return arrival_times.tolist()

num_requests = 1000
brown_text = ' '.join(brown_words)
prompts = [brown_text.split('.')[i] for i in range(num_requests)]
sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10, min_tokens=1)
goodput = 0
current_time = time.time()
arrival_times = poisson_arrival_times(rate=1.0, num_requests=num_requests, start_time=current_time)
llm = LLM(model=model_name, trust_remote_code=True)
outputs = llm.generate(prompts=prompts, sampling_params=sampling_params, arrivals=arrival_times)
for output in outputs:
    print("prompt:", output.prompt)
    print("arrival time:", output.metrics.arrival_time)
    print("finish time:", output.metrics.finished_time)
    print("deadline:", output.metrics.deadline)
    print("generated text:", output.outputs[0].text)
    print("generated token:", output.metrics.processed_token)
    if output.metrics.finished_time <= output.metrics.deadline:
        goodput += 1
print("goodput:", goodput)
