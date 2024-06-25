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

def generate_meaningful_prompt(token_length):
    # Create a meaningful sentence with the exact number of tokens
    sentence = " ".join(brown_words[:token_length])
    return sentence

def poisson_arrival_times(rate: float, num_requests: int, start_time: float) -> List[float]:
    intervals = np.random.exponential(scale=1/rate, size=num_requests)
    arrival_times = np.cumsum(intervals) + start_time
    return arrival_times.tolist()

class CustomLLM(LLM):
    def generate_with_custom_arrival(
        self,
        prompts: List[str],
        arrivals: List[float],
        sampling_params: Optional[Union[SamplingParams, List[SamplingParams]]] = None,
        use_tqdm: bool = True
    ) -> List[RequestOutput]:
        if sampling_params is None:
            sampling_params = SamplingParams()
        elif isinstance(sampling_params, list) and len(sampling_params) != len(prompts):
            raise ValueError("The lengths of prompts and sampling_params must be the same.")
        
        if len(arrivals) != len(prompts):
            raise ValueError("The lengths of arrivals and prompts must be the same.")
        
        for i, prompt in enumerate(prompts):
            current_sampling_params = sampling_params[i] if isinstance(sampling_params, list) else sampling_params
            self._add_request_with_custom_arrival(
                prompt=prompt,
                sampling_params=current_sampling_params,
                prompt_token_ids=None,
                arrival_time=arrivals[i]
            )
        
        outputs = self._run_engine(use_tqdm)
        return outputs

    def _add_request_with_custom_arrival(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]],
        arrival_time: float
    ) -> None:
        request_id = str(next(self.request_counter))
        self.llm_engine.add_request(
            request_id,
            prompt,
            sampling_params,
            prompt_token_ids,
            arrival_time=arrival_time,
            lora_request=None,
            multi_modal_data=None
        )

num_requests = 6
prompts = [
    "hi, how are you",
    "hello",
    "can you tell me something",
    "hi, how are you",
    "hello",
    "can you tell me something"
]

sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10, min_tokens=1)
max_iterations = 1000
goodput = 0
current_time = time.time()
arrival_times = poisson_arrival_times(rate=1.0, num_requests=num_requests, start_time=current_time)
llm = CustomLLM(model=model_name, trust_remote_code=True)
outputs = llm.generate_with_custom_arrival(prompts=prompts, sampling_params=sampling_params, arrivals=arrival_times)
for output in outputs:
    print("prompt:", output.prompt)
    print("arrival time:", output.metrics.arrival_time)
    print("finish time:", output.metrics.finished_time)
    print("deadline:", output.metrics.deadline)
    print("generated text:", output.outputs[0].text)
