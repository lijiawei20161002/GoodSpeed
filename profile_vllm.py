import nltk
from nltk.corpus import brown
import time
import sys
import numpy as np
from typing import List
from vllm import LLM, SamplingParams, RequestOutput
import multiprocessing
import os

sys.path.append('/data/jiawei_li/GoodSpeed/vllm_source')
multiprocessing.set_start_method('spawn', force=True)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Ensure NLTK Brown corpus is downloaded
nltk.download('brown')

# Function to profile the inference time of a given model
def profile_model_inference_time(model_name: str, num_requests: int = 1000) -> None:
    # Generate prompts from the Brown corpus
    brown_words = brown.words()
    brown_text = ' '.join(brown_words)
    prompts = [brown_text.split('.')[i] for i in range(num_requests)]

    # Sampling parameters for the model
    sampling_params = SamplingParams(temperature=0, top_p=0.95)

    # Initialize the LLM model
    llm = LLM(model=model_name, trust_remote_code=True, enable_chunked_prefill=False, max_num_seqs=16, tensor_parallel_size=4)

    # Generate outputs from the model
    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)

    # Metrics to collect
    total_inference_time = 0
    total_tokens = 0

    # Calculate total inference time and total tokens
    for output in outputs:
        # Calculate the time taken for the inference
        inference_time = float(output.metrics.finished_time) - float(output.metrics.first_token_time)
        total_inference_time += inference_time
        
        # Get the number of tokens generated in the output
        num_tokens = len(output.outputs[0].text.split()) -1  
        total_tokens += num_tokens

    # Calculate averages
    average_inference_time = total_inference_time / num_requests
    average_tokens = total_tokens / num_requests
    average_time_per_token = average_inference_time / average_tokens if average_tokens > 0 else float('inf')

    # Print the results
    print(f"Model: {model_name}")
    print(f"Total Inference Time: {total_inference_time:.4f} seconds")
    print(f"Average Inference Time: {average_inference_time:.4f} seconds")
    print(f"Total Tokens Generated: {total_tokens}")
    print(f"Average Tokens per Request: {average_tokens:.2f} tokens")
    print(f"Average Time per Token: {average_time_per_token:.4f} seconds/token\n")

if __name__ == "__main__":
    model_names = [
        #'/data/public_models/huggingface/meta-llama/Llama-2-13b-chat-hf',
        #'/data/public_models/huggingface/deepseek-ai/deepseek-llm-7b-base',
        #'/data/public_models/huggingface/meta-llama/Meta-Llama-3-8B',
        #'/data/public_models/huggingface/Qwen/Qwen1.5-14B',
        '/data/public_models/huggingface/meta-llama/Meta-Llama-3-70B-Instruct',
        #'/data/public_models/Llama-3-70B',
        #'/data/public_models/huggingface/deepseek-ai/deepseek-llm-67b-chat',
    ]

    for model in model_names:
        profile_model_inference_time(model)