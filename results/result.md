### Performance Comparison on Different Models (prefills+decodes, 1000 requests)
| Model                   | GPU             | Request Num | Oracle | FCFS  | Solver | Deadline | Random | Bidding |
|-------------------------|-----------------|-------------|--------|-------|--------|----------|--------|---------|
| deepseek-llm-7b-base    | 1 Nvidia A100   | 1000        | 978    | 936   | 342    | 908      | 941    | 961     |
| Llama-2-13b-chat-hf     | 1 Nvidia A100   | 1000        | 978    | 693   | 304    | 677      | 754    | 840     |
| Qwen1.5-14B             | 1 Nvidia A100   | 1000        | 978    | 693   |        | 640      | 719    | 819     |
| deepseek-llm-67b-chat   | 4 Nvidia A100   | 1000        | 978    | 335   | 290    | 329      | 337    | 467     |
| Llama-3-70B             | 4 Nvidia A100   | 1000        | 978    | 335   | 308    | 335      | 355    | 557     |

### Performance Comparison on Different Models (chunked prefill, 1000 requests)
| Model                     | GPU             | Request Num | Oracle | FCFS | Deadline | Random | Bidding |
|---------------------------|-----------------|-------------|--------|------|----------|--------|---------|
| deepseek-llm-7b-base      | 1 Nvidia A100   | 1000        | 978    | 910  | 894      | 904    | 933     |
| Llama-2-13b-chat-hf       | 1 Nvidia A100   | 1000        | 978    | 753  | 668      | 762    | 856     |
| Qwen1.5-14B               | 1 Nvidia A100   | 1000        | 978    | 703  | 651      | 732    | 853     |
| deepseek-llm-67b-chat     | 4 Nvidia A100   | 1000        | 978    | 335  | 332      | 346    | 563     |
| Meta-Llama-3-70B-Instruct | 4 Nvidia A100   | 1000        | 978    | 335  | 335      | 341    | 488     |

### Performance Comparison of Policies (solver only in decodes, window=10)
| request_num | oracle | fcfs | solver | deadline | random | bidding |
|-------------|--------|------|--------|----------|--------|---------|
| 100         | 77     | 55   | 47     | 40       | 41     | 68      |
| 500         | 474    | 387  | 192    | 375      | 363    | 361     |
| 1000        | 978    | 693  | 335    | 636      | 763    | 680     |

### Performance Comparison of Policies (solver only in prefills + decodes, window=10)
- **Model**: Llama-2-13b-chat-hf
| request_num     | oracle | fcfs  | solver  | deadline | random | bidding |
|-----------------|--------|-------|---------|----------|--------|---------|
| 100             | 77     | 55    | 37      | 39       | 50     | 55      | 
| 500             | 474    | 387   | 155     | 332      | 368    | 396     | 
| 1000            | 978    | 693   | 304     | 677      | 754    | 840     |
| 5000            | 4923   | 4290  | 876     | 4048     | 4192   | 4605    |
| Timeline(h:m:s) | -      | 34:10 | 1:33:17 | 34:17    | 34:17  | 34:00   | 

- **Model**: Meta-Llama-3-70B-Instruct (chunked prefill)
| request_num     | fcfs  | bidding | random | deadline |
|-----------------|-------|---------|--------|----------|
| 100             | 37    | 37      | 37     | 37       | 
| 500             | 174   | 256     | 189    | 174      | 
| 1000            | 335   | 509     | 365    | 335      |
| 5000            | 1625  | 2852    | 1716   | 1984     |
| Timeline(h:m:s) | 36:12 | 35:08   | 36:26  | 35:29    |
 

### Workload Configuration
| Workload Type   | Proportion | Deadline          | Max Token |
|-----------------|------------|-------------------|-----------|
| Search          | 33%        | U(1, 2)           | 10        |
| Chatbox         | 33%        | 0.2 * max_token=10| 50        |
| Batch Analysis  | 33%        | U(300, 3600)      | 500       |

### Summary Statistics of Prompt Lengths
- **Minimum length:** 1
- **Maximum length:** 614
- **Mean length:** 90.5002
- **Standard deviation of lengths:** 73.17146173558722
- **Median:** 80.5
- **90% Percentile:** 188.0
- **95% Percentile:** 223.0

### Request Scheduling Metrics
- **Number of Requests**: 100
- **Penalty**: True
- **Batch Size**: 16
- **Window Size**: 10

### Reserve Capacity and Goodput

| Reserve Capacity | Goodput |
|------------------|---------|
| 0                | 40      |
| 4                | 38      |
| 5                | 39      |
| 6                | 39      |
| 8                | 42      |
| 10               | 43      |
| 12               | 41      |
| 16               | 37      |

### Batch Size Impact
- **Model**: Llama-2-13b-chat-hf
| Batch Size | Scheduling Policy | Timeline | Goodput |
|------------|-------------------|----------|---------|
| 2          | Bidding           | 31:06    | 580     |
| 2          | FCFS              | 31:10    | 281     |
| 4          | Bidding           | 16:04    | 644     |
| 4          | FCFS              | 16:09    | 328     |
| 8          | Bidding           | 08:31    | 710     |
| 8          | FCFS              | 08:35    | 335     |
| 16         | Bidding           | 06:35    | 839     |
| 16         | FCFS              | 06:37    | 741     |

- **Model**: Meta-Llama-3-70B-Instruct (chunked prefill)
| Batch Size | Scheduling Policy | Timeline | Goodput |
|------------|-------------------|----------|---------|
| 2          | FCFS              | 50:07    | 231     |
| 2          | Bidding           | 49:45    | 405     |
| 4          | FCFS              | 26:02    | 292     |
| 4          | Bidding           | 25:38    | 496     |
| 8          | FCFS              | 13:42    | 333     |
| 8          | Bidding           | 13:27    | 535     |
| 16         | FCFS              | 07:35    | 335     |
| 16         | Bidding           | 07:19    | 509     |
