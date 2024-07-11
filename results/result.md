### Performance Comparison of Policies

| Number of Requests | Solver Goodput | FCFS Goodput | Timeline        |
|--------------------|----------------|--------------|-----------------|
| 100                | 100            | 82           | 01:21 min       |
| 200                | 200            | 173          | 02:53 min       |
| 300                | 300            | 256          | 04:43 min       |
| 400                | 400            | 360          | 06:09 min       |
| 500                | 500            | 459          | 07:20 min       |
| 600                | 600            | 547          | 09:44 min       |
| 700                | 700            | 641          | 11:17 min       |
| 800                | 800            | 732          | 13:45 min       |
| 900                | 900            | 827          | 13:41 min       |
| 1000               | 1000           | 930          | 20:41 min       |

### Workload Configuration
| Workload Type   | Proportion | Deadline          | Max Token |
|-----------------|------------|-------------------|-----------|
| Search          | 33%        | U(0.1, 0.5)       | 10        |
| Chatbox         | 33%        | 0.5 * max_token=5 | 10        |
| Batch Analysis  | 33%        | U(300, 3600)      | 10        |