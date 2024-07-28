### Performance Comparison of Policies (solver only in decodes, window=10)
| request_num | oracle | fcfs | solver | deadline | random | bidding |
|-------------|--------|------|--------|----------|--------|---------|
| 100         | 77     | 55   | 47     | 40       | 41     | 68      |
| 500         | 474    | 387  | 192    | 375      | 363    | 361     |
| 1000        | 978    | 693  | 335    | 636      | 763    | 680     |

### Performance Comparison of Policies (solver only in prefills + decodes, window=100)
| request_num | oracle | fcfs | solver | deadline | random | bidding |
|-------------|--------|------|--------|----------|--------|---------|
| 100         | 77     | 55   | 37     | 39       | 50     | 55      |
| 500         | 474    | 387  | 155    | 332      | 368    | 393     |
| 1000        | 978    | 693  | 304    | 677      | 754    | 840     |

### Workload Configuration
| Workload Type   | Proportion | Deadline          | Max Token |
|-----------------|------------|-------------------|-----------|
| Search          | 33%        | U(1, 2)           | 10        |
| Chatbox         | 33%        | 0.2 * max_token=10| 50        |
| Batch Analysis  | 33%        | U(300, 3600)      | 500       |

## Request Scheduling Metrics

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