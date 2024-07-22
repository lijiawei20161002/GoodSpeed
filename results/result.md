### Performance Comparison of Policies
| request_num | oracle | fcfs | solver | deadline | random | bidding |
|-------------|--------|------|--------|----------|--------|---------|
| 100         | 77     | 55   | 47     | 40       | 41     | 68      |
| 500         | 474    | 387  | 192    | 375      | 363    | 361     |
| 1000        | 978    | 693  | 335    | 636      | 763    | 680     |

### Workload Configuration
| Workload Type   | Proportion | Deadline          | Max Token |
|-----------------|------------|-------------------|-----------|
| Search          | 33%        | U(1, 2)           | 10        |
| Chatbox         | 33%        | 0.2 * max_token=10| 50        |
| Batch Analysis  | 33%        | U(300, 3600)      | 500       |