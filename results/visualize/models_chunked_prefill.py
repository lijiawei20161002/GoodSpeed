import matplotlib.pyplot as plt

# Define new data based on your table
models = ["deepseek-llm-7b-base", "Llama-2-13b-chat-hf", "Qwen1.5-14B", "deepseek-llm-67b-chat", "Llama-3-70B"]
oracle = [978, 904, 943, 670, 589]
fcfs = [910, 753, 703, 335, 335]
deadline = [894, 668, 651, 332, 335]
random = [904, 762, 732, 346, 341]
bidding = [933, 856, 853, 563, 488]

# Create the plot
plt.rcParams['font.family'] = 'Comic Sans MS'
fig, ax = plt.subplots(figsize=(12, 6))

# Bar positions
bar_width = 0.15
index = range(len(models))

# Bars
bars1 = plt.bar(index, oracle, bar_width, color='pink', hatch='-', label='Oracle Goodput')
bars2 = plt.bar([i + bar_width for i in index], fcfs, bar_width, color='orchid', hatch='/', label='FCFS Goodput')
bars3 = plt.bar([i + bar_width*2 for i in index], deadline, bar_width, color='steelblue', hatch='+', label='Deadline Goodput')
bars4 = plt.bar([i + bar_width*3 for i in index], random, bar_width, color='cadetblue', hatch='.', label='Random Goodput')
bars5 = plt.bar([i + bar_width*4 for i in index], bidding, bar_width, color='green', label='Bidding Goodput')

# Add text labels on the bars
for i, v in enumerate(oracle):
    plt.text(i, v + 10, str(v), ha='center', fontsize=10)
for i, v in enumerate(fcfs):
    plt.text(i + bar_width, v + 10, str(v), ha='center', fontsize=10)
for i, v in enumerate(deadline):
    plt.text(i + bar_width*2, v + 10, str(v), ha='center', fontsize=10)
for i, v in enumerate(random):
    plt.text(i + bar_width*3, v + 10, str(v), ha='center', fontsize=10)
for i, v in enumerate(bidding):
    plt.text(i + bar_width*4, v + 10, str(v), ha='center', fontsize=10)

# Labels and titles
plt.xlabel('Models', fontsize=16)
plt.ylabel('Goodput', fontsize=16)
plt.title('Performance Comparison of Different Scheduling Policies for Batching Requests (chunked prefill)', fontsize=16)
plt.xticks([i + bar_width*2 for i in index], models, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc='upper right', bbox_to_anchor=(1.0, 0.95), fontsize=12)
plt.tight_layout()
plt.grid()
plt.savefig("goodput_comparison_models_chunked_prefill.png")