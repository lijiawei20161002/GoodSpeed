import matplotlib.pyplot as plt

# Data
batch_sizes = [2, 4, 8, 16]
goodput_bidding = [580, 644, 710, 839]
timeline_bidding = ["31:06", "16:04", "08:31", "06:35"]
goodput_fcfs = [281, 328, 335, 741]
timeline_fcfs = ["31:10", "16:09", "08:35", "06:37"]

# Plotting
plt.rcParams['font.family'] = 'Comic Sans MS'
fig, ax = plt.subplots(figsize=(6, 4))

# Bidding
plt.plot(batch_sizes, goodput_bidding, marker='*', color='green', linewidth=3, label='Bidding')
for i, txt in enumerate(timeline_bidding):
    plt.annotate(txt, (batch_sizes[i], goodput_bidding[i]), textcoords="offset points", xytext=(0,10), ha='center')

# FCFS
plt.plot(batch_sizes, goodput_fcfs, marker='o', color='pink', linewidth=3, label='FCFS')
for i, txt in enumerate(timeline_fcfs):
    plt.annotate(txt, (batch_sizes[i], goodput_fcfs[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Adding titles and labels
plt.title('Goodput Comparison: FCFS vs Bidding')
plt.xlabel('Batch Size')
plt.ylabel('Goodput')
plt.legend()
plt.grid(True)

# Display the plot
plt.savefig("goodput-batch-size-Llama-2-13b-chat-hf.png")
