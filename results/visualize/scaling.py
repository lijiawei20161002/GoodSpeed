import matplotlib.pyplot as plt

# Data
requests = [100, 500, 1000, 5000]
oracle_goodput = [77, 474, 978, 4923]
fcfs_goodput = [55, 387, 693, 4290]
deadline_goodput = [39, 332, 677, 4048]
random_goodput = [50, 368, 754, 4192]
bidding_goodput = [55, 393, 840, 4605]

# Create the plot
plt.rcParams['font.family'] = 'Comic Sans MS'
fig, ax = plt.subplots(figsize=(12, 6))

# Plot lines
plt.plot(requests, oracle_goodput, color='pink', marker='o', label='Oracle Goodput')
plt.plot(requests, fcfs_goodput, color='orchid', marker='*', label='FCFS Goodput')
plt.plot(requests, deadline_goodput, color='steelblue', marker='^', label='Deadline Prioritize Goodput')
plt.plot(requests, random_goodput, color='cadetblue', marker='+', label='Random Goodput')
plt.plot(requests, bidding_goodput, color='green', marker='x', label='Bidding Goodput')

# Adding text labels for the last point of each series
# Adjust position based on the context of each line
label_offset = 50
horizontal_positions = [50, -50, 100, -100, 50]  # Positive for right, negative for left adjustments

# Draw labels near the last point but adjust positions to avoid crowding
for i, (label, hp_offset) in enumerate(zip(
    [oracle_goodput, fcfs_goodput, deadline_goodput, random_goodput, bidding_goodput],
    horizontal_positions
)):
    last_x = requests[-1]
    last_y = label[-1]
    # Conditional horizontal adjustment to position labels more clearly
    if hp_offset > 0:
        ha = 'left'
    else:
        ha = 'right'
    plt.text(last_x + hp_offset, last_y, str(last_y), ha=ha, va='center', fontsize=14)

# Labels and titles
plt.xlabel('Number of Requests', fontsize=20)
plt.ylabel('Goodput', fontsize=20)
plt.title('Performance Comparison of Different Scheduling Policies for Batching Requests', fontsize=20)
plt.xticks(requests, fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig("scaling-llama2-13b-chat-hf.png")