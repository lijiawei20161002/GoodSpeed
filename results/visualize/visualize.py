import matplotlib.pyplot as plt

# Data
requests = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
solver_goodput = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
fcfs_goodput = [82, 173, 256, 360, 459, 547, 641, 732, 827, 930]
timeline = ["01:21", "02:53", "04:43", "06:09", "07:20", "09:44", "11:17", "13:45", "13:41", "20:41"]

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Bar positions
bar_width = 0.35
index = range(len(requests))

# Bars
bars1 = plt.bar(index, solver_goodput, bar_width, color='lavender', label='Solver Goodput')
bars2 = plt.bar([i + bar_width for i in index], fcfs_goodput, bar_width, color='rosybrown', hatch='//', label='FCFS Goodput')

# Labels and titles
plt.xlabel('Number of Requests', fontsize=14)
plt.ylabel('Goodput', fontsize=14)
plt.title('Performance Comparison of Solver and FCFS', fontsize=14)
plt.xticks([i + bar_width / 2 for i in index], requests)
plt.legend(fontsize=14)

# Add timeline annotations
for i in range(len(requests)):
    plt.text(i - 0.1, max(solver_goodput[i], fcfs_goodput[i]) + 10, timeline[i], ha='center', va='bottom', fontsize=14)

plt.tight_layout()
plt.grid()
plt.savefig("goodput.png")