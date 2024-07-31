import matplotlib.pyplot as plt

# Data
requests = [100, 500, 1000, 5000]
oracle_goodput = [77, 474, 978, 4923]
fcfs_goodput = [55, 387, 693, 4290]
solver_goodput = [37, 155, 304, 876]
deadline_goodput = [39, 332, 677, 4048]
random_goodput = [50, 368, 754, 4192]
bidding_goodput = [55, 393, 840, 4605]

# Create the plot
plt.rcParams['font.family'] = 'Comic Sans MS'
fig, ax = plt.subplots(figsize=(12, 6))

# Bar positions
bar_width = 0.12
index = range(len(requests))

# Bars
bars1 = plt.bar(index, oracle_goodput, bar_width, color='pink', hatch='.', label='Oracle Goodput')
bars2 = plt.bar([i + bar_width for i in index], fcfs_goodput, bar_width, color='orchid', hatch='//', label='FCFS Goodput')
bars3 = plt.bar([i + bar_width*2 for i in index], solver_goodput, bar_width, color='mediumslateblue', hatch='*', label='Sovler Goodput')
bars4 = plt.bar([i + bar_width*3 for i in index], deadline_goodput, bar_width, color='steelblue', hatch='|', label='Deadline Prioritize Goodput')
bars5 = plt.bar([i + bar_width*4 for i in index], random_goodput, bar_width, color='cadetblue', hatch='-', label='Random Goodput')
bars6 = plt.bar([i + bar_width*5 for i in index], bidding_goodput, bar_width, color='green', hatch='+', label='Bidding Goodput')

# Add text labels on the bars
for i, v in enumerate(oracle_goodput):
    plt.text(i, v + 20, str(v), ha='center', fontsize=14)
for i, v in enumerate(fcfs_goodput):
    plt.text(i + bar_width, v + 20, str(v), ha='center', fontsize=14)
for i, v in enumerate(solver_goodput):
    plt.text(i + bar_width*2, v + 20, str(v), ha='center', fontsize=14)
for i, v in enumerate(deadline_goodput):
    plt.text(i + bar_width*3, v + 20, str(v), ha='center', fontsize=14)
for i, v in enumerate(random_goodput):
    plt.text(i + bar_width*4, v + 20, str(v), ha='center', fontsize=14)
for i, v in enumerate(bidding_goodput):
    plt.text(i + bar_width*5, v + 20, str(v), ha='center', fontsize=14)

# Labels and titles
plt.xlabel('Number of Requests', fontsize=20)
plt.ylabel('Goodput', fontsize=20)
plt.title('Performance Comparison of Different Scheduling Policies for Batching Requests', fontsize=20)
plt.xticks([i + bar_width * 2 for i in index], requests, fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.tight_layout()
plt.grid()
plt.savefig("goodput.png")