import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
file_path = 'bidding_output_metrics.csv'  
df = pd.read_csv(file_path)

plt.rcParams['font.family'] = 'Comic Sans MS'
plt.rcParams['font.size'] = 20

# Visualize the frequency distribution of deadline
plt.figure(figsize=(10, 7))
plt.hist(df['deadline']-df['arrival_time'], bins=30, alpha=0.7, edgecolor='black')
plt.title('Frequency Distribution of Deadline')
plt.xlabel('Deadline (seconds)')
plt.ylabel('Frequency')
plt.grid(True)

# Save the figure
plt.savefig('bidding_deadline_distribution.png')

# Visualize the frequency distribution of finish_time
plt.figure(figsize=(10, 7))
plt.hist(df['finish_time']-df['arrival_time'], bins=30, alpha=0.7, edgecolor='black', color='orange')
plt.title('Frequency Distribution of Finish Time')
plt.xlabel('Finish Time (seconds)')
plt.ylabel('Frequency')
plt.grid(True)

# Save the figure
plt.savefig('bidding_finish_time_distribution.png')

# Visualize the frequency distribution of price
plt.figure(figsize=(10, 7))
plt.hist(df['price']/1000.0, bins=30, alpha=0.7, edgecolor='black', color='green')
plt.title('Frequency Distribution of Price')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.grid(True)

# Save the figure
plt.savefig('bidding_price_distribution.png')