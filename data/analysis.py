import pandas as pd
import matplotlib.pyplot as plt

# Load the data into a DataFrame
file_path = "BurstGPT_1.csv"
df = pd.read_csv(file_path)

# Display the first few rows of the DataFrame
print("Data Preview:")
print(df.head())

# Display general information about the DataFrame
print("\nData Information:")
print(df.info())

# Combine Model and Log Type into a new column
df['Model_Log_Type'] = df['Model'] + ' (' + df['Log Type'] + ')'

# Count the number of unique model-log type combinations
unique_combinations = df['Model_Log_Type'].nunique()
print(f"\nNumber of Unique Model-Log Type Combinations: {unique_combinations}")

# Group by combined Model and Log Type and calculate statistics
combination_statistics = df.groupby('Model_Log_Type').agg(
    total_requests=('Timestamp', 'count'),
    average_input_length=('Request tokens', 'mean'),
    average_output_length=('Response tokens', 'mean'),
    total_tokens=('Total tokens', 'sum')
).reset_index()

print("\nModel-Log Type Statistics:")
print(combination_statistics)

# Display distribution of input and output sequence lengths based on the combination
print("\nInput Sequence Length Distribution by Model-Log Type:")
print(df.groupby('Model_Log_Type')['Request tokens'].describe())
print("\nOutput Sequence Length Distribution by Model-Log Type:")
print(df.groupby('Model_Log_Type')['Response tokens'].describe())

'''
# Additional analysis: Distribution of arrival times (Timestamp)
print("\nDistribution of Arrival Times:")
df['Timestamp'].hist(bins=30, grid=False)
plt.title('Arrival Time Distribution')
plt.xlabel('Timestamp')
plt.ylabel('Frequency')
plt.savefig("arrival_time.png")

# Additional analysis: Distribution of request tokens (input sequence length)
print("\nDistribution of Request Tokens (Input Sequence Length):")
df['Request tokens'].hist(bins=30, grid=False)
plt.title('Request Tokens Distribution')
plt.xlabel('Request tokens (Input Length)')
plt.ylabel('Frequency')
plt.savefig("request_token.png")

# Additional analysis: Distribution of response tokens (output sequence length)
print("\nDistribution of Response Tokens (Output Sequence Length):")
df['Response tokens'].hist(bins=30, grid=False)
plt.title('Response Tokens Distribution')
plt.xlabel('Response tokens (Output Length)')
plt.ylabel('Frequency')
plt.savefig("response_token.png")

# Visualizing input/output lengths for each model-log type combination
plt.figure(figsize=(12, 6))
df.boxplot(column='Request tokens', by='Model_Log_Type', grid=False)
plt.xticks(rotation=90)
plt.title('Boxplot of Request Tokens by Model-Log Type')
plt.suptitle('')  # Suppress the automatic title
plt.xlabel('Model (Log Type)')
plt.ylabel('Request Tokens')
plt.savefig("request_token_model_log.png")

plt.figure(figsize=(12, 6))
df.boxplot(column='Response tokens', by='Model_Log_Type', grid=False)
plt.xticks(rotation=90)
plt.title('Boxplot of Response Tokens by Model-Log Type')
plt.suptitle('')  # Suppress the automatic title
plt.xlabel('Model (Log Type)')
plt.ylabel('Response Tokens')
plt.savefig("response_token_model_log.png")
'''