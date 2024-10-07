import pandas as pd
file_path = "BurstGPT_1.csv"
# Load the data into a DataFrame
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