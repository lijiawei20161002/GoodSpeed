import csv

# Input and output file paths
input_file = 'BurstGPT_1.csv'
output_file = 'gpt4.csv'

# Open the input CSV file for reading
with open(input_file, mode='r', newline='') as infile:
    reader = csv.DictReader(infile)

    # Prepare the modified records with adjusted Log Type
    modified_records = []
    for row in reader:
        # Update the 'Log Type' based on the condition
        if row['Model'] == 'GPT-4':
            if row['Log Type'] == 'API log':
                row['Log Type'] = 'search'
            elif row['Log Type'] == 'Conversation log':
                row['Log Type'] = 'chatbox'

            # Add the modified row to the list
            modified_records.append(row)

# Define fieldnames for the output CSV (same as the input file)
fieldnames = reader.fieldnames  # Use the same fieldnames as the input

# Open the output CSV file for writing
with open(output_file, mode='w', newline='') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)

    # Write the header
    writer.writeheader()

    # Write the modified records to the output file
    writer.writerows(modified_records)

print(f"Modified records with updated Log Type written to {output_file}")