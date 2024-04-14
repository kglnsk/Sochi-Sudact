# Function to extract text from files in a given folder.

import textract
import pandas as pd 
import os 

csv_path = 'sample.csv'
directory_path = 'supplement/'

df = pd.read_csv(csv_path)


def extract_text_from_files(folder_path):
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".rtf"):  # Check if the file is an RTF file
            file_path = os.path.join(folder_path, filename)
            text = textract.process(file_path).decode('utf-8')  # Extract text and decode bytes to string
            texts.append(text)
    return texts

# Loop through each folder in the directory and read RTF files.
for folder_name in os.listdir(directory_path):
    full_folder_path = os.path.join(directory_path, folder_name)
    if os.path.isdir(full_folder_path):  # Check if it is a directory
        class_texts = extract_text_from_files(full_folder_path)
        # Append each text as a new row in the DataFrame with the corresponding class name.
        for text in class_texts:
            df = df.append({'class': folder_name, 'text': text}, ignore_index=True)

            
# Save the merged data into a new CSV file.
output_csv_path = 'merged_output.csv'
df.to_csv(output_csv_path, index=False)