import os
import csv
import random

random.seed(42)  # For reproducibility
def create_csv_of_files(folder_path, output_csv, percentage):
    # Open the CSV file for writing
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        
        # Write the header
        writer.writerow(['File Name', 'File Path'])
        
        # Walk through the folder and its subfolders
        for root, _, files in os.walk(folder_path):
            if files:
                # Filter only .wav files
                flac_files = [file_ for file_ in files if file_.lower().endswith('.flac')]
                if not flac_files:
                    continue
                
                # Shuffle and select a percentage of files
                num_files_to_select = max(1, int(len(flac_files) * (percentage / 100)))
                selected_files = random.sample(flac_files, num_files_to_select)
                
                for file_ in selected_files:
                    file_path = os.path.join(root, file_)
                    if os.path.isfile(file_path):  # Check if it's a file_
                        writer.writerow([file_, file_path])

# Example usage
folder_path = './VCTK/wav48_silence_trimmed'  # Replace with your folder path
output_csv = './files_list.csv'              # Replace with your desired CSV file name
percentage = 50                              # Replace with the percentage of files to include
create_csv_of_files(folder_path, output_csv, percentage)
