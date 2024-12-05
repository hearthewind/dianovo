import csv
import os
import sys

output_file_name = 'combined.csv'

if __name__ == '__main__':
    input_folder = sys.argv[1]
    csv_files = [file_name for file_name in os.listdir(input_folder) if file_name.endswith('csv')]

    all_records = []

    for csv_file in csv_files:
        for record in csv.DictReader(open(os.path.join(input_folder, csv_file), 'r')):
            all_records.append(record)
    #     all_records = all_records[:-1]

    print('total number of records, ', len(all_records))

    fieldnames = all_records[0].keys()
    writer = csv.DictWriter(open(os.path.join(input_folder, output_file_name), 'w'), fieldnames=fieldnames)
    writer.writeheader()

    writer.writerows(all_records)
