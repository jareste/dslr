import csv
import math
import numpy as np
import sys
from utils.utils import parse_csv, is_float, calculate_statistics

def print_describe(parsed_data):
    headers = parsed_data[0]
    data = parsed_data[1:]

    columns = np.transpose(data)
    
    stat_labels = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    
    stats_dict = {}

    for i, col in enumerate(columns):
        col_name = headers[i]
        if is_float(col[1]):
            col_data = [float(x) if is_float(x) else float('nan') for x in col[1:]]
            stats_dict[col_name] = calculate_statistics(col_data)
        else:
            stats_dict[col_name] = ['N/A'] * len(stat_labels)
    
    stat_line = f"{'':<15}" + ''.join([f"{label:<12}" for label in stat_labels])
    print(stat_line)
    
    for col_name, stats in stats_dict.items():
        stats_line = f"{col_name[:12]:<15}" + ''.join([f"{stat:<12}" if stat == 'N/A' else f"{stat:<12.3f}" for stat in stats])
        print(stats_line)


def describe(file_name):
    parsed_data = parse_csv(file_name)
    print_describe(parsed_data)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python describe.py <dataset.csv>")
    else:
        file_name = sys.argv[1]
        describe(file_name)

