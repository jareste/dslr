import csv
import numpy as np
import math

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def percentile(data, p):
    p = p / 100
    return data[int(p * (len(data) - 1))] if len(data) > 0 else float('nan')

def calculate_statistics(data):
    data = [x for x in data if not (math.isnan(x) or x == 'N/A')]
    
    count = len(data)
    mean = sum(data) / count if count > 0 else float('nan')
    
    variance = sum((x - mean) ** 2 for x in data) / count if count > 0 else float('nan')
    std = math.sqrt(variance)
    
    min_val = min(data) if count > 0 else float('nan')
    max_val = max(data) if count > 0 else float('nan')
    
    data_sorted = sorted(data)
    percentile_25 = percentile(data_sorted, 25)
    percentile_50 = percentile(data_sorted, 50)
    percentile_75 = percentile(data_sorted, 75)
    
    return count, mean, std, min_val, percentile_25, percentile_50, percentile_75, max_val

def parse_csv(file_name):
    dataset = list()
    with open(file_name, mode='r') as file:
        csv_reader = csv.reader(file)
        
        headers = next(csv_reader)
        dataset.append(headers)
        
        for row in csv_reader:
            processed_row = list()
            for value in row:
                try:
                    value = float(value)
                except ValueError:
                    if not value:
                        value = np.nan
                processed_row.append(value)
            dataset.append(processed_row)
    
    return np.array(dataset, dtype=object)
