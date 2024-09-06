import csv
import math

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def calculate_statistics(data):
    data = [x for x in data if not math.isnan(x)]
    
    count = len(data)
    mean = sum(data) / count if count > 0 else float('nan')
    
    variance = sum((x - mean) ** 2 for x in data) / count if count > 0 else float('nan')
    std = math.sqrt(variance)
    
    min_val = min(data) if count > 0 else float('nan')
    max_val = max(data) if count > 0 else float('nan')
    
    data_sorted = sorted(data)
    percentile_25 = data_sorted[int(0.25 * (count - 1))] if count > 0 else float('nan')
    percentile_50 = data_sorted[int(0.50 * (count - 1))] if count > 0 else float('nan')
    percentile_75 = data_sorted[int(0.75 * (count - 1))] if count > 0 else float('nan')
    
    return count, mean, std, min_val, percentile_25, percentile_50, percentile_75, max_val

def describe(file_name):
    with open(file_name, mode='r') as file:
        csv_reader = csv.reader(file)
        
        headers = next(csv_reader)
        
        first_row = next(csv_reader)
        numerical_columns = [i for i, value in enumerate(first_row) if is_float(value)]
        
        numerical_data = {headers[i]: [] for i in numerical_columns}
        
        for i in numerical_columns:
            numerical_data[headers[i]].append(float(first_row[i]) if first_row[i] else float('nan'))
        
        for row in csv_reader:
            for i in numerical_columns:
                numerical_data[headers[i]].append(float(row[i]) if row[i] else float('nan'))
    
    stats = {}
    for feature, data in numerical_data.items():
        stats[feature] = calculate_statistics(data)
    
    print(f"{'':<8}", end="")
    for feature in numerical_data.keys():
        print(f"{feature[:10]:<15}", end="")
    print()
    
    stat_labels = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    for i, label in enumerate(stat_labels):
        print(f"{label:<8}", end="")
        for feature in numerical_data.keys():
            print(f"{stats[feature][i]:<15.3f}", end="")
        print()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python describe.py <dataset.csv>")
    else:
        describe(sys.argv[1])
