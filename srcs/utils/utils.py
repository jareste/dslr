import csv
import numpy as np
import math
import matplotlib.pyplot as plt
from sys import exit

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def calculate_correlation(feature1_data, feature2_data):
    valid_indices = ~np.isnan(feature1_data) & ~np.isnan(feature2_data)
    feature1_data = feature1_data[valid_indices]
    feature2_data = feature2_data[valid_indices]

    if len(feature1_data) > 1 and len(feature2_data) > 1:
        return np.corrcoef(feature1_data, feature2_data)[0, 1]
    else:
        return 0

def find_most_similar_features(numerical_data, headers):
    features = list(numerical_data.keys())
    max_correlation = -1
    most_similar_features = (None, None)

    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            feature1 = features[i]
            feature2 = features[j]
            correlation = abs(calculate_correlation(np.array(numerical_data[feature1], dtype=float),
                                                    np.array(numerical_data[feature2], dtype=float)))
            if correlation > max_correlation:
                max_correlation = correlation
                most_similar_features = (feature1, feature2)

    return most_similar_features, max_correlation

def plot_scatter(feature1, feature2, feature1_data, feature2_data, houses, ax=None, save_plot=False):
    house_colors = {'Gryffindor': 'red', 'Hufflepuff': 'yellow', 'Ravenclaw': 'blue', 'Slytherin': 'green'}

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    for house in house_colors:
        house_mask = (houses == house)
        ax.scatter(feature1_data[house_mask], feature2_data[house_mask], alpha=0.5, color=house_colors[house], label=house)

    if ax is None or save_plot:
        ax.set_title(f"Scatter Plot: {feature1} vs {feature2}")
        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
        ax.grid(True)
        ax.legend(loc='upper right')
        plt.tight_layout()
        try:
            plt.savefig(f'/output/scatter_{feature1}_vs_{feature2}.png')
        except:
            print("Failed to save scatter plot")
        plt.show()


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
    try:
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
    except FileNotFoundError:
        print(f"Fatal error: file {file_name} not found. Exiting...")
        exit(1)
    except Exception as e:
        print(f"Fatal error: {e}. Exiting...")
        exit(1)

    return np.array(dataset, dtype=object)


class StandardScaler:
    def __init__(self):
        self._mean = None
        self._std = None

    def fit(self, X):
        self._mean = np.mean(X, axis=0)
        self._std = np.std(X, axis=0)

    def transform(self, X):
        return (X - self._mean) / self._std

def sigmoid(z):
    return 1 / (1 + np.exp(-z))