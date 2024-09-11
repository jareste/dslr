import numpy as np
import matplotlib.pyplot as plt
import sys
from utils.utils import parse_csv, is_float, find_most_similar_features, plot_scatter

def main(file_name):
    parsed_data = parse_csv(file_name)
    headers = list(parsed_data[0])
    data = parsed_data[1:]
    
    house_index = headers.index('Hogwarts House')
    houses = np.array([row[house_index] for row in data])
    numerical_columns = [i for i, value in enumerate(data[0]) if is_float(value)]
    numerical_data = {headers[i]: [] for i in numerical_columns}

    for row in data:
        for i in numerical_columns:
            numerical_data[headers[i]].append(float(row[i]) if row[i] else np.nan)

    most_similar_features, max_correlation = find_most_similar_features(numerical_data, headers)

    feature1, feature2 = most_similar_features
    print(f"The most similar features are: {feature1} and {feature2} with a correlation of {max_correlation:.3f}")
    
    plot_scatter(feature1, feature2, np.array(numerical_data[feature1], dtype=float),
                 np.array(numerical_data[feature2], dtype=float), houses, ax=None, save_plot=True)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scatter_plot.py <dataset.csv>")
    else:
        file_name = sys.argv[1]
        main(file_name)
