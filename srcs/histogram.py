import math
import sys
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import parse_csv, is_float, calculate_statistics

def describe_by_house(parsed_data):
    headers = list(parsed_data[0])
    data = parsed_data[1:]
    
    house_index = headers.index('Hogwarts House')
    
    houses_data = {'Ravenclaw': {}, 'Slytherin': {}, 'Gryffindor': {}, 'Hufflepuff': {}}
    
    numerical_columns = [i for i, value in enumerate(data[0]) if is_float(value)]
    
    for house in houses_data.keys():
        for i in numerical_columns:
            houses_data[house][headers[i]] = []
    
    for row in data:
        house = row[house_index]
        for i in numerical_columns:
            houses_data[house][headers[i]].append(float(row[i]) if row[i] else float('nan'))
    
    return houses_data, [headers[i] for i in numerical_columns]

def calculate_variances(houses_data, features):
    feature_variances = {}

    for feature in features:
        feature_variances[feature] = {}
        for house, data in houses_data.items():
            _, _, std, _, _, _, _, _ = calculate_statistics(data[feature])
            feature_variances[feature][house] = std
    
    return feature_variances

def find_most_homogeneous(feature_variances):
    min_variance_difference = float('inf')
    most_homogeneous_feature = None

    for feature, variances in feature_variances.items():
        variance_values = list(variances.values())
        variance_difference = max(variance_values) - min(variance_values)

        if variance_difference < min_variance_difference:
            min_variance_difference = variance_difference
            most_homogeneous_feature = feature
    
    return most_homogeneous_feature, min_variance_difference

def plot_histograms(numerical_data, house):
    features = list(numerical_data.keys())
    num_features = len(features)
    
    plt.figure(figsize=(12, 8))
    
    for idx, feature in enumerate(features):
        if feature == 'Index':
            continue
        plt.subplot((num_features // 3) + 1, 3, idx)
        plt.hist(numerical_data[feature], bins=30, color='blue', alpha=0.7)
        plt.title(f"{feature[:15]}")
        plt.tight_layout()
    
    plt.savefig(f'/output/histogram_{house}.png')
    plt.show()

def plot_most_homogeneous(houses_data, most_homogeneous_feature):
    h1 = np.array(houses_data['Gryffindor'][most_homogeneous_feature], dtype=float)
    h2 = np.array(houses_data['Hufflepuff'][most_homogeneous_feature], dtype=float)
    h3 = np.array(houses_data['Ravenclaw'][most_homogeneous_feature], dtype=float)
    h4 = np.array(houses_data['Slytherin'][most_homogeneous_feature], dtype=float)

    h1 = h1[~np.isnan(h1)]
    h2 = h2[~np.isnan(h2)]
    h3 = h3[~np.isnan(h3)]
    h4 = h4[~np.isnan(h4)]

    plt.hist(h1, color='red', alpha=0.6)
    plt.hist(h2, color='yellow', alpha=0.6)
    plt.hist(h3, color='blue', alpha=0.6)
    plt.hist(h4, color='green', alpha=0.6)

    plt.legend(['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin'], loc='upper right', frameon=False)
    plt.title(f"Most Homogeneous Feature: {most_homogeneous_feature}")
    plt.xlabel('Marks')
    plt.ylabel('Students')
    plt.tight_layout()

    plt.savefig(f'/output/homogeneous_{most_homogeneous_feature}.png')
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python histogram.py <dataset.csv>")
    else:
        file_name = sys.argv[1]
        
        parsed_data = parse_csv(file_name)
        
        houses_data, features = describe_by_house(parsed_data)
        
        feature_variances = calculate_variances(houses_data, features)
        
        most_homogeneous_feature, variance_difference = find_most_homogeneous(feature_variances)
        
        print(f"The most homogeneous feature across houses is: {most_homogeneous_feature}")
        print(f"Variance difference across houses: {variance_difference:.3f}")
        
        plot_most_homogeneous(houses_data, most_homogeneous_feature)

        if '--all' in sys.argv:
            for house, data in houses_data.items():
                print(f"\n--- Histograms for {house} ---")
                plot_histograms(data, house)
        
