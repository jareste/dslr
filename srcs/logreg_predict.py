import numpy as np
import sys
from utils.utils import sigmoid, parse_csv

def predict_one_vs_all(X, theta_all):
    probs = sigmoid(X @ theta_all.T)
    return np.argmax(probs, axis=1)

def load_weights(weights_file):
    try:
        weights = np.loadtxt(weights_file, delimiter=',', skiprows=1, usecols=range(4))
    except:
        print(f"Fatal error: failed to load weights from {weights_file}. Exiting...")
        exit(1)

    means, stds = [], []
    with open(weights_file, 'r') as f:
        next(f)
        for line in f:
            line = line.strip().split(',')
            mean_val = line[4] if line[4] != '' else '0'
            std_val = line[5] if line[5] != '' else '1'
            means.append(float(mean_val))
            stds.append(float(std_val))
    
    return weights.T, np.array(means[1:]), np.array(stds[1:])

def calculate_accuracy(y_true, y_pred):
    accuracy = np.mean(y_true == y_pred) * 100
    return accuracy

def calculate_precision(y_true, y_pred, num_labels=4):
    precisions = []
    for label in range(num_labels):
        true_positive = np.sum((y_true == label) & (y_pred == label))
        false_positive = np.sum((y_true != label) & (y_pred == label))
        if true_positive + false_positive > 0:
            precision = true_positive / (true_positive + false_positive)
        else:
            precision = 0
        precisions.append(precision)
    macro_precision = np.mean(precisions)
    return precisions, macro_precision

def main(test_file, weights_file='/output/weights.csv'):
    dataset = parse_csv(test_file)
    headers = list(dataset[0])
    data = dataset[1:]

    features = ['Astronomy', 'Defense Against the Dark Arts', 'Herbology', 'Charms', 'Flying']
    feature_indices = [headers.index(f) for f in features]
    
    X_test = np.array([[row[i] for i in feature_indices] for row in data], dtype=float)
    
    theta_all, means, stds = load_weights(weights_file)

    X_test_std = (X_test - means) / stds
    X_test_std = np.c_[np.ones(X_test_std.shape[0]), X_test_std]

    predictions = predict_one_vs_all(X_test_std, theta_all)

    house_mapping = {0: 'Gryffindor', 1: 'Hufflepuff', 2: 'Ravenclaw', 3: 'Slytherin'}
    predicted_houses = [house_mapping[p] for p in predictions]

    output_df = {'Index': [row[0] for row in data], 'Hogwarts House': predicted_houses}

    try:
        with open('/output/predict.csv', 'w') as f:
            f.write('Index,Hogwarts House\n')
            for index, house in zip(output_df['Index'], output_df['Hogwarts House']):
                f.write(f"{int(index)},{house}\n")
    except:
        print("Failed to write predictions to /output/predict.csv")

    if 'Hogwarts House' in headers and any(row[headers.index('Hogwarts House')] is not np.nan for row in data):
        house_index = headers.index('Hogwarts House')
        y_true = np.array([{'Gryffindor': 0, 'Hufflepuff': 1, 'Ravenclaw': 2, 'Slytherin': 3}[row[house_index]] for row in data if row[house_index] is not np.nan])

        predictions = predictions[:len(y_true)]
        
        accuracy = calculate_accuracy(y_true, predictions)
        precisions, macro_precision = calculate_precision(y_true, predictions)
        
        print(f"Test Accuracy: {accuracy:.2f}%")
        for i, house in enumerate(house_mapping.values()):
            print(f"Precision for {house}: {precisions[i]:.2f}")
        print(f"Macro-average Precision: {macro_precision:.2f}")
    else:
        print("No true labels available for accuracy and precision calculation.")

    for index, house in zip(output_df['Index'], output_df['Hogwarts House']):
        print(f"{index}. {house}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python logreg_predict.py <dataset.csv> [weights.csv]")
    else:
        file_name = sys.argv[1]
        try:
            if len(sys.argv) > 2:
                weights_file = sys.argv[2]
                main(file_name, weights_file)
            else:
                main(file_name)
        except Exception as e:
            print(f"Fatal error: {e}. Exiting...")