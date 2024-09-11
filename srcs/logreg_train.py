import numpy as np
import sys
from utils.utils import StandardScaler, sigmoid, parse_csv

def cost_function(X, y, theta, Lambda):
    m = len(y)
    h = sigmoid(X @ theta)
    regularization = (Lambda / (2 * m)) * np.sum(np.square(theta[1:]))
    cost = (-1 / m) * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h)) + regularization
    return cost

def calculate_accuracy(X, y, theta):
    predictions = sigmoid(X @ theta) >= 0.5
    accuracy = np.mean(predictions == y) * 100
    return accuracy

def gradient_descent(X, y, theta, alpha, num_iters, Lambda):
    m = len(y)
    cost_history = []
    accuracy_history = []
    
    for _ in range(num_iters):
        h = sigmoid(X @ theta)
        gradient = (1 / m) * (X.T @ (h - y))
        gradient[1:] = gradient[1:] + (Lambda / m) * theta[1:]
        theta = theta - alpha * gradient
        cost = cost_function(X, y, theta, Lambda)
        cost_history.append(cost)
        
        accuracy = calculate_accuracy(X, y, theta)
        accuracy_history.append(accuracy)
        
    return theta, cost_history, accuracy_history

def one_vs_all(X, y, num_labels, alpha, num_iters, Lambda):
    m, n = X.shape
    all_theta = np.zeros((num_labels, n))
    cost_histories = []
    accuracy_histories = []
    
    for i in range(num_labels):
        theta = np.zeros(n)
        y_i = np.where(y == i, 1, 0)
        theta, cost_history, accuracy_history = gradient_descent(X, y_i, theta, alpha, num_iters, Lambda)
        all_theta[i, :] = theta
        cost_histories.append(cost_history)
        accuracy_histories.append(accuracy_history)
    
    return all_theta, cost_histories, accuracy_histories

def main(file_name):
    dataset = parse_csv(file_name)
    headers = list(dataset[0])
    data = dataset[1:]
    
    features = ['Astronomy', 'Defense Against the Dark Arts', 'Herbology', 'Charms', 'Flying']
    feature_indices = [headers.index(f) for f in features]

    clean_data = [row for row in data if all(row[i] is not np.nan for i in feature_indices)]
    
    X = np.array([[row[i] for i in feature_indices] for row in clean_data], dtype=float)
    y = np.array([{'Gryffindor': 0, 'Hufflepuff': 1, 'Ravenclaw': 2, 'Slytherin': 3}[row[headers.index('Hogwarts House')]] for row in clean_data])

    scaler = StandardScaler()
    scaler.fit(X)
    X_std = scaler.transform(X)

    X_std = np.c_[np.ones(X_std.shape[0]), X_std]

    alpha = 0.01
    num_iters = 5000
    num_labels = 4
    Lambda = 10

    theta_all, cost_histories, accuracy_histories = one_vs_all(X_std, y, num_labels, alpha, num_iters, Lambda)

    for i, house in enumerate(['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']):
        print(f"Final training accuracy for {house}: {accuracy_histories[i][-1]:.2f}%")
        print(f"Final cost for {house}: {cost_histories[i][-1]:.4f}")

    try:
        with open('/output/weights.csv', 'w') as f:
            f.write('Gryffindor,Hufflepuff,Ravenclaw,Slytherin,Mean,Std\n')
            for i in range(theta_all.shape[1]):
                for j in range(theta_all.shape[0]):
                    f.write(f'{theta_all[j, i]},')
                f.write(f'{scaler._mean[i - 1] if i > 0 else ""},{scaler._std[i - 1] if i > 0 else ""}\n')
    except:
        print("Failed to write weights to /output/weights.csv")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python logreg_train.py <dataset.csv>")
    else:
        file_name = sys.argv[1]
        try:
            main(file_name)
        except Exception as e:
            print(f"Fatal error: {e}. Exiting...")
            sys.exit(1)
