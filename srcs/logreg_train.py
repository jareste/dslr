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

def batch_gradient_descent(X, y, theta, alpha, num_iters, Lambda):
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

def stochastic_gradient_descent(X, y, theta, alpha, num_iters, Lambda):
    m = len(y)
    cost_history = []
    accuracy_history = []

    for _ in range(num_iters):
        for i in range(m):
            x_i = X[i].reshape(1, -1)
            y_i = y[i]
            
            h_i = sigmoid(x_i @ theta)
            gradient = (x_i.T @ (h_i - y_i)).flatten()
            gradient[1:] = gradient[1:] + (Lambda / m) * theta[1:]

            theta = theta - alpha * gradient

        cost = cost_function(X, y, theta, Lambda)
        cost_history.append(cost)
        accuracy = calculate_accuracy(X, y, theta)
        accuracy_history.append(accuracy)

    return theta, cost_history, accuracy_history

def mini_batch_gradient_descent(X, y, theta, alpha, num_iters, Lambda, batch_size=32):
    m = len(y)
    cost_history = []
    accuracy_history = []

    for _ in range(num_iters):
        indices = np.arange(m)
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

        for start_idx in range(0, m, batch_size):
            end_idx = min(start_idx + batch_size, m)
            x_batch = X[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]

            h = sigmoid(x_batch @ theta)
            gradient = (1 / len(y_batch)) * (x_batch.T @ (h - y_batch))
            gradient[1:] = gradient[1:] + (Lambda / m) * theta[1:]

            theta = theta - alpha * gradient

        cost = cost_function(X, y, theta, Lambda)
        cost_history.append(cost)
        accuracy = calculate_accuracy(X, y, theta)
        accuracy_history.append(accuracy)

    return theta, cost_history, accuracy_history

def adam_optimizer(X, y, theta, alpha, num_iters, Lambda, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m = len(y)
    cost_history = []
    accuracy_history = []

    mt = np.zeros_like(theta)
    vt = np.zeros_like(theta)

    for t in range(1, num_iters + 1):
        h = sigmoid(X @ theta)
        gradient = (1 / m) * (X.T @ (h - y))
        gradient[1:] = gradient[1:] + (Lambda / m) * theta[1:]

        mt = beta1 * mt + (1 - beta1) * gradient
        vt = beta2 * vt + (1 - beta2) * (gradient ** 2)

        mt_hat = mt / (1 - beta1 ** t)
        vt_hat = vt / (1 - beta2 ** t)

        theta = theta - alpha * (mt_hat / (np.sqrt(vt_hat) + epsilon))

        cost = cost_function(X, y, theta, Lambda)
        cost_history.append(cost)
        accuracy = calculate_accuracy(X, y, theta)
        accuracy_history.append(accuracy)

    return theta, cost_history, accuracy_history

def one_vs_all(X, y, num_labels, alpha, num_iters, Lambda, optimizer='batch', batch_size=32):
    m, n = X.shape
    all_theta = np.zeros((num_labels, n))
    cost_histories = []
    accuracy_histories = []
    
    optimizer_functions = {
        'batch': batch_gradient_descent,
        'stochastic': stochastic_gradient_descent,
        'mini-batch': mini_batch_gradient_descent,
        'adam': adam_optimizer,
        'normal': gradient_descent
    }

    for i in range(num_labels):
        theta = np.zeros(n)
        y_i = np.where(y == i, 1, 0)
        
        if optimizer == 'mini-batch':
            theta, cost_history, accuracy_history = optimizer_functions[optimizer](X, y_i, theta, alpha, num_iters, Lambda, batch_size)
        else:
            theta, cost_history, accuracy_history = optimizer_functions[optimizer](X, y_i, theta, alpha, num_iters, Lambda)

        all_theta[i, :] = theta
        cost_histories.append(cost_history)
        accuracy_histories.append(accuracy_history)

    return all_theta, cost_histories, accuracy_histories

def main(file_name, optimizer='batch', num_iters=300):
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
    num_labels = 4
    Lambda = 10

    theta_all, cost_histories, accuracy_histories = one_vs_all(X_std, y, num_labels, alpha, num_iters, Lambda, optimizer)

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
        print("Usage: python logreg_train.py <dataset.csv> [optimizer] [batch_size]")
        print("Optimizer options: 'batch', 'stochastic', 'mini-batch', 'adam'")
    else:
        file_name = sys.argv[1]
        optimizer = sys.argv[2] if len(sys.argv) > 2 else 'stochastic'
        num_iters = int(sys.argv[3]) if len(sys.argv) > 3 else 300
        try:
            main(file_name, optimizer, num_iters)
        except Exception as e:
            print(f"Fatal error: {e}. Exiting...")
            sys.exit(1)
