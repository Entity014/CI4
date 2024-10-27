import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel("AirQualityUCI.xlsx", na_values=-200)

data.dropna(inplace=True)
X = data.iloc[:, [2, 6, 8, 10, 11, 12, 13, 14]].values
y = data.iloc[:, 5].values

X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_scaled = (X - X_mean) / X_std

hidden_layer = [20]


class MLP:
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size

        # Initialize weights
        self.weights = []
        prev_size = input_size
        for size in hidden_layer_sizes:
            self.weights.append(np.random.rand(prev_size, size))
            prev_size = size
        self.weights.append(np.random.rand(prev_size, output_size))

    def forward(self, X):
        self.layers = [X]
        for weight in self.weights:
            X = self.relu(np.dot(X, weight))
            self.layers.append(X)
        return X

    def linear(self, x):
        return x

    def tanh(self, x):
        return np.tanh(x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def predict(self, X):
        return self.forward(X)


# PSO Algorithm
class Particle:
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        self.mlp = MLP(input_size, hidden_layer_sizes, output_size)
        self.position = [w.copy() for w in self.mlp.weights]
        self.velocity = [np.random.rand(*w.shape) * 0.1 for w in self.position]
        self.best_position = self.position
        self.best_error = float("inf")


def pso(X, y, num_particles, num_iterations):
    particles = [Particle(X.shape[1], hidden_layer, 1) for _ in range(num_particles)]
    global_best_position = None
    global_best_error = float("inf")

    for _ in range(num_iterations):
        for particle in particles:
            y_pred = particle.mlp.predict(X)
            error = np.mean(np.abs(y - y_pred.flatten()))

            if error < particle.best_error:
                particle.best_error = error
                particle.best_position = [w.copy() for w in particle.position]

            if error < global_best_error:
                global_best_error = error
                global_best_position = [w.copy() for w in particle.position]

            inertia = 0.5
            cognitive = 1.5
            social = 1.5

            for i in range(len(particle.position)):
                r1 = np.random.rand(*particle.position[i].shape)
                r2 = np.random.rand(*particle.position[i].shape)
                particle.velocity[i] = (
                    inertia * particle.velocity[i]
                    + cognitive
                    * r1
                    * (particle.best_position[i] - particle.position[i])
                    + social * r2 * (global_best_position[i] - particle.position[i])
                )
                particle.position[i] += particle.velocity[i]

    return global_best_position


num_folds = 10
fold_size = len(X_scaled) // num_folds
mae_results = []

for fold in range(num_folds):
    test_indices = list(range(fold * fold_size, (fold + 1) * fold_size))
    train_indices = list(set(range(len(X_scaled))) - set(test_indices))

    X_train = X_scaled[train_indices]
    y_train = y[train_indices]
    X_test = X_scaled[test_indices]
    y_test = y[test_indices]

    best_weights = pso(X_train, y_train, num_particles=15, num_iterations=200)

    final_mlp = MLP(X_train.shape[1], hidden_layer, 1)
    final_mlp.weights = best_weights
    y_test_pred = final_mlp.predict(X_test)

    mae = np.mean(np.abs(y_test - y_test_pred.flatten()))
    mae_results.append(mae)

    print(f"Fold {fold + 1}/{num_folds} - Mean Absolute Error: {mae}")

average_mae = np.mean(mae_results)
print(f"Average Mean Absolute Error across all folds: {average_mae}")

# Best weights after final training on the entire dataset
best_weights = pso(X_scaled, y, num_particles=15, num_iterations=200)
final_mlp = MLP(X_scaled.shape[1], hidden_layer, 1)
final_mlp.weights = best_weights
y_pred_all = final_mlp.predict(X_scaled)

# Path where you want to save the graph
output_path = f"benzene_concentration_{hidden_layer[0]}_{average_mae}.png"  # Change this to your desired file path

plt.figure(figsize=(12, 6))
plt.plot(y, label="Actual Benzene Concentration", color="blue", alpha=0.5)
plt.plot(y_pred_all, label="Predicted Benzene Concentration", color="orange", alpha=0.5)

# Adding vertical lines for each fold
for fold in range(1, num_folds):
    plt.axvline(x=fold * fold_size, color="red", linestyle="--", linewidth=1)

plt.title("Benzene Concentration: Actual vs Predicted")
plt.xlabel("Sample Index")
plt.ylabel("Benzene Concentration (µg/m³)")
plt.legend()
plt.grid()

# Save the figure
plt.savefig(output_path)

# Show the plot
plt.show()
