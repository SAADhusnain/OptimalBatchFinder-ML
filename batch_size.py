import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load dataset
file_path = 'Your data set'
dataset = pd.read_excel(file_path)

# Select Features and Target
features = dataset[['Parameters', '-', '-', '-']]
target = dataset['Target Variable']

# Assign to X and y
X, y = features, target

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
def create_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# List of batch sizes to test
batch_sizes = [8, 16, 32, 64, 128, 256]

# Dictionary to store the results
results = {}

for batch_size in batch_sizes:
    print(f"Testing batch size: {batch_size}")
    
    # Create a new model for each batch size
    model = create_model()
    
    # Train the model with the current batch size
    model.fit(X_train, y_train, epochs=10, batch_size=batch_size, verbose=0)
    
    # Make predictions on the test set
    predictions = model.predict(X_test)
    
    # Calculate the Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)  # Additional metric for better understanding
    
    # Store the results
    results[batch_size] = {'MSE': mse, 'RMSE': rmse}
    print(f"MSE for batch size {batch_size}: {mse:.2f}")
    print(f"RMSE for batch size {batch_size}: {rmse:.2f}\n")

# Plot the results
plt.figure(figsize=(10,6))
plt.plot(results.keys(), [value['MSE'] for value in results.values()], marker='o', label='Mean Squared Error (MSE)')
plt.plot(results.keys(), [value['RMSE'] for value in results.values()], marker='o', label='Root Mean Squared Error (RMSE)')
plt.title('Batch Size vs Error Metrics')
plt.xlabel('Batch Size')
plt.ylabel('Error Value')
plt.legend()
plt.grid(True)
plt.show()

# Additional Insight: Print the best batch size based on the lowest RMSE
best_batch_size = min(results, key=lambda x: results[x]['RMSE'])
print(f"\nBest Batch Size (based on lowest RMSE): {best_batch_size} with RMSE: {results[best_batch_size]['RMSE']:.2f}")
