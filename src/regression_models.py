import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

def scatter_plot_with_regression_line(y_test, y_pred, text):
  # Calculate the difference between actual and predicted values
  differences = y_pred - y_test

  # Scatter plot of actual vs predicted values
  plt.figure(figsize=(10, 6))
  
  sc = plt.scatter(y_test, y_pred, c=differences, cmap='viridis', edgecolor='k', alpha=0.6)

  # Plot the ideal line where predictions == actual values
  plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'p--', linewidth=2)

  # Add color bar to indicate the magnitude of differences
  plt.colorbar(sc, label='Prediction Error')

  # Plot labels and title
  plt.xlabel('Actual Values')
  plt.ylabel('Predicted Values')
  plt.title('Actual vs Predicted Values ' + text)

  # Show the plot
  plt.grid(True)
  plt.show()

def evaluate_regression_model(y_test, y_pred, tolerance = 0.10):
  # Mean Squared Error (MSE)
  mse = mean_squared_error(y_test, y_pred)
  print(f"Mean Squared Error (MSE): {round(mse, 2)}")

  # Mean Absolute Error (MAE)
  mae = mean_absolute_error(y_test, y_pred)
  print(f"Mean Absolute Error (MAE): {round(mae, 2)}")

  # R-squared (R²)
  r2 = r2_score(y_test, y_pred)
  print(f"R-squared (R²): {round(r2, 2)}")

  # Custom Accuracy: Percentage of predictions within ±10% of the actual values
  accuracy = (abs((y_pred - y_test) / y_test) <= tolerance).mean()
  print(f"Custom Accuracy (within ±10%): {accuracy * 100:.2f}%")

def train_and_evaluate_r(X, y, random_state, graph_title_text):
  # split the data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

  # Create the XGBoost regressor model
  xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)

  # Train the model
  xgb_regressor.fit(X_train, y_train)

  # Make predictions on the test set
  y_pred = xgb_regressor.predict(X_test)

  evaluate_regression_model(y_test, y_pred)
  scatter_plot_with_regression_line(y_test, y_pred, graph_title_text)
