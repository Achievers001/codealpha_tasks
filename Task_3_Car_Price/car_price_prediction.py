# Task 3: Car Price Prediction with Machine Learning
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Generate synthetic dataset for immediate execution
np.random.seed(42)
data = pd.DataFrame({
    'Year': np.random.randint(2010, 2024, 100),
    'Horsepower': np.random.randint(100, 400, 100),
    'Mileage': np.random.randint(10000, 150000, 100),
    'Price': np.random.randint(5000, 50000, 100)
})

# Feature selection and Target
X = data[['Year', 'Horsepower', 'Mileage']]
y = data['Price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the regression model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate performance
predictions = model.predict(X_test)
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
print("Model Training Complete.")
