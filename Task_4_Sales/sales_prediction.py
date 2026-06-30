# Task 4: Sales Prediction using Machine Learning
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate synthetic dataset for immediate execution
np.random.seed(42)
data = pd.DataFrame({
    'TV_Ad_Spend': np.random.rand(100) * 200,
    'Radio_Ad_Spend': np.random.rand(100) * 50,
    'Social_Media_Spend': np.random.rand(100) * 100,
    'Sales': np.random.rand(100) * 500 + 100
})

# Feature selection
X = data[['TV_Ad_Spend', 'Radio_Ad_Spend', 'Social_Media_Spend']]
y = data['Sales']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, predictions))
print("Model R^2 Score:", model.score(X_test, y_test))
