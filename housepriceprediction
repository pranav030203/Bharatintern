# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset into a pandas DataFrame
# You can replace 'your_dataset.csv' with the path to your dataset
data = pd.read_csv('your_dataset.csv')

# Assuming you have a dataset with features (X) and target variable (y)
X = data[['feature1', 'feature2', 'feature3']]  # Replace with your actual feature columns
y = data['target']  # Replace with your actual target variable column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display the model's performance
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Now, you can use the trained model to make predictions on new data
# For example, to predict the price of a house with specific features:
new_data = np.array([[feature1_value, feature2_value, feature3_value]])  # Replace with actual feature values
predicted_price = model.predict(new_data)
print("Predicted House Price:", predicted_price[0])
