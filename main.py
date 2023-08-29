import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load your data
data = pd.read_csv('customer_transactions.csv')

# Calculate Recency, Frequency, and Monetary Value (RFM)
current_date = datetime.today()
data['Purchase_Date'] = pd.to_datetime(data['Purchase_Date'])
recency = data.groupby('Customer_ID')['Purchase_Date'].max()
data['Recency'] = (current_date - recency).dt.days
frequency = data.groupby('Customer_ID')['Purchase_Date'].count()
data['Frequency'] = frequency
monetary_value = data.groupby('Customer_ID')['Purchase_Amount'].sum()
data['Monetary_Value'] = monetary_value
data['Log_Monetary_Value'] = np.log1p(data['Monetary_Value'])

# Create scatter plots for RFM
plt.figure(figsize=(15, 5))

# Recency vs. Monetary Value
plt.subplot(131)
plt.scatter(data['Recency'], data['Log_Monetary_Value'], alpha=0.5)
plt.title('Recency vs. Log(Monetary Value)')
plt.xlabel('Recency (Days)')
plt.ylabel('Log(Monetary Value)')

# Frequency vs. Monetary Value
plt.subplot(132)
plt.scatter(data['Frequency'], data['Log_Monetary_Value'], alpha=0.5)
plt.title('Frequency vs. Log(Monetary Value)')
plt.xlabel('Frequency')
plt.ylabel('Log(Monetary Value)')

# Histogram of Monetary Value
plt.subplot(133)
plt.hist(data['Log_Monetary_Value'], bins=30, alpha=0.7)
plt.title('Histogram of Log(Monetary Value)')
plt.xlabel('Log(Monetary Value)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Prepare the features and target variable
X = data[['Recency', 'Frequency', 'Log_Monetary_Value']]
y = data['Monetary_Value']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Plot actual vs. predicted values
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Actual vs. Predicted CLV')
plt.xlabel('Actual CLV')
plt.ylabel('Predicted CLV')
plt.show()

# Example: Predict CLV for a specific customer
customer_data = [[90, 5, np.log1p(2000)]]
clv_prediction = model.predict(customer_data)
print(f"Predicted CLV: {clv_prediction[0]}")
